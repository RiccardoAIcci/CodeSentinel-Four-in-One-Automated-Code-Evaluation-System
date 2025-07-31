import json
import math
import re
import os
import traceback
from typing import Dict, Any, List, Tuple
from collections import OrderedDict

# 支持的语言
SUPPORTED_LANGUAGES = ["python", "java", "c++"]

def extract_code_from_response(response: str, language: str) -> str:
    """从模型响应中提取代码块"""
    if not response:
        return ""
        
    # 修复正则表达式，避免"multiple repeat"错误
    code_pattern = r"```(?:" + re.escape(language) + r")?[\s\n]*([\s\S]*?)```"
    matches = re.findall(code_pattern, response, re.IGNORECASE)
    
    if matches:
        return "\n".join(matches)
    
    # 如果没有找到代码块，尝试其他格式
    generic_patterns = [
        r"```[\s\n]*([\s\S]*?)```",  # 任何语言的代码块
        r"`{3,}([\s\S]*?)`{3,}",     # 多行反引号
    ]
    
    for pattern in generic_patterns:
        matches = re.findall(pattern, response, re.DOTALL)
        if matches:
            return "\n".join(matches)
    
    # 如果仍然没有找到代码块，尝试直接提取代码
    lines = response.split("\n")
    code_lines = []
    in_code = False
    
    for line in lines:
        # 检测可能的代码行（根据语言特征）
        if language.lower() == "python":
            if line.strip().startswith(("def ", "class ", "import ", "from ", "if ", "for ", "while ")):
                in_code = True
        elif language.lower() in ["java", "c++"]:
            if line.strip().startswith(("public ", "private ", "class ", "import ", "package ", "#include", "void ", "int ")):
                in_code = True
        else:
            # 通用代码特征
            if (re.search(r'\b(def|class|if|for|while|return|import|function)\b', line) or
                re.search(r'[{}\[\]();=]', line)):
                in_code = True
        
        if in_code:
            code_lines.append(line)
        elif line.strip() == "" and in_code:
            # 空行可能是代码的一部分
            code_lines.append(line)
        elif in_code:
            # 如果遇到非代码行，结束代码块
            in_code = False
    
    return "\n".join(code_lines) if code_lines else ""

def count_loc(code: str) -> int:
    """计算有效代码行数（排除注释和空行）"""
    if not code:
        return 0
        
    lines = code.split("\n")
    count = 0
    
    for line in lines:
        line = line.strip()
        if line and not line.startswith("#") and not line.startswith("//") and not line.startswith("/*"):
            count += 1
    
    return max(1, count)  # 确保至少返回1行

def calculate_cc(code: str, language: str) -> int:
    """计算循环复杂度（简化版）"""
    if not code:
        return 1
        
    if language == "python":
        # 计算Python的CC（简化版）
        keywords = ["if", "for", "while", "and", "or", "elif", "except", "finally"]
        cc = 1  # 基础复杂度为1
        
        for keyword in keywords:
            cc += code.count(f" {keyword} ") + code.count(f"{keyword} ")
        
        return cc
    
    elif language == "java" or language == "c++":
        # 计算Java/C++的CC（简化版）
        keywords = ["if", "for", "while", "case", "catch", "&&", "||", "?"]
        cc = 1  # 基础复杂度为1
        
        for keyword in keywords:
            cc += code.count(f" {keyword} ") + code.count(f"{keyword} ")
        
        return cc
    
    return 1  # 默认返回1

def calculate_hv(code: str, loc: int) -> float:
    """使用论文公式(3)计算Halstead Volume"""
    if loc <= 0:
        return 1.0
    return max(1.0, 45 * loc - 428)  # 确保至少为1.0

def calculate_mi(loc: int, cc: int, hv: float) -> float:
    """计算可维护性指数"""
    try:
        # 防止无效值
        loc = max(1, loc)
        cc = max(1, cc)
        hv = max(1.0, hv)
        
        mi = 171 - 5.2 * math.log(hv) - 0.23 * cc - 16.2 * math.log(loc)
        return max(0, min(100, mi))  # 确保MI在0-100之间
    except (ValueError, ZeroDivisionError) as e:
        print(f"计算MI时出错: {e}")
        # 处理可能的数学错误（如log(0)）
        return 50.0  # 返回中等值

def map_to_score(mi: float) -> float:
    """将MI值映射到1-5分"""
    if mi >= 85:
        return 5.0
    elif 65 <= mi < 85:
        return 4.0
    elif 50 <= mi < 65:
        return 3.0
    elif 30 <= mi < 50:
        return 2.0
    else:
        return 1.0

def normalize_language(language: str) -> str:
    """标准化语言名称"""
    if not language:
        return "python"  # 默认使用Python
        
    language = language.lower()
    
    # 映射到支持的语言
    language_map = {
        # Python
        'python': 'python',
        'py': 'python',
        'python3': 'python',
        
        # Java
        'java': 'java',
        
        # C++
        'cpp': 'c++',
        'c++': 'c++',
        'c': 'c++',
    }
    
    return language_map.get(language, 'python')  # 如果未知语言，默认使用Python

def process_model_response(response: str, language: str) -> Dict[str, float]:
    """处理单个模型的响应，返回得分（30分制）"""
    try:
        code = extract_code_from_response(response, language)
        
        if not code:
            return {"score": 0.0}  # 无法提取代码时返回0分
        
        loc = count_loc(code)
        cc = calculate_cc(code, language)
        hv = calculate_hv(code, loc)
        
        # 防止无效值
        if loc <= 0:
            loc = 1
        if hv <= 0:
            hv = 1
        
        mi = calculate_mi(loc, cc, hv)
        score = map_to_score(mi)
        
        # 将5分制转换为30分制
        score_30 = score * 6
        
        return {"score": round(score_30, 1)}
    except Exception as e:
        print(f"处理模型响应时出错: {e}")
        traceback.print_exc()
        return {"score": 0.0}  # 出错时返回0分

def calculate_simplicity_scores(input_file: str, output_file: str) -> None:
    """计算所有模型的简洁性得分并输出到文件"""
    results = []
    
    try:
        # 检查输入文件是否存在
        if not os.path.exists(input_file):
            print(f"输入文件 {input_file} 不存在，尝试查找其他jsonl文件...")
            jsonl_files = [f for f in os.listdir('.') if f.endswith('.jsonl')]
            if jsonl_files:
                input_file = jsonl_files[0]
                print(f"使用 {input_file} 作为输入文件")
            else:
                print("未找到任何jsonl文件")
                return
        
        # 计算总行数以显示进度
        with open(input_file, 'r', encoding='utf-8') as f:
            total_lines = sum(1 for _ in f)
        
        with open(input_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f, 1):
                try:
                    print(f"处理第 {i}/{total_lines} 条数据...")
                    data = json.loads(line)
                    
                    index = data.get("index")
                    
                    # 从results中提取代码，而不是从programming_language字段
                    model_results = data.get("results", {})
                    
                    # 使用OrderedDict确保模型顺序一致
                    ordered_results = OrderedDict()
                    
                    # 检测语言类型，从代码中推断
                    language = "python"  # 默认使用Python
                    
                    # 尝试从base字段中提取语言信息
                    base_response = model_results.get("base", "")
                    if "```java" in base_response or "import java." in base_response:
                        language = "java"
                    elif "```c++" in base_response or "#include" in base_response:
                        language = "c++"
                    
                    print(f"  检测到语言: {language}")
                    
                    output_results = {"index": index, "programming_language": language, "results": ordered_results}
                    
                    # 处理所有可能的模型字段
                    for model_key in model_results:
                        try:
                            response = model_results[model_key]
                            if not response:  # 如果响应为空
                                ordered_results[model_key] = 0.0
                                print(f"  模型 {model_key} 响应为空，得分: 0.0/30")
                                continue
                                
                            print(f"  处理模型: {model_key}")
                            metrics = process_model_response(response, language)
                            ordered_results[model_key] = metrics["score"]
                            print(f"  得分: {metrics['score']}/30")
                        except Exception as e:
                            print(f"  处理模型 {model_key} 时出错: {e}")
                            traceback.print_exc()
                            ordered_results[model_key] = 0.0  # 出错时返回0分
                    
                    results.append(output_results)
                except json.JSONDecodeError as e:
                    print(f"JSON解析错误: {e}")
                    continue
                except Exception as e:
                    print(f"处理行 {i} 时出错: {e}")
                    traceback.print_exc()
                    continue
        
        print(f"写入结果到 {output_file}...")
        with open(output_file, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result) + '\n')
        
        print(f"处理完成，共评估 {len(results)} 个代码片段")
                
    except Exception as e:
        print(f"处理文件时出错: {e}")
        traceback.print_exc()
        
        # 尝试保存已处理的结果
        if results:
            backup_file = output_file + ".backup"
            try:
                with open(backup_file, 'w', encoding='utf-8') as f:
                    for result in results:
                        f.write(json.dumps(result) + '\n')
                print(f"已将部分结果保存到 {backup_file}")
            except:
                print("无法保存部分结果")

if __name__ == "__main__":
    try:
        print("开始计算简洁性得分...")
        input_file = "6model_result.jsonl"
        output_file = "simplicity_scores.jsonl"
        
        calculate_simplicity_scores(input_file, output_file)
        print("处理完成")
    except Exception as e:
        print(f"程序执行出错: {e}")
        traceback.print_exc()
