import json
import re
import os
from pathlib import Path

def detect_language(code_instruction):
    """
    从code_instruction中检测编程语言
    """
    if "python" in code_instruction.lower():
        return "python"
    elif "java" in code_instruction.lower():
        return "java"
    elif "cpp" in code_instruction.lower() or "c++" in code_instruction.lower():
        return "cpp"
    else:
        # 尝试从代码中推断语言
        if "def " in code_instruction or "import " in code_instruction or "print(" in code_instruction:
            return "python"
        elif "public class" in code_instruction or "System.out" in code_instruction:
            return "java"
        elif "cout <<" in code_instruction or "#include" in code_instruction:
            return "cpp"
        return "python"  # 默认使用python

def extract_code_from_response(response):
    """
    从模型响应中提取代码块
    """
    if not response:
        return ""
        
    # 使用正则表达式匹配代码块
    code_pattern = r"```(?:python|java|cpp|c\+\+)?\n([\s\S]*?)\n```"
    matches = re.findall(code_pattern, response)
    
    if matches:
        # 返回所有匹配的代码块，用换行符连接
        return "\n".join(matches)
    
    # 如果没有找到标准代码块，尝试匹配其他形式的代码块
    alt_pattern = r"```([\s\S]*?)```"
    matches = re.findall(alt_pattern, response)
    if matches:
        # 返回所有匹配的代码块，用换行符连接
        return "\n".join(matches)
    
    # 如果仍然没有找到代码块，尝试提取整个响应
    return response

def calculate_standardization_score(code, language):
    """
    计算代码规范性评分（30分制）
    """
    if not code:
        return 0.0
        
    score = 10.0  # 初始满分（10分制）
    
    # 基本检查项
    checks = {
        "python": [
            (r"def\s+\w+\s*\([^:]*\):", 0.5, "函数缺少类型标注"),
            (r"[a-zA-Z_][a-zA-Z0-9_]*\s*=", 0.3, "变量命名不规范"),
            (r"#\s*\w+", 0.2, "注释不规范"),
            (r"if\s+\w+\s*==\s*True", 0.3, "冗余的True比较"),
            (r"except:", 0.5, "捕获所有异常"),
            (r"print\(", 0.2, "使用print而非日志"),
            (r"^\s*[^#\n]{80,}", 0.2, "行长度超过80字符"),
            (r"class\s+\w+[^(]", 0.3, "类定义不规范"),
            (r"import\s+\*", 0.4, "导入所有模块")
        ],
        "java": [
            (r"public\s+class\s+\w+\s*{", 0.3, "类定义不规范"),
            (r"catch\s*\(\s*Exception\s+\w+\s*\)", 0.5, "捕获所有异常"),
            (r"System\.out\.print", 0.2, "使用System.out而非日志"),
            (r"[a-z][a-zA-Z0-9]*[A-Z]", 0.3, "驼峰命名不规范"),
            (r"//\s*\w+", 0.2, "注释不规范"),
            (r"if\s*\(\s*\w+\s*==\s*true\s*\)", 0.3, "冗余的true比较"),
            (r"^.{100,}", 0.2, "行长度超过100字符"),
            (r"private\s+\w+\s*;", 0.3, "字段定义不规范")
        ],
        "cpp": [
            (r"using\s+namespace\s+std\s*;", 0.5, "使用整个std命名空间"),
            (r"catch\s*\(\s*\.\.\.\s*\)", 0.5, "捕获所有异常"),
            (r"cout\s*<<", 0.2, "使用cout而非日志"),
            (r"#define\s+\w+", 0.3, "使用宏定义"),
            (r"//\s*\w+", 0.2, "注释不规范"),
            (r"if\s*\(\s*\w+\s*==\s*true\s*\)", 0.3, "冗余的true比较"),
            (r"^.{80,}", 0.2, "行长度超过80字符"),
            (r"void\s+\w+\s*\(\s*\)", 0.3, "函数定义不规范")
        ]
    }
    
    # 根据语言选择检查项
    if language in checks:
        for pattern, penalty, reason in checks[language]:
            if re.search(pattern, code):
                score -= penalty
    
    # 检查空行和缩进
    lines = code.split('\n')
    for i, line in enumerate(lines):
        # 检查缩进是否一致
        if i > 0 and line.strip() and lines[i-1].strip():
            prev_indent = len(lines[i-1]) - len(lines[i-1].lstrip())
            curr_indent = len(line) - len(line.lstrip())
            if abs(prev_indent - curr_indent) > 0 and abs(prev_indent - curr_indent) != 4:
                score -= 0.2
    
    # 确保分数在0-10之间
    score = max(0, min(10, score))
    
    # 转换为30分制
    score_30 = score * 3
    
    # 保留一位小数
    return round(score_30, 1)

def detect_language_from_code(code):
    """
    从代码内容推断编程语言
    """
    if not code:
        return "unknown"
        
    # Python特征
    if re.search(r"def\s+\w+\s*\(", code) or re.search(r"import\s+\w+", code) or "print(" in code:
        return "python"
    # Java特征
    elif re.search(r"public\s+class", code) or re.search(r"System\.out\.", code) or re.search(r"public\s+static\s+void\s+main", code):
        return "java"
    # C++特征
    elif re.search(r"#include", code) or re.search(r"std::", code) or re.search(r"cout\s*<<", code):
        return "cpp"
    else:
        return "python"  # 默认使用python

def process_jsonl_file(input_file, output_file):
    """
    处理输入的jsonl文件，计算每个模型回答的规范性评分
    """
    results = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                
                # 获取编程语言
                language = detect_language(data.get("code-instruction", ""))
                
                # 初始化结果
                result = {
                    "index": data.get("index", 0),
                    "programming_language": data.get("programming_language", language),
                    "results": {}
                }
                
                # 处理每个模型的回答
                for model, response in data.get("results", {}).items():
                    if response:
                        code = extract_code_from_response(response)
                        # 如果语言未知，尝试从代码中推断
                        if language == "unknown":
                            language = detect_language_from_code(code)
                            result["programming_language"] = language
                        
                        score = calculate_standardization_score(code, language)
                        result["results"][model] = {"score": score}
                
                results.append(result)
            except Exception as e:
                print(f"处理数据时出错 (index {data.get('index', 'unknown')}): {e}")
    
    # 写入输出文件
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            # 调整输出格式
            formatted_result = {
                "index": result["index"],
                "programming_language": result["programming_language"],
                "results": {}
            }
            
            # 使用实际数据中的模型名称
            model_order = ["gpt", "claude", "deepseek", "qwen"]
            for model in model_order:
                if model in result["results"]:
                    formatted_result["results"][model] = result["results"][model]["score"]
            
            # 确保至少有一个模型的结果
            if not formatted_result["results"]:
                # 检查原始数据中是否有模型结果
                for model, response in data.get("results", {}).items():
                    if response:
                        code = extract_code_from_response(response)
                        language = detect_language_from_code(code) if language == "unknown" else language
                        score = calculate_standardization_score(code, language)
                        formatted_result["results"][model] = score
            
            f.write(json.dumps(formatted_result, ensure_ascii=False) + '\n')

def main():
    # 获取当前脚本所在目录
    current_dir = Path(__file__).parent
    
    # 输入和输出文件路径
    input_file = current_dir / "6model_result.jsonl"
    output_file = current_dir / "standardization_scores.jsonl"
    
    # 处理文件
    process_jsonl_file(input_file, output_file)
    print(f"规范性评分已保存到 {output_file}")

if __name__ == "__main__":
    main()
