import json
import re
import math
import os
import ast
import javalang
import pycparser
from radon.metrics import mi_visit
from radon.complexity import cc_visit

def extract_code(response):
    """从LLM响应中提取代码块"""
    if not response:
        return ""
    
    # 确保response是字符串类型
    if not isinstance(response, str):
        try:
            response = str(response)
        except:
            print(f"无法将响应转换为字符串: {type(response)}")
            return ""
        
    # 尝试多种代码块格式
    patterns = [
        r'```(?:python|java|cpp|c\+\+)?\s*([\s\S]*?)```',  # 标准格式
        r'```(.*?)```',                                    # 简单格式
        r'`{3,}([\s\S]*?)`{3,}',                           # 多行反引号
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, response, re.DOTALL)
        if matches:
            return matches[0].strip()
    
    # 如果没有找到代码块，尝试提取可能的代码片段
    # 修改：更精确地识别代码部分，避免提取非代码内容
    lines = response.split('\n')
    code_lines = []
    in_code = False
    code_indent = None
    
    for line in lines:
        stripped = line.strip()
        # 检测可能的代码行
        if (re.search(r'\b(def|class|if|for|while|return|import|function)\b', line) or
            re.search(r'[{}\[\]();=]', line)):
            if not in_code:
                in_code = True
                # 检测缩进
                indent = len(line) - len(line.lstrip())
                if indent > 0:
                    code_indent = indent
            code_lines.append(line)
        elif in_code:
            # 如果已经在代码块中，继续收集代码行
            if stripped:  # 非空行
                code_lines.append(line)
            elif len(code_lines) > 3:  # 如果已经收集了足够的代码行，空行可能表示代码块结束
                in_code = False
    
    if code_lines:
        return '\n'.join(code_lines)
    
    # 最后尝试：如果响应看起来像代码，直接返回
    if re.search(r'\b(def|class|if|for|while|return|import|function)\b', response):
        return response
    
    return ""  # 如果无法识别代码块，返回空字符串

def detect_language(code):
    """检测代码的编程语言"""
    # 更健壮的语言检测逻辑
    if not code:
        return 'unknown'
        
    # Python特征
    python_features = ['def ', 'import ', 'print(', 'if __name__ == "__main__":', '# ', 'class ', 'self.', 'None', 'True', 'False']
    python_score = sum(1 for feature in python_features if feature in code)
    
    # Java特征
    java_features = ['public class', 'public static void main', 'System.out.println', 'import java.', 'private ', 'protected ', '@Override', 'extends ', 'implements ']
    java_score = sum(1 for feature in java_features if feature in code)
    
    # C++特征
    cpp_features = ['#include', 'int main()', 'std::', 'cout <<', 'cin >>', 'using namespace', 'void ', 'return 0;', '->', '::']
    cpp_score = sum(1 for feature in cpp_features if feature in code)
    
    # 根据特征得分判断语言
    scores = {'python': python_score, 'java': java_score, 'cpp': cpp_score}
    max_lang = max(scores, key=scores.get)
    
    # 如果最高分太低，可能无法确定语言
    if scores[max_lang] < 2:
        # 使用简单的关键字检测作为备选
        if 'def ' in code or 'import ' in code or 'print(' in code:
            return 'python'
        elif 'public class' in code or 'public static void main' in code:
            return 'java'
        elif '#include' in code or 'int main()' in code:
            return 'cpp'
        return 'unknown'
    
    return max_lang

def calculate_mi_score(code, language):
    """计算维护性指数(MI)得分"""
    try:
        if language == 'python':
            # 使用radon计算Python代码的MI值
            try:
                # 首先检查代码是否有效
                import ast as python_ast  # 重命名以避免命名冲突
                python_ast.parse(code)
                mi_result = mi_visit(code, True)
                mi_value = mi_result
                # MI值通常在0-100之间，可以直接作为得分
                score = round(mi_value, 1) if mi_value <= 100 else 100.0
            except SyntaxError as e:
                print(f"Python代码语法错误: {e}")
                # 使用简化的计算方法作为备选
                lines = len(code.split('\n'))
                complexity = len(re.findall(r'if|for|while|except|with|def|class', code))
                mi_value = 100 - (lines * 0.2 + complexity * 0.5)
                score = round(max(0, min(mi_value, 100)), 1)
        elif language == 'java':
            # 简化的Java代码MI计算
            try:
                # 尝试使用javalang解析Java代码
                javalang.parse.parse(code)
                lines = len(code.split('\n'))
                complexity = len(re.findall(r'if|for|while|switch|catch|try|instanceof', code))
                mi_value = 100 - (lines * 0.2 + complexity * 0.5)
                score = round(max(0, min(mi_value, 100)), 1)
            except Exception as e:
                print(f"Java代码解析错误: {e}")
                # 使用更简单的方法
                lines = len(code.split('\n'))
                complexity = len(re.findall(r'if|for|while|switch|catch', code))
                mi_value = 100 - (lines * 0.2 + complexity * 0.5)
                score = round(max(0, min(mi_value, 100)), 1)
        elif language == 'cpp':
            # 简化的C++代码MI计算
            try:
                # 尝试使用pycparser解析C++代码
                # 注意：pycparser只支持C代码，这里只是尝试解析
                # 添加一个简单的main函数包装，使其成为有效的C代码
                wrapped_code = f"""
#include <stdio.h>
int main() {{
    {code}
    return 0;
}}
"""
                with open('temp.c', 'w') as f:
                    f.write(wrapped_code)
                parser = pycparser.c_parser.CParser()
                cpp_ast = parser.parse(wrapped_code)  # 重命名变量避免冲突
                os.remove('temp.c')
                
                lines = len(code.split('\n'))
                complexity = len(re.findall(r'if|for|while|switch|catch|try|throw', code))
                mi_value = 100 - (lines * 0.2 + complexity * 0.5)
                score = round(max(0, min(mi_value, 100)), 1)
            except Exception as e:
                print(f"C++代码解析错误: {e}")
                # 使用更简单的方法
                lines = len(code.split('\n'))
                complexity = len(re.findall(r'if|for|while|switch|catch', code))
                mi_value = 100 - (lines * 0.2 + complexity * 0.5)
                score = round(max(0, min(mi_value, 100)), 1)
        else:
            # 对于未知语言，使用通用方法
            lines = len(code.split('\n'))
            complexity = len(re.findall(r'if|for|while|switch|catch|try', code))
            mi_value = 100 - (lines * 0.2 + complexity * 0.5)
            score = round(max(0, min(mi_value, 100)), 1)
        
        return {"mi_value": mi_value, "score": score}
    except Exception as e:
        print(f"计算MI得分时出错: {e}")
        # 出错时使用基于代码长度的简单计算
        try:
            lines = len(code.split('\n'))
            # 简单公式：较短的代码通常更易维护
            mi_value = max(0, 100 - lines * 0.5)
            score = round(max(0, min(mi_value, 100)), 1)
            return {"mi_value": mi_value, "score": score}
        except:
            # 如果连长度都无法计算，返回默认值
            return {"mi_value": 50, "score": 50.0}

def check_error_handling(code, language):
    """检查代码中的错误处理"""
    error_handling_score = 0
    
    if language == 'python':
        try_count = code.count('try:')
        except_count = code.count('except')
        finally_count = code.count('finally:')
        
        if try_count > 0 and except_count > 0:
            error_handling_score = 20
            if finally_count > 0:
                error_handling_score += 10
    
    elif language == 'java':
        try_count = code.count('try {')
        catch_count = code.count('catch')
        finally_count = code.count('finally')
        
        if try_count > 0 and catch_count > 0:
            error_handling_score = 20
            if finally_count > 0:
                error_handling_score += 10
    
    elif language == 'cpp':
        try_count = code.count('try {')
        catch_count = code.count('catch')
        
        if try_count > 0 and catch_count > 0:
            error_handling_score = 25
    
    return error_handling_score

def check_input_validation(code, language):
    """检查代码中的输入验证"""
    input_validation_score = 0
    
    if language == 'python':
        # 检查类型检查、条件判断等
        if 'isinstance(' in code or 'type(' in code:
            input_validation_score += 10
        if 'if ' in code and (' < ' in code or ' > ' in code or ' == ' in code):
            input_validation_score += 10
    
    elif language == 'java':
        # 检查null检查、条件判断等
        if 'null' in code and 'if' in code:
            input_validation_score += 10
        if 'if ' in code and (' < ' in code or ' > ' in code or ' == ' in code):
            input_validation_score += 10
    
    elif language == 'cpp':
        # 检查null检查、条件判断等
        if 'nullptr' in code or 'NULL' in code:
            input_validation_score += 10
        if 'if ' in code and (' < ' in code or ' > ' in code or ' == ' in code):
            input_validation_score += 10
    
    return input_validation_score

def check_resource_management(code, language):
    """检查代码中的资源管理"""
    resource_management_score = 0
    
    if language == 'python':
        # 检查with语句、close()方法等
        if 'with ' in code:
            resource_management_score += 15
        if '.close()' in code:
            resource_management_score += 10
    
    elif language == 'java':
        # 检查try-with-resources、close()方法等
        if 'try (' in code:
            resource_management_score += 15
        if '.close()' in code:
            resource_management_score += 10
    
    elif language == 'cpp':
        # 检查RAII模式、delete关键字等
        if 'delete ' in code or 'delete[]' in code:
            resource_management_score += 15
        if 'unique_ptr' in code or 'shared_ptr' in code:
            resource_management_score += 10
    
    return resource_management_score

def calculate_robustness_score(code, language):
    """计算代码的健壮性得分"""
    # 基础得分
    base_score = 50
    
    # 错误处理得分
    error_handling_score = check_error_handling(code, language)
    
    # 输入验证得分
    input_validation_score = check_input_validation(code, language)
    
    # 资源管理得分
    resource_management_score = check_resource_management(code, language)
    
    # 计算总得分
    total_score = base_score + error_handling_score + input_validation_score + resource_management_score
    
    # 确保得分不超过100
    total_score = min(total_score, 100)
    
    return round(total_score, 1)

def process_jsonl_file(input_file, output_file):
    """处理JSONL文件并计算健壮性得分"""
    results = []
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            line_count = sum(1 for _ in f)
        
        with open(input_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f, 1):
                try:
                    print(f"处理第 {i}/{line_count} 条数据...")
                    data = json.loads(line)
                    index = data.get('index')
                    programming_language = data.get('programming_language', '')
                    
                    # 检查是否为需要跳过的索引
                    if index in [41, 55, 61, 67, 71, 77, 82, 94]:
                        print(f"  跳过索引为{index}的数据项")
                        # 为跳过的数据添加结果，得分为0
                        result_entry = {
                            "index": index,
                            "programming_language": programming_language,
                            "results": {model: 0.0 for model in data.get('results', {})}
                        }
                        results.append(result_entry)
                        continue
                    
                    result_entry = {
                        "index": index,
                        "programming_language": programming_language,
                        "results": {}
                    }
                    
                    for model, response in data.get('results', {}).items():
                        try:
                            print(f"  处理模型: {model}")
                            # 检查响应类型
                            if not isinstance(response, str):
                                print(f"  处理模型 {model} 时出错: expected string or bytes-like object")
                                result_entry['results'][model] = 0.0
                                continue
                                
                            code = extract_code(response)
                            if not code:
                                print(f"  无法从模型 {model} 的响应中提取代码")
                                result_entry['results'][model] = 0.0
                                continue
                            
                            # 如果代码长度过短，可能不是有效代码
                            if len(code.strip()) < 20:
                                print(f"  提取的代码过短，可能不是有效代码")
                                result_entry['results'][model] = 0.0
                                continue
                            
                            # 如果没有明确指定语言，则尝试检测
                            if not programming_language or programming_language in ['python或Java或C++'] or 'python' in programming_language.lower() or 'java' in programming_language.lower() or 'c++' in programming_language.lower():
                                lang = detect_language(code)
                                print(f"  检测到语言: {lang}")
                            else:
                                lang = programming_language.lower()
                                print(f"  使用指定语言: {lang}")
                            
                            # 验证语言是否被正确识别和支持
                            if lang == 'unknown' or lang not in ['python', 'java', 'cpp', 'c++', 'c']:
                                print(f"  不支持的语言或无法识别语言: {lang}")
                                result_entry['results'][model] = 0.0
                                continue
                            
                            # 标准化语言名称
                            if lang in ['py', 'python3']:
                                lang = 'python'
                            elif lang in ['c++', 'c']:
                                lang = 'cpp'
                            
                            # 计算MI得分
                            print(f"  计算MI得分...")
                            try:
                                mi_result = calculate_mi_score(code, lang)
                                if mi_result['score'] <= 0:
                                    print(f"  MI得分计算异常，得分为0")
                                    result_entry['results'][model] = 0.0
                                    continue
                            except Exception as e:
                                print(f"  MI得分计算失败: {e}")
                                result_entry['results'][model] = 0.0
                                continue
                            
                            # 计算健壮性得分
                            print(f"  计算健壮性得分...")
                            try:
                                robustness_score = calculate_robustness_score(code, lang)
                                if robustness_score <= 0:
                                    print(f"  健壮性得分计算异常，得分为0")
                                    result_entry['results'][model] = 0.0
                                    continue
                            except Exception as e:
                                print(f"  健壮性得分计算失败: {e}")
                                result_entry['results'][model] = 0.0
                                continue
                            
                            # 综合得分 (MI得分占50%，健壮性得分占50%)
                            final_score = (mi_result['score'] + robustness_score) / 2
                            
                            # 将100分制映射到30分制
                            final_score_30 = round((final_score / 100) * 30, 1)
                            
                            print(f"  最终得分: {final_score_30}/30")
                            result_entry['results'][model] = final_score_30
                        except Exception as e:
                            print(f"  处理模型 {model} 时出错: {e}")
                            result_entry['results'][model] = 0.0
                    
                    results.append(result_entry)
                except Exception as e:
                    print(f"处理第 {i} 条数据时出错: {e}")
                    continue
    except Exception as e:
        print(f"读取输入文件时出错: {e}")
    
    # 写入输出文件
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        print(f"结果已写入到 {output_file}")
    except Exception as e:
        print(f"写入输出文件时出错: {e}")
        # 尝试写入备用文件
        try:
            backup_file = "robustness_scores_backup.jsonl"
            with open(backup_file, 'w', encoding='utf-8') as f:
                for result in results:
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
            print(f"结果已写入到备用文件 {backup_file}")
        except:
            print("无法写入结果到任何文件")

# 示例用法
if __name__ == "__main__":
    try:
        print("开始处理健壮性评分...")
        input_file = "6model_result.jsonl"
        output_file = "robustness_scores.jsonl"
        
        # 检查输入文件是否存在
        if not os.path.exists(input_file):
            print(f"输入文件 {input_file} 不存在，尝试查找其他jsonl文件...")
            jsonl_files = [f for f in os.listdir('.') if f.endswith('.jsonl')]
            if jsonl_files:
                input_file = jsonl_files[0]
                print(f"使用 {input_file} 作为输入文件")
            else:
                print("未找到任何jsonl文件")
                exit(1)
        
        process_jsonl_file(input_file, output_file)
        print("处理完成")
    except Exception as e:
        print(f"程序执行出错: {e}")
        import traceback
        traceback.print_exc()
