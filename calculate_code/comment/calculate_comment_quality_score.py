import re
from typing import Set
import math
import json
import traceback
import time


def extract_solution(text):
    try:
        if not isinstance(text, str):
            print(f"警告：输入不是字符串类型，而是 {type(text)}")
            try:
                text = str(text)
            except:
                return None
        
        if not text:
            return None
            
        patterns = [
            r"```[\s\S]*?\n([\s\S]*?)```",
            r"```([\s\S]*?)```",
            r"`{3,}([\s\S]*?)`{3,}",
            r"(?<=\n)((?:    |\t).*(?:\n|$))+",
            r"(?<=\n)((?!```).*\n)*(?=\n|$)"
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            if matches:
                code = max(matches, key=len).strip()
                if code:
                    return clean_extracted_code(code)
        
        lines = text.split('\n')
        code_lines = []
        in_code = False
        
        for line in lines:
            if (re.search(r'\b(def|class|if|for|while|return|import|function|print|var|let|const)\b', line) or
                re.search(r'[{}\[\]();=]', line) or
                re.search(r'^\s*[a-zA-Z_][a-zA-Z0-9_]*\s*[=:]', line)):
                in_code = True
                code_lines.append(line)
            elif in_code and line.strip():
                code_lines.append(line)
            elif in_code:
                in_code = False
        
        if code_lines:
            return clean_extracted_code('\n'.join(code_lines))
        
        return None
        
    except Exception as e:
        print(f"代码提取出错: {str(e)}")
        return None

def clean_extracted_code(code):
    try:
        code = re.sub(r'^(python|java|cpp)\n', '', code, flags=re.IGNORECASE)
        
        code = code.strip()
        
        lines = code.split('\n')
        cleaned_lines = []
        min_indent = float('inf')
        
        for line in lines:
            if line.strip():
                indent = len(line) - len(line.lstrip())
                min_indent = min(min_indent, indent)
        
        for line in lines:
            if line.strip():
                if min_indent < float('inf'):
                    line = line[min_indent:]
            cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    except Exception as e:
        print(f"代码清理出错: {str(e)}")
        return code

def extract_comment_in_java_cpp(text):
    if text is None:
        return [], 0
    lines = text.splitlines()
    result = []
    comment_lines_count = 0
    multi_line_comment_pattern = r"(\/\*[\s\S]*?\*\/|\/\*\*[\s\S]*?\*\/)"
    multi_line_comments = re.findall(multi_line_comment_pattern, text)
    for multi_comment in multi_line_comments:
        result.append(multi_comment.strip("/*").strip("*/").strip())
        comment_lines_count += multi_comment.count('\n') + 1  

    for i in range(len(lines)):
        current_line = lines[i]
        prev_line = lines[i - 1] if i > 0 else None 
        next_line = lines[i + 1] if i < len(lines) - 1 else None 
        
        if current_line.strip().startswith('//'):
            comment_content = current_line.split('//', 1)[1].strip()  
            
            if i == 0:
                if next_line is None or not next_line.strip().startswith('//'):
                    result.append(comment_content)
                    comment_lines_count += 1
       
            elif i == len(lines) - 1:
                if prev_line is None or not prev_line.strip().startswith('//'):
                    result.append(comment_content)
                    comment_lines_count += 1
           
            else:
                if not prev_line.strip().startswith('//') and not next_line.strip().startswith('//'):
                    result.append(comment_content)
                    comment_lines_count += 1
                elif not prev_line.strip().startswith('//') and next_line.strip().startswith('//'):
                    combined_comment = comment_content
                    j = i + 1
                    while j < len(lines) and lines[j].strip().startswith('//'):
                        combined_comment += ' ' + lines[j].split('//', 1)[1].strip()
                        j += 1
                    result.append(combined_comment)
                    comment_lines_count += j - i  
                else:
                    pass

    return result, comment_lines_count  

def extract_comment_in_python(text):
    if text is None:
        return [], 0
    lines = text.splitlines()
    result = []
    comment_lines_count = 0
    multi_line_comment_pattern = r"('''[\s\S]*?'''|\"\"\"[\s\S]*?\"\"\")"
    multi_line_comments = re.findall(multi_line_comment_pattern, text)

    for multi_comment in multi_line_comments:
        result.append(multi_comment.strip("'''").strip("\"\"\""))
        comment_lines_count += multi_comment.count('\n') + 1  

    for i in range(len(lines)):
        current_line = lines[i]
        prev_line = lines[i - 1] if i > 0 else None  
        next_line = lines[i + 1] if i < len(lines) - 1 else None  

        if current_line.strip().startswith('#'):
            comment_content = current_line.split('#', 1)[1].strip()  
            
      
            if i == 0:
                if next_line is None or not next_line.strip().startswith('#'):
                    result.append(comment_content)
                    comment_lines_count += 1
            
            elif i == len(lines) - 1:
                if prev_line is None or not prev_line.strip().startswith('#'):
                    result.append(comment_content)
                    comment_lines_count += 1
            else:
                if not prev_line.strip().startswith('#') and not next_line.strip().startswith('#'):
                    result.append(comment_content)
                    comment_lines_count += 1
                elif not prev_line.strip().startswith('#') and next_line.strip().startswith('#'):
               
                    combined_comment = comment_content
                
                    j = i + 1
                    while j < len(lines) and lines[j].strip().startswith('#'):
                        combined_comment += ' ' + lines[j].split('#', 1)[1].strip()
                        j += 1
                    result.append(combined_comment)
                  
                    comment_lines_count += j - i  
                else:
                    pass
                    
        elif '#' in current_line:
            comment_content = current_line.split('#', 1)[1].strip()  
            result.append(comment_content)
            comment_lines_count += 1  

    return result, comment_lines_count 

def extract_pure_code_in_solution(code, language='java'):
    if code is None:
        return ""
    
    if language.lower() in ['java', 'cpp', 'c++', 'c', 'csharp', 'c#', 'javascript', 'js', 'typescript', 'ts']:
        code = re.sub(r'//.*', '', code)
        code = re.sub(r'/\*[\s\S]*?\*/', '', code)
        code = re.sub(r'/\*\*[\s\S]*?\*/', '', code)
    
    elif language.lower() in ['python', 'py']:
        code = re.sub(r'#.*', '', code)
        code = re.sub(r'\'\'\'[\s\S]*?\'\'\'', '', code)
        code = re.sub(r'\"\"\"[\s\S]*?\"\"\"', '', code)
    
    elif language.lower() in ['ruby', 'rb']:
        code = re.sub(r'#.*', '', code)
        code = re.sub(r'=begin[\s\S]*?=end', '', code)
    
    return code

def normalize_language(language):
    if not language:
        return "python"
        
    language = language.lower()
    
    language_map = {
        'python': 'python',
        'py': 'python',
        'python3': 'python',
        
        'java': 'java',
        
        'cpp': 'cpp',
        'c++': 'cpp',
        'c': 'cpp',
        
        'javascript': 'java',
        'js': 'java',
        'typescript': 'java',
        'ts': 'java',
        'csharp': 'java',
        'c#': 'java',
        'php': 'java',
        'go': 'java',
        'ruby': 'python',
        'rb': 'python',
        'rust': 'cpp',
        'swift': 'java',
        'kotlin': 'java',
        'scala': 'java',
    }
    
    return language_map.get(language, 'python')

def extract_identifiers_in_code(code, language='java'):
    code = extract_pure_code_in_solution(code, language)
    if not code:
        return set()
    
    normalized_language = normalize_language(language)
    
    java_keywords = {
        "abstract", "assert", "boolean", "break", "byte", "case", "catch", "char", "class",
        "const", "continue", "default", "do", "double", "else", "enum", "extends", "final",
        "finally", "float", "for", "goto", "if", "implements", "import", "instanceof", "int",
        "interface", "long", "native", "new", "null", "package", "private", "protected",
        "public", "return", "short", "static", "strictfp", "super", "switch", "synchronized",
        "this", "throw", "throws", "transient", "try", "void", "volatile", "while"
    }
    
    cpp_keywords = {
        "alignas", "alignof", "and", "and_eq", "asm", "auto", "bitand", "bitor", "bool", "break", "case",
        "catch", "char", "char16_t", "char32_t", "class", "compl", "concept", "const", "consteval", "constexpr",
        "constinit", "continue", "co_await", "co_return", "decltype", "default", "delete", "do", "double", "dynamic_cast",
        "else", "enum", "explicit", "export", "extern", "false", "final", "float", "for", "friend", "goto",
        "if", "import", "inline", "int", "long", "mutable", "namespace", "new", "noexcept", "not", "not_eq", 
        "nullptr", "operator", "or", "or_eq", "private", "protected", "public", "reflexpr", "register", "reinterpret_cast", 
        "requires", "return", "short", "signed", "sizeof", "static", "static_assert", "static_cast", "struct", "switch", 
        "synchronized", "template", "this", "throw", "throws", "true", "try", "typedef", "typeid", "typename", "union", 
        "unsigned", "using", "virtual", "void", "volatile", "wchar_t", "while"
    }
    
    python_keywords = {
        "False", "None", "True", "and", "as", "assert", "async", "await", "break", "class", "continue", "def", 
        "del", "elif", "else", "except", "finally", "for", "from", "global", "if", "import", "in", "is", "lambda", 
        "nonlocal", "not", "or", "pass", "raise", "return", "try", "while", "with", "yield"
    }

    if normalized_language == 'java':
        keywords = java_keywords
    elif normalized_language == 'cpp':
        keywords = cpp_keywords
    elif normalized_language == 'python':
        keywords = python_keywords
    else:
        keywords = python_keywords

    pattern = r'\b[a-zA-Z_][a-zA-Z0-9_]*\b'

    identifiers = re.findall(pattern, code)
    camel_case_parts = re.findall(r'[a-zA-Z][a-z]*', code)
    snake_case_parts = re.findall(r'[a-zA-Z]+', code.replace('_', ' '))
    upper_case_parts = re.findall(r'[A-Z]+(?:[a-z]+)?', code)
    all_parts = camel_case_parts + snake_case_parts + upper_case_parts + identifiers
    all_parts = set(all_parts)
    filtered_identifiers = [identifier for identifier in all_parts if identifier not in keywords]

    return set(filtered_identifiers)

def extract_terms_from_comment(lst):
    terms = set()
    
    for sentence in lst:
        words = sentence.split()
        for word in words:
            terms.add(word.strip().lower())  
        
    return terms

def calculate_cic(comments: Set[str], identifiers: Set[str]) -> float:
    if not comments and not identifiers:
        return 1.0  
    
    intersection = comments.intersection(identifiers)
    union = comments.union(identifiers)
    
    return len(intersection) / len(union)

def calculate_cls(c):
    abs_c = abs(c)
    
    if abs_c <= 2:
        return 0
    elif 2 < abs_c <= 30:
        return (abs_c - 2) / 28
    else:
        return 1

def calculate_average_cls(comments):
    if not comments:
        return 0
    cls_values = [calculate_cls(len(comment)) for comment in comments]
    average_cls = sum(cls_values) / len(cls_values) if cls_values else 0
    return average_cls

def calculate_ccrs(comment_lines_count, code_text, alpha=3, k=5):
    if code_text is None:
        return 0
    
    total_lines = len(code_text.splitlines())
    if total_lines == 0:
        return 0
        
    r = comment_lines_count / total_lines
    
    if r < 0.2:
        return (r / 0.2) ** alpha
    elif 0.2 <= r <= 0.3:
        return 1
    else:
        return math.exp(-k * (r - 0.3))
    
    
def load_data_from_jsonl(file_path):
    """从jsonl文件加载数据"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def calculate_comment_quality_score(input_file, output_file):
    """计算注释质量分数并将结果保存到输出文件"""
    print(f"开始处理数据文件: {input_file}")
    
    data = []
    try:
        data = load_data_from_jsonl(input_file)
        print(f"成功加载数据，共 {len(data)} 条记录")
    except Exception as e:
        print(f"加载数据文件时出错: {str(e)}")
        traceback.print_exc()
        return
    
    # 结果列表和缓存的分组结果
    results = []
    grouped_results = {}
    
    for item_index, item in enumerate(data):
        print(f"\n处理第 {item_index+1}/{len(data)} 条数据...")
        
        try:
            # 提取必要信息
            index = item.get("index", item_index)  # 如果没有index，使用循环索引
            
            # 检查是否为需要跳过的索引
            if index == 833:
                print(f"  跳过索引为{index}的数据项")
                # 为跳过的数据添加结果，得分为0
                results.append({
                    "index": index,
                    "programming_language": item.get("programming_language", "python"),
                    "results": {model: 0 for model in item.get("results", {}).keys()}
                })
                
                # 每处理5条数据保存一次
                if (item_index + 1) % 5 == 0:
                    save_partial_results(results, output_file)
                    print(f"已保存 {len(results)} 条结果到 {output_file}")
                
                continue
                
            programming_language = item.get("programming_language", "python")  # 默认为python
            
            print(f"  项目索引: {index}, 语言: {programming_language}")
            
            # 获取不同模型的结果
            model_results = item.get("results", {})
            print(f"  该项目有 {len(model_results)} 个模型结果")
            
            model_scores = {}
            
            for model_name, response in model_results.items():
                try:
                    print(f"    处理模型: {model_name}")
                    # 设置模型处理超时（30秒）
                    start_time = time.time()
                    
                    solution_code = extract_solution(response)
                    
                    # 检查处理时间
                    if time.time() - start_time > 30:
                        print(f"    警告：模型 {model_name} 处理超时，跳过")
                        model_scores[model_name] = 0
                        continue
                    
                    if not solution_code:
                        print(f"    警告：模型 {model_name} 无法提取代码")
                        model_scores[model_name] = 0
                        continue
                    
                    lang = normalize_language(programming_language)
                    print(f"    模型 {model_name}: 提取到代码，标准化语言为 {lang}")
                    
                    # 检查处理时间
                    if time.time() - start_time > 30:
                        print(f"    警告：模型 {model_name} 处理超时，跳过")
                        model_scores[model_name] = 0
                        continue
                    
                    if lang == "python":
                        comments, comment_lines_count = extract_comment_in_python(solution_code)
                    else:
                        comments, comment_lines_count = extract_comment_in_java_cpp(solution_code)
                    
                    # 检查处理时间
                    if time.time() - start_time > 30:
                        print(f"    警告：模型 {model_name} 处理超时，跳过")
                        model_scores[model_name] = 0
                        continue
                    
                    print(f"    模型 {model_name}: 提取到 {len(comments)} 条注释，共 {comment_lines_count} 行")
                    
                    ids = extract_identifiers_in_code(solution_code, lang)
                    comment_word = extract_terms_from_comment(comments)
                    cic_score = round(calculate_cic(comment_word, ids), 3)
                    cls_score = round(calculate_average_cls(comments), 3)
                    ccrs_score = round(calculate_ccrs(comment_lines_count, solution_code), 3)
                    
                    comment_quality_score = 0.7 * cic_score + 0.3 * cls_score 
                    final_score = round(comment_quality_score * 30, 1)
                    model_scores[model_name] = final_score
                    print(f"    模型 {model_name}: 计算得分 {final_score}")
                except Exception as e:
                    print(f"    处理模型 {model_name} 时出错: {str(e)}")
                    model_scores[model_name] = 0
            
            results.append({
                "index": index,
                "programming_language": programming_language,
                "results": model_scores
            })
            
            # 每处理5条数据保存一次
            if (item_index + 1) % 5 == 0:
                save_partial_results(results, output_file)
                print(f"已保存 {len(results)} 条结果到 {output_file}")
                
        except Exception as e:
            print(f"处理数据项 {item_index} 时出错: {str(e)}")
            # 保存当前结果并记录错误
            results.append({
                "index": item.get("index", item_index),
                "programming_language": item.get("programming_language", "python"),
                "results": {"error": str(e)}
            })
            # 错误后立即保存
            save_partial_results(results, output_file)
            continue
    
    # 保存所有结果
    save_partial_results(results, output_file)
    print(f"数据处理完成，结果已保存到 {output_file}")

def save_partial_results(results, output_file):
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
        print(f"已保存 {len(results)} 条结果到 {output_file}")
    except Exception as e:
        print(f"保存结果时出错: {str(e)}")

if __name__ == "__main__":   
    try:
        print("开始加载数据...")
        input_file = "6model_result.jsonl"
        output_file = "calculate_comment_quality_score.jsonl"
        
        # 创建或清空输出文件
        with open(output_file, 'w', encoding='utf-8') as f:
            pass  # 只是创建或清空文件
            
        print(f"已创建输出文件: {output_file}")
        
        print(f"开始计算注释质量分数...")
        calculate_comment_quality_score(input_file, output_file)
        print("处理完成")
    except Exception as e:
        print(f"程序执行出错: {str(e)}")
        traceback.print_exc()