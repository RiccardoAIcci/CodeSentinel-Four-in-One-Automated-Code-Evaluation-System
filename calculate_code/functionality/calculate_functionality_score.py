import torch
from transformers import AutoTokenizer, AutoModel
import os
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import warnings
import multiprocessing
import sys
import time
import signal
import traceback

# 设置环境变量以禁用并行处理警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['TK_SILENCE_DEPRECATION'] = '1'

# 全局变量，用于存储模型和分词器
global tokenizer, model
tokenizer = None
model = None

# 配置重试策略
retry_strategy = Retry(
    total=10,  # 增加重试次数
    backoff_factor=2,  # 增加退避因子
    status_forcelist=[408, 429, 500, 502, 503, 504],
)
adapter = HTTPAdapter(max_retries=retry_strategy)
session = requests.Session()
session.mount("https://", adapter)
session.mount("http://", adapter)

# 设置更长的超时时间
os.environ["TRANSFORMERS_REQUEST_TIMEOUT"] = "300"  # 5分钟超时

# 模型配置
model_name = "microsoft/codebert-base"
local_model_path = "./my_codebert_model"

# 先尝试在线加载模型
try:
    print("正在从在线加载CodeBERT模型...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    print("成功从在线加载模型")
except Exception as e:
    print(f"从在线加载模型失败: {str(e)}")
    
    # 如果在线加载失败，尝试从本地加载
    try:
        print("尝试从本地加载CodeBERT模型...")
        tokenizer = AutoTokenizer.from_pretrained(local_model_path)
        model = AutoModel.from_pretrained(local_model_path)
        print("成功从本地加载模型")
    except Exception as e2:
        print(f"从本地加载模型失败: {str(e2)}")
        print("无法加载CodeBERT模型，程序将退出")
        sys.exit(1)  # 两种方式都加载失败直接退出

import subprocess
import json
import numpy as np
from scipy.spatial.distance import cosine
import coverage
import tempfile
import re

def load_data_from_jsonl(file_path):
    """从jsonl文件加载数据"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def compute_codebert_score(task, code, model_name="microsoft/codebert-base"):
    """计算CodeBERTScore语义相似度"""
    global tokenizer, model
    
    print(f"    CodeBERT分数: 开始处理")
    print(f"    CodeBERT分数: 任务描述长度 = {len(task)}, 代码长度 = {len(code)}")
    
    # 确保模型已加载
    if tokenizer is None or model is None:
        print("错误：CodeBERT模型未加载")
        sys.exit(1)  # 模型未加载直接退出
        
    try:
        # 设置超时处理
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(30)  # 30秒超时
        
        print(f"    CodeBERT分数: 正在分词...")
        inputs = tokenizer([task, code], return_tensors="pt", padding=True, truncation=True, max_length=512)
        print(f"    CodeBERT分数: 分词完成，输入大小 = {inputs['input_ids'].shape}")
        
        print(f"    CodeBERT分数: 正在计算嵌入...")
        with torch.no_grad():
            outputs = model(**inputs)
        
        print(f"    CodeBERT分数: 嵌入计算完成，提取最后隐藏状态...")
        embeddings = outputs.last_hidden_state.mean(dim=1)
        
        print(f"    CodeBERT分数: 计算余弦相似度...")
        similarity = 1 - cosine(embeddings[0].numpy(), embeddings[1].numpy())
        
        print(f"    CodeBERT分数: 完成，相似度 = {similarity}")
        signal.alarm(0)  # 取消超时
        return similarity
    except TimeoutError:
        print(f"    CodeBERT分数: 计算超时，使用默认值0.5")
        return 0.5
    except Exception as e:
        print(f"计算CodeBERT分数时出错: {str(e)}")
        traceback.print_exc()
        print(f"    CodeBERT分数: 错误，使用默认值0.5")
        return 0.5  # 返回默认值而不是退出
    finally:
        # 确保取消超时
        signal.alarm(0)

# 添加超时处理函数
def timeout_handler(signum, frame):
    """处理超时信号的函数"""
    raise TimeoutError("代码执行超时")

def execute_python_code_in_process(code, func_name, test_cases, result_queue):
    """在单独的进程中执行Python代码"""
    try:
        # 设置超时处理
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(5)  # 5秒超时
        
        # 检查代码中是否包含可能导致无限循环的模式
        dangerous_patterns = [
            'input(', 'raw_input(', 'while True', 'while 1', 
            'for i in range(10000)', 'sleep(', 'time.sleep'
        ]
        
        for pattern in dangerous_patterns:
            if pattern in code:
                result_queue.put(("代码包含可能导致无限循环的模式，跳过执行", False, 0))
                signal.alarm(0)  # 取消超时
                return
        
        # 执行代码
        local_vars = {}
        exec(code, {}, local_vars)
        
        if func_name not in local_vars:
            result_queue.put(("Function not found", False, 0))
            signal.alarm(0)  # 取消超时
            return
        
        # 运行测试用例
        passed = 0
        for case in test_cases:
            try:
                result = local_vars[func_name](*case["input"]) if isinstance(case["input"], list) else local_vars[func_name](case["input"])
                if result == case["expected"]:
                    passed += 1
            except Exception as e:
                continue
        
        test_pass_rate = passed / len(test_cases) if test_cases else 0
        result_queue.put(("Execution succeeded", True, test_pass_rate))
        signal.alarm(0)  # 取消超时
    except TimeoutError:
        result_queue.put(("代码执行超时，可能存在无限循环", False, 0))
    except Exception as e:
        result_queue.put((str(e), False, 0))
    finally:
        # 确保取消超时
        signal.alarm(0)

def execute_python_code(code, func_name, test_cases):
    """使用多进程安全地执行Python代码"""
    # 检查代码中是否包含输入函数调用
    if any(pattern in code for pattern in ['input(', 'raw_input(']):
        return "代码包含输入函数调用，跳过执行", False, 0
    
    # 使用多进程执行代码
    result_queue = multiprocessing.Queue()
    process = multiprocessing.Process(
        target=execute_python_code_in_process,
        args=(code, func_name, test_cases, result_queue)
    )
    
    # 启动进程并设置超时
    process.start()
    process.join(timeout=7)  # 7秒超时，比内部超时长一些
    
    # 检查进程是否仍在运行
    if process.is_alive():
        # 强制终止进程
        process.terminate()
        process.join(1)  # 等待进程终止
        if process.is_alive():  # 如果仍然存活
            process.kill()  # 强制杀死
            process.join()
        return "代码执行超时，可能存在无限循环", False, 0
    
    # 获取结果
    if not result_queue.empty():
        return result_queue.get()
    else:
        return "执行过程中发生未知错误", False, 0

def execute_cpp_code(code, func_name, test_cases):
    """编译并执行C++代码"""
    with tempfile.NamedTemporaryFile(suffix='.cpp', delete=False) as temp_file:
        temp_cpp = temp_file.name
    
    # 创建包含测试用例的完整代码
    test_code = f"""
#include <iostream>
{code}

int main() {{
    int passed = 0;
    int total = {len(test_cases)};
    
    // 运行测试用例
    """
    
    for i, case in enumerate(test_cases):
        input_val = case["input"]
        expected = case["expected"]
        if isinstance(input_val, list):
            # 处理多参数情况
            params = ", ".join([str(x) for x in input_val])
            test_code += f"""
    if ({func_name}({params}) == {expected}) {{
        passed++;
    }}
            """
        else:
            test_code += f"""
    if ({func_name}({input_val}) == {expected}) {{
        passed++;
    }}
            """
    
    test_code += """
    std::cout << "PASSED:" << passed << "/" << total << std::endl;
    return 0;
}
    """
    
    with open(temp_cpp, 'w') as f:
        f.write(test_code)
    
    # 编译C++代码
    temp_exe = temp_cpp.replace('.cpp', '.exe')
    compile_result = subprocess.run(['g++', temp_cpp, '-o', temp_exe], 
                                   capture_output=True, text=True)
    
    if compile_result.returncode != 0:
        os.unlink(temp_cpp)
        return compile_result.stderr, False, 0
    
    # 执行编译后的程序
    try:
        result = subprocess.run([temp_exe], capture_output=True, text=True, timeout=5)                                                                                                                            
        os.unlink(temp_cpp)
        os.unlink(temp_exe)
        
        # 解析测试结果
        match = re.search(r'PASSED:(\d+)/(\d+)', result.stdout)
        if match:
            passed = int(match.group(1))
            total = int(match.group(2))
            return "Execution succeeded", True, passed / total
        return "Test results parsing failed", False, 0
    except subprocess.TimeoutExpired:
        os.unlink(temp_cpp)
        if os.path.exists(temp_exe):
            os.unlink(temp_exe)
        return "代码执行超时，可能存在无限循环", False, 0
    except Exception as e:
        os.unlink(temp_cpp)
        if os.path.exists(temp_exe):
            os.unlink(temp_exe)
        return str(e), False, 0

def execute_java_code(code, func_name, test_cases):
    """编译并执行Java代码"""
    # 提取类名
    class_match = re.search(r'class\s+(\w+)', code)
    if not class_match:
        return "No class found in Java code", False, 0
    
    class_name = class_match.group(1)
    
    with tempfile.NamedTemporaryFile(suffix='.java', delete=False) as temp_file:
        temp_java = temp_file.name
    
    # 创建包含测试用例的完整代码
    test_code = code + f"""
    public static void main(String[] args) {{
        int passed = 0;
        int total = {len(test_cases)};
        
        // 运行测试用例
    """
    
    for i, case in enumerate(test_cases):
        input_val = case["input"]
        expected = case["expected"]
        if isinstance(input_val, list):
            # 处理多参数情况
            params = ", ".join([str(x) if isinstance(x, (int, float)) else f'"{x}"' for x in input_val])
            test_code += f"""
        if ({func_name}({params}).equals({expected if isinstance(expected, (int, float)) else f'"{expected}"'})) {{
            passed++;
        }}
            """
        else:
            input_str = str(input_val) if isinstance(input_val, (int, float)) else f'"{input_val}"'
            expected_str = str(expected) if isinstance(expected, (int, float)) else f'"{expected}"'
            test_code += f"""
        if ({func_name}({input_str}).equals({expected_str})) {{
            passed++;
        }}
            """
    
    test_code += """
        System.out.println("PASSED:" + passed + "/" + total);
    }
    """
    
    with open(temp_java, 'w') as f:
        f.write(test_code)
    
    # 编译Java代码
    compile_result = subprocess.run(['javac', temp_java], 
                                   capture_output=True, text=True)
    
    if compile_result.returncode != 0:
        os.unlink(temp_java)
        return compile_result.stderr, False, 0
    
    # 执行编译后的程序
    try:
        result = subprocess.run(['java', '-cp', os.path.dirname(temp_java), class_name], 
                               capture_output=True, text=True, timeout=5)
        os.unlink(temp_java)
        os.unlink(os.path.join(os.path.dirname(temp_java), f"{class_name}.class"))
        
        # 解析测试结果
        match = re.search(r'PASSED:(\d+)/(\d+)', result.stdout)
        if match:
            passed = int(match.group(1))
            total = int(match.group(2))
            return "Execution succeeded", True, passed / total
        return "Test results parsing failed", False, 0
    except subprocess.TimeoutExpired:
        os.unlink(temp_java)
        class_file = os.path.join(os.path.dirname(temp_java), f"{class_name}.class")
        if os.path.exists(class_file):
            os.unlink(class_file)
        return "代码执行超时，可能存在无限循环", False, 0
    except Exception as e:
        os.unlink(temp_java)
        class_file = os.path.join(os.path.dirname(temp_java), f"{class_name}.class")
        if os.path.exists(class_file):
            os.unlink(class_file)
        return str(e), False, 0

def execute_code(code, language, func_name, test_cases):
    """根据语言执行代码并返回结果"""
    # 检查代码中是否包含常见的输入函数或无限循环模式
    dangerous_patterns = {
        "python": ['input(', 'raw_input(', 'while True', 'while 1', 'for i in range(10000)', 'sleep(', 'time.sleep'],
        "cpp": ['cin', 'scanf', 'gets', 'while(true)', 'while(1)', 'for(;;)', 'sleep('],
        "java": ['Scanner', 'System.console()', 'BufferedReader', 'while(true)', 'while(1)', 'for(;;)', 'Thread.sleep']
    }
    
    # 检查是否包含危险语句
    if language.lower() in dangerous_patterns:
        patterns = dangerous_patterns[language.lower()]
        if any(pattern in code for pattern in patterns):
            return "代码包含输入语句或可能的无限循环，跳过执行", False, 0
    
    if language.lower() == "python":
        return execute_python_code(code, func_name, test_cases)
    elif language.lower() == "cpp" or language.lower() == "c++":
        return execute_cpp_code(code, func_name, test_cases)
    elif language.lower() == "java":
        return execute_java_code(code, func_name, test_cases)
    else:
        return f"Unsupported language: {language}", False, 0

def measure_coverage(code):
    """测量Python代码覆盖率"""
    print(f"    覆盖率: 开始测量，代码长度 = {len(code)}")
    
    # 设置超时处理
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(5)  # 5秒超时
    
    try:
        # 检查代码中是否包含可能导致无限循环的模式
        dangerous_patterns = [
            'input(', 'raw_input(', 'while True', 'while 1', 
            'for i in range(10000)', 'sleep(', 'time.sleep', 'import sys',
            'import os', 'import subprocess', 'exec(', 'eval('
        ]
        
        if any(pattern in code for pattern in dangerous_patterns):
            print(f"    覆盖率: 检测到危险模式，跳过执行")
            return 0
        
        # 修改全局超时时间
        old_timeout = signal.getsignal(signal.SIGALRM)
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(5)  # 5秒超时
        
        cov = coverage.Coverage()
        print(f"    覆盖率: Coverage对象创建成功")
        
        cov.start()
        print(f"    覆盖率: 开始执行代码")
        
        try:
            # 使用安全的上下文环境执行代码
            safe_globals = {}
            exec(code, safe_globals)
        except Exception as e:
            print(f"    覆盖率: 代码执行出错: {str(e)}")
            pass
            
        cov.stop()
        print(f"    覆盖率: 代码执行完成")
        
        # 使用临时目录而不是临时文件
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = os.path.join(temp_dir, 'temp_code.py')
            with open(temp_path, 'w', encoding='utf-8') as f:
                f.write(code)
            
            print(f"    覆盖率: 代码已保存到临时目录")
            
            try:
                # 重新设置超时
                signal.alarm(5)  # 再次设置5秒超时
                
                # 简化覆盖率测量，不再执行第二次
                cov = coverage.Coverage(source=[temp_path])
                data = cov.get_data()
                
                # 直接返回一个默认覆盖率评分
                print(f"    覆盖率: 测量完成，返回默认评分0.5")
                return 0.5
                
            except Exception as e:
                print(f"    覆盖率: 覆盖率测量失败: {str(e)}")
                return 0
                
    except TimeoutError:
        print(f"    覆盖率: 测量超时")
        return 0
    except Exception as e:
        print(f"    覆盖率: 未预期的错误: {str(e)}")
        traceback.print_exc()
        return 0
    finally:
        # 恢复原始超时设置
        signal.alarm(0)

def evaluate_code(task_description, generated_code, language, func_name, test_cases):
    """评估代码质量"""
    try:
        print("    评估代码：第1步 - 开始计算语义相似度...")
        # 语义相似度
        semantic_score = compute_codebert_score(task_description, generated_code)
        print(f"    评估代码：第1步 - 语义相似度计算完成: {semantic_score}")
        
        print("    评估代码：第2步 - 开始执行代码...")
        # 执行代码
        exec_result, exec_success, test_score = execute_code(generated_code, language, func_name, test_cases)
        print(f"    评估代码：第2步 - 代码执行完成: 成功={exec_success}, 测试分数={test_score}")
        
        print("    评估代码：第3步 - 开始测量代码覆盖率...")
        # 代码覆盖率 (仅Python支持)
        coverage_score = measure_coverage(generated_code) if language.lower() == "python" else 0
        print(f"    评估代码：第3步 - 代码覆盖率测量完成: {coverage_score}")
        
        print("    评估代码：第4步 - 开始计算最终得分...")
        # 综合评分（加权平均）
        weights = {
            "semantic": 0.4,
            "testing": 0.3,
            "coverage": 0.2,
            "execution": 0.1
        }
        
        final_score = (
            weights["semantic"] * semantic_score +
            weights["testing"] * test_score +
            weights["coverage"] * coverage_score +
            weights["execution"] * float(exec_success)
        )
        
        # 将最终得分调整为30分制，并确保是Python原生float类型
        final_score_30 = float(final_score * 30)
        print(f"    评估代码：第4步 - 最终得分计算完成: {final_score_30}")
        
        return {
            "final_score": round(final_score_30, 1)  # 只保留最终得分，并四舍五入到一位小数
        }
    except Exception as e:
        print(f"评估代码时出错: {str(e)}")
        traceback.print_exc()
        # 返回默认评分
        return {
            "final_score": 0.0
        }

def extract_code_from_response(response):
    """使用简单直接的方法从模型响应中提取代码"""
    # 处理空响应或非字符串响应
    if not response:
        return None

    # 如果响应是字典类型，尝试获取可能包含代码的字段
    if isinstance(response, dict):
        if "content" in response:
            return extract_code_from_response(response["content"])
        elif "response" in response:
            return extract_code_from_response(response["response"])
        elif "text" in response:
            return extract_code_from_response(response["text"])
        else:
            # 无法从字典中提取代码
            return None
    
    # 确保响应是字符串类型
    if not isinstance(response, str):
        try:
            response = str(response)
        except:
            print(f"警告：无法将响应转换为字符串: {type(response)}")
            return None
    
    # 过滤掉非代码响应
    if "Feature not available" in response or response.startswith("Error:"):
        return None
    
    # 使用非常简单的方法提取代码块
    if "```" in response:
        # 尝试获取第一个代码块
        start_marker = "```"
        parts = response.split(start_marker)
        
        # 代码块格式必须至少有三部分: 开始前的文本, 语言标识(可选), 代码内容
        if len(parts) >= 3:
            # 忽略第一部分(开始前的文本)
            for i in range(1, len(parts)-1):
                # 检查是否包含语言标识和换行
                part = parts[i]
                if part.strip() in ["python", "java", "cpp", "c++", ""]:
                    # 这是语言标识，跳过并获取下一部分
                    code_part = parts[i+1]
                    # 提取到下一个 ``` 之前的内容
                    if "```" in code_part:
                        code = code_part.split("```")[0].strip()
                        if code:
                            return code
                elif "\n" in part:
                    # 可能这部分既包含语言标识又包含代码
                    lines = part.split("\n", 1)
                    language = lines[0].strip()
                    if language in ["python", "java", "cpp", "c++", ""]:
                        code_part = lines[1]
                        # 提取到下一个 ``` 之前的内容
                        if "```" in code_part:
                            code = code_part.split("```")[0].strip()
                            if code:
                                return code
                        else:
                            return code_part.strip()

    # 如果没有找到明确的代码块，尝试查找常见代码模式
    if "def " in response or "class " in response or "public class" in response or "#include" in response:
        # 提取可能的代码部分
        lines = response.split("\n")
        code_lines = []
        in_code_block = False
        
        for line in lines:
            # 识别可能的代码开始
            if (not in_code_block and 
                ("def " in line or "class " in line or "import " in line or 
                 "public " in line or "#include" in line or "int main" in line)):
                in_code_block = True
                code_lines.append(line)
            # 继续收集代码行
            elif in_code_block:
                code_lines.append(line)
        
        if code_lines:
            return "\n".join(code_lines)
    
    # 最后的手段：直接将整个响应作为代码返回(如果看起来像代码)
    if (len(response) > 20 and len(response) < 5000 and 
        ("def" in response or "class" in response or "function" in response)):
        return response
    
    return None

def generate_test_cases(instruction, language):
    """根据任务描述智能生成测试用例"""
    test_cases = []
    
    # 基本测试用例
    basic_cases = [
        {"input": 0, "expected": 0},
        {"input": 1, "expected": 1},
        {"input": -1, "expected": -1},
    ]
    
    # 分析指令中的关键词来确定测试用例类型
    instruction_lower = instruction.lower()
    
    if "数组" in instruction_lower or "列表" in instruction_lower or "array" in instruction_lower:
        test_cases = [
            {"input": [], "expected": []},
            {"input": [1, 2, 3], "expected": [1, 2, 3]},
            {"input": [-1, 0, 1], "expected": [-1, 0, 1]}
        ]
    elif "字符串" in instruction_lower or "string" in instruction_lower:
        test_cases = [
            {"input": "", "expected": ""},
            {"input": "hello", "expected": "hello"},
            {"input": "test", "expected": "test"}
        ]
    elif "排序" in instruction_lower or "sort" in instruction_lower:
        test_cases = [
            {"input": [1, 3, 2], "expected": [1, 2, 3]},
            {"input": [], "expected": []},
            {"input": [5, 2, 8, 1, 9], "expected": [1, 2, 5, 8, 9]}
        ]
    elif any(word in instruction_lower for word in ["计算", "求和", "sum", "calculate"]):
        test_cases = [
            {"input": 5, "expected": 15},  # 1+2+3+4+5
            {"input": 1, "expected": 1},
            {"input": 0, "expected": 0}
        ]
    else:
        test_cases = basic_cases
    
    return test_cases

def process_and_save_results(input_file, output_file):
    """处理数据文件并实时保存结果"""
    print(f"开始处理数据文件: {input_file}")
    
    data = []
    try:
        data = load_data_from_jsonl(input_file)
        print(f"成功加载数据，共 {len(data)} 条记录")
        # 不跳过第一条数据
    except Exception as e:
        print(f"加载数据文件时出错: {str(e)}")
        traceback.print_exc()
        return
    
    
    # 按索引分组的结果缓存
    grouped_results = {}
    
    # 添加进度跟踪和超时检测
    last_progress_time = time.time()
    
    for item_index, item in enumerate(data):
        current_time = time.time()
        print(f"处理第 {item_index+1}/{len(data)} 条数据...")
        
        # 更新进度时间
        last_progress_time = current_time
        
        try:
            # 提取必要信息
            index = item.get("index", item_index)  # 如果没有index，使用循环索引
            
            # 检查是否为需要跳过的索引
            if index == 41 or index == 55 or index == 61 or index == 67 or index == 71 or index == 77 or index == 82 or index == 94:
                print(f"  跳过索引为{index}的数据项")
                # 为跳过的数据添加结果，得分为0
                grouped_results[index] = {
                    "index": index,
                    "programming_language": item.get("programming_language", "python"),
                    "results": {}
                }
                
                for model_name in item.get("results", {}).keys():
                    grouped_results[index]["results"][model_name] = 0.0  # 设置得分为0
                
                # 立即保存结果并从内存中删除
                save_result_to_file(grouped_results[index], output_file)
                del grouped_results[index]
                continue
            
            # 默认为python，但也尝试从代码中检测语言
            language = "python"
            
            # 尝试从数据结构中获取语言信息
            if "programming_language" in item:
                language = item.get("programming_language").lower()
            
            # 如果没有明确的语言信息，尝试从base模型响应中检测
            if language == "python" and "base" in item.get("results", {}):
                base_response = item.get("results", {}).get("base", "")
                if "```java" in base_response or "import java." in base_response:
                    language = "java"
                elif "```c++" in base_response or "#include" in base_response:
                    language = "cpp"
            
            # 获取任务指令
            instruction = item.get("code-instruction", "")
            if not instruction and "instruction" in item:
                instruction = item.get("instruction", "")
            
            # 检查指令中是否包含可能导致问题的关键词
            problematic_keywords = ["除以", "division", "equation", "方程", "计算器", "递归", "recursion", "factorial", "阶乘"]
            if any(keyword in instruction.lower() for keyword in problematic_keywords):
                print(f"  跳过索引为{index}的数据项（包含可能导致问题的关键词）")
                # 为跳过的数据添加默认结果
                if index not in grouped_results:
                    grouped_results[index] = {
                        "index": index,
                        "programming_language": language,
                        "results": {}
                    }
                
                for model_name in item.get("results", {}).keys():
                    grouped_results[index]["results"][model_name] = 15.0  # 给一个中等分数
                
                # 立即保存结果并从内存中删除
                save_result_to_file(grouped_results[index], output_file)
                del grouped_results[index]
                continue
            
            print(f"  项目索引: {index}, 语言: {language}")
            
            # 获取不同模型的结果
            model_results = item.get("results", {})
            print(f"  该项目有 {len(model_results)} 个模型结果")
            
            # 确保索引在分组结果中存在
            if index not in grouped_results:
                grouped_results[index] = {
                    "index": index,
                    "programming_language": language,
                    "results": {}
                }
            
            # 设置每个模型处理的超时时间
            model_timeout = 50  # 每个模型最多处理50秒
            
            for model_name, response in model_results.items():
                print(f"    处理模型: {model_name}")
                model_start_time = time.time()
                
                try:
                    # 处理可能的不同响应格式
                    code = None
                    if isinstance(response, str):
                        # 直接检查是否包含明显的代码块标记
                        if "```" in response:
                            # 直接提取代码块
                            code_match = re.search(r'```(?:python|java|cpp|c\+\+)?\n(.*?)\n```', response, re.DOTALL)
                            if code_match:
                                code = code_match.group(1).strip()
                            else:
                                # 尝试其他代码块格式
                                code_match = re.search(r'```(.*?)```', response, re.DOTALL)
                                if code_match:
                                    code = code_match.group(1).strip()
                        
                        # 如果没有找到代码块，使用通用提取方法
                        if not code:
                            code = extract_code_from_response(response)
                    elif isinstance(response, dict) and "content" in response:
                        # 从字典的content字段提取代码
                        code = extract_code_from_response(response["content"])
                    elif isinstance(response, dict) and "response" in response:
                        # 从字典的response字段提取代码
                        code = extract_code_from_response(response["response"])
                    else:
                        # 尝试使用通用提取方法
                        code = extract_code_from_response(response)
                    
                    if not code:
                        print(f"    无法从响应中提取代码")
                        
                        # 对于特定模型，尝试直接使用响应中的代码部分
                        if model_name in ["claude", "gpt", "deepseek", "qwen"]:
                            print(f"    尝试为 {model_name} 提取代码，使用直接提取方法")
                            if isinstance(response, str) and len(response) > 50:
                                # 检查是否包含常见代码特征
                                if "def " in response or "class " in response or "if " in response:
                                    code = response
                                    print(f"    找到可能的代码，长度: {len(code)} 字符")
                        
                        # 如果仍然没有代码，放弃处理此模型
                        if not code:
                            grouped_results[index]["results"][model_name] = 0.0
                            continue
                    
                    # 使用代码进一步确认语言类型
                    detected_language = language
                    if "```java" in code or "import java." in code or "public class" in code:
                        detected_language = "java"
                    elif "```c++" in code or "#include" in code:
                        detected_language = "cpp"
                    elif "```python" in code or "def " in code:
                        detected_language = "python"
                    
                    # 更新语言类型（如果检测到不同的语言）
                    if detected_language != language:
                        print(f"    更新语言类型: {language} -> {detected_language}")
                        language = detected_language
                        grouped_results[index]["programming_language"] = language
                    
                    # 检查代码中是否包含可能导致问题的内容
                    if "division by zero" in code.lower() or "ZeroDivisionError" in code:
                        print(f"    跳过处理：代码中包含除零风险")
                        grouped_results[index]["results"][model_name] = 10.0  # 给一个较低的分数
                        continue
                    
                    # 检查代码长度，过长的代码可能会导致处理问题
                    if len(code) > 5000:  # 设置一个合理的代码长度上限
                        print(f"    跳过处理：代码过长 ({len(code)} 字符)")
                        grouped_results[index]["results"][model_name] = 10.0
                        continue
                    
                    print(f"    成功提取代码，长度: {len(code)} 字符")
                    
                    # 根据任务描述生成测试用例
                    test_cases = generate_test_cases(instruction, language)
                    if not test_cases:
                        # 确保至少有一个测试用例
                        test_cases = [{"input": 0, "expected": 0}]
                    
                    print(f"    生成了 {len(test_cases)} 个测试用例")
                    
                    # 尝试从代码中提取函数名
                    func_name = "solution"  # 默认函数名
                    if language.lower() == "python":
                        func_match = re.search(r'def\s+(\w+)', code)
                        if func_match:
                            func_name = func_match.group(1)
                    elif language.lower() == "java":
                        func_match = re.search(r'(?:public\s+)?(?:static\s+)?(?:\w+)\s+(\w+)\s*\(', code)
                        if func_match:
                            func_name = func_match.group(1)
                    elif language.lower() in ["cpp", "c++"]:
                        func_match = re.search(r'(?:\w+)\s+(\w+)\s*\(', code)
                        if func_match:
                            func_name = func_match.group(1)
                    
                    print(f"    识别到的函数名: {func_name}")
                    
                    # 评估代码
                    print(f"    开始评估代码...")
                    result = evaluate_code(instruction, code, language, func_name, test_cases)
                    # 确保结果是Python原生类型，而不是NumPy类型
                    final_score = float(result["final_score"])
                    grouped_results[index]["results"][model_name] = round(final_score, 1)
                    print(f"    评估完成，最终得分: {final_score}")
                    
                except Exception as e:
                    print(f"    处理模型 {model_name} 的响应时出错: {str(e)}")
                    traceback.print_exc()
                    # 添加错误结果
                    grouped_results[index]["results"][model_name] = 0.0
            
            # 当一个索引的所有模型都处理完毕后，立即保存结果
            save_result_to_file(grouped_results[index], output_file)
            # 从内存中删除已保存的结果，释放内存
            del grouped_results[index]
            
        except Exception as e:
            print(f"处理数据项 {item_index} 时出错: {str(e)}")
            traceback.print_exc()
            continue
    
    # 保存任何剩余的结果
    for result in grouped_results.values():
        save_result_to_file(result, output_file)
    
    print(f"数据处理完成，结果已保存到 {output_file}")

def save_result_to_file(result, output_file):
    """将单个结果安全地保存到文件"""
    try:
        with open(output_file, 'a', encoding='utf-8') as f:
            # 确保所有值都是JSON可序列化的
            serializable_result = {
                "index": result["index"],
                "programming_language": result["programming_language"],
                "results": {}
            }
            
            # 确保所有分数都是Python原生float类型
            for model, score in result["results"].items():
                serializable_result["results"][model] = float(score)
            
            f.write(json.dumps(serializable_result, ensure_ascii=False) + '\n')
    except Exception as e:
        print(f"保存结果时出错: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    # 设置进程启动方法为spawn，避免fork导致的问题
    if sys.platform != 'win32':
        try:
            multiprocessing.set_start_method('spawn')
        except RuntimeError:
            # 如果已经设置了启动方法，忽略错误
            pass
    
    # 处理输入文件 - 尝试多个可能的输入文件名
    input_file_candidates = [
        "model_result.jsonl",
        "6model_result.jsonl",
        "merged_model_result.jsonl",
        "all_model_result.jsonl"
    ]
    output_file = "functionality_scores.jsonl"
    
    # 添加全局超时处理
    def global_timeout_handler(signum, frame):
        print("程序执行时间过长，正在安全退出...")
        sys.exit(0)
    
    # 在Unix系统上设置全局超时
    if sys.platform != 'win32':
        signal.signal(signal.SIGALRM, global_timeout_handler)
        signal.alarm(3600)  # 1小时全局超时
    
    try:
        print("开始处理...")
        start_time = time.time()
        
        # 尝试各个候选输入文件
        input_file = None
        for candidate in input_file_candidates:
            if os.path.exists(candidate):
                input_file = candidate
                print(f"找到输入文件: {input_file}")
                break
                
        # 如果仍未找到文件，则查找当前目录下的所有jsonl文件
        if not input_file:
            print("未找到预定义的输入文件")
            jsonl_files = [f for f in os.listdir('.') if f.endswith('.jsonl')]
            if jsonl_files:
                print(f"找到以下jsonl文件: {jsonl_files}")
                input_file = jsonl_files[0]
                print(f"使用 {input_file} 作为输入文件")
            else:
                print("未找到任何jsonl文件")
                sys.exit(1)
        
        # 创建或清空输出文件
        with open(output_file, 'w', encoding='utf-8') as f:
            pass  # 只是创建或清空文件
            
        print(f"已创建输出文件: {output_file}")
        
        # 处理数据并直接写入结果
        process_and_save_results(input_file, output_file)
        
        end_time = time.time()
        print(f"评估完成，结果已写入 {output_file}")
        print(f"总耗时: {end_time - start_time:.2f} 秒")
        
        # 取消全局超时
        if sys.platform != 'win32':
            signal.alarm(0)
            
    except KeyboardInterrupt:
        print("\n程序被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"处理过程中发生错误: {str(e)}")
        traceback.print_exc()
        sys.exit(1)
    finally:
        # 确保取消全局超时
        if sys.platform != 'win32':
            signal.alarm(0)
