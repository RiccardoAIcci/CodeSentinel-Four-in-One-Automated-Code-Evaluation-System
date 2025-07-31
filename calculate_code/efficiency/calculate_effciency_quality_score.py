import json
import ast
import re
import os
import math
import random
from typing import Dict, List, Tuple, Any
import pandas as pd
import traceback
import time

class CodeAnalyzer:
    """代码分析器基类"""
    def __init__(self):
        self.loc = 0
        self.num_methods = 0
        self.exec_statements = 0
        self.max_inputs = 0
        self.max_outputs = 0
        self.cc = 0
    
    def analyze(self, code: str) -> Dict[str, Any]:
        """分析代码并返回指标"""
        self.count_loc(code)
        return {
            "loc": self.loc,
            "num_methods": self.num_methods,
            "exec_statements": self.exec_statements,
            "max_inputs": self.max_inputs,
            "max_outputs": self.max_outputs,
            "cc": self.cc,
            "ecc": self.calculate_ecc()
        }
    
    def count_loc(self, code: str) -> None:
        """计算代码行数（排除空行和仅含括号的行）"""
        lines = code.strip().split('\n')
        self.loc = sum(1 for line in lines if line.strip() and not re.match(r'^\s*[{}()\[\]]*\s*$', line))
    
    def calculate_ecc(self) -> float:
        """计算增强圈复杂度"""
        if self.loc == 0:
            return 0
        return (self.num_methods + self.exec_statements + self.max_inputs + self.max_outputs) / self.loc
    
    def calculate_risk_level(self, ecc: float) -> str:
        """根据ECC值确定风险等级"""
        if ecc < 1:
            return "低风险"
        elif ecc < 2:
            return "中风险"
        else:
            return "高风险"
    
    def calculate_score(self, ecc: float) -> float:
        """根据ECC值计算30分制得分"""
        if ecc >= 3:
            return 10  # 最低分
        elif ecc <= 0.5:
            return 30  # 最高分
        else:
            # 线性映射: ECC从0.5到3对应分数从30到10
            return 30 - ((ecc - 0.5) / 2.5) * 20

class PythonAnalyzer(CodeAnalyzer):
    """Python代码分析器"""
    def analyze(self, code: str) -> Dict[str, Any]:
        """分析Python代码"""
        self.count_loc(code)
        try:
            tree = ast.parse(code)
            self.num_methods = sum(1 for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)))
            
            # 计算可执行语句
            exec_nodes = [ast.If, ast.For, ast.While, ast.Assign, ast.AugAssign, ast.Expr, 
                         ast.Return, ast.Raise, ast.Assert, ast.Import, ast.ImportFrom]
            self.exec_statements = sum(1 for node in ast.walk(tree) if any(isinstance(node, t) for t in exec_nodes))
            
            # 计算输入输出
            inputs = []
            outputs = []
            
            for node in ast.walk(tree):
                # 检查函数调用
                if isinstance(node, ast.Call):
                    func_name = ""
                    if isinstance(node.func, ast.Name):
                        func_name = node.func.id
                    elif isinstance(node.func, ast.Attribute):
                        func_name = node.func.attr
                    
                    # 检查输入函数
                    if func_name == 'input':
                        inputs.append(len(node.args))
                    
                    # 检查输出函数
                    if func_name == 'print':
                        outputs.append(len(node.args))
                
                # 检查函数定义的参数
                if isinstance(node, ast.FunctionDef):
                    inputs.append(len(node.args.args))
                
                # 检查返回语句
                if isinstance(node, ast.Return) and node.value:
                    outputs.append(1)
            
            self.max_inputs = max(inputs) if inputs else 0
            self.max_outputs = max(outputs) if outputs else 0
            
            # 计算圈复杂度（简化版）
            self.cc = 1 + sum(1 for node in ast.walk(tree) if isinstance(node, (ast.If, ast.For, ast.While, ast.Assert)))
            
            return super().analyze(code)
        except SyntaxError:
            return {
                "loc": self.loc,
                "num_methods": 0,
                "exec_statements": 0,
                "max_inputs": 0,
                "max_outputs": 0,
                "cc": 0,
                "ecc": 0,
                "error": "语法错误，无法解析"
            }

class JavaAnalyzer(CodeAnalyzer):
    """Java代码分析器（简化版）"""
    def analyze(self, code: str) -> Dict[str, Any]:
        """分析Java代码"""
        self.count_loc(code)
        
        # 简单的方法计数
        self.num_methods = len(re.findall(r'(public|private|protected|static|\s) +[\w\<\>\[\]]+\s+(\w+) *\([^\)]*\) *(\{?|[^;])', code))
        
        # 简单的可执行语句计数
        patterns = [r'\bif\s*\(', r'\bfor\s*\(', r'\bwhile\s*\(', r'\bswitch\s*\(', r'\breturn\s+', r'\bthrow\s+', r'=']
        self.exec_statements = sum(len(re.findall(pattern, code)) for pattern in patterns)
        
        # 简单的输入输出计数
        self.max_inputs = max([len(re.findall(r',', params)) + 1 for params in re.findall(r'\([^)]*\)', code) if params.strip('()')], default=0)
        self.max_outputs = len(re.findall(r'System\.out\.print', code)) + len(re.findall(r'\breturn\s+', code))
        
        # 简单的圈复杂度计算
        self.cc = 1 + sum(len(re.findall(pattern, code)) for pattern in [r'\bif\s*\(', r'\bfor\s*\(', r'\bwhile\s*\(', r'\bcase\s+', r'\bcatch\s*\('])
        
        return super().analyze(code)

class CppAnalyzer(CodeAnalyzer):
    """C++代码分析器（简化版）"""
    def analyze(self, code: str) -> Dict[str, Any]:
        """分析C++代码"""
        self.count_loc(code)
        
        # 简单的方法计数
        self.num_methods = len(re.findall(r'([\w\*]+\s+)+(\w+)\s*\([^)]*\)\s*(\{|:)', code))
        
        # 简单的可执行语句计数
        patterns = [r'\bif\s*\(', r'\bfor\s*\(', r'\bwhile\s*\(', r'\bswitch\s*\(', r'\breturn\s+', r'\bthrow\s+', r'=']
        self.exec_statements = sum(len(re.findall(pattern, code)) for pattern in patterns)
        
        # 简单的输入输出计数
        self.max_inputs = max([len(re.findall(r',', params)) + 1 for params in re.findall(r'\([^)]*\)', code) if params.strip('()')], default=0)
        self.max_outputs = len(re.findall(r'std::cout|printf|cout', code)) + len(re.findall(r'\breturn\s+', code))
        
        # 简单的圈复杂度计算
        self.cc = 1 + sum(len(re.findall(pattern, code)) for pattern in [r'\bif\s*\(', r'\bfor\s*\(', r'\bwhile\s*\(', r'\bcase\s+', r'\bcatch\s*\('])
        
        return super().analyze(code)

def get_analyzer(language: str) -> CodeAnalyzer:
    """根据语言获取相应的分析器"""
    analyzers = {
        "python": PythonAnalyzer(),
        "java": JavaAnalyzer(),
        "cpp": CppAnalyzer(),
        "c++": CppAnalyzer()
    }
    return analyzers.get(language.lower(), CodeAnalyzer())

def extract_code_from_response(response: str) -> Tuple[str, str]:
    """从响应中提取代码块和语言类型"""
    # 尝试匹配带有语言标识的代码块
    code_match = re.search(r'```(\w+)?\n(.*?)```', response, re.DOTALL)
    if code_match:
        language = code_match.group(1) or ""
        return code_match.group(2), language.lower()
    
    # 尝试匹配不带语言标识的代码块
    code_match = re.search(r'```(.*?)```', response, re.DOTALL)
    if code_match:
        return code_match.group(1), ""
    
    # 尝试匹配可能的代码段（没有明确的代码块标记）
    # 查找可能包含代码的行（例如，有缩进的多行）
    lines = response.split('\n')
    code_lines = []
    in_code_section = False
    
    for line in lines:
        # 检查是否是可能的代码行
        if re.match(r'^\s*\w+\s*[=({:]', line) or re.match(r'^\s*(def|class|if|for|while|import)\s+', line):
            in_code_section = True
            code_lines.append(line)
        elif in_code_section and line.strip():
            code_lines.append(line)
        elif in_code_section and not line.strip():
            # 空行可能是代码块的一部分
            code_lines.append(line)
    
    if code_lines:
        # 尝试从代码内容推断语言
        code = '\n'.join(code_lines)
        if re.search(r'\bdef\s+|import\s+|class\s+.*:', code):
            return code, "python"
        elif re.search(r'public\s+class|void\s+main|System\.out', code):
            return code, "java"
        elif re.search(r'#include|std::|int\s+main', code):
            return code, "cpp"
        return code, ""
    
    # 如果没有找到明确的代码，返回空字符串
    return "", ""

def load_data_from_jsonl(file_path):
    """从jsonl文件加载数据"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def calculate_efficiency_score(input_file, output_file):
    """处理数据文件并计算效率分数"""
    print(f"开始处理数据文件: {input_file}")
    
    data = []
    try:
        data = load_data_from_jsonl(input_file)
        print(f"成功加载数据，共 {len(data)} 条记录")
    except Exception as e:
        print(f"加载数据文件时出错: {str(e)}")
        traceback.print_exc()
        return
    
    # 结果列表
    results = []
    
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
                    
                    # 检查响应是否已经是错误信息
                    if isinstance(response, str) and response.startswith("Error:"):
                        model_scores[model_name] = 0
                        continue
                    
                    # 从响应中提取代码和语言
                    code, code_language = extract_code_from_response(str(response))
                    
                    # 检查处理时间
                    if time.time() - start_time > 30:
                        print(f"    警告：模型 {model_name} 处理超时，跳过")
                        model_scores[model_name] = 0
                        continue
                        
                    if not code:
                        print(f"    警告：模型 {model_name} 无法提取代码")
                        model_scores[model_name] = 0
                        continue
                    
                    # 确定使用哪种语言的分析器
                    language = code_language if code_language else programming_language
                    
                    # 如果仍然没有语言信息，尝试从代码内容推断
                    if not language:
                        if re.search(r'\bdef\s+|import\s+|class\s+.*:', code):
                            language = "python"
                        elif re.search(r'public\s+class|void\s+main|System\.out', code):
                            language = "java"
                        elif re.search(r'#include|std::|int\s+main', code):
                            language = "cpp"
                    
                    print(f"    模型 {model_name}: 提取到代码，语言为 {language}")
                    
                    # 检查处理时间
                    if time.time() - start_time > 30:
                        print(f"    警告：模型 {model_name} 处理超时，跳过")
                        model_scores[model_name] = 0
                        continue
                        
                    # 分析代码
                    analyzer = get_analyzer(language)
                    analysis = analyzer.analyze(code)
                    
                    # 检查处理时间
                    if time.time() - start_time > 30:
                        print(f"    警告：模型 {model_name} 处理超时，跳过")
                        model_scores[model_name] = 0
                        continue
                        
                    # 检查是否有错误
                    if "error" in analysis:
                        print(f"    警告：模型 {model_name} 代码分析出错")
                        model_scores[model_name] = 0
                        continue
                    
                    # 计算ECC和得分
                    ecc = analysis["ecc"]
                    score = analyzer.calculate_score(ecc)
                    
                    # 添加到模型结果
                    model_scores[model_name] = round(score, 1)
                    
                    # 调试信息
                    print(f"    模型 {model_name}: ECC={ecc}, 得分={score}")
                    
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
    """保存中间结果到输出文件"""
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
        print(f"已保存 {len(results)} 条结果到 {output_file}")
    except Exception as e:
        print(f"保存结果时出错: {str(e)}")

def main():
    """主函数"""
    try:
        print("开始加载数据...")
        input_file = "6model_result.jsonl"
        output_file = "code_efficiency_score.jsonl"
        
        if not os.path.exists(input_file):
            print(f"错误: 找不到输入文件 {input_file}")
            return
        
        # 创建或清空输出文件
        with open(output_file, 'w', encoding='utf-8') as f:
            pass  # 只是创建或清空文件
            
        print(f"已创建输出文件: {output_file}")
        
        print(f"开始分析 {input_file}...")
        calculate_efficiency_score(input_file, output_file)
        print(f"分析完成，结果已保存到 {output_file}")
    except Exception as e:
        print(f"程序执行出错: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
