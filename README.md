# CodeSentinel-Four-in-One-Automated-Code-Evaluation-System

## Project Core Advantages (项目核心优势)

The core strength of this code evaluation framework can be summarized as a **"Four-in-One Automated Evaluation System"**. It integrates four key characteristics that make it powerful and unique.

---

### 1. Comprehensive & Multi-Dimensional 

The framework covers **seven crucial dimensions** of code quality: Functionality, Modularity, Simplicity, Standardization, Robustness, Efficiency, and Comment Quality. This scope goes far beyond typical linters or complexity analysis tools, enabling a holistic assessment of code from high-level architecture to low-level implementation details.


---

### 2. Advanced & In-Depth Methodologies

It creatively combines four different levels of analysis techniques, creating a deeply insightful evaluation process:

*   **AI-Powered Semantic Analysis**: Leverages the `CodeBERT` model to understand the *intent* behind the code, assessing if its functionality aligns with the requirements. This marks a leap from "syntactically correct" to "semantically correct."
*   **Dynamic Execution & Testing**: Verifies code correctness by actually running it in a secure sandbox with auto-generated unit tests. This serves as a "gold standard" for functional validation.
*   **Classic Software Metrics**: Integrates battle-tested software engineering metrics like Modularity (Coupling & Cohesion), Cyclomatic Complexity (CC), and Maintainability Index (MI), ensuring the evaluation is grounded in established principles.
*   **Static Rule-Based Scanning**: Efficiently checks for adherence to language-specific best practices and common standards through pattern matching.

---

### 3. Quantitative & Objective Results 

One of its most significant highlights is the ability to distill complex code quality issues into a clear **30-point score** for each dimension. This makes code quality tangible, comparable, and trackable, providing objective data for developers to improve their code or for assessing the capabilities of different code generation models.

---

### 4. Automated & Extensible Workflow 

The entire evaluation process is designed as an automated pipeline of scripts that can batch-process code submissions. This allows for seamless integration into Continuous Integration (CI) workflows or large-scale model evaluation systems. Furthermore, the project has a clean, modular structure where each dimension is a self-contained component, making it highly **extensible** for adding new evaluation dimensions in the future.

## A automated Code Evaluation Framework 

## Introduction

This project provides a comprehensive framework for automatically evaluating code quality across seven key dimensions. By leveraging a combination of static and dynamic analysis techniques, including AI-based semantic analysis and traditional software metrics, the framework offers a multi-faceted and in-depth assessment of source code written in Python, Java, and C++. Each dimension is scored on a 30-point scale, providing a quantitative measure of code quality.


---

## Evaluation Dimensions 

The framework evaluates code based on the following seven dimensions:

1.  **Functionality (功能性)**
2.  **Modularity (模块化)**
3.  **Simplicity (简洁性)**
4.  **Standardization (规范性)**
5.  **Robustness (鲁棒性)**
6.  **Efficiency (效率)**
7.  **Comment Quality (注释质量)**

---

### 1. Functionality (功能性)

This dimension assesses whether the code correctly and completely implements the required functionalities. It combines AI-based semantic understanding with dynamic execution analysis.

*   **Methodology (方法论):**
    *   **Semantic Similarity**: Uses the `CodeBERT` model to calculate the cosine similarity between the vector embeddings of the task description and the generated code. A higher score indicates the code is more semantically aligned with the requirements.
    *   **Unit Testing**: Dynamically executes the code in a secure sandbox environment. It automatically generates simple test cases based on the task description and calculates the pass rate.
    *   **Code Coverage**: Measures the percentage of code lines executed by the test cases (currently for Python only) using the `coverage.py` library.
    *   **Execution Success**: A binary check to see if the code compiles and runs without any runtime errors.

*   **Final Score (最终得分):**
    The final score is a weighted average of the four metrics above, mapped to a 30-point scale.
    `Score = (Semantic * 0.4 + Testing * 0.3 + Coverage * 0.2 + Execution * 0.1) * 30`

### 2. Modularity (模块化)

This dimension evaluates the code's structure, focusing on low coupling and high cohesion, which are essential for maintainability and reusability.


*   **Methodology (方法论):**
    The evaluation is based on established software engineering metrics, primarily calculated through static analysis (AST parsing for Python, regex for Java/C++).
    *   **Coupling (耦合度)**:
        *   **Efferent Coupling (CE)**: Measures how many other modules this code depends on. Approximated by the number of imported modules.
        *   **Afferent Coupling (CA)**: Measures how many other modules depend on this code. Estimated based on code features like the number of classes and methods.
    *   **Cohesion (内聚度)**:
        *   **Lack of Cohesion in Methods (LCOM2)**: Measures how well methods within a class belong together. It's calculated by analyzing the sets of instance variables used by different methods. Less shared variables lead to a higher LCOM2 value and lower cohesion.
    *   **Complexity Factor**: A multiplier that adjusts the score based on code length, nesting depth, and the number of functions/classes.

*   **Final Score (最终得分):**
    The final score is a weighted average of the CE, CA, and LCOM2 scores, adjusted by the complexity factor and mapped to a 30-point scale.

### 3. Simplicity (简洁性)

This dimension assesses how easy the code is to understand and maintain.


*   **Methodology (方法论):**
    The core of this evaluation is the **Maintainability Index (MI)**, a metric that provides a single value for maintainability. The script calculates MI using simplified versions of its components.
    *   **Halstead Volume (HV)**: A measure of program size and complexity based on operators and operands. It's estimated here using an empirical formula based on LOC.
    *   **Cyclomatic Complexity (CC)**: Measures the number of independent paths in the code. It's approximated by counting control flow keywords (e.g., `if`, `for`, `while`).
    *   **Lines of Code (LOC)**: The number of effective lines of code, excluding comments and blank lines.

*   **Final Score (最终得分):**
    The MI value (ranging from 0-100) is calculated using the classic formula: `MI = 171 - 5.2 * log(HV) - 0.23 * CC - 16.2 * log(LOC)`. This value is then mapped to a 5-point scale, which is finally converted to a 30-point scale.

### 4. Standardization (规范性)

This dimension checks if the code adheres to common coding standards and best practices for the specific language.


*   **Methodology (方法论):**
    A rule-based static analysis approach is used. The script maintains a list of common coding anti-patterns for Python, Java, and C++. It scans the code for violations and deducts points for each one found.

*   **Metrics (评估指标):**
    Checks include, but are not limited to:
    *   Improper naming conventions.
    *   Excessive line length.
    *   Use of `print` instead of a proper logger.
    *   Catching overly broad exceptions (e.g., `except:` in Python).
    *   Using wildcard imports (`import *`).
    *   Inconsistent indentation.

*   **Final Score (最终得分):**
    The code starts with a perfect score, and points are deducted for each violation. The final result is then normalized to a 30-point scale.

### 5. Robustness (鲁棒性)

This dimension evaluates the code's ability to handle errors, exceptional conditions, and invalid inputs.


*   **Methodology (方法论):**
    The assessment is a hybrid approach, combining a high-level maintainability score with checks for specific robustness-enhancing patterns.
    *   **Maintainability Index (MI)**: This script also calculates MI, contributing 50% to the final score. It uses the `radon` library for a more accurate MI calculation for Python.
    *   **Robustness Feature Checks**: The script statically scans the code for key robustness features:
        *   **Error Handling**: Presence of `try-except/catch-finally` blocks.
        *   **Input Validation**: Presence of checks for inputs (e.g., `isinstance()`, `null` checks).
        *   **Resource Management**: Use of safe resource handling patterns (e.g., `with` statements in Python, smart pointers in C++).

*   **Final Score (最终得分):**
    The final score is an average of the MI score and the robustness feature score, mapped to a 30-point scale.

### 6. Efficiency (效率)

This dimension provides a static, indirect measure of code efficiency by analyzing its structural complexity and potential resource consumption.


*   **Methodology (方法论):**
    The evaluation is based on a custom metric called **Enhanced Cyclomatic Complexity (ECC)**. This metric aims to quantify the "density" of operations per line of code.
    `ECC = (Num of Methods + Num of Statements + Max Inputs + Max Outputs) / Lines of Code`
    *   A lower ECC value suggests that the code is structurally simpler and potentially more efficient, resulting in a higher score.

*   **Implementation Details (实现细节):**
    *   The **Python** analyzer uses the `ast` module for accurate metric extraction.
    *   The **Java** and **C++** analyzers use simplified, regex-based approximations.

*   **Final Score (最终得分):**
    The calculated ECC value is mapped linearly to a 30-point scale, where a lower ECC yields a higher score.

### 7. Comment Quality (注释质量)

This dimension evaluates the quality and relevance of the code's comments.


*   **Methodology (方法论):**
    The score is based on three metrics derived from static analysis of comments and code.
    *   **Comment-Identifier Consistency (CIC)**: This is the most heavily weighted metric (70%). It measures the relevance of a comment to the code it describes by calculating the Jaccard similarity between the words in the comments and the identifiers (variable/function names) in the code.
    *   **Comment Length Score (CLS)**: This metric (weighted at 30%) scores comments based on their length. Comments that are too short or too long receive lower scores.
    *   **Comment-to-Code Ratio Score (CCRS)**: This metric is calculated but **not used** in the final score. It assesses if the overall amount of comments is reasonable relative to the code size.

*   **Final Score (最终得分):**
    The final score is a weighted average of the CIC and CLS scores, mapped to a 30-point scale.
    `Score = (CIC * 0.7 + CLS * 0.3) * 30` 
