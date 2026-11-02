#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试 OpenEvals CODE_CORRECTNESS_PROMPT_WITH_REFERENCE_OUTPUTS - 带参考输出的代码正确性评估示例
"""

import os
from langchain_community.chat_models import ChatTongyi
from openevals.prompts import CODE_CORRECTNESS_PROMPT_WITH_REFERENCE_OUTPUTS
from openevals.llm import create_llm_as_judge

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
if not DASHSCOPE_API_KEY:
    print("错误: 请设置 DASHSCOPE_API_KEY 环境变量")
    exit(1)

eval_llm = ChatTongyi(model_name="qwen-flash", dashscope_api_key=DASHSCOPE_API_KEY, temperature=0)

evaluator = create_llm_as_judge(
    prompt=CODE_CORRECTNESS_PROMPT_WITH_REFERENCE_OUTPUTS,
    feedback_key="code_correctness",
    judge=eval_llm,
    continuous=True,
    use_reasoning=False,
)

print("=" * 60)
print("CODE_CORRECTNESS_PROMPT_WITH_REFERENCE_OUTPUTS - 带参考输出的代码正确性评估测试")
print("=" * 60)

test_cases = [
    {
        "name": "与参考输出匹配",
        "inputs": "编写一个函数计算两个数的和",
        "outputs": "def add(a, b):\n    return a + b",
        "reference_outputs": "def add(a, b):\n    return a + b"
    },
    {
        "name": "功能正确但实现不同",
        "inputs": "编写一个函数计算两个数的和",
        "outputs": "def add(x, y):\n    result = x + y\n    return result",
        "reference_outputs": "def add(a, b):\n    return a + b"
    },
    {
        "name": "功能错误",
        "inputs": "编写一个函数计算两个数的和",
        "outputs": "def add(a, b):\n    return a * b",
        "reference_outputs": "def add(a, b):\n    return a + b"
    },
]

for i, test_case in enumerate(test_cases, 1):
    print(f"\n【测试 {i}: {test_case['name']}】")
    print(f"输入: {test_case['inputs']}")
    print(f"输出代码:\n{test_case['outputs']}")
    print(f"参考代码:\n{test_case['reference_outputs']}")
    
    try:
        result = evaluator(
            inputs=test_case['inputs'],
            outputs=test_case['outputs'],
            reference_outputs=test_case['reference_outputs'],
        )
        
        score = result.get("score") if isinstance(result, dict) else result
        print(f"[OK] 评估分数: {score:.3f}")
    except Exception as e:
        print(f"[ERROR] 评估失败: {e}")

