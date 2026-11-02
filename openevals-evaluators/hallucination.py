#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试 OpenEvals HALLUCINATION_PROMPT - 幻觉检测示例
"""

import os
from langchain_community.chat_models import ChatTongyi
from openevals.prompts import HALLUCINATION_PROMPT
from openevals.llm import create_llm_as_judge

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
if not DASHSCOPE_API_KEY:
    print("错误: 请设置 DASHSCOPE_API_KEY 环境变量")
    exit(1)

eval_llm = ChatTongyi(model_name="qwen-flash", dashscope_api_key=DASHSCOPE_API_KEY, temperature=0)

evaluator = create_llm_as_judge(
    prompt=HALLUCINATION_PROMPT,
    feedback_key="hallucination",
    judge=eval_llm,
    continuous=True,
    use_reasoning=False,
)

print("=" * 60)
print("HALLUCINATION_PROMPT - 幻觉检测测试")
print("=" * 60)

test_cases = [
    {
        "name": "无幻觉（基于上下文）",
        "context": "Python是一种高级编程语言，由Guido van Rossum在1991年发布。",
        "inputs": "Python是什么时候发布的？",
        "outputs": "Python是在1991年发布的。",
        "reference_outputs": "1991年"
    },
    {
        "name": "包含幻觉（未支持的信息）",
        "context": "Python是一种高级编程语言。",
        "inputs": "Python是什么时候发布的？",
        "outputs": "Python是在1991年由Guido van Rossum发布的。",
        "reference_outputs": "1991年"
    },
    {
        "name": "包含幻觉（与上下文矛盾）",
        "context": "Python是一种高级编程语言。",
        "inputs": "Python是什么？",
        "outputs": "Python是一种数据库管理系统。",
        "reference_outputs": "Python是一种高级编程语言"
    },
]

for i, test_case in enumerate(test_cases, 1):
    print(f"\n【测试 {i}: {test_case['name']}】")
    print(f"上下文: {test_case['context']}")
    print(f"输入: {test_case['inputs']}")
    print(f"输出: {test_case['outputs']}")
    
    try:
        result = evaluator(
            context=test_case['context'],
            inputs=test_case['inputs'],
            outputs=test_case['outputs'],
            reference_outputs=test_case.get('reference_outputs'),
        )
        
        score = result.get("score") if isinstance(result, dict) else result
        print(f"[OK] 评估分数: {score:.3f} (分数越高表示幻觉越少)")
    except Exception as e:
        print(f"[ERROR] 评估失败: {e}")

