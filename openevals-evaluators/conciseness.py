#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试 OpenEvals CONCISENESS_PROMPT - 简洁性评估示例
"""

import os
from langchain_community.chat_models import ChatTongyi
from openevals.prompts import CONCISENESS_PROMPT
from openevals.llm import create_llm_as_judge

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
if not DASHSCOPE_API_KEY:
    print("错误: 请设置 DASHSCOPE_API_KEY 环境变量")
    exit(1)

eval_llm = ChatTongyi(model_name="qwen-flash", dashscope_api_key=DASHSCOPE_API_KEY, temperature=0)

evaluator = create_llm_as_judge(
    prompt=CONCISENESS_PROMPT,
    feedback_key="conciseness",
    judge=eval_llm,
    continuous=True,
    use_reasoning=False,
)

print("=" * 60)
print("CONCISENESS_PROMPT - 简洁性评估测试")
print("=" * 60)

test_cases = [
    {
        "name": "简洁回答",
        "inputs": "什么是Python？",
        "outputs": "Python是一种高级编程语言。"
    },
    {
        "name": "包含冗余信息",
        "inputs": "什么是Python？",
        "outputs": "嗯，Python是一种高级编程语言，我想你可能想知道更多信息。希望这能帮到你！"
    },
    {
        "name": "包含修饰语",
        "inputs": "什么是Python？",
        "outputs": "我认为Python可能是一种高级编程语言，至少据我所知是这样的。"
    },
]

for i, test_case in enumerate(test_cases, 1):
    print(f"\n【测试 {i}: {test_case['name']}】")
    print(f"输入: {test_case['inputs']}")
    print(f"输出: {test_case['outputs']}")
    
    try:
        result = evaluator(
            inputs=test_case['inputs'],
            outputs=test_case['outputs'],
        )
        
        score = result.get("score") if isinstance(result, dict) else result
        print(f"[OK] 评估分数: {score:.3f}")
    except Exception as e:
        print(f"[ERROR] 评估失败: {e}")

