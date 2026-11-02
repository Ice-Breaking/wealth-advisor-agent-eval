#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试 OpenEvals RAG_GROUNDEDNESS_PROMPT - RAG 基础性评估示例
"""

import os
from langchain_community.chat_models import ChatTongyi
from openevals.prompts import RAG_GROUNDEDNESS_PROMPT
from openevals.llm import create_llm_as_judge

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
if not DASHSCOPE_API_KEY:
    print("错误: 请设置 DASHSCOPE_API_KEY 环境变量")
    exit(1)

eval_llm = ChatTongyi(model_name="qwen-flash", dashscope_api_key=DASHSCOPE_API_KEY, temperature=0)

evaluator = create_llm_as_judge(
    prompt=RAG_GROUNDEDNESS_PROMPT,
    feedback_key="groundedness",
    judge=eval_llm,
    continuous=True,
    use_reasoning=False,
)

print("=" * 60)
print("RAG_GROUNDEDNESS_PROMPT - RAG 基础性评估测试")
print("=" * 60)

test_cases = [
    {
        "name": "基于上下文的输出",
        "context": "Python是一种高级编程语言，由Guido van Rossum在1991年发布。",
        "outputs": "Python是由Guido van Rossum在1991年发布的编程语言。"
    },
    {
        "name": "包含未支持信息",
        "context": "Python是一种高级编程语言。",
        "outputs": "Python是由Guido van Rossum在1991年发布的编程语言，广泛用于数据科学。"
    },
    {
        "name": "与上下文矛盾",
        "context": "Python是一种高级编程语言。",
        "outputs": "Python是一种数据库管理系统。"
    },
]

for i, test_case in enumerate(test_cases, 1):
    print(f"\n【测试 {i}: {test_case['name']}】")
    print(f"上下文: {test_case['context']}")
    print(f"输出: {test_case['outputs']}")
    
    try:
        result = evaluator(
            context=test_case['context'],
            outputs=test_case['outputs'],
        )
        
        score = result.get("score") if isinstance(result, dict) else result
        print(f"[OK] 评估分数: {score:.3f}")
    except Exception as e:
        print(f"[ERROR] 评估失败: {e}")

