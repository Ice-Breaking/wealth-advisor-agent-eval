#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试 OpenEvals RAG_RETRIEVAL_RELEVANCE_PROMPT - RAG 检索相关性评估示例
"""

import os
from langchain_community.chat_models import ChatTongyi
from openevals.prompts import RAG_RETRIEVAL_RELEVANCE_PROMPT
from openevals.llm import create_llm_as_judge

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
if not DASHSCOPE_API_KEY:
    print("错误: 请设置 DASHSCOPE_API_KEY 环境变量")
    exit(1)

eval_llm = ChatTongyi(model_name="qwen-flash", dashscope_api_key=DASHSCOPE_API_KEY, temperature=0)

evaluator = create_llm_as_judge(
    prompt=RAG_RETRIEVAL_RELEVANCE_PROMPT,
    feedback_key="retrieval_relevance",
    judge=eval_llm,
    continuous=True,
    use_reasoning=False,
)

print("=" * 60)
print("RAG_RETRIEVAL_RELEVANCE_PROMPT - RAG 检索相关性评估测试")
print("=" * 60)

test_cases = [
    {
        "name": "高度相关的检索内容",
        "inputs": "什么是机器学习？",
        "context": "机器学习是人工智能的一个分支，通过算法让计算机从数据中学习。"
    },
    {
        "name": "部分相关的检索内容",
        "inputs": "什么是机器学习？",
        "context": "人工智能是一个广泛的领域，包括多个子领域。"
    },
    {
        "name": "不相关的检索内容",
        "inputs": "什么是机器学习？",
        "context": "今天天气很好，适合出门散步。"
    },
]

for i, test_case in enumerate(test_cases, 1):
    print(f"\n【测试 {i}: {test_case['name']}】")
    print(f"输入: {test_case['inputs']}")
    print(f"检索上下文: {test_case['context']}")
    
    try:
        result = evaluator(
            inputs=test_case['inputs'],
            context=test_case['context'],
        )
        
        score = result.get("score") if isinstance(result, dict) else result
        print(f"[OK] 评估分数: {score:.3f}")
    except Exception as e:
        print(f"[ERROR] 评估失败: {e}")

