#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试 OpenEvals PLAN_ADHERENCE_PROMPT - 计划遵循度评估示例
"""

import os
from langchain_community.chat_models import ChatTongyi
from openevals.prompts import PLAN_ADHERENCE_PROMPT
from openevals.llm import create_llm_as_judge

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
if not DASHSCOPE_API_KEY:
    print("错误: 请设置 DASHSCOPE_API_KEY 环境变量")
    exit(1)

eval_llm = ChatTongyi(model_name="qwen-flash", dashscope_api_key=DASHSCOPE_API_KEY, temperature=0)

evaluator = create_llm_as_judge(
    prompt=PLAN_ADHERENCE_PROMPT,
    feedback_key="plan_adherence",
    judge=eval_llm,
    continuous=True,
    use_reasoning=False,
)

print("=" * 60)
print("PLAN_ADHERENCE_PROMPT - 计划遵循度评估测试")
print("=" * 60)

test_cases = [
    {
        "name": "完全遵循计划",
        "inputs": "查询天气信息",
        "plan": "1. 获取用户位置\n2. 调用天气API\n3. 返回天气信息",
        "outputs": "步骤1: 获取用户位置 - 完成\n步骤2: 调用天气API - 完成\n步骤3: 返回天气信息 - 完成"
    },
    {
        "name": "部分遵循计划",
        "inputs": "查询天气信息",
        "plan": "1. 获取用户位置\n2. 调用天气API\n3. 返回天气信息",
        "outputs": "步骤1: 获取用户位置 - 完成\n步骤3: 返回天气信息 - 完成"
    },
    {
        "name": "未遵循计划",
        "inputs": "查询天气信息",
        "plan": "1. 获取用户位置\n2. 调用天气API\n3. 返回天气信息",
        "outputs": "直接返回了默认天气信息"
    },
]

for i, test_case in enumerate(test_cases, 1):
    print(f"\n【测试 {i}: {test_case['name']}】")
    print(f"输入: {test_case['inputs']}")
    print(f"计划:\n{test_case['plan']}")
    print(f"实际输出:\n{test_case['outputs']}")
    
    try:
        result = evaluator(
            inputs=test_case['inputs'],
            plan=test_case['plan'],
            outputs=test_case['outputs'],
        )
        
        score = result.get("score") if isinstance(result, dict) else result
        print(f"[OK] 评估分数: {score:.3f} (分数越高表示越遵循计划)")
    except Exception as e:
        print(f"[ERROR] 评估失败: {e}")

