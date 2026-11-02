#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试 OpenEvals 评估器示例

本文件用于测试评估器的使用，查看评估结果的格式和内容。
"""

import os
from langchain_community.chat_models import ChatTongyi
from openevals.prompts import CORRECTNESS_PROMPT
from openevals.llm import create_llm_as_judge

# ==================== 配置 ====================

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
if not DASHSCOPE_API_KEY:
    print("错误: 请设置 DASHSCOPE_API_KEY 环境变量")
    print("Windows PowerShell: $env:DASHSCOPE_API_KEY='your-key'")
    print("Linux/Mac: export DASHSCOPE_API_KEY='your-key'")
    exit(1)

# ==================== 创建评估 LLM ====================

print("=" * 60)
print("创建评估 LLM...")
print("=" * 60)

eval_llm = ChatTongyi(
    model_name="qwen-flash",
    dashscope_api_key=DASHSCOPE_API_KEY,
    temperature=0
)

print("[OK] 评估 LLM 创建成功\n")

# ==================== 创建评估器 ====================

print("=" * 60)
print("创建正确性评估器...")
print("=" * 60)

evaluator = create_llm_as_judge(
    prompt=CORRECTNESS_PROMPT,
    feedback_key="correctness",
    judge=eval_llm,
    continuous=True,
    use_reasoning=False,
)

print("[OK] 评估器创建成功\n")

# ==================== 测试评估 ====================

print("=" * 60)
print("开始评估测试...")
print("=" * 60)

# 测试用例1: 基本评估
print("\n【测试用例 1】")
print("-" * 60)
inputs_1 = "什么是机器学习？"
outputs_1 = "机器学习是人工智能的一个分支，通过算法让计算机从数据中学习。"
reference_outputs_1 = "机器学习是让计算机从数据中学习的技术。"

print(f"输入: {inputs_1}")
print(f"输出: {outputs_1}")
print(f"参考输出: {reference_outputs_1}")
print("\n正在评估...")

try:
    result_1 = evaluator(
        inputs=inputs_1,
        outputs=outputs_1,
        reference_outputs=reference_outputs_1,
    )
    
    print("\n[OK] 评估完成")
    print(f"\n评估结果类型: {type(result_1)}")
    print(f"评估结果内容:\n{result_1}")
    
    # 尝试提取分数
    if isinstance(result_1, dict):
        print(f"\n结果字典键: {result_1.keys()}")
        if "score" in result_1:
            print(f"分数: {result_1['score']}")
        if "key" in result_1:
            print(f"评估键: {result_1['key']}")
        if "comment" in result_1:
            print(f"评论: {result_1['comment']}")
    elif isinstance(result_1, (int, float)):
        print(f"\n分数: {result_1}")
    elif isinstance(result_1, tuple):
        print(f"\n元组内容: {result_1}")
        if len(result_1) >= 1:
            print(f"分数: {result_1[0]}")
        if len(result_1) >= 2:
            print(f"理由: {result_1[1]}")
    
except Exception as e:
    print(f"\n❌ 评估失败: {e}")
    import traceback
    traceback.print_exc()

# 测试用例2: 不同质量的输出
print("\n\n" + "=" * 60)
print("【测试用例 2】- 测试不同质量的输出")
print("=" * 60)

test_cases = [
    {
        "name": "高质量回答",
        "inputs": "什么是Python？",
        "outputs": "Python是一种高级编程语言，具有简洁的语法和强大的功能。",
        "reference_outputs": "Python是一种高级编程语言。"
    },
    {
        "name": "部分正确回答",
        "inputs": "什么是Python？",
        "outputs": "Python是一种编程语言。",
        "reference_outputs": "Python是一种高级编程语言，具有简洁的语法和强大的功能。"
    },
    {
        "name": "错误回答",
        "inputs": "什么是Python？",
        "outputs": "Python是一种数据库管理系统。",
        "reference_outputs": "Python是一种高级编程语言。"
    },
]

for i, test_case in enumerate(test_cases, 1):
    print(f"\n--- 测试 {i}: {test_case['name']} ---")
    print(f"输入: {test_case['inputs']}")
    print(f"输出: {test_case['outputs']}")
    print(f"参考输出: {test_case['reference_outputs']}")
    
    try:
        result = evaluator(
            inputs=test_case['inputs'],
            outputs=test_case['outputs'],
            reference_outputs=test_case['reference_outputs'],
        )
        
        # 提取分数
        score = None
        if isinstance(result, dict):
            score = result.get("score")
        elif isinstance(result, (int, float)):
            score = result
        elif isinstance(result, tuple) and len(result) > 0:
            score = result[0]
        
        if score is not None:
            print(f"[OK] 评估分数: {score:.3f}")
        else:
            print(f"[WARN] 无法提取分数，结果: {result}")
            
    except Exception as e:
        print(f"[ERROR] 评估失败: {e}")

# ==================== 总结 ====================

print("\n\n" + "=" * 60)
print("测试完成")
print("=" * 60)
print("\n说明:")
print("1. 评估结果格式: 字典 {'key': 'correctness', 'score': 0.8, 'comment': None, 'metadata': None}")
print("2. 分数范围: 0.0 到 1.0")
print("3. 分数含义: 分数越高表示输出越正确")
print("   - 0.8-1.0: 高质量回答")
print("   - 0.5-0.8: 部分正确回答")
print("   - 0.0-0.5: 低质量或错误回答")
print("4. 如果 use_reasoning=True，结果可能是 (分数, 理由) 元组")
print("5. 如果 use_reasoning=False，结果通常是包含分数的字典")
print("\n实际运行结果示例:")
print("  - 高质量回答: score = 0.800")
print("  - 部分正确回答: score = 0.600")
print("  - 错误回答: score = 0.000")

