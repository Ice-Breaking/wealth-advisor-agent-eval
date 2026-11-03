#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LangSmith 测试与评估 - 快速开始示例

这个脚本演示如何快速使用 LangSmith 进行测试和评估。
"""

import os
# 导入评估模块：用 importlib 按文件名加载同目录模块
import importlib.util
import sys
from datetime import datetime

spec = importlib.util.spec_from_file_location(
    "langsmith_evaluation",
    "langsmith_evaluation.py"
)
eval_module = importlib.util.module_from_spec(spec)
sys.modules["langsmith_evaluation"] = eval_module
spec.loader.exec_module(eval_module)

from langsmith_evaluation import (
    create_test_dataset,
    run_evaluation,
    run_single_test
)

def main():
    """主函数：演示完整的测试和评估流程"""
    
    print("=" * 60)
    print("LangSmith 测试与评估 - 快速开始示例")
    print("=" * 60)
    print()
    
    # 检查 LangSmith 是否启用
    if not os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_TRACING_V2", "").lower() != "true":
        print("❌ 错误: LangSmith 未启用")
        print()
        print("请先设置环境变量:")
        print("  $env:LANGSMITH_API_KEY='your-api-key'")
        print("  $env:LANGCHAIN_TRACING_V2='true'")
        print()
        return
    
    print("✓ LangSmith 已启用")
    print()
    
    # 步骤 1: 创建测试数据集
    print("步骤 1: 创建测试数据集...")
    print("-" * 60)
    dataset_name = "wealth-advisor-test-dataset"
    
    try:
        created_dataset = create_test_dataset(dataset_name)
        if created_dataset:
            print(f"✓ 数据集创建成功: {dataset_name}")
        else:
            print("⚠ 数据集可能已存在，继续使用现有数据集")
    except Exception as e:
        print(f"⚠ 数据集创建警告: {str(e)}")
        print("   继续使用现有数据集（如果存在）")
    
    print()
    
    # 步骤 2: 运行单个测试（演示）
    print("步骤 2: 运行单个测试用例（演示）...")
    print("-" * 60)
    
    test_query = "今天上证指数的表现如何？"
    print(f"测试查询: {test_query}")
    
    try:
        result = run_single_test(test_query, "customer1")
        print("✓ 单个测试完成")
    except Exception as e:
        print(f"⚠ 单个测试失败: {str(e)}")
    
    print()
    
    # 步骤 3: 询问是否运行完整评估
    print("步骤 3: 运行完整评估")
    print("-" * 60)
    print("注意: 完整评估会运行所有测试用例，可能需要较长时间")
    
    user_input = input("是否运行完整评估？(y/n): ").strip().lower()
    
    if user_input == 'y':
        experiment_name = f"wealth-advisor-eval-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        print(f"\n开始运行评估...")
        print(f"实验名称: {experiment_name}")
        print("这可能需要几分钟时间，请耐心等待...")
        print()
        
        try:
            results = run_evaluation(dataset_name, experiment_name)
            if results:
                print()
                print("=" * 60)
                print("✓ 评估完成！")
                print("=" * 60)
                print()
                print("查看结果:")
                print(f"  https://smith.langchain.com")
                print()
                print("在 LangSmith 界面中:")
                print("  1. 进入 'Experiments' 页面")
                print(f"  2. 查找实验: {experiment_name}")
                print("  3. 查看详细的评估结果和分数")
            else:
                print("⚠ 评估完成，但未返回结果")
        except Exception as e:
            print(f"❌ 评估失败: {str(e)}")
    else:
        print("跳过完整评估")
    
    print()
    print("=" * 60)
    print("完成！")
    print("=" * 60)
    print()
    print("提示:")
    print("  - 查看详细使用说明: 阅读 LANGSMITH_TESTING_GUIDE.md")
    print("  - 运行完整工具: python langsmith_evaluation.py")
    print("  - 访问 LangSmith: https://smith.langchain.com")

if __name__ == "__main__":
    main()

