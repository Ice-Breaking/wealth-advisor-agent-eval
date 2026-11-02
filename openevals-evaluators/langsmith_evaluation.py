#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LangSmith 测试与评估模块

本模块提供使用 LangSmith 进行 Agent 测试和评估的功能，包括：
1. 创建和管理测试数据集
2. 运行批量测试
3. 使用评估器评估输出质量
4. 比较不同版本的性能
5. 生成评估报告

使用方法：
1. 确保已设置 LangSmith 环境变量
2. 运行此脚本进行测试和评估
"""

import os
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
from langsmith import Client, RunEvaluator
from langsmith.evaluation import evaluate
from langsmith.schemas import Example, Run

# 导入主智能体模块
# 导入主智能体模块：用 importlib 按文件名加载同目录模块
import importlib.util
import sys

spec = importlib.util.spec_from_file_location(
    "hybrid_advisor_langgraph",
    "hybrid_advisor_langgraph.py"
)
hybrid_module = importlib.util.module_from_spec(spec)
sys.modules["hybrid_advisor_langgraph"] = hybrid_module
spec.loader.exec_module(hybrid_module)

from hybrid_advisor_langgraph import (
    run_wealth_advisor,
    SAMPLE_CUSTOMER_PROFILES
)

# LangSmith 客户端初始化
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
LANGSMITH_ENABLED = os.getenv("LANGCHAIN_TRACING_V2", "").lower() == "true"

if not LANGSMITH_ENABLED or not LANGSMITH_API_KEY:
    print("警告: LangSmith 未启用，无法进行测试和评估")
    print("请设置环境变量:")
    print("  - LANGSMITH_API_KEY: 您的 API 密钥")
    print("  - LANGCHAIN_TRACING_V2: true")
    client = None
else:
    client = Client(api_key=LANGSMITH_API_KEY)
    print(f"[LangSmith] 客户端已初始化，可以开始测试和评估")

# ==================== 测试数据集定义 ====================

# 测试用例：反应式查询（简单查询）
REACTIVE_TEST_CASES = [
    {
        "inputs": {
            "user_query": "今天上证指数的表现如何？",
            "customer_id": "customer1"
        },
        "expected_outputs": {
            "processing_mode": "reactive",
            "should_contain": ["上证指数", "点位", "涨跌"]
        }
    },
    {
        "inputs": {
            "user_query": "我的投资组合中科技股占比是多少？",
            "customer_id": "customer1"
        },
        "expected_outputs": {
            "processing_mode": "reactive",
            "should_contain": ["科技", "占比", "投资组合"]
        }
    },
    {
        "inputs": {
            "user_query": "请解释一下什么是ETF？",
            "customer_id": "customer1"
        },
        "expected_outputs": {
            "processing_mode": "reactive",
            "should_contain": ["ETF", "基金", "交易"]
        }
    }
]

# 测试用例：深思熟虑查询（复杂分析）
DELIBERATIVE_TEST_CASES = [
    {
        "inputs": {
            "user_query": "根据当前市场情况，我应该如何调整投资组合以应对可能的经济衰退？",
            "customer_id": "customer1"
        },
        "expected_outputs": {
            "processing_mode": "deliberative",
            "should_contain": ["投资组合", "调整", "经济衰退", "建议"]
        }
    },
    {
        "inputs": {
            "user_query": "考虑到我的退休目标，请评估我当前的投资策略并提供优化建议。",
            "customer_id": "customer1"
        },
        "expected_outputs": {
            "processing_mode": "deliberative",
            "should_contain": ["退休", "投资策略", "评估", "建议"]
        }
    },
    {
        "inputs": {
            "user_query": "我想为子女准备教育金，请帮我设计一个10年期的投资计划。",
            "customer_id": "customer1"
        },
        "expected_outputs": {
            "processing_mode": "deliberative",
            "should_contain": ["教育金", "10年", "投资计划", "建议"]
        }
    }
]

# 测试用例：边界情况
EDGE_CASE_TEST_CASES = [
    {
        "inputs": {
            "user_query": "",  # 空查询
            "customer_id": "customer1"
        },
        "expected_outputs": {
            "should_handle_error": True
        }
    },
    {
        "inputs": {
            "user_query": "这是一个非常长的查询" * 100,  # 超长查询
            "customer_id": "customer1"
        },
        "expected_outputs": {
            "should_handle": True
        }
    }
]

# 合并所有测试用例
ALL_TEST_CASES = REACTIVE_TEST_CASES + DELIBERATIVE_TEST_CASES + EDGE_CASE_TEST_CASES

# ==================== 评估器定义 ====================

def _get_example_inputs(example):
    """安全获取 example 的 inputs，支持 Example 对象和字典"""
    try:
        # 尝试多种方式获取 inputs
        if hasattr(example, 'inputs'):
            inputs = example.inputs
            if inputs is not None:
                return inputs if isinstance(inputs, dict) else {}
        if isinstance(example, dict):
            if "inputs" in example:
                return example["inputs"] if isinstance(example["inputs"], dict) else {}
            # 如果 example 本身就是 inputs
            if "user_query" in example:
                return example
        # 尝试直接访问属性
        if hasattr(example, '__dict__'):
            if 'inputs' in example.__dict__:
                inputs = example.__dict__['inputs']
                return inputs if isinstance(inputs, dict) else {}
    except Exception as e:
        pass
    return {}

def _get_example_outputs(example):
    """安全获取 example 的 outputs，支持 Example 对象和字典"""
    try:
        # 尝试多种方式获取 outputs
        if hasattr(example, 'outputs'):
            outputs = example.outputs
            if outputs is not None:
                return outputs if isinstance(outputs, dict) else {}
        if isinstance(example, dict):
            if "outputs" in example:
                return example["outputs"] if isinstance(example["outputs"], dict) else {}
        # 尝试直接访问属性
        if hasattr(example, '__dict__'):
            if 'outputs' in example.__dict__:
                outputs = example.__dict__['outputs']
                return outputs if isinstance(outputs, dict) else {}
    except Exception as e:
        pass
    return {}

class ProcessingModeEvaluator(RunEvaluator):
    """评估处理模式选择是否正确"""
    
    def evaluate_run(self, run: Run, example: Example, **kwargs) -> Dict[str, Any]:
        """评估处理模式是否正确"""
        try:
            example_outputs = _get_example_outputs(example)
            expected_mode = example_outputs.get("processing_mode")
            
            if not expected_mode:
                return {
                    "key": "processing_mode",
                    "score": None,
                    "comment": "未指定期望的处理模式"
                }
            
            actual_mode = run.outputs.get("processing_mode") if run.outputs else None
            
            if actual_mode == expected_mode:
                return {
                    "key": "processing_mode",
                    "score": 1.0,
                    "comment": f"处理模式正确: {actual_mode}"
                }
            else:
                return {
                    "key": "processing_mode",
                    "score": 0.0,
                    "comment": f"处理模式不匹配: 期望 {expected_mode}, 实际 {actual_mode}"
                }
        except Exception as e:
            return {
                "key": "processing_mode",
                "score": 0,
                "comment": f"评估错误: {str(e)}"
            }

class ResponseCompletenessEvaluator(RunEvaluator):
    """评估响应完整性"""
    
    def evaluate_run(self, run: Run, example: Example, **kwargs) -> Dict[str, Any]:
        """评估响应是否完整"""
        try:
            if not run.outputs or "final_response" not in run.outputs:
                return {
                    "key": "response_completeness",
                    "score": 0,
                    "comment": "无响应输出"
                }
            
            response = run.outputs.get("final_response", "")
            example_outputs = _get_example_outputs(example)
            expected_keywords = example_outputs.get("should_contain", [])
            
            if not expected_keywords:
                return {
                    "key": "response_completeness",
                    "score": None,
                    "comment": "未指定期望的关键词"
                }
            
            # 检查是否包含期望的关键词
            response_lower = response.lower()
            found_keywords = [kw for kw in expected_keywords if kw.lower() in response_lower]
            completeness = len(found_keywords) / len(expected_keywords) if expected_keywords else 0
            
            return {
                "key": "response_completeness",
                "score": completeness,
                "comment": f"找到 {len(found_keywords)}/{len(expected_keywords)} 个期望关键词"
            }
        except Exception as e:
            return {
                "key": "response_completeness",
                "score": 0,
                "comment": f"评估错误: {str(e)}"
            }

# ==================== 测试运行函数 ====================

def create_test_dataset(dataset_name: str = "wealth-advisor-test-dataset") -> str:
    """创建测试数据集"""
    if not client:
        print("错误: LangSmith 客户端未初始化")
        return None
    
    try:
        # 检查数据集是否已存在
        try:
            existing_dataset = client.read_dataset(dataset_name=dataset_name)
            print(f"[LangSmith] 数据集已存在: {dataset_name}")
            # 检查现有示例数量
            existing_examples = list(client.list_examples(dataset_name=dataset_name))
            print(f"  现有测试用例数量: {len(existing_examples)}")
            if len(existing_examples) > 0:
                return dataset_name
            else:
                print(f"  数据集为空，将添加测试用例...")
        except Exception as e:
            # 数据集不存在，创建新数据集
            print(f"[LangSmith] 数据集不存在，正在创建...")
            try:
                dataset = client.create_dataset(
                    dataset_name=dataset_name,
                    description="投顾AI助手测试数据集，包含反应式、深思熟虑和边界情况测试用例"
                )
                print(f"[LangSmith] 数据集已创建: {dataset_name}")
            except Exception as create_error:
                error_msg = str(create_error)
                if "already exists" in error_msg.lower() or "duplicate" in error_msg.lower():
                    print(f"[LangSmith] 数据集已存在: {dataset_name}")
                else:
                    raise create_error
        
        # 添加测试用例到数据集
        print(f"[LangSmith] 正在添加测试用例...")
        added_count = 0
        for i, test_case in enumerate(ALL_TEST_CASES):
            try:
                client.create_example(
                    inputs=test_case["inputs"],
                    outputs=test_case.get("expected_outputs", {}),
                    dataset_name=dataset_name
                )
                added_count += 1
            except Exception as e:
                # 如果示例已存在，跳过
                error_msg = str(e)
                if "already exists" in error_msg.lower() or "duplicate" in error_msg.lower():
                    continue
                else:
                    print(f"  警告: 添加测试用例 {i+1} 失败: {error_msg}")
        
        # 验证数据集
        existing_examples = list(client.list_examples(dataset_name=dataset_name))
        print(f"[LangSmith] 测试数据集准备完成: {dataset_name}")
        print(f"  总测试用例数量: {len(existing_examples)}")
        if added_count > 0:
            print(f"  本次新增: {added_count} 个测试用例")
        
        return dataset_name
    except Exception as e:
        error_msg = str(e)
        print(f"创建测试数据集失败: {error_msg}")
        import traceback
        traceback.print_exc()
        return None

def run_evaluation(
    dataset_name: str,
    experiment_name: Optional[str] = None,
    evaluators: Optional[List[RunEvaluator]] = None
) -> Dict[str, Any]:
    """运行评估"""
    if not client:
        print("错误: LangSmith 客户端未初始化")
        return None
    
    if not experiment_name:
        experiment_name = f"wealth-advisor-eval-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    
    # 默认评估器
    if not evaluators:
        evaluators = [
            ProcessingModeEvaluator(),
            ResponseCompletenessEvaluator()
        ]
    
    # 定义测试函数
    def test_function(example: Example) -> Dict[str, Any]:
        """测试函数，运行智能体并返回结果"""
        try:
            # 尝试多种方式获取 inputs
            example_inputs = None
            
            # 方式1: 直接访问 example.inputs
            if hasattr(example, 'inputs'):
                example_inputs = example.inputs
            
            # 方式2: 如果是字典
            if not example_inputs and isinstance(example, dict):
                example_inputs = example.get("inputs")
            
            # 方式3: 使用辅助函数
            if not example_inputs:
                example_inputs = _get_example_inputs(example)
            
            # 确保 example_inputs 是字典
            if not isinstance(example_inputs, dict):
                # 如果 example_inputs 不是字典，尝试转换或使用空字典
                if example_inputs is None:
                    example_inputs = {}
                elif hasattr(example_inputs, '__dict__'):
                    example_inputs = example_inputs.__dict__
                else:
                    example_inputs = {}
            
            user_query = example_inputs.get("user_query") if example_inputs else ""
            customer_id = example_inputs.get("customer_id", "customer1") if example_inputs else "customer1"
            
            # 如果 user_query 为空或 None，返回错误
            if not user_query or (isinstance(user_query, str) and not user_query.strip()):
                return {
                    "error": "用户查询为空",
                    "final_response": "用户查询为空，无法处理",
                    "processing_mode": "unknown",
                    "query_type": "unknown"
                }
            
            result = run_wealth_advisor(
                user_query=user_query,
                customer_id=customer_id
            )
            return {
                "final_response": result.get("final_response", ""),
                "processing_mode": result.get("processing_mode", "unknown"),
                "query_type": result.get("query_type", "unknown"),
                "error": result.get("error")
            }
        except Exception as e:
            return {
                "error": str(e),
                "final_response": f"执行错误: {str(e)}",
                "processing_mode": "unknown",
                "query_type": "unknown"
            }
    
    try:
        print(f"[LangSmith] 开始评估...")
        print(f"  实验名称: {experiment_name}")
        print(f"  数据集: {dataset_name}")
        print(f"  评估器数量: {len(evaluators)}")
        print()
        
        # 运行评估
        results = evaluate(
            test_function,
            data=dataset_name,
            evaluators=evaluators,
            experiment_prefix=experiment_name,
            max_concurrency=1  # 串行执行，避免API限流
        )
        
        print()
        print(f"[LangSmith] 评估完成")
        return results
    except Exception as e:
        print(f"运行评估失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def run_single_test(user_query: str, customer_id: str = "customer1") -> Dict[str, Any]:
    """运行单个测试用例"""
    print(f"\n=== 运行测试 ===")
    print(f"查询: {user_query}")
    print(f"客户: {customer_id}")
    
    start_time = datetime.now()
    result = run_wealth_advisor(user_query, customer_id)
    end_time = datetime.now()
    
    process_time = (end_time - start_time).total_seconds()
    
    print(f"\n处理模式: {result.get('processing_mode', '未知')}")
    print(f"查询类型: {result.get('query_type', '未知')}")
    print(f"处理时间: {process_time:.2f}秒")
    
    if result.get("error"):
        print(f"错误: {result['error']}")
    
    print(f"\n响应预览: {result.get('final_response', '')[:200]}...")
    
    return {
        "result": result,
        "process_time": process_time,
        "timestamp": datetime.now().isoformat()
    }

def compare_experiments(experiment_names: List[str]) -> Dict[str, Any]:
    """比较不同实验的结果"""
    if not client:
        print("错误: LangSmith 客户端未初始化")
        return None
    
    try:
        comparison_results = {}
        for exp_name in experiment_names:
            # 获取实验的评估结果
            runs = list(client.list_runs(project_name=exp_name, limit=100))
            
            # 计算平均分数
            scores = {}
            for run in runs:
                if run.feedback_stats:
                    for feedback in run.feedback_stats:
                        key = feedback.get("key", "unknown")
                        score = feedback.get("score", 0)
                        if key not in scores:
                            scores[key] = []
                        scores[key].append(score)
            
            # 计算平均值
            avg_scores = {
                key: sum(values) / len(values) if values else 0
                for key, values in scores.items()
            }
            
            comparison_results[exp_name] = {
                "total_runs": len(runs),
                "average_scores": avg_scores
            }
        
        return comparison_results
    except Exception as e:
        print(f"比较实验失败: {str(e)}")
        return None

# ==================== 主函数 ====================

if __name__ == "__main__":
    print("=" * 60)
    print("LangSmith 测试与评估工具")
    print("=" * 60)
    print()
    
    if not LANGSMITH_ENABLED or not client:
        print("❌ 错误: LangSmith 未启用，无法进行测试和评估")
        print()
        print("请设置环境变量:")
        print("  - LANGSMITH_API_KEY: 您的 API 密钥")
        print("  - LANGCHAIN_TRACING_V2: true")
        exit(1)
    
    # 固定的数据集名称
    dataset_name = "wealth-advisor-test-dataset"
    
    # 步骤 1: 创建或使用测试数据集
    print("步骤 1: 准备测试数据集...")
    print("-" * 60)
    created_dataset = None
    try:
        created_dataset = create_test_dataset(dataset_name)
        if created_dataset:
            print(f"✓ 测试数据集准备完成: {dataset_name}")
        else:
            print(f"❌ 数据集准备失败")
            print()
            print("请检查:")
            print("  1. LangSmith API 密钥是否正确")
            print("  2. 网络连接是否正常")
            print("  3. 或者手动在 https://smith.langchain.com 创建数据集")
            exit(1)
    except Exception as e:
        print(f"❌ 数据集准备失败: {str(e)}")
        print()
        print("请检查:")
        print("  1. LangSmith API 密钥是否正确")
        print("  2. 网络连接是否正常")
        print("  3. 或者手动在 https://smith.langchain.com 创建数据集")
        exit(1)
    
    # 验证数据集是否存在
    try:
        existing_examples = list(client.list_examples(dataset_name=dataset_name))
        if len(existing_examples) == 0:
            print(f"❌ 数据集为空，无法运行评估")
            print("请确保数据集包含测试用例")
            exit(1)
    except Exception as e:
        print(f"❌ 无法验证数据集: {str(e)}")
        print("请确保数据集存在且可访问")
        exit(1)
    
    print()
    
    # 步骤 2: 运行评估
    print("步骤 2: 运行测试集评估...")
    print("-" * 60)
    experiment_name = f"wealth-advisor-eval-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    print(f"实验名称: {experiment_name}")
    print(f"数据集: {dataset_name}")
    
    # 获取实际测试用例数量
    try:
        existing_examples = list(client.list_examples(dataset_name=dataset_name))
        print(f"测试用例数量: {len(existing_examples)}")
    except:
        print(f"测试用例数量: {len(ALL_TEST_CASES)} (预期)")
    
    print()
    print("开始运行评估，这可能需要几分钟时间，请耐心等待...")
    print()
    
    try:
        results = run_evaluation(dataset_name, experiment_name)
        
        if results:
            print()
            print("=" * 60)
            print("✓ 评估完成！")
            print("=" * 60)
            print()
            print(f"实验名称: {experiment_name}")
            print(f"数据集: {dataset_name}")
            print()
            print("查看详细结果:")
            print(f"  https://smith.langchain.com")
            print()
            print("在 LangSmith 界面中:")
            print("  1. 进入 'Experiments' 页面")
            print(f"  2. 查找实验: {experiment_name}")
            print("  3. 查看详细的评估结果、分数和统计信息")
        else:
            print()
            print("⚠ 评估完成，但未返回结果")
            print("请检查 LangSmith 界面查看评估状态")
    except Exception as e:
        print()
        print("=" * 60)
        print(f"❌ 评估失败: {str(e)}")
        print("=" * 60)
        print()
        print("请检查:")
        print("  1. LangSmith API 密钥是否正确")
        print("  2. 数据集是否存在")
        print("  3. 网络连接是否正常")
        exit(1)

