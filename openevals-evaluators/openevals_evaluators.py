#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
OpenEvals 评估器使用 - 投顾助手评估示例
使用 openevals 创建多种评估器，对投顾助手进行全面评估
"""

import os
from langsmith import Client
from langsmith.evaluation import evaluate
from langchain_community.chat_models import ChatTongyi
from openevals.llm import create_llm_as_judge
from openevals.prompts import (
    ANSWER_RELEVANCE_PROMPT,
    CONCISENESS_PROMPT,
    RAG_HELPFULNESS_PROMPT,
    HALLUCINATION_PROMPT,
    TOXICITY_PROMPT,
)

# ==================== 投顾助手评估（使用 OpenEvals）====================

def evaluate_wealth_advisor():
    """使用 OpenEvals 评估投顾助手"""
    print("=" * 60)
    print("投顾助手评估（使用 OpenEvals）")
    print("=" * 60)
    
    # 检查环境变量
    if not os.getenv("DASHSCOPE_API_KEY"):
        print("\n[ERROR] 错误: 请设置 DASHSCOPE_API_KEY 环境变量")
        return
    
    if not os.getenv("LANGSMITH_API_KEY"):
        print("\n[WARN] 未配置 LANGSMITH_API_KEY，跳过 LangSmith 集成")
        print("如需使用 LangSmith，请设置环境变量:")
        print("  Windows PowerShell: $env:LANGSMITH_API_KEY='your-key'")
        print("  Linux/Mac: export LANGSMITH_API_KEY='your-key'")
        return
    
    # 尝试导入投顾助手模块
    try:
        import importlib.util
        import sys
        
        spec = importlib.util.spec_from_file_location(
            "hybrid_advisor_langgraph",
            "hybrid_advisor_langgraph.py"
        )
        if spec is None or spec.loader is None:
            raise ImportError("无法加载投顾助手模块")
        
        hybrid_module = importlib.util.module_from_spec(spec)
        sys.modules["hybrid_advisor_langgraph"] = hybrid_module
        spec.loader.exec_module(hybrid_module)
        
        from hybrid_advisor_langgraph import run_wealth_advisor
    except Exception as e:
        print(f"\n[WARN] 无法导入投顾助手模块: {e}")
        print("  请确保 hybrid_advisor_langgraph.py 文件存在")
        return
    
    # 1. 定义要评估的函数
    def predict(inputs: dict) -> dict:
        """投顾助手预测函数，用于 LangSmith 评估"""
        try:
            user_query = inputs.get("user_query", "")
            customer_id = inputs.get("customer_id", "customer1")
            
            if not user_query:
                return {
                    "output": "用户查询为空，无法处理",
                    "final_response": "用户查询为空，无法处理"
                }
            
            result = run_wealth_advisor(user_query=user_query, customer_id=customer_id)
            
            # 返回格式化的输出
            # 为了支持评估器，将 processing_mode 信息包含在输出中
            final_response = result.get("final_response", "")
            processing_mode = result.get("processing_mode", "unknown")
            
            # 将 processing_mode 信息添加到输出文本中（用于评估器提取）
            output_text = f"[处理模式: {processing_mode}]\n\n{final_response}"
            
            return {
                "output": output_text,
                "final_response": final_response,
                "processing_mode": processing_mode,
                "query_type": result.get("query_type", "unknown")
            }
        except Exception as e:
            return {
                "output": f"执行错误: {str(e)}",
                "final_response": f"执行错误: {str(e)}",
                "error": str(e)
            }
    
    # 2. 创建评估 LLM
    eval_llm = ChatTongyi(
        model_name="qwen-flash",
        dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"),
        temperature=0
    )
    
    # 3. 创建投顾助手专用的自定义评估器（参考 langsmith_evaluation.py）
    
    # 处理模式评估 Prompt（对应 ProcessingModeEvaluator）
    PROCESSING_MODE_PROMPT = """你是一位评估员，评估投顾助手选择的处理模式是否正确。

<说明>
投顾助手有两种处理模式：
- reactive（反应式）：用于简单查询，需要快速响应
- deliberative（深思熟虑）：用于复杂分析，需要深入思考

<用户查询>
{inputs}
</用户查询>

<实际输出>
输出的开头包含 "[处理模式: xxx]" 格式的信息，请从中提取实际选择的处理模式（应该是 "reactive" 或 "deliberative"）。
{outputs}
</实际输出>

<期望的处理模式>
参考输出中可能包含字典格式，如 {{"processing_mode": "reactive", "should_contain": [...]}}，请提取其中的 "processing_mode" 字段值。
如果参考输出是字符串格式，请直接使用该字符串。
{reference_outputs}
</期望的处理模式>

请评估实际选择的处理模式是否与期望的处理模式一致。
如果一致，给出 1.0 分；如果不一致，给出 0.0 分。"""

    # 响应完整性评估 Prompt（对应 ResponseCompletenessEvaluator）
    RESPONSE_COMPLETENESS_PROMPT = """你是一位评估员，评估投顾助手的回答是否完整，是否包含了期望的关键信息。

<用户查询>
{inputs}
</用户查询>

<投顾助手的回答>
注意：回答开头可能包含 "[处理模式: xxx]" 格式的信息，请忽略这部分，只关注实际的回答内容。
{outputs}
</投顾助手的回答>

<期望包含的关键词>
参考输出中可能包含字典格式，如 {{"processing_mode": "reactive", "should_contain": ["关键词1", "关键词2"]}}，请提取其中的 "should_contain" 字段值（这是一个关键词列表）。
如果参考输出是字符串格式，请尝试解析为关键词列表。
{reference_outputs}
</期望包含的关键词>

请评估回答是否包含了期望的关键信息。
根据包含的关键词比例给出 0-1 之间的分数：
- 如果包含了所有期望的关键词，给出 1.0 分
- 如果包含了部分关键词，给出相应的比例分数（例如包含 2/3 的关键词，给出 0.67 分）
- 如果没有包含任何关键词，给出 0.0 分"""

    # 创建评估器列表（包含预定义和自定义评估器）
    evaluators = [
        # 预定义评估器
        create_llm_as_judge(
            prompt=ANSWER_RELEVANCE_PROMPT,
            feedback_key="relevance",
            judge=eval_llm,
            continuous=True,
            use_reasoning=False,
        ),
        create_llm_as_judge(
            prompt=CONCISENESS_PROMPT,
            feedback_key="conciseness",
            judge=eval_llm,
            continuous=True,
            use_reasoning=False,
        ),
        create_llm_as_judge(
            prompt=RAG_HELPFULNESS_PROMPT,
            feedback_key="helpfulness",
            judge=eval_llm,
            continuous=True,
            use_reasoning=False,
        ),
        create_llm_as_judge(
            prompt=HALLUCINATION_PROMPT,
            feedback_key="hallucination",
            judge=eval_llm,
            continuous=True,
            use_reasoning=False,
        ),
        create_llm_as_judge(
            prompt=TOXICITY_PROMPT,
            feedback_key="toxicity",
            judge=eval_llm,
            continuous=True,
            use_reasoning=False,
        ),
        # 自定义评估器（参考 langsmith_evaluation.py）
        create_llm_as_judge(
            prompt=PROCESSING_MODE_PROMPT,
            feedback_key="processing_mode",
            judge=eval_llm,
            continuous=True,
            use_reasoning=False,
        ),
        create_llm_as_judge(
            prompt=RESPONSE_COMPLETENESS_PROMPT,
            feedback_key="response_completeness",
            judge=eval_llm,
            continuous=True,
            use_reasoning=False,
        ),
    ]
    
    print(f"\n创建了 {len(evaluators)} 个评估器:")
    print("  预定义评估器:")
    print("    - relevance (相关性)")
    print("    - conciseness (简洁性)")
    print("    - helpfulness (帮助性)")
    print("    - hallucination (幻觉检测)")
    print("    - toxicity (毒性检测)")
    print("  自定义评估器（参考 langsmith_evaluation.py）:")
    print("    - processing_mode (处理模式)")
    print("    - response_completeness (响应完整性)")
    
    # 4. LangSmith 集成
    print("\n[步骤1] 创建 LangSmith 客户端")
    client = Client(api_key=os.getenv("LANGSMITH_API_KEY"))
    dataset_name = "wealth-advisor-openevals-dataset"
    
    # 定义测试用例（参考 langsmith_evaluation.py）
    test_cases = [
        {
            "inputs": {
                "user_query": "今天上证指数的表现如何？",
                "customer_id": "customer1"
            },
            "outputs": {
                "processing_mode": "reactive",
                "should_contain": ["上证指数", "点位", "涨跌"]
            }
        },
        {
            "inputs": {
                "user_query": "我的投资组合中科技股占比是多少？",
                "customer_id": "customer1"
            },
            "outputs": {
                "processing_mode": "reactive",
                "should_contain": ["科技", "占比", "投资组合"]
            }
        },
        {
            "inputs": {
                "user_query": "请解释一下什么是ETF？",
                "customer_id": "customer1"
            },
            "outputs": {
                "processing_mode": "reactive",
                "should_contain": ["ETF", "基金", "交易"]
            }
        },
        {
            "inputs": {
                "user_query": "根据当前市场情况，我应该如何调整投资组合以应对可能的经济衰退？",
                "customer_id": "customer1"
            },
            "outputs": {
                "processing_mode": "deliberative",
                "should_contain": ["投资组合", "调整", "经济衰退", "建议"]
            }
        },
        {
            "inputs": {
                "user_query": "考虑到我的退休目标，请评估我当前的投资策略并提供优化建议。",
                "customer_id": "customer1"
            },
            "outputs": {
                "processing_mode": "deliberative",
                "should_contain": ["退休", "投资策略", "评估", "建议"]
            }
        },
        {
            "inputs": {
                "user_query": "我想为子女准备教育金，请帮我设计一个10年期的投资计划。",
                "customer_id": "customer1"
            },
            "outputs": {
                "processing_mode": "deliberative",
                "should_contain": ["教育金", "10年", "投资计划", "建议"]
            }
        }
    ]
    
    print(f"[步骤2] 创建数据集: {dataset_name}")
    try:
        # 删除已存在的数据集（如果存在）
        try:
            client.delete_dataset(dataset_name=dataset_name)
        except:
            pass
        
        # 创建新数据集
        client.create_dataset(
            dataset_name=dataset_name,
            description="投顾助手 OpenEvals 评估数据集"
        )
        
        # 添加测试用例
        for test_case in test_cases:
            client.create_example(
                inputs=test_case["inputs"],
                outputs=test_case.get("outputs", {}),
                dataset_name=dataset_name
            )
        
        print(f"[OK] 数据集创建成功，包含 {len(test_cases)} 个测试用例")
    except Exception as e:
        print(f"[ERROR] 数据集创建失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print(f"[步骤3] 使用 evaluate() 函数运行评估")
    print(f"  评估器数量: {len(evaluators)}")
    print(f"  数据集: {dataset_name}")
    print(f"  测试用例数: {len(test_cases)}")
    print(f"  预计评估次数: {len(evaluators) * len(test_cases)}")
    
    try:
        results = evaluate(
            predict,
            data=dataset_name,
            evaluators=evaluators,
            client=client,
            experiment_prefix="wealth-advisor-openevals",
            max_concurrency=1  # 串行执行，避免 API 限流
        )
        
        print(f"[步骤4] 评估完成，查看结果:")
        print(f"  https://smith.langchain.com")
        print(f"\n[OK] 投顾助手评估成功")
        
        # 尝试获取结果（可能因为网络问题失败）
        try:
            results_list = list(results)
            print(f"[OK] 成功评估 {len(results_list)} 个测试样本")
            
            # 统计评估分数
            if results_list:
                print("\n评估结果摘要:")
                scores_summary = {}
                for result in results_list:
                    if "feedback_stats" in result:
                        for feedback in result["feedback_stats"]:
                            key = feedback.get("key", "unknown")
                            score = feedback.get("score", 0)
                            if key not in scores_summary:
                                scores_summary[key] = []
                            scores_summary[key].append(score)
                
                # 按类别分组显示
                predefined_keys = ["relevance", "conciseness", "helpfulness", "hallucination", "toxicity"]
                custom_keys = ["processing_mode", "response_completeness"]
                
                print("\n  预定义评估器结果:")
                for key in predefined_keys:
                    if key in scores_summary:
                        scores = scores_summary[key]
                        avg_score = sum(scores) / len(scores) if scores else 0
                        print(f"    {key:20s}: 平均分数 {avg_score:.3f} ({len(scores)} 个样本)")
                
                print("\n  自定义评估器结果:")
                for key in custom_keys:
                    if key in scores_summary:
                        scores = scores_summary[key]
                        avg_score = sum(scores) / len(scores) if scores else 0
                        print(f"    {key:20s}: 平均分数 {avg_score:.3f} ({len(scores)} 个样本)")
        except Exception as e:
            print(f"[WARN] 无法获取详细结果: {e}")
            print("  但评估可能已经完成，请访问 LangSmith 网站查看")
            
    except Exception as e:
        print(f"[ERROR] 评估失败: {e}")
        import traceback
        traceback.print_exc()


# ==================== 主函数 ====================

def main():
    """运行投顾助手评估"""
    print("=" * 60)
    print("OpenEvals 评估器使用 - 投顾助手评估")
    print("=" * 60)
    
    try:
        evaluate_wealth_advisor()
    except Exception as e:
        print(f"\n[ERROR] 运行评估时出错: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("评估完成")
    print("=" * 60)
    print("\n注意:")
    print("  - 需要配置 DASHSCOPE_API_KEY 和 LANGSMITH_API_KEY 环境变量")
    print("  - 需要 hybrid_advisor_langgraph.py 文件存在")
    print("  - 评估结果可以在 LangSmith 网站上查看: https://smith.langchain.com")
    print("\n评估器说明:")
    print("  预定义评估器:")
    print("    - relevance: 回答与问题的相关性")
    print("    - conciseness: 回答的简洁性")
    print("    - helpfulness: 回答的帮助性")
    print("    - hallucination: 幻觉检测（分数越高表示幻觉越少）")
    print("    - toxicity: 毒性检测（分数越高表示越有害）")
    print("  自定义评估器:")
    print("    - processing_mode: 处理模式选择是否正确")
    print("    - response_completeness: 响应是否包含期望的关键信息")


if __name__ == "__main__":
    main()
