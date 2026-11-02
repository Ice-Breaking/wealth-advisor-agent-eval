#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LangChain Agent 自动化测试

被测对象：混合智能体 - 财富管理投顾 AI 助手（hybrid_advisor_langgraph.py）
本脚本基于 langsmith_evaluation.py 改写，实现三个步骤：

    Step1 定义测试数据：针对不同意图（反应式 / 深思熟虑）+ 针对边界情况
    Step2 创建评估器：ProcessingModeEvaluator + ResponseCompletenessEvaluator
    Step3 自动化测试：上传数据集到 LangSmith -> evaluate() 串行执行 -> 打印分数汇总

运行方式（使用装齐依赖的解释器）：
    python3.11 agent_automation_testing.py

可选环境变量：
    AGENT_MODEL   切换被测模型（默认沿用 Agent 内的 qwen3.8-2.4t-a95b）
"""

import os
import sys
import time
import importlib.util
from datetime import datetime
from typing import Any, Dict, List, Optional

from langsmith import Client, RunEvaluator
from langsmith.evaluation import evaluate
from langsmith.schemas import Example, Run

# ==================== 导入被测 Agent ====================
# 同目录模块，使用 importlib 按绝对路径加载，
# 这样无论从哪个目录启动脚本都能找到被测模块
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AGENT_FILE = os.path.join(BASE_DIR, "hybrid_advisor_langgraph.py")
AGENT_MODULE_NAME = "hybrid_advisor_langgraph"

_spec = importlib.util.spec_from_file_location(AGENT_MODULE_NAME, AGENT_FILE)
_agent_module = importlib.util.module_from_spec(_spec)
sys.modules[AGENT_MODULE_NAME] = _agent_module
_spec.loader.exec_module(_agent_module)

run_wealth_advisor = _agent_module.run_wealth_advisor
AGENT_MODEL = getattr(_agent_module, "AGENT_MODEL", "unknown")

# ==================== LangSmith 客户端 ====================
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
LANGSMITH_ENABLED = os.getenv("LANGCHAIN_TRACING_V2", "").lower() == "true"

if not LANGSMITH_ENABLED or not LANGSMITH_API_KEY:
    print("错误: LangSmith 未启用，无法进行测试和评估")
    print("请设置环境变量 LANGSMITH_API_KEY 与 LANGCHAIN_TRACING_V2=true")
    sys.exit(1)

client = Client(api_key=LANGSMITH_API_KEY)

# 自动化测试专用数据集名，避免与 langsmith_evaluation.py 的 wealth-advisor-test-dataset 混用
DATASET_NAME = "wealth-advisor-eval-dataset"
EXPERIMENT_PREFIX = "agent-eval"


# ==================== Step1: 定义测试数据 ====================
# 说明：expected_outputs 中的字段会被评估器读取
#   processing_mode     期望的处理模式（reactive / deliberative）
#   should_contain      期望响应中出现的业务关键词
#   should_handle_error 边界用例：期望 Agent 给出兜底响应而不是崩溃

# 针对不同意图 - 简单查询（应由反应式层处理）
REACTIVE_TEST_CASES = [
    {
        "inputs": {"user_query": "今天上证指数的表现如何？", "customer_id": "customer1"},
        "expected_outputs": {
            "processing_mode": "reactive",
            "should_contain": ["上证指数", "点位", "涨跌"],
        },
    },
    {
        "inputs": {"user_query": "我的投资组合中科技股占比是多少？", "customer_id": "customer1"},
        "expected_outputs": {
            "processing_mode": "reactive",
            "should_contain": ["科技", "占比", "投资组合"],
        },
    },
    {
        "inputs": {"user_query": "请解释一下什么是ETF？", "customer_id": "customer1"},
        "expected_outputs": {
            "processing_mode": "reactive",
            "should_contain": ["ETF", "基金", "交易"],
        },
    },
]

# 针对不同意图 - 复杂分析（应由深思熟虑层处理）
DELIBERATIVE_TEST_CASES = [
    {
        "inputs": {
            "user_query": "根据当前市场情况，我应该如何调整投资组合以应对可能的经济衰退？",
            "customer_id": "customer1",
        },
        "expected_outputs": {
            "processing_mode": "deliberative",
            "should_contain": ["投资组合", "调整", "经济衰退", "建议"],
        },
    },
    {
        "inputs": {
            "user_query": "考虑到我的退休目标，请评估我当前的投资策略并提供优化建议。",
            "customer_id": "customer1",
        },
        "expected_outputs": {
            "processing_mode": "deliberative",
            "should_contain": ["退休", "投资策略", "评估", "建议"],
        },
    },
    {
        "inputs": {
            "user_query": "我想为子女准备教育金，请帮我设计一个10年期的投资计划。",
            "customer_id": "customer1",
        },
        "expected_outputs": {
            "processing_mode": "deliberative",
            "should_contain": ["教育金", "投资", "计划"],
        },
    },
]

# 针对边界情况 - 期望不崩溃、给出兜底响应
EDGE_CASE_TEST_CASES = [
    {
        "inputs": {"user_query": "", "customer_id": "customer1"},
        "expected_outputs": {"should_handle_error": True},
    },
    {
        "inputs": {"user_query": "这是一个非常长的查询" * 100, "customer_id": "customer1"},
        "expected_outputs": {"should_handle_error": True},
    },
]

ALL_TEST_CASES = REACTIVE_TEST_CASES + DELIBERATIVE_TEST_CASES + EDGE_CASE_TEST_CASES

# 兜底话术关键词：边界用例命中任意一个即视为"优雅处理"
ERROR_HINTS = ["无法处理", "抱歉", "错误", "未能", "无法", "请提供"]


# ==================== Step2: 创建评估器 ====================

def _get_inputs(run_or_example: Any) -> Dict[str, Any]:
    """安全获取 inputs，兼容 None 与非 dict 的情况"""
    data = getattr(run_or_example, "inputs", None)
    return data if isinstance(data, dict) else {}


def _get_outputs(run_or_example: Any) -> Dict[str, Any]:
    """安全获取 outputs，兼容 None 与非 dict 的情况"""
    data = getattr(run_or_example, "outputs", None)
    return data if isinstance(data, dict) else {}


class ProcessingModeEvaluator(RunEvaluator):
    """处理模式评估器：评估 Agent 是否正确选择了 reactive / deliberative"""

    def evaluate_run(self, run: Run, example: Example, **kwargs) -> Dict[str, Any]:
        expected_mode = _get_outputs(example).get("processing_mode")

        # 边界用例没有期望模式，跳过该指标
        if not expected_mode:
            return {
                "key": "processing_mode",
                "score": None,
                "comment": "边界用例未指定期望的处理模式，跳过",
            }

        actual_mode = _get_outputs(run).get("processing_mode")
        if actual_mode == expected_mode:
            return {
                "key": "processing_mode",
                "score": 1.0,
                "comment": f"处理模式正确: {actual_mode}",
            }
        return {
            "key": "processing_mode",
            "score": 0.0,
            "comment": f"处理模式不匹配: 期望 {expected_mode}, 实际 {actual_mode}",
        }


class ResponseCompletenessEvaluator(RunEvaluator):
    """响应完整性评估器：检查响应是否包含期望的关键信息

    对普通用例计算关键词命中比例；对边界用例改为校验是否给出兜底响应。
    """

    def evaluate_run(self, run: Run, example: Example, **kwargs) -> Dict[str, Any]:
        outputs = _get_outputs(run)
        response = str(outputs.get("final_response") or "")
        expected = _get_outputs(example)

        # 情况一：边界用例，只要求优雅兜底，不崩溃
        if expected.get("should_handle_error"):
            if not response.strip():
                return {
                    "key": "response_completeness",
                    "score": 0.0,
                    "comment": "边界用例返回空响应",
                }
            hit = any(hint in response for hint in ERROR_HINTS)
            return {
                "key": "response_completeness",
                "score": 1.0 if hit else 0.5,
                "comment": "已给出兜底响应" if hit else "有响应但未见兜底话术",
            }

        # 情况二：普通用例，按关键词命中比例打分
        expected_keywords = expected.get("should_contain") or []
        if not expected_keywords:
            return {
                "key": "response_completeness",
                "score": None,
                "comment": "未指定期望的关键词",
            }

        response_lower = response.lower()
        found = [kw for kw in expected_keywords if str(kw).lower() in response_lower]
        return {
            "key": "response_completeness",
            "score": len(found) / len(expected_keywords),
            "comment": f"命中 {len(found)}/{len(expected_keywords)} 个关键词: {found}",
        }


# ==================== Step3: 自动化测试 ====================

def ensure_dataset(dataset_name: str = DATASET_NAME) -> bool:
    """创建数据集并写入用例；已存在则复用，不重复写入"""
    try:
        client.read_dataset(dataset_name=dataset_name)
        print(f"[数据集] 已存在: {dataset_name}")
    except Exception:
        client.create_dataset(
            dataset_name=dataset_name,
            description="投顾AI助手自动化测试数据集（意图 + 边界）",
        )
        print(f"[数据集] 已创建: {dataset_name}")

    existing = list(client.list_examples(dataset_name=dataset_name))
    if existing:
        print(f"[数据集] 复用已有 {len(existing)} 条用例")
        return len(existing) > 0

    for case in ALL_TEST_CASES:
        client.create_example(
            inputs=case["inputs"],
            outputs=case.get("expected_outputs", {}),
            dataset_name=dataset_name,
        )
    total = len(list(client.list_examples(dataset_name=dataset_name)))
    print(f"[数据集] 写入 {total} 条用例")
    return total > 0


def build_target_function():
    """构造被测函数：LangSmith 会用数据集里的每条 inputs 调用它"""

    def target(inputs: Dict[str, Any]) -> Dict[str, Any]:
        user_query = str(inputs.get("user_query") or "")
        customer_id = inputs.get("customer_id", "customer1")

        # 空查询直接返回兜底，避免无意义的模型调用
        if not user_query.strip():
            return {
                "final_response": "用户查询为空，无法处理，请提供具体问题。",
                "processing_mode": "unknown",
                "query_type": "unknown",
                "error": "empty_query",
            }

        result = run_wealth_advisor(user_query=user_query, customer_id=customer_id)
        return {
            "final_response": result.get("final_response", ""),
            "processing_mode": result.get("processing_mode", "unknown"),
            "query_type": result.get("query_type", "unknown"),
            "error": result.get("error"),
        }

    return target


def print_summary(experiment_name: str, expected_count: int = len(ALL_TEST_CASES)) -> None:
    """评估结束后从云端拉取根 run 的反馈，打印逐条结果与平均分

    evaluate() 返回时反馈可能还在异步写入，因此这里做有限次重试，
    直到样本数齐全或达到重试上限。
    """
    root_runs: List[Run] = []
    for attempt in range(6):
        try:
            root_runs = list(
                client.list_runs(project_name=experiment_name, is_root=True, limit=50)
            )
        except Exception as exc:
            print(f"[汇总] 拉取实验结果失败: {exc}")
            return

        complete = all(
            bool(list(client.list_feedback(run_ids=[run.id]))) for run in root_runs
        )
        if len(root_runs) >= expected_count and complete:
            break
        print(f"[汇总] 结果尚在写入（{len(root_runs)}/{expected_count}），3 秒后重试...")
        time.sleep(3)

    print()
    print("=" * 78)
    print(f"逐条结果（共 {len(root_runs)} 条）")
    print("=" * 78)
    print(f"{'查询':<34}{'模式':<16}{'模式分':<8}{'完整性':<8}")
    print("-" * 78)

    mode_scores: List[float] = []
    completeness_scores: List[float] = []

    for run in root_runs:
        query = str(_get_inputs(run).get("user_query") or "")[:30] or "(空查询)"
        mode = str(_get_outputs(run).get("processing_mode") or "unknown")

        scores: Dict[str, Optional[float]] = {}
        try:
            for feedback in client.list_feedback(run_ids=[run.id]):
                scores[feedback.key] = feedback.score
        except Exception:
            pass

        mode_score = scores.get("processing_mode")
        comp_score = scores.get("response_completeness")
        if isinstance(mode_score, (int, float)):
            mode_scores.append(float(mode_score))
        if isinstance(comp_score, (int, float)):
            completeness_scores.append(float(comp_score))

        def fmt(value: Optional[float]) -> str:
            return "-" if value is None else f"{value:.2f}"

        print(f"{query:<34}{mode:<16}{fmt(mode_score):<8}{fmt(comp_score):<8}")

    print("-" * 78)

    def avg(values: List[float]) -> str:
        return f"{sum(values) / len(values):.2f}" if values else "-"

    print(f"平均分：处理模式 {avg(mode_scores)}（{len(mode_scores)} 条有效）"
          f"，响应完整性 {avg(completeness_scores)}（{len(completeness_scores)} 条有效）")
    print("=" * 78)


def main() -> None:
    print("=" * 78)
    print("LangChain Agent 自动化测试")
    print("=" * 78)
    print(f"被测模型: {AGENT_MODEL}")
    print(f"数据集: {DATASET_NAME}")
    print()

    print("Step1 定义测试数据")
    print("-" * 78)
    print(f"  不同意图-反应式: {len(REACTIVE_TEST_CASES)} 条")
    print(f"  不同意图-深思熟虑: {len(DELIBERATIVE_TEST_CASES)} 条")
    print(f"  边界情况: {len(EDGE_CASE_TEST_CASES)} 条")
    print(f"  合计: {len(ALL_TEST_CASES)} 条")

    if not ensure_dataset():
        print("数据集准备失败，退出")
        sys.exit(1)
    print()

    print("Step2 创建评估器")
    print("-" * 78)
    evaluators = [ProcessingModeEvaluator(), ResponseCompletenessEvaluator()]
    for evaluator in evaluators:
        print(f"  - {evaluator.__class__.__name__}")
    print()

    print("Step3 自动化测试")
    print("-" * 78)
    experiment_name = f"{EXPERIMENT_PREFIX}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    print(f"  实验名称: {experiment_name}")
    print("  串行执行（max_concurrency=1），预计需要数分钟")
    print()

    results = evaluate(
        build_target_function(),
        data=DATASET_NAME,
        evaluators=evaluators,
        experiment_prefix=EXPERIMENT_PREFIX,
        max_concurrency=1,
        metadata={"model": AGENT_MODEL, "task": "automation-testing"},
    )

    actual_experiment = getattr(results, "experiment_name", experiment_name)
    print()
    print(f"[完成] 实验已上传: {actual_experiment}")
    print_summary(actual_experiment)

    print()
    print("在 LangSmith 页面查看:")
    print("  https://smith.langchain.com -> 左侧 Experiments")
    print(f"  选择实验: {actual_experiment}")


if __name__ == "__main__":
    # 用法：
    #   python3.11 agent_automation_testing.py                 跑全量评估
    #   python3.11 agent_automation_testing.py --summary 实验名  只看已有实验的分数（不消耗模型额度）
    if len(sys.argv) >= 3 and sys.argv[1] == "--summary":
        print_summary(sys.argv[2])
    else:
        main()
