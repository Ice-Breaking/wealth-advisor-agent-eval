#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DeepEval 评估脚本 - 投顾AI助手质量测试

使用 DeepEval 对投顾AI助手进行自动化质量评估，共 5 个指标：
1. AnswerRelevancy  - 回答是否切题
2. Hallucination    - 是否编造与上下文矛盾的信息
3. GEval(RiskConsideration) - 是否考虑了客户风险偏好
4. TaskCompletion   - 是否真正完成了用户任务（Agentic）
5. ToolCorrectness  - 工具调用是否选对（Agentic，确定性比对）

判官模型：DashScope 兼容接口的 LLM（经 dashscope_judge.py 适配，不再依赖 OpenAI）。
默认 deepseek-v4.1-flash-1017，可用 JUDGE_MODEL 环境变量切换；
qwen3.8-max-0902 曾是最稳的结构化输出判官，但其免费额度已耗尽（403）。

安装依赖：
    pip install deepeval        # 注意：不要整装 requirements.txt，会降级 langchain

运行方式：
    python deepeval_wealth_advisor.py                    # 全量（4 条用例）
    deepeval test run deepeval_wealth_advisor.py         # pytest 门控，可进 CI

环境变量：
    DASHSCOPE_API_KEY: 通义千问 API 密钥（投顾助手 + 判官共用）
    AGENT_MODEL: 被测模型，默认 deepseek-v4.1-flash-1017
"""

import importlib.util
import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, List

# deepeval 默认整轮预算 = per-task 180s + gather buffer 27s，4 条用例 × 5 指标必然被掐断
# （表现为 asyncio TimeoutError）。这里放宽到 1 小时，必须在 import deepeval 之前设置。
os.environ.setdefault("DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE", "3600")

# DeepEval 导入
from deepeval import assert_test, evaluate
from deepeval.evaluate.configs import AsyncConfig, DisplayConfig, ErrorConfig
from deepeval.metrics import (
    AnswerRelevancyMetric,
    GEval,
    HallucinationMetric,
    TaskCompletionMetric,
    ToolCorrectnessMetric,
)
from deepeval.test_case import LLMTestCase, LLMTestCaseParams, ToolCall

# 导入投顾助手：用 importlib 按文件名加载同目录模块
spec = importlib.util.spec_from_file_location(
    "hybrid_advisor_langgraph",
    "hybrid_advisor_langgraph.py"
)
hybrid_module = importlib.util.module_from_spec(spec)
sys.modules["hybrid_advisor_langgraph"] = hybrid_module
spec.loader.exec_module(hybrid_module)

from hybrid_advisor_langgraph import (  # noqa: E402
    run_wealth_advisor,
    SAMPLE_CUSTOMER_PROFILES
)
from dashscope_judge import get_judge  # noqa: E402

# 结果落盘位置（相对本文件目录），可用 DEEPEVAL_RESULT_FILE 指定，
# 便于同时保留"改前基线"与"改后复测"两份结果
RESULT_FILE = os.getenv("DEEPEVAL_RESULT_FILE") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "deepeval_results.json"
)

# 工具名需与 hybrid_advisor_langgraph 中 tools_called 记录的名称一致
TOOL_INDEX = "上证指数查询"

# 判官输入截断：深思熟虑链路回答可达 4000+ 字，判官（思考型模型）在超长输入上偶发超时
# 结论性内容集中在开头，截断到 3000 字既稳定又不失真；ToolCorrectness 只比对工具名，不受影响
MAX_OUTPUT_CHARS = 3000


# ==================== 测试用例定义 ====================

# 反应式查询测试用例（简单查询）
REACTIVE_TEST_CASES = [
    {
        "input": "今天上证指数表现如何？",
        "customer_id": "customer1",
        "context": ["上证指数是中国股市的重要指标", "提供实时行情数据"],
        "expected_keywords": ["上证指数", "点位"],
        "expected_tools": [TOOL_INDEX],  # 必须调用行情工具
    },
    {
        "input": "请解释一下什么是ETF？",
        "customer_id": "customer1",
        "context": ["ETF是交易所交易基金", "可以在交易所买卖"],
        "expected_keywords": ["ETF", "基金", "交易"],
        "expected_tools": [],  # 概念解释，不应调工具
    },
    {
        "input": "我的投资组合中科技股占比是多少？",
        "customer_id": "customer1",
        "context": ["科技股包括半导体、软件、互联网等行业", "需要持仓明细才能计算占比"],
        "expected_keywords": ["科技股", "占比"],
        "expected_tools": [],
    },
]

# 深思熟虑查询测试用例（复杂分析）
DELIBERATIVE_TEST_CASES = [
    {
        "input": "我想为子女准备教育金，请帮我设计一个10年期的投资规划",
        "customer_id": "customer1",
        "context": ["教育金规划需要考虑期限和风险", "建议采用多元化投资组合"],
        "expected_keywords": ["教育金", "投资", "规划"],
        "expected_tools": [],  # 深度分析不走工具
    },
    {
        "input": "考虑到我的退休目标，请评估我当前的投资策略并提供优化建议",
        "customer_id": "customer1",
        "context": ["退休规划需要长期视角", "需要平衡风险和收益"],
        "expected_keywords": ["退休", "投资策略", "优化"],
        "expected_tools": [],
    },
    {
        "input": "根据当前市场情况，我应该如何调整投资组合以应对可能的经济衰退？",
        "customer_id": "customer1",
        "context": ["经济衰退时建议增加防御性资产", "需要评估风险承受能力"],
        "expected_keywords": ["市场", "调整", "衰退"],
        "expected_tools": [],
    },
]

# 本轮运行使用的用例：2 条反应式 + 2 条深思熟虑
# 控制理由：深思熟虑单条约 2.5 分钟、判官单条约 20-40 秒，4 条可在 25 分钟内跑完；
# 其余用例保留在文件中，需要更大样本时改这里即可
ACTIVE_CASES = REACTIVE_TEST_CASES[:2] + DELIBERATIVE_TEST_CASES[:2]

# 调试开关：DEEPEVAL_CASE_LIMIT=1 只跑第一条，便于先验证判官链路再全量
_case_limit = os.getenv("DEEPEVAL_CASE_LIMIT")
if _case_limit and _case_limit.isdigit():
    ACTIVE_CASES = ACTIVE_CASES[: int(_case_limit)]


# ==================== 测试用例构造 ====================

def run_agent_and_get_response(query: str, customer_id: str) -> Dict[str, Any]:
    """运行投顾助手并获取响应"""
    print(f"\n[测试] 运行查询: {query[:50]}...")
    result = run_wealth_advisor(user_query=query, customer_id=customer_id)
    return result


def build_tool_calls(raw_calls: List[Dict[str, Any]]) -> List[ToolCall]:
    """把 Agent 记录的工具轨迹转成 DeepEval 的 ToolCall 对象"""
    return [
        ToolCall(
            name=call.get("name", ""),
            input_parameters=call.get("input_parameters") or {},
            output=call.get("output"),
        )
        for call in raw_calls
    ]


def create_test_case(
    test_data: Dict[str, Any],
    actual_output: str,
    retrieval_context: List[str],
    tools_called: List[Dict[str, Any]],
) -> LLMTestCase:
    """创建 DeepEval 测试用例"""
    return LLMTestCase(
        input=test_data["input"],
        actual_output=actual_output[:MAX_OUTPUT_CHARS],
        context=retrieval_context,  # HallucinationMetric 读取 context 字段
        retrieval_context=retrieval_context,
        tools_called=build_tool_calls(tools_called),
        expected_tools=[ToolCall(name=name) for name in test_data.get("expected_tools", [])],
    )


# ==================== 评估指标定义 ====================

def get_metrics():
    """获取评估指标列表（判官由 dashscope_judge.DEFAULT_MODEL / JUDGE_MODEL 决定）"""
    judge = get_judge()

    # 1. 答案相关性：评估回答是否与问题相关
    relevancy_metric = AnswerRelevancyMetric(
        threshold=0.6,
        model=judge,
        include_reason=True
    )

    # 2. 幻觉检测：检测回答是否与给定上下文矛盾
    hallucination_metric = HallucinationMetric(
        threshold=0.5,
        model=judge,
        include_reason=True
    )

    # 3. 自定义评估：是否考虑客户风险偏好
    risk_consideration_metric = GEval(
        name="RiskConsideration",
        criteria="评估投资建议是否考虑了客户的风险承受能力和投资偏好",
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT
        ],
        evaluation_steps=[
            "检查回答是否提及风险相关内容",
            "评估建议是否与客户风险等级匹配",
            "判断是否有个性化的考虑"
        ],
        threshold=0.5,
        model=judge
    )

    # 4. Agentic：任务是否完成（task 不传，由 DeepEval 从用例输入/输出推导）
    task_completion_metric = TaskCompletionMetric(
        threshold=0.7,
        model=judge,
        include_reason=True
    )

    # 5. Agentic：工具调用是否正确（确定性比对，不调 LLM）
    tool_correctness_metric = ToolCorrectnessMetric(
        threshold=1.0,
        model=judge,
        should_exact_match=True,
    )

    return [
        relevancy_metric,
        hallucination_metric,
        risk_consideration_metric,
        task_completion_metric,
        tool_correctness_metric,
    ]


# ==================== 结果落盘 ====================

def to_case_dict(test_result: Any) -> Dict[str, Any]:
    """把 deepeval 的单个用例结果转成可序列化 dict（含失败指标）"""
    case: Dict[str, Any] = {
        "input": test_result.input,
        # 保留判官实际看到的回答正文，便于复现与人工复核
        # 注意：deepeval 的 TestResult 没有 tools_called 属性（工具轨迹在 metrics_data 里体现），
        # 这里全部用 getattr 兜底，避免不同版本字段缺失导致整条用例丢失
        "actual_output": (getattr(test_result, "actual_output", "") or "")[:MAX_OUTPUT_CHARS],
        "tools_called": [
            getattr(tool, "name", str(tool))
            for tool in (getattr(test_result, "tools_called", None) or [])
        ],
        "metrics": [],
    }
    for data in getattr(test_result, "metrics_data", []):
        case["metrics"].append({
            "name": data.name,
            "score": data.score,
            "success": data.success,
            "threshold": data.threshold,
            "reason": (data.reason or "")[:500],
            "error": getattr(data, "error", None),
        })
    return case


def dump_results(cases: List[Dict[str, Any]], skipped: List[str]) -> None:
    """把逐条指标分数与理由写成 JSON，便于贴进报告（逐条覆盖写，中途失败也保留已有结果）"""
    payload = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "judge_model": get_judge().get_model_name(),
        "agent_model": os.getenv("AGENT_MODEL", "deepseek-v4.1-flash-1017"),
        "skipped_cases": skipped,
        "cases": cases,
    }
    with open(RESULT_FILE, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    print(f"[落盘] 已写入 {RESULT_FILE}（累计 {len(cases)} 条用例）")


def print_summary(cases: List[Dict[str, Any]], skipped: List[str]) -> None:
    """终端打印紧凑分数表"""
    print("\n" + "=" * 78)
    print("逐条结果（√ 表示达到阈值，! 表示判官调用失败）")
    print("=" * 78)
    for case in cases:
        print(f"\n用例: {str(case['input'])[:40]}")
        for data in case["metrics"]:
            if data["error"]:
                print(f"  ! {data['name']:<28} 判官失败: {str(data['error'])[:60]}")
                continue
            flag = "√" if data["success"] else "×"
            score = data["score"] if isinstance(data["score"], (int, float)) else float("nan")
            print(f"  {flag} {data['name']:<28} {score:.2f} (阈值 {data['threshold']})")
    if skipped:
        print("\n[跳过用例]（Agent 未返回有效响应）")
        for item in skipped:
            print(f"  - {item}")


# ==================== 主评估流程 ====================

def run_evaluation():
    """运行完整评估流程"""

    print("=" * 60)
    print("DeepEval 投顾AI助手评估")
    print("=" * 60)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"判官模型: {get_judge().get_model_name()}")
    print(f"被测模型: {os.getenv('AGENT_MODEL', 'deepseek-v4.1-flash-1017')}")
    print()

    test_cases = []
    skipped: List[str] = []

    print(f"[准备] 共 {len(ACTIVE_CASES)} 个测试用例")
    print("-" * 60)

    for i, test_data in enumerate(ACTIVE_CASES, 1):
        print(f"\n[{i}/{len(ACTIVE_CASES)}] 测试: {test_data['input'][:40]}...")

        result = run_agent_and_get_response(
            query=test_data["input"],
            customer_id=test_data["customer_id"]
        )

        actual_output = result.get("final_response", "")
        if not actual_output:
            print("  [警告] 未获取到响应，跳过此用例")
            skipped.append(test_data["input"])
            continue

        print(f"  [响应] {actual_output[:100]}...")
        print(f"  [工具] {result.get('tools_called') or []}")

        customer_profile = SAMPLE_CUSTOMER_PROFILES.get(test_data["customer_id"], {})
        retrieval_context = test_data["context"] + [
            f"客户风险等级: {customer_profile.get('risk_tolerance', '未知')}",
            f"投资期限: {customer_profile.get('investment_horizon', '未知')}"
        ]

        test_cases.append(create_test_case(
            test_data=test_data,
            actual_output=actual_output,
            retrieval_context=retrieval_context,
            tools_called=result.get("tools_called") or [],
        ))

    if not test_cases:
        print("\n[中止] 没有可评估的用例")
        return None

    print("\n" + "=" * 60)
    print(f"[评估] {len(test_cases)} 个用例 × {len(get_metrics())} 个指标")
    print("=" * 60)

    # 逐条用例单独评估：判官偶发超时只影响当前用例，已完成的分数立即落盘
    cases: List[Dict[str, Any]] = []
    for idx, test_case in enumerate(test_cases, 1):
        print(f"\n[判官] {idx}/{len(test_cases)} {str(test_case.input)[:30]}...")
        try:
            result = evaluate(
                test_cases=[test_case],
                metrics=get_metrics(),
                # 判官是思考型模型，并发压到 2 避免限流
                async_config=AsyncConfig(run_async=True, throttle_value=0, max_concurrent=2),
                # 关闭 deepeval 自带富文本表格，改用本脚本的紧凑输出 + JSON 落盘
                display_config=DisplayConfig(print_results=False, show_indicator=True),
                # 单个指标失败不中断整条用例
                error_config=ErrorConfig(ignore_errors=True, skip_on_missing_params=True),
            )
            for test_result in result.test_results:
                cases.append(to_case_dict(test_result))
        except Exception as exc:  # 极端情况（如整轮超时）也要保住已有结果
            print(f"  [错误] 该用例评估失败: {type(exc).__name__}: {str(exc)[:100]}")
        dump_results(cases, skipped)

    print_summary(cases, skipped)

    print("\n" + "=" * 60)
    print("评估完成")
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    return cases


# ==================== Pytest 风格测试（CI 门控） ====================

def _build_case(test_data: Dict[str, Any]) -> LLMTestCase:
    """跑一次 Agent 并构造测试用例（pytest 用例共用）"""
    result = run_agent_and_get_response(
        query=test_data["input"],
        customer_id=test_data["customer_id"]
    )

    customer_profile = SAMPLE_CUSTOMER_PROFILES.get(test_data["customer_id"], {})
    context = test_data["context"] + [
        f"客户风险等级: {customer_profile.get('risk_tolerance', '未知')}"
    ]

    return create_test_case(
        test_data=test_data,
        actual_output=result.get("final_response", ""),
        retrieval_context=context,
        tools_called=result.get("tools_called") or [],
    )


def test_reactive_query():
    """门控：反应式查询需切题且选对工具"""
    test_case = _build_case(REACTIVE_TEST_CASES[0])
    judge = get_judge()
    metrics = [
        AnswerRelevancyMetric(threshold=0.6, model=judge),
        ToolCorrectnessMetric(threshold=1.0, model=judge, should_exact_match=True),
    ]
    assert_test(test_case, metrics)


def test_deliberative_query():
    """门控：深思熟虑查询需切题、无幻觉、完成任务"""
    test_case = _build_case(DELIBERATIVE_TEST_CASES[0])
    judge = get_judge()
    metrics = [
        AnswerRelevancyMetric(threshold=0.6, model=judge),
        HallucinationMetric(threshold=0.5, model=judge),
        TaskCompletionMetric(threshold=0.7, model=judge),
    ]
    assert_test(test_case, metrics)


# ==================== 入口 ====================

if __name__ == "__main__":
    if not os.getenv("DASHSCOPE_API_KEY"):
        print("[错误] 请设置 DASHSCOPE_API_KEY 环境变量（投顾助手与判官共用）")
        exit(1)

    run_evaluation()
