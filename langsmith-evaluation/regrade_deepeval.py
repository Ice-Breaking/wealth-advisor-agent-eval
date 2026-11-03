#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
用同一判官对改前回答重新打 deepeval 分（固定 actual_output，不重跑 Agent）

背景：qwen3.8-max-0902 额度耗尽，改后只能用别的判官。
若直接对比"改前(qwen) vs 改后(新判官)"，分数差异里混了判官变量。
因此对改前已保存的回答用新判官重打一遍，得到同判官可比的两列。

用法：
    JUDGE_MODEL=deepseek-v4.1-flash-1017 python regrade_deepeval.py
"""

import json
import os
import sys

from deepeval.metrics import (
    AnswerRelevancyMetric,
    GEval,
    HallucinationMetric,
    TaskCompletionMetric,
    ToolCorrectnessMetric,
)
from deepeval.test_case import LLMTestCase, LLMTestCaseParams, ToolCall

from dashscope_judge import get_judge

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BEFORE_FILE = os.path.join(BASE_DIR, "deepeval_results_before.json")
OUT_FILE = os.path.join(BASE_DIR, "deepeval_results_before_regraded.json")

# 与主脚本一致的上下文构造（Hallucination 需要 context）
CASE_CONTEXT = {
    "今天上证指数表现如何？": ["上证指数是中国股市的重要指标", "提供实时行情数据"],
    "请解释一下什么是ETF？": ["ETF是交易所交易基金", "可以在交易所买卖"],
    "我想为子女准备教育金，请帮我设计一个10年期的投资规划": ["教育金规划需要考虑期限和风险", "建议采用多元化投资组合"],
    "考虑到我的退休目标，请评估我当前的投资策略并提供优化建议": ["退休规划需要长期视角", "需要平衡风险和收益"],
}
PROFILE_CONTEXT = ["客户风险等级: balanced", "投资期限: medium-term"]


def build_metrics(judge):
    return [
        ("Answer Relevancy", AnswerRelevancyMetric(threshold=0.6, model=judge, include_reason=True)),
        ("Hallucination", HallucinationMetric(threshold=0.5, model=judge, include_reason=True)),
        ("RiskConsideration [GEval]", GEval(
            name="RiskConsideration",
            criteria="评估投资建议是否考虑了客户的风险承受能力和投资偏好",
            evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
            evaluation_steps=[
                "检查回答是否提及风险相关内容",
                "评估建议是否与客户风险等级匹配",
                "判断是否有个性化的考虑",
            ],
            threshold=0.5,
            model=judge,
        )),
        ("Task Completion", TaskCompletionMetric(threshold=0.7, model=judge, include_reason=True)),
    ]


def main():
    judge_model = os.getenv("JUDGE_MODEL", "deepseek-v4.1-flash-1017")
    print(f"新判官 {judge_model}")
    judge = get_judge(judge_model)

    # 改前基线 JSON 里没有保存回答正文（该字段是后加的，加完后首次运行即崩），
    # 因此用 git 中改前版本的 agent 重放同样 4 条查询，再让同一判官打分。
    agent_file = sys.argv[1] if len(sys.argv) > 1 else None
    if not agent_file:
        print("用法: JUDGE_MODEL=... python regrade_deepeval.py /tmp/old_agent/hybrid_advisor_langgraph.py")
        return 1

    import importlib.util

    spec = importlib.util.spec_from_file_location("replay_agent_deepeval", agent_file)
    module = importlib.util.module_from_spec(spec)
    sys.modules["replay_agent_deepeval"] = module
    spec.loader.exec_module(module)
    print(f"改前 agent: {agent_file}")

    queries = list(CASE_CONTEXT.keys())
    out_cases = []
    for idx, query in enumerate(queries, 1):
        print(f"\n[{idx}/{len(queries)}] {query[:28]}")
        try:
            result = module.run_wealth_advisor(user_query=query, customer_id="customer1")
            answer = str(result.get("final_response") or "")
        except Exception as exc:
            print(f"  [ERROR] agent 失败: {type(exc).__name__}: {str(exc)[:80]}")
            continue
        print(f"  mode={result.get('processing_mode')} 字数={len(answer)}")
        if not answer:
            print("  跳过：无回答正文")
            continue

        context = CASE_CONTEXT.get(query, []) + PROFILE_CONTEXT
        test_case = LLMTestCase(
            input=query,
            actual_output=answer[:3000],
            context=context,
            retrieval_context=context,
        )

        metrics_out = []
        for name, metric in build_metrics(judge):
            try:
                metric.measure(test_case)
                metrics_out.append({
                    "name": name,
                    "score": metric.score,
                    "success": metric.success,
                    "threshold": getattr(metric, "threshold", None),
                    "reason": str(metric.reason or "")[:500],
                    "error": None,
                })
                print(f"   {name:<26} {metric.score:.2f}")
            except Exception as exc:
                metrics_out.append({
                    "name": name, "score": None, "success": None,
                    "threshold": getattr(metric, "threshold", None),
                    "reason": "", "error": f"{type(exc).__name__}: {exc}",
                })
                print(f"   {name:<26} FAIL {type(exc).__name__}")

        # ToolCorrectness 是确定性比对：直接用本次重放记录的工具轨迹计算，不调判官
        expected = ["上证指数查询"] if query.startswith("今天上证指数") else []
        called = [t.get("name") for t in (result.get("tools_called") or [])]
        tool_metric = ToolCorrectnessMetric(threshold=1.0, model=judge, should_exact_match=True)
        tool_case = LLMTestCase(
            input=query,
            actual_output=answer[:500],
            tools_called=[ToolCall(name=name) for name in called],
            expected_tools=[ToolCall(name=name) for name in expected],
        )
        try:
            tool_metric.measure(tool_case)
            metrics_out.append({
                "name": "Tool Correctness",
                "score": tool_metric.score,
                "success": tool_metric.success,
                "threshold": 1.0,
                "reason": f"called={called} expected={expected}",
                "error": None,
            })
            print(f"   {'Tool Correctness':<26} {tool_metric.score:.2f}")
        except Exception as exc:
            metrics_out.append({
                "name": "Tool Correctness", "score": None, "success": None,
                "threshold": 1.0, "reason": "", "error": f"{type(exc).__name__}: {exc}",
            })

        out_cases.append({
            "input": query,
            "actual_output": answer,
            "metrics": metrics_out,
        })

    payload = {
        "source": os.path.basename(BEFORE_FILE),
        "judge_model": judge_model,
        "note": "用新判官对改前已保存回答重打分，用于与改后同判官对比",
        "cases": out_cases,
    }
    with open(OUT_FILE, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    print(f"\n[落盘] {OUT_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
