#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
判官稳定性量化（deepeval 侧）：同一判官对同一份固定回答重复打分

为什么需要它：
- 报告第十章的"改前 vs 改后"结论都建立在单次打分上，但 LLM 判官并非确定性函数；
- 本脚本用固定回答（不重跑 Agent）让同一判官重复打 3 次，产出逐指标"均值/最小/最大"，
  据此为报告结论划定可信边界（判读规则：极差 ≤0.05 稳定；0.05-0.2 结论成立但属单次抽样；>0.2 降级为方向性结论）。

固定回答来源：deepeval_results.json 已落盘的 actual_output（改后复测），
选取 1 条 reactive 短回答 + 1 条 deliberative 长回答。除重复次数外不引入任何新变量。

用法：
    python3.11 judge_stability_deepeval.py
"""

import json
import os
from datetime import datetime

from deepeval.test_case import LLMTestCase

from dashscope_judge import get_judge

# 复用改前重打分脚本的上下文映射与指标构造（Hallucination 需要 context，不重复实现）
from regrade_deepeval import CASE_CONTEXT, PROFILE_CONTEXT, build_metrics

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_FILE = os.path.join(BASE_DIR, "deepeval_results.json")
OUT_FILE = os.path.join(BASE_DIR, "judge_stability_deepeval.json")

REPEATS = 3
MAX_OUTPUT_CHARS = 3000

# 选取的固定回答：reactive 短回答 + deliberative 长回答（覆盖两种链路）
SELECTED_INPUTS = [
    ("今天上证指数表现如何？", "reactive"),
    ("考虑到我的退休目标，请评估我当前的投资策略并提供优化建议", "deliberative"),
]


def load_answers():
    """从主评估结果中取固定回答；确定性指标（ToolCorrectness）分数一并带出"""
    with open(SOURCE_FILE, encoding="utf-8") as fh:
        data = json.load(fh)
    by_input = {case["input"]: case for case in data.get("cases", [])}

    answers = []
    for query, kind in SELECTED_INPUTS:
        case = by_input.get(query)
        if not case or not case.get("actual_output"):
            raise SystemExit(f"[ERROR] 源结果中找不到回答：{query[:30]}")
        deterministic = {
            m["name"]: m["score"]
            for m in case.get("metrics", [])
            if m.get("name") == "Tool Correctness" and isinstance(m.get("score"), (int, float))
        }
        answers.append({
            "input": query,
            "kind": kind,
            "output": case["actual_output"][:MAX_OUTPUT_CHARS],
            "deterministic": deterministic,
        })
    return answers


def measure_all(test_case):
    """对固定 test_case 打一轮全部 LLM 指标。指标实例 measure 后有状态，必须每次新建。"""
    scores = {}
    for name, metric in build_metrics(get_judge()):
        try:
            metric.measure(test_case)
            scores[name] = metric.score if isinstance(metric.score, (int, float)) else None
            print(f"   {name:<26} {scores[name]}")
        except Exception as exc:
            scores[name] = None
            print(f"   {name:<26} FAIL {type(exc).__name__}: {str(exc)[:60]}")
    return scores


def summarize(scores_list):
    """对单指标的多次分数算均值/最小/最大/极差（None 不参与统计）"""
    values = [s for s in scores_list if isinstance(s, (int, float))]
    if not values:
        return {"scores": scores_list, "mean": None, "min": None, "max": None, "range": None}
    return {
        "scores": scores_list,
        "mean": round(sum(values) / len(values), 4),
        "min": round(min(values), 4),
        "max": round(max(values), 4),
        "range": round(max(values) - min(values), 4),
    }


def dump(payload):
    """逐条覆盖写，中途失败也保留已有结果"""
    with open(OUT_FILE, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    print(f"[落盘] {OUT_FILE}（已完成 {len(payload['answers'])}/{len(SELECTED_INPUTS)} 条）")


def main():
    judge_model = get_judge().get_model_name()
    answers = load_answers()
    print(f"判官 {judge_model} | 每条回答重复 {REPEATS} 次 | LLM 指标 {len(build_metrics(get_judge()))} 个")

    payload = {
        "source": "deepeval_results.json",
        "judge_model": judge_model,
        "repeats": REPEATS,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "answers": [],
    }

    for idx, item in enumerate(answers, 1):
        print(f"\n[{idx}/{len(answers)}] {item['input'][:28]}（{item['kind']}，{len(item['output'])} 字）")
        context = CASE_CONTEXT.get(item["input"], []) + PROFILE_CONTEXT
        test_case = LLMTestCase(
            input=item["input"],
            actual_output=item["output"],
            context=context,
            retrieval_context=context,
        )

        # 按指标名汇总各次分数
        per_metric = {}
        for rep in range(1, REPEATS + 1):
            print(f"  第 {rep}/{REPEATS} 次：")
            scores = measure_all(test_case)
            for name, score in scores.items():
                per_metric.setdefault(name, []).append(score)
            payload["answers"].append({
                "input": item["input"],
                "kind": item["kind"],
                "answer_chars": len(item["output"]),
                "rep": rep,
                "scores": scores,
            })
            dump(payload)  # 每轮增量落盘

        # 汇总统计并替换逐轮明细
        stats = {name: summarize(scores) for name, scores in per_metric.items()}
        payload["answers"] = [a for a in payload["answers"] if a["input"] != item["input"]]
        payload["answers"].append({
            "input": item["input"],
            "kind": item["kind"],
            "answer_chars": len(item["output"]),
            "deterministic": item["deterministic"],
            "stats": stats,
        })
        dump(payload)

    # 终端紧凑表
    print("\n" + "=" * 70)
    print("稳定性汇总（极差 = max - min）")
    print("=" * 70)
    for item in payload["answers"]:
        print(f"\n{item['input'][:30]}（{item['kind']}）")
        for name, stat in item["stats"].items():
            print(f"  {name:<26} 均值 {stat['mean']}  极差 {stat['range']}  （{stat['scores']}）")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
