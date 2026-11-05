#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
判官稳定性量化（openevals 侧）：同一判官对同一份固定回答重复打分

与 judge_stability_deepeval.py 同一目的、同一批固定回答、同一判读规则，
区别只在框架与判官模型：本脚本用 JsonTextJudge（默认 qwen3.8-omni-flash）+ 7 个质量指标。

固定回答来源：../langsmith-evaluation/deepeval_results.json 的 actual_output（改后复测，不重跑 Agent）。
reference_outputs 按 STANDARD_CASES 索引显式映射——两边 query 文本有细微差异（如"表现如何"vs"的表现如何"），
禁止按文本匹配。

用法：
    python3.11 judge_stability_openevals.py
"""

import json
import os
from datetime import datetime

from openevals_evaluators import STANDARD_CASES, make_trajectory_evaluator

# 复用重打分脚本的指标工厂（7 个质量指标），不重复实现
from regrade_openevals import build_quality
from json_text_judge import make_judge

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_FILE = os.path.join(os.path.dirname(BASE_DIR), "langsmith-evaluation", "deepeval_results.json")
# 默认文件保留 glm-5.3 判官时代的 P1 数据；换判官重测时用 STABILITY_OUT 指定新文件，
# 避免覆盖历史基线（本脚本启动不读旧文件，无续跑能力）
OUT_FILE = os.environ.get("STABILITY_OUT") or os.path.join(BASE_DIR, "judge_stability_openevals.json")

REPEATS = 3
MAX_OUTPUT_CHARS = 3000

# 选取的固定回答：ref_index 指向 openevals STANDARD_CASES 的参考 outputs（按索引，不按文本匹配）
SELECTED = [
    {"source_input": "今天上证指数表现如何？", "kind": "reactive", "ref_index": 0},
    {
        "source_input": "考虑到我的退休目标，请评估我当前的投资策略并提供优化建议",
        "kind": "deliberative",
        "ref_index": 4,
    },
]


def load_answers():
    with open(SOURCE_FILE, encoding="utf-8") as fh:
        data = json.load(fh)
    by_input = {case["input"]: case for case in data.get("cases", [])}

    answers = []
    for item in SELECTED:
        case = by_input.get(item["source_input"])
        if not case or not case.get("actual_output"):
            raise SystemExit(f"[ERROR] 源结果中找不到回答：{item['source_input'][:30]}")
        answers.append({
            "input": item["source_input"],
            "kind": item["kind"],
            "output": case["actual_output"][:MAX_OUTPUT_CHARS],
            "reference": STANDARD_CASES[item["ref_index"]]["outputs"],
        })
    return answers


def summarize(scores_list):
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
    with open(OUT_FILE, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    print(f"[落盘] {OUT_FILE}（已完成 {len(payload['answers'])}/{len(SELECTED)} 条）")


def main():
    judge_model = os.getenv("JUDGE_MODEL", "qwen3.8-omni-flash")
    judge = make_judge(judge_model)
    evaluators = build_quality(judge)
    trajectory_evaluator = make_trajectory_evaluator()
    answers = load_answers()
    print(f"判官 {judge_model} | 每条回答重复 {REPEATS} 次 | 质量指标 {len(evaluators)} 个")

    payload = {
        "source": "deepeval_results.json（回答正文） + openevals_evaluators.STANDARD_CASES（参考）",
        "judge_model": judge_model,
        "repeats": REPEATS,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "answers": [],
    }

    for idx, item in enumerate(answers, 1):
        print(f"\n[{idx}/{len(answers)}] {item['input'][:28]}（{item['kind']}，{len(item['output'])} 字）")
        inputs = {"user_query": item["input"], "customer_id": "customer1"}
        outputs = {"final_response": item["output"]}

        per_metric = {}
        for rep in range(1, REPEATS + 1):
            print(f"  第 {rep}/{REPEATS} 次：")
            for name, evaluator in evaluators:
                try:
                    result = evaluator(inputs=inputs, outputs=outputs, reference_outputs=item["reference"])
                    per_metric.setdefault(name, []).append(result.get("score"))
                    print(f"   {name:<24} {result.get('score')}")
                except Exception as exc:
                    per_metric.setdefault(name, []).append(None)
                    print(f"   {name:<24} FAIL {type(exc).__name__}: {str(exc)[:60]}")
            dump(payload)

        # 轨迹比对是确定性计算，跑一次标注方差 0
        try:
            traj = trajectory_evaluator(inputs=inputs, outputs=outputs, reference_outputs=item["reference"])
            deterministic = {"trajectory_strict": traj.get("score")}
        except Exception as exc:
            deterministic = {"trajectory_strict": f"FAIL {type(exc).__name__}: {str(exc)[:60]}"}

        payload["answers"].append({
            "input": item["input"],
            "kind": item["kind"],
            "answer_chars": len(item["output"]),
            "deterministic": deterministic,
            "stats": {name: summarize(scores) for name, scores in per_metric.items()},
        })
        dump(payload)

    print("\n" + "=" * 70)
    print("稳定性汇总（极差 = max - min）")
    print("=" * 70)
    for item in payload["answers"]:
        print(f"\n{item['input'][:30]}（{item['kind']}）")
        for name, stat in item["stats"].items():
            print(f"  {name:<24} 均值 {stat['mean']}  极差 {stat['range']}  （{stat['scores']}）")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
