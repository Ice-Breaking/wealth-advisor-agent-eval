#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
对已保存的 Agent 回答重新打判官分（固定 outputs，不重跑 Agent）

为什么需要它：
- 改前基线（实验 wealth-advisor-openevals-f41d8fde）的判官是 qwen3.8-max-0902，
  该模型免费额度已耗尽（403 Free quota exhausted），无法继续当判官；
- 改后必须换判官（deepseek-v4-pro-0813 + 纯文本 JSON 模式）。
- 判官一换，"改前 vs 改后"就不可比。因此用**同一新判官**对改前、改后两批
  已保存的回答各打一遍，得到可比的两列；原始 qwen 分数保留在 LangSmith 里作为历史记录。

用法：
    python regrade_openevals.py <project_name> quality <out.json>
    python regrade_openevals.py <project_name> security <out.json>
"""

import json
import sys
from datetime import datetime

from langsmith import Client
from openevals.llm import create_llm_as_judge
from openevals.prompts import (
    ANSWER_RELEVANCE_PROMPT,
    CONCISENESS_PROMPT,
    RAG_HELPFULNESS_PROMPT,
)

import openevals_evaluators as m
from json_text_judge import make_judge


def build_quality(judge):
    """7 个质量指标（与主脚本保持一致）"""
    specs = [
        ("relevance", ANSWER_RELEVANCE_PROMPT),
        ("conciseness", CONCISENESS_PROMPT),
        ("helpfulness", RAG_HELPFULNESS_PROMPT),
        ("factuality", m.FACTUALITY_PROMPT),
        ("toxicity", m.TOXICITY_SINGLE_PROMPT),
        ("processing_mode", m.PROCESSING_MODE_PROMPT),
        ("response_completeness", m.RESPONSE_COMPLETENESS_PROMPT),
    ]
    return [
        (name, create_llm_as_judge(prompt=prompt, feedback_key=name, judge=judge,
                                   continuous=True, use_reasoning=False))
        for name, prompt in specs
    ]


def build_security(judge):
    """安全指标：主口径 PII 披露 + 内置对照 + 注入防护"""
    return [
        ("pii_disclosure", m.make_pii_disclosure_evaluator(judge)),
        ("pii_raw_inverted", m.make_pii_evaluator(judge)),
        ("injection_defense", create_llm_as_judge(
            prompt=m.INJECTION_DEFENSE_PROMPT, feedback_key="injection_defense",
            judge=judge, continuous=True, use_reasoning=False)),
    ]


def main():
    if len(sys.argv) < 4:
        print(__doc__)
        return 1

    project, mode, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
    judge_model = sys.argv[4] if len(sys.argv) > 4 else "deepseek-v4.1-flash"
    judge = make_judge(judge_model)

    client = Client()

    # 支持两种来源：LangSmith 实验名，或本地重放落盘的 JSON（改前基线被误删后走本地）
    items = []
    if project.endswith(".json"):
        with open(project, encoding="utf-8") as fh:
            payload = json.load(fh)
        want_kind = "standard" if mode == "quality" else "adversarial"
        for case in payload.get("cases", []):
            if case.get("kind") and case["kind"] != want_kind:
                continue
            items.append({
                "inputs": {"user_query": case["query"], "customer_id": case.get("customer_id", "customer1")},
                "outputs": {
                    "final_response": case.get("final_response") or "",
                    "node_trace": case.get("node_trace"),
                },
            })
        print(f"本地文件 {project}：{len(items)} 条 | 判官 {judge_model} | 模式 {mode}")
    else:
        runs = [r for r in client.list_runs(project_name=project, is_root=True, limit=20)
                if r.status == "success"]
        runs.sort(key=lambda r: str(r.start_time))
        for run in runs:
            ref = {}
            ref_id = getattr(run, "reference_example_id", None)
            if ref_id:
                try:
                    ref = client.read_example(ref_id).outputs or {}
                except Exception:
                    ref = {}
            items.append({"inputs": run.inputs or {}, "outputs": run.outputs or {}, "reference": ref})
        print(f"项目 {project}：{len(items)} 条成功 run | 判官 {judge_model} | 模式 {mode}")

    evaluators = build_quality(judge) if mode == "quality" else build_security(judge)
    trajectory_evaluator = m.make_trajectory_evaluator()

    # reference 兜底：数据集每次重建后，旧实验的 reference_example_id 可能已失效，
    # 因此按 query 文本回退到脚本内的静态用例定义
    static_refs = {c["inputs"]["user_query"]: c["outputs"] for c in m.STANDARD_CASES}
    static_refs.update({c["inputs"]["user_query"]: c["outputs"] for c in m.ADVERSARIAL_CASES})

    cases = []
    for idx, item in enumerate(items, 1):
        inputs = item["inputs"]
        outputs = item["outputs"]
        query = str(inputs.get("user_query") or "")
        reference = item.get("reference") or static_refs.get(query, {})
        print(f"\n[{idx}/{len(items)}] {query[:30]}")

        scores = {}
        for name, evaluator in evaluators:
            try:
                result = evaluator(inputs=inputs, outputs=outputs, reference_outputs=reference)
                scores[name] = result.get("score")
                print(f"   {name:<22} {scores[name]}")
            except Exception as exc:
                scores[name] = None
                print(f"   {name:<22} FAIL {type(exc).__name__}: {str(exc)[:60]}")

        try:
            traj = trajectory_evaluator(inputs=inputs, outputs=outputs, reference_outputs=reference)
            scores["trajectory_strict"] = traj.get("score")
            print(f"   trajectory_strict      {scores['trajectory_strict']}  {traj.get('comment')}")
        except Exception as exc:
            scores["trajectory_strict"] = None
            print(f"   trajectory_strict      FAIL {str(exc)[:60]}")

        cases.append({
            "query": query,
            "node_trace": outputs.get("node_trace"),
            "answer_chars": len(str(outputs.get("final_response") or "")),
            "scores": scores,
        })

    payload = {
        "project": project,
        "mode": mode,
        "judge_model": judge_model,
        "regraded_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "cases": cases,
    }
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    print(f"\n[落盘] {out_path}（{len(cases)} 条）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
