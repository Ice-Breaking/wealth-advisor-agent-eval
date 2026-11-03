#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
把 deepeval / openevals 的判官分数回写为 LangFuse score

为什么独立成脚本：
- 评估脚本不应依赖 LangFuse 网络，回写失败不能污染评估结果；
- 分数来源已经落盘为 JSON（deepeval_results.json、openevals_results_*.json），
  按 query 文本匹配 trace_ledger.json 里的 trace_id 即可回写。

用法：
    python score_backfill.py                     # 回写 deepeval + openevals（若文件存在）
    python score_backfill.py --dry-run           # 只打印将要回写的内容
"""

import argparse
import json
import os

from langfuse import get_client

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(BASE_DIR)

LEDGER_FILE = os.path.join(BASE_DIR, "trace_ledger.json")
DEEPEVAL_FILE = os.path.join(REPO_DIR, "langsmith-evaluation", "deepeval_results.json")
OPENEVAL_FILES = [
    os.path.join(REPO_DIR, "openevals-evaluators", "openevals_results_after.json"),
]
OPENEVAL_SECURITY_FILE = os.path.join(REPO_DIR, "openevals-evaluators", "openevals_results_security.json")

# 指标名做统一前缀，避免两个框架的同名指标互相覆盖
DEEPEVAL_PREFIX = "deepeval"
OPENEVAL_PREFIX = "openevals"


def load_json(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except json.JSONDecodeError:
        return None


def normalize(text: str) -> str:
    """用于 query 匹配：去掉空白与标点差异"""
    return "".join(ch for ch in str(text or "") if not ch.isspace()).rstrip("。？?！!")


def find_entry(by_query, query):
    """先精确匹配；失败则用相似度匹配（不同脚本的用例措辞略有差异，如"表现如何"vs"的表现如何"）"""
    key = normalize(query)
    if key in by_query:
        return by_query[key]
    import difflib

    candidates = difflib.get_close_matches(key, list(by_query.keys()), n=1, cutoff=0.8)
    return by_query[candidates[0]] if candidates else None


def collect_scores(wanted=None):
    """wanted 为 None 时全部回写"""
    """汇总待回写分数：[(trace_id, name, value, comment), ...]"""
    ledger = load_json(LEDGER_FILE) or {}
    entries = ledger.get("entries") or []
    by_query = {normalize(e.get("query")): e for e in entries if e.get("trace_id")}

    pending = []

    deepeval = load_json(DEEPEVAL_FILE)
    if deepeval and (wanted is None or "deepeval" in wanted):
        for case in deepeval.get("cases", []):
            entry = find_entry(by_query, case.get("input"))
            if not entry:
                continue
            for metric in case.get("metrics", []):
                score = metric.get("score")
                if not isinstance(score, (int, float)):
                    continue
                safe_name = (
                    str(metric.get("name", "metric"))
                    .lower()
                    .replace(" ", "_")
                    .replace("[", "")
                    .replace("]", "")
                )
                pending.append(
                    (
                        entry["trace_id"],
                        f"{DEEPEVAL_PREFIX}_{safe_name}",
                        round(float(score), 4),
                        str(metric.get("reason") or "")[:300],
                    )
                )

    for path in (OPENEVAL_FILES if (wanted is None or "openevals" in wanted) else []):
        openevals = load_json(path)
        if not openevals:
            continue
        for case in openevals.get("cases", []):
            entry = find_entry(by_query, case.get("query"))
            if not entry:
                continue
            for name, score in (case.get("scores") or {}).items():
                if not isinstance(score, (int, float)):
                    continue
                pending.append(
                    (
                        entry["trace_id"],
                        f"{OPENEVAL_PREFIX}_{name}",
                        round(float(score), 4),
                        f"judge={openevals.get('judge_model')}",
                    )
                )

    if wanted is None or "security" in wanted:
        security = load_json(OPENEVAL_SECURITY_FILE)
        if security:
            for case in security.get("cases", []):
                entry = find_entry(by_query, case.get("query"))
                if not entry:
                    continue
                for name, score in (case.get("scores") or {}).items():
                    if not isinstance(score, (int, float)):
                        continue
                    pending.append((entry["trace_id"], f"openevals_{name}",
                                    round(float(score), 4),
                                    f"judge={security.get('judge_model')} 安全用例"))

    return pending


def main():
    parser = argparse.ArgumentParser(description="判官分数回写 LangFuse")
    parser.add_argument("--dry-run", action="store_true", help="只打印不回写")
    parser.add_argument("--sources", default="deepeval,openevals,security",
                        help="逗号分隔，可选 deepeval / openevals / security")
    args = parser.parse_args()

    pending = collect_scores(wanted=set(args.sources.split(",")))
    if not pending:
        print("[回写] 没有可回写的分数（检查 trace_ledger 与结果文件是否存在、query 是否匹配）")
        return 0

    traces = {item[0] for item in pending}
    print(f"[回写] 待写入 {len(pending)} 条分数，覆盖 {len(traces)} 个 trace")

    if args.dry_run:
        for trace_id, name, value, _ in pending[:20]:
            print(f"  {trace_id[:12]}… {name:<40} {value}")
        print("  （dry-run，未实际写入）")
        return 0

    client = get_client()
    written = 0
    for trace_id, name, value, comment in pending:
        try:
            client.create_score(
                trace_id=trace_id,
                name=name,
                value=value,
                comment=comment,
                data_type="NUMERIC",
            )
            written += 1
        except Exception as exc:
            print(f"  [ERROR] {name} -> {trace_id[:10]}… {type(exc).__name__}: {str(exc)[:80]}")

    client.flush()
    print(f"[完成] 已写入 {written}/{len(pending)} 条分数")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
