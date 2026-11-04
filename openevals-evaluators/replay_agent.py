#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
用指定版本的 agent 重放用例，把回答落盘（供"同一判官重打分"使用）

背景：改前基线实验（wealth-advisor-openevals-f41d8fde）因脚本里 delete_dataset 重建
被级联删除，改前回答原文随之丢失。为让"改前 vs 改后"在同一判官下可比，
从 git 历史取出改前版本的 hybrid_advisor_langgraph.py，重放同一批用例。

用法：
    python replay_agent.py /tmp/old_agent/hybrid_advisor_langgraph.py openevals_baseline_outputs.json
"""

import importlib.util
import json
import os
import sys
from datetime import datetime

import openevals_evaluators as m


def load_agent(path):
    spec = importlib.util.spec_from_file_location("replay_agent_module", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载 {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["replay_agent_module"] = module
    spec.loader.exec_module(module)
    return module


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 1

    agent_path, out_path = sys.argv[1], sys.argv[2]
    module = load_agent(agent_path)
    print(f"agent 版本: {agent_path}")
    print(f"是否含 node_trace 支持: {hasattr(module, '_with_trace')}")

    cases = [("standard", c) for c in m.STANDARD_CASES] + [("adversarial", c) for c in m.ADVERSARIAL_CASES]
    results = []

    for idx, (kind, case) in enumerate(cases, 1):
        query = case["inputs"]["user_query"]
        customer_id = case["inputs"].get("customer_id", "customer1")
        print(f"\n[{idx}/{len(cases)}] {query[:32]}…")
        try:
            result = module.run_wealth_advisor(user_query=query, customer_id=customer_id)
            answer = str(result.get("final_response") or "")
            entry = {
                "query": query,
                "customer_id": customer_id,
                "kind": kind,
                "final_response": answer,
                "node_trace": result.get("node_trace"),
                "tools_called": [t.get("name") for t in (result.get("tools_called") or [])],
                "processing_mode": result.get("processing_mode"),
                "error": result.get("error"),
            }
            print(f"  mode={entry['processing_mode']} 字数={len(answer)} trace={entry['node_trace']}")
        except Exception as exc:
            entry = {
                "query": query,
                "customer_id": customer_id,
                "kind": kind,
                "final_response": "",
                "node_trace": None,
                "tools_called": [],
                "processing_mode": None,
                "error": f"{type(exc).__name__}: {exc}",
            }
            print(f"  [ERROR] {entry['error'][:120]}")
        results.append(entry)

    payload = {
        "agent_file": agent_path,
        "replayed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "agent_model": os.getenv("AGENT_MODEL", "deepseek-v4-pro-0813"),
        "cases": results,
    }
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    print(f"\n[落盘] {out_path}（{len(results)} 条）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
