#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""pytest 路径引导：让离线测试能直连导入 openevals-evaluators 下的模块。

agent 模块在 import 期构造模块级 ChatOpenAI 实例——无 DASHSCOPE_API_KEY 时
langchain-openai 会直接抛 OpenAIError。离线测试不打真实 API（全部 monkeypatch），
因此这里兜底注入假 key，保证 CI static-checks（无 secret）也能跑测试。
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("DASHSCOPE_API_KEY", "offline-test-dummy-key")

# 无 LangSmith 凭证时关闭追踪，避免测试期 401 噪音与后台重试
if not (os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")):
    os.environ["LANGCHAIN_TRACING_V2"] = "false"

EVAL_DIR = Path(__file__).resolve().parent.parent / "openevals-evaluators"
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))
