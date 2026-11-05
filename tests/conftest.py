#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""pytest 路径引导：让离线测试能直连导入 openevals-evaluators 下的模块。

不安装任何额外依赖；导入 agent 模块只构造 ChatOpenAI 实例（不发起网络请求）。
"""
import sys
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent.parent / "openevals-evaluators"
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))
