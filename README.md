# 投顾 Agent 效果评估（wealth-advisor-agent-eval）

一个「混合智能体」形式的财富管理投顾 AI 助手，以及围绕它的三套可观测性与效果评估实现：

- **LangSmith**：逐节点追踪 + 数据集/实验体系 + 自定义评估器
- **LangFuse**：替换式观测方案（同一 Agent，换上报后端）
- **OpenEvals**：LLM-as-judge 判官库，覆盖相关性、简洁性、幻觉、代码正确性等通用维度

核心设计是把请求分成两条处理路径，并由协调层（LLM）动态决定走哪条：

```
开始 → 协调层 assess（判断查询类型）
  ├─ 简单/实时查询 → 反应式 reactive：快速回答，可调用行情工具
  └─ 复杂分析请求 → 深思熟虑 deliberative：数据收集 → 深度分析 → 生成建议
两条路径都 → 响应生成 respond → 结束
```

“路径选得对不对”“回答全不全”这两件事由自动化评估持续测量，从而形成「改 prompt / 换模型 → 重跑评估 → 对比实验分数」的闭环。

## 目录结构

```
.
├── langsmith-evaluation/            LangSmith 追踪 + 数据集评估 + DeepEval
│   ├── hybrid_advisor_langgraph.py      混合智能体（LangGraph 状态机）
│   ├── langsmith_evaluation.py          数据集 + 两个自定义评估器 + evaluate()
│   ├── langsmith_evaluation_example.py  单条用例的最小评估示例
│   ├── agent_automation_testing.py      完整自动化测试（支持 --summary 只读查看）
│   ├── hybrid_advisor_qwen_agent.py     同一 Agent 的 qwen-agent 声明式实现
│   ├── deepeval_wealth_advisor.py       DeepEval 指标（相关性 / 幻觉 / 自定义 GEval）
│   ├── regrade_deepeval.py              对已保存回答重打分（换判官保证可比）
│   ├── judge_stability_deepeval.py      判官稳定性量化（同一回答重复打 3 次）
│   └── assistant_ticket_bot.py          客服工单助手（另一个被测样例）
├── langfuse-observability/          LangFuse 观测
│   ├── hybrid_advisor_langgraph.py             同一 Agent + LangFuse 上报
│   ├── hybrid_advisor_qwen_agent.py            qwen-agent 版
│   ├── hybrid_advisor_qwen_agent_langfuse.py   TracedAssistant 包裹 + 手动 span
│   └── assistant_ticket_bot.py                 客服工单助手
└── openevals-evaluators/            OpenEvals 判官
    ├── hybrid_advisor_langgraph.py        被测 Agent
    ├── langsmith_evaluation.py            LangSmith 数据集评估
    ├── openevals_evaluators.py            7 个判官 × 24 条标准 + 6 条对抗用例
    ├── json_text_judge.py                 纯文本 JSON 判官（绕开厂商结构化输出差异）
    ├── regrade_openevals.py / replay_agent.py   历史实验重打分 / 指定版本 agent 重放
    ├── judge_stability_openevals.py       判官稳定性量化（同一回答重复打 3 次；STABILITY_OUT 可指定输出文件保留多判官基线）
    ├── kimi_regrade_quality.json / kimi_regrade_security.json   kimi-k3 跨判官重打分（判官可信度对照数据）
    ├── run_expanded_eval.py               P3 扩样本跑批（断点续跑 + --stats 汇总）
    ├── test_eval_gate.py                  CI 门控（pytest，阈值按实测噪声设定）
    ├── correctness.py / conciseness.py / answer_relevance.py
    ├── rag_helpfulness.py / rag_groundedness.py / rag_retrieval_relevance.py
    ├── toxicity.py / hallucination.py
    └── code_correctness.py / code_correctness_with_reference.py / plan_adherence.py
```

## 环境要求

- Python **3.11**（已在 3.11.15 上验证；3.10 / 3.12 未实测）
- 模型服务：阿里云百炼 DashScope（通过 OpenAI 兼容接口调用）
- 追踪与实验：LangSmith 账号

## 安装

各子目录依赖独立，按需安装：

```bash
python3.11 -m venv .venv && source .venv/bin/activate   # 可选

pip install -r langsmith-evaluation/requirements.txt
pip install -r langfuse-observability/requirements.txt   # 需要 LangFuse 时
pip install -r openevals-evaluators/requirements.txt     # 需要 OpenEvals 时
pip install -r requirements-ci.txt                       # 只跑 CI 门控时的最小依赖
```

## 自动化门控（CI）

push 到 main 时 GitHub Actions（`.github/workflows/eval-gate.yml`）自动运行两道检查：

- **static-checks**（免密钥、秒级）：脚本编译、三份 agent 副本 md5 一致、默认模型不指向已耗尽额度模型
- **llm-gate**（需 secret）：真实跑 2 条标准用例 + 1 条对抗用例，指标阈值按判官稳定性实测噪声设定（详见 `docs/evaluation-report.md` 第十、十五章）

首次使用需在仓库 Settings → Secrets and variables → Actions 配置 `DASHSCOPE_API_KEY`；判官/被测模型额度耗尽（403）时，改同名 Variables（`AGENT_MODEL` / `JUDGE_MODEL`）即可，无需改代码。

**扩样本统计**：全量用例（24 标准 + 6 对抗 + 6 多轮）离线跑批用 `openevals-evaluators/run_expanded_eval.py`——中断后重跑同命令即自动续跑（已完成条目跳过），`--stats` 只汇总不调模型。CI 门控刻意保持 3 条小样本以控制每次推送成本。结果与结论见 `docs/evaluation-report.md` 第十六、十八、十九章。

**离线测试与规格**：`tests/` 含 75 项零额度特征测试与可执行规格（秒级），CI static-checks 每次推送必跑；核心功能规格见 `docs/specs.md`（SPEC-ID 与测试一一对应，改规格必须同步改测试）。重构 agent 前先跑 `pytest tests/ -q` 建立基线，改完再跑全绿即行为未变。

**数据文件角色**：`openevals_results_expanded.json` 为微调前基线（P4 36 条）；`openevals_results_post_tune.json` 为 P5 prompt 微调后基线（36/36 全量）。两份保留可追溯，对比结论见报告第十九章。

**多轮支持**：agent 的 `run_wealth_advisor` 接受可选 `chat_history`（`[{"role": "user"/"assistant", "content": ...}]`），单轮调用不传即行为不变；多轮用例见 `openevals_evaluators.py` 的 `MULTI_TURN_CASES`（含跨轮路由变化与多轮铺垫注入场景）。

## 环境变量

复制 `.env.example` 为 `.env` 并填入真实值，或直接导出到 shell：

| 变量 | 必填 | 说明 |
|---|---|---|
| `DASHSCOPE_API_KEY` | 是 | 百炼平台 API Key，用于模型调用 |
| `LANGSMITH_API_KEY` | 是 | LangSmith API Key |
| `LANGCHAIN_TRACING_V2` | 是 | 设为 `true` 才会上报 trace |
| `LANGCHAIN_PROJECT` | 否 | 项目名，默认 `wealth-advisor-hybrid-agent` |
| `AGENT_MODEL` | 否 | 切换被测模型；默认值以 `.env.example` 为准（免费额度会耗尽，403 时换模型即可） |

```bash
export DASHSCOPE_API_KEY="sk-..."
export LANGSMITH_API_KEY="lsv2_pt_..."
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_PROJECT=wealth-advisor-hybrid-agent
```

## 运行

脚本之间按**同目录文件名**加载被测 Agent，因此请在对应目录内运行：

```bash
cd langsmith-evaluation

# 单轮交互（先选样例查询，再选客户）
python3.11 hybrid_advisor_langgraph.py

# 数据集 + 评估器全量评估
python3.11 langsmith_evaluation.py

# 完整自动化测试：3 条意图用例 + 3 条复杂用例 + 2 条边界用例
python3.11 agent_automation_testing.py

# 只看已有实验的分数，不重新调用模型
python3.11 agent_automation_testing.py --summary <实验名>

# 换模型跑同一套评估
AGENT_MODEL=deepseek-v4-pro-0813 python3.11 agent_automation_testing.py
```

OpenEvals 的单个判官示例可以直接跑：

```bash
cd openevals-evaluators
python3.11 conciseness.py
```

## 评估指标

`langsmith_evaluation.py` 与 `agent_automation_testing.py` 使用两个自定义 `RunEvaluator`：

| 指标 | 打分方式 | 衡量什么 |
|---|---|---|
| `processing_mode` | 期望模式与 `processing_mode` 一致得 1.0，否则 0.0；边界用例跳过 | 协调层是否把请求分到了正确的处理路径 |
| `response_completeness` | 期望关键词命中比例 0~1；边界用例校验是否给出兜底响应 | 回答是否覆盖了该有的信息点 |

结果在 LangSmith 左侧 **Experiments** 页面查看，每次运行生成一个实验，可横向对比不同模型 / prompt 版本的分数。

最新一次全量评估（qwen3.8-2.4t-a95b，8 用例）的结果与失分分析见 **[docs/evaluation-report.md](docs/evaluation-report.md)**。

## 已知限制

- 关键词命中率会惩罚同义改写（例如答“上涨 6.32 点”而期望词是“涨跌”），需要语义判断时请改用 LLM-as-judge
- 深思熟虑链路每条用例串行 4 次模型调用，`max_concurrency=1` 下全量运行需要数分钟
- 行情数据为模拟数据，仅用于演示评估流程，不构成投资建议
