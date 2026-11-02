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
│   └── assistant_ticket_bot.py          客服工单助手（另一个被测样例）
├── langfuse-observability/          LangFuse 观测
│   ├── hybrid_advisor_langgraph.py             同一 Agent + LangFuse 上报
│   ├── hybrid_advisor_qwen_agent.py            qwen-agent 版
│   ├── hybrid_advisor_qwen_agent_langfuse.py   TracedAssistant 包裹 + 手动 span
│   └── assistant_ticket_bot.py                 客服工单助手
└── openevals-evaluators/            OpenEvals 判官
    ├── hybrid_advisor_langgraph.py        被测 Agent
    ├── langsmith_evaluation.py            LangSmith 数据集评估
    ├── openevals_evaluators.py            7 个判官 × 6 条用例的完整套件
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
```

## 环境变量

复制 `.env.example` 为 `.env` 并填入真实值，或直接导出到 shell：

| 变量 | 必填 | 说明 |
|---|---|---|
| `DASHSCOPE_API_KEY` | 是 | 百炼平台 API Key，用于模型调用 |
| `LANGSMITH_API_KEY` | 是 | LangSmith API Key |
| `LANGCHAIN_TRACING_V2` | 是 | 设为 `true` 才会上报 trace |
| `LANGCHAIN_PROJECT` | 否 | 项目名，默认 `wealth-advisor-hybrid-agent` |
| `AGENT_MODEL` | 否 | 切换被测模型，默认 `qwen3.8-2.4t-a95b` |

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

## 已知限制

- 关键词命中率会惩罚同义改写（例如答“上涨 6.32 点”而期望词是“涨跌”），需要语义判断时请改用 LLM-as-judge
- 深思熟虑链路每条用例串行 4 次模型调用，`max_concurrency=1` 下全量运行需要数分钟
- 行情数据为模拟数据，仅用于演示评估流程，不构成投资建议
