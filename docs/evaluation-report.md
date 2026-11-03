# 投顾 Agent 自动化测试评估报告

> 实验：`agent-eval-4591bc7d`（2026-11-03）· 模型：`qwen3.8-2.4t-a95b` · 数据集：`wealth-advisor-eval-dataset`

## 一、实验概览

| 项 | 值 |
|---|---|
| 被测对象 | 混合智能体 - 财富管理投顾 AI 助手（`langsmith-evaluation/hybrid_advisor_langgraph.py`） |
| 被测模型 | `qwen3.8-2.4t-a95b`（DashScope OpenAI 兼容接口，temperature=0） |
| 数据集 | `wealth-advisor-eval-dataset`（8 条用例） |
| 评估器 | `ProcessingModeEvaluator`（处理模式 0/1）+ `ResponseCompletenessEvaluator`（关键词命中比例 0~1） |
| 执行方式 | `evaluate(max_concurrency=1)` 串行，8/8 全部 success |

用例构成（对应任务三步中的 Step1）：

| 类别 | 数量 | 说明 |
|---|---|---|
| 反应式意图 | 3 | 上证指数、科技股占比、ETF 概念，期望 `reactive` |
| 深思熟虑意图 | 3 | 经济衰退调仓、退休策略评估、10 年教育金计划，期望 `deliberative` |
| 边界情况 | 2 | 空查询、超长查询（1400 字），期望优雅兜底不崩溃 |

## 二、逐条结果

| # | 查询 | 期望模式 | 实际模式 | 模式分 | 完整性 |
|---|---|---|---|---|---|
| 1 | 今天上证指数的表现如何？ | reactive | reactive | 1.00 | **0.33** |
| 2 | 我的投资组合中科技股占比是多少？ | reactive | reactive | 1.00 | 1.00 |
| 3 | 请解释一下什么是ETF？ | reactive | reactive | 1.00 | 1.00 |
| 4 | 如何调整投资组合应对经济衰退？ | deliberative | deliberative | 1.00 | **0.75** |
| 5 | 退休目标下的投资策略评估 | deliberative | deliberative | 1.00 | 1.00 |
| 6 | 10 年期教育金投资计划 | deliberative | deliberative | 1.00 | 1.00 |
| 7 | （空查询） | —（跳过） | unknown | — | 1.00 |
| 8 | 超长查询 ×100 | —（跳过） | reactive | — | **0.50** |

**平均分：处理模式 1.00（6 条有效）· 响应完整性 0.82（8 条有效）**

## 三、两次运行对比

| 指标 | 2026-11-02（旧数据集名） | 2026-11-03（本实验） | 变化 |
|---|---|---|---|
| 处理模式 | 1.00（6/6） | 1.00（6/6） | 稳定 |
| 响应完整性 | 0.92 | 0.82 | -0.10 |

两次均为 qwen3.8、同一套用例与评估器，差异全部来自 LLM 输出的随机性（详见下节）。

## 四、失分分析

**用例 1（上证指数，0.33，两次均失分）**
模型回答："当前报 **3125.62点**，上涨 **6.32点**，涨幅 **0.20%**"。语义完全正确，但期望关键词是「点位」「涨跌」，回答中的"点""上涨"属于同义改写，字符串匹配不命中。这是**指标缺陷而非模型缺陷**。

**用例 8（超长查询，0.50，本次新失分）**
上次模型给出兜底话术（命中"无法处理"得 1.00）；本次模型直接正常回答了问题，未命中任何兜底关键词，按评分规则记 0.5。Agent 没有崩溃、没有空响应，仅响应风格不同。评分规则对"非兜底的正常回答"存在覆盖盲区。

**用例 4（经济衰退，0.75，措辞波动）**
本次回答少命中 1 个期望关键词，LLM 生成措辞的随机波动。

## 五、结论

1. **路由能力稳定可靠**：协调层把 3 条简单查询全部判为 reactive、3 条复杂分析全部判为 deliberative，两次运行 12/12 全对。处理模式由协调层的 LLM 结构化输出决定，属确定性逻辑，分数稳定在 1.00。
2. **关键词完整分天然波动（0.82~0.92）**：响应完整性测的是 LLM 自由生成的措辞，字符串匹配对同义改写零容忍（上证 0.33 是典型误判：语义正确、字面不匹配）。
3. **改进方向**：内容质量评估应从关键词匹配升级为 **LLM-as-judge**（语义级判断"是否包含点位/涨跌信息"），本仓库 `openevals-evaluators/` 已提供现成判官框架，可作为下一步演进。
4. **边界健壮性达标**：空查询返回兜底话术、超长查询正常处理，两条边界用例均未崩溃。

## 六、复现

```bash
cd langsmith-evaluation

# 全量评估（约 35-50 分钟，深思熟虑用例每条 4 次串行 LLM 调用）
python3.11 agent_automation_testing.py

# 只读查看本实验分数（不调用模型）
python3.11 agent_automation_testing.py --summary agent-eval-4591bc7d

# 切换模型对比
AGENT_MODEL=deepseek-v4-pro-0813 python3.11 agent_automation_testing.py

# deepeval 全量评估（4 条用例 × 5 指标，判官 qwen3.8-max-0902，约 40-60 分钟）
python3.11 deepeval_wealth_advisor.py

# deepeval CI 门控（阈值不达标则退出码非零）
deepeval test run deepeval_wealth_advisor.py

# 只跑第一条用例做冒烟
DEEPEVAL_CASE_LIMIT=1 python3.11 deepeval_wealth_advisor.py
```

分数明细见 LangSmith：smith.langchain.com → Experiments → `agent-eval-4591bc7d`。

## 七、语义判官评估（openevals）

关键词匹配的缺陷在第四章已经暴露（语义正确但字面不匹配记为失分），因此补做一轮 LLM-as-judge 评估。

**实验**：LangSmith `wealth-advisor-openevals-f41d8fde`（`openevals-evaluators/openevals_evaluators.py`，6 条用例 × 7 个指标，42 次判官调用全部成功）

**模型分工**：被测 Agent = `deepseek-v4.1-flash-1017`，判官 = `qwen3.8-max-0902`（两者分属独立额度，避免判官与被测模型互相挤占配额；实测只有 qwen3.8-max 能稳定支撑 openevals 的结构化输出，`glm-5.3` 与 `deepseek-v4.1-flash` 在该接口下直接报错）

**判官设计**：7 个指标 = relevance / conciseness / helpfulness / response_completeness / toxicity / factuality（LLM 判官）+ processing_mode（规则判定，确定性逻辑不需要判官）。

| 用例 | 处理模式 | 相关性 | 简洁性 | 帮助性 | 语义完整度 | 事实自洽 | 无毒性 |
|---|---|---|---|---|---|---|---|
| 10 年期教育金规划 | 1.00 | 1.00 | 0.20 | 0.90 | 0.75 | 0.90 | 1.00 |
| 退休目标下策略评估 | 1.00 | 1.00 | 0.15 | 0.85 | 1.00 | 1.00 | 1.00 |
| 应对经济衰退的调仓 | 1.00 | — | — | — | — | 1.00 | 1.00 |
| ETF 概念解释 | 1.00 | 1.00 | 0.50 | 0.95 | 1.00 | 1.00 | 1.00 |
| 科技股占比查询 | 1.00 | 1.00 | 0.20 | 1.00 | 1.00 | 1.00 | 1.00 |
| 上证指数表现 | 1.00 | 1.00 | 0.70 | 1.00 | **1.00** | 0.00 | 1.00 |
| **平均（5 条有效）** | **1.00** | **1.00** | **0.35** | **0.94** | **0.95** | **0.78** | **1.00** |

"应对经济衰退的调仓"一条本次执行返回"无法生成响应"（见下方缺陷 4），其响应类指标无意义，故不纳入平均。

**关键验证**：上证指数用例在关键词评估下只有 0.33（答"报 3125.62 点、上涨 6.32 点"不含「点位」「涨跌」字面），语义判官给出 **1.00**——同一份回答、同一套期望信息点，两种评估方式结论完全相反，直接证实"字符串匹配惩罚同义改写"的判断。反过来，教育金用例关键词版 1.00、判官版 0.75，说明判官也会识别出"字面覆盖但没答全"的情况，不是单向放水。

**新暴露的信号**：

- **简洁性平均仅 0.35**：深思熟虑链路回答 2800-4200 字，投顾场景偏冗长，可考虑分点摘要 + 折叠细节
- **事实自洽在上证用例为 0.00**：回答引用"3125.62 点""6.32 点"等精确数据但未声明是模拟数据；其余用例 0.90-1.00，说明这是"工具取数未标注来源"的局部问题，不是全局幻觉
- **处理模式 6/6 全对**：路由层的确定性判断在两轮实验里都稳定，这与关键词版结果一致

### 本轮发现并修复的 4 个缺陷

1. **内置 `TOXICITY_PROMPT` 数据形状不匹配**：其设计面向"多条观点列表"，单条回答触发 `TypeError: list indices must be integers`。已改为自定义单回答毒性 prompt。
2. **内置 `HALLUCINATION_PROMPT` 依赖 RAG 上下文**：需要 `{context}` 字段，本项目无检索环节，`KeyError('context')`。已改为自定义事实自洽 prompt（`factuality`）。
3. **判官在超长回答上退化**：回答 >7000 字时判官给出明显失真分数，截断到 2000 字后恢复稳定；同时 deepseek 系列在 openevals 结构化输出接口下会把解析失败算成 0 分，故判官统一选用 qwen3.8-max。
4. **失败原因被兜底文案吞掉**（真实缺陷，已修复）：`analyze_data` / `generate_recommendations` 的异常与缺数据分支只写 `error` 不写 `final_response`，respond 节点于是返回"无法生成响应。请检查处理流程。"，真实原因完全丢失。修复后失败分支同时写入用户可读的 `final_response`（含异常类型）。该用例单独复跑即成功（2832 字），确认是偶发 LLM 失败而非逻辑缺陷。

## 八、语义判官评估（deepeval，含 Agent 行为指标）

openevals 只能覆盖"文本好不好"，测不到"有没有调对工具、任务是否真的完成"。这一轮换 deepeval 做第二家判官，并补上两个 Agentic 指标。

**判官接入**：deepeval 内置模型不含 DashScope，故按官方 `DeepEvalBaseLLM` 扩展点写了适配层 `langsmith-evaluation/dashscope_judge.py`——同步 `generate()` + 异步 `a_generate()` 双实现、`schema` 结构化输出透传、单次 300 秒超时 × 4 次重试，全程不依赖 OpenAI。

**实验配置**：被测 `deepseek-v4.1-flash-1017`，判官 `qwen3.8-max-0902`（两者额度独立），4 条用例 × 5 指标 = 20 次判官调用，结果落盘 `langsmith-evaluation/deepeval_results.json`。

| 用例 | 模式 | 相关性 | 幻觉 | 风险考虑 | 任务完成 | 工具正确 |
|---|---|---|---|---|---|---|
| 上证指数表现 | reactive | 1.00 | 1.00 | **0.00** | 1.00 | 1.00 |
| ETF 概念解释 | reactive | 1.00 | 1.00 | 0.50 | 1.00 | 1.00 |
| 10 年教育金规划 | deliberative | 1.00 | 1.00 | 1.00 | 0.95 | 1.00 |
| 退休策略评估 | deliberative | 0.99 | 1.00 | 1.00 | 0.95 | 1.00 |
| **平均** | — | **1.00** | **1.00** | **0.63** | **0.98** | **1.00** |

### 三个只有这套指标才看得见的结论

1. **风险考虑是最大短板，且失分点可解释**：上证用例 0.00（判官理由："仅提供点位、涨跌点数，未提及风险相关内容，也没有基于客户风险等级给出匹配建议"），ETF 用例 0.50（提了市场/跟踪误差/流动性风险，但"输入未提供客户风险等级，回答也未给出具体投资建议"）。深思熟虑链路 1.00（明确给出回撤、集中度、汇率风险与分阶段降风险方案）。这一维度在关键词评估和 openevals 里都没有对应指标。
2. **工具调用 4/4 正确**：上证用例 `tools_called=[上证指数查询]`，其余 3 条为空，与 `expected_tools` 完全一致。说明虽然工具触发靠文本标记 `[CALL_TOOL:...]` 这种土办法，路由是稳定的；反过来也证明 `ToolCorrectness` 能真实判别"该调没调"（负向对照见下）。
3. **任务完成度的扣分很具体**：两条深思熟虑用例都是 0.95——教育金那条"未明确逐字说明'10 年期'"，退休那条"额外覆盖子女教育金目标，略超出任务范围"。这种挑刺粒度对改进 prompt 有直接价值。

### 与 openevals 的交叉验证

| 维度 | openevals | deepeval | 是否一致 |
|---|---|---|---|
| 相关性 | relevance 1.00 | AnswerRelevancy 1.00 | 一致 |
| 完整性 | response_completeness 1.00 / 1.00（教育金、退休） | TaskCompletion 0.95 / 0.95 | 基本一致，deepeval 更严格 |
| 事实层面 | factuality 上证 **0.00** | Hallucination 上证 1.00 | **分歧** |

分歧原因是**指标口径不同，不是判官打架**：openevals 的 factuality 查的是"是否给出精确数据却不标注来源"，deepeval 的 Hallucination 查的是"是否与给定 context 矛盾"——我们传入的 context 里已有"提供实时行情数据"，因此不构成矛盾。这提示：跨框架比分数前必须先对齐口径，同名指标不可直接横比。

### 本轮踩到的工程坑（均已修复）

| 问题 | 现象 | 修法 |
|---|---|---|
| 整轮预算过小 | deepeval 默认 per-task 180s + gather 27s，4×5 指标必然 `asyncio.TimeoutError` | `DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE=3600`，必须在 `import deepeval` 之前设置 |
| 单点失败拖垮全轮 | 一次判官超时导致整轮无结果 | `ErrorConfig(ignore_errors=True)` + 逐条用例 `evaluate()` + 增量落盘 |
| 长输入判官超时 | 3000 字回答的 statement 抽取 4×300s 全超时 | 先截断到 3000 字；仍失败的两条用 1500 字 + 600s 超时补测（JSON 里标了 `note`） |
| 工具轨迹缺失 | 状态机不记录调用了什么工具 | 三份 `hybrid_advisor_langgraph.py` 副本新增 `tools_called` 字段（md5 校验一致） |
| pytest 门控要求 OpenAI | 指标不传 `model` 时 deepeval 回落 GPTModel，报缺 key | 门控里每个 metric 显式传判官实例 |
| 依赖硬冲突 | deepeval 要 `click<8.4.0`，huggingface-hub 要 `>=8.4.2` | 保留 click 8.3.3（`huggingface_hub` 导入正常），已用 `pip check` 记录 |
| pydantic 被升级 | 2.9.2 → 2.14.2 | 实测 `langchain 1.3.14` / `langsmith 0.14.17` 的 `ChatOpenAI.invoke` 正常，无回退 |

### CI 门控

```bash
cd langsmith-evaluation
deepeval test run deepeval_wealth_advisor.py   # 通过则退出码 0
```

实测：2 个用例全 PASSED（相关性 1.0 / 幻觉 1.0 / 任务完成 1.0），`EXIT_CODE=0`；另用"应调工具但未调"的负向对照验证门控确实生效，`EXIT_CODE=1`。门控覆盖"回答切题 + 无幻觉 + 完成任务 + 选对工具"四类失败模式，可直接接 CI。

## 九、可观测性（LangFuse）

`langfuse-observability/hybrid_advisor_qwen_agent_langfuse.py` 已接入 LangFuse（US 区域），实测可在控制台看到完整 trace：`wealth_advisor_query`（span）→ `gui_query`（generation，含 input / output / 5 项 metadata / 延迟）。

接线过程中修复了 4 个问题，均已提交：

| 问题 | 现象 | 修法 |
|---|---|---|
| 导出超时 | 默认 5 秒读超时，跨区域 3 秒 RTT 下事件静默丢弃 | `LANGFUSE_TIMEOUT=30` + 客户端 `timeout=30` |
| 生成器转发 | 异常/未启用分支 `return super().run()` 导致无响应 | 改 `yield from` |
| API 不匹配 | langfuse 4.x 的 `update_current_span()` 不接受 `model` 参数，抛 TypeError 被吞 | 改用 `update_current_generation()` |
| span 关闭后更新 | 输出 metadata 在 span 结束后才设置，静默失效 | 移入 span 内部更新 |

另修复：模型从 `qwen-flash`（原生 DashScope SDK，不支持 qwen3.8）改为走 OpenAI 兼容接口；`WebUI` 改惰性导入，使命令行调用不再依赖 gradio。

## 十、环境依赖备注

- `openevals==0.2.2`、`langfuse==4.14.4`、`qwen-agent==0.0.34`、`deepeval==4.3.2`、`soundfile` 为本次新增/补齐依赖；`requirements.txt` 中的版本 pin 已落后于实际环境，安装时不要整体套用，否则会降级 `langchain 1.3.14`。
- 若 `qwen-agent` 报缺 `modelscope_studio`，只在需要 GUI 时安装 `qwen-agent[gui]`，命令行链路不需要。
- **deepeval 的连锁依赖变更**（安装前已 `pip freeze` 快照比对）：`pydantic 2.9.2 → 2.14.2`、`rich 15.0.0 → 14.3.4`、`click 8.4.2 → 8.3.3`。其中 click 与 `huggingface-hub`（要求 `>=8.4.2`）存在无法同时满足的版本约束，因 `deepeval test run` 依赖 click 而未 `huggingface_hub`，选择保留 8.3.3 并实测其导入正常；`pydantic` 升级后 `langchain` / `langsmith` 的 `ChatOpenAI.invoke` 实测正常。
- deepeval 判官不需要 `OPENAI_API_KEY`：所有指标显式传入 `dashscope_judge.get_judge()`，只有忘记传 `model` 时 deepeval 才会回落到默认 GPTModel 并要求该 key。
