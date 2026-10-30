# ReAct Agent 能力升级说明

更新时间：2026-05-23

本次升级没有推倒现有 deterministic workflow，而是在现有金融投研链路上补齐三类 Agent 工程能力：

```text
固定投研 workflow
  + Function Calling 结构化决策
  + 受控 Python 表格分析工具
  + workflow trace / 失败记录
= ReAct-lite 投研 Agent 基座
```

## 1. Function Calling 结构化输出

新增文件：

```text
src/agent/structured_decision.py
```

核心能力：

- 定义 `InvestmentDecision` JSON Schema
- 通过 `submit_investment_decision` 函数约束 LLM 输出
- 输出字段包括：
  - `verdict`: Buy / Watch / Avoid
  - `confidence`
  - `analysis`
  - `key_reasons`
  - `risk_factors`
  - `evidence_refs`
  - `missing_info`

接入位置：

```text
src/agent/llm_synthesizer.py
```

调用逻辑：

1. OpenAI-compatible 后端优先使用 Function Calling
2. 如果 provider 不支持 tools，则尝试解析 JSON 文本
3. 如果仍失败，回退到原有自由文本 + 正则 verdict

## 2. 受控 Python 分析工具

新增文件：

```text
src/agent/python_executor.py
```

核心能力：

- 使用 AST 检查屏蔽 `import/open/eval/exec` 等高风险操作
- 子进程隔离执行，支持超时终止
- 执行环境只暴露：
  - `pandas`
  - `numpy`
  - `load_panel`
  - `DEFAULT_TUSHARE_DATA_DIR`
- 要求脚本把最终答案写入 `result`

当前内置示例：

```text
fac_eps_revision_60d Top20% - Bottom20% 未来5日收益差
```

报告中会生成：

```text
Python 动态分析
- EPS 修正 Top20% 组合未来5日平均收益
- EPS 修正 Bottom20% 组合未来5日平均收益
- 多空收益差
- 收益差为正的交易日占比
```

这个能力对应 ReAct 中的“需要计算时调用 Python 工具”。

## 3. Workflow Trace

新增文件：

```text
src/agent/trace.py
```

核心能力：

- 记录每个工具调用：
  - 工具名
  - 是否成功
  - 耗时
  - 输出摘要
  - 异常信息
- 报告中展示 trace 摘要
- CLI 会保存完整 JSON trace 到：

```text
reports/traces/
```

示例命令：

```bash
.venv/bin/python scripts/run_research.py \
  --stock-code 300308 \
  --theme "AI 光模块" \
  --question "中际旭创现在还值得关注吗？" \
  --top-k 2 \
  --no-llm \
  --output reports/smoke_research.md
```

其中 `--no-llm` 用于离线 smoke test；去掉该参数后会启用结构化 LLM 决策。

## 4. 当前边界

当前还不是完整 LangGraph ReAct：

- 工具选择仍由确定性 workflow 控制
- LLM 负责结构化决策，不负责自由循环调工具
- Python 工具是受控执行器，不是开放 REPL

这是有意保守的设计。它已经能体现 Agent 工程能力，并保持现有投研系统稳定可演示。

下一步如果要继续靠近完整 ReAct，可以新增：

```text
src/agent/planner.py
src/agent/research_agent.py
```

让 planner 只决定工具开关，例如：

```json
{
  "need_policy_docs": true,
  "need_company_docs": true,
  "need_news": true,
  "need_peer_comparison": true,
  "need_python_analysis": false
}
```

不要一开始让 LLM 无限循环调工具。先做有限状态机，最多 3 轮，失败回退到当前 deterministic workflow。

