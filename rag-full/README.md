# 金融 RAG 与投研 Agent 工程

本目录整理了原有 `RAG_for_report/Fin-RAG-Knowledge-Base` 的完整工程源码、脚本、依赖和中文说明，作为统一作品集中的大模型与 Agent 项目。保留 `src/`、`scripts/`、`app/` 的相对结构，共 **45 个 Python 文件**，不再只是少量代码摘录。

原项目来源：[RAG_for_report / Fin-RAG-Knowledge-Base](https://github.com/Walter-prog88/RAG_for_report/tree/main/Fin-RAG-Knowledge-Base)。原文件路径、Git blob SHA 与本次整理变更见 [来源与整理记录.json](来源与整理记录.json)。

这里是可公开维护的工程源码版，**不是包含数据和运行环境的完整镜像**。原文库、真实行情／财务／持仓数据、评测问题及原始回答、FAISS 索引、模型权重、缓存、凭据和本机设置均未打包。运行前需要自行安装依赖，准备有权使用的数据，并下载或配置需要的模型。

## 已有功能

- **文档处理**：政策、交易所公告、研报和新闻的采集脚本，HTML／PDF 解析，Markdown 标准化、元数据、去重和增量索引。
- **RAG 检索**：LangChain 文档与切块、FAISS 向量索引、可配置 embedding、中文词法检索和候选融合；支持 BGE Cross-Encoder 重排序、时效权重及可选 HyDE 查询改写。
- **投研工作流**：按固定流程调用资料检索、行情、因子、风险、分析师共识及受限 Python 分析工具，组织证据，生成结构化判断和报告，保留工具调用记录。
- **模型接口**：OpenAI 兼容接口、Ollama、可选 Anthropic；支持结构化 function calling、解析失败回退及流式输出。
- **交互与评价**：Streamlit 页面、命令行报告、检索评价脚本、本地模型基准及云端／本地模型对照脚本。

工具调用顺序目前由 **Python 确定性工作流**控制，LLM 负责证据组织与结构化输出；没有将其描述为已经完成的自主 LangGraph／ReAct 循环。原有词法检索也不等同于已经实现 BM25。

## 目录

```text
rag-full/
├── app/                         # Streamlit 界面
├── configs/sources.example.yaml # 公开配置模板
├── docs/                        # 原项目中文说明与历史记录
├── scripts/                     # 采集、入库、索引、报告、评价与基准
├── src/
│   ├── agent/                   # 工作流、工具、证据、结构化输出、trace
│   ├── crawler/                 # 采集器
│   ├── indexer/                 # embedding、FAISS、检索和重排
│   ├── market/                  # 行情／因子／共识／风险数据接口
│   └── parser/                  # HTML、PDF、Markdown 处理
├── .env.example
├── requirements.txt
└── 来源与整理记录.json
```

`docs/原始项目说明.md` 保留旧 README。历史文档中的个人路径已改为占位路径；历史方案、日期和叙述不自动代表当前已经完成的验证，启动步骤与评测口径以本 README 为准。

## 安装与配置

建议使用 Python 3.10 或以上版本。依赖沿用原工程的最低版本约束，另补上 pandas 读取 Parquet 所需的 `pyarrow`；本次整理未安装依赖，也未验证所有最新依赖组合。

```bash
cd rag-full
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp configs/sources.example.yaml configs/sources.yaml
cp .env.example .env
```

模板默认配置为本机 Ollama 的 `qwen2.5:3b`；需要先自行安装并启动 Ollama、准备对应模型。该后端不需要 API 密钥。如改用云端，在本机 `.env` 中填写相应环境变量，再修改 `configs/sources.yaml` 的 `llm` 配置。不要将真实 `.env` 或本机 `sources.yaml` 提交到仓库。

所有采集源在模板中默认关闭，股票列表和本地笔记路径为空。根据自己的数据权限与任务需求修改后再显式运行采集脚本。模板中的公开政策源 URL 是接口示例，不保证外部网站始终兼容。

## 准备数据

### 文档与索引

把自己有权使用的原始文件放到 `data/raw/`，或在配置中指定本地 Markdown 来源，然后运行解析与索引脚本。文档正文和必要元数据会写入 `data/processed/markdown/`，FAISS 索引写入 `data/index/faiss/`。

默认 embedding 为 `BAAI/bge-small-zh-v1.5`；重排可使用 `BAAI/bge-reranker-base`。首次使用可能下载模型。Hashing fallback 只适合验证数据管线，不能等同于语义 embedding 的检索质量。HyDE 会调用配置的 LLM。

```bash
python scripts/convert_to_markdown.py --config configs/sources.yaml
python scripts/build_index.py --config configs/sources.yaml
python scripts/test_retrieval.py --query "政策对行业需求有什么影响" --top-k 5
```

### 行情与因子

完整投研报告还需要自行准备本地行情／因子面板和股票基本信息。原代码读取的主要文件为：

- `panel_with_factors.parquet`
- `stock_basic.parquet`
- `factor_ic_summary.csv`（可选）
- `data/facts/stock_consensus.parquet`（分析师共识，可由提取脚本生成）

表结构与字段要求以 `src/market/` 的读取函数为准。整理版新增 `RAG_MARKET_DATA_DIR` 环境变量指定数据目录；未设置时保留原工程相邻 `tushare_quant/quant_data_6y` 路径作为兼容默认值。单独运行市场相关脚本时，建议在 shell 明确设置：

```bash
export RAG_MARKET_DATA_DIR="$PWD/data/market"
```

只准备文档可以进行文档检索；缺少行情和因子数据时，不保证完整投研报告能执行。这里没有用演示数据冒充真实金融研究结果。

## 生成报告与启动界面

完成文档索引、市场数据和模型配置后：

```bash
python scripts/run_research.py \
  --stock-code 300308 \
  --theme "行业研究" \
  --question "请整理已有证据、风险与需要继续核查的问题" \
  --no-llm

streamlit run app/streamlit_app.py
```

`--no-llm` 跳过报告的 LLM 综合步骤，但仍需要相应本地数据；它不是无数据运行开关。去掉该参数会调用所配置的模型。交互页面及采集／评价脚本也可能按操作触发模型、下载或网络请求。本次整理没有执行这些流程。

## 原始评测口径与可写结论

以下数字来自原仓库中已经保存的文件与结果，本次只读检查，**没有重跑**：

| 记录 | 实际含义 |
|---|---|
| 3,031 个 processed Markdown 文件 | 仓库中的文件数量；未证明去重后文档数量、原始网页数量或当前全部可用 |
| 60 条检索问题 | 根据源文档片段由模型合成的问题；不是独立人工标注集，也不是60条回答正确率 |
| Recall@k、MRR、NDCG 脚本 | 检索命中按来源文件名计算；来源命中不等于答案正确。未找到支持“Recall 从72%提升至89%”的已存结果 |
| 8 条金融问答模型基准 | 记录 Qwen2.5 3B／7B 的速度、内存及云端模型评分，样本很小，不是通用模型评测 |
| 10 条云端／本地 RAG 对照 | 使用同一检索上下文，替换 LLM 综合步骤；不是多轮 Agent 任务完成率 |

8 问基准记录的 3B／7B 速度为 **13.42／6.24 tok/s**，平均延迟 **25.6／51.2 秒**，质量分 **4.50／4.88（满分5）**。质量由云端模型评分，汇总文件未保存逐问原始输出，不能据此推断普遍质量优劣。

10 问对照记录的平均延迟为云端 **7.27秒**、本地 7B **138.882秒**。其中一次云端质量评分失败；旧文档使用云端9条有效评分与本地10条评分，分母不一致。按两侧均有效的9条配对计算，均分为 **4.33／3.89**。这些都是 LLM-as-judge 分数，不能当作人工正确率或稳定任务完成率。

没有根据这些资料宣称已完成“120题76%任务完成率”、Qwen 领域 LoRA 训练或 vLLM 部署。若重新评价，应自行生成或人工标注新的评测集，冻结配置并保存逐问结果。

评价脚本需要自备评测文件和索引，例如：

```bash
python scripts/build_eval_set.py --n-chunks 60 --out data/eval/eval_set.jsonl
python scripts/run_eval.py --help
python scripts/ollama_benchmark.py --help
python scripts/rag_cloud_vs_local.py --help
```

合成问题和模型评分会产生相应模型调用，不会因为脚本存在就自动视为已完成评价。

## 本次整理范围

- 保留原工程全部 `src/`、`scripts/`、`app/` 源码及中文技术文档；未迁入原仓库其他目录的零散交易／Agent 脚本。
- 将原 README 移到历史文档；将真实配置改成公开模板，清空研究股票池和本地笔记路径，默认关闭采集。
- 清理个人绝对路径；新增市场数据目录环境变量，补充 Parquet 读取依赖和忽略规则。
- 不包含原文库、行情／财务／持仓数据、评测原文、权重、索引、缓存、凭据和本机配置。
- 仅做 Python AST、配置解析、文件完整性与敏感标记静态检查；未运行采集、回测、推理或评测。
- 未自动添加 MIT 或其他新许可。原代码与依赖来源保持可追溯，第三方库遵循各自许可。

