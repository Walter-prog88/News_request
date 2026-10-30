# FinResearch-Agent 项目状态与修改历史

更新时间：2026-05-19

## 1. 项目最终目标

目标是构建一个基于 RAG + Agent + 金融数据分析的 AI 投研助手，能够围绕某个主题或个股自动检索证据、计算结构化金融信号，并生成可追溯的投研结论。

典型输入：

```text
股票代码：300308
主题：AI 光模块
问题：中际旭创现在还值得关注吗？AI 光模块主题是否还能持续？
```

目标输出：

```text
1. 公司基本情况
2. 最近行情表现
3. 光模块主题逻辑
4. 政策和产业支持证据
5. 公司公告 / 年报证据
6. 因子信号：动量、成交量、波动率、回撤
7. 风险提示
8. 最终结论：Buy / Watch / Avoid
9. 证据来源
```

最终系统分为 4 层：

```text
FinResearch-Agent
├── 第一层：RAG 知识库
├── 第二层：金融数据分析模块
├── 第三层：Agent 调度层
└── 第四层：报告生成和展示层
```

## 2. 当前已经完成的内容

### 2.0 最新验证状态

截至 2026-05-19，项目已经从“RAG 骨架”推进到一个可演示的最小闭环：

```text
RAG 原始数据采集
-> Markdown 标准化
-> FAISS 向量库
-> 混合检索
-> 本地行情/因子/风险计算
-> 规则化证据综合
-> Markdown 投研报告
-> Streamlit Demo
```

已完成验证：

- 代码编译：`python -m compileall -q src scripts app`
- 检索 smoke test：`python scripts/evaluate_retrieval.py`，4/4 通过
- 示例报告生成：`python scripts/run_research.py --stock-code 300308 --theme "AI 光模块" ...`
- 示例报告保存：`reports/research_300308_SZ_AI_光模块.md`
- Streamlit 页面：`app/streamlit_app.py`

### 2.1 第一层：RAG 知识库骨架已经完成

当前目录：`Fin-RAG-Knowledge-Base/`

已完成能力：

- 政府政策网页/PDF 小批量采集框架
- 国家发改委网页/PDF 小批量采集框架
- 上交所/深交所上市公司公告 API + PDF 采集框架
- 本地 Markdown 投研笔记导入配置
- 原始文件保存到 `data/raw/`
- 原始文件 `.meta.json` metadata 侧车文件
- HTML 文本抽取
- PDF 文本抽取：优先 `pdfplumber`，失败后用 `PyMuPDF`
- Markdown 标准化
- YAML front matter metadata
- hash 去重
- chunk 切分
- embedding
- FAISS 向量库
- 检索测试脚本
- README 中已写明每个模块的作用、输入、输出、运行方式、失败原因和扩展方向

核心文件：

```text
requirements.txt
configs/sources.yaml
src/crawler/base_crawler.py
src/crawler/gov_policy_crawler.py
src/crawler/ndrc_crawler.py
src/crawler/exchange_announcement_crawler.py
src/parser/html_parser.py
src/parser/pdf_parser.py
src/parser/markdown_cleaner.py
src/indexer/build_vectorstore.py
src/indexer/retriever.py
scripts/crawl_policies.py
scripts/convert_to_markdown.py
scripts/build_index.py
scripts/test_retrieval.py
README.md
```

验证状态：

- 已执行 Python 静态编译检查：`python -m compileall -q src scripts app`
- 已完成 NDRC 政策网页采集和 Markdown 转换
- 已完成深交所 `300308` 公告 PDF 采集、解析和 Markdown 转换
- 已构建 FAISS 向量库：`data/index/faiss/`
- 当前 processed Markdown 数量：政策、公告和本地笔记共 14 篇
- 当前向量库 chunk 数量：499
- 已增加确定性词法检索层，优先保证公司名、股票代码和公告证据命中
- 已增加检索 smoke test：`scripts/evaluate_retrieval.py`

LangChain 使用状态：

- 已用于 `Document` 封装、chunk 切分、FAISS 向量库构建和向量检索
- 当前没有使用 LangChain Agent；Agent 调度层先用确定性 Python workflow 实现
- 详细说明见 `docs/LANGCHAIN_USAGE.md`

### 2.2 第二层：金融数据分析模块已有重要基础

当前目录：`tushare_quant/`

已完成能力：

- 6 年 A 股/沪深300相关数据集
- 对齐面板：`quant_data_6y/aligned_panel.parquet`
- 22 个候选因子：`quant_data_6y/panel_with_factors.parquet`
- 单因子 IC / RankIC / ICIR 检验：`quant_data_6y/factor_ic_summary.csv`
- 每日 IC 时间序列：`quant_data_6y/all_ic_series.pkl`

关键数据结果：

```text
aligned_panel.parquet:
shape: (668841, 25)
日期范围: 2020-01-02 ~ 2026-05-15
股票数: 481
交易日数: 1540
沪深300成分股样本数: 461412
```

已构造的因子类型：

- 价值类：EP、BP、SP、log_mv
- 质量类：ROE、毛利率、负债率、EPS
- 动量/反转类：5日反转、20日反转、60日动量、120日动量、20日波动率
- 量价/流动性类：20日换手、量比、20日资金流、北向资金
- 预期类：评级、EPS yield、20/60日盈利预测修正、预测EPS/实际EPS

当前最有价值的研究发现：

```text
fac_eps_revision_60d:
RankIC = +0.0232
RankICIR = +0.2249
方向 = 正向
```

含义：过去 60 个交易日分析师预测 EPS 上调的公司，未来 5 日收益横截面排序更可能靠前。它可以作为项目差异化亮点，因为它利用的是盈利预期变化，而不只是价格历史。

### 2.3 已经做过的关键工程修正

目标变量修正：

```python
panel["y_future_5d"] = panel.groupby("ts_code")["close"].transform(
    lambda x: x.shift(-5) / x - 1
)
```

意义：

- 在每只股票内部计算未来 5 日收益
- 避免股票边界串行
- 口径是用 T 日信号预测 T 到 T+5 的收盘收益

因子变化率修正：

```python
pct_change(..., fill_method=None)
```

意义：

- 显式关闭 Pandas 默认前向填充
- 避免 EPS 预测缺失值被隐式填充后产生虚假的 revision 信号

IC 检验修正：

- 屏蔽常数/近常数截面的 SciPy warning
- 保留样本数 `< 30` 跳过
- 保留 Pearson IC 和 Spearman RankIC

## 3. 当前还差什么

### 3.1 第一层 RAG 知识库还差

优先级 P0：

- 安装依赖并真实跑通完整管线

```bash
cd Fin-RAG-Knowledge-Base
pip install -r requirements.txt
python scripts/crawl_policies.py
python scripts/convert_to_markdown.py
python scripts/build_index.py
python scripts/test_retrieval.py --query "AI 算力 光模块 政策 支持"
```

- 验证真实抓取是否成功
- 验证 PDF 能否抽出有效文本
- 验证 Markdown front matter 是否完整
- 验证 FAISS 检索结果是否和 query 相关

优先级 P1：

- 增加年报/季报/投资者关系记录的数据源
- 增加巨潮资讯或交易所公告更稳定的公告下载入口
- 为 gov.cn、NDRC、交易所公告写更细的正文抽取规则
- 给每篇文档增加更标准的 metadata 字段：

```text
source
doc_type
company_name
stock_code
industry
theme
published_at
url
raw_path
content_hash
```

优先级 P2：

- 增加近重复去重：simhash/minhash
- 增加 OCR，用于扫描版 PDF
- 增加增量抓取和断点续跑
- 增加引用高亮和页码级溯源

### 3.2 第二层金融数据分析模块还差

当前已有因子研究基础，但还不是可被 Agent 调用的工具。

优先级 P0：

- 把 `tushare_quant` 中的行情和因子逻辑封装成可调用函数

目标函数：

```python
get_market_data(stock_code, start_date, end_date)
calculate_technical_signals(stock_code)
calculate_factor_score(stock_code)
risk_check(stock_code)
```

需要输出：

```text
近5/20/60日涨跌幅
成交量变化
均线状态
20日波动率
最大回撤
估值 PE/PB
ROE/EPS
因子分位数
风险标签
```

当前进展：

- `src/market/data_loader.py` 已封装本地 parquet 股票信息和面板读取
- `src/market/technical.py` 已支持收益率、均线、成交量比、波动率、最大回撤和价格历史
- `src/market/factor_signal.py` 已支持基于 IC 结果的因子分位打分
- `src/market/risk.py` 已支持规则化风险标签
- `src/agent/tools.py` 已把 RAG 检索、行情、因子和风险函数封装为工具层函数

优先级 P1：

- 对进入模型的因子做 winsorize 去极值
- 做行业/市值中性化
- 重新计算中性化后的 IC / RankIC
- 建立个股打分表
- 增加简单回测：top quantile vs bottom quantile

优先级 P2：

- 接入 AKShare 实时行情
- 接入本地 CSV/Parquet 缓存
- 支持主题股票池，如 AI 光模块、半导体、机器人

### 3.3 第三层 Agent 调度层还差

目前还没有真正的 Agent，只完成了 RAG 和金融数据的底层材料。

需要新增目录：

```text
src/agent/
├── tools.py
├── planner.py
├── prompts.py
└── research_agent.py
```

目标工具：

```python
search_policy_docs(query)
search_company_docs(stock_code, query)
search_industry_docs(theme)
get_market_data(stock_code)
calculate_factor_score(stock_code)
risk_check(stock_code)
generate_report(context)
```

Agent 需要完成的判断：

- 用户问的是主题、个股，还是主题 + 个股
- 是否需要检索政策
- 是否需要检索公告/年报
- 是否需要计算行情和风险
- 最终结论应该是 Buy、Watch 还是 Avoid

优先级 P0：

- 先不做复杂多轮 Agent
- 先做 deterministic workflow：

```text
输入 stock_code + theme + question
-> RAG 检索
-> 行情/因子工具
-> 风险检查
-> 报告模板生成
```

优先级 P1：

- 用 LangChain tools 封装各模块
- 增加 LLM 总结层
- 增加引用来源强制输出

### 3.4 第四层报告生成和展示层还差

当前没有前端。

建议新增：

```text
app/streamlit_app.py
```

页面输入：

```text
股票代码
主题关键词
问题
风险偏好
```

页面输出：

```text
结论：Buy / Watch / Avoid
行情摘要
RAG 检索证据
因子信号
风险提示
结构化投研报告
引用来源
```

优先级 P0：

- 做一个能演示的 Streamlit 页面
- 先用固定报告模板，不追求复杂交互

优先级 P1：

- 增加证据卡片
- 增加行情图
- 增加因子雷达图或分位数条形图
- 增加一键导出 Markdown/PDF

## 4. 建议下一步执行顺序

### Step 1：先把 RAG 管线真实跑通

目标：确认第一层不是空架子。

命令：

```bash
cd Fin-RAG-Knowledge-Base
pip install -r requirements.txt
python scripts/crawl_policies.py --sources gov_policy ndrc
python scripts/convert_to_markdown.py
python scripts/build_index.py
python scripts/test_retrieval.py --query "AI 算力 光模块 政策 支持" --top-k 5
```

验收标准：

```text
data/raw/ 有 HTML/PDF 原始文件
data/processed/markdown/ 有标准 Markdown
data/index/faiss/ 有 FAISS 索引
检索结果能返回相关政策/行业资料片段
```

### Step 2：新增金融数据工具层

目标：把 `tushare_quant` 的数据能力变成 Agent 可调用工具。

建议新增：

```text
src/market/
├── data_loader.py
├── technical.py
├── factor_signal.py
└── risk.py
```

验收标准：

```python
calculate_factor_score("300308.SZ")
```

能返回：

```text
动量信号
成交量信号
波动率信号
回撤风险
估值水平
综合评分
```

### Step 3：做 deterministic research workflow

目标：先不用复杂 Agent，先跑通完整报告链路。

建议新增：

```text
src/agent/research_workflow.py
```

输入：

```python
stock_code = "300308.SZ"
theme = "AI 光模块"
question = "这个主题是否还能持续？"
```

输出：

```text
结构化投研报告 Markdown
```

### Step 4：做 Streamlit Demo

目标：让项目可展示、可面试演示。

建议新增：

```text
app/streamlit_app.py
```

验收标准：

```bash
streamlit run app/streamlit_app.py
```

页面能输入股票代码和主题，输出完整报告。

## 5. 修改历史

### 2026-05-16：Stage 1B，6 年面板重新对齐

新增/修改：

- 复制 `tushare_quant/stage1_align_panel.py`
- 生成 `tushare_quant/stage1_align_panel_6y.py`
- 运行 6 年数据对齐
- 新增 `tushare_quant/validate_stage1b.py`

结果：

```text
aligned_panel.parquet:
shape: (668841, 25)
日期范围: 2020-01-02 ~ 2026-05-15
股票数: 481
交易日数: 1540
沪深300成分股样本数: 461412
```

重要结论：

- 数据质量足够支撑专业项目
- `eps_forecast` 在沪深300样本可用率约 98.1%
- `target_price` 缺失率高，后续不作为核心字段

### 2026-05-16：Stage 2 Step 1，构造 22 个候选因子

新增：

- `tushare_quant/stage2_step1_build_factors.py`
- 输出 `tushare_quant/quant_data_6y/panel_with_factors.parquet`

构造因子：

```text
fac_ep
fac_bp
fac_sp
fac_log_mv
fac_roe
fac_gross_margin
fac_debt
fac_eps
fac_rev_5d
fac_rev_20d
fac_mom_60d
fac_mom_120d
fac_vol_20d
fac_turn_20d
fac_volume_ratio
fac_mf_20d
fac_hsgt_5d
fac_rating
fac_eps_yield
fac_eps_revision_60d
fac_eps_revision_20d
fac_eps_growth_expect
```

关键修正：

- `y_future_5d` 用 `groupby("ts_code").transform(lambda x: x.shift(-5) / x - 1)`，避免跨股票错位
- 所有 `pct_change` 显式使用 `fill_method=None`，避免隐式前向填充

### 2026-05-16：Stage 2 Step 2，单因子 IC / RankIC 检验

新增：

- `tushare_quant/stage2_step2_factor_ic.py`
- 输出 `factor_ic_summary.csv`
- 输出 `all_ic_series.pkl`

核心结果：

```text
fac_eps_revision_60d:
RankIC = +0.0232
RankICIR = +0.2249
方向 = 正向
```

重要认知：

- RankIC 比 Pearson IC 更稳健，因为它基于排序，不容易被极端值拖偏
- 后续进入模型前需要补 winsorize
- 后续需要做行业/市值中性化，否则 IC 可能被市值暴露污染
- `fac_hsgt_5d` 更适合作为市场层面变量，不适合直接做横截面选股因子

### 2026-05-19：RAG 知识库项目骨架搭建

新增项目：

```text
Fin-RAG-Knowledge-Base/
```

新增核心文件：

```text
requirements.txt
configs/sources.yaml
src/crawler/base_crawler.py
src/crawler/gov_policy_crawler.py
src/crawler/ndrc_crawler.py
src/crawler/exchange_announcement_crawler.py
src/parser/html_parser.py
src/parser/pdf_parser.py
src/parser/markdown_cleaner.py
src/indexer/build_vectorstore.py
src/indexer/retriever.py
scripts/crawl_policies.py
scripts/convert_to_markdown.py
scripts/build_index.py
scripts/test_retrieval.py
README.md
```

实现能力：

- 小批量公开网页/PDF 采集
- 所有请求支持 `delay_seconds`
- 原始文件保存到 `data/raw/`
- HTML/PDF 文本抽取
- Markdown 标准化
- YAML front matter metadata
- hash 去重
- LangChain chunk 切分
- embedding
- FAISS 向量库
- 检索测试

验证：

```bash
python3 -m compileall -q src scripts
```

结果：语法检查通过。

尚未完成：

- 尚未安装完整 requirements
- 尚未真实跑通采集、解析、建库、检索

### 2026-05-19：最小投研闭环与 Streamlit Demo

新增目录：

```text
src/market/
src/agent/
app/
reports/
```

新增核心文件：

```text
src/market/data_loader.py
src/market/technical.py
src/market/factor_signal.py
src/market/risk.py
src/agent/tools.py
src/agent/research_workflow.py
scripts/run_research.py
app/streamlit_app.py
reports/research_300308_SZ_AI_光模块.md
```

实现能力：

- 复用 `../tushare_quant/quant_data_6y/panel_with_factors.parquet`
- 支持股票代码标准化，如 `300308 -> 300308.SZ`
- 读取公司基础信息
- 计算近5/20/60日收益
- 计算成交量比、20/60日均线状态
- 计算20日年化波动率
- 计算60/120日最大回撤
- 计算核心因子截面分位和综合因子得分
- 生成风险标签
- 调用 RAG 检索政策/产业证据和公司证据
- 如 FAISS 索引不存在，退回本地 Markdown 关键词检索
- 生成结构化 Markdown 投研报告
- 输出 Buy / Watch / Avoid 规则结论
- 提供 Streamlit 页面演示

示例命令：

```bash
python scripts/run_research.py \
  --stock-code 300308 \
  --theme "AI 光模块" \
  --question "中际旭创现在还值得关注吗？AI 光模块主题是否还能持续？"
```

示例输出：

```text
公司：中际旭创（300308.SZ）
主题：AI 光模块
结论：Watch
理由：主题和趋势仍有关注价值，但短期风险收益比需要更多证据确认，适合跟踪或分批关注。
近20日收益：35.98%
近60日收益：94.65%
综合因子得分：47.64%
风险等级：medium
```

生成报告：

```text
reports/research_300308_SZ_AI_光模块.md
```

Streamlit Demo：

```bash
streamlit run app/streamlit_app.py
```

本地验证：

```text
Streamlit 已启动
http://localhost:8501 返回 HTTP 200
```

当前限制：

- RAG 已有第一批真实证据，但政策/行业语义检索质量仍受 hashing embedding 限制
- 结论规则是 deterministic rule，不是完整 LLM Agent
- 行情数据来自本地 parquet，不是实时行情
- 因子没有做 winsorize、行业中性化、市值中性化
- Streamlit 已有行情图、成交量图、证据卡片和因子表，但还没有更精细的交互图表

### 2026-05-19：RAG 证据层真实跑通

新增/修改：

- 修复 `src/crawler/ndrc_crawler.py` 中 NDRC 页面编码和详情页正则匹配问题
- 修复 `src/crawler/gov_policy_crawler.py` 中列表页 UTF-8 解码
- 在 `configs/sources.yaml` 的深交所股票池中加入 `300308`
- 修复 `src/indexer/build_vectorstore.py` 中 fallback embedding 与 LangChain FAISS 的兼容问题
- 优化 `src/agent/tools.py`：
  - RAG 检索合并关键词检索和 FAISS 检索
  - 对 `AI 光模块` 做中文扩展词：人工智能、算力、数据中心、集成电路、通信设备等
  - 公司证据必须包含公司名或股票代码，避免误引无关材料
  - 政策证据过滤掉公告类 source，避免政策区和公司公告区混杂

真实执行：

```bash
.venv/bin/python scripts/crawl_policies.py --sources ndrc
.venv/bin/python scripts/crawl_policies.py --sources exchange_announcements
.venv/bin/python scripts/convert_to_markdown.py
.venv/bin/python scripts/build_index.py
.venv/bin/python scripts/test_retrieval.py --query "中际旭创 年度报告 AI 光模块" --top-k 2
.venv/bin/python scripts/run_research.py --stock-code 300308 --theme "AI 光模块" --question "中际旭创现在还值得关注吗？AI 光模块主题是否还能持续？"
```

结果：

```text
NDRC 抓取：5 篇政策详情文档
交易所公告抓取：10 份 PDF
深交所中际旭创公告解析：5 份 Markdown
Markdown 文档数：14 篇
FAISS chunks：499 个
检索测试：可以命中中际旭创一季报、三季报、年报等公告
最终报告：政策/产业证据 3 条，公司公告证据 5 条
```

最终报告文件：

```text
reports/research_300308_SZ_AI_光模块.md
```

当前报告核心输出：

```text
结论：Watch
公司：中际旭创（300308.SZ）
主题：AI 光模块
政策/产业证据数量：3
公司公告证据数量：5
近20日收益：35.98%
近60日收益：94.65%
综合因子得分：47.64%
风险等级：medium
```

已知问题：

- 上交所抓到的若干 PDF 响应不是有效 PDF，解析器已跳过
- 当前 embedding 使用 `HashingEmbeddings`，检索可用但不是高质量语义检索
- 尚未安装 `sentence-transformers`，后续可切回 HuggingFace embedding
- 中际旭创公告主要来自深交所年报、季报、半年报，投资者关系记录还未接入

### 2026-05-19：Streamlit 展示层增强

新增/修改：

- `src/market/technical.py` 增加 `get_price_history()`，为前端提供收盘价、20日均线、60日均线、成交量和单日收益
- `src/agent/research_workflow.py` 增加 `collect_research_payload()`，一次性返回结构化 payload 和 Markdown 报告
- `app/streamlit_app.py` 从单页 Markdown 展示升级为多标签页面

页面能力：

```text
行情概览：价格走势、20/60日均线、成交量图、核心行情指标
RAG 证据：政策/产业证据卡片、公司公告/年报证据卡片
因子与风险：综合因子得分、因子明细表、风险标签、最大回撤
Markdown 报告：完整报告展示和下载按钮
```

验证：

```bash
.venv/bin/python -m compileall -q src scripts app
.venv/bin/python - <<'PY'
from src.agent.research_workflow import collect_research_payload
payload = collect_research_payload("300308", "AI 光模块", "中际旭创现在还值得关注吗？AI 光模块主题是否还能持续？", top_k=3)
print(payload["conclusion"])
print(len(payload["policy_docs"]), len(payload["company_docs"]), len(payload["report"]))
PY
curl -I http://localhost:8501
```

结果：

```text
结论：Watch
policy_docs: 2
company_docs: 3
Streamlit HTTP 200
```

### 2026-05-19：Lexical Retriever 与检索评估

新增/修改：

- 新增 `src/indexer/lexical_retriever.py`
- 新增 `scripts/evaluate_retrieval.py`
- `src/agent/tools.py` 改为优先使用中文友好的 lexical retrieval，再用 FAISS 补充结果
- 报告和 Streamlit 证据卡片增加匹配词、retriever 和 score

检索增强能力：

```text
标题/正文/URL 加权打分
中文关键词扩展
source include/exclude 过滤
公司证据 required_terms 过滤
相关段落 snippet 抽取
匹配词输出
```

验证：

```bash
.venv/bin/python scripts/evaluate_retrieval.py --top-k 5
```

结果：

```text
company_annual_report: PASS
company_optical_module: PASS
policy_ic_software: PASS
generic_rag_query: PASS
Passed 4/4
```

## 6. 当前项目短板总结

最重要的短板已经从“没有闭环”变成“证据层还没有真实填充、Agent 还不够智能”。

当前状态：

```text
RAG 知识库：骨架完成，待真实跑通和扩源
金融数据分析：已完成第一版工具化
Agent 调度层：已完成 deterministic workflow 最小闭环
报告/前端展示：已完成 Streamlit Demo 第一版
```

下一阶段最有价值的目标：

```text
把 RAG 证据层真实跑通：

采集政策/公告 -> 转 Markdown -> 建 FAISS -> 让报告中出现真实引用来源
```

最小闭环已经可运行，下一步要提高“证据质量”和“报告可信度”。

## 7. 简历表达草稿

项目名称：金融 AI 投研 Agent 系统

项目描述：

构建面向 A 股主题投资的 AI 投研助手，整合政策文件、上市公司公告、行业资料、研报摘要和股票行情数据，实现从 RAG 知识库构建、因子信号计算、Agent 工具调用到结构化投研报告生成的完整流程。

技术栈：

```text
Python, Pandas, Requests, BeautifulSoup, pdfplumber, PyMuPDF,
LangChain, FAISS, sentence-transformers, Tushare, Streamlit
```

可量化亮点：

```text
构建 2020-2026 年 A 股/沪深300面板数据，覆盖 66.8 万行、481 只股票、1540 个交易日；
构造 22 个候选因子并完成 IC/RankIC 检验；
发现盈利预测上修因子 fac_eps_revision_60d 具备正向预测能力，RankIC=0.0232，RankICIR=0.2249；
搭建政策文件、交易所公告、本地研究笔记的 RAG 知识库管线，支持原始文件归档、Markdown 标准化、hash 去重、FAISS 检索和可追溯引用。
```

