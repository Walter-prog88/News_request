# Fin-RAG-Knowledge-Base 优化日志

> 这个文件记录系统架构层面的优化决策和背后的数据依据。
> 请不要删除。每次重要改动追加一节，而不是覆盖历史。

---

## 2026-05-20 — 索引清理 + 检索质量提升

### 问题诊断（数据依据）

运行前对索引结构做了定量分析，发现以下问题：

| 来源 | 文件数 | Chunks | 均值 chunks/文件 | 最大 |
|------|--------|--------|----------------|------|
| `szse_announcement` | 385 | 67,667 | **175.8** | 1,436 |
| `research_report`   | 2,218 | 2,218 | 1.0 | — |
| 政策/local_note     | 19 | 58 | 3 | — |

研报占总 chunks 的 3.2%，公告占 96.7%。FAISS 检索时公告会大量"稀释"研报。

**英文年报是主要根因**：最重的13个文件全是英文版年报，单文件最多 1,436 chunks：
- 海康威视 2025 年报（英文版）：1,436 chunks
- 云南白药 2025 年报（英文版）：1,362 chunks
- 恩捷股份 2025 年报（英文版）：1,348 chunks
- 合计 13 个文件：**6,826 chunks**（占总量 9.8%）

这些内容与中文版完全重复，对中文投研系统没有增量价值。

---

### 优化 1：删除英文年报的 FAISS 条目

**改动位置**：
- `configs/sources.yaml` → `chunking.exclude_title_patterns: ["英文版"]`
- `src/indexer/build_vectorstore.py` → `load_markdown_documents()` 接受 config，按 title 模式过滤
- `scripts/update_index_incremental.py` → `_current_documents()` 接受 config 并透传

**机制**：
`load_markdown_documents` 在加载时跳过 title 含"英文版"的文件 → 这些文件不进入 `current_docs` → 增量更新的 `delete_missing` 逻辑认为它们已"消失" → 自动删除对应 FAISS entries。**源 Markdown 文件完整保留**，只清理向量数据库。

**效果（2026-05-20 实测）**：
```
执行前：69,943 chunks（含 6,826 英文年报 chunks）
执行后：63,117 chunks
删除：13 个文件，6,826 chunks
误删：0（通过 dry-run 验证精确匹配）
```

**未来防范**：新爬取的英文年报不会进入索引，config 里的 `exclude_title_patterns` 自动拦截。

---

### 优化 2：单文档 chunk 数量上限

**改动位置**：
- `configs/sources.yaml` → `chunking.max_chunks_per_doc: 80`
- `src/indexer/build_vectorstore.py` → `split_documents()` 增加 cap 逻辑

**机制**：
单个文件切分后若超过 80 个 chunks，则保留前 25%（摘要/亮点部分）并从剩余部分等距采样，确保文档全局覆盖而不是只保留开头。

**为什么是 80？**  
当前最大的中文年报约 400 pages，切成 ~400 chunks。80 已足以覆盖"业务概况、财务摘要、研发投入、客户结构、风险"等关键章节，且不会让单篇文档占据 FAISS 的显著比例。此参数可在 config 中调整。

**注意**：此限制只对**下次重建索引**生效（通过 `build_index.py` 或全量增量更新）。已入库的超大文件需要文件内容变更触发重索引才会被 cap。

---

### 优化 3：元数据增强（为下次重索引生效）

**改动位置**：
- `src/indexer/build_vectorstore.py` → 新增 `_normalize_ts_code()`, `_derive_doc_type()`, `_enrich_doc_metadata()`
- `load_markdown_documents()` 在每个文档加载后调用 `_enrich_doc_metadata()`

**新增字段**：

| 字段 | 来源 | 说明 |
|------|------|------|
| `ts_code` | `stock_code` + `exchange` 推导 | 统一格式 `XXXXXX.SZ/SH`，用于公司文档过滤 |
| `doc_type` | `source` + `title` 推导 | `annual_report / quarterly_report / research_report / policy / announcement` |

`ts_code` 推导规则：
- `stock_code: ['002415']` + `exchange: SZSE` → `002415.SZ`
- 前缀 `000/001/002/003/300/301/302` → `.SZ`；`600/601/603/605/688` → `.SH`

**注意**：现有 FAISS 索引中的 chunks 不含 `doc_type` 字段（因为它们在本次改动前已入库）。新入库的 chunks 会自动获得这两个字段。

---

### 优化 4：时间权重检索重排序

**改动位置**：
- `src/indexer/retriever.py` → `_parse_doc_date()`, `_recency_factor()`, `_apply_time_weights()`, `retrieve()` 新增 `time_weighted=True`

**机制**：
FAISS 返回 L2 距离（越小越相似）。在返回结果前，将语义相似度和时效性混合：
```
sim     = 1 / (1 + raw_l2_dist)     ∈ (0, 1]
blended = sim × (0.6 + 0.4 × recency)
```

`recency` 衰减参数：
- 研报 / 公告：半衰期 180 天（6个月）
- 政策文件（ndrc/csrc/miit/gov_policy/local_note）：半衰期 730 天（2年）

日期字段依次尝试：`published_at` → `report_date` → `fetched_at`。无日期的文档获得中性权重 0.75。

**为什么 60/40 混合而不是纯时间排序？**  
投研场景中语义相关性比时效性更重要。旧的深度研究报告（行业 / 公司逻辑）比昨天发布的无关公告更有价值。60% 语义 + 40% 时效是在"拉最新"和"找最准"之间的平衡，可在代码中调整。

**FAISS 获取候选量**：开启时效权重时，`retrieve()` 获取 `top_k × 3` 个候选再重排，给时效性调整留出空间。

**实测效果（2026-05-20）**：
```
查询："中际旭创 光模块 800G 业绩"
权重前 Top-1：中际旭创深度研究（2026-04-22）
权重后 Top-1：中际旭创 26Q1 业绩点评（2026-04-29）← 更新、更具操作参考价值
```

---

### 验证命令

```bash
# 1. 确认索引无英文年报
TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 EMBEDDING_DEVICE=mps \
  .venv/bin/python -c "
import json
d = json.load(open('data/index/faiss/index_manifest.json'))
print('Total:', d['total_chunks'])
print('By source:', d['chunks_by_source'])
"

# 2. 时间权重检索验证（应以最新研报优先）
TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 EMBEDDING_DEVICE=mps \
  .venv/bin/python -m src.indexer.retriever

# 3. 增量索引 dry-run（应显示 files_to_add=0, files_to_delete=0）
TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 EMBEDDING_DEVICE=mps \
  .venv/bin/python scripts/update_index_incremental.py --dry-run

# 4. 端到端报告生成
TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 EMBEDDING_DEVICE=mps \
  .venv/bin/python scripts/run_research.py \
    --stock-code 300308 --theme "AI 光模块" \
    --question "中际旭创现在还值得关注吗？"
```

---

## 后续待实现（按优先级）

### P0 — 最大收益，代码量小

**Cross-encoder Reranker**
- 模型：`BAAI/bge-reranker-base`（~100MB，可本地运行）
- 位置：`src/indexer/retriever.py` 的 `retrieve()` 末尾，FAISS 召回后精排
- 为什么：当前 BGE 双编码器向量召回精度约 70%；加入 cross-encoder 精排后可到 90%+，对"年报淹没研报"这类 false positive 问题效果最显著
- 参考实现：`FlagReranker(model).compute_score([(query, passage)])`

**大股东增减持信号**
- 数据：Tushare `stk_holdertrade` 表（已有本地 parquet 数据）
- 位置：`src/market/factor_signal.py`，新增 `insider_buy_sell_ratio` 因子
- 为什么：A 股中机构/大股东增持是少数有统计显著性的先行指标之一

### P1 — 中等收益

**融资融券数据（margin trading）**
- Tushare `margin` 表 → 融资余额快速增加意味着交易拥挤，是反向信号

**事件分类器**
- 从已爬取的公告 Markdown 中提取：业绩预告、重大合同、股权激励
- 用简单正则 + 关键词匹配实现，不需要 LLM

**查询改写（Query Rewriting）**
- 在 `search_company_docs` / `search_policy_docs` 前，把用户问题拆成 3 个子查询分别检索再合并
- 解决"中际旭创值不值得买"这类模糊问题召回不足的问题

### P2 — 长期改进

**幻觉校验层**
- 对 LLM 生成的 Section 4 中每个数字主张，验证是否有对应检索 chunk 支持
- 无支持的标注 `[未验证]`

**供应链图谱**
- 从年报"主要客户/供应商"章节提取关联股票
- 查询"中际旭创"时也能检索客户（Nvidia 相关政策）和上游（光芯片）

**指数重建时的进一步 chunk 质量优化**
- 年报关键章节提取（只保留：管理层讨论、主营业务、客户结构、财务摘要、风险）
- 其余章节（脚注、格式页、重复表格）排除，可把年报平均 chunk 数从 ~175 降到 ~30

---

## 架构说明（写给未来的自己）

```
crawler → HTML/PDF 解析 → Markdown（带 front matter）→ FAISS（BGE-small-zh 512dim）
                                                               ↓
          Tushare 6年面板（parquet） → 技术信号 + 因子打分     ↓
                                                               ↓
                           research_workflow.py ← tools.py（hybrid retrieve）
                                    ↓
                           LLM（SiliconFlow/Qwen2.5-72B）
                                    ↓
                           Markdown 报告 / Streamlit 流式界面
```

**重要约束**：
- 不要全量重建 FAISS，除非明确说明原因（重建耗时 ~2小时 on MPS）
- 新文档入库：`scripts/update_index_incremental.py`（增量，不全量）
- 英文年报已被 `configs/sources.yaml exclude_title_patterns` 永久排除，不需要手动干预
- FAISS 使用 L2 距离（越小越相似），时间权重在检索层做混合排序，不影响索引本身

