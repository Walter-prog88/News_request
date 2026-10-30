# LangChain 使用说明

更新时间：2026-05-19

## 当前是否使用了 LangChain

已经使用。

但当前 LangChain 主要用在第一层 RAG 基础设施，不是用来做自主 Agent 调度。

## 当前用在什么地方

### 1. 文档对象封装

文件：`src/indexer/build_vectorstore.py`

使用：

```python
from langchain_core.documents import Document
```

作用：

- 把标准 Markdown 文档包装成 LangChain `Document`
- 每个 `Document` 同时保留正文和 metadata
- 后续切 chunk、embedding、FAISS 入库都基于这个统一对象

### 2. Chunk 切分

文件：`src/indexer/build_vectorstore.py`

使用：

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter
```

作用：

- 把长政策、年报、公告切成适合 embedding 的片段
- 使用 `chunk_size` 和 `chunk_overlap` 控制片段大小和上下文重叠
- 避免整篇年报直接入向量库导致检索不准

当前配置：

```yaml
chunking:
  chunk_size: 900
  chunk_overlap: 150
```

### 3. FAISS 向量库

文件：

- `src/indexer/build_vectorstore.py`
- `src/indexer/retriever.py`

使用：

```python
from langchain_community.vectorstores import FAISS
```

作用：

- 把 chunk embedding 后写入本地 FAISS 索引
- 保存到 `data/index/faiss/`
- 查询时加载 FAISS 并返回 top-k 相似片段

输出文件：

```text
data/index/faiss/index.faiss
data/index/faiss/index.pkl
```

### 4. Embedding 接口

文件：`src/indexer/build_vectorstore.py`

当前支持两种模式：

```yaml
embedding:
  provider: hashing
  model_name: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

当前实际使用的是 `HashingEmbeddings` fallback。

原因：

- 本地轻量、无需下载大模型
- 适合验证端到端管线
- 可复现、启动快

正式版本建议改为：

```yaml
embedding:
  provider: huggingface
  model_name: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

或者替换成中文金融 embedding 服务。

## 当前为什么没有用 LangChain Agent

当前第三层 Agent 采用的是确定性 workflow：

文件：`src/agent/research_workflow.py`

它按固定流程调用：

```text
search_policy_docs()
search_company_docs()
get_market_data()
calculate_factor_score()
risk_check()
synthesize_evidence()
generate_report()
```

这样做的原因：

- 当前项目重点是把数据层和证据链打稳
- 固定 workflow 更容易调试和复现
- 可以避免早期 Agent 自主调用导致证据漂移或结论不可控
- 金融投研场景里，可追溯性比“看起来智能”更重要

## 后续如何扩展成真正 Agent

下一步可以把现有函数包装成 LangChain Tool 或 LangGraph 节点：

```text
RAG 检索工具：
- search_policy_docs
- search_company_docs

结构化数据工具：
- get_market_data
- calculate_factor_score
- risk_check

报告工具：
- synthesize_evidence
- generate_report
```

推荐路线：

```text
当前 deterministic workflow
-> LangChain Tools
-> LangGraph 状态机
-> LLM 负责规划、改写 query、生成最终报告
```

不建议一开始就让 LLM 自由调用工具。更稳的方式是先用 LangGraph 定义清楚状态、工具边界和输出 schema。

## 一句话解释

LangChain 现在是 RAG 的工程底座：负责文档对象、chunk 切分、embedding 接口和 FAISS 向量库；Agent 调度层目前先用可控的 Python workflow 实现，后续再升级为 LangChain Tools 或 LangGraph。

