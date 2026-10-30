# 教学 RAG 课程问答系统：项目结构草案

当前公开内容是一份**项目结构与实施计划草案**。原仓库仅有占位文件和结构说明，没有应用源码、已构建的知识库或评测结果；因此这里不将其描述为已实现、已部署或已验证的 Agent 系统。

来源：`Computer_Science_teaching_agent`（原私有草稿，完整内容已转录于本页），原文件 `teaching_rag_minimal_project_structure.md`，Git blob SHA：`9564458f17d82cdf14ed738e36bbe969fa4a73c3`。本页保留草案内容，未修改原仓库，也未更改其可见性。原仓库访问受其权限限制，本页用于统一展示该草案。

以下参数与第一周安排均为原始建议，不代表已完成的实验或开发记录。

---

# 教学 RAG 系统：最小可讲清项目结构（第一版）

## 目标
做一个可运行、可面试讲清楚、可逐步扩展的课程问答系统（RAG）。

## 推荐目录
- data/raw：原始资料（pdf/docx/md）
- data/cleaned：清洗后文本
- knowledge_base：chunks、向量、索引
- src/ingest：文档解析
- src/chunking：切块（含 overlap）
- src/embed：向量化
- src/retrieve：检索
- src/generate：生成
- src/pipeline：build_kb / ask
- src/app：CLI 或 Web 入口
- prompts：系统提示词
- tests：基础测试

## 第一版参数建议
- chunk_size: 400
- chunk_overlap: 80
- top_k: 4
- temperature: 0.2

## 第一周计划
1. Day1: 项目骨架和依赖
2. Day2: 文档解析（先 md/pdf）
3. Day3: 切块 + 抽样人检查
4. Day4: embedding + 向量库存储
5. Day5: 检索 + 生成打通
6. Day6: 小规模问答评测
7. Day7: 准备面试讲解材料


