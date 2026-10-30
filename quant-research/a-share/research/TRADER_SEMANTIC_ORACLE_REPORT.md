# 交易者决策层：静态行业／产业链语义上限实验

生成日期：2026-08-31。样本为 20 笔 L4 普通股买入、8 个独立交易日。本轮只检验“排得像不像他的选择”，不评估收益。

## 1. 结论先行

**这版九产业链弱文本代理没有复现最后选股，冻结闸门失败。**

- 原 F1–F6 基线：Top-20 命中 **2/20**，覆盖 **2/8** 日。
- 冻结完整模型（原六族 + 结构行业方向 + 行业内龙头 + 科技指数 + 九产业链）：Top-20 命中 **1/20**，覆盖 **1/8** 日。
- 完整模型的逐日平均 penalty rank 在 **8/8** 日改善，但没有新增 Top-20 命中：它把一批真实买入从几百／几千名推到更前，却没有完成最后的尖子筛选。
- 最有信息的固定消融是“基线 + 九链语义”：**3/20、3/8 日**；仍低于预定的 4/20、4/8 日门槛，且不能在看到结果后改选为正式模型。

最合理的解释是：**产业链／题材像中间筛选层，不是最后的 3–5 只选股器。** 还缺的更像是当时催化强度、更细题材内的地位、盘中确认和组合约束，而不是再加几个量价窗口。

## 2. 固定消融结果

| 实验 | Top-5 | Top-10 | Top-20 | Top-20命中日 | Top-100 | 真实买入中位日内分位 |
|---|---:|---:|---:|---:|---:|---:|
| 原 F1–F6 基线 | 1 | 1 | **2** | **2** | 5 | 89.8% |
| 仅结构行业方向 | 0 | 0 | 1 | 1 | 1 | 47.2% |
| 仅行业内龙头 | 0 | 1 | 2 | 2 | 3 | 87.5% |
| 仅科技指数题材 | 0 | 0 | 0 | 0 | 2 | 80.9% |
| 仅九链语义 | 0 | 1 | 2 | 1 | 3 | 87.3% |
| 基线 + 九链语义 | 1 | 2 | **3** | **3** | 7 | **93.4%** |
| 冻结完整模型 | 0 | 1 | **1** | **1** | 7 | 92.4% |
| 仅四个新族 | 0 | 0 | 0 | 0 | 4 | 92.7% |

“基线 + 九链语义”的三个 Top-20 是：6月30日麦格米特（第18）、8月14日蓝色光标（第4）、8月27日远东股份（第7）。其中前两只在原基线已是第11和第1，真正新增只有远东股份。

四个新族全加后反而丢失尖子：结构行业方向单独较弱，把部分本来靠前的个股拉低；行业内龙头和科技指数提高了普遍排名，却没有 Top-20 增量。

## 3. 闸门裁决

| 闸门 | 要求 | 观察 | 结果 |
|---|---:|---:|---|
| Recall@20 | ≥4/20 | **1/20** | 失败 |
| Top-20命中日 | ≥4/8 | **1/8** | 失败 |
| 逐日平均 penalty rank 改善 | ≥6/8 | **8/8** | 通过 |
| 新增命中单日集中度 | ≤50% | 新增命中为0 | 失败 |
| 无代码白名单／标签注入 | 必须通过 | 8项审计均通过 | 通过 |

闸门只评冻结的 `complete_f1_f10`，没有按结果事后挑最好的消融变体。

## 4. 映射覆盖与“选择力”不是一回事

- 固定九链共生成 **2,877** 条股票–主题关系，覆盖 **1,984** 只股票。
- 20 笔实际买入中，**19/20** 至少有一个九链标签；百润股份按冻结规则保留为消费残差，没有为追求 20/20 另造主题。
- 九链可评分集每日中位约 **1,707** 只，相对全合格池中位 **4,732** 只，压缩到约 **36.1%**。
- 但仅九链语义的 Top-20 仍只命中 2/20，而且全集中在 8月27日。高覆盖只说明这些交易容易被事后归到当期主线，不代表可以从主线中提前挑出它们。

## 5. 一个必须披露的偏差：当前映射主要是研报标题代理

本轮虽按“静态分类已知”授权回填，但本地没有完整的全市场年报主营／官方产品表。实际映射主要靠当前申万标签、科技指数成分和研报行业／标题关键词：

- 2,877 条关系中，**2,039** 条仅有研报文本证据。
- 1,984 只映射股中，**1,445** 只至少有一条“仅研报”主题，**1,192** 只的所有主题都只靠研报。
- 股票研报数与映射主题数的 Spearman 相关约 **0.376**。没有研报的股票映射率为 9.7%，21 篇以上时上升到 53.3%。
- 20 只实际买入的研报数中位数是 **50.5**，全市场是 **4**；17/20 至少有一个仅研报主题。

因此，`mapping audit pass` 只证明没有读取交易标签、股票白名单或 PnL，**不证明分类本身准确**。研报多的高关注股更容易撞中关键词，19/20 的标签覆盖中混入了关注度偏差。

所以本报告的严格结论是：**“这版九链弱文本代理不足以复现最后买入”，不是“真实、完整的已知产业链也无效”。**

## 6. 数据与时钟边界

- 价格、成交额、申万指数和 `daily_basic` 数值只用 T-1 及更早；研报动态行业标签在发布后下一交易日才生效。
- 当前申万、科技指数和全期研报文本允许静态回填，全部标成 `oracle_static`；这不是前向策略。
- `daily_basic` 在前 6 个日期的合格池覆盖为 99.8%–100%，8月26、27日为 0。结构龙头仍可靠动量和流动性组件得分，因此“族分有值”不等于换手率、量比、流通市值等所有计划组件都齐全。
- 九主题是看过这批案例后提出的同样本解释性假说。词典冻结后统一扫全市场，不含公司名、股票代码或逐股修正，但仍不是独立样本外验证。
- manifest 对较大原始文件使用首尾分块哈希，对小文件使用完整 SHA-256；它可发现常见本地漂移，不是第三方公证。

## 7. 主题定义的公开依据

| 主题 | 主要定义依据 |
|---|---|
| PCB | [生态环境部电子电路制造业产排污系数手册](https://www.mee.gov.cn/xxgk2018/xxgk/xxgk06/202109/W020210926520414533835.pdf) |
| AI算力／高速网络 | [中证算力基础设施主题指数编制方案](https://oss-ch.csindex.com.cn/static/html/csindex/public/uploads/indices/detail/files/zh_CN/931657_Index_Methodology_cn.pdf) |
| 半导体 | [中证全指半导体产品与设备指数编制方案](https://oss-ch.csindex.com.cn/static/html/csindex/public/uploads/indices/detail/files/zh_CN/20231208180055-H30007_Index_Methodology_cn.pdf) |
| 液冷 | [工信部数据中心先进适用技术指南](https://www.miit.gov.cn/jgsj/jns/nyjy/art/2022/art_fbb57e69a8b148cfa263953a71b53e2e.html) |
| 卫星／军工 | [中证卫星产业指数编制方案](https://oss-ch.csindex.com.cn/static/html/csindex/public/uploads/indices/detail/files/zh_CN/931594_Index_Methodology_cn.pdf) |
| 机器人／高端制造 | [中证机器人指数编制方案](https://oss-ch.csindex.com.cn/static/html/csindex/public/uploads/indices/detail/files/zh_CN/20231208180051-H30590_Index_Methodology_cn.pdf) |
| 电力电网／电力电子 | [中证电网设备主题指数编制方案](https://oss-ch.csindex.com.cn/static/html/csindex/public/uploads/indices/detail/files/zh_CN/20231208180048-931994_Index_Methodology_cn.pdf) |
| 先进材料 | [中证新材料主题指数编制方案](https://oss-ch.csindex.com.cn/static/html/csindex/public/uploads/indices/detail/files/zh_CN/20231208180056-H30597_Index_Methodology_cn.pdf) |
| AI应用 | [中证人工智能主题指数编制方案](https://oss-ch.csindex.com.cn/static/html/csindex/public/uploads/indices/detail/files/zh_CN/931449_Index_Methodology_cn.pdf) |

## 8. 后续最有价值的补充

如果继续做当前静态上限，优先级不是继续扩词典，而是：

1. 用官方指数成分、年报主营和官网正式产品表建“强证据”全市场多标签，替换研报标题 OR 匹配。
2. 做来源消融：申万 + 正式指数，再与“+研报标题”对比，量化关注度混杂。
3. 在主题已知的前提下，增加买入当日前的新闻／公告催化、主题内地位和盘中确认；这才是目前最可能的最后选择层。

不建议再围绕这 20 笔增加主题、关键词或调整族权重；那只会将解释性上限变成逐股过拟合。

## 9. 可复跑产物

- 冻结协议：`research/TRADER_SEMANTIC_ORACLE_PROTOCOL.md`
- 运行器：`scripts/run_trader_semantic_oracle.py`
- 产业／数据层：`src/afactor/trader_choice_industry.py`
- 九链语义：`src/afactor/trader_choice_semantic.py`
- 结构方向／龙头：`src/afactor/trader_choice_oracle.py`
- 精简全截面：`research/ledger/trader_oracle_panel.parquet`
- 15 组消融：`research/ledger/trader_oracle_experiment_{daily,summary}.csv`
- 逐笔排名：`research/ledger/trader_oracle_label_ranks.csv`
- 主题／科技成分审计：`research/ledger/trader_oracle_{theme,tech}_audit.csv`
- 覆盖与压缩：`research/ledger/trader_oracle_{feature_coverage,coverage_curve}.csv`
- 终局闸门：`research/ledger/trader_oracle_gate.json`
- 输入、代码与输出哈希：`research/ledger/trader_oracle_manifest.json`

最终校验：24 项相关测试通过，Ruff 和编译检查通过；面板 44,520 行，日表 120 行，汇总 15 行，逐笔 20 行，主键无重复。
