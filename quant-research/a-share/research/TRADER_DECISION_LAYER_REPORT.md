# 交易者决策层：首轮结果

生成日期：2026-08-31 11:47。本报告全部为探索性结果；统计独立单位为买入日。模仿度与未来收益分别裁决。

## 1. 候选池先验检查

| 池 | 每日候选中位数 | L4逐笔召回 | 有命中日期占比 |
|---|---:|---:|---:|
| full_eligible | 4732 | 95.0% | 100.0% |
| legacy_P150 | 117 | 35.0% | 50.0% |
| high_recall_parent | 604 | 50.0% | 62.5% |
| pv_top3pct | 140 | 30.0% | 50.0% |

`legacy_P150` 是旧阈值规则（名称仅为历史沿用，并不表示每日固定 104–150 只）；`pv_top3pct` 是参考 VCP/leader/reacceleration 构造的固定 3% 多形态池；`high_recall_parent` 是较宽的多形态并集。任何真实买入都没有被强行补入候选池。

## 2. 分层选择消融

| 排名池 | 累加实验 | 候选中位数 | Recall@5 | Recall@10 | Recall@20 |
|---|---|---:|---:|---:|---:|
| full_eligible | pool_only | 4732 | 0.0% | 0.0% | 0.0% |
| full_eligible | only_price_volume | 4732 | 0.0% | 0.0% | 10.0% |
| full_eligible | only_dynamic_direction | 4732 | 0.0% | 0.0% | 0.0% |
| full_eligible | only_within_direction_leader | 4732 | 0.0% | 0.0% | 0.0% |
| full_eligible | only_event_attention | 4732 | 0.0% | 0.0% | 0.0% |
| full_eligible | only_familiarity | 4732 | 0.0% | 0.0% | 5.0% |
| full_eligible | only_portfolio_replacement | 4732 | 0.0% | 0.0% | 0.0% |
| full_eligible | through_price_volume | 4732 | 0.0% | 0.0% | 10.0% |
| full_eligible | through_dynamic_direction | 4732 | 0.0% | 0.0% | 0.0% |
| full_eligible | through_within_direction_leader | 4732 | 0.0% | 0.0% | 0.0% |
| full_eligible | through_event_attention | 4732 | 0.0% | 0.0% | 0.0% |
| full_eligible | through_familiarity | 4732 | 5.0% | 5.0% | 10.0% |
| full_eligible | through_portfolio_replacement | 4732 | 5.0% | 5.0% | 10.0% |
| high_recall_parent | pool_only | 604 | 0.0% | 0.0% | 0.0% |
| high_recall_parent | only_price_volume | 604 | 0.0% | 5.0% | 10.0% |
| high_recall_parent | only_dynamic_direction | 604 | 0.0% | 0.0% | 0.0% |
| high_recall_parent | only_within_direction_leader | 604 | 0.0% | 0.0% | 0.0% |
| high_recall_parent | only_event_attention | 604 | 0.0% | 0.0% | 0.0% |
| high_recall_parent | only_familiarity | 604 | 0.0% | 5.0% | 5.0% |
| high_recall_parent | only_portfolio_replacement | 604 | 0.0% | 0.0% | 0.0% |
| high_recall_parent | through_price_volume | 604 | 0.0% | 5.0% | 10.0% |
| high_recall_parent | through_dynamic_direction | 604 | 0.0% | 0.0% | 10.0% |
| high_recall_parent | through_within_direction_leader | 604 | 0.0% | 0.0% | 5.0% |
| high_recall_parent | through_event_attention | 604 | 0.0% | 0.0% | 5.0% |
| high_recall_parent | through_familiarity | 604 | 5.0% | 5.0% | 5.0% |
| high_recall_parent | through_portfolio_replacement | 604 | 5.0% | 5.0% | 5.0% |

族顺序固定为：量价 → 动态方向 → 方向内龙头 → 事件/关注度 → 熟悉池；组合替换仅在严格 T-1 完整持仓日进入。当前只有 8 个独立 L4 买入日，因此没有训练机器学习权重，以上均为日内百分位和固定等权。

## 3. 同统计方向代理随机篮子收益对照

| 持有口径 | 实际篮子均值 | 同 PIT 收益聚类随机均值 | 实际平均分位 | 有效模拟数 |
|---|---:|---:|---:|---:|
| fwd_open_to_open_d1 | 2.9% | 2.1% | 60.3% | 70000 |
| fwd_open_to_open_d3 | 4.3% | 2.6% | 63.6% | 60000 |
| fwd_open_to_open_d5 | 5.0% | 1.8% | 59.4% | 50000 |

收益为 `open[T+h]/open[T]-1`；净收益另扣 7bp 往返成本。这里的“方向”只是严格 PIT 的收益聚类代理，不是已经验证的行业或题材。该检验回答“统计方向代理给定后，他挑的个股是否优于随机”，不用于反向优化选择分数。

## 4. 首轮裁决

1. **旧候选池不是高召回入口。** `legacy_P150` 每日中位 117 只，只覆盖 35.0% 的 L4 买入；扩大到中位 604 只也只覆盖 50.0%。参考 VCP/leader/reacceleration 的固定 3% 池覆盖更低（30.0%）。因此不能在这 100–600 只里直接学习最后 3–5 只，否则一半以上答案已在第一层丢失。
2. **量价适合作为画像，不足以还原最终决策。** 单独量价使真实买入的中位日内分位升至 92.7%，但 Recall@10 仍是 0.0%。动态方向、方向内龙头、事件关注单独的中位分位分别为 52.4%、73.9%、69.0%，均未产生 Top-20 命中。
3. **熟悉池是唯一可见的行为线索，但证据只有一例。** 单独熟悉度的中位分位为 87.5%；加入后完整模型首次出现 Top-5 命中，最终也只有 1/20。逐笔核对表明该命中来自此前已披露持有过的蓝色光标，不能凭一例外推为稳定因子。
4. **组合替换没有观察到增量。** 它只在严格 T-1 有完整普通股持仓的 3 个买入日可计算；加入前后 Recall@5/10/20 不变。ETF 没有被当作待卖股票，持仓快照也没有向前携带。
5. **统计方向代理给定后的个股收益只有弱线索，尚未过证据门槛。** 相对同一 PIT 收益聚类随机篮子的均值差为：1日 +0.75pp，胜 5/7 日，单边符号检验 p=0.227；3日 +1.79pp，胜 4/6 日，单边符号检验 p=0.344；5日 +3.15pp，胜 4/5 日，单边符号检验 p=0.188；与更严格的同聚类、流动性、波动匹配股相比则为：1日 -0.91pp；3日 +0.33pp；5日 +0.51pp。严格匹配覆盖 18/20 笔标签；可计算的独立日期分别为 1日 7 个、3日 6 个、5日 5 个。两类对照方向不一致，不能认定已有稳定选股 alpha。

综合判断：现有证据与“**量价/形态做宽筛，题材语义和人工判断完成最后选择，纪律化执行买卖**”相容，但不支持“当前这些可见因子已经构成可复现的纯量化模型”。下一道最有价值的数据闸门是带历史生效日期的概念/题材成员与新闻语义，而不是继续堆叠量价窗口。

## 5. 数据边界

过期而记为未知的源：reports: 2026-08-26, 2026-08-27；forecast: 2026-08-21, 2026-08-26, 2026-08-27；express: 2026-08-26, 2026-08-27；toplist: 2026-08-26, 2026-08-27。

- 当前申万快照不进入主模型；方向主口径为前季末 120 日收益聚类、下一季度生效的 PIT 统计行业。
- 8 月 28 日天华新能尚未同时具备规范事件与全市场行情，因此没有混入主样本。
- 无帖子或缺图日期不产生负标签；事件字段在源截止后是 NA，不是 0。

## 6. 相似方法的使用边界

- [IBD 官方 CAN SLIM / SwingTrader 指南](https://get.investors.com/wp-content/uploads/2025/01/How-To-Use-SwingTrader-2025.pdf)：只借鉴“基本面/质量预筛 + 技术执行”的分层流程，不移植美股仓位和止损阈值。
- [Mark Minervini 官方 SEPA 说明](https://www.minervini.com/)：只映射高动量筛选、领导力和图形确认；未公开的 VCP 专有阈值不作假设。
- [George & Hwang 52周高点动量论文](https://onlinelibrary.wiley.com/doi/10.1111/j.1540-6261.2004.00695.x)：支持检验既往领导地位，不支持把深回撤百分比当成论文结论。
- [Opening-range breakout 原始论文](https://www.sciencedirect.com/science/article/pii/S1544612312000438)：只作为后续盘中确认/执行层假设；其原油期货窗口和阈值不进入 A 股选股模型。

完整逐日结果见 `research/ledger/trader_choice_*`；冻结协议见 `research/TRADER_DECISION_LAYER_PROTOCOL.md`。
