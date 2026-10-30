# INC-T1 执行优化：审查后修订

> Latest bounded replay: W5 same-rule pool excess delta +0.689/+0.681/+0.627 pp/year; W9 +0.241/+0.232/+0.114. W9 at 1B fails development-sign consistency (dev -0.031; full paired t=1.205). These post-review account diagnostics supersede deployability inferences from the original mechanical tables. No investment adoption.

本报告替代旧主报告。预注册 `INC_T1_PREREG.md` 原文不变；旧报告保留在 `INC_T1_REPORT_prereview.md`，旧结论不可再引用为当前结论。只用已落盘的2021–2025计算（B×GBT为2023–2025）；未读取2026数据，未联网、未提交、未改其他轨道。

排雷的 W5（13:01–14:00）和 W9（全日VWAP）在修正后仍满足预注册机械判据。W7降为三档资金均证据不足，W8三档均证据不足。T1是样本内执行增量证据，尚未完成真实可成交验证；不能直接修改现行冻结执行方案或部署。Top20逐格通过结果保留，不能用“策略本身不赚钱”覆盖它们。

## 修订结果与基线

所有数值为年化百分点。第一层 ΔC 是恒定AUM算术节省，main相对同口径W1 VWAP（09:31–09:40），不是相对W1 TWAP。第二层是收盘终点计价后的几何年化收益差，与第一层不能相加；Δ同窗超额再扣各自同窗同频池基准。VWAP方案对W1 VWAP；单价W0/W8亦对W1 VWAP；D1 TWAP对W1 TWAP。

| 资金 | scheme | dC_full | dC_dev | dC_test | t_full | d_realized_A | t_realized_A | d_model_B | l2_total_diff_full | l2_d_excess_full | verdict_mechanical | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1000万 | W5 | 0.381 | 0.385 | 0.374 | 4.692 | 0.335 | 4.130 | 0.046 | 0.489 | 0.375 | 有增量 | 有增量 |
| 1000万 | W7 | 0.180 | 0.222 | 0.117 | 2.042 | 0.094 | 1.064 | 0.086 | 0.276 | 0.149 | 证据不足 | 证据不足 |
| 1000万 | W8 | 0.206 | 0.258 | 0.128 | 2.244 | 0.079 | 0.857 | 0.127 | 0.126 | -0.044 | 证据不足 | 证据不足 |
| 1000万 | W9 | 0.410 | 0.413 | 0.405 | 7.186 | 0.375 | 6.576 | 0.035 | 0.252 | 0.525 | 有增量 | 有增量 |
| 1亿 | W5 | 0.481 | 0.504 | 0.447 | 5.927 | 0.337 | 4.160 | 0.144 | 0.605 | 0.490 | 有增量 | 有增量 |
| 1亿 | W7 | 0.372 | 0.435 | 0.277 | 4.238 | 0.100 | 1.140 | 0.272 | 0.495 | 0.368 | 证据不足 | 证据不足 |
| 1亿 | W8 | 0.551 | 0.620 | 0.448 | 5.729 | 0.141 | 1.456 | 0.410 | 0.480 | 0.310 | 证据不足 | 证据不足 |
| 1亿 | W9 | 0.485 | 0.501 | 0.460 | 8.504 | 0.375 | 6.585 | 0.109 | 0.337 | 0.611 | 有增量 | 有增量 |
| 10亿 | W5 | 0.778 | 0.836 | 0.691 | 9.616 | 0.370 | 4.619 | 0.408 | 0.821 | 0.707 | 有增量 | 有增量 |
| 10亿 | W7 | 0.954 | 1.043 | 0.822 | 10.758 | 0.184 | 2.074 | 0.771 | 1.023 | 0.899 | 有增量 | 证据不足 |
| 10亿 | W8 | 1.338 | 1.438 | 1.187 | 10.213 | 0.144 | 1.094 | 1.193 | 1.043 | 0.876 | 证据不足 | 证据不足 |
| 10亿 | W9 | 0.675 | 0.710 | 0.621 | 11.989 | 0.373 | 6.637 | 0.302 | 0.345 | 0.621 | 有增量 | 有增量 |

W5可供综合使用的两层点估计区间为1000万 +0.375～+0.381、1亿 +0.481～+0.490、10亿 +0.707～+0.778 pp/年；这是不同估计对象的区间，不是置信区间或可承诺收益。W9相应为 +0.410～+0.525、+0.485～+0.611、+0.621～+0.675。W9更少受日内局部封板影响，但仍依赖成本与基准假设。

W5原第一层 +0.465/+0.565/+0.850 已下降到 +0.381/+0.481/+0.778；原第二层 +0.45/+0.56/+0.78 混入价格口径差，现为 +0.375/+0.490/+0.707。第二层总差现 +0.490/+0.605/+0.821，仅作组合总收益参考。旧“+5.28→+5.73”是TWAP与VWAP混比，不再使用。旧口径列 `*_vs_w1twap` 与当前 `*_vs_ref` 并存，二者差才是同族W1的口径差，不能加入任何增量。

## 封板修复与未关闭的模型限制

原审查的例子002222.SZ、2021-12-23在W5整窗涨停，买入不能按10% POV视为成交。修复第一层main采用方向性dir：封涨停挡买、封跌停挡卖，整腿顺延次日同窗；tight和any并列敏感性。第二层 `_seal` 采用any，双向禁成交，继承引擎约定，不能称为方向性可成交验证。旧无后缀族保留用于前后对照。

10亿W7的单次顺延机械结果仍是“有增量”（A=+0.184、t=2.074），不是原审查声称的所有规模都自动失败。多次顺延的已实现价格诊断降到+0.060、t=0.681，故修订判断降为证据不足；`verdict_mechanical`保留机械结果，`verdict`记录审查后判断。1000万W7也只有+0.059、t=0.662；W8检验窗多次顺延为负。该诊断不含价差、冲击、POV截量、持仓反馈，只有1000万和10亿，不能冒充第二次完整回测或把数值插入原ΔC。

| X | window | mode | px_rel_saving_pp | t_nw |
|---|---|---|---|---|
| 10000000.000 | W5 | none | 0.419 | 5.183 |
| 10000000.000 | W7 | none | 0.238 | 2.727 |
| 10000000.000 | W8 | none | 0.243 | 2.692 |
| 10000000.000 | W9 | none | 0.371 | 6.493 |
| 10000000.000 | W5 | roll1 | 0.335 | 4.129 |
| 10000000.000 | W7 | roll1 | 0.093 | 1.053 |
| 10000000.000 | W8 | roll1 | 0.070 | 0.762 |
| 10000000.000 | W9 | roll1 | 0.375 | 6.576 |
| 10000000.000 | W5 | rollN | 0.315 | 3.885 |
| 10000000.000 | W7 | rollN | 0.059 | 0.662 |
| 10000000.000 | W8 | rollN | 0.033 | 0.352 |
| 10000000.000 | W9 | rollN | 0.375 | 6.564 |
| 1000000000.000 | W5 | none | 0.421 | 5.282 |
| 1000000000.000 | W7 | none | 0.239 | 2.770 |
| 1000000000.000 | W8 | none | 0.242 | 2.722 |
| 1000000000.000 | W9 | none | 0.369 | 6.562 |
| 1000000000.000 | W5 | roll1 | 0.336 | 4.199 |
| 1000000000.000 | W7 | roll1 | 0.093 | 1.064 |
| 1000000000.000 | W8 | roll1 | 0.069 | 0.758 |
| 1000000000.000 | W9 | roll1 | 0.373 | 6.637 |
| 1000000000.000 | W5 | rollN | 0.316 | 3.947 |
| 1000000000.000 | W7 | rollN | 0.060 | 0.681 |
| 1000000000.000 | W8 | rollN | 0.033 | 0.358 |
| 1000000000.000 | W9 | rollN | 0.373 | 6.625 |

第一层只顺延一次，次日仍封板也可能按其价格计成本；缺价及无量顺延也不是无限重试。rollN最多检查5次延迟，仍封板或无法定价的残余按零项处理，并非真实订单取消现金账。因此P0的错误收益主张已处置，但持续封板订单的完整执行模型未完成。W5的rollN已实现节省约+0.315/+0.316（1000万/10亿），t=3.88/3.95，方向未消失；W9约+0.375/+0.373、t=6.56/6.62。不能把修复后的模型点估计称为完全可实现收入。

## 冲击假设与报价测量

s_W用同年池均值波动/成交额比，含全年未来信息，只是事后成本分解；逐股票日缩放也用当日实现量。Y未校准，POV用当天实现成交额。要形成可部署方案，需冻结过去窗口估计的系数、真实订单执行规则，并以报价与订单回放校准。这里保留预注册消融，未事后优化参数。

| X | scheme | variant | d_credible_pp |
|---|---|---|---|
| 10000000.000 | W5 | Y0.5 | 0.358 |
| 10000000.000 | W9 | Y0.5 | 0.392 |
| 10000000.000 | W5 | Y1.5 | 0.403 |
| 10000000.000 | W9 | Y1.5 | 0.427 |
| 10000000.000 | W5 | twap | 0.353 |
| 10000000.000 | W9 | twap | 0.359 |
| 10000000.000 | W5 | s_one | 0.335 |
| 10000000.000 | W9 | s_one | 0.375 |
| 100000000.000 | W5 | Y0.5 | 0.409 |
| 100000000.000 | W9 | Y0.5 | 0.430 |
| 100000000.000 | W5 | Y1.5 | 0.553 |
| 100000000.000 | W9 | Y1.5 | 0.539 |
| 100000000.000 | W5 | twap | 0.454 |
| 100000000.000 | W9 | twap | 0.434 |
| 100000000.000 | W5 | s_one | 0.337 |
| 100000000.000 | W9 | s_one | 0.375 |
| 1000000000.000 | W5 | Y0.5 | 0.574 |
| 1000000000.000 | W9 | Y0.5 | 0.524 |
| 1000000000.000 | W5 | Y1.5 | 0.982 |
| 1000000000.000 | W9 | Y1.5 | 0.826 |
| 1000000000.000 | W5 | twap | 0.748 |
| 1000000000.000 | W9 | twap | 0.624 |
| 1000000000.000 | W5 | s_one | 0.381 |
| 1000000000.000 | W9 | s_one | 0.373 |

W5去窗口冲击缩放（s_one）仅 +0.335/+0.337/+0.381，10亿main约一半来自模型项，不能把+0.778全称为模型无关收益。成交价来自分钟bar成交价近似，不是买卖盘中间价；开盘调出股可能多在买价成交，已实现路径含交易方向组成，另加半价差可能重复计量。没有报价数据，无法分离数bp的组成效应；外部5分钟成交量核对只验证VWAP近似，不验证买入排队或半价差。

## Top20逐格结果与部署判断

| strategy | X | scheme | dC_full | t_full | d_realized_A | t_realized_A | d_px_mkt_excluded | verdict |
|---|---|---|---|---|---|---|---|---|
| B_gbt_top20 | 10000000.000 | S:W7\|B:W9 | 5.561 | 6.389 | 1.807 | 2.143 | -1.617 | 有增量 |
| COMP_v5_top20 | 10000000.000 | S:W1\|B:W3 | 2.035 | 6.522 | 0.784 | 2.585 | -1.922 | 有增量 |
| COMP_v5_top20 | 10000000.000 | S:W1\|B:W5 | 2.322 | 5.525 | 1.195 | 2.884 | -3.349 | 有增量 |
| COMP_v5_top20 | 10000000.000 | S:W1\|B:W9 | 2.378 | 7.688 | 1.561 | 5.133 | -3.250 | 有增量 |
| COMP_v5_top20 | 10000000.000 | S:W3\|B:W9 | 3.079 | 7.655 | 0.992 | 2.662 | -1.356 | 有增量 |
| COMP_v5_top20 | 10000000.000 | S:W7\|B:W9 | 4.302 | 7.531 | 1.174 | 2.221 | -1.633 | 有增量 |

以上是当前修复账本的通过格，保留其执行增量判定。买卖分离没有第二层表达，池子级漂移另列且不纳入判据；真实净收益必须计入漂移与现金暴露。Top20策略自身扣成本超额为负只影响投资/部署判断。撤回原报告未经登记的“总体证据不足”覆盖逐格结论，其他格以summary为准；不能将本轨道增量当成Top20策略转正证明。

## 原审查逐条处置

原文从review代理ad3fa5ac884a784c8.jsonl的StructuredOutput提取，保存为 `inc_t1_original_review.json`；原输出JSON按 `d['result']` 解码后T1的fix为null，说明中断而非完成。

| id | severity | disposition | evidence_and_limit |
|---|---|---|---|
| R1 | P0 | 采纳；收益伪影已修正，成交模型仍受限 | 两层既有封板修复复用；W7三档、W8三档均证据不足。单次顺延仍可能再次封板；rollN只是价格诊断，不能声称完整可成交模拟。 |
| R2 | P1 | 采纳；已修复并回归验证 | 第二层每行按ref_mode同价格口径扣减；VWAP对W1 VWAP，延迟TWAP对W1 TWAP；旧TWAP差保留为对照。 |
| R3 | P2 | 采纳；撤销整体覆盖逐格判据 | Top20通过格逐一保留；策略净超额为负和池子漂移风险只影响部署判断，不改执行增量机械判定。 |
| R4 | P2 | 采纳；已修正文案 | 第一层main相对W1 VWAP；不称为TWAP切换到VWAP的执行alpha。 |
| R5 | P2 | 采纳；明确未解决的测量局限 | 成交bar价含买卖方向组成、可能与半价差重复；同年s_W含前视；s_one是消融，不是实际报价/冲击校准。 |

## 复用、偏离与验证

复用a9e5bc3c7316ffd08已跑完的封板腿、引擎模式、summarize、seal诊断及三档资金账本，没有重跑大规模收益计算。此次只刷新判定、图和报告，并补同基线逐行代数、roll1诊断对账、审查降级回归测试。旧窗口聚合37项、旧/新28模式引擎断言、成交清单门槛仍从原账本验证。

偏离登记：原报告§7记录全部保留于prereview档案；新增D08封板规则修复（dir第一层/any第二层），D09第二层同口径基线修正，D10撤销Top20整体覆盖逐格判据，D11基线名称修正，D12披露同年冲击及成交价组成，D13连续封板价格诊断对10亿W7增加事后保守降级。D13不回写预注册、不冒称新预注册研究；诊断未进入原机械判据。

完整逐格：`inc_t1_summary.csv`；旧新主窗口对照：`inc_t1_review_changed_results.csv`；复查意见：`inc_t1_review.csv`；机器可读综合交接：`inc_t1_completion.json`。图 `reports/inc_t1_layer1_single.png`、`inc_t1_layer2.png`、`inc_t1_paths.png` 已从修复账本重新生成，只展示模型结果。

旧全网格账本仅完成文档审查处置；方向性订单、有限生命周期现金/库存和事前系数随后已在本报告D14的三个窗口账户回放中实现，报价校准仍是证据缺口。所有结果在历史样本内，五个年份并非独立前向验证；长历史换手率版本无分钟数据，不能直接外推。现行冻结方案保持不变。


## Bounded account replay: engineering follow-up (D14)

This follow-up completes the feasible bounded engineering requested after the earlier documentation closure. It supersedes any inference that the old model increment is a deployable gain. It does not relabel preregistered cells or rerun the entire grid. Scope: exclusion W1/W5/W9, three initial AUMs, five independent phases, plus identical-rule full-pool controls.

Orders are fixed at previous-close NAV and prices, generated every five days, active only in the selected window, and explicitly CANCELLED at that window's end. Remaining inventory stays invested; residual orders do not persist overnight and are only regenerated on the next five-day rebalance. There are no invented next-day fills, including repeated sealed limits. Buy orders at upper-limit minute closes and sell orders at lower-limit minute closes are blocked; opposite directions are permitted. The stock/order participation rate is locked at previous close: requested notional divided by the preceding 20-day median selected-window turnover, capped at 10%. Each completed bar uses this fixed rate times its observed turnover; no day's final turnover is used to set its rate. This approximates causal volume-weighted pacing rather than front-loading every order at 10%. Each stock's minute-side notional is <=10% of that minute's observed turnover. Buys use start-of-minute cash, with commission/spread/impact included; contemporaneous sells cannot finance them. Sells cannot exceed opening inventory (T+1). Cash cannot be borrowed (roundoff tolerance 1e-6 yuan).

The word causal refers to state/order recurrence: a bar consumes no subsequent bar. A completed minute CLOSE and its full observed turnover are still retrospective execution proxies, not proof of executable intrabar order placement, quote access or queue priority. Empirical quote/orderbook calibration is unavailable and remains an evidence gap.

Window impact scaling is FIXED1, the existing preregistered sensitivity; no same-year s_W is consumed. Sigma, ADV and half-spread use the production cost model's shift(1), at most trade_date-1 data. Future perturbation and truncated-prefix tests validate the actual cost model. Impact is charged as the incremental 1.5 power of each stock's daily cumulative filled notional, preventing artificial cost reduction by slicing into minutes. Commission and date-correct sell stamp tax debit cash.

Each of the five phases uses the FULL initial AUM independently; daily returns are averaged. This is conservative for impact versus a real product splitting capital five ways. Capital subsequently grows with NAV, unlike the old constant-AUM model. Inventory uses fractional adjusted-price units, without 100-share rounding, minimum commissions or actual corporate-action settlement. These are disclosed research/product approximations, not hidden sealed-limit fills.

Full-pool controls execute EXACTLY the same five-day phases, prices, directional limits, cash financing, POV capacity, lagged estimates, fixed1 costs and cancellation rule. `total_diff_vs_W1_pp` is the raw exclusion annual-return difference. `d_excess_vs_W1_pp` subtracts each window's own same-rule pool excess before comparing against W1; they are different objects and never added. `paired_excess_arith_pp` and `t_nw` use paired daily differences of these excess returns. All pp columns are annual percentage points.

| X | window | ann_pp | pool_ann_pp | total_diff_vs_W1_pp | d_excess_vs_W1_pp | paired_excess_arith_pp | t_nw | cost_ann_pp | mean_exposure |
|---|---|---|---|---|---|---|---|---|---|
| 10000000.000 | W1 | 15.925 | 10.921 | 0.000 | 0.000 | 0.000 | — | 1.049 | 0.975 |
| 10000000.000 | W5 | 16.706 | 11.012 | 0.780 | 0.689 | 0.602 | 6.935 | 1.057 | 0.983 |
| 10000000.000 | W9 | 16.220 | 10.974 | 0.295 | 0.241 | 0.208 | 3.763 | 1.072 | 0.981 |
| 100000000.000 | W1 | 15.560 | 10.849 | 0.000 | 0.000 | 0.000 | — | 1.366 | 0.975 |
| 100000000.000 | W5 | 16.337 | 10.945 | 0.777 | 0.681 | 0.597 | 6.877 | 1.381 | 0.983 |
| 100000000.000 | W9 | 15.841 | 10.898 | 0.281 | 0.232 | 0.201 | 3.645 | 1.400 | 0.981 |
| 1000000000.000 | W1 | 14.575 | 10.679 | 0.000 | 0.000 | 0.000 | — | 2.227 | 0.978 |
| 1000000000.000 | W5 | 15.278 | 10.755 | 0.703 | 0.627 | 0.554 | 6.250 | 2.344 | 0.985 |
| 1000000000.000 | W9 | 14.675 | 10.665 | 0.100 | 0.114 | 0.094 | 1.205 | 2.417 | 0.982 |

Development/test split:

| X | window | period | d_excess_vs_W1_pp | paired_excess_arith_pp | t_nw |
|---|---|---|---|---|---|
| 10000000.000 | W5 | 2021-2025 | 0.689 | 0.602 | 6.935 |
| 10000000.000 | W9 | 2021-2025 | 0.241 | 0.208 | 3.763 |
| 10000000.000 | W5 | 2021-2023 | 0.638 | 0.591 | 6.391 |
| 10000000.000 | W9 | 2021-2023 | 0.124 | 0.110 | 2.421 |
| 10000000.000 | W5 | 2024-2025 | 0.773 | 0.617 | 3.701 |
| 10000000.000 | W9 | 2024-2025 | 0.445 | 0.353 | 2.998 |
| 100000000.000 | W5 | 2021-2025 | 0.681 | 0.597 | 6.877 |
| 100000000.000 | W9 | 2021-2025 | 0.232 | 0.201 | 3.645 |
| 100000000.000 | W5 | 2021-2023 | 0.634 | 0.589 | 6.332 |
| 100000000.000 | W9 | 2021-2023 | 0.117 | 0.105 | 2.294 |
| 100000000.000 | W5 | 2024-2025 | 0.761 | 0.610 | 3.663 |
| 100000000.000 | W9 | 2024-2025 | 0.432 | 0.345 | 2.932 |
| 1000000000.000 | W5 | 2021-2025 | 0.627 | 0.554 | 6.250 |
| 1000000000.000 | W9 | 2021-2025 | 0.114 | 0.094 | 1.205 |
| 1000000000.000 | W5 | 2021-2023 | 0.572 | 0.534 | 6.017 |
| 1000000000.000 | W9 | 2021-2023 | -0.031 | -0.039 | -0.408 |
| 1000000000.000 | W5 | 2024-2025 | 0.718 | 0.583 | 3.294 |
| 1000000000.000 | W9 | 2024-2025 | 0.367 | 0.294 | 2.243 |

W5 paired same-rule excess is +0.602/+0.597/+0.554 pp/year arithmetically, with full NW t=6.94/6.88/6.25; both subperiods are positive. Its geometric excess difference is +0.689/+0.681/+0.627. W9 is +0.241/+0.232/+0.114 geometrically; 1B development excess is -0.031 and full t=1.205, so it is not a robust three-AUM recommendation. The larger W5 result than the old fixed-trade ledger is model/rule sensitivity, not an additional increment to stack: previous-close sizing, growing NAV, volume pacing and finite cancellation change inventory/cash paths. W5 modeled annual transaction cost is actually slightly higher than W1 at all AUMs; its gain is not a reduction in that modeled cost.

These are POST-REVIEW diagnostics, not new preregistered discoveries. The older W5/W9 mechanical pass labels remain historical evidence; bounded implementation conclusions must use the new table, especially same-rule excess and development/test consistency. No investment adoption, freeze change or new forward-test claim follows.

Artifacts: `inc_t1_causal_daily.csv` (phase cash/NAV/cost/cancellation), `inc_t1_causal_summary.csv`, `inc_t1_causal_checks.csv` (runtime capacity/cash/cancellation/cutoff), `inc_t1_causal_sources.csv` (actually read historical months), `inc_t1_causal_rules.json` (rule frozen before replay outcomes). `cache/_tmp/inc_t1_causal/orders_YYYYMM.parquet` contains per-stock requested/filled/cancelled units, notional, costs, blocked-minute counts and maximum actual minute participation. `inc_t1_causal_journal_checks.csv` independently reconstructs cash, inventory and close-valued NAV from these saved fills, rather than rerunning prices to produce another backtest. Terminal units are in `cache/_tmp/inc_t1_causal/terminal_inventory.parquet`.

The initial diagnostic was manually interrupted after completed month 202207: filling as fast as 10% POV allowed small W9 orders to finish near the open, rather than providing long-window volume-weighted pacing. The corrected replay locks the rate from lagged selected-window volume. The superseded account journals remain in `cache/_tmp/inc_t1_causal_superseded_frontload/`; the preserved log `logs/inc_t1_causal_superseded_frontload.log` ends with the manual KeyboardInterrupt. This was a rule correction, not a runtime assertion failure. `inc_t1_causal_run_manifest.json` records the change and earlier corrected startup/report-edit errors. Superseded outcomes are not used.

Directional execution, finite order lifecycle, cash/inventory feedback and same-year impact look-ahead are now implemented for this bounded replay. Quote/queue calibration and actual-share product constraints remain evidence gaps. The original five-year, sample-internal signal and survivor/adjustment limitations also remain. No 2026 inputs, commits or changes to other tracks.
