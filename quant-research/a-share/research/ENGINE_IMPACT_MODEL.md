# 回测引擎改造与冲击成本模型(前缀 eng,2026-10-10)

> 组合层(指数增强优化器、容量曲线、PEAD 事件组合)要用的底座:一个能喂任意权重或路径依赖
> target_fn 的成交模拟器,加上一个引擎与优化器共用的成本模型(佣金 + 印花税 + 半价差 + 平方根冲击)。
>
> 代码:`src/afactor/strategy.py`(simulate / run_weights / build_weights,新口径 vwap1)、
> `src/afactor/impact.py`(CostModel)。测试:`tests/test_engine_regression.py`。
> 账本:`research/ledger/eng_*.csv`。金标准:`tests/golden_engine.json`。
> 研究期 2020-01-02 ~ 2025-12-31,未触碰封存数据。**数据不含已退市股(约 209 只),
> 所有收益数字有幸存者偏差,方向是高估**(README 估计池等权高估 3~6pp/年)。
>
> **2026-10-10 审查修复**(详见文末"审查记录"):callable 的 `ctx` 只给决策时点可得的信息
> (`cap_w` 改为事前估计,新增 `h_dec / nav_dec / budget_dec`),目标里 NaN = 该股不交易、负权重报错;
> vwap1 是平均价口径,日收益正自相关,引擎结果追加 `vol_adj / sharpe_adj`、新增 `excess_stats()`
> 给自相关稳健的 TE / IR。修复后 18 条金标准仍逐位一致,已有字段数值不变。
> 证据脚本 `scripts/eng_review_fixes.py`,账本 `eng_review_capw.csv / eng_review_hdec.csv /
> eng_vwap_autocorr.csv / eng_review_edge.csv`。
>
> **2026-10-10 第二轮修订(前缀 fixa,详见文末)**:`benchmark()` 新增 `rebalance`(默认 1,金标准逐位一致),
> k 日调仓的策略改为对**同频再平衡**的池等权 `benchmark(..., rebalance=k)` 算超额;§3.1"年化收益不受影响"的表述
> 收窄为只对单条序列成立;§7 的 vwap1 行改用同频基准(open10 表保留原值,附同频数字);§9 编号改正。
> 证据脚本 `scripts/fixa_benchmark_fix.py`,账本 `fixa_bench_compare.csv / fixa_fullpool_checks.csv / fixa_eng_cost_demo.csv /
> fixa_eng_excess_ir.csv`。

---

## 1. 改了什么,保证什么

| | |
|---|---|
| 抽出 `simulate()` | 旧 `_run_one_phase` 里"目标权重 -> 逐日成交"的循环原样抽出,目标可以是 T×N 数组或 callable |
| 新增 `run_weights()` | 相位平均逻辑与 `run_strategy` 逐字相同(共用 `_aggregate_phases`),callable 时每个相位是独立路径 |
| 新增 `build_weights()` | 打分 -> W,与旧 `_run_one_phase` 内部同法;`run_strategy` 现在只算一次 W 再跑 k 个相位(旧版每相位重算) |
| `run_strategy` 新可选参数 | `capital / cost_model / max_participation / participation_on_nav / record`,默认值下行为不变 |
| 新口径 `EXEC_MODES["vwap1"]` | T 收盘信号 -> T+1 全日 VWAP 成交,参与率按全日成交额 |
| 新模块 `impact.py` | `CostModel`:半价差 + 平方根冲击,`cost_coefficients()` 给优化器 |
| `_run_one_phase` | 保留旧签名(`run_barra_attribution_top20.py` 等直接调用) |
| 审查修复(2026-10-10) | `ctx` 分成"决策时点可得"与"按成交价计价"两组;目标 NaN = 保持不交易;负权重报错、噪声级负数截零;`ctx` 数组为副本;结果追加 `vol_adj / sharpe_adj / vr_lr / ret_ac1`;新增 `excess_stats / lr_variance_ratio / decision_window_amount` |

**逐位不变的证据**:改造前先在未改动的 `strategy.py`(sha256 `56d74351…e4b0`)上跑
`scripts/eng_make_golden.py`,存下 9 个配置 × {全期, 2024} = 18 条记录的全部标量与逐日收益
(COMP_v5 标准排雷 / top20 r5 / top50 r1 / V100 top20 / V100 tail1450 top50 / 单相位 top20 /
两个 benchmark / `backtest.run_standard`)。新代码先在 `_strategy_new.py` 开发、对照这 18 条
**逐日收益 max|Δ| = 0(逐位相等)**,再一次 `mv` 原子替换。`pytest` 以 1e-10 容差复核
(`tests/test_engine_regression.py::test_golden_bitwise`)。`backtest.py` 未改动。

---

## 2. 接口

### 2.1 签名

```python
from afactor import strategy as ST, impact as IM

ST.build_weights(score, p, version, scheme="top", n=20, exclude_frac=0.3, buffer=None,
                 date_mask=None, sign=1, universe_override=None) -> np.ndarray   # (T, N) float32

ST.run_weights(target, p, mode, capital, rebalance=5, avg_phases=True, date_mask=None,
               cost_model=None, max_participation=None, record=False,
               normalize=True, participation_on_nav=False, phases=None) -> dict

ST.simulate(target, p, mode, capital, rebalance=1, date_mask=None, phase=0, cost_model=None,
            max_participation=None, record=False, normalize=True,
            participation_on_nav=False) -> dict                                 # 单相位

ST.run_strategy(score, p, version, mode, scheme="top", n=20, rebalance=1, exclude_frac=0.3,
                buffer=None, date_mask=None, sign=1, avg_phases=True, universe_override=None,
                capital=None, cost_model=None, max_participation=None,
                participation_on_nav=False, record=False) -> dict

ST.save_record(res, prefix) -> dict      # cache/_tmp/<prefix>_{holdings,daily,daily_by_phase}.parquet

ST.benchmark(p, version, mode, date_mask=None, rebalance=1, universe_override=None) -> dict
                                         # 池等权基准;rebalance=k(第二轮修订新增)= 同频再平衡、k 相位平均,
                                         # k 日调仓的策略请对它算超额;默认 1 与改造前逐位一致

# 2026-10-10 审查后新增
ST.excess_stats(rets, bench_rets, bandwidth=20, t_bandwidth=5) -> dict(n_days, excess_arith, te, ir,
                te_adj, ir_adj, vr_lr, excess_ac1, t_naive, t_nw)   # 两条日收益按索引取交集
ST.lr_variance_ratio(r, bandwidth=20) -> float   # Newey-West(Bartlett)长期方差 / 日方差
ST.decision_window_amount(p, mode, window=20, min_periods=10) -> ndarray (T, N)
                                         # 行 te = 截至 te-1 的 20 日窗口成交额中位数(ctx['cap_w'] 的原料)
# simulate / run_weights / run_strategy / benchmark 的返回值另有:
#   ret_ac1(日收益一阶自相关)、vr_lr(NW 带宽 20 的方差比)、vol_adj = vol·sqrt(vr_lr)、sharpe_adj = sharpe/sqrt(vr_lr)

IM.CostModel(p, impact_y=1.0, spread_mult=1.0, window=20, min_periods=10, tick=0.01,
             spread_method="tick")       # spread_method ∈ {"tick", "ar_window", "ar_2day"}
cm.cost_coefficients(te, capital, nav=1.0) -> dict(lin_buy, lin_sell, c15, half_spread, sigma,
                                                   adv, stamp, commission, te)
cm.trade_cost(te, delta, capital, nav=1.0) -> dict(commission, stamp, spread, impact, total)
cm.extra_cost(te, abs_delta, capital, nav) -> (spread, impact)          # 引擎内部调用
cm.half_spread(te) / cm.sigma(te) / cm.adv(te) / cm.impact_coef(te, capital, nav)
cm.impact_rate(te, q_yuan) / cm.stamp(te) / cm.te_of(t, mode) / cm.describe()
```

`te` 是成交日在 `p.dates` 里的行号(也可传 Timestamp);信号日 t 对应 `te = t + EXEC_MODES[mode]["lag"]`。
所有成本都以**占组合净值的比例**计。`CostModel` 必须用主口径 `Panel()` 构建(尾盘口径同样适用;
传 TailPanel 会报错,因为它的原料已滞后一天,会被重复滞后)。估计面板缓存在该 Panel 对象上,
同一面板上建多个 CostModel(不同 Y / spread_mult)共享同一份 σ / ADV / 价差。

### 2.2 target 的两种形式

- **ndarray** `W (T × N)`:行对齐 `p.dates`、列对齐 `p.codes`,只在调仓日使用;`date_mask`
  之外的行按 0 处理(与 `run_strategy` 把池子按 mask 截断同义,所以传不传截断后的 W 结果相同)。
- **callable** `target_fn(t, h, ctx) -> ndarray(N,) | None`:只在 `date_mask` 内的调仓日被调用;
  返回 `None` = 今天不交易(持仓自然漂移)。`ctx` 里的数组都是副本,原地改不影响引擎。
  `ctx` 字段分两组(2026-10-10 审查修复后):

  | 组 | 字段 | 含义 |
  |---|---|---|
  | **决策时点可得**(决策规则只能用这一组) | `t, te, date, trade_date, phase, mode, capital, cost_model` | 信号日 / 成交日行号与日期等 |
  | | `h_dec` | 上一成交日 te-1 成交后、按 `px[te-1]` 计价的持仓(相对当时净值,含现金腿)。open10 下 `px[te-1]` = t 日开盘 TWAP,vwap1 下 = t 日全日 VWAP,尾盘口径 = t-1 日窗口价 |
  | | `nav_dec / aum_dec` | 按 `px[te-1]` 计价、扣完上一笔成本后的净值 / 资金 |
  | | `cap_w` | 参与率上限的**事前估计**:`max_participation × 截至 te-1 的 20 日窗口成交额中位数 / (capital [× nav_dec])`。引擎实际仍按 te 当天窗口成交额截量 |
  | | `budget_dec` | `1 - h_dec[stuck].sum()` |
  | | `can, stuck` | te 日可成交 / 卖不掉的残仓。含"te+1 有价格""一字板用 te 全日振幅判定"两条旧引擎约定(停牌一般事先公告,不是新增前视,但对集中持仓略有利) |
  | **按成交价 `px[te]` 计价**(te 当天才知道,vwap1 下要到 te 收盘) | 位置参数 `h` | 成交前、按 `px[te]` 漂移后的持仓 —— 引擎实际从这里成交 |
  | | `nav, aum` | 按 `px[te]` 计价的成交前净值(引擎算冲击 Q 与 `participation_on_nav` 分母用它) |
  | | `stuck_w, budget` | 按 `px[te]` 计价的残仓权重与可用预算(`normalize` 归一到它) |

  ⚠ **不得用 `h / nav` 做判断**(如换手带 `|h-w|<band`、按 `h` 算换手惩罚):vwap1 下等于用了 te 全日 VWAP。
  要表达"某只股票不交易",在目标里给 **NaN**(引擎把它保持在实际持仓 `h_i`),不要回传 `h` 的数值。
  `target_fn` 只允许用信号日 t 及以前的信息,由调用方保证。

引擎对返回的目标做与旧逻辑相同的处理:不可交易的置 0、卖不掉的残仓先从预算扣掉、
`normalize=True`(默认)时在可交易标的上归一到 `budget`;`normalize=False` 时按原值使用
(可以留现金),只在超预算时等比压缩。之后依次施加参与率上限、买单按剩余预算压缩。
审查修复后新增三条规则(ndarray 目标同样适用;无 NaN、无负数时走与改造前逐字相同的代码路径):
- **NaN = 保持**:可交易且目标为 NaN 的股票保持实际持仓;保持部分先占预算,其余目标在剩余预算里归一 / 压缩。
- **负权重**:任一目标 < -1e-6(`ST.NEG_TOL`)报 `ValueError`(引擎只做多头;不防的话虚卖照样计费,
  `normalize` 还会把负数计入分母、放大多头);-1e-6 ~ 0 的求解器噪声截零;inf 报错。
- callable 的返回形状必须是 `(N,)`。

### 2.3 资金口径(容量分析必读)

- `capital` 是**初始**资金(元)。冲击成本的成交金额 `Q = |Δw| × capital × 当前净值`,即 AUM 随净值滚动。
- 参与率上限默认 `max_participation × 窗口成交额 / capital`(旧口径,按初始资金,不随净值变)。
  净值上涨后实际参与率会按净值倍数超过 10%(record 的 `max_part` 列可直接看到)。
  **做容量曲线请传 `participation_on_nav=True`**,让上限按 `capital × 当前净值` 折算。
- 相位平均时每个相位都按**全额** `capital` 模拟(与旧引擎一致,相位平均是降噪手段,不是把资金切成 k 份);
  对冲击与参与率而言这是偏保守的。
- callable 的 `ctx['cap_w']` 是**事前估计**(截至 te-1 的 20 日窗口成交额中位数),引擎截量用 te 实际值。
  这个估计并不保守:2021-2025 V1000 池 te 实际 / 估计 中位 0.95,open10 / vwap1 下分别有 53.1% / 54.1%
  的股票日实际低于估计、14.7% / 6.9% 低于估计的一半(`eng_review_capw.csv` A4)。按 `cap_w` 下单仍会被截量,
  看 record 的 `part_cut / unfilled`。容量曲线与指增优化器**只能用 `cap_w`(或自己的事前估计)做约束**。

### 2.4 record=True 返回的明细

`res["record"] = dict(daily, daily_by_phase, holdings, meta)`:

| 字段 | 含义 |
|---|---|
| `holdings` | DataFrame[信号日 × 股票],成交后持仓 newh 的相位平均(= 合并组合持仓),float32 |
| `daily.ret / gross_ret` | 净收益 / 扣成本前收益(`ret = gross_ret - cost_total`) |
| `daily.trade_value / buy_value / sell_value` | 成交金额(元)= |Δw| 之和 × capital × 成交前净值 |
| `daily.cost_total / cost_commission / cost_stamp / cost_spread / cost_impact` | 成本分解(占净值比例) |
| `daily.exposure / cash / tgt_exposure / unfilled` | 仓位、现金(=1-仓位)、调仓日目标仓位、调仓日"目标-实际"仓位差(未成交导致的现金) |
| `daily.desired_turnover / part_cut / part_bound_frac / n_part_bound / max_part` | 调仓意图、被参与率截掉的 |Δw|、截掉占意图比例、触上限股票数、实际最大参与率(按 capital×净值折算) |
| `daily.stuck_w / n_hold / n_rebalance_phases` | 卖不掉的残仓权重、持仓数、当天调仓的相位数 |
| `daily_by_phase` | 同上字段逐相位明细(含 `phase`、布尔 `rebalance`);只看调仓日用 `.query("rebalance")` |

多相位时 `daily` 是 k 个相位的均值(与收益的相位平均同口径)。

### 2.5 调用示例

```python
import numpy as np, pandas as pd
from afactor.panel import Panel
from afactor import strategy as ST, impact as IM

p = Panel()
score = pd.read_parquet("cache/_tmp/comp_v5_score_open10.parquet")
m_test = np.asarray((p.dates.year >= 2024))                      # 检验窗 2024-2025

# (1) 旧方案 + 冲击成本,1 亿资金,参与率按 AUM
cm = IM.CostModel(p, impact_y=1.0)                               # Y=1、一跳价差
res = ST.run_strategy(score, p, "V1000", "open10", scheme="exclude", buffer=(0.20, 0.35),
                      rebalance=5, capital=1e8, cost_model=cm, participation_on_nav=True,
                      record=True, date_mask=m_test)
res["record"]["daily"][["cost_spread", "cost_impact", "unfilled"]].mean()

# (2) 直接喂权重(如外部优化器离线算好的 W)
W = ST.build_weights(score, p, "V1000", "top", n=20)
res = ST.run_weights(W, p, "vwap1", 1e7, rebalance=5, cost_model=cm)

# (3) 路径依赖:优化器用**决策口径**持仓 h_dec 与同一个成本模型算换手成本
def target_fn(t, h, ctx):                                        # 位置参数 h 是 te 价,决策里不要用
    hd = ctx["h_dec"]
    co = ctx["cost_model"].cost_coefficients(ctx["te"], ctx["capital"], ctx["nav_dec"])
    # 目标: max  alpha_t @ w - Σ lin*|w-hd| - Σ c15*|w-hd|^1.5   s.t. 0 <= w - hd <= cap_w(买)、...
    # (cvxpy 可直接写 power(abs(w-hd), 1.5);cap_w 是事前估计,引擎仍按实际成交额截量)
    w = my_optimizer(t, hd, co, ctx["can"], ctx["budget_dec"], ctx["cap_w"])   # 只用 t 及以前的信息
    w = np.clip(w, 0, None)                                      # 求解器的 -1e-9 级负数也可交给引擎截零
    w[no_trade_mask] = np.nan                                    # 可选:不想动的股票给 NaN(= 保持实际持仓)
    return w                                                     # 或 None = 今天不交易
res = ST.run_weights(target_fn, p, "open10", 1e8, rebalance=5, cost_model=cm,
                     normalize=False, participation_on_nav=True, date_mask=m_test)
ST.excess_stats(res["rets"], ST.benchmark(p, "V1000", "open10", date_mask=m_test, rebalance=5)["rets"])
                                                                 # TE / IR(含 *_adj);5 日调仓对同频基准(第二轮修订)
```

返回的 `w` 是相对 te 成交前净值的目标。用 `h_dec` 算换手惩罚意味着优化器看到的"当前持仓"比实际
晚一天漂移;没有被 NaN 标为保持的股票,引擎会把 te 当天的漂移一起调回去(一天漂移量级:
2024 年 Top50 日调仓下 `|h - h_dec|` 单股最大约 0.4~0.5pp 权重,`eng_review_hdec.csv` B1)。

---

## 3. 新执行口径 vwap1

原料核实(`build_daily._day_features`,2026-10-10 实测;数字出自 `research/ledger/eng_vwap_checks.csv`):

- `vwap = Σ_bar amt / Σ_bar (amt / mid)`,`mid = (open_adj + close_adj)/2`:全日 241 根 bar
  (含 09:30 开盘与 15:00 收盘集合竞价)的成交额加权 bar 中价。原始数据没有成交量,"股数"是
  amt/mid 的近似,所以是**近似 VWAP**;与 `twap_open10` 同源(同一套前复权分钟 open/close),
  复权口径一致,收益率无前视。vwap/twap_open10 中位 1.000,1%/99% 分位 0.965/1.052;
  vwap 日收益 |r|>35% 的股票日只占 3.9e-6(除权日不跳,复权口径一致)。
- ~~"648 万个股票日 vwap 都落在 [low_px, high_px] 内"作为原料核实证据~~(审查 P2 撤回):这是**按构造恒成立**的
  —— vwap 是各 bar 中价的成交额加权调和平均,`high_px / low_px` 是同一套 bar 的 max/min(open, close)
  (`scripts/build_hilo.py`),每个 mid 都在区间内,加权平均也必然在。测试里保留这条断言,只用来防原料文件错位。
- **外部佐证**(审查后新增):新浪日线有真实成交量(`daily_basic` 换手率 = 成交量 / 流通股本 × 100,
  见 `data_8:21/DATA_ACQUISITION.md`),于是真实 VWAP(未复权)= 全日成交额 / (换手率/100 × 流通股本)。
  复权因子在同日比值里约掉,比较 `vwap/close_px`(本项目)与 `真实VWAP/未复权收盘`:
  2020-2025 共 645 万个股票日,偏离中位 1.2bp,90% / 99% 分位 4.0 / 11.9bp,98.6% 在 10bp 以内、
  99.96% 在 50bp 以内;V1000 池 2021-2025(514 万个股票日)几乎相同(中位 1.2bp、99% 分位 11.7bp)。
  对照:`twap_open10` 与真实 VWAP 的偏离中位 72.5bp。所以"amt/mid 近似股数"的误差在 1bp 量级,
  vwap 字段就是全日 VWAP(只读 2025-12-31 及以前的行)。
- `amount = Σ_bar amt`,全日成交额,单位**元**(600519 2024 年日中位 45.6 亿元)。
- 持有收益 `vwap[te+1]/vwap[te]-1`,lag=1,参与率上限 = 10% × 全日 amount;一字板不可成交规则不变。
  已有口径的 `exec_prices` 返回对象与改造前完全相同(测试断言)。

### 3.1 ⚠ vwap1 的日收益有平均价自相关:日频二阶矩(波动 / Sharpe / TE / IR)有偏(审查 P1)

全日 VWAP 是一天价格的均值(Working 1960 平均价效应),相邻两日的 VWAP 收益共享半天的价格信息,
所以日收益**正自相关**。**单条序列**的年化收益不受影响(VWAP 到 VWAP 的收益可以连乘还原),偏的是二阶矩;
但**再平衡频率不同的两个组合之间的超额**(一阶矩)会被改变(第二轮修订更正,原文"年化收益不受影响"只对单条序列成立):
每天重置等权的池基准比 5 日再平衡的同频池等权 2021-2025 低 2.09pp/年(vwap1,
NW t 14.1;open10 0.50pp/年,t 3.6,`fixa_bench_compare.csv` 的 `d_pool_r5_minus_d1`;tail1450 0.50pp/年,t 3.7,
尾盘研究脚本实际用的 TailPanel 上是 0.60pp/年,`fixa_r2_mech.csv`),5 日调仓的策略对它的超额被抬高同样幅度。
k 日调仓的策略请对 `benchmark(..., rebalance=k)` 算超额(§8 判据 ④b)。

**机制(复核更正)**:两个等权组合的市场暴露相同,差别来自**个股相对池均值收益的截面序列相关**,不是池等权序列本身的自相关。
相对收益有正的滞后相关时,"每天卖相对涨的、买相对跌的回到等权"系统性吃亏。2021-2025 滞后 1 期截面相关
vwap1 **0.175**、open10 0.037、tail1450 0.036(逐日截面相关的均值,`fixa_r2_mech.csv` 的 `xs_corr_l1`)。
一阶展开:5 日漂移组合相位平均后每天比等权多 (1/5)Σ_{l=1..4}(5−l)·γ_l(γ_l = 截面相对收益滞后 l 期的截面协方差),
预测的算术年化差 vwap1 1.78 / open10 0.41 / tail1450 0.42pp,实测 1.90 / 0.44 / 0.44pp,解释了 93~96%
(分窗口 82~105%,`pred_gap_arith` / `pred_over_actual`)。下表"池等权基准 日收益一阶自相关"是二阶矩偏差的证据,
对这段一阶矩差只是旁证:open10 2021-2023 池序列一阶自相关只有 0.021,同频差仍有 +0.38pp/年。

2021-2025、V1000 池(`research/ledger/eng_vwap_autocorr.csv`;表中第一轮的 COMP_v5 各行都是 2021-2025 **窗口掩码冷启动**、
对**每日重置**池基准算的):

| | vwap1 | open10 | tail1450 |
|---|---|---|---|
| 池等权基准 日收益一阶自相关 | **0.308** | 0.068 | 0.096 |
| 个股日收益一阶自相关(中位,4894 只) | **0.203**(vwap) | 0.041(twap_open10) | 0.046(twap_tail8);close 0.033 |
| 基准 5 日 / 20 日重叠方差比 | 1.499 / 1.392 | 1.129 / 1.000 | 1.156 / 1.062 |
| 基准年化波动:日频 std×√242 | **19.76%** | 23.74% | 22.64% |
| 基准年化波动:NW 带宽 20(`vol_adj`) | 23.27% | 23.68% | 23.28% |
| 基准年化波动:不重叠 20 日和 | 22.39% | 22.71% | 22.48% |
| COMP_v5 标准排雷(旧成本)Sharpe 日频 → `sharpe_adj` | **0.938 → 0.808** | 0.809 → 0.817 | 0.858 → 0.846 |
| 同上 超额 TE 日频 → NW20 | 2.54% → 3.15% | 2.93% → 3.05% | 2.89% → 3.03% |
| 同上 超额 IR 日频 → NW20 | **3.09 → 2.48** | 2.03 → 1.95 | 2.22 → 2.11 |
| 同上 超额日收益一阶自相关 | 0.277 | 0.043 | 0.040 |
| **(复核后主口径)同上,对同频(5 日)池基准,2020H2 预热的连续路径切 2021-2025**:超额几何年化 / TE 日频 → NW20 / IR 日频 → NW20 | **+6.39pp** / 2.43% → 2.83% / **2.26 → 1.94** | +6.09pp / 2.83% → 2.83% / 1.82 → 1.82 | +6.65pp / 2.85% → 2.84% / 1.99 → 1.99 |
| (第二轮修订原数,含冷启动伪超额)同上,但 2021-2025 窗口掩码冷启动 | +6.91pp / 2.50% → 3.10% / 2.38 → 1.91 | +6.50pp / 2.89% → 3.03% / 1.91 → 1.82 | +7.00pp / 2.87% → 2.99% / 2.07 → 1.99 |

(复核新增两行出自 `research/ledger/fixa_r2_eng_warm.csv`,`plan=exclude_std`、`cost_model=legacy`、`start=warm_2020H2 / cold_mask`;
冷启动行与 `fixa_eng_excess_ir.csv` 逐位相同,tail1450 与第一轮同法:研究期 Panel、同一打分。)

分窗口(同表 CSV,**对每日重置池基准、窗口掩码冷启动**,第一轮原数):vwap1 排雷超额 IR 2021-2023 为 3.09 → 2.31、2024-2025 为 3.27 → 2.82;
open10 为 1.97 → 1.78、2.27 → 2.31。
分窗口(复核后主口径:**对同频池基准、2020H2 预热**,`fixa_r2_eng_warm.csv`):vwap1 超额 2021-2023 +5.39pp、IR 2.17 → 1.82,
2024-2025 +8.10pp、IR 2.39 → 2.14;open10 +5.08pp、1.71 → 1.68,+7.82pp、1.97 → 2.04;tail1450 +5.46pp、1.76 → 1.79,+8.68pp、2.33 → 2.31。
带宽敏感性(vwap1 基准方差比):NW 5 / 10 / 20 / 40 = 1.495 / 1.393 / 1.386 / 1.256。

读法与规则:
- vwap1 的日频波动低估约 15%(19.76% vs 23.27%;按不重叠 20 日和是 12%),IR 日频口径高估:
  **预热连续路径上约 16%**(对同频基准 2.26 vs 1.94;对每日重置基准 2.99 vs 2.57),第一轮的"约 25%"(3.09 vs 2.48)是窗口掩码
  冷启动下的数,冷启动那几天的相位空仓让超额序列多了一段自相关,NW 口径的 TE 被抬高(2.54% → 3.15%,预热路径 2.47% → 2.87%)。
  对同频基准排雷的超额与 IR 都更低:超额 +9.00 → +6.39pp(预热、同频;冷启动同频 +6.91pp),IR_adj 2.48 → 1.94
  (`fixa_r2_eng_warm.csv`)—— 自相关同时影响二阶矩(IR 分母)与跨频率比较的一阶矩(IR 分子:截面相对收益的序列相关)。
  open10 / tail1450 全期的日频口径与 NW20 口径差 ≤5%,分窗口最多约 10%(open10 2021-2023 冷启动排雷 IR 1.97 → 1.78;预热同频 1.71 → 1.68)。
- **k 日调仓策略的窗口超额要用预热的连续路径切片**(复核新增):窗口掩码在窗口首日冷启动,相位 k 前 k 个信号日空仓,
  而同频基准从面板起点预热、始终满仓。旧成本排雷对同频基准的冷启动伪超额 vwap1 +0.52 / +0.80 / +0.93pp、open10 +0.41 / +0.63 / +0.91pp、
  tail1450 +0.34 / +0.53 / +0.91pp(2021-2025 / 2021-2023 / 2024-2025,`cold_minus_warm_excess_same`),和再平衡频率差同量级、同方向。
  它几乎全出在窗口首 5 个信号日(相位空仓期):冷启动 − 预热的日收益差累计,首 5 日占全窗口的 96~100%,同期同频池基准累计
  −3.9% ~ −5.7%(2021 年初、2024 年初小盘下跌,`cold_minus_warm_cum_first5_pp / _total_pp`、`bench_same_cum_first5_pct`)。
  容量报告一开始就用 2020H2 预热,不受影响。
- 修复:引擎结果统一追加 `ret_ac1 / vr_lr / vol_adj / sharpe_adj`(Newey-West Bartlett 核、带宽 20 个交易日,
  约等于"按月度持有期折算",对正自相关的 vwap1 和中期略负自相关的 open10 都可比);
  新增 `ST.excess_stats(rets, bench_rets)` 同时给日频与 NW20 两种 TE / IR,以及 NW(带宽 5)t 值。
  原有 `vol / sharpe` 字段数值不变(金标准逐位一致)。
- **下游规则**:vwap1 下的波动、Sharpe、TE、IR 一律用 `*_adj`;不得拿 vwap1 的日频 Sharpe / IR 与 open10 横比
  (换成 `*_adj` 后两口径可比:排雷 Sharpe 0.808 vs 0.817)。`eng_cost_demo.csv` 已追加 `sharpe_adj / te_adj / ir_adj` 列。

---

## 4. 成本模型

单只股票单边成本(占净值比例):

```
cost_i = |Δw_i| × (佣金 1bp + 半价差 hs_i [+ 卖出印花税])  +  c_i × |Δw_i|^1.5
c_i    = Y × σ_i × sqrt(capital × nav / ADV_i)
⇔ 冲击成本率 = Y × σ_i × sqrt(Q / ADV_i),Q = |Δw_i| × capital × nav(元)
```

| 参数 | 取值 | 来源 / 说明 |
|---|---|---|
| Y | **1.0 基准**,0.5 / 1.5 敏感性(预注册,`IM.PREREG_Y`) | Almgren et al. (2005) 美股大单 η≈0.3~1;Tóth et al. (2011) 平方根定律 Y=O(1);Frazzini-Israel-Moskowitz (2018) 实盘冲击低于多数学术模型。**文献量级,未用真实成交数据校准** |
| σ_i | 20 日日收益标准差(close_px 收盘对收盘,前复权),min_periods=10 | 截至 te-1 |
| ADV_i | 20 日日均**全日**成交额(元),min_periods=10 | 与决策池 adv20 同一公式;截至 te-1 |
| hs_i | 一跳价差 / 2 = 0.005 元 / 未复权收盘价(**默认,下界**);`ar_window` 为上界 | 见第 5 节 |
| 佣金 / 印花税 | 1bp / 卖出 10bp(2023-08-28 前)、5bp | 与 config 同一常数 |

- **时序**:面板行 te 存截至 te-1 的估计(统一 `.shift(1)`)。`test_cost_estimates_no_lookahead`
  把第 k 行之后的全部原料抹成 NaN,第 k+1 行的 σ / ADV / 价差逐位不变。
- **与旧引擎的关系**:`cost_model=None` 时成本表达式与改造前同一行代码;给定 `CostModel` 时在其上
  **加** `extra_cost`。`CostModel(impact_y=0, spread_mult=0)` 与旧引擎逐位一致(`test_cost_model_zero_is_legacy`,
  12 个金标准配置)。
- **优化器与引擎同一模型**:`Σ lin_buy·buy + Σ lin_sell·sell + Σ c15·|Δw|^1.5` 与 `trade_cost` 相对误差 <1e-12
  (`test_cost_coefficients_match_engine`)。c15 项是 |Δw| 的 3/2 次幂,凸,cvxpy 可直接用 `power(abs(.), 1.5)`。
- **缺失兜底**(行内,不用未来数据):σ -> 当日截面中位数;ADV 缺失或 ≤0 -> 当日截面正值 5% 分位
  (偏保守 = 高冲击);价差 -> 当日截面中位数;未复权价缺失(约 0.6% 股票日)-> 用前复权收盘价
  (前复权价 ≤ 未复权价,下限偏高,保守)。

---

## 5. 价差:Abdi-Ranaldo 在本数据上不可识别,默认用一跳下限

任务书原定"Abdi-Ranaldo (2017) 日频估计量、20 日滚动均值、负值按文献处理、下限为一跳"。
实现了,但**有效性诊断不通过**(判据只看估计量本身,与任何策略收益无关),所以默认改成一跳价差,
逐股 AR(`spread_method="ar_window"`)保留为悲观上界。这是对任务书的偏离,理由如下。

AR 的期望值是价差²:`E[4(c_t-η_t)(c_t-η_{t+1})] = s²`。诊断(2021-2025,V1000 决策池):

1. **合并后均值为负**:按市值或 ADV 十分位,每天桶内两日乘积的截面均值、再对时间序列做 Newey-West
   (带宽 5),20 个桶的 t 值全部为负,在 -2.10 ~ -4.69 之间;2021-2023 与 2024-2025 两个子窗也全部为负
   (市值桶 -1.55 ~ -4.26、-1.68 ~ -2.89)。合并后"负值截零"就是 0,只剩一跳下限。
   (`eng_spread_by_size.csv` 的 `pooled_prod_t`)
2. **逐股估计是截断噪声**:逐股 20 日估计 49.7% 被截成 0(60 日 54.1%、120 日 59.9%,窗口越长越多);
   与 -ln ADV 的截面秩相关日均 -0.016、与一跳价差 -0.045(方向都是反的)。(`eng_spread_ar_diagnostics.csv`)
3. **桶内横截面合并也救不回来**:市场共同成分在截面上不分散,ADV 第 1/5/10 十分位的 20 日桶均值
   分别有 39.4% / 29.2% / 18.2% 的天为正;250 日也还有 10.9% / 6.0% / 0.0%。(`eng_spread_ar_diagnostics.csv`)

可能原因:close_px 是 15:00 收盘集合竞价价(没有买卖价反弹,违反 AR 的识别假设)、high/low 由分钟
open/close 近似(比真实极值窄)、日频收益有非鞅成分。

**按市值十分位的半价差(bp,2021-2025 V1000 池,`eng_spread_by_size.csv`)**

> 2026-10-10 注:本表按**修复前**的 `ND.total_mv` 分档(见 `TOTAL_MV_FIX.md`),分档边界会略有变化;本节结论(AR 不可识别、默认一跳半价差)只看估计量本身,不依赖市值。

| 市值十分位 | 总市值中位(亿元) | 未复权价中位(元) | 一跳半价差 中位 | AR 逐股 中位 | AR 逐股 均值 | AR two-day 中位 | AR 被截零比例 | 合并乘积 NW t |
|---|---|---|---|---|---|---|---|---|
| 1(最小) | 22.3 | 9.20 | 5.43 | 17.52 | 33.95 | 46.13 | 50.3% | -2.99 |
| 2 | 31.7 | 9.96 | 5.02 | 17.01 | 33.74 | 45.87 | 49.1% | -3.53 |
| 3 | 40.3 | 10.20 | 4.90 | 17.14 | 33.75 | 45.97 | 48.8% | -3.49 |
| 4 | 50.4 | 10.80 | 4.63 | 15.88 | 32.83 | 45.85 | 49.9% | -4.07 |
| 5 | 63.5 | 12.13 | 4.12 | 15.33 | 32.54 | 45.90 | 50.0% | -4.03 |
| 6 | 82.7 | 12.91 | 3.87 | 14.97 | 32.00 | 45.92 | 50.2% | -4.16 |
| 7 | 113.2 | 13.68 | 3.65 | 13.58 | 30.87 | 44.77 | 50.2% | -3.83 |
| 8 | 167.8 | 16.12 | 3.10 | 13.37 | 30.19 | 43.96 | 50.1% | -3.26 |
| 9 | 290.4 | 17.69 | 2.83 | 14.12 | 29.50 | 42.48 | 49.7% | -2.67 |
| 10(最大) | 891.6 | 21.99 | 2.27 | 13.23 | 27.33 | 38.96 | 48.7% | -2.10 |

读法:一跳半价差随市值单调下降(5.4 -> 2.3bp),只因为大市值股价高;它是**下界**
(实盘 A 股小盘常有 1~2 跳价差,TWAP 分批成交只有主动成交部分付价差)。AR 逐股的中位 13~18bp、
均值 27~34bp 与市值基本无关,two-day 版本 39~46bp 更高,都是截断噪声的产物。
**建议下游把价差当区间报告**:`spread_method="tick"`(下界)、`spread_mult=2`(两跳)、`"ar_window"`(上界)。
按 ADV 十分位的同一张表见 CSV(`bucket_type=adv`),结论相同。

---

## 6. 冲击示例:1000 万 / 1 亿 / 10 亿资金买入 1% 权重

成交金额 Q = 1% × 资金 = 10 万 / 100 万 / 1000 万元。2021-2025 每个成交日、信号日 V1000 决策池内
按 ADV 十分位取股票日中位数(`eng_impact_examples.csv`)。冲击 = Y·σ·sqrt(Q/ADV),单位 bp。

| ADV 十分位 | ADV 中位(万元) | σ 中位 | 1000 万:冲击 Y=1 | 1 亿:冲击 Y=1 | 10 亿:冲击 Y=1 | 10 亿:Q/ADV | 1 亿:开盘窗口参与率上限会截量的比例 |
|---|---|---|---|---|---|---|---|
| 1(最差) | 2,621 | 1.8% | 11.2 | 35.6 | 112.5 | 38.2% | 94.0% |
| 2 | 3,773 | 2.0% | 10.1 | 31.8 | 100.5 | 26.5% | 88.0% |
| 3 | 5,092 | 2.1% | 9.3 | 29.4 | 93.1 | 19.6% | 80.3% |
| 4 | 6,749 | 2.3% | 8.7 | 27.5 | 86.9 | 14.8% | 70.5% |
| 5 | 8,915 | 2.5% | 8.1 | 25.5 | 80.7 | 11.2% | 58.1% |
| 6 | 11,915 | 2.6% | 7.5 | 23.6 | 74.5 | 8.4% | 43.4% |
| 7 | 16,242 | 2.8% | 6.8 | 21.5 | 67.9 | 6.2% | 27.4% |
| 8 | 23,436 | 2.9% | 6.0 | 19.1 | 60.3 | 4.3% | 12.8% |
| 9 | 37,349 | 3.2% | 5.1 | 16.1 | 51.0 | 2.7% | 3.2% |
| 10(最好) | 88,960 | 3.4% | 3.4 | 10.8 | 34.1 | 1.1% | 0.2% |
| 全池 | 10,760 | 2.5% | 7.4 | 23.4 | 74.0 | 9.3% | 47.8% |

- Y=0.5 / 1.5 时冲击按比例缩放(全池中位:1000 万 3.7 / 11.1bp,1 亿 11.7 / 35.1bp,10 亿 37.0 / 111.0bp)。
- 加上佣金 1bp 与一跳半价差,1000 万买入单边总成本全池中位 13.6bp,1 亿 30.1bp,10 亿 80.8bp(`buy_cost_bp_Y1_tick`)。
- 资金每扩大 10 倍,冲击放大 sqrt(10)=3.16 倍,这是平方根定律的直接推论。
- **参与率比冲击更早成为约束**:1 亿资金在开盘 10 分钟窗口买 1%,全池 47.8% 的股票日会被 10%
  参与率上限截量(`share_cap_bind_open10`);同一笔单在全日 VWAP 口径只有 0.3% 被截(`share_cap_bind_vwap1`)。
  10 亿资金时开盘窗口 92.5% 被截、全日 VWAP 也有 51.4% 被截。
- ADV 越低的十分位 σ 反而越低(1.8% vs 3.4%):池子里低成交额的股票也更不活跃,部分抵消了 sqrt(1/ADV)。

---

## 7. 接口演示:COMP_v5 两个方案在 1000 万下的成本(⚠ 信号样本内)

COMP_v5 成员在 2020-2025 研究期内选出,下面的收益**不是信号的样本外能力**,只用来说明成本层的量级。
V1000、open10、5 相位平均,基准为池等权(`eng_cost_demo.csv`;超额 = 策略年化 - 基准年化,
成本年化 = 日均成本 × 242;价差 / 冲击分项年化为 CSV 的 `spread_ann_pp / impact_ann_pp`)。

| 方案 | 成本模型 | 2021-2023 超额 | 2024-2025 超额 | 2021-2025 成本年化 | 其中 半价差 / 冲击(pp/年,2021-2025) |
|---|---|---|---|---|---|
| 排雷 20/35 r5 | 旧(1bp+印花税) | +6.09pp | +9.44pp | 0.46pp | 0 / 0 |
| | Y=0.5,一跳 | +5.41pp | +8.66pp | 1.07pp | 0.52 / 0.09 |
| | **Y=1.0,一跳(基准)** | **+5.31pp** | **+8.56pp** | **1.15pp** | **0.52 / 0.17** |
| | Y=1.5,一跳 | +5.22pp | +8.46pp | 1.24pp | 0.52 / 0.26 |
| | Y=1.0,两跳 | +4.74pp | +7.88pp | 1.67pp | 1.04 / 0.17 |
| | Y=1.0,AR 逐股(上界) | +2.31pp | +5.55pp | 3.75pp | 3.12 / 0.17 |
| Top20 r5 | 旧(1bp+印花税) | -0.99pp | +8.10pp | 2.59pp | 0 / 0 |
| | Y=0.5,一跳 | -7.08pp | -1.50pp | 9.31pp | 2.90 / 3.82 |
| | **Y=1.0,一跳(基准)** | **-10.03pp** | **-6.88pp** | **12.80pp** | **2.90 / 7.30** |
| | Y=1.5,一跳 | -12.77pp | -11.80pp | 16.01pp | 2.90 / 10.52 |
| | Y=1.0,两跳 | -12.57pp | -10.02pp | 15.45pp | 5.80 / 7.05 |
| | Y=1.0,AR 逐股(上界) | -16.35pp | -17.78pp | 20.74pp | 11.47 / 6.66 |

(第二轮修订注:上表超额对每日重置的池等权、且每个窗口首日冷启动,保留 `eng_cost_demo.csv` 原值作接口演示,**超额读下面的补表**。
两个方案都是 5 日调仓,按 §3.1 应对同频池基准算:open10 各行同幅下移 2021-2023 0.38pp、2024-2025 0.71pp(`fixa_eng_cost_demo.csv`);
复核又发现窗口掩码的冷启动伪超额(相位 k 前 k 天空仓,同频基准却始终满仓),所以补表改用 2020H2 预热的连续路径切片。)

**§7 补表(复核后主口径):对同频(5 日)池等权、2020-07-01 起一条连续路径切窗口**(`research/ledger/fixa_r2_eng_warm.csv`,
`start=warm_2020H2`;"冷启动原值"= `fixa_eng_cost_demo.csv` 的同频超额,即第二轮修订原先报告的数;成本模型、资金、方案与上表相同)

| 口径 | 方案 | 成本模型 | 2021-2023 超额 | 2024-2025 超额 | 2021-2025 超额 | 2021-2025 成本年化 | 冷启动原值(2021-2023 / 2024-2025) | 冷启动 − 预热 | 预热路径进入 2024 的净值 |
|---|---|---|---|---|---|---|---|---|---|
| open10 | 排雷 20/35 r5 | 旧(1bp+印花税) | +5.08pp | +7.82pp | +6.09pp | 0.46pp | +5.71 / +8.73 | +0.63 / +0.91 | 1.40 |
| open10 | 排雷 20/35 r5 | Y=0.5,一跳 | +4.42pp | +7.06pp | +5.39pp | 1.06pp | +5.03 / +7.94 | +0.61 / +0.88 | 1.37 |
| open10 | 排雷 20/35 r5 | **Y=1.0,一跳(基准)** | **+4.32pp** | **+6.95pp** | **+5.29pp** | **1.14pp** | +4.93 / +7.85 | +0.61 / +0.90 | 1.37 |
| open10 | 排雷 20/35 r5 | Y=1.5,一跳 | +4.22pp | +6.84pp | +5.19pp | 1.23pp | +4.83 / +7.75 | +0.61 / +0.91 | 1.36 |
| open10 | 排雷 20/35 r5 | Y=1.0,两跳 | +3.77pp | +6.31pp | +4.70pp | 1.65pp | +4.36 / +7.17 | +0.59 / +0.86 | 1.34 |
| open10 | 排雷 20/35 r5 | Y=1.0,AR 逐股(上界) | +1.41pp | +4.04pp | +2.37pp | 3.70pp | +1.93 / +4.84 | +0.52 / +0.80 | 1.24 |
| open10 | Top20 r5 | 旧(1bp+印花税) | -1.94pp | +6.52pp | +1.09pp | 2.61pp | -1.37 / +7.39 | +0.56 / +0.87 | 1.05 |
| open10 | Top20 r5 | Y=0.5,一跳 | -7.90pp | -2.58pp | -6.01pp | 9.20pp | -7.46 / -2.21 | +0.44 / +0.37 | 0.85 |
| open10 | Top20 r5 | **Y=1.0,一跳(基准)** | **-10.71pp** | **-7.06pp** | **-9.43pp** | **12.54pp** | -10.42 / -7.59 | +0.30 / -0.53 | 0.77 |
| open10 | Top20 r5 | Y=1.5,一跳 | -13.29pp | -10.82pp | -12.44pp | 15.57pp | -13.16 / -12.52 | +0.13 / -1.69 | 0.70 |
| open10 | Top20 r5 | Y=1.0,两跳 | -13.21pp | -9.84pp | -12.04pp | 15.16pp | -12.95 / -10.74 | +0.26 / -0.90 | 0.70 |
| open10 | Top20 r5 | Y=1.0,AR 逐股(上界) | -16.95pp | -17.15pp | -17.07pp | 20.40pp | -16.74 / -18.50 | +0.22 / -1.35 | 0.60 |
| vwap1 | 排雷 20/35 r5 | 旧(1bp+印花税) | +5.39pp | +8.10pp | +6.39pp | 0.46pp | +6.20 / +9.03 | +0.80 / +0.93 | 1.39 |
| vwap1 | 排雷 20/35 r5 | **Y=1.0,一跳(基准)** | **+4.64pp** | **+7.24pp** | **+5.60pp** | **1.14pp** | +5.42 / +8.15 | +0.78 / +0.91 | 1.36 |
| vwap1 | Top20 r5 | 旧(1bp+印花税) | +0.70pp | +10.96pp | +4.37pp | 3.36pp | +1.56 / +11.81 | +0.86 / +0.85 | 1.12 |
| vwap1 | Top20 r5 | **Y=1.0,一跳(基准)** | **-12.90pp** | **-7.33pp** | **-10.95pp** | **17.68pp** | -12.63 / -9.07 | +0.27 / -1.74 | 0.69 |

补表读法:
- **排雷各行的"冷启动 − 预热"就是冷启动伪超额**:2021-2023 +0.52~+0.80pp、2024-2025 +0.80~+0.93pp,各成本模型几乎相同
  (成本年化与冷启动版相差 ≤0.09pp,`cost_total_ann_pp` vs `cold_cost_ann_pp_eng_cost_demo`)。2021-01 与 2024-01 都是小盘下跌月,
  冷启动的相位空仓躲过了一部分跌幅。
- **Top20 计入冲击的各行混了两件事**:冷启动伪超额(旧成本行 +0.56 / +0.87pp),以及引擎默认"冲击按 capital × 当前净值"——
  预热路径在 2024 年初净值已跌到 0.60~0.85,实际规模小于 1000 万,冲击更低(Y=1 2024-2025 冲击 8.42 vs 冷启动 9.60pp/年,
  `cost_impact_ann_pp` vs `cold_impact_ann_pp_eng_cost_demo`),所以 2024-2025 的差为负。要比"同一规模"请看容量报告(恒定 AUM)。
- 读法不变:排雷在所有成本模型、两个窗口都为正;Top20 计入冲击后在两个口径、两个窗口都为负,Y=0.5 也为负。

读法:
- 排雷方案日换手约 0.02。基准设定(一跳半价差 + Y=1 平方根冲击)在 1000 万下比旧成本每年多扣约 0.7pp,
  **其中 3/4 是半价差(0.52pp)、冲击只有 0.17pp**;价差口径从一跳换成 AR 上界多扣 3.3pp/年 ——
  宽分散低换手的方案对成本模型不敏感,**两个窗口超额都保持正值**。(审查 P2 更正:原文把这 0.7pp 笼统写成"冲击")
- Top20 周频日换手约 0.11(年单边约 27 倍),基准设定下比旧成本每年多扣约 10.2pp(半价差 2.90pp、冲击 7.30pp),
  两个窗口超额都转负;
  即使 Y=0.5 也是负的。这说明 README 里"Top20 选优"的结论在计入冲击后**不成立于 1000 万资金**
  (信号样本内结论,COMP_v5;B×GBT 样本外打分需下游另测)。
- vwap1 口径(`mode=vwap1`,复核后主口径:对 vwap1 的**同频(5 日)池等权**、2020H2 预热连续路径切片,`fixa_r2_eng_warm.csv`):
  排雷方案旧成本 2021-2023 / 2024-2025 超额 +5.39 / +8.10pp,计入 Y=1 冲击后 +4.64 / +7.24pp;Top20 在 vwap1 下几乎不受
  参与率约束(仓位 1.000),旧成本超额 +0.70 / +10.96pp,计入冲击后 −12.90 / −7.33pp —— 不设窗口上限时单笔更大,冲击更高。
  (第二轮修订原先报告的同频冷启动数是 +6.20 / +9.03、+5.42 / +8.15、+1.56 / +11.81、−12.63 / −9.07pp,`fixa_eng_cost_demo.csv`,
  含冷启动伪超额 +0.78~+0.93pp;修订前对每日重置的池基准是 +8.10 / +11.42、+7.32 / +10.54、+3.47 / +14.20、−10.73 / −6.68pp,
  `eng_cost_demo.csv`,那个基准在 vwap1 下又被再平衡频率差压低 1.90 / 2.39pp/年。)
  vwap1 与 open10 的成交时点不同,超额不宜直接横比。**vwap1 的日频 Sharpe / IR 被平均价自相关高估**(见 3.1):
  排雷旧成本 2021-2025 Sharpe 0.938 → `sharpe_adj` 0.808;对同频基准、预热路径的 IR 2.26 → `ir_adj` 1.94
  (冷启动口径 2.38 → 1.91;修订前对每日重置基准、冷启动 3.09 → 2.48)。
  `eng_cost_demo.csv` 的 `ir / ir_adj` 列仍是对每日重置基准、冷启动,读 `fixa_r2_eng_warm.csv`。

---

## 8. 引擎回归判据(全部通过)

| 判据 | 结果 | 出处 |
|---|---|---|
| ① 金标准逐位一致 | 18 条记录逐日收益 max\|Δ\|=0 | `test_golden_bitwise` |
| ② exposure ≤ 1 | 含冲击模型、callable、normalize=False、vwap1、tail+10 亿 全部 ≤1 | `test_exposure_le_one` + 循环内硬断言 |
| ③ 对齐峰值 k=0 | 三口径 × 两方案 全部 k=0;全池等权 k=0 相关 1.000,k=±1 为 0.05~0.29 | `test_alignment_peak_at_zero`;`eng_engine_checks.csv`(`align_peak_k`) |
| ④ 全池等权超额 ≈ -成本 | 日调仓 vs 每日重置的池基准:残差(超额+成本)三口径两版本**逐年 0.10~0.91pp**、全期 0.27~0.63pp,全部为正 | `test_fullpool_excess_equals_minus_cost`;`eng_engine_checks.csv`(`fullpool_resid_pp`, score=const) |
| ④b 同频基准(第二轮修订新增) | 全池等权 rebalance=k vs `benchmark(rebalance=k)`,连续路径切片:k=5 open10 2021-2025 +0.23pp、逐年 +0.11~+0.34;vwap1 +0.05、逐年 −0.06~+0.12(k=1 同 ④)。反例:k=5 对每日重置基准 vwap1 +1.95pp、2024 年 +2.56pp,超出容差。另核对同频实现:k=1 特例与旧路径差 ≤ 8.4e-9、两调仓日之间 = 等权买入持有(1e-12)、date_mask 截断 = 全样本切片 | `test_fullpool_samefreq_excess_equals_minus_cost`、`test_samefreq_benchmark_implementation`;`fixa_fullpool_checks.csv` |
| ⑤ Y=0 且价差=0 == 旧引擎 | 12 个策略配置逐位相等 | `test_cost_model_zero_is_legacy` |
| ⑥ 冲击随资金单调不减 | 函数层严格递增;引擎层(参与率不设限)100 万→10 亿单调 | `test_impact_monotone_in_capital` |
| ⑦ run_weights(topn W) == run_strategy(top) | 逐位相等(含 mask、未截断 W、单相位) | `test_run_weights_matches_run_strategy` |
| ⑧ callable 固定 W == ndarray | 逐位相等(含成本模型、2024 mask) | `test_callable_fixed_matches_ndarray` |
| ⑨ backtest.py == strategy.py | 排雷单相位逐日逐位相等,年化差 0.000pp;top100 r1 年化差 <1e-12 | `test_backtest_matches_strategy`;`eng_engine_checks.csv` |
| ⑩ ctx 只含决策时点信息(审查后新增) | te 及以后成交价 / 成交额乘 U(0.5,2) 扰动后,`cap_w / h_dec / nav_dec / budget_dec / can / stuck` 逐位不变,`h / nav` 变化(对照);`cap_w` == 0.1×截至 te-1 的 20 日中位数 / capital | `test_ctx_is_decision_time_only`(open10、vwap1);`eng_review_capw.csv` A1/A2 |
| ⑪ `h_dec` 是精确回滚 | 与"h 按 px[te-1]/px[te] 回滚、含现金腿归一"的差 ≤1.7e-17(vwap1 / open10 / tail1450) | `test_hdec_is_exact_rollback`;`eng_review_hdec.csv` B1 |
| ⑫ 目标校验与 ctx 副本 | 负权重 / inf 报错;-1e-8 级负数截零后与干净目标逐位相等;原地改 `ctx['can']` 不影响结果 | `test_target_validation_and_ctx_copies`;`eng_review_edge.csv` |
| ⑬ NaN = 保持 | 全 NaN 与返回 None 逐位相等;部分 NaN 与"回传 h 数值"(不超预算时)逐位相等 | `test_nan_means_hold` |
| ⑭ 自相关稳健字段 | 白噪声 vr≈1、AR(1) ρ=0.3 时 vr∈(1.5, 1.96);`vol_adj / sharpe_adj / ir` 公式自洽 | `test_lr_fields_and_excess_stats` |

`pytest tests/test_engine_regression.py`:25 项全部通过(审查前 18 项 + 新增 7 项,`logs/eng_pytest.log`)。
第二轮修订后 28 项全部通过(新增 ④b 3 项,`logs/fixa_pytest.log`)。

判据 ④ 的两个注释:

- 残差全部为正、量级与 README 记录的 0.6~0.9pp 一致,是一字板可成交性差异(策略买不进 / 卖不掉一字板,
  持有中的连板因漂移超配),已定位非 bug;测试容差 [-0.5, +1.5]pp。
- **5 日调仓的组合必须对同频基准做这个判据**(第二轮修订):对每日重置的基准,vwap1 下 5 日调仓的全池等权残差
  是 +1.95pp(连续路径;2021-01-04 冷启动口径 +2.44pp,`fixa_fullpool_checks.csv`),这一段是再平衡频率差,不是可成交性残差。
- **必须用常数打分做这个判据**。用 COMP_v5 做时 2020 年残差是 +10.30pp(`score=COMP_v5(预热缺失)`),
  原因是 COMP_v5 在 **2020-02-06 之前没有打分**(`comp_v5_first_scored_day`),策略空仓,而池基准满仓
  经历了 2020-02-03 的暴跌。这不是引擎问题,但它影响所有用 COMP_v5 报**全期**数字的结论,见下一节。

### 8.1 顺带发现:COMP_v5 全期超额里约 1.4pp/年是 2020 年 1 月的空仓伪超额

标准排雷方案在 2020-01-02 ~ 2020-02-05 打分缺失、空仓(收益 0.00%),同期池基准 -7.57%,
累计伪超额 +7.57pp(`comp_v5_std_prewarmup_excess_pp`)。从首个有效打分日 2020-02-06 起算:
策略年化 19.15%、基准 12.61%,超额 **6.54pp**;全期口径是 18.88% / 10.97%,超额 7.90pp
(`comp_v5_std_ann_from_first_scored_day`)。2020 本就是预热年,正式窗口(2021-2023 / 2024-2025)
不受影响,但**引用全期数字时应改用 2020-02-06 起算,或明确标注**。

---

## 9. 已知局限与给下游的提示

1. **成本参数是文献量级,没有用真实成交数据校准**。Y 的真值可能在 0.5~1.5 之外;一跳价差是下界。
2. **平方根冲击用全日 ADV**。open10 口径实际在开盘 10 分钟窗口成交,窗口流动性只占全日一小部分,
   用全日 ADV **很可能低估**集中开盘成交的冲击;10% 窗口参与率上限是另一道闸。
3. 只算临时冲击,不建模永久冲击对后续交易的传递,也不建模买卖不对称。
4. 参与率上限默认按初始资金(旧口径);做容量曲线请传 `participation_on_nav=True`。
5. 相位平均时每个相位按全额资金模拟,对冲击与参与率偏保守。
6. `can` 里"te+1 有价格"是旧引擎约定;一字板判定用 te 全日振幅,开盘时点并不完全已知。两者都是沿用,
   对集中持仓略有利。
7. **callable 决策只能用"决策时点可得"那组 ctx 字段**(2.2 节):`h_dec / nav_dec / budget_dec / cap_w`;
   位置参数 `h`、`nav / aum / budget / stuck_w` 按 te 成交价计价,只供"保持不动"的数值表达,
   不动的股票请返回 NaN。`cap_w` 是事前估计、并不保守,引擎仍按实际成交额截量。
8. **vwap1 的二阶矩必须用 `*_adj`**(3.1 节):日频口径低估波动约 15%、高估 IR 约 16%(预热连续路径;窗口掩码冷启动下约 25%),
    不得与 open10 的日频 Sharpe / IR 横比。open10 / tail1450 全期差 ≤5%,做跨口径比较时也建议统一报 `*_adj`。
9. 数据不含已退市股(约 209 只),小盘低价股的价差 / 冲击估计样本偏向"活下来的"股票,所有收益偏高。
10. 用 COMP_v5 跑的任何组合结果都是**信号样本内**;B_gbt walk-forward 打分在 2023-2025 是样本外。
11. **k 日调仓的策略对同频基准算超额**(第二轮修订):`benchmark(p, version, mode, rebalance=k)`。个股相对收益有正的截面序列相关时,
    每天重置等权的基准系统性偏低(2021-2025):vwap1 约 2.1pp/年、open10 约 0.5pp/年、tail1450 约 0.5pp/年
    (研究期 Panel;尾盘研究实际用的 TailPanel 上 0.60pp/年,2020-2025 0.71pp/年、2024-2025 0.80pp/年,`fixa_r2_mech.csv`),
    5 日调仓策略对它的超额会被同幅抬高;同频基准仍不计成本、不加可成交约束,对一字板 / 停牌的可成交性残差还在
    (open10 约 0.2pp、vwap1 约 0.05pp)。
    **窗口超额还要用预热的连续路径切片**(复核新增):窗口首日冷启动(相位 k 前 k 天空仓)会再多出 0.3~0.9pp/年的伪超额
    (§3.1 读法、§7 补表),不论对哪个基准。
    **仍对每日重置基准、未改的报告与脚本**(复核盘点;本轮只改了 fixa 负责的 CAPACITY / ENGINE / CSI1000_PROXY / IE / IE-PEAD / PEAD / STRESS
    七份报告的相关段落):`run_barra_attribution.py`(BARRA_ATTRIBUTION;另被 SELECTION_ABILITY_AUDIT、RISK_MODEL、STRESS_TEST、
    CAPACITY §7.7 引用)、`run_barra_attribution_top20.py`(BARRA_TOP20)、`run_barra_attribution_topfrac.py` / `run_comp_split.py`
    (LONG_SIDE_TEST)、`run_ir_analysis.py` / `run_baselines.py`(IR_AND_NOISE_THRESHOLD)、`m5_bridge.py`(MINUTE5_BRIDGE)、
    `run_risk_model.py`(RISK_MODEL)、`run_index_proxy.py`(CSI1000_PROXY 第一轮表格)、`run_attribution.py`,以及因子层 / 模型层的
    `eval_scores.py`、`run_ml_models.py`、`run_final_strategy.py`、`run_featureset_ablation.py`、`run_ablation_paired.py`、
    `run_industry_layer*.py`、`run_gate_test.py`、`run_eventbt*.py`、`run_e1_theme_pool.py`、`run_e4_upper_bound.py`
    (多为 5 日调仓、tail1450 / open10)。这些报告"相对池等权"的超额水平普遍偏高约 0.5pp/年(同一基准下方案之间的比较不受影响)。
    已量化的一例:BARRA_ATTRIBUTION(open10,2020-2025)换同频基准后总超额 +5.32 → +4.89pp、流动性贡献 +5.20 → +5.13pp
    (占比 98% → 105%)、残差 alpha −0.31 → −1.22pp,NW t −0.48~−0.54 → −1.94~−2.15;tail1450 alpha +0.00 → −0.78pp
    (OLS 对超额线性,Δ 精确;`fixa_r2_barra_delta.csv`)。"alpha 不为正"的方向不变,但 open10 的 alpha 从"与零不可分"变成
    "边际显著为负"。交主控统一加注或重算。【2026-10-10 市值修复后重跑(`BARRA_ATTRIBUTION.md`,`barra2_attribution_summary.csv` mv=new bench=same):open10 alpha −0.98pp(NW t −1.57 ~ −1.72),tail1450 −0.54pp,均不显著——"边际显著为负"不再成立,"没有正 alpha"不变】

---

## 一句话结论(可直接引用)

引擎已抽象为 `simulate / run_weights`,支持数组或路径依赖的 target_fn,改造与审查修复前后 18 条金标准逐日收益逐位一致;
路径依赖接口只向决策函数提供决策时点可得的信息(参与率上限用截至 te-1 的事前估计、持仓按上一成交价计价)。
新增全日 VWAP 口径(外部成交量反推的真实 VWAP 与之偏离中位 1.2bp),但它是平均价口径,日收益一阶自相关约 0.3,
日频 Sharpe / IR 高估约 16%(第一轮在窗口冷启动口径下报 25%),必须看 `*_adj`;而且个股相对收益在这种口径下有正的截面序列相关
(滞后 1 期 0.175),每天重置等权的池基准每年吃亏约 2.1pp(open10 / tail1450 约 0.5pp),
k 日调仓的策略必须对同频再平衡的池基准 `benchmark(rebalance=k)` 算超额,窗口超额还要用预热的连续路径(冷启动多出 0.3~0.9pp/年)(第二轮修订及复核)。成本模型为"佣金 + 印花税 + 一跳半价差 + 平方根冲击(Y=1,0.5/1.5 敏感性)",
引擎与优化器共用同一套系数;Abdi-Ranaldo 价差在本数据上不可识别(合并后均值全为负),默认退回一跳下限、AR 逐股作上界。
计入半价差(一跳下界)与平方根冲击后,1000 万资金下低换手排雷方案每年多扣约 0.7pp(其中价差 0.52pp、冲击 0.17pp),
两个窗口超额仍为正;周频 Top20 每年多扣约 10pp(价差 2.9pp、冲击 7.3pp),超额转负(COMP_v5,信号样本内)。
另发现 COMP_v5 全期超额中约 1.4pp/年来自 2020 年 1 月打分缺失期的空仓。

---

## 审查记录(2026-10-10)

两位独立审查者复核了金标准等价性(旧代码 = golden.json = 新代码,另加 14 个配置逐位相等)、会计重算、
CostModel 无前视与全部核心数字,未发现 P0。共 6 条发现,逐条处理如下。证据脚本 `scripts/eng_review_fixes.py`
(`logs/eng_review_fixes.log`、`logs/eng_review_fixes_A.log`)与 `scripts/eng_review_edge_probe.py`
(需要修复前快照,见其文档串);账本 `eng_review_capw.csv`、`eng_review_hdec.csv`、`eng_vwap_autocorr.csv`、
`eng_review_edge.csv`,`eng_vwap_checks.csv` 与 `eng_cost_demo.csv` 按修复后的脚本重跑。

**修复对已有数字的影响**:引擎默认行为不变 —— 修复后 18 条金标准仍逐日逐位一致(`test_golden_bitwise`,
worst |diff| = 0),重跑 `eng_cost_model_tables.py` 后 `eng_spread_by_size / eng_spread_ar_diagnostics /
eng_impact_examples / eng_engine_checks` 四张表与修复前逐字节相同,`eng_cost_demo.csv` 原有列数值完全相同(只追加列)。
本报告此前的数字都没有用到 `ctx['cap_w']` 或位置参数 `h`,所以没有数字因 P1-a / P2-a 改变;
改变的是:(1) 一句话结论与第 7 节对"0.7pp"的归因;(2) 第 3 节撤回"vwap∈[low,high]"这条证据、换成外部佐证;
(3) 新增 3.1 节,vwap1 的日频 Sharpe / IR 不再可直接引用。

| # | 级别 | 发现 | 处理 | 修复前 → 修复后 |
|---|---|---|---|---|
| 1 | P1 | `ctx['cap_w']` 用成交日 te 的**实际**窗口成交额(open10 = te 开盘 10 分钟,vwap1 = te 全日),决策时不可知;优化器拿它做约束 / 重分配会虚增大资金结果 | **已修复**。`cap_w` 改为 `max_participation × 截至 te-1 的 20 日窗口成交额中位数 / (capital [× nav_dec])`(`ST.decision_window_amount`);引擎内部截量仍用 te 实际值(模拟成交,与旧口径同)。不另外暴露 te 实际值。2.2 / 2.3 节写明只能用 `cap_w` 做约束,且它并不保守 | 恒等式:新 `cap_w` 等于事前中位数公式 = 1、等于 te 实际值公式 = 0(open10、vwap1,`eng_review_capw.csv` A1);te 及以后成交价 / 成交额随机扰动后 `cap_w` 逐位不变(A2)。审查者的"截量后重分配"规则(COMP_v5 Top50 r5,2021-2025,信号样本内,A3):open10 1 亿 年化 **10.25% → 4.78%**、仓位 0.807 → 0.663,按仓位调整的超额 +0.30 → -1.54pp/年(差 +1.84pp,NW t 1.87);vwap1 10 亿 **7.21% → 4.21%**、仓位 0.756 → 0.690(差 +1.06pp,NW t 1.71);open10 10 亿、vwap1 1 亿 差 +0.003 / -0.05pp(t 0.02 / -0.93)。修复后 ctx 组与手算事前中位数组逐位相等。"修复前"组 = 修复前快照的 ctx 驱动结果,逐位核对相等(`eng_review_edge.csv`) |
| 2 | P2 | 传给 target_fn 的 `h`(及 `ctx['nav']`)按 te 成交价漂移;vwap1 下等于用了 te 全日 VWAP,基于 h 的决策规则吃到决策后信息 | **已修复(接口层)**。新增 `ctx['h_dec' / 'nav_dec' / 'aum_dec' / 'budget_dec']`(按 px[te-1] 计价,= 上一笔成交后的持仓);目标里 **NaN = 保持实际持仓不交易**,让规则在 h_dec 口径下表达"不动";位置参数 `h` 保留(引擎从这里成交,也是表达"不动"的数值),文档分组标注、明确禁止用于判断 | `h_dec` 与精确回滚公式之差 ≤1.7e-17(`eng_review_hdec.csv` B1);扰动 te 及以后价格后 `h_dec / nav_dec / budget_dec` 逐位不变(`eng_review_capw.csv` A2)。换手带规则(COMP_v5 Top50,1000 万,2021-2025,信号样本内,B2):vwap1 日调仓 band 0.0003 按 h 判定 12.80% vs 按 h_dec 判定 13.10%,差 **-0.267pp/年(NW t -6.95)**;band 0.0006 差 -0.224(t -5.30);open10 日调仓 -0.087(t -2.32)/ -0.058(t -1.58);周频两档 |差| ≤0.012pp(|t| ≤1.17)。与审查者数字基本一致(审查者为 -0.267 / -0.224 / -0.086 / -0.057,t 为朴素口径)。用 NaN 表达保持与回传 h 数值的差 ≤0.022pp(来自超预算时是否连保持仓一起压缩) |
| 3 | P2 | "vwap 100% 落在 [low, high] 内"由构造保证,作为原料核实证据是空的 | **已修复**。strategy.py `exec_prices` 文档串、本报告第 3 节、测试注释、`eng_vwap_checks.csv` 注释都改为"按构造恒成立,只防原料错位";新增外部佐证:用新浪日成交量(换手率 × 流通股本)反推真实 VWAP | 证据从"恒等式"换成外部对照:645 万个股票日,本项目 vwap 与真实 VWAP 偏离中位 **1.2bp**,99% 分位 11.9bp,98.6% 在 10bp 内;对照 twap_open10 偏离中位 72.5bp(`eng_vwap_checks.csv` ext_*) |
| 4 | P1 | vwap1 日收益有平均价自相关,引擎的 vol / sharpe 与下游按日收益算的 TE / IR 系统性偏差,文档未警告 | **已修复**。引擎结果追加 `ret_ac1 / vr_lr / vol_adj / sharpe_adj`(NW Bartlett,带宽 20),新增 `ST.excess_stats()` 给日频与 NW20 两种 TE / IR 和 NW t;原字段数值不变。EXEC_MODES / exec_prices 注释、新 3.1 节、第 7 / 9 节写明下游规则;`eng_cost_demo.csv` 追加 `vol_adj / sharpe_adj / te / ir / te_adj / ir_adj` 列 | 复现审查数字(`eng_vwap_autocorr.csv`,2021-2025):基准一阶自相关 vwap1 0.308 / open10 0.068 / tail1450 0.096,个股中位 0.203 / 0.041 / 0.046;基准波动 19.76%(日频)→ 23.27%(NW20)/ 22.39%(不重叠 20 日);COMP_v5 排雷 Sharpe **0.938 → 0.808**,超额 TE 2.54% → 3.15%,IR **3.09 → 2.48**;open10 Sharpe 0.809 → 0.817、IR 2.03 → 1.95 |
| 5 | P2 | `run_weights / simulate` 不防负权重(虚卖成本、多头被放大),`ctx['can']` 按引用传入 | **已修复**。目标 < -1e-6 报 `ValueError`、噪声级负数截零、inf 报错、callable 返回形状校验;`ctx` 的 `can / stuck / h_dec` 都是副本 | 负权重用例(Top20 r5,1000 万,2024,`eng_review_edge.csv`):修复前不报错,卖出额 272 → 402 百万元、印花税累计 143 → 212bp、仓位 0.889 → 0.968、年化 9.887% → 10.508%;修复后报 ValueError。原地改 `ctx['can']`:修复前年化 9.887% → 9.813%;修复后与基线逐位相等 |
| 6 | P2 | 一句话结论把"价差 + 冲击"写成"冲击",排雷方案多扣的成本 3/4 其实是半价差 | **已修复**。一句话结论与第 7 节改为分项;表格最后一列改为年化 pp 分项(`spread_ann_pp / impact_ann_pp` 新列) | 排雷 Y=1 一跳 2021-2025:多扣 0.69pp/年 = 半价差 0.52 + 冲击 0.17;Top20:10.21pp/年 = 半价差 2.90 + 冲击 7.30(`eng_cost_demo.csv`) |

**审查之外的顺带改动**:新增 `budget_dec`(审查只点名了 `h / nav`,但 `budget` 同样按 te 价计算);`cap_w` 的事前估计
实测**并不保守**(te 实际 / 估计 中位 0.95,约一半股票日实际低于估计),在 2.3 节写明。回归测试从 18 项增至 25 项,
全部通过(`logs/eng_pytest.log`,约 124 秒)。

**留给下游(未在本轮处理)**:`can / stuck` 里"te+1 有价格""一字板按 te 全日振幅判定"两条旧约定仍在(审查认定
不是新增前视,对集中持仓略有利);引擎对 vwap1 成交仍按 te 全日成交额截量(模拟成交的口径,不是决策信息)。

---

## 第二轮修订(fixa,2026-10-10)

第二轮审计的两个"高"级残留问题都出在基准构造上(详见 `research/CAPACITY_REPORT.md` 文末同名小节)。本报告相关的处理:

| # | 问题 | 处理 | 修订前 → 修订后(出处) |
|---|---|---|---|
| 1 | `benchmark()` 每天按决策池重置等权;k 日调仓的策略其间权重漂移,日收益正自相关时二者之差是与选股无关的再平衡频率差,被记成策略超额(vwap1 最严重) | `benchmark()` 新增 `rebalance`(默认 1 走原代码路径)与 `universe_override`;`rebalance=k` = 同频再平衡池等权(每 k 个信号日按当日决策池、在当日有持有收益的股票上重置等权,其间按成交价漂移,缺价日收益记 0;k 相位平均,相位日历锚在 date_mask 首日;不计成本、不加可成交约束)。新增判据 ④b 与实现核对(3 个测试用例) | 18 条金标准仍逐位一致(`test_golden_bitwise` worst \|diff\| = 0,`logs/fixa_pytest_golden.log`);`test_engine_regression.py` 28 项全部通过(`logs/fixa_pytest.log`)。全池等权 5 日调仓、旧成本、连续路径 2021-2025 的残差(算术):对每日重置基准 vwap1 +1.95 / open10 +0.67pp → 对同频基准 +0.05 / +0.23pp;审计探针口径(2021-01-04 冷启动)的超额 vwap1 +2.53 / open10 +1.07pp 逐位复现,对同频基准 +0.45 / +0.57pp,差额是冷启动的相位空仓(`fixa_fullpool_checks.csv`)。池基准年化 2021-2025:vwap1 8.38% → 10.47%,open10 10.15% → 10.66%(`fixa_bench_compare.csv`) |
| 2 | §3.1"年化收益不受影响"只对单条序列成立 | §3.1 改写并追加同频基准行;`strategy.exec_prices` 文档串与 `EXEC_MODES` 注释同步改 | COMP_v5 排雷(旧成本、2021-2025)vwap1 超额 +9.00 → +6.91pp,IR 日频 3.09 → 2.38、NW20 2.48 → 1.91;open10 +7.00 → +6.50pp,IR_adj 1.95 → 1.82(`fixa_eng_excess_ir.csv`)。**复核更正**:这些同频数仍是 2021-2025 窗口掩码冷启动,含冷启动伪超额 vwap1 +0.52 / open10 +0.41pp;2020H2 预热的连续路径上是 vwap1 +6.39pp、IR 2.26 → 1.94,open10 +6.09pp、1.82 → 1.82,tail1450 +6.65pp、1.99 → 1.99(`fixa_r2_eng_warm.csv`,见文末复核记录 #1) |
| 3 | §7 的 vwap1 行对每日重置基准 | 改为对 vwap1 的同频基准;open10 表保留原值、表后附同频数字 | vwap1 排雷旧成本 2021-2023 / 2024-2025 +8.10 / +11.42 → +6.20 / +9.03pp,Y=1 +7.32 / +10.54 → +5.42 / +8.15pp;Top20 旧成本 +3.47 / +14.20 → +1.56 / +11.81pp,Y=1 −10.73 / −6.68 → −12.63 / −9.07pp;open10 各行下移 0.38 / 0.71pp(`fixa_eng_cost_demo.csv`)。**复核更正**:这些是窗口冷启动口径,含冷启动伪超额(排雷 2021-2023 +0.52~+0.80、2024-2025 +0.80~+0.93pp);复核后主口径见 §7 补表(2020H2 预热,`fixa_r2_eng_warm.csv`):vwap1 排雷旧成本 +5.39 / +8.10pp、Y=1 +4.64 / +7.24pp,Top20 旧成本 +0.70 / +10.96pp、Y=1 −12.90 / −7.33pp;open10 排雷 Y=1 +4.32 / +6.95pp、Top20 Y=1 −10.71 / −7.06pp |
| 4 | §9 编号 1–6、9、10、7、8 | 改为 1–10 顺序,新增第 11 条(k 日调仓对同频基准) | — |
| 5 | (问题 2)中证1000 代理旧基准 `IP.engine_consistent_bench_returns` 系统性偏低 | `exec_time_bench_returns` 搬入 `index_proxy` 作为推荐基准(`optimizer` 旧名再导出,数值逐位相同),旧函数发 `DeprecationWarning`;`tests/test_fixa_benchmarks.py` 5 项 | 代理年化 2021-2025 open10 4.69% → 5.49%、vwap1 2.93% → 5.45%(`fixa_bench_compare.csv`;以上为修复前市值,市值修复后新口径 open10 5.13% → 5.97%、vwap1 3.38% → 5.95%,见 `CSI1000_PROXY.md`);详见 `research/CSI1000_PROXY.md` 文末 |

**修订改变了哪些结论**:§7 的 vwap1 超额与 §3.1 的 vwap1 IR 都下移约 2pp / 0.5;"vwap1 日频 Sharpe / IR 高估约 25%、必须看 `*_adj`"
仍成立,另加一条"k 日调仓对同频基准"的下游规则。§7 的读法(排雷两个窗口为正、Top20 计入冲击后转负)、成本模型与回归判据 ①~⑭ 不变。
(复核后:"高估约 25%"改为"预热连续路径上约 16%",同频超额再扣掉 0.3~0.9pp 的冷启动伪超额,见下节。)

---

## 第二轮修订的复核审查记录(fixa,2026-10-10)

复核审查用独立实现重算了第二轮修订的核心数字(同频基准、容量网格、IE 判据、§3.1 排雷 IR、成交时点代理),全部对上,没有 P0。
共 5 条发现(1 条 P1、4 条 P2),逐条处理如下。证据脚本 `scripts/fixa_benchmark_fix.py review2 [mech|barra|warm]`
(`logs/fixa_r2_mech_barra.log`、`logs/fixa_r2_barra.log`、`logs/fixa_r2_warm.log`);
账本 `research/ledger/fixa_r2_eng_warm.csv`、`fixa_r2_mech.csv`、`fixa_r2_barra_delta.csv`。审查探针的数字全部逐位复现。
`strategy.py` / 测试只改了文档串:金标准逐位一致(worst |diff| = 0,`logs/fixa_r2_pytest_golden.log`),
`test_engine_regression.py` + `test_fixa_benchmarks.py` 33 项全部通过(`logs/fixa_r2_pytest.log`)。

| # | 级别 | 发现 | 处理 | 修订前 → 修订后(出处) |
|---|---|---|---|---|
| 1 | P1 | §3.1 / §7 的"对同频基准超额"仍含窗口掩码冷启动的伪超额:策略在窗口首日冷启动(相位 k 前 k 天空仓),同频基准从面板起点预热、始终满仓 | **已修复**(两种处理都做;伪超额 96~100% 出在窗口首 5 个信号日,同期池基准累计 −3.9%~−5.7%):§3.1 表与分窗口 IR、§7 改用 2020H2 预热的连续路径切片(与容量报告同法),新增 §7 补表;冷启动原数保留并标注,表后写明冷启动伪超额的量级;第二轮修订表第 2/3 行加注。顺带更正:vwap1 的"IR 高估约 25%"也是冷启动口径下的数,预热路径上约 16% | 旧成本排雷对同频基准,冷启动 → 预热:vwap1 2021-2025 / 2021-2023 / 2024-2025 +6.91 / +6.20 / +9.03 → **+6.39 / +5.39 / +8.10pp**(伪超额 +0.52 / +0.80 / +0.93);open10 +6.50 / +5.71 / +8.73 → **+6.09 / +5.08 / +7.82pp**(+0.41 / +0.63 / +0.91);tail1450 +7.00 / +5.99 / +9.59 → +6.65 / +5.46 / +8.68pp。IR(同频,日频 → NW20)vwap1 2.38 → 1.91 改为 **2.26 → 1.94**,open10 1.91 → 1.82 改为 1.82 → 1.82 —— 冷启动同时抬高了分子(超额)和分母(NW 口径的 TE 2.50% → 3.10%,预热 2.43% → 2.83%),所以 IR_adj 几乎不变。§7 Y=1 一跳:open10 排雷 +4.93 / +7.85 → +4.32 / +6.95pp,Top20 −10.42 / −7.59 → −10.71 / −7.06pp;vwap1 排雷 +5.42 / +8.15 → +4.64 / +7.24pp,Top20 −12.63 / −9.07 → −12.90 / −7.33pp(Top20 计入冲击的 2024-2025 还混了"冲击按 capital × 净值"、预热路径净值 0.6~0.85 的规模效应,见 §7 补表读法)。核对:冷启动重跑与 `eng_cost_demo.csv` 年化逐位相同(差 0);同频基准在冷 / 热两种掩码下逐日相同(差 0)(`fixa_r2_eng_warm.csv`) |
| 2 | P2 | 同频基准没有传播到其他"5 日调仓 vs 每日重置池基准"的报告;tail1450 的偏差未量化;"all affected reports updated"说过头 | **已处理(量化 + 盘点,其他负责人的报告不改)**:§3.1 与 §9 第 11 条补 tail1450 量级;§9 第 11 条列出仍用每日重置基准的报告与脚本,交主控统一加注或重算;BARRA_ATTRIBUTION 用"OLS 对超额线性"精确算出换基准后的变化;CAPACITY §7.7 引用 98% 处注明口径与新数。上一轮摘要的"all affected reports"应读作"fixa 负责的七份报告" | 同频 − 每日重置(2021-2025):tail1450 +0.50pp(Panel,t 3.7)/ +0.60pp(TailPanel);2020-2025 +0.60 / +0.71,2024-2025 +0.76 / +0.80(`fixa_r2_mech.csv`)。BARRA_ATTRIBUTION open10:总超额 +5.32 → +4.89pp,流动性 +5.20 → +5.13pp(占比 98% → 105%),残差 alpha −0.31 → **−1.22pp**,NW t −0.48~−0.54 → **−1.94~−2.15**(用缓存打分逐位复现原回归后重算);tail1450 alpha +0.00 → −0.78pp(t 未重算)(`fixa_r2_barra_delta.csv`)—— open10 的 alpha 从"与零不可分"变成"边际显著为负",结论方向(无正 alpha)不变。【2026-10-10 市值修复后重跑(`BARRA_ATTRIBUTION.md`,`barra2_attribution_summary.csv` mv=new bench=same):open10 alpha −0.98pp(NW t −1.57 ~ −1.72),tail1450 −0.54pp,均不显著——"边际显著为负"不再成立,"没有正 alpha"不变】 |
| 3 | P2 | 容量报告一句话结论里 vwap1"Y=1 盈亏平衡 > 100 亿"只剩点估计支撑 | **已修复**(见 `research/CAPACITY_REPORT.md` 文末同名小节):§0 第 1 条与一句话结论加上 100 亿格的 NW t、±1SE 盈亏平衡与开发窗,明确"> 100 亿"是预注册定义下的点估计 | vwap1 100 亿 +0.99pp(NW t 0.57),±1SE 盈亏平衡 67.2亿 ~ >100亿,开发窗 +0.22pp(t 0.10)、±1SE 下界 33.1亿(`capacity_grid.csv` / `capacity_summary.csv`) |
| 4 | P2 | 再平衡频率差的机制证据引用了池等权序列自相关,起作用的是截面相对收益的序列相关 | **已修复**:§3.1 新增"机制"段,`strategy.py` 文件头 / `exec_prices` / `benchmark` 文档串、`fixa_benchmark_fix.py` 文档串、`test_engine_regression.py` 文档串、容量报告 §1.6 与修订小节都改为截面相对收益的序列相关,池序列 ac1 只作旁证;并用一阶展开做了定量核对 | 机制统计量 0.308 / 0.068(池序列)→ **0.175 / 0.037 / 0.036**(vwap1 / open10 / tail1450,截面相对收益滞后 1 期相关,2021-2025);一阶展开预测的算术差 1.78 / 0.41 / 0.42pp,实测 1.90 / 0.44 / 0.44pp,解释 93~96%(分窗口 82~105%)(`fixa_r2_mech.csv`) |
| 5 | P2 | §3.1 分窗口 IR 仍对每日重置基准,未标注、未重算 | **已修复**:原行标注"对每日重置池基准、窗口掩码冷启动";新增同频、预热的分窗口行 | vwap1 2021-2023 / 2024-2025 IR(日频 → NW20)3.09 → 2.31 / 3.27 → 2.82(每日重置、冷启动)→ **2.17 → 1.82 / 2.39 → 2.14**(同频、预热);open10 1.97 → 1.78 / 2.27 → 2.31 → 1.71 → 1.68 / 1.97 → 2.04(`fixa_r2_eng_warm.csv`) |

**复核改变了哪些结论**:k 日调仓策略对同频基准的窗口超额再下移 0.3~0.9pp/年(冷启动伪超额);vwap1 日频 IR 的高估幅度从"约 25%"
改为"约 16%"(冷启动口径下才是 25%);BARRA_ATTRIBUTION 在同频基准下残差 alpha 边际显著为负(该报告由其负责人改)。【2026-10-10 市值修复后重跑(`BARRA_ATTRIBUTION.md`,`barra2_attribution_summary.csv` mv=new bench=same):open10 alpha −0.98pp(NW t −1.57 ~ −1.72),tail1450 −0.54pp,均不显著——"边际显著为负"不再成立,"没有正 alpha"不变】
不变的:§7 的读法(排雷各成本模型、两个窗口都为正;Top20 计入冲击后在两个口径、两个窗口都为负,Y=0.5 也为负)、
"k 日调仓对同频基准"的规则、成本模型、金标准与回归判据 ①~⑭。COMP_v5 仍是信号样本内,数据无退市股,收益偏高。
