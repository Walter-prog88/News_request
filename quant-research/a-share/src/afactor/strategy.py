"""策略层:两个资金版本的股票池、基线策略、以及统一的绩效口径。

版本(2026-08-22 与用户确认):
  V1000  1000 万:核心排雷 + 小市值卫星;决策池 ADV >= 2000 万
  V100   100 万:集中持仓、小市值可做主仓、接受高波动;决策池 ADV >= 300 万
                 **只设流动性下限,不按 ST/北交所/小市值身份剔除**

基线(模型的及格线):多因子打分 -> Top N 等权满仓 -> 按频率调仓。
模型必须接近或超过基线才算有价值。同时报排雷基线做对照,防止"基线太弱"造成错觉。

## 2026-10-10 引擎改造(前缀 eng)

把"给定目标权重 -> 逐日成交模拟"抽成公开的 `simulate()`,新增 `run_weights()` 入口,
供组合层(指增优化器、容量曲线、事件组合)直接喂权重或路径依赖的 target_fn:

    build_weights(...)   打分 -> 目标权重 W (T x N),与旧 _run_one_phase 内部完全同法
    simulate(target,...) 单相位逐日成交模拟;target = ndarray W 或 callable target_fn(t, h, ctx)
    run_weights(...)     相位平均入口(逻辑与 run_strategy 完全一致),可选冲击成本 / 记录明细
    run_strategy(...)    = build_weights + run_weights,默认参数下**逐位**等于改造前

改造前金标准见 tests/golden_engine.json(scripts/eng_make_golden.py 在未改动代码上生成),
tests/test_engine_regression.py 以 1e-10 容差逐日比对。新增可选参数全部默认关闭:
cost_model=None 时成本公式与改造前同一表达式;capital=None 时用版本资金。

新增执行口径 vwap1:T 收盘信号 -> T+1 全日 VWAP 成交(原料定义见 exec_prices 注释)。

## 2026-10-10 审查修复(前缀 eng,见 research/ENGINE_IMPACT_MODEL.md "审查记录")

callable 的 ctx 只给**决策时点可得**的信息(审查 P1/P2):
  - ctx['cap_w'] 改为参与率上限的**事前估计**:截至 te-1 的 20 日窗口成交额中位数折算
    (旧版直接给 te 当天实际窗口成交额,vwap1 下是 te 全日成交额,决策时不可知)。
    引擎内部截量仍按 te 实际窗口成交额(模拟成交,与旧口径相同)。
  - 新增 ctx['h_dec'] / ctx['nav_dec']:按上一成交价 px[te-1] 计价的持仓与净值(决策时可得)。
    位置参数 h 与 ctx['nav'] 仍按 px[te] 计价(引擎实际从这里成交),决策规则不得用它们做判断。
  - 目标里的 NaN = "该股保持实际持仓、不交易",让决策函数在 h_dec 口径下表达"不动"。
  - 目标权重不得为负(< -NEG_TOL 报错;求解器噪声级的负数截零);ctx 里的数组都是副本。
二阶矩:日收益有自相关时(vwap1 平均价口径一阶自相关约 0.3),日频 std×sqrt(242) 低估波动。
结果新增 vol_adj / sharpe_adj / vr_lr / ret_ac1(Newey-West 长期方差,带宽 LR_BANDWIDTH),
跟踪误差 / IR 请用 excess_stats()。原有字段数值逐位不变。

## 2026-10-10 第二轮修订(前缀 fixa):同频再平衡的池等权基准

benchmark() 新增可选参数 rebalance(默认 1 = 原口径,金标准逐位一致)。k 日调仓的策略请对
benchmark(..., rebalance=k) 算超额:k=1 基准每天重置等权,与其间漂移的 k 日调仓组合之间有一段
与选股无关的"再平衡频率差"。2021-2025 同频(k=5)池基准比 k=1 的年化高 vwap1 2.09pp、open10 0.50pp
(research/ledger/fixa_bench_compare.csv),k 日调仓策略对 k=1 基准的超额被抬高同样幅度(见 benchmark 文档串)。
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from afactor.config import (COST_BPS_ONEWAY, STAMP_TAX_BPS_BEFORE, STAMP_TAX_BPS_AFTER,
                            STAMP_TAX_CHANGE_DATE, MAX_PARTICIPATION, TMP)
from afactor.panel import Panel

TRADING_DAYS = 242
# (不用 @:numpy 2.0 + Accelerate 的 matmul 会报假的浮点警告)
# Newey-West(Bartlett 核)长期方差的带宽(交易日)。20 日 ≈ 月度持有期:Bartlett 带宽 L 的长期方差
# 约等于 (L+1) 日重叠收益和的方差 /(L+1),所以 vol_adj ≈ "按月度持有期折算的年化波动",
# 对 vwap1(平均价口径,正自相关)和 open10(略负的中期自相关)都可比。
LR_BANDWIDTH = 20
# 目标权重的负值容差:> -NEG_TOL 视为求解器噪声截零;更负则报错(引擎只做多头,负权重会产生虚卖成本)。
NEG_TOL = 1e-6

VERSIONS = {
    "V1000": dict(capital=1e7, min_adv=2e7, twap_minutes=10),
    "V100": dict(capital=1e6, min_adv=3e6, twap_minutes=5),
}

# 执行口径:(信号截断, 成交窗口 bar 索引区间, 是否隔夜)
EXEC_MODES = {
    "open10":   dict(desc="T收盘信号 -> T+1 09:31-09:40 TWAP", win=(1, 11), lag=1, tail=False),
    "open5":    dict(desc="T收盘信号 -> T+1 09:31-09:35 TWAP", win=(1, 6), lag=1, tail=False),
    "tail1450": dict(desc="T日14:49截断 -> T日14:50-14:57 TWAP", win=(230, 238), lag=0, tail=True),
    "tail1445": dict(desc="T日14:44截断 -> T日14:45-14:57 TWAP", win=(225, 238), lag=0, tail=True),
    "close":    dict(desc="T日14:49截断 -> T日15:00 收盘集合竞价", win=(240, 241), lag=0, tail=True),
    # 2026-10-10 新增。持有收益 vwap[te+1]/vwap[te]-1,参与率按全日成交额 amount 计。
    # ⚠ 全日平均价口径(Working 1960 效应):日收益正自相关,日频 std*sqrt(242) 低估波动、
    #   高估 Sharpe / IR。vwap1 下请用 vol_adj / sharpe_adj 与 excess_stats() 的 te_adj / ir_adj,
    #   且不得拿 vwap1 的日频 Sharpe / IR 与 open10 横比(数字见 exec_prices 注释)。
    #   k 日调仓策略的"相对池等权"超额必须对同频基准 benchmark(..., rebalance=k) 算(fixa)。
    "vwap1":    dict(desc="T收盘信号 -> T+1 全日 VWAP(09:30-15:00,含两次集合竞价)", win=(0, 241),
                     lag=1, tail=False),
}


def universe(p: Panel, version: str) -> pd.DataFrame:
    """版本对应的决策池。只用 T 时点信息,不含前视。"""
    cfg = VERSIONS[version]
    from afactor.config import MIN_LIST_DAYS
    return (p["valid"] & (p["days_listed"] >= MIN_LIST_DAYS)
            & (p["adv20"] >= cfg["min_adv"])).fillna(False)


def exec_prices(p: Panel, mode: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """成交价与成交窗口成交额。open* 用现成的 twap_open10;其余用尾盘窗口。

    vwap1(2026-10-10 核实原料,见 build_daily._day_features):
      vwap   = Σ_bar amt / Σ_bar (amt / mid),mid = (open_adj + close_adj)/2,
               即**全日 241 根 bar(含 09:30 开盘与 15:00 收盘集合竞价)的成交额加权 bar 中价**。
               原始数据没有成交量,"股数"是 amt/mid 的近似,故它是近似 VWAP;
               与 twap_open10 同源(同一套前复权分钟 open/close),复权口径一致,收益率无前视。
               "vwap 落在 [low_px, high_px] 内"是**按构造恒成立**的(bar 中价的加权调和平均必落在同一套
               bar 的 max/min 之间),只防原料错位,不是口径证据(2026-10-10 审查 P2)。
               外部佐证(eng_vwap_checks.csv 的 ext_* 行):用新浪日成交量(换手率×流通股本)反推真实 VWAP,
               比较同日 vwap/收盘价,645 万个股票日偏离中位 1.2bp、99% 分位 11.9bp,98.6% 在 10bp 内
               (对照:twap_open10 与真实 VWAP 偏离中位 72.5bp)。vwap/twap_open10 中位 1.000,1%/99% 分位 0.965/1.052。
      amount = Σ_bar amt,全日成交额,单位**元**(600519 2024 年日中位 45.6 亿元)。

    ⚠ vwap1 是全日**平均价**口径(Working 1960):价格是全天均值,相邻两日收益共享半天信息 -> 正自相关。
      2021-2025 池等权基准日收益一阶自相关 vwap1 0.308(open10 0.068、tail1450 0.096),个股中位 0.203;
      基准年化波动 日频 19.76% vs 长期方差口径(NW 带宽 20)23.27%;COMP_v5 排雷超额 IR 日频 3.09 vs 2.48。
      **单条序列**的年化收益不受影响(VWAP 到 VWAP 的收益可连乘还原),偏的是二阶矩。vwap1 下请看 vol_adj /
      sharpe_adj / excess_stats() 的 te_adj / ir_adj,不得拿 vwap1 的日频 Sharpe / IR 与 open10 横比
      (数字见 research/ledger/eng_vwap_autocorr.csv)。
      ⚠ 但**再平衡频率不同的两个组合之间的超额**(一阶矩)会被自相关改变(2026-10-10 第二轮修订,前缀 fixa):
      起作用的是**个股相对池均值收益的截面序列相关**(不是上面那个池等权序列的自相关;每日重置与 k 日漂移的
      两个等权组合市场暴露相同):2021-2025 滞后 1 期截面相关 vwap1 0.175、open10 0.037、tail1450 0.036
      (research/ledger/fixa_r2_mech.csv xs_corr_l1),每天重置等权的池基准因此系统性吃亏,k 日调仓策略对它的
      超额被抬高(vwap1 约 +2.1pp/年、open10 / tail1450 约 +0.5pp/年)。k 日调仓的策略请对 benchmark(..., rebalance=k) 算超额。
    """
    m = EXEC_MODES[mode]
    if mode == "vwap1":
        return p.raw("vwap"), p.raw("amount")
    if mode.startswith("open"):
        return p.raw("twap_open10"), p.raw("amt_open10")
    if mode == "close":
        return p.raw("close_px"), p.raw("amt_close_auction")
    return p.raw("twap_tail8"), p.raw("amt_tail8")


def decision_window_amount(p: Panel, mode: str, window: int = 20, min_periods: int = 10) -> np.ndarray:
    """成交窗口成交额的**事前估计**(元),行 te = 截至 te-1 的 window 日窗口成交额中位数。

    给 callable 的 ctx['cap_w'] 用(审查 P1):te 当天的实际窗口成交额(open10 = te 开盘 10 分钟、
    vwap1 = te 全日)在信号日 t 收盘后的决策时点不可知,决策函数拿它做约束或重分配会虚增大资金结果。
    用中位数而非均值:成交额右偏,均值会被放量日抬高。但它**并不保守**:2021-2025 V1000 池 te 实际 / 估计
    中位 0.95,约 53%(open10)/ 54%(vwap1)的股票日实际低于估计(eng_review_capw.csv A4);
    所以决策函数按 cap_w 下单仍可能被引擎按实际成交额截量,record 的 part_cut 记录落差。
    min_periods 不足(新股、长期停牌复牌)时为 NaN -> cap_w=0,即决策函数视为不可买。
    结果缓存在 Panel 对象的 _derived 上(随面板生灭),按原料字段名区分(open5/open10 共用)。"""
    field = ("amount" if mode == "vwap1" else "amt_open10" if mode.startswith("open")
             else "amt_close_auction" if mode == "close" else "amt_tail8")     # 与 exec_prices 同一映射
    key = f"__eng_capest__{field}_{window}_{min_periods}"
    if key not in p._derived:
        _, amt_df = exec_prices(p, mode)
        est = amt_df.astype(np.float64).rolling(window, min_periods=min_periods).median().shift(1)
        p._derived[key] = est.to_numpy(np.float64)
    return p._derived[key]


def lr_variance_ratio(r, bandwidth: int = LR_BANDWIDTH) -> float:
    """Newey-West(Bartlett 核)长期方差 / 日方差。无自相关时 ≈1;平均价口径的正自相关使其 >1。
    vol_adj = vol * sqrt(比值),sharpe_adj = sharpe / sqrt(比值)。"""
    x = np.asarray(r, np.float64)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 2 * bandwidth + 10:
        return np.nan
    e = x - x.mean()
    g0 = float(np.sum(e * e)) / n
    if g0 <= 0:
        return np.nan
    s = g0
    for k in range(1, int(bandwidth) + 1):
        s += 2.0 * (1.0 - k / (bandwidth + 1.0)) * float(np.sum(e[k:] * e[:-k])) / n
    return s / g0


def _ac1(r) -> float:
    x = np.asarray(r, np.float64)
    x = x[np.isfinite(x)]
    if len(x) < 10:
        return np.nan
    e = x - x.mean()
    g0 = float(np.sum(e * e))
    return float(np.sum(e[1:] * e[:-1])) / g0 if g0 > 0 else np.nan


def _lr_fields(r_s: pd.Series, vol: float, sharpe: float) -> dict:
    """在原有 vol / sharpe 之外**追加**的自相关稳健字段(原字段数值不动)。"""
    vr = lr_variance_ratio(r_s.to_numpy(np.float64))
    ok = np.isfinite(vr) and vr > 0
    return dict(ret_ac1=_ac1(r_s.to_numpy(np.float64)), vr_lr=vr,
                vol_adj=float(vol * np.sqrt(vr)) if ok else np.nan,
                sharpe_adj=float(sharpe / np.sqrt(vr)) if ok and np.isfinite(sharpe) else np.nan)


def excess_stats(rets: pd.Series, bench_rets: pd.Series, bandwidth: int = LR_BANDWIDTH,
                 t_bandwidth: int = 5) -> dict:
    """超额的跟踪误差 / IR,同时给日频口径与自相关稳健口径。两条序列按索引取交集对齐。

    excess_arith = mean(日超额)*242(IR 分子,与 scripts/run_ir_analysis.py 同口径);
    te / ir       = 日频 std*sqrt(242) 口径(open10 等成交价口径基本无偏);
    te_adj / ir_adj = Newey-West 长期方差(带宽 bandwidth)口径 —— vwap1 必须用这一组;
    t_nw          = 日超额均值的 Newey-West t(带宽 t_bandwidth,默认 5 = max(5, 2h), h=1)。"""
    a, b = rets.align(bench_rets, join="inner")
    d = (a - b).dropna()
    n = len(d)
    if n < 30 or d.std() <= 0:
        return dict(n_days=n)
    te = float(d.std() * np.sqrt(TRADING_DAYS))
    ex = float(d.mean() * TRADING_DAYS)
    vr = lr_variance_ratio(d.to_numpy(np.float64), bandwidth)
    vr_t = lr_variance_ratio(d.to_numpy(np.float64), t_bandwidth)
    ok = np.isfinite(vr) and vr > 0
    t_naive = float(d.mean() / d.std() * np.sqrt(n))
    return dict(n_days=n, excess_arith=ex, te=te, ir=ex / te,
                te_adj=te * np.sqrt(vr) if ok else np.nan, ir_adj=ex / te / np.sqrt(vr) if ok else np.nan,
                vr_lr=vr, excess_ac1=_ac1(d.to_numpy(np.float64)), t_naive=t_naive,
                t_nw=t_naive / np.sqrt(vr_t) if np.isfinite(vr_t) and vr_t > 0 else np.nan)


def tilt_weights(score: np.ndarray, M: np.ndarray, power: float = 1.0) -> np.ndarray:
    """篮子内倾斜:始终满仓持有 mask 内全部股票,权重 ∝ 打分名次百分位^power。
    power=1 线性(最高分权重≈最低分的 N 倍中位的 2 倍);power=0 退化为等权。
    用于实验 b:"篮子内按因子调权 vs 篮子等权拿着"。"""
    T, N = score.shape
    W = np.zeros((T, N), np.float32)
    for t in range(T):
        m = M[t] & np.isfinite(score[t])
        k = int(m.sum())
        if k < 5:
            continue
        idx = np.flatnonzero(m)
        r = np.argsort(np.argsort(score[t, idx])) + 1.0          # 1..k,分高名次大
        w = (r / k) ** power
        W[t, idx] = w / w.sum()
    return W


def topn_weights(score: np.ndarray, M: np.ndarray, n: int) -> np.ndarray:
    """Top-N 等权满仓。score 高者优先。"""
    T, N = score.shape
    W = np.zeros((T, N), np.float32)
    for t in range(T):
        m = M[t] & np.isfinite(score[t])
        k = int(m.sum())
        if k < 5:
            continue
        idx = np.flatnonzero(m)
        kk = min(n, k)
        sel = idx[np.argpartition(-score[t, idx], kk - 1)[:kk]]
        W[t, sel] = 1.0 / kk
    return W


def build_weights(score: pd.DataFrame, p: Panel, version: str, scheme: str = "top", n: int = 20,
                  exclude_frac: float = 0.3, buffer: tuple | None = None,
                  date_mask: np.ndarray | None = None, sign: int = 1,
                  universe_override: pd.DataFrame | None = None) -> np.ndarray:
    """打分 -> 目标权重 W(T x N,float32,行和为 1 或 0)。与改造前 _run_one_phase 内部完全同法。

    W 与调仓相位无关,所以 run_strategy 现在只算一次 W 再跑 k 个相位(旧版每个相位重算一遍)。"""
    dates, codes = p.dates, p.codes
    U = (universe_override.reindex(index=dates, columns=codes).fillna(False) if universe_override is not None
         else universe(p, version)).to_numpy(bool)
    if date_mask is not None:
        U = U & date_mask[:, None]
    S = (score.reindex(index=dates, columns=codes).to_numpy(np.float32) * sign)

    if scheme == "top":
        W = topn_weights(S, U, n)
    elif scheme == "tilt":
        W = tilt_weights(S, U, power=float(n) if n else 1.0)   # 复用 n 作 power
    elif scheme == "vw":
        # 按打分**数值**加权(而非名次):score 即目标权重比例,负值截零。
        # 用于复现 ETF 的市值加权,回答"ETF 赢等权是不是仅仅因为加权方式"。
        W = np.where(U & np.isfinite(S), np.clip(S, 0, None), 0.0).astype(np.float32)
        rs = W.sum(axis=1, keepdims=True)
        W = np.divide(W, rs, out=np.zeros_like(W), where=rs > 0)
    else:
        from afactor.backtest import weights_exclude_buffer, target_weights
        W = (weights_exclude_buffer(S, U, 1, *buffer) if buffer
             else target_weights(S, U, top_n=None, sign=1, scheme="exclude", top_frac=exclude_frac))
    return W


def _aggregate_phases(subs: list[dict]) -> dict:
    """k 个相位子组合等权 = 相位平均。与改造前 run_strategy 的聚合逐字相同。"""
    R = pd.concat([x["rets"] for x in subs], axis=1)
    r_s = R.mean(axis=1)                      # k 个子组合等权 = 相位平均
    nav = (1.0 + r_s).cumprod()
    yrs = len(r_s) / TRADING_DAYS
    agg = lambda key: float(np.mean([x[key] for x in subs if np.isfinite(x.get(key, np.nan))]))
    out = dict(nav=nav, rets=r_s, n_phases=len(subs),
               ann_return=float(nav.iloc[-1] ** (1 / yrs) - 1),
               sharpe=float(r_s.mean() / r_s.std() * np.sqrt(TRADING_DAYS)) if r_s.std() > 0 else np.nan,
               max_dd=float((nav / nav.cummax() - 1).min()),
               turnover=agg("turnover"), cost_bps=agg("cost_bps"),
               holdings=agg("holdings"), exposure=agg("exposure"),
               n_days=len(r_s), vol=float(r_s.std() * np.sqrt(TRADING_DAYS)),
               phase_ann=[x["ann_return"] for x in subs])
    # 2026-10-10 审查 P1:追加自相关稳健的 vol_adj / sharpe_adj(相位平均后的序列上算),原字段不动
    out.update(_lr_fields(r_s, out["vol"], out["sharpe"]))
    return out


def run_strategy(score: pd.DataFrame, p: Panel, version: str, mode: str,
                 scheme: str = "top", n: int = 20, rebalance: int = 1,
                 exclude_frac: float = 0.3, buffer: tuple | None = None,
                 date_mask: np.ndarray | None = None, sign: int = 1,
                 avg_phases: bool = True, universe_override: pd.DataFrame | None = None,
                 capital: float | None = None, cost_model=None,
                 max_participation: float | None = None, participation_on_nav: bool = False,
                 record: bool = False) -> dict:
    """统一回测入口。scheme ∈ {top, exclude}。返回净值与绩效。

    avg_phases=True(默认):k 日调仓时跑满 **k 个相位**并对日收益取平均,
    等价于把资金等分成 k 个子组合、各自错开一天调仓(经典 overlapping portfolio)。

    ⚠ 这不是可选的精细化,是**结论成立的前提**。实测同一打分在 5 个合法周频相位下,
    V1000 Top20 的逐年超额摆动可达:rev20 2024 [+42.01,+9.78,+8.84,+15.23,+30.41]
    (极差 33.2pp)、mom60 2025 极差 41.7pp。单相位的逐年超额是噪声主导的,
    任何"模型超额 X%"的结论在单相位下都不可信。

    2026-10-10 新增的可选参数(默认值下与改造前逐位一致,见 tests/golden_engine.json):
      capital            覆盖版本资金(决策池仍按 version 的 min_adv);容量曲线用
      cost_model         afactor.impact.CostModel;None = 旧成本(佣金 1bp + 印花税)
      max_participation  覆盖 config.MAX_PARTICIPATION
      participation_on_nav  参与率上限按 capital*当前净值 计(默认 False = 旧口径按初始资金)
      record             返回 res["record"](每日持仓、成交额、成本分解、现金、参与率约束)
    """
    W = build_weights(score, p, version, scheme, n, exclude_frac, buffer, date_mask, sign,
                      universe_override)
    cap = VERSIONS[version]["capital"] if capital is None else float(capital)
    return run_weights(W, p, mode, cap, rebalance=rebalance, avg_phases=avg_phases,
                       date_mask=date_mask, cost_model=cost_model,
                       max_participation=max_participation, record=record,
                       participation_on_nav=participation_on_nav)


def _run_one_phase(score, p, version, mode, scheme, n, rebalance, exclude_frac,
                   buffer, date_mask, sign, phase: int, universe_override=None) -> dict:
    """单相位回测(保留旧签名:scripts/run_barra_attribution_top20.py 等直接调用)。"""
    W = build_weights(score, p, version, scheme, n, exclude_frac, buffer, date_mask, sign,
                      universe_override)
    return simulate(W, p, mode, VERSIONS[version]["capital"], rebalance=rebalance,
                    date_mask=date_mask, phase=phase)


def run_weights(target, p: Panel, mode: str, capital: float, rebalance: int = 5,
                avg_phases: bool = True, date_mask: np.ndarray | None = None,
                cost_model=None, max_participation: float | None = None, record: bool = False,
                normalize: bool = True, participation_on_nav: bool = False,
                phases: list[int] | None = None, stamp_fn=None) -> dict:
    """权重 -> 回测。相位平均逻辑与 run_strategy 完全一致。

    target
      - ndarray (T x N),行对齐 p.dates、列对齐 p.codes,**只在调仓日使用**;
        date_mask 之外的行视为 0(与 run_strategy 把决策池按 mask 截断同义)。
      - callable target_fn(t, h, ctx) -> ndarray(N,) 或 None(None = 今天不交易)。
        只在 date_mask 之内的调仓日被调用;每个相位是一条独立路径(h 各自漂移)。
        ⚠ 位置参数 h 按成交价 px[te] 计价(te 当天才知道),决策规则请用 ctx['h_dec'] / ctx['nav_dec'];
        ctx['cap_w'] 是参与率上限的事前估计;"某股不交易"请返回 NaN。字段与规则见 simulate()。
      - 两种形式的目标都不得含负数(< -NEG_TOL 报错);NaN = 该股保持实际持仓不交易。
    capital      初始资金(元)。参与率上限 = max_participation * 窗口成交额 / capital
                 (participation_on_nav=True 时分母为 capital*当前净值);冲击成本的
                 Q = |Δw| * capital * 当前净值。
    rebalance    k 日调仓;avg_phases=True 且 k>1 时跑满 k 个相位取日收益平均。
                 每个相位都按**全额 capital** 模拟(与旧引擎一致:相位平均是降噪手段,
                 不是真的把资金切成 k 份;对冲击/参与率而言这是偏保守的)。
    phases       只跑指定相位(调试/省时);None = 全部。
    normalize    True(默认,与 run_strategy 一致):目标在可交易标的上归一到"1 - 卖不掉的残仓";
                 False:目标权重按原值使用(可留现金),仅当超出预算时等比压缩。
    record       True 时返回 res["record"] = dict(daily, daily_by_phase, holdings, meta),见 simulate()。
    """
    k = max(int(rebalance), 1)
    sim_kw = dict(cost_model=cost_model, max_participation=max_participation, record=record,
                  normalize=normalize, participation_on_nav=participation_on_nav)
    if stamp_fn is not None:                    # lh(2026-10-11):分段印花税钩子,默认 None 时不传,路径逐位不变
        sim_kw["stamp_fn"] = stamp_fn
    if avg_phases and k > 1:
        phs = list(range(k)) if phases is None else [int(x) for x in phases]
        subs = [simulate(target, p, mode, capital, rebalance=k, date_mask=date_mask, phase=ph, **sim_kw)
                for ph in phs]
        subs = [x for x in subs if x.get("rets") is not None and len(x["rets"]) > 5]
        if not subs:
            return dict(nav=pd.Series(dtype=float), n_days=0)
        out = _aggregate_phases(subs)
        if record:
            out["record"] = _merge_records([x["record"] for x in subs])
        return out
    ph = 0 if phases is None else int(phases[0])
    res = simulate(target, p, mode, capital, rebalance=k, date_mask=date_mask, phase=ph, **sim_kw)
    if record and "record" in res:
        res["record"] = _merge_records([res["record"]])      # 与多相位同结构(holdings -> DataFrame)
    return res


def simulate(target, p: Panel, mode: str, capital: float, rebalance: int = 1,
             date_mask: np.ndarray | None = None, phase: int = 0, cost_model=None,
             max_participation: float | None = None, record: bool = False,
             normalize: bool = True, participation_on_nav: bool = False, stamp_fn=None) -> dict:
    """单相位逐日成交模拟(由改造前 _run_one_phase 的循环原样抽出)。

    时序口径:信号日 t 收盘(或尾盘截断)后定目标 -> 成交日 te = t + lag 在成交窗口按窗口均价成交
    -> 持有到 te+1 同窗口,收益 px[te+1]/px[te]-1,**打在信号日 t**(与 benchmark 同索引)。

    target 为 callable 时,在调仓日调用 target_fn(t, h, ctx)。ctx 里的数组都是副本(改了不影响引擎)。
    ── 决策时点(信号日 t 收盘 / 尾盘截断)可得,决策规则只能用这一组 ──
      t, te, date(信号日), trade_date(成交日), phase, mode, capital, cost_model
      h_dec    float(N,):上一成交日 te-1 成交后、按 px[te-1] 计价的持仓(相对当时净值,含现金腿)。
               open10 下 px[te-1] = t 日开盘 TWAP,vwap1 下 = t 日全日 VWAP,尾盘口径 = t-1 日窗口价。
      nav_dec  按 px[te-1] 计价、扣完上一笔成本后的净值(起点 1.0);aum_dec = capital * nav_dec
      cap_w    float(N,):参与率上限的**事前估计**折算的单日最大 |Δw|
               = max_participation * 截至 te-1 的 20 日窗口成交额中位数 / (capital [* nav_dec])。
               引擎实际按 te 当天窗口成交额截量,record 的 part_cut 记录估计与实际的落差。
               (审查 P1:旧版直接给 te 实际窗口成交额,决策函数拿它做约束/重分配会虚增大资金结果)
      can      bool(N,):te 日可成交(te/te+1 均有价格、非一字板、窗口成交额>0)。
               注意"te+1 有价格"与"一字板用 te 全日振幅判定"是旧引擎沿用的约定(停牌一般事先公告),
               不是新增前视,但对集中持仓略有利。
      stuck    bool(N,):持有但 te 不可成交的残仓(判定同 can 的旧约定)
      budget_dec  1 - h_dec[stuck].sum():决策口径的可用预算
    ── 成交价 px[te] 计价(te 当天才知道;vwap1 下要到 te 收盘),只供"保持不动"的数值表达 ──
      h(位置参数)  成交前、按 px[te] 漂移后的持仓 —— 引擎实际从这里成交
      nav, aum     按 px[te] 计价的成交前净值(引擎算冲击 Q 与 participation_on_nav 的分母用它)
      stuck_w, budget  按 px[te] 计价的残仓权重与可用预算 1 - stuck_w(引擎 normalize 归一到它)
      ⚠ 不得用 h / nav 做判断(如换手带 |h-w|<band、按 h 算换手惩罚):vwap1 下等于用了 te 全日 VWAP。
        要"某只股票不交易"请在目标里给 NaN(见下),不要回传 h 的数值。
    返回 ndarray(N,) 目标权重(相对 te 成交前净值),或 None = 今天不交易(持仓自然漂移)。
      NaN  = 该股保持实际持仓、不交易(无论 h 漂到多少);其余目标在扣掉这些持仓后的预算内处理。
      负数 < -NEG_TOL 报错(引擎只做多头);-NEG_TOL ~ 0 的求解器噪声截零。
    target_fn 只允许使用信号日 t 及以前的信息,由调用方负责。ndarray 目标同样支持 NaN / 截零规则。

    cost_model=None 时成本 = (买+卖)*1bp + 卖*印花税,与改造前同一表达式(逐位一致);
    stamp_fn(lh 2026-10-11,默认 None = 上式):给定时印花税改为 买*b + 卖*s,(b, s) = stamp_fn(te)(te = 成交日行号),
    用于长历史分段税率(含 2008-09-19 前的买方印花税);返回 (0.0, 现行卖方税率) 时与默认路径逐位相同;
    给定时在其上**加**半价差与平方根冲击:cost_model.extra_cost(te, |Δw|, capital, nav)。

    record=True 时额外返回 res["record"]:
      daily     DataFrame[信号日] 列:ret, gross_ret, nav_prev, aum, turnover, trade_value(元),
                buy_value, sell_value, cost_total/commission/stamp/spread/impact(占净值比例),
                exposure, cash(=1-exposure), tgt_exposure, unfilled(调仓日 目标-实际 仓位差),
                desired_turnover, part_cut(被参与率截掉的 |Δw| 之和), part_bound_frac(截掉占意图比例),
                n_part_bound(触及上限的股票数), max_part(实际最大参与率,按 capital*nav 折算),
                stuck_w, n_hold, rebalance(是否调仓日)
      holdings  ndarray float32 (n_days x N),成交后持仓 newh(信号日 t 行);外层包成 DataFrame
    """
    em = EXEC_MODES[mode]
    dates = p.dates
    is_fn = callable(target)
    if is_fn:
        T, N = len(dates), len(p.codes)
    else:
        W = target
        if date_mask is not None:
            W = np.where(date_mask[:, None], W, 0)
        T, N = W.shape
        assert T == len(dates) and N == len(p.codes), f"W 形状 {W.shape} 与面板不符"
    mp = MAX_PARTICIPATION if max_participation is None else float(max_participation)

    px_df, amt_df = exec_prices(p, mode)
    px = px_df.to_numpy(np.float64)
    win_amt = amt_df.to_numpy(np.float64)
    dead = (p["one_word_up"] | p["one_word_down"]).to_numpy(bool)
    lag = em["lag"]

    cap_est = decision_window_amount(p, mode) if is_fn else None   # 只有 callable 需要事前估计

    h = np.zeros(N)
    nav_prev = 1.0
    # 决策口径的持仓与净值:上一轮成交后、按上一成交价 px[te-1] 计价(见 docstring)。
    h_dec = np.zeros(N)
    nav_dec = 1.0
    rets, ridx, turns, costs, held, exps = [], [], [], [], [], []
    rec_rows, rec_h = ([], []) if record else (None, None)
    stamp_switch = np.searchsorted(dates.values, np.datetime64(STAMP_TAX_CHANGE_DATE))
    # 调仓相位锚在 date_mask 起点,而非面板起点。否则逐年切片时年初会有
    # (-t_year) % rebalance 天恰好空仓(W[t]=0 且非调仓日),实测给 2025 年
    # 注入 +1.54pp、2023 年 -0.95pp 的伪超额,与被测超额同量级、足以翻转符号。
    # 锚在 mask 的**首个信号日**(不是首个成交日):该日 U 已被 mask 放行、W 有效,
    # 当天即可建仓。若减去 lag 会落到年前一天,那天 U 被 mask 掉、W=0,反而空仓 4 天。
    t0 = int(np.flatnonzero(date_mask)[0]) if date_mask is not None and date_mask.any() else 0
    t0 += int(phase)          # 相位偏移:phase=0..k-1 覆盖全部合法调仓日历
    for t in range(T - lag - 1):
        te = t + lag
        if te >= T - 1:
            break
        p_now, p_next = px[te], px[te + 1]
        ok_px = np.isfinite(p_now) & np.isfinite(p_next) & (p_now > 0)
        can = ok_px & ~dead[te] & (win_amt[te] > 0)
        is_reb = not (rebalance > 1 and ((t - t0) % rebalance))
        w_t = None
        if is_reb and is_fn:
            if date_mask is not None and not date_mask[t]:
                w_t = np.zeros(N)                        # mask 外不建仓(与 ndarray 分支同义)
            else:
                st_ = (~can) & (h > 1e-12)
                # cap_w 用事前估计(截至 te-1),**不是** te 实际窗口成交额(审查 P1)。
                cap_den = capital * (nav_dec if participation_on_nav else 1.0)
                ctx = dict(t=t, te=te, date=dates[t], trade_date=dates[te], phase=int(phase), mode=mode,
                           can=can.copy(), stuck=st_.copy(), stuck_w=float(h[st_].sum()),
                           budget=max(0.0, 1.0 - float(h[st_].sum())),
                           cap_w=_cap_weights(cap_est[te], mp, cap_den),
                           h_dec=h_dec.copy(), nav_dec=nav_dec, aum_dec=capital * nav_dec,
                           budget_dec=max(0.0, 1.0 - float(h_dec[st_].sum())),
                           nav=nav_prev, capital=capital, aum=capital * nav_prev, cost_model=cost_model)
                out = target(t, h.copy(), ctx)
                if out is None:
                    is_reb = False                       # 今天不交易,持仓自然漂移
                else:
                    w_t = np.asarray(out, dtype=np.float64)
                    if w_t.shape != (N,):
                        raise ValueError(f"target_fn 返回形状 {w_t.shape},应为 ({N},)")
        elif is_reb:
            w_t = W[t]
        stuck_w = 0.0; tgt_exp = np.nan; desired_abs = 0.0; part_cut = 0.0; n_bound = 0
        if not is_reb:
            delta = np.zeros(N)
        else:
            # 卖不掉的旧仓(停牌/一字板)必须先从预算里扣掉,剩下的才归一给可交易标的。
            # 旧写法把 tgt 在"可交易 ∪ 已持仓"上归一到 1,再把不可交易的 delta 抹成 0,
            # 于是卖不掉的仓位叠在已经满仓的新目标之上 -> sum(newh) > 1 的无成本杠杆。
            # 实测 15.3% 的交易日 exposure>1,最大 1.5375。
            stuck = (~can) & (h > 1e-12)
            budget = max(0.0, 1.0 - float(h[stuck].sum()))
            # 负权重:引擎只做多头。不防的话 newh 虽被截到 0,虚卖照样计费,normalize 还会把负数
            # 计入分母、放大多头(审查 P2:实测印花税 143->212bp、仓位 0.889->0.968)。
            if (w_t < 0).any():
                wmin = float(np.nanmin(w_t))
                if wmin < -NEG_TOL:
                    raise ValueError(f"目标权重含负数(最小 {wmin:.3g},t={t}):引擎只支持多头")
                w_t = np.where(w_t < 0, 0.0, w_t)       # 求解器噪声级负数截零
            if np.isinf(w_t).any():
                raise ValueError(f"目标权重含 inf(t={t})")
            if not np.isnan(w_t).any():
                # ---- 与改造前逐字相同的路径(金标准逐位一致) ----
                tgt = np.where(can, w_t.astype(np.float64), 0.0)
                s = tgt.sum()
                if normalize:
                    if s > 0:
                        tgt = tgt / s * budget
                elif s > budget:                             # normalize=False:允许留现金,只防超预算
                    tgt = tgt / s * budget
                tgt = tgt + np.where(stuck, h, 0.0)
            else:
                # NaN = 保持实际持仓、不交易。保持部分先占预算,其余目标在剩余预算里归一 / 压缩。
                hold = np.isnan(w_t) & can
                free = max(0.0, budget - float(h[hold].sum()))
                tgt = np.where(can & ~hold, np.nan_to_num(w_t.astype(np.float64), nan=0.0), 0.0)
                s = tgt.sum()
                if normalize:
                    if s > 0:
                        tgt = tgt / s * free
                elif s > free:
                    tgt = tgt / s * free
                tgt = tgt + np.where(stuck | hold, h, 0.0)
            delta = tgt - h
            pre_cap = delta
            if participation_on_nav:
                cap_w = _cap_weights(win_amt[te], mp, capital * nav_prev)
            else:
                cap_w = np.where(np.isfinite(win_amt[te]),
                                 mp * win_amt[te] / capital, 0.0)
            delta = np.clip(delta, -cap_w, cap_w)
            delta = np.where(can, delta, 0.0)
            # 卖出侧被参与率截断时,卖不完的残仓仍占资金;买单必须按剩余预算等比压缩,
            # 否则 sum(newh) > 1(审查 E3:实测最大 1.19,年化虚增 0.3~1.2pp)。
            over = float(np.clip(h + delta, 0, None).sum()) - 1.0
            if over > 1e-12:
                buys = np.clip(delta, 0, None); bs = buys.sum()
                if bs > 0:
                    delta = delta - buys * min(1.0, over / bs)
            if record:
                stuck_w = float(h[stuck].sum()); tgt_exp = float(tgt.sum())
                d_int = np.where(can, pre_cap, 0.0)
                desired_abs = float(np.abs(d_int).sum())
                cut = np.abs(d_int) - np.abs(np.clip(d_int, -cap_w, cap_w))
                part_cut = float(cut.sum()); n_bound = int((cut > 1e-12).sum())
        newh = np.clip(h + delta, 0, None)
        buy = np.clip(delta, 0, None).sum(); sell = np.clip(-delta, 0, None).sum()
        if stamp_fn is None:
            stamp = (STAMP_TAX_BPS_BEFORE if te < stamp_switch else STAMP_TAX_BPS_AFTER) * 1e-4
            stamp_cost = sell * stamp
            cost = (buy + sell) * COST_BPS_ONEWAY * 1e-4 + sell * stamp
        else:
            # lh 可选钩子(预注册 LONG_HISTORY_PREREG §14.1):stamp_fn(te) -> (买方税率, 卖方税率),按成交日行号
            sb_, ss_ = stamp_fn(te)
            stamp_cost = buy * sb_ + sell * ss_
            cost = (buy + sell) * COST_BPS_ONEWAY * 1e-4 + stamp_cost
        c_spread = c_impact = 0.0
        if cost_model is not None and (buy + sell) > 0:
            c_spread, c_impact = cost_model.extra_cost(te, np.abs(delta), capital, nav_prev)
            cost = cost + c_spread + c_impact
        with np.errstate(invalid="ignore"):
            r = np.where(ok_px, p_next / p_now - 1.0, 0.0)
        gross = float(np.nansum(newh * r))
        port_r = gross - cost
        # 保留现金腿:参与率上限没买进的部分留作现金,不得凭空补满(旧写法 grown/tot
        # 把未成交部分零成本、无约束地补齐,等于 MAX_PARTICIPATION 从未生效)。
        grown = newh * (1.0 + np.nan_to_num(r))
        h = grown / (1.0 + port_r) if (1.0 + port_r) > 1e-9 else grown
        hs = h.sum()
        if hs > 1.0 + 1e-12:          # 成本把现金扣成负 -> 实际是从持仓里出(审查 E3)
            h = h / hs
        # 收益打在**信号日 t**;benchmark 必须用同一索引口径(见 benchmark 的 shift(-lag))。
        # 旧版 benchmark 按成交日索引,主口径 lag=1 时整条曲线错开一天,
        # 实测同持仓同池的"超额"跑出 -3.18%/-1.24%/+10.17%,而非应有的 -成本。
        rets.append(port_r); ridx.append(dates[t])
        turns.append(np.abs(delta).sum() / 2); costs.append(cost)
        held.append(int((newh > 1e-8).sum())); exps.append(float(newh.sum()))
        assert newh.sum() <= 1.0 + 1e-9, f"exposure>1: {newh.sum():.6f} at t={t}"
        if record:
            aum = capital * nav_prev
            ad = np.abs(delta)
            with np.errstate(divide="ignore", invalid="ignore"):
                part = np.where((ad > 0) & (win_amt[te] > 0), ad * aum / win_amt[te], 0.0)
            ex = float(newh.sum())
            rec_rows.append((port_r, gross, nav_prev, aum, float(ad.sum() / 2), float((buy + sell) * aum),
                             float(buy * aum), float(sell * aum), float(cost),
                             float((buy + sell) * COST_BPS_ONEWAY * 1e-4), float(stamp_cost),
                             float(c_spread), float(c_impact), ex, 1.0 - ex, tgt_exp,
                             (max(0.0, tgt_exp - ex) if np.isfinite(tgt_exp) else np.nan),
                             desired_abs / 2, part_cut,
                             (part_cut / desired_abs if desired_abs > 1e-12 else np.nan), n_bound,
                             float(np.nanmax(part)) if part.size else 0.0, stuck_w,
                             int((newh > 1e-8).sum()), bool(is_reb)))
            rec_h.append(newh.astype(np.float32))
        if is_fn:
            # 下一轮的决策口径持仓 = 本轮成交后、按本轮成交价 px[te] 计价(下一轮的 px[te'-1]),
            # 相对扣完本轮成本后的净值。不含 px[te+1] 的任何信息。
            den = 1.0 - cost
            h_dec = newh / den if den > 1e-9 else newh.copy()
            hsd = h_dec.sum()
            if hsd > 1.0 + 1e-12:
                h_dec = h_dec / hsd
            nav_dec = nav_prev * (1.0 - cost)
        nav_prev = nav_prev * (1.0 + port_r)

    r_s = pd.Series(rets, index=pd.DatetimeIndex(ridx))
    turns, costs = np.asarray(turns), np.asarray(costs)
    held, exps = np.asarray(held), np.asarray(exps)
    keep = None
    if date_mask is not None:
        keep = pd.Series(date_mask[:len(dates)], index=dates).reindex(
            r_s.index).fillna(False).to_numpy(bool)
        if keep.sum() < 10:
            return dict(nav=(1.0 + r_s[keep]).cumprod(), n_days=int(keep.sum()))
        r_s = r_s[keep]
        turns, costs, held, exps = turns[keep], costs[keep], held[keep], exps[keep]
    nav = (1.0 + r_s).cumprod()
    yrs = len(r_s) / TRADING_DAYS
    res = dict(nav=nav, rets=r_s, ann_return=float(nav.iloc[-1] ** (1 / yrs) - 1),
               sharpe=float(r_s.mean() / r_s.std() * np.sqrt(TRADING_DAYS)) if r_s.std() > 0 else np.nan,
               max_dd=float((nav / nav.cummax() - 1).min()),
               turnover=float(turns.mean()), cost_bps=float(costs.mean() * 1e4),
               holdings=float(held.mean()), exposure=float(exps.mean()),
               n_days=len(r_s), vol=float(r_s.std() * np.sqrt(TRADING_DAYS)))
    res.update(_lr_fields(r_s, res["vol"], res["sharpe"]))      # 追加字段,原字段不动
    if record:
        daily = pd.DataFrame(rec_rows, columns=_REC_COLS, index=pd.DatetimeIndex(ridx))
        H = np.vstack(rec_h) if rec_h else np.zeros((0, N), np.float32)
        if keep is not None:
            daily, H = daily[keep], H[keep]
        daily.index.name = "date"
        res["record"] = dict(daily=daily, holdings=H, phase=int(phase), codes=p.codes,
                             meta=dict(mode=mode, capital=float(capital), rebalance=int(rebalance),
                                       max_participation=mp, participation_on_nav=bool(participation_on_nav),
                                       cost_model=(cost_model.describe() if cost_model is not None else None)))
    return res


_REC_COLS = ["ret", "gross_ret", "nav_prev", "aum", "turnover", "trade_value", "buy_value", "sell_value",
             "cost_total", "cost_commission", "cost_stamp", "cost_spread", "cost_impact",
             "exposure", "cash", "tgt_exposure", "unfilled", "desired_turnover", "part_cut",
             "part_bound_frac", "n_part_bound", "max_part", "stuck_w", "n_hold", "rebalance"]


def _cap_weights(amt_row: np.ndarray, mp: float, denom: float) -> np.ndarray:
    """参与率上限折算成单日最大 |Δw|。"""
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(np.isfinite(amt_row), mp * amt_row / denom, 0.0)


def _merge_records(recs: list[dict]) -> dict:
    """相位记录 -> 相位平均。daily 数值列取相位均值(与收益的相位平均同口径:k 个等额子组合),
    trade_value 等金额列同样取均值(= 每个子组合按全额 capital 模拟时的平均金额);
    holdings 取相位均值 = 合并组合的持仓。daily_by_phase 保留逐相位明细。"""
    by = pd.concat([r["daily"].assign(phase=r["phase"]) for r in recs])
    idx = recs[0]["daily"].index
    for r in recs[1:]:
        idx = idx.union(r["daily"].index)
    num = [c for c in _REC_COLS if c != "rebalance"]
    daily = by.groupby(level=0)[num].mean().reindex(idx)
    daily["n_rebalance_phases"] = by.groupby(level=0)["rebalance"].sum().reindex(idx)
    H = None
    for r in recs:
        hh = pd.DataFrame(r["holdings"], index=r["daily"].index).reindex(idx).fillna(0.0).to_numpy(np.float64)
        H = hh if H is None else H + hh
    H = (H / len(recs)).astype(np.float32)
    return dict(daily=daily, daily_by_phase=by, holdings=pd.DataFrame(H, index=idx, columns=recs[0]["codes"]),
                meta=dict(recs[0]["meta"], n_phases=len(recs)))


def record_to_frames(res: dict) -> dict:
    """单相位结果的 record 也包成与相位平均相同的结构(holdings -> DataFrame)。"""
    rec = res["record"]
    if isinstance(rec.get("holdings"), pd.DataFrame):
        return rec
    return _merge_records([rec])


def save_record(res: dict, prefix: str) -> dict:
    """把 record 落盘到 cache/_tmp/<prefix>_{holdings,daily,daily_by_phase}.parquet,返回路径。
    holdings 只存非零列(稀疏持仓省盘)。"""
    rec = record_to_frames(res)
    paths = {}
    H = rec["holdings"]
    H = H.loc[:, (H != 0).any(axis=0)]
    paths["holdings"] = TMP / f"{prefix}_holdings.parquet"; H.to_parquet(paths["holdings"])
    paths["daily"] = TMP / f"{prefix}_daily.parquet"; rec["daily"].to_parquet(paths["daily"])
    paths["daily_by_phase"] = TMP / f"{prefix}_daily_by_phase.parquet"
    rec["daily_by_phase"].to_parquet(paths["daily_by_phase"])
    return paths


def _pool_ew_drift_returns(p: Panel, version: str, mode: str, rebalance: int, t0: int = 0,
                           universe_override: pd.DataFrame | None = None) -> np.ndarray:
    """同频再平衡的池等权基准的逐相位日收益,返回 (k, T-1-lag) 数组,列 = 信号日 t。

    相位 ph 的调仓日 = {t : (t - t0 - ph) % k == 0}(与 simulate 的 is_reb 同一日历,t0 = mask 首日)。
    调仓日:按信号日 t 的决策池、在 te=t+lag 有持有收益(px[te], px[te+1] 均有)的股票上重置等权
            —— 与 k=1 基准"池内有收益的股票取均值"同一组股票;
    其余日:权重按 te->te+1 收益漂移 w <- w(1+g)/(1+r);当天无收益(停牌 / 缺价)的持仓收益记 0、权重保留
            (与引擎对持仓的处理一致,复牌跳空同样丢失)。
    尚未持仓(面板起点、或池子首次出现)时当天即建仓:基准始终满仓,**没有**策略那种"相位 k 前 k 天空仓"的冷启动;
    t0 之前同样按日历计算(相位在 mask 起点之前已预热)。k=1 时每天都是调仓日,与 benchmark() 的旧路径
    同一组股票、同一均值(浮点求和顺序不同,tests/test_engine_regression.py::test_samefreq_benchmark_implementation 核对)。"""
    k = max(int(rebalance), 1)
    px, _ = exec_prices(p, mode)
    lag = EXEC_MODES[mode]["lag"]
    T, N = len(p.dates), len(p.codes)
    P = px.reindex(index=p.dates, columns=p.codes).to_numpy(np.float64)
    U = (universe(p, version) if universe_override is None else universe_override).reindex(
        index=p.dates, columns=p.codes).fillna(False).to_numpy(bool)
    n_sig = T - 1 - lag
    with np.errstate(invalid="ignore", divide="ignore"):
        G = P[1 + lag:1 + lag + n_sig] / P[lag:lag + n_sig] - 1.0      # 行 t = te->te+1 收益,te = t+lag
    W = np.zeros((k, N))
    R = np.zeros((k, n_sig))
    for t in range(n_sig):
        g = G[t]
        fin = np.isfinite(g)
        gz = np.where(fin, g, 0.0)
        reb = ((t - t0 - np.arange(k)) % k) == 0
        empty = W.sum(axis=1) <= 0
        todo = reb | empty
        if todo.any():
            m = U[t] & fin
            n = int(m.sum())
            W[todo] = np.where(m, 1.0 / n, 0.0) if n > 0 else 0.0
        r = (W * gz).sum(axis=1)                      # 不用 @(见文件头注释)
        R[:, t] = r
        grow = 1.0 + r
        ok = grow > 1e-9
        W[ok] = W[ok] * (1.0 + gz) / grow[ok, None]
    return R


def benchmark(p: Panel, version: str, mode: str, date_mask=None, rebalance: int = 1,
              universe_override: pd.DataFrame | None = None) -> dict:
    """池等权基准(不含成本),用于算超额。

    口径与 run_strategy 对齐:按**信号日 t 的决策池**建仓,吃 te=t+lag -> te+1 的
    收益,收益打在**信号日 t**。二者必须同索引,否则逐年超额会带 1~10pp 伪信号。

    rebalance(2026-10-10 新增,前缀 fixa;默认 1 = 原行为,金标准逐位一致):
      k=1  每天按当日决策池重置等权(原口径)。
      k>1  **同频再平衡的池等权基准**:每 k 个信号日按当日决策池重置等权,其间权重随成交价漂移,
           k 个调仓相位的日收益取平均(与 run_strategy / run_weights 的相位平均同一口径,相位日历锚在
           date_mask 首日;k 个相位覆盖全部日历,所以平均与锚点无关)。不计成本、不加可成交约束,
           与 k=1 的语义相同,只改再平衡频率。细节见 _pool_ew_drift_returns。

    为什么需要 k>1(2026-10-10 审计,"高"级残留问题):k 日调仓的策略其间让权重漂移,k=1 的基准却每天
      重置等权。两者市场暴露相同,差别来自**个股相对池均值收益的截面序列相关**:相对收益有正的滞后相关时,
      "每天卖相对涨的、买相对跌的回到等权"系统性吃亏,于是**与选股无关**的再平衡频率差被记成策略超额。
      一阶展开:相位平均后每天差 ≈ (1/k) Σ_{l=1}^{k-1} (k−l)·γ_l(γ_l = 截面相对收益滞后 l 期的截面协方差),
      它解释了 2021-2025 实测差的 93~96%(研究期 Panel;分窗口 82~105%,research/ledger/fixa_r2_mech.csv pred_over_actual)。
      vwap1 是全日平均价口径,滞后 1 期截面相关 0.175(open10 0.037、tail1450 0.036;池等权序列一阶自相关
      0.308 / 0.068 只是旁证 —— open10 2021-2023 池序列只有 0.021,差仍有 +0.38pp),偏差最大:
      常数打分全池等权、旧成本、2020H2 起的连续路径切 2021-2025,5 日调仓对 k=1 基准的几何超额 vwap1 +1.99pp、
      open10 +0.61pp(日调仓只有 +0.07 / +0.30pp),对同频基准 −0.10 / +0.11pp;审计探针用 2021-2025 掩码
      (2021-01-04 冷启动)得到 +2.53 / +1.07pp,多出的约 0.5pp 是冷启动的相位空仓
      (research/ledger/fixa_fullpool_checks.csv)。所以"全池等权超额 ≈ -成本"这条引擎判据、
      以及 k 日调仓策略的"相对池等权"超额,都应对**同频**基准算:benchmark(..., rebalance=k)。
      ⚠ "vwap1 年化收益不受自相关影响"只对单条序列成立(VWAP 到 VWAP 可连乘还原);两个再平衡频率不同的
      组合之间的**差**会受自相关影响,这正是本参数要消掉的部分。
    策略在 date_mask 起点的冷启动(相位 k 前 k 个信号日空仓)不在基准里,仍会进入超额(与 k=1 时相同)。
    universe_override(fixa 新增,默认 None = 版本决策池,行为不变):信号日 t 的自定义池(T x N 布尔,
      如 PEAD 的合格池),与 run_strategy 的同名参数同义;k=1 / k>1 两条路径都适用。
    """
    if int(rebalance) > 1:
        return _benchmark_rebalanced(p, version, mode, date_mask, int(rebalance), universe_override)
    px, _ = exec_prices(p, mode)
    lag = EXEC_MODES[mode]["lag"]
    u = (universe(p, version) if universe_override is None else
         universe_override.reindex(index=p.dates, columns=p.codes).fillna(False).astype(bool))
    g = px.shift(-1) / px - 1                       # 索引在成交日 te
    m = g.where(u.shift(lag).fillna(False).astype(bool)).mean(axis=1)  # te 日、用信号日 t 的池
    r = m.shift(-lag).iloc[:len(p.dates) - 1 - lag].fillna(0.0)   # 搬回信号日 t
    if date_mask is not None:
        keep = pd.Series(date_mask[:len(p.dates)], index=p.dates).reindex(
            r.index).fillna(False).to_numpy(bool)
        r = r[keep]
    nav = (1 + r).cumprod()
    yrs = len(nav) / TRADING_DAYS
    out = dict(nav=nav, rets=r, ann_return=float(nav.iloc[-1] ** (1 / yrs) - 1),
               vol=float(r.std() * np.sqrt(TRADING_DAYS)),
               sharpe=float(r.mean() / r.std() * np.sqrt(TRADING_DAYS)) if r.std() > 0 else np.nan,
               max_dd=float((nav / nav.cummax() - 1).min()))
    out.update(_lr_fields(r, out["vol"], out["sharpe"]))     # 追加字段(vwap1 必看),原字段不动
    return out


def _benchmark_rebalanced(p: Panel, version: str, mode: str, date_mask, k: int,
                          universe_override: pd.DataFrame | None = None) -> dict:
    """benchmark(rebalance=k>1) 的实现:相位平均后的日收益 + 与 k=1 相同的统计字段。"""
    lag = EXEC_MODES[mode]["lag"]
    t0 = int(np.flatnonzero(date_mask)[0]) if date_mask is not None and np.asarray(date_mask).any() else 0
    R = _pool_ew_drift_returns(p, version, mode, k, t0, universe_override)
    r = pd.Series(R.mean(axis=0), index=p.dates[:len(p.dates) - 1 - lag])   # k 个相位等权 = 相位平均
    if date_mask is not None:
        keep = pd.Series(date_mask[:len(p.dates)], index=p.dates).reindex(
            r.index).fillna(False).to_numpy(bool)
        r = r[keep]
    nav = (1 + r).cumprod()
    yrs = len(nav) / TRADING_DAYS
    out = dict(nav=nav, rets=r, ann_return=float(nav.iloc[-1] ** (1 / yrs) - 1),
               vol=float(r.std() * np.sqrt(TRADING_DAYS)),
               sharpe=float(r.mean() / r.std() * np.sqrt(TRADING_DAYS)) if r.std() > 0 else np.nan,
               max_dd=float((nav / nav.cummax() - 1).min()), rebalance=k, n_phases=k,
               phase_ann=[float((1 + pd.Series(R[i], index=p.dates[:len(p.dates) - 1 - lag])[r.index]).prod()
                                ** (1 / yrs) - 1) for i in range(k)])
    out.update(_lr_fields(r, out["vol"], out["sharpe"]))
    return out
