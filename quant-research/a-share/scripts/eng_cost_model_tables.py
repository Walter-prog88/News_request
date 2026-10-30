"""成本模型与引擎改造的账本表(前缀 eng,2026-10-10)。

产出(全部 research/ledger/,research/ENGINE_IMPACT_MODEL.md 里的每个数字都出自这里):
  eng_spread_by_size.csv        市值 / ADV 十分位的半价差(bp):一跳下限、AR 逐股、AR two-day;
                                AR 两日乘积的十分位合并均值与 Newey-West t(有效性诊断)
  eng_spread_ar_diagnostics.csv AR 逐股估计的零值比例、与流动性的截面秩相关、桶合并后为正的天数比例
  eng_impact_examples.csv       1000 万 / 1 亿 / 10 亿资金买入 1% 权重的冲击(bp),按 ADV 十分位
  eng_engine_checks.csv         引擎回归判据:全池等权 vs 池基准残差(逐年、三口径、两版本)、
                                COMP_v5 的 2020 预热空仓伪超额、backtest/strategy 一致性
  eng_vwap_checks.csv           vwap1 口径原料核实:与 twap_open10 的比值、amount 单位、除权日不跳;
                                外部佐证 ext_*:用新浪日成交量(换手率×流通股本)反推真实 VWAP 对照(2026-10-10 审查 P2);
                                vwap∈[low,high] **按构造恒成立**,只防原料错位,不作口径证据
  eng_cost_demo.csv             接口演示:COMP_v5 两个方案在 1000 万资金下,旧成本 vs 冲击模型(Y 三档、
                                (2026-10-10 审查后追加:价差/冲击分项年化、vol_adj/sharpe_adj、TE/IR 日频与 NW20 口径)
                                价差上下界)、open10 vs vwap1。⚠ COMP_v5 成员在 2020-2025 内选出,
                                结果是**信号样本内**的,只说明成本层量级,不是信号的样本外能力

研究期面板(2020-01-02 ~ 2025-12-31),未触碰封存数据;数据不含已退市股(幸存者偏差)。
"""
from __future__ import annotations
import os, sys, time, warnings
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

from afactor.config import LEDGER, TMP
from afactor.panel import Panel
from afactor import strategy as ST, backtest as BT, impact as IM, factors as FA
from afactor.strategy import TRADING_DAYS

PERIODS = {"2021-2025": (2021, 2025), "2021-2023": (2021, 2023), "2024-2025": (2024, 2025)}
CAPITALS = (1e7, 1e8, 1e9)


def nw_t(x, q: int = 5) -> float:
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    n = len(x)
    if n < 30:
        return np.nan
    e = x - x.mean()
    s = float((e ** 2).sum() / n)
    for L in range(1, q + 1):
        s += 2.0 * (1.0 - L / (q + 1.0)) * float((e[L:] * e[:-L]).sum() / n)
    return float(x.mean() / np.sqrt(s / n)) if s > 0 else np.nan


def floor_bind(est: np.ndarray, floor: np.ndarray) -> float:
    """逐股 AR 估计(未取下限)<= 一跳下限的比例 = 下限生效的比例。"""
    ok = np.isfinite(est) & np.isfinite(floor)
    return float(np.mean(est[ok] <= floor[ok]))


def deciles(key: pd.DataFrame, mask: pd.DataFrame) -> np.ndarray:
    """当日池内按 key 分十分位(1 = 最小)。"""
    q = key.where(mask).rank(axis=1, pct=True)
    return np.ceil(q * 10).clip(1, 10).to_numpy()


def spread_tables(p: Panel) -> tuple[pd.DataFrame, pd.DataFrame]:
    U = ST.universe(p, "V1000")
    yrs = p.dates.year
    floor = IM.tick_floor(p)
    S_tick = IM.ar_spread(p, method="tick")
    S_win_nf = IM.ar_spread(p, method="ar_window", floor=False)
    S_win = np.fmax(S_win_nf, floor).where(S_win_nf.notna())
    S_2d = IM.ar_spread(p, method="ar_2day", floor=True)
    prod = IM.ar_products(p)
    raw = p.raw("close_px")  # 仅用于非空判断
    from afactor import newdata as ND
    rawc = ND.raw_close(p.dates, p.codes)
    keys = {"size": FA._real(p, "mv"), "adv": p["adv20"]}
    Uv = U.to_numpy(bool)
    rows = []
    arr = {k: v.to_numpy(np.float64) for k, v in
           dict(tick=S_tick, arwin=S_win, arwin_nf=S_win_nf, ar2d=S_2d, prod=prod, rawc=rawc).items()}
    for btype, key in keys.items():
        D = deciles(key, U)
        kv = key.to_numpy(np.float64)
        for pname, (y0, y1) in PERIODS.items():
            sel = np.asarray((yrs >= y0) & (yrs <= y1))
            for d in range(1, 11):
                m = (D == d) & Uv & sel[:, None]
                def v(name):
                    x = arr[name][m]; return x[np.isfinite(x)]
                tick, arwin, arwin_nf, ar2d = v("tick"), v("arwin"), v("arwin_nf"), v("ar2d")
                # AR 两日乘积:每天先取桶内截面均值(一天一个数),再对时间序列做 NW t(带宽 5)
                pm = np.where(m, arr["prod"], np.nan)
                with np.errstate(invalid="ignore"):
                    daily = np.nanmean(pm[sel], axis=1)
                daily = daily[np.isfinite(daily)]
                rows.append(dict(
                    period=pname, bucket_type=btype, decile=d, n_obs=int(m.sum()),
                    key_median=float(np.nanmedian(kv[m])),
                    key_unit=("元(总市值)" if btype == "size" else "元(20日ADV)"),
                    raw_close_median=float(np.nanmedian(arr["rawc"][m])),
                    hs_tick_med_bp=float(np.median(tick) / 2 * 1e4),
                    hs_tick_mean_bp=float(np.mean(tick) / 2 * 1e4),
                    hs_arwin_med_bp=float(np.median(arwin) / 2 * 1e4),
                    hs_arwin_mean_bp=float(np.mean(arwin) / 2 * 1e4),
                    hs_arwin_p25_bp=float(np.percentile(arwin, 25) / 2 * 1e4),
                    hs_arwin_p75_bp=float(np.percentile(arwin, 75) / 2 * 1e4),
                    hs_ar2day_med_bp=float(np.median(ar2d) / 2 * 1e4),
                    arwin_zero_share=float(np.mean(arwin_nf == 0)),
                    arwin_floor_bind_share=floor_bind(arr["arwin_nf"][m], arr["tick"][m]),
                    pooled_prod_mean=float(daily.mean()), pooled_prod_t=nw_t(daily, 5),
                    pooled_implied_hs_bp=float(np.sqrt(max(daily.mean(), 0.0)) / 2 * 1e4)))
    by_size = pd.DataFrame(rows)

    # ---- 诊断:截面秩相关、桶合并为正的天数比例 ----
    from scipy.stats import spearmanr
    diag = []
    sel = np.asarray((yrs >= 2021) & (yrs <= 2025))
    nlogadv = -np.log(p["adv20"])
    cs_adv, cs_tick = [], []
    for i in np.flatnonzero(sel):
        u = Uv[i]
        a = arr["arwin_nf"][i]; b = nlogadv.to_numpy()[i]; c = arr["tick"][i]
        ok = u & np.isfinite(a) & np.isfinite(b) & np.isfinite(c)
        if ok.sum() > 50:
            cs_adv.append(spearmanr(a[ok], b[ok])[0]); cs_tick.append(spearmanr(a[ok], c[ok])[0])
    diag.append(dict(metric="spearman(AR逐股20日, -lnADV) 日均", window="2021-2025", value=float(np.nanmean(cs_adv)),
                     nw_t=nw_t(np.asarray(cs_adv), 5)))
    diag.append(dict(metric="spearman(AR逐股20日, 一跳价差) 日均", window="2021-2025", value=float(np.nanmean(cs_tick)),
                     nw_t=nw_t(np.asarray(cs_tick), 5)))
    x = arr["arwin_nf"][Uv & sel[:, None]]; x = x[np.isfinite(x)]
    diag.append(dict(metric="AR逐股20日 被截零的股票日比例(V1000池)", window="2021-2025", value=float(np.mean(x == 0)), nw_t=np.nan))
    for w in (20, 60, 120):
        Sw = IM.ar_spread(p, window=w, min_periods=w // 2, method="ar_window", floor=False).to_numpy(np.float64)
        x = Sw[Uv & sel[:, None]]; x = x[np.isfinite(x)]
        diag.append(dict(metric=f"AR逐股{w}日 被截零比例", window="2021-2025", value=float(np.mean(x == 0)), nw_t=np.nan))
    # ADV 十分位桶内合并:每天桶内两日乘积截面均值,再按 w 日滚动平均,统计为正的天数比例
    D = deciles(p["adv20"], U)
    for j in (1, 5, 10):
        with np.errstate(invalid="ignore"):
            daily = pd.Series(np.nanmean(np.where((D == j) & Uv, arr["prod"], np.nan), axis=1), index=p.dates)
        for w in (20, 250):
            mu = daily.rolling(w, min_periods=w // 2).mean()
            diag.append(dict(metric=f"ADV第{j}十分位 桶内合并{w}日均值>0 的天数比例", window="2021-2025",
                             value=float((mu[sel] > 0).mean()), nw_t=np.nan))
    return by_size, pd.DataFrame(diag)


def impact_examples(p: Panel) -> pd.DataFrame:
    cm = IM.CostModel(p, impact_y=1.0)                       # 默认:一跳价差
    cm_w = IM.CostModel(p, impact_y=1.0, spread_method="ar_window")
    yrs = p.dates.year
    U_t = ST.universe(p, "V1000").to_numpy(bool)
    te_rows = np.flatnonzero(np.asarray((yrs >= 2021) & (yrs <= 2025)))
    te_rows = te_rows[te_rows >= 1]
    pool = U_t[te_rows - 1]                                 # 信号日 t = te-1 的决策池
    adv = cm._adv[te_rows].astype(np.float64); sig = cm._sigma[te_rows].astype(np.float64)
    hs_t = cm._hs[te_rows].astype(np.float64); hs_w = cm_w._hs[te_rows].astype(np.float64)
    a10 = p.raw("amt_open10").to_numpy(np.float64)[te_rows]; aday = p.raw("amount").to_numpy(np.float64)[te_rows]
    q = np.where(pool, adv, np.nan).copy()
    rk = pd.DataFrame(q).rank(axis=1, pct=True).to_numpy()
    D = np.ceil(rk * 10).clip(1, 10)
    rows = []
    for cap in CAPITALS:
        Q = 0.01 * cap
        for d in list(range(1, 11)) + ["all"]:
            m = pool & (np.isfinite(D) if d == "all" else (D == d))
            r = dict(capital=cap, trade_weight=0.01, trade_value=Q, adv_decile=d, n_obs=int(m.sum()),
                     adv_median=float(np.median(adv[m])), sigma_median=float(np.median(sig[m])),
                     q_over_adv_median=float(np.median(Q / adv[m])))
            for y in IM.PREREG_Y:
                r[f"impact_bp_Y{y}"] = float(np.median(y * sig[m] * np.sqrt(Q / adv[m])) * 1e4)
            r["hs_tick_bp"] = float(np.median(hs_t[m]) * 1e4)
            r["hs_arwin_bp"] = float(np.median(hs_w[m]) * 1e4)
            r["buy_cost_bp_Y1_tick"] = float(np.median(1.0 + hs_t[m] * 1e4 + sig[m] * np.sqrt(Q / adv[m]) * 1e4))
            r["buy_cost_bp_Y1_arwin"] = float(np.median(1.0 + hs_w[m] * 1e4 + sig[m] * np.sqrt(Q / adv[m]) * 1e4))
            with np.errstate(invalid="ignore", divide="ignore"):
                r["share_cap_bind_open10"] = float(np.mean(Q > 0.10 * a10[m]))
                r["share_cap_bind_vwap1"] = float(np.mean(Q > 0.10 * aday[m]))
            rows.append(r)
    return pd.DataFrame(rows)


def engine_checks(p: Panel, score: pd.DataFrame) -> pd.DataFrame:
    rows = []
    yrs = p.dates.year
    ones = pd.DataFrame(1.0, index=p.dates, columns=p.codes, dtype=np.float32)
    for mode in ("open10", "tail1450", "vwap1"):
        for ver in ("V1000", "V100"):
            for y in [None] + list(range(2020, 2026)):
                m = None if y is None else np.asarray(yrs == y)
                s = ST.run_strategy(ones, p, ver, mode, scheme="exclude", exclude_frac=0.0, rebalance=1, date_mask=m)
                b = ST.benchmark(p, ver, mode, date_mask=m)
                assert s["rets"].index.equals(b["rets"].index)
                cost = s["cost_bps"] * 1e-4 * TRADING_DAYS
                rows.append(dict(check="fullpool_resid_pp", score="const", mode=mode, version=ver,
                                 window="2020-2025" if y is None else str(y),
                                 value=(s["ann_return"] - b["ann_return"] + cost) * 100,
                                 note=f"excess={100*(s['ann_return']-b['ann_return']):.3f}pp cost={100*cost:.3f}pp exposure={s['exposure']:.4f}"))
                if y is None:
                    cs = {k: float(s["rets"].corr(b["rets"].shift(k))) for k in (-2, -1, 0, 1, 2)}
                    rows.append(dict(check="align_peak_k", score="const", mode=mode, version=ver, window="2020-2025",
                                     value=float(max(cs, key=cs.get)), note=str({k: round(v, 4) for k, v in cs.items()})))
    # COMP_v5 预热:打分在 2020-02-06 前缺失 -> 策略空仓,而池基准满仓经历了 2020-02-03 的暴跌
    U = ST.universe(p, "V1000")
    cov = (score.notna() & U).sum(axis=1) / U.sum(axis=1)
    first = cov[cov >= 0.5].index[0]
    rows.append(dict(check="comp_v5_first_scored_day", score="COMP_v5", mode="open10", version="V1000",
                     window="2020", value=float(cov.loc[first]), note=f"首个池覆盖>=50%的日期 {first.date()}"))
    for y in range(2020, 2026):
        m = np.asarray(yrs == y)
        s = ST.run_strategy(score, p, "V1000", "open10", scheme="exclude", exclude_frac=0.0, rebalance=1, date_mask=m)
        b = ST.benchmark(p, "V1000", "open10", date_mask=m)
        cost = s["cost_bps"] * 1e-4 * TRADING_DAYS
        rows.append(dict(check="fullpool_resid_pp", score="COMP_v5(预热缺失)", mode="open10", version="V1000",
                         window=str(y), value=(s["ann_return"] - b["ann_return"] + cost) * 100, note=""))
    s = ST.run_strategy(score, p, "V1000", "open10", scheme="exclude", buffer=(0.20, 0.35), rebalance=5)
    b = ST.benchmark(p, "V1000", "open10")
    pre = s["rets"].index < first
    d = (s["rets"] - b["rets"])[pre]
    rows.append(dict(check="comp_v5_std_prewarmup_excess_pp", score="COMP_v5", mode="open10", version="V1000",
                     window=f"2020-01-02~{(first - pd.Timedelta(days=1)).date()}",
                     value=float(((1 + s["rets"][pre]).prod() - (1 + b["rets"][pre]).prod()) * 100),
                     note=f"标准排雷方案在打分缺失期空仓的累计伪超额;策略{100*((1+s['rets'][pre]).prod()-1):.2f}% vs 基准{100*((1+b['rets'][pre]).prod()-1):.2f}%;逐日差之和 {d.sum()*100:.2f}pp"))
    s2 = ST.run_strategy(score, p, "V1000", "open10", scheme="exclude", buffer=(0.20, 0.35), rebalance=5,
                         date_mask=np.asarray(p.dates >= first))
    b2 = ST.benchmark(p, "V1000", "open10", date_mask=np.asarray(p.dates >= first))
    rows.append(dict(check="comp_v5_std_ann_from_first_scored_day", score="COMP_v5", mode="open10", version="V1000",
                     window=f"{first.date()}~2025-12-31", value=s2["ann_return"] * 100,
                     note=f"基准 {b2['ann_return']*100:.3f}%;全期口径为 策略 {s['ann_return']*100:.3f}% / 基准 {b['ann_return']*100:.3f}%"))
    rows.append(dict(check="comp_v5_std_bench_from_first_scored_day", score="COMP_v5", mode="open10", version="V1000",
                     window=f"{first.date()}~2025-12-31", value=b2["ann_return"] * 100, note=""))
    # backtest.py 与 strategy.py 同配置
    a = BT.run_standard(score, p)
    c = ST.run_strategy(score, p, "V1000", "open10", scheme="exclude", buffer=(0.20, 0.35), rebalance=5, avg_phases=False)
    rows.append(dict(check="backtest_vs_strategy_ann_diff_pp", score="COMP_v5", mode="open10", version="V1000",
                     window="2020-2025", value=(a["ann_return"] - c["ann_return"]) * 100,
                     note=f"逐日收益逐位相等={np.array_equal(a['rets'].to_numpy(), c['rets'].to_numpy())}"))
    return pd.DataFrame(rows)


def cost_demo(p: Panel, score: pd.DataFrame) -> pd.DataFrame:
    yrs = p.dates.year
    models = {"legacy(1bp+印花税)": None,
              "Y0.5_tick": IM.CostModel(p, impact_y=0.5), "Y1.0_tick": IM.CostModel(p, impact_y=1.0),
              "Y1.5_tick": IM.CostModel(p, impact_y=1.5),
              "Y1.0_2tick": IM.CostModel(p, impact_y=1.0, spread_mult=2.0),
              "Y1.0_arwindow": IM.CostModel(p, impact_y=1.0, spread_method="ar_window")}
    plans = {"exclude_std": dict(scheme="exclude", buffer=(0.20, 0.35), rebalance=5),
             "top20_r5": dict(scheme="top", n=20, rebalance=5)}
    windows = {"2021-2025": (2021, 2025), "2021-2023": (2021, 2023), "2024-2025": (2024, 2025)}
    rows = []
    for mode in ("open10", "vwap1"):
        for wname, (y0, y1) in windows.items():
            m = np.asarray((yrs >= y0) & (yrs <= y1))
            b = ST.benchmark(p, "V1000", mode, date_mask=m)
            for pname, plan in plans.items():
                for mname, cm in models.items():
                    if mode == "vwap1" and mname not in ("legacy(1bp+印花税)", "Y1.0_tick"):
                        continue
                    t0 = time.time()
                    r = ST.run_strategy(score, p, "V1000", mode, date_mask=m, cost_model=cm, record=True, **plan)
                    dd = r["record"]["daily"]
                    es = ST.excess_stats(r["rets"], b["rets"])      # 日频与 NW20 两种口径的 TE / IR
                    rows.append(dict(mode=mode, window=wname, plan=pname, cost_model=mname, capital=1e7,
                                     ann_return=r["ann_return"], bench_ann=b["ann_return"],
                                     excess_ann=r["ann_return"] - b["ann_return"], sharpe=r["sharpe"],
                                     max_dd=r["max_dd"], turnover=r["turnover"], exposure=r["exposure"],
                                     cost_bps_per_day=r["cost_bps"],
                                     commission_bps_per_day=dd["cost_commission"].mean() * 1e4,
                                     stamp_bps_per_day=dd["cost_stamp"].mean() * 1e4,
                                     spread_bps_per_day=dd["cost_spread"].mean() * 1e4,
                                     impact_bps_per_day=dd["cost_impact"].mean() * 1e4,
                                     cost_ann_pp=r["cost_bps"] * 1e-4 * TRADING_DAYS * 100,
                                     commission_ann_pp=dd["cost_commission"].mean() * TRADING_DAYS * 100,
                                     stamp_ann_pp=dd["cost_stamp"].mean() * TRADING_DAYS * 100,
                                     spread_ann_pp=dd["cost_spread"].mean() * TRADING_DAYS * 100,
                                     impact_ann_pp=dd["cost_impact"].mean() * TRADING_DAYS * 100,
                                     vol=r["vol"], vol_adj=r["vol_adj"], sharpe_adj=r["sharpe_adj"], ret_ac1=r["ret_ac1"],
                                     te=es["te"], ir=es["ir"], te_adj=es["te_adj"], ir_adj=es["ir_adj"],
                                     excess_ac1=es["excess_ac1"],
                                     part_bound_frac_rebal=float(r["record"]["daily_by_phase"].query("rebalance")["part_bound_frac"].mean()),
                                     unfilled_rebal=float(r["record"]["daily_by_phase"].query("rebalance")["unfilled"].mean()),
                                     note=("COMP_v5 信号样本内;只用于说明成本层量级"
                                           + (";vwap1 平均价口径:sharpe/ir 日频口径高估,看 sharpe_adj/ir_adj" if mode == "vwap1" else ""))))
                    print(f"  {mode} {wname} {pname} {mname}: ann {r['ann_return']*100:.2f}% cost {r['cost_bps']:.2f}bp/d {time.time()-t0:.1f}s", flush=True)
    return pd.DataFrame(rows)


def vwap_checks(p: Panel) -> pd.DataFrame:
    """vwap1 口径原料核实:vwap 落在 [low, high] 内、与 twap_open10 同一复权口径、amount 单位为元。"""
    vw, tw, hi, lo, am = (p.raw(k) for k in ("vwap", "twap_open10", "high_px", "low_px", "amount"))
    ok = vw.notna() & hi.notna() & lo.notna()
    inside = ((vw >= lo * (1 - 1e-4)) & (vw <= hi * (1 + 1e-4))) | ~ok
    r = (vw / tw).to_numpy(); r = r[np.isfinite(r)]
    q = np.quantile(r, [0.01, 0.5, 0.99])
    rv = (vw / vw.shift(1) - 1).to_numpy(); rc = (p.raw("close_px") / p.raw("close_px").shift(1) - 1).to_numpy()
    m = np.isfinite(rv) & np.isfinite(rc)
    rows = [dict(check="vwap_obs", value=float(ok.to_numpy().sum()), note="vwap/high/low 同时有效的股票日数"),
            dict(check="vwap_inside_low_high_share", value=float(inside.to_numpy().mean()),
                 note="vwap 在 [low,high](±1bp)内的比例。按构造恒成立(bar 中价的成交额加权调和平均必落在同一套 bar 的 max/min 内),只防原料错位,不作口径证据"),
            dict(check="vwap_over_twap_open10_q01", value=float(q[0]), note=""),
            dict(check="vwap_over_twap_open10_median", value=float(q[1]), note=""),
            dict(check="vwap_over_twap_open10_q99", value=float(q[2]), note=""),
            dict(check="share_abs_vwap_ret_gt_35pct", value=float((np.abs(rv[m]) > 0.35).mean()),
                 note="vwap 日收益 |r|>35% 的比例;前复权一致时除权日不跳"),
            dict(check="amount_600519_2024_median_yuan", value=float(am.loc["2024", "600519.SH"].median()),
                 note="单位核实:元")]
    rows += vwap_external_check(p)
    return pd.DataFrame(rows)


def vwap_external_check(p: Panel) -> list[dict]:
    """外部佐证(审查 P2):分钟数据没有成交量,vwap 用 Σamt/Σ(amt/mid) 近似"股数";新浪日线有真实成交量
    (daily_basic: 换手率% = 成交量/流通股本×100,float_share 单位万股,见 data_8:21/DATA_ACQUISITION.md),
    于是 真实 VWAP(未复权)= 全日成交额 amount / (换手率/100 × 流通股本)。复权因子在同一天的比值里约掉:
        比较 vwap/close_px(本项目,前复权)与 VWAP_ext/未复权收盘(新浪成交量 + 本项目成交额)。
    只读 trade_date <= 2025-12-31 的行(封存纪律),再对齐到研究期面板。"""
    from afactor.newdata import DATA
    d = pd.read_csv(DATA / "daily_basic.csv", usecols=["ts_code", "trade_date", "close", "turnover_rate", "float_share"])
    d = d[d["trade_date"] <= 20251231]
    d["trade_date"] = pd.to_datetime(d["trade_date"].astype(str), format="%Y%m%d")
    piv = lambda c: d.pivot(index="trade_date", columns="ts_code", values=c).reindex(
        index=p.dates, columns=p.codes).astype(np.float64)
    vol_sh = piv("turnover_rate") / 100.0 * piv("float_share") * 1e4          # 股
    raw_c = piv("close")
    del d
    amt = p.raw("amount").astype(np.float64)
    vw, cp = p.raw("vwap").astype(np.float64), p.raw("close_px").astype(np.float64)
    ext = (amt / vol_sh) / raw_c
    own = vw / cp
    ok = (np.isfinite(ext) & np.isfinite(own) & (vol_sh > 0) & (amt > 0) & (raw_c > 0)).to_numpy()
    dev = (own / ext - 1.0).to_numpy()
    U = ST.universe(p, "V1000").to_numpy(bool)
    yrs = np.asarray(p.dates.year)[:, None]
    out = []
    for lab, m in (("all_2020_2025", ok), ("v1000_2021_2025", ok & U & (yrs >= 2021))):
        x = dev[m]; ax = np.abs(x)
        out += [dict(check=f"ext_{lab}_n", value=float(m.sum()), note="vwap、新浪成交量、未复权收盘同时有效的股票日"),
                dict(check=f"ext_{lab}_dev_median", value=float(np.median(x)),
                     note="(vwap/close_px)/(VWAP_ext/未复权收盘)-1 的中位数;VWAP_ext=amount/新浪成交量"),
                dict(check=f"ext_{lab}_absdev_q50", value=float(np.quantile(ax, 0.5)), note=""),
                dict(check=f"ext_{lab}_absdev_q90", value=float(np.quantile(ax, 0.9)), note=""),
                dict(check=f"ext_{lab}_absdev_q99", value=float(np.quantile(ax, 0.99)), note=""),
                dict(check=f"ext_{lab}_share_absdev_lt_10bp", value=float(np.mean(ax < 1e-3)), note=""),
                dict(check=f"ext_{lab}_share_absdev_lt_50bp", value=float(np.mean(ax < 5e-3)), note=""),
                dict(check=f"ext_{lab}_share_absdev_gt_2pct", value=float(np.mean(ax > 2e-2)), note="")]
    # 对照:同一比值下 twap_open10 与真实 VWAP 的差(说明 vwap 不是"随便一个日内价")
    tw = p.raw("twap_open10").astype(np.float64)
    dv2 = ((tw / cp) / ext - 1.0).to_numpy()
    m2 = ok & np.isfinite(dv2)
    out.append(dict(check="ext_all_2020_2025_twap_open10_absdev_q50", value=float(np.quantile(np.abs(dv2[m2]), 0.5)),
                    note="对照:twap_open10 与真实 VWAP 的偏离中位数"))
    return out


def main():
    t0 = time.time()
    p = Panel()
    vc = vwap_checks(p)
    vc.to_csv(LEDGER / "eng_vwap_checks.csv", index=False)
    print(vc.to_string(), flush=True)
    if "--only-vwap" in sys.argv:
        return
    score = pd.read_parquet(TMP / "comp_v5_score_open10.parquet")
    if "--only-demo" in sys.argv:
        demo = cost_demo(p, score)
        demo.to_csv(LEDGER / "eng_cost_demo.csv", index=False)
        print(f"done {time.time()-t0:.0f}s")
        return
    by_size, diag = spread_tables(p)
    by_size.to_csv(LEDGER / "eng_spread_by_size.csv", index=False)
    diag.to_csv(LEDGER / "eng_spread_ar_diagnostics.csv", index=False)
    print(by_size.query("period=='2021-2025'")[["bucket_type", "decile", "hs_tick_med_bp", "hs_arwin_med_bp",
                                                  "hs_arwin_mean_bp", "arwin_zero_share", "pooled_prod_t"]].to_string())
    print(diag.to_string()); print(f"spread {time.time()-t0:.0f}s", flush=True)
    ex = impact_examples(p)
    ex.to_csv(LEDGER / "eng_impact_examples.csv", index=False)
    print(ex[["capital", "adv_decile", "adv_median", "q_over_adv_median", "impact_bp_Y1.0", "hs_tick_bp",
              "buy_cost_bp_Y1_tick", "share_cap_bind_open10"]].to_string()); print(f"impact {time.time()-t0:.0f}s", flush=True)
    ck = engine_checks(p, score)
    ck.to_csv(LEDGER / "eng_engine_checks.csv", index=False)
    print(ck.to_string()); print(f"checks {time.time()-t0:.0f}s", flush=True)
    demo = cost_demo(p, score)
    demo.to_csv(LEDGER / "eng_cost_demo.csv", index=False)
    print(demo[["mode", "window", "plan", "cost_model", "ann_return", "excess_ann", "cost_ann_pp",
                "spread_bps_per_day", "impact_bps_per_day"]].to_string())
    print(f"done {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
