"""PEAD 第 4 步:事件层诊断(预注册 7.5 h10、7.6 价格条件、7.7 新鲜度、7.8 经典事件研究图)。

这些都是**理论收益**(TWAP 窗口价或收盘价),不施加整手/参与率/成本,只作诊断,不拼净值、不年化
(h5 事件层配对差另给"每批平均"口径,方便与账户层对照)。

口径:
  事件层收益 r_h(T, i) = twap_open10[T+h] / twap_open10[T] − 1;T 日买不进(无价/一字板/开盘窗口无量)记 0;
              T+h 无价(停牌)用 T..T+h 内最近一个有价日的价格(近似"复牌前卖不掉")。
  配对超额 e_i = r(E 入选股) − mean(其 3 个对照的 r)(B 或 C;匹配失败的对照不计入均值)。
  每个 T 求平均得 D_T;窗口按 t_signal 切;NW 带宽 h5 用 10、h10 用 20(预注册 7.5/7.6)。
  价格条件 rel20_i = Π(1+r_i,d) − Π(1+m_d),d 取截至 t_signal 的 20 个交易日,r 为 ret_c2c(停牌记 0
  且为无效日,有效日 < 15 不参与分组),m_d 为 d 日合格池(按 d−1 信号行判断)等权 ret_c2c;
  按当期**事件池**中位数分低/高;D_T = mean_低(e) − mean_高(e),两组各 >= 3 只才计入。
  事件研究:主事件(V >= 2021-01-04 且 V+60 不超过面板末日),按 V 所在季度把 fu_op_yoy 分五等分;
  AR_d = ret_c2c − 当日合格池等权 ret_c2c;CAR(τ) = Σ_{s=-5..τ} AR,τ ∈ [-5, 60];缺行情的 AR 记 0 并计数;
  95% 区间以 V 所在季度为聚类单位。V 日收益含公告后的隔夜跳空,不可交易,另把它拆成
  "前收盘 -> V 日开盘 TWAP"(不可交易:跳空 + 开盘 10 分钟)与"V 日 TWAP -> V 日收盘"(可交易)两段。
  时钟诊断(预注册外,只描述;审查 P2 补充):τ=−1 同样拆两段,并给"τ=−1 开盘 TWAP -> V 日开盘 TWAP"
  (若在 P 日开盘入场能多拿到的部分)。
产出:research/ledger/pead_car.csv、pead_car_summary.csv、pead_event_level.csv;
     reports/pead_car.png(同图另存 reports/pead_event_study_quintiles.png,预注册 7.8 的文件名)。
"""
from __future__ import annotations
import os, sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from afactor.config import TMP, LEDGER, REPORTS
from afactor.panel import Panel
from afactor import earnings_events as EE, pead_engine as PE

WINDOWS = {"dev": ("2021-01-01", "2023-12-31"), "test": ("2024-01-01", "2025-12-31"),
           "all2125": ("2021-01-01", "2025-12-31")}
TAUS = np.arange(-5, 61)


def win_mask(idx, w):
    a, b = WINDOWS[w]
    idx = pd.DatetimeIndex(idx)
    return (idx >= pd.Timestamp(a)) & (idx <= pd.Timestamp(b))


def cluster_mean_se(x: np.ndarray, g: np.ndarray):
    """x (n, k) 的列均值与以 g 为聚类的稳健标准误(CR1 小样本修正)。"""
    n = x.shape[0]
    mu = x.mean(axis=0)
    e = x - mu
    G = pd.Series(np.arange(n)).groupby(g).indices
    S = np.vstack([e[ix].sum(axis=0) for ix in G.values()])
    nG = len(G)
    se = np.sqrt((S ** 2).sum(axis=0) * nG / max(nG - 1, 1)) / n
    return mu, se, nG


def cluster_diff_se(x1, g1, x0, g0):
    """两组均值差 mean(x1) − mean(x0) 的聚类稳健标准误(影响函数法)。"""
    n1, n0 = len(x1), len(x0)
    m1, m0 = x1.mean(axis=0), x0.mean(axis=0)
    infl = np.concatenate([(x1 - m1) / n1, -(x0 - m0) / n0], axis=0)
    g = np.concatenate([g1, g0])
    G = pd.Series(np.arange(len(g))).groupby(g).indices
    S = np.vstack([infl[ix].sum(axis=0) for ix in G.values()])
    nG = len(G)
    return m1 - m0, np.sqrt((S ** 2).sum(axis=0) * nG / max(nG - 1, 1)), nG


def main():
    t0 = time.time()
    p = Panel()
    dates, codes = p.dates, p.codes
    T, N = len(dates), len(codes)
    Q = EE.qualified_pool(p)
    Qn = Q.to_numpy(bool)
    twap = p.raw("twap_open10").to_numpy(np.float64)
    amt10 = p.raw("amt_open10").to_numpy(np.float64)
    dead = (p["one_word_up"] | p["one_word_down"]).to_numpy(bool)
    close = p.raw("close_px").to_numpy(np.float64)
    rc2c = p["ret_c2c"].to_numpy(np.float64)
    buyable = np.isfinite(twap) & (twap > 0) & ~dead & (amt10 > 0)
    twap_ff = pd.DataFrame(twap).ffill().to_numpy()
    print(f"[{time.time()-t0:.0f}s] 面板就绪")

    # ---- 合格池等权市场收益(按 d−1 信号行的池)----
    Qprev = np.vstack([np.zeros((1, N), bool), Qn[:-1]])
    with np.errstate(invalid="ignore"):
        m_c2c = np.nanmean(np.where(Qprev & np.isfinite(rc2c), rc2c, np.nan), axis=1)
        prevc = np.vstack([np.full((1, N), np.nan), close[:-1]])
        gap = twap / prevc - 1.0
        rest = close / twap - 1.0
        m_gap = np.nanmean(np.where(Qprev & np.isfinite(gap) & np.isfinite(rest), gap, np.nan), axis=1)
        m_rest = np.nanmean(np.where(Qprev & np.isfinite(gap) & np.isfinite(rest), rest, np.nan), axis=1)

    # =========================================================== 事件层:h5 / h10 / 价格条件 / 新鲜度
    B = pd.read_parquet(TMP / "pead_batches.parquet", columns=["arm", "rep", "T_pos", "t_pos", "slot", "j"])
    E = B[B["arm"] == "E"].copy()

    def r_h(Tp, j, h):
        Tp = np.asarray(Tp); j = np.asarray(j)
        out = np.full(len(Tp), np.nan)
        ok = (j >= 0) & (Tp + h < T)
        Te, je = Tp[ok], j[ok]
        r = twap_ff[Te + h, je] / twap[Te, je] - 1.0
        r = np.where(buyable[Te, je], r, 0.0)
        out[ok] = r
        return out

    for h in (5, 10):
        E[f"r{h}"] = r_h(E["T_pos"], E["j"], h)
    ctrl = {}
    for arm in ("B", "C"):
        c = B[B["arm"] == arm].copy()
        for h in (5, 10):
            c[f"r{h}"] = r_h(c["T_pos"], c["j"], h)
        ctrl[arm] = c.groupby(["T_pos", "slot"])[["r5", "r10"]].mean()     # 失败对照(j<0)为 NaN,mean 自动略过
    for arm in ("B", "C"):
        cm = ctrl[arm].reindex(pd.MultiIndex.from_arrays([E["T_pos"], E["slot"]]))
        for h in (5, 10):
            E[f"e{h}_{arm}"] = E[f"r{h}"].to_numpy() - cm[f"r{h}"].to_numpy()
    E["t_signal"] = dates[E["t_pos"].to_numpy()]

    # 价格条件 rel20(事件池内中位数分组)
    r0 = np.where(np.isfinite(rc2c), rc2c, 0.0)
    nvalid = pd.DataFrame(np.isfinite(rc2c).astype(float)).rolling(20, min_periods=20).sum().to_numpy()
    lr = pd.DataFrame(np.log1p(r0)).rolling(20, min_periods=20).sum().to_numpy()
    lm = pd.Series(np.log1p(np.nan_to_num(m_c2c))).rolling(20, min_periods=20).sum().to_numpy()
    rel20 = np.expm1(lr) - np.expm1(lm)[:, None]
    rel20[nvalid < 15] = np.nan
    pool = EE.event_pool(EE.main_events(EE.load_events()))
    pool["i"] = dates.get_indexer(pool["t_signal"]); pool["j"] = codes.get_indexer(pool["stock_code"])
    pool = pool[(pool["i"] >= 0) & (pool["j"] >= 0)]
    pool = pool[Qn[pool["i"].to_numpy(), pool["j"].to_numpy()]].copy()
    pool["rel20"] = rel20[pool["i"].to_numpy(), pool["j"].to_numpy()]
    med = pool.groupby("i")["rel20"].median()
    E["rel20"] = rel20[E["t_pos"].to_numpy(), E["j"].to_numpy()]
    E["rel_med"] = med.reindex(E["t_pos"]).to_numpy()
    E["grp"] = np.where(E["rel20"].isna() | np.isnan(E["rel_med"]), "na",
                        np.where(E["rel20"] <= E["rel_med"], "low", "high"))
    fr = pool.set_index(["i", "j"])["freshness"]
    E["freshness"] = fr.reindex(pd.MultiIndex.from_arrays([E["t_pos"], E["j"]])).to_numpy()
    E["fresh_grp"] = np.where(E["freshness"] == "预告在先", "预告在先",
                              np.where(E["freshness"] == "无在先记录", "无在先记录", "其他"))
    E.to_parquet(TMP / "pead_event_level_slots.parquet", index=False)

    rows = []

    def add(kind, ctrl_arm, h, s: pd.Series, L, extra=None):
        for w in WINDOWS:
            x = s[win_mask(s.index, w)]
            ci = PE.mean_ci(x.to_numpy(), L)
            rows.append(dict(kind=kind, control=ctrl_arm, h=h, window=w, n_T=ci["n"], mean=ci["est"],
                             se_nw=ci["se"], t_nw=ci["t"], lo95=ci["lo"], hi95=ci["hi"], nw_lag=L,
                             **(extra or {})))

    for arm in ("B", "C"):
        for h, L in ((5, 10), (10, 20)):
            s = E.groupby("t_signal")[f"e{h}_{arm}"].mean()
            add("paired_excess_per_T", arm, h, s, L)
            sr = E.groupby("t_signal")[f"r{h}"].mean()
        # 价格条件
        g = E[E["grp"] != "na"].groupby(["t_signal", "grp"])[f"e5_{arm}"].agg(["mean", "size"]).unstack("grp")
        okT = (g[("size", "low")] >= 3) & (g[("size", "high")] >= 3)
        D = (g[("mean", "low")] - g[("mean", "high")])[okT]
        add("price_cond_low_minus_high", arm, 5, D, 10)
        add("price_cond_low_group", arm, 5, g[("mean", "low")][okT], 10)
        add("price_cond_high_group", arm, 5, g[("mean", "high")][okT], 10)
        # 新鲜度
        for fg in ("预告在先", "无在先记录", "其他"):
            s = E[E["fresh_grp"] == fg].groupby("t_signal")[f"e5_{arm}"].mean()
            add(f"freshness:{fg}", arm, 5, s, 10, extra=dict(n_slots=int((E["fresh_grp"] == fg).sum())))
    # 预注册外诊断:COMP_v5 最差 20%(合格池内)与其余股票的 h5 理论收益差 —— 全池 vs E 入选股内部。
    # 用来核对 F 臂方向(分数越高越好)与"排雷在事件池里是否还成立",不进入判定。
    S = pd.read_parquet(TMP / "comp_v5_score_open10.parquet").reindex(index=dates, columns=codes)
    Sq = S.where(Q)
    worst = (Sq.rank(axis=1).div(Sq.notna().sum(axis=1), axis=0) <= 0.20).to_numpy(bool)
    del S, Sq
    r5_all = np.full((T, N), np.nan)                       # 行 t:t+1 窗口买入、t+6 窗口卖出(理论)
    r5_all[:T - 6] = np.where(buyable[1:T - 5], twap_ff[6:] / twap[1:T - 5] - 1.0, 0.0)
    okp = Qn & np.isfinite(r5_all)
    with np.errstate(invalid="ignore"):
        dpool = (np.nanmean(np.where(okp & worst, r5_all, np.nan), axis=1)
                 - np.nanmean(np.where(okp & ~worst, r5_all, np.nan), axis=1))
    add("comp_worst_minus_rest:qualified_pool_daily", "-", 5, pd.Series(dpool, index=dates).dropna(), 10)
    E["comp_worst"] = worst[E["t_pos"].to_numpy(), E["j"].to_numpy()]
    g = E.groupby(["t_signal", "comp_worst"])["r5"].mean().unstack()
    if True in g.columns and False in g.columns:
        add("comp_worst_minus_rest:E_picks_per_T", "-", 5, (g[True] - g[False]).dropna(), 10)
        add("comp_worst_minus_rest:E_picks_per_T_pairedB", "B", 5,
            (E[E["comp_worst"]].groupby("t_signal")["e5_B"].mean()
             - E[~E["comp_worst"]].groupby("t_signal")["e5_B"].mean()).dropna(), 10)
    # 理论 h5 收益本身(E 入选股,不扣对照)
    for h, L in ((5, 10), (10, 20)):
        add("E_raw_return_per_T", "-", h, E.groupby("t_signal")[f"r{h}"].mean(), L)
    EL = pd.DataFrame(rows)
    EL.to_csv(LEDGER / "pead_event_level.csv", index=False)
    print(EL[EL.window != "all2125"].to_string())
    print(f"[{time.time()-t0:.0f}s] 事件层完成")

    # =========================================================== 经典事件研究(7.8)
    ev = EE.main_events(EE.load_events())
    ev["j"] = codes.get_indexer(ev["stock_code"])
    ev["vpos"] = dates.get_indexer(ev["V"])
    ev = ev[(ev["j"] >= 0) & (ev["vpos"] >= 5) & (ev["V"] >= pd.Timestamp("2021-01-04"))
            & (ev["vpos"] + 60 <= T - 1)].copy()
    ev["vq"] = ev["V"].dt.to_period("Q").astype(str)
    pct = ev.groupby("vq")["value"].rank(method="average", pct=True)
    ev["quint"] = np.clip(np.ceil(pct * 5), 1, 5).astype(int)
    ev["qual"] = Qn[ev["vpos"].to_numpy() - 1, ev["j"].to_numpy()]
    AR = rc2c - m_c2c[:, None]
    vp, jj = ev["vpos"].to_numpy(), ev["j"].to_numpy()
    rows_idx = vp[:, None] + TAUS[None, :]
    A = AR[rows_idx, jj[:, None]]
    miss = ~np.isfinite(A)
    A = np.where(miss, 0.0, A)
    CAR = np.cumsum(A, axis=1)
    ar_gap = gap[vp, jj] - m_gap[vp]
    ar_rest = rest[vp, jj] - m_rest[vp]
    # 时钟诊断补充(审查 P2,预注册外、只描述):τ=−1 同样拆成"前收盘 -> 开盘 TWAP"与"开盘 TWAP -> 收盘"。
    # P 为交易日时 τ=−1 就是 P 日;若在 P 日开盘 TWAP 入场(修正时钟),能多拿到的是
    # "P 日开盘 TWAP -> P 收盘" + "P 收盘 -> V 日开盘 TWAP"(= V_gap_to_open_twap),两段之和另列一项。
    ar_gap_m1 = gap[vp - 1, jj] - m_gap[vp - 1]
    ar_rest_m1 = rest[vp - 1, jj] - m_rest[vp - 1]
    i0 = int(np.flatnonzero(TAUS == 0)[0])
    comp = {
        "pre_run_m5_m1": CAR[:, i0 - 1],
        "V_day_total": A[:, i0],
        "V_gap_to_open_twap": np.where(np.isfinite(ar_gap), ar_gap, np.nan),
        "V_open_twap_to_close": np.where(np.isfinite(ar_rest), ar_rest, np.nan),
        "drift_p1_p5": CAR[:, i0 + 5] - CAR[:, i0],
        "drift_p1_p20": CAR[:, i0 + 20] - CAR[:, i0],
        "drift_p1_p60": CAR[:, i0 + 60] - CAR[:, i0],
        "drift_p6_p60": CAR[:, i0 + 60] - CAR[:, i0 + 5],
        # 时钟诊断(预注册外,只描述):单日 AR。P 为交易日时 τ=−1 就是 P 日本身;
        # 若公告实为 P 前一晚发布(报纸日),τ=−1 已含公告反应。P 为周末/节假日时 τ=−1 是 P 之前的最后一个交易日。
        "AR_tau_m2": A[:, i0 - 2],
        "AR_tau_m1": A[:, i0 - 1],
        "AR_tau_p1": A[:, i0 + 1],
        "tau_m1_gap_to_open_twap": np.where(np.isfinite(ar_gap_m1), ar_gap_m1, np.nan),
        "tau_m1_open_twap_to_close": np.where(np.isfinite(ar_rest_m1), ar_rest_m1, np.nan),
        "tau_m1_open_twap_to_V_open_twap": np.where(np.isfinite(ar_rest_m1) & np.isfinite(ar_gap),
                                                    ar_rest_m1 + ar_gap, np.nan),
    }
    car_rows, sum_rows = [], []
    ptd = ev["P_is_trading_day"].astype("boolean").fillna(False).to_numpy(bool)
    samples = {"main": np.ones(len(ev), bool), "qualified_at_Vm1": ev["qual"].to_numpy(),
               "main_P_trading_day": ptd, "main_P_non_trading_day": ~ptd}
    vdates = pd.DatetimeIndex(ev["V"])
    wins = {"all2125": np.ones(len(ev), bool), "dev": win_mask(vdates, "dev"), "test": win_mask(vdates, "test")}
    for y in range(2021, 2026):
        wins[f"y{y}"] = vdates.year == y
    qa = ev["quint"].to_numpy(); ga = ev["vq"].to_numpy()
    for sname, sm in samples.items():
        for wname, wm in wins.items():
            base = sm & wm
            for q in range(1, 6):
                m = base & (qa == q)
                if m.sum() < 30:
                    continue
                if not wname.startswith("y") and sname in ("main", "qualified_at_Vm1"):
                    mu, se, nG = cluster_mean_se(CAR[m], ga[m])
                    for k, tau in enumerate(TAUS):
                        car_rows.append(dict(sample=sname, window=wname, quintile=q, tau=int(tau), n=int(m.sum()),
                                             n_clusters=nG, car=mu[k], lo95=mu[k] - 1.96 * se[k],
                                             hi95=mu[k] + 1.96 * se[k],
                                             ar_missing_share=float(miss[m, k].mean())))
                for cn, x in comp.items():
                    mm = m & np.isfinite(x)
                    mu, se, nG = cluster_mean_se(x[mm][:, None], ga[mm])
                    sum_rows.append(dict(sample=sname, window=wname, group=f"Q{q}", component=cn, n=int(mm.sum()),
                                         n_clusters=nG, mean=float(mu[0]), lo95=float(mu[0] - 1.96 * se[0]),
                                         hi95=float(mu[0] + 1.96 * se[0])))
            m5, m1 = base & (qa == 5), base & (qa == 1)
            if m5.sum() >= 30 and m1.sum() >= 30:
                if not wname.startswith("y") and sname in ("main", "qualified_at_Vm1"):
                    d, se, nG = cluster_diff_se(CAR[m5], ga[m5], CAR[m1], ga[m1])
                    for k, tau in enumerate(TAUS):
                        car_rows.append(dict(sample=sname, window=wname, quintile="Q5-Q1", tau=int(tau),
                                             n=int(m5.sum() + m1.sum()), n_clusters=nG, car=d[k],
                                             lo95=d[k] - 1.96 * se[k], hi95=d[k] + 1.96 * se[k]))
                for cn, x in comp.items():
                    a5, a1 = m5 & np.isfinite(x), m1 & np.isfinite(x)
                    d, se, nG = cluster_diff_se(x[a5][:, None], ga[a5], x[a1][:, None], ga[a1])
                    sum_rows.append(dict(sample=sname, window=wname, group="Q5-Q1", component=cn,
                                         n=int(a5.sum() + a1.sum()), n_clusters=nG, mean=float(d[0]),
                                         lo95=float(d[0] - 1.96 * se[0]), hi95=float(d[0] + 1.96 * se[0])))
    CARdf = pd.DataFrame(car_rows)
    SUM = pd.DataFrame(sum_rows)
    CARdf.to_csv(LEDGER / "pead_car.csv", index=False)
    SUM.to_csv(LEDGER / "pead_car_summary.csv", index=False)
    meta = dict(n_events=int(len(ev)), ar_missing_share=float(miss.mean()),
                n_quarters=int(ev["vq"].nunique()))
    print(meta)
    pd.DataFrame([meta]).to_csv(LEDGER / "pead_car_meta.csv", index=False)
    s = SUM[(SUM["sample"] == "main") & (SUM["group"] == "Q5-Q1")]
    print(s.pivot(index="window", columns="component", values="mean").round(4).to_string())
    plot(CARdf, SUM)
    print(f"完成 [{time.time()-t0:.0f}s]")


def plot(CARdf: pd.DataFrame, SUM: pd.DataFrame):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "Arial Unicode MS", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False
    INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
    # 五分位 = 有序且有正负含义 -> 发散配色:Q1/Q2 红臂、Q3 中性灰、Q4/Q5 蓝臂(参考调色板的 blue<->red)
    QC = {1: "#c23b3a", 2: "#ec8a88", 3: "#898781", 4: "#6da7ec", 5: "#1c5cab"}
    fig = plt.figure(figsize=(15, 9.5), facecolor="#fcfcfb")
    gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1], hspace=0.48, wspace=0.22)
    titles = {"all2125": "全期 2021–2025", "dev": "开发窗 2021–2023", "test": "检验窗 2024–2025"}
    d = CARdf[CARdf["sample"] == "main"]
    ylim = (d[d.quintile.isin([1, 2, 3, 4, 5])]["lo95"].min() * 100 - 0.3,
            d[d.quintile.isin([1, 2, 3, 4, 5])]["hi95"].max() * 100 + 0.3)
    for k, w in enumerate(["all2125", "dev", "test"]):
        ax = fig.add_subplot(gs[0, k], facecolor="#fcfcfb")
        ax.axvspan(-0.5, 0.5, color="#f0efec", zorder=0)
        ax.axhline(0, color=AXIS, lw=1, zorder=1)
        ends = {}
        for q in (1, 2, 3, 4, 5):
            x = d[(d.window == w) & (d.quintile == q)]
            if x.empty:
                continue
            ax.fill_between(x.tau, x.lo95 * 100, x.hi95 * 100, color=QC[q], alpha=0.13, lw=0)
            ax.plot(x.tau, x.car * 100, color=QC[q], lw=2, label=f"Q{q}")
            ends[q] = float(x.car.iloc[-1] * 100)
        # 直接标注:按终值排序后最小间距 0.28pp,避免重叠
        gap_ = 0.28
        order = sorted(ends, key=ends.get)
        ys_ = [ends[q] for q in order]
        for k_ in range(1, len(ys_)):
            ys_[k_] = max(ys_[k_], ys_[k_ - 1] + gap_)
        for q, yv in zip(order, ys_):
            ax.text(61.5, yv, f"Q{q}", va="center", fontsize=9, color=INK2)
        n = int(d[(d.window == w) & (d.quintile == 1)]["n"].iloc[0] + d[(d.window == w) & (d.quintile == 5)]["n"].iloc[0])
        ax.set_title(titles[w], fontsize=12, color=INK, loc="left")
        ax.text(0.01, 0.97, "灰带 = V 日(含公告后隔夜跳空,不可交易)", transform=ax.transAxes, fontsize=8.5,
                color=MUTED, va="top")
        ax.set_xlim(-6, 68); ax.set_ylim(*ylim)
        ax.set_xlabel("相对 V 的交易日 τ", color=INK2, fontsize=10)
        if k == 0:
            ax.set_ylabel("累计超额 CAR(%,相对合格池等权)", color=INK2, fontsize=10)
            ax.legend(loc="lower left", frameon=False, fontsize=9, ncol=5)
        ax.grid(axis="y", color=GRID, lw=0.8); ax.tick_params(colors=MUTED, labelsize=9)
        for s_ in ax.spines.values():
            s_.set_color(AXIS)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    # 下排:Q5−Q1 分解(按年)
    ax = fig.add_subplot(gs[1, :], facecolor="#fcfcfb")
    comps = [("pre_run_m5_m1", "公告前 τ=−5..−1"), ("V_gap_to_open_twap", "V 日:前收盘→开盘 TWAP(不可交易)"),
             ("V_open_twap_to_close", "V 日:开盘 TWAP→收盘"), ("drift_p1_p5", "漂移 τ=1..5"),
             ("drift_p6_p60", "漂移 τ=6..60")]
    CC = ["#898781", "#eb6834", "#eda100", "#2a78d6", "#1c5cab"]
    groups = ["dev", "test", "y2021", "y2022", "y2023", "y2024", "y2025"]
    glab = ["开发窗\n2021–23", "检验窗\n2024–25", "2021", "2022", "2023", "2024", "2025"]
    s = SUM[(SUM["sample"] == "main") & (SUM["group"] == "Q5-Q1")]
    width = 0.15
    xs = np.arange(len(groups))
    for c_i, ((cn, lab), col) in enumerate(zip(comps, CC)):
        vals, lo, hi = [], [], []
        for g in groups:
            r = s[(s.window == g) & (s.component == cn)]
            vals.append(r["mean"].iloc[0] * 100 if len(r) else np.nan)
            lo.append(r["lo95"].iloc[0] * 100 if len(r) else np.nan)
            hi.append(r["hi95"].iloc[0] * 100 if len(r) else np.nan)
        vals, lo, hi = map(np.array, (vals, lo, hi))
        xx = xs + (c_i - 2) * (width + 0.01)
        ax.bar(xx, vals, width=width, color=col, label=lab, zorder=2, edgecolor="#fcfcfb", linewidth=1)
        # 区间只画两个窗口:单个年份只有 4 个季度聚类,聚类区间不可靠(数字仍在 pead_car_summary.csv)
        ax.errorbar(xx[:2], vals[:2], yerr=[vals[:2] - lo[:2], hi[:2] - vals[:2]], fmt="none", ecolor=INK2,
                    elinewidth=0.9, capsize=2, zorder=3)
    ax.axhline(0, color=AXIS, lw=1)
    ax.axvline(1.5, color=GRID, lw=1)
    ax.set_xticks(xs); ax.set_xticklabels(glab, fontsize=10, color=INK2)
    ax.set_ylabel("Q5 − Q1(%,95% 区间按季度聚类)", color=INK2, fontsize=10)
    ax.set_title("最高 vs 最低五分位的超额分解:公告跳空 vs 之后的漂移(按 V 所在年份)", fontsize=12,
                 color=INK, loc="left", pad=28)
    ax.legend(frameon=False, fontsize=9, ncol=5, loc="lower left", bbox_to_anchor=(0.0, 1.0))
    ax.text(0.995, 0.97, "误差线 = 95% 区间(季度聚类),只画两个窗口;单年仅 4 个季度聚类,不画区间",
            transform=ax.transAxes, ha="right", va="top", fontsize=8.5, color=MUTED)
    ax.grid(axis="y", color=GRID, lw=0.8); ax.tick_params(colors=MUTED, labelsize=9)
    for s_ in ax.spines.values():
        s_.set_color(AXIS)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.suptitle("PEAD 经典事件研究:按营业利润单季同比(fu_op_yoy)季度内五分位的累计超额收益", fontsize=14,
                 color=INK, x=0.06, ha="left", y=0.985)
    fig.text(0.06, 0.012, "样本:主事件(首次披露、依赖项已公开、V = P 后首个交易日),V 在 2021-01-04 之后且 V+60 在面板内;"
             "AR = ret_c2c − 合格池等权 ret_c2c;缺行情的 AR 记 0。数据不含已退市股(幸存者偏差,事件收益偏高估)。"
             "理论收益,不含成本与成交约束。", fontsize=8.5, color=MUTED)
    for name in ("pead_car.png", "pead_event_study_quintiles.png"):
        fig.savefig(REPORTS / name, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    main()
