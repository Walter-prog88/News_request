"""INC-T3 S2:限售解禁事件研究(机制诊断,不改主判定)。

预注册 research/INC_T3_PREREG.md §7.1、§7.2、§8.1。
  AR_d = ret_c2c(i, d) − mean_k ret_c2c(ctrl_k, d)(B1 / B2 匹配对照,k=3;另报池等权对照 POOL = d−1 的
         V1000&非ST 池等权 ret_c2c),缺行情记 0 并计数;CAR(τ) = Σ_{s=−20..τ} AR_s,τ ∈ [−20, 60]。
  子窗口 [−20,−4] [−3,−1] [0,0] [1,5] [1,20] [0,20] [21,60],另加 H1 的 [−20,−1] 与 §8.1 的 [−3,+20]。
  只用 D + 窗口终点 <= 面板最后一日的事件(短窗口可用到 2025-11,长窗口以最后可用事件为界)。
  标准误:按事件日 D 聚类(CR1);稳健性按日历月聚类。分组 n < 100 只描述(judgeable=False)。
  可交易回避收益 R(V, h) = twap_open10[D+h] / twap_open10[V] − 1(事件股 − B1 对照均值),V 日不可买记 0
  并计数,D+h 停牌用 [V, D+h] 内最近有价日价格;V_B = D−2(全部层)、V_A = D−10(A 层),h ∈ {0,5,10,20}。

用法  python3 scripts/inc_t3_event_study.py
产出  research/ledger/inc_t3_car.csv(CAR 路径)、inc_t3_car_summary.csv(子窗口)、inc_t3_tradable.csv、
     inc_t3_h3_monotonic.csv、inc_t3_event_level.csv.gz(逐事件子窗口 CAR,供复核);
     reports/inc_t3_car.png(B1)、reports/inc_t3_car_b2.png(B2)
"""
from __future__ import annotations
import os, sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import inc_t3_common as C                                   # noqa: E402
from afactor.panel import Panel                             # noqa: E402
from afactor import strategy as ST                          # noqa: E402

TAUS = np.arange(-20, 61)
SUBWIN = {"[-20,-4]": (-20, -4), "[-3,-1]": (-3, -1), "[0,0]": (0, 0), "[1,5]": (1, 5), "[1,20]": (1, 20),
          "[0,20]": (0, 20), "[21,60]": (21, 60), "[-20,-1]": (-20, -1), "[-3,20]": (-3, 20), "[-20,60]": (-20, 60)}
EVWIN = {"2021-2025": ("2021-01-01", "2025-12-31"), "2021-2023": ("2021-01-01", "2023-12-31"),
         "2024-2025": ("2024-01-01", "2025-12-31"), "2020": ("2020-01-01", "2020-12-31")}
CONTROLS = ["B1", "B2", "POOL"]
MIN_N = 100
HS = [0, 5, 10, 20]


def groups(ev: pd.DataFrame) -> dict:
    """样本组(全部为在池事件)。ge5 是主样本;规模桶按 ratio_plan;类型 / 层在 >=10% 与 >=5% 内各报。"""
    rp, ra = ev["ratio_plan"], ev["ratio_actual"]
    g = {"ge5": rp >= 0.05, "ge10": rp >= 0.10, "ge20": rp >= 0.20, "all": rp.notna(),
         "ge10_actual": ra >= 0.10, "ge10_nooverlap": (rp >= 0.10) & ~ev["overlap"].astype(bool),
         "lt5": rp < 0.05}
    for b in C.SIZE_LABELS:
        g[f"size{b}"] = ev["size_bucket"] == b
    for b in C.SIZE_LABELS:
        g[f"size_actual{b}"] = ev["size_bucket_actual"] == b
    for tg in ("首发", "定增", "追加承诺", "股权激励", "其他"):
        g[f"ge10_tg_{tg}"] = (rp >= 0.10) & (ev["tg"] == tg)
        g[f"ge5_tg_{tg}"] = (rp >= 0.05) & (ev["tg"] == tg)
        g[f"all_tg_{tg}"] = rp.notna() & (ev["tg"] == tg)
    g["ge10_首发原股东"] = (rp >= 0.10) & ev["first_type"].str.startswith("首发原股东")
    g["ge10_首发机构配售"] = (rp >= 0.10) & ev["first_type"].str.startswith("首发机构配售")
    for t in ("A", "B"):
        g[f"ge10_tier{t}"] = (rp >= 0.10) & (ev["tier"] == t)
        g[f"ge5_tier{t}"] = (rp >= 0.05) & (ev["tier"] == t)
    for a, lab in enumerate(("<250", "250-750", ">750")):
        g[f"ge10_age{lab}"] = (rp >= 0.10) & (ev["age_bucket"] == a)
    return {k: v.fillna(False).to_numpy(bool) for k, v in g.items()}


def ci_rows(x: np.ndarray, gD: np.ndarray, gM: np.ndarray) -> dict:
    mu, se, nG = C.cluster_mean_se(x, gD)
    _, seM, nM = C.cluster_mean_se(x, gM)
    return dict(mean=mu, se_D=se, lo=mu - 1.96 * se, hi=mu + 1.96 * se, t_D=mu / np.where(se > 0, se, np.nan),
                nG_D=nG, se_month=seM, t_month=mu / np.where(seM > 0, seM, np.nan), nG_month=nM)


def main():
    t0 = time.time()
    p = Panel()
    dates, codes = p.dates, p.codes
    T, N = len(dates), len(codes)
    ev = C.load_events()
    ev = ev[ev["in_pool"] & (ev["d_pos"] >= 21)].copy().sort_values("event_idx").reset_index(drop=True)
    M = pd.read_parquet(C.TMP / "inc_t3_matches.parquet")
    assert (M["j_ctrl"] >= 0).all(), "匹配有失败行"
    rc = p["ret_c2c"].to_numpy(np.float64)
    twap = p.raw("twap_open10").to_numpy(np.float64)
    amt10 = p.raw("amt_open10").to_numpy(np.float64)
    dead = (p["one_word_up"] | p["one_word_down"]).to_numpy(bool)
    buyable = np.isfinite(twap) & (twap > 0) & ~dead & (amt10 > 0)
    twap_ff = pd.DataFrame(twap).ffill().to_numpy()
    U = ST.universe(p, C.VERSION).to_numpy(bool)
    st = p["is_st"].reindex(index=dates, columns=codes).fillna(False).to_numpy(bool)
    Q = U & ~st
    Qprev = np.vstack([np.zeros((1, N), bool), Q[:-1]])
    with np.errstate(invalid="ignore"):
        m_pool = np.nanmean(np.where(Qprev & np.isfinite(rc), rc, np.nan), axis=1)
    print(f"[{time.time()-t0:.0f}s] 面板就绪;在池事件 {len(ev)}", flush=True)

    nE = len(ev)
    d = ev["d_pos"].to_numpy(np.int64)
    j = ev["j"].to_numpy(np.int64)
    tt = d[:, None] + TAUS[None, :]                       # (nE, 81)
    okT = tt <= T - 1
    ttc = np.clip(tt, 0, T - 1)
    r_ev = rc[ttc, j[:, None]]
    miss_ev = ~np.isfinite(r_ev) & okT
    r_ev = np.where(np.isfinite(r_ev), r_ev, 0.0)

    def ctrl_mean(arm):
        mm = M[M["arm"] == arm].pivot_table(index="event_idx", columns="rep", values="j_ctrl").reindex(ev["event_idx"])
        J = mm.to_numpy(np.int64)                        # (nE, 3)
        R = rc[ttc[:, :, None], J[:, None, :]]           # (nE, 81, 3)
        miss = (~np.isfinite(R) & okT[:, :, None]).sum()
        R = np.where(np.isfinite(R), R, 0.0)
        return R.mean(axis=2), J, int(miss)

    AR, CJ, miss_c = {}, {}, {}
    for arm in ("B1", "B2"):
        cm, CJ[arm], miss_c[arm] = ctrl_mean(arm)
        AR[arm] = np.where(okT, r_ev - cm, np.nan)
    mp = m_pool[ttc]
    mp = np.where(np.isfinite(mp), mp, 0.0)
    AR["POOL"] = np.where(okT, r_ev - mp, np.nan)
    print(f"[{time.time()-t0:.0f}s] AR 就绪;缺行情记 0:事件股 {int(miss_ev.sum())} / {int(okT.sum())} 个事件日,"
          f"对照 B1 {miss_c['B1']}、B2 {miss_c['B2']}", flush=True)

    # ---- 逐事件子窗口 CAR ----
    tau0 = -TAUS[0]
    evl = ev[["event_idx", "code", "D", "d_pos", "tier", "tg", "first_type", "ratio_plan", "ratio_actual",
              "size_bucket", "age_bucket", "overlap"]].copy()
    for arm in CONTROLS:
        for wn, (a, b) in SUBWIN.items():
            seg = AR[arm][:, a + tau0:b + tau0 + 1]
            evl[f"car_{arm}_{wn}"] = np.where(np.isfinite(seg).all(axis=1), np.nansum(seg, axis=1), np.nan)

    # ---- 可交易回避收益 ----
    def R_vh(V, h, J):
        """J: (nE,) 或 (nE, 3) 股票列号;返回 (nE,) 或 (nE, 3)。"""
        e = d + h
        ok = (V >= 0) & (e <= T - 1)
        Vc, ec = np.clip(V, 0, T - 1), np.clip(e, 0, T - 1)
        if J.ndim == 1:
            r = twap_ff[ec, J] / twap[Vc, J] - 1.0
            r = np.where(buyable[Vc, J] & np.isfinite(r), r, 0.0)
            nb = int((~buyable[Vc, J] & ok).sum())
            return np.where(ok, r, np.nan), nb
        r = twap_ff[ec[:, None], J] / twap[Vc[:, None], J] - 1.0
        bb = buyable[Vc[:, None], J]
        r = np.where(bb & np.isfinite(r), r, 0.0)
        nb = int((~bb & ok[:, None]).sum())
        return np.where(ok[:, None], r, np.nan), nb

    nb_rows = []
    for vname, off in (("V_B=D-2", 2), ("V_A=D-10", 10)):
        V = d - off
        for h in HS:
            re_, nb_e = R_vh(V, h, j)
            for arm in ("B1", "B2"):
                rcx, nb_c = R_vh(V, h, CJ[arm])
                evl[f"R_{vname}_h{h}_{arm}"] = re_ - rcx.mean(axis=1)
                if arm == "B1":
                    nb_rows.append(dict(V=vname, h=h, n_unbuyable_event=nb_e, n_unbuyable_ctrl=nb_c))
            evl[f"Rraw_{vname}_h{h}"] = re_
    evl.to_csv(C.LEDGER / "inc_t3_event_level.csv.gz", index=False, compression="gzip")

    G = groups(ev)
    Dts = ev["D"]
    month = Dts.dt.to_period("M").astype(str).to_numpy()
    gD = d

    # ---- CAR 路径 ----
    path_rows, sum_rows = [], []
    for gname, gm in G.items():
        for wn, (a, b) in EVWIN.items():
            wm = gm & (Dts >= a).to_numpy() & (Dts <= b).to_numpy()
            if wm.sum() < 5:
                continue
            for arm in CONTROLS:
                A = AR[arm][wm]
                if gname in ("ge5", "ge10", "ge20", "all") or gname.startswith("size") and not gname.startswith("size_actual"):
                    ok_ev = np.isfinite(A).all(axis=1)            # 路径只用 [−20, 60] 完整的事件
                    if ok_ev.sum() >= 5:
                        car = np.cumsum(A[ok_ev], axis=1)
                        st_ = ci_rows(car, gD[wm][ok_ev], month[wm][ok_ev])
                        for k, tau in enumerate(TAUS):
                            path_rows.append(dict(group=gname, window=wn, control=arm, tau=int(tau), n=int(ok_ev.sum()),
                                                  mean=st_["mean"][k], lo=st_["lo"][k], hi=st_["hi"][k],
                                                  se_D=st_["se_D"][k], nG_D=st_["nG_D"]))
                for sw in SUBWIN:
                    x = evl.loc[wm, f"car_{arm}_{sw}"].to_numpy()
                    ok = np.isfinite(x)
                    if ok.sum() < 5:
                        continue
                    st_ = ci_rows(x[ok], gD[wm][ok], month[wm][ok])
                    sum_rows.append(dict(group=gname, window=wn, control=arm, subwin=sw, n=int(ok.sum()),
                                         judgeable=bool(ok.sum() >= MIN_N),
                                         **{k: (float(v[0]) if isinstance(v, np.ndarray) else v) for k, v in st_.items()}))
    pd.DataFrame(path_rows).to_csv(C.LEDGER / "inc_t3_car.csv", index=False)
    S = pd.DataFrame(sum_rows)
    S.to_csv(C.LEDGER / "inc_t3_car_summary.csv", index=False)
    print(f"[{time.time()-t0:.0f}s] CAR 汇总 {len(S)} 行", flush=True)

    # ---- 可交易 ----
    tr_rows = []
    for gname in ("ge10", "ge5", "ge20", "ge10_tierA", "ge10_首发原股东", "ge10_tg_定增", "ge10_nooverlap",
                  "size[10,20)", "size[20,50)", "size>=50"):
        gm = G[gname]
        for wn, (a, b) in EVWIN.items():
            wm = gm & (Dts >= a).to_numpy() & (Dts <= b).to_numpy()
            for vname in ("V_B=D-2", "V_A=D-10"):
                if vname == "V_A=D-10":
                    wm2 = wm & (ev["tier"] == "A").to_numpy()
                else:
                    wm2 = wm
                for h in HS:
                    for arm in ("B1", "B2"):
                        x = evl.loc[wm2, f"R_{vname}_h{h}_{arm}"].to_numpy()
                        ok = np.isfinite(x)
                        if ok.sum() < 5:
                            continue
                        st_ = ci_rows(x[ok], gD[wm2][ok], month[wm2][ok])
                        raw = evl.loc[wm2, f"Rraw_{vname}_h{h}"].to_numpy()[ok]
                        tr_rows.append(dict(group=gname, window=wn, V=vname, h=h, control=arm, n=int(ok.sum()),
                                            judgeable=bool(ok.sum() >= MIN_N), raw_event_mean=float(np.mean(raw)),
                                            **{k: (float(v[0]) if isinstance(v, np.ndarray) else v) for k, v in st_.items()}))
    TR = pd.DataFrame(tr_rows)
    TR.to_csv(C.LEDGER / "inc_t3_tradable.csv", index=False)
    pd.DataFrame(nb_rows).to_csv(C.LEDGER / "inc_t3_tradable_unbuyable.csv", index=False)

    # ---- H3 单调性:规模桶序号 vs 事件 CAR[0,20](B1)Spearman,按 D 聚类 bootstrap 999 次 ----
    from scipy.stats import spearmanr
    h3 = []
    bidx = ev["size_bucket"].map({b: i for i, b in enumerate(C.SIZE_LABELS)})
    for wn, (a, b) in EVWIN.items():
        for arm in ("B1", "B2"):
            for sw in ("[0,20]", "[-20,-1]", "[-3,20]"):
                wm = G["ge5"] & (Dts >= a).to_numpy() & (Dts <= b).to_numpy()
                x = evl.loc[wm, f"car_{arm}_{sw}"].to_numpy()
                bi = bidx[wm].to_numpy(np.float64)
                ok = np.isfinite(x) & np.isfinite(bi)
                x, bi, gg = x[ok], bi[ok], gD[wm][ok]
                if len(x) < 20:
                    continue
                rho = spearmanr(bi, x).correlation
                rng = np.random.default_rng([C.SEED, 33, len(h3)])
                uniq = np.unique(gg)
                idx = pd.Series(np.arange(len(gg))).groupby(gg).indices
                boots = []
                for _ in range(999):
                    pick = rng.choice(uniq, size=len(uniq), replace=True)
                    ii = np.concatenate([idx[u] for u in pick])
                    boots.append(spearmanr(bi[ii], x[ii]).correlation)
                means = [float(np.mean(x[bi == k])) if (bi == k).any() else np.nan for k in range(4)]
                h3.append(dict(window=wn, control=arm, subwin=sw, n=len(x), spearman=rho,
                               lo=float(np.nanpercentile(boots, 2.5)), hi=float(np.nanpercentile(boots, 97.5)),
                               **{f"mean_{C.SIZE_LABELS[k]}": means[k] for k in range(4)},
                               **{f"n_{C.SIZE_LABELS[k]}": int((bi == k).sum()) for k in range(4)}))
    pd.DataFrame(h3).to_csv(C.LEDGER / "inc_t3_h3_monotonic.csv", index=False)
    print(f"[{time.time()-t0:.0f}s] 可交易 / H3 完成", flush=True)

    # ---- 图 ----
    plot_car(pd.DataFrame(path_rows), "B1", C.REPORTS / "inc_t3_car.png")
    plot_car(pd.DataFrame(path_rows), "B2", C.REPORTS / "inc_t3_car_b2.png")
    print(f"done event study {time.time()-t0:.0f}s")


def plot_car(P: pd.DataFrame, arm: str, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    cols = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]          # dataviz 参考调色板分类 1-4
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for ax, wn in zip(axes, ("2021-2023", "2024-2025")):
        for c, b in zip(cols, C.SIZE_LABELS):
            q = P[(P["group"] == f"size{b}") & (P["window"] == wn) & (P["control"] == arm)].sort_values("tau")
            if q.empty:
                continue
            ax.plot(q["tau"], q["mean"] * 100, color=c, lw=1.8, label=f"{b}%  n={int(q['n'].iloc[0])}")
            ax.fill_between(q["tau"], q["lo"] * 100, q["hi"] * 100, color=c, alpha=0.12, lw=0)
        ax.axhline(0, color="#888", lw=0.8)
        ax.axvline(0, color="#888", lw=0.8, ls="--")
        ax.axvline(-3, color="#bbb", lw=0.8, ls=":")
        ax.set_title(f"{'开发窗' if wn == '2021-2023' else '检验窗'} {wn}(解禁日 D)", fontsize=11)
        ax.set_xlabel("相对解禁日的交易日 τ")
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel(f"CAR(%),对照 {arm},按 D 聚类 95% 带")
    axes[0].legend(title="解禁/流通股本", fontsize=9, frameon=False)
    fig.suptitle(f"限售解禁事件 CAR[−20, τ](在池事件,{arm} 匹配对照;只含 [−20,60] 完整事件)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "plot":
        P = pd.read_csv(C.LEDGER / "inc_t3_car.csv")
        plot_car(P, "B1", C.REPORTS / "inc_t3_car.png")
        plot_car(P, "B2", C.REPORTS / "inc_t3_car_b2.png")
    else:
        main()
