"""路径 A:从赢家反推形状(120 日启动前形态 + 量能形态 聚类 -> 全样本模板匹配)。

信息时点:特征只用 T 日及之前收盘数据。
前瞻:tw=twap_open10, fwd_h = tw.shift(-(1+h))/tw.shift(-1)-1。
样本外:2020-2023 聚类/搜索,2024-2025 检验。
"""
from __future__ import annotations
import sys, json, time
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import fftconvolve
from sklearn.cluster import KMeans

sys.path.insert(0, "/path/to/user/Desktop/A stock market/src")
from afactor.panel import Panel
import afactor.strategy as st

W = 120                      # 启动前窗口
HZ = 60                      # 赢家定义窗口
THRESHOLDS = [0.30, 0.50, 1.00]
OUT = "/path/to/user/Desktop/A stock market/research/pathA/pathA"
TRAIN_END = "2023-12-31"
SAMPLE_STEP = 3              # 聚类样本按日期抽样

log = lambda *a: (print(*a, flush=True))


# ---------------- 工具 ----------------
def roll_sum(X, w):
    """X (T,N) float64 NaN->0 已处理。返回 (T-w+1, N) 的窗口和,行 k 对应 t=k+w-1。"""
    cs = np.empty((X.shape[0] + 1, X.shape[1]), np.float64)
    cs[0] = 0.0
    np.cumsum(X, axis=0, out=cs[1:])
    return cs[w:] - cs[:-w]


def roll_mean_std_cnt(X, w):
    fin = np.isfinite(X)
    Xf = np.where(fin, X, 0.0)
    cnt = roll_sum(fin.astype(np.float64), w)
    s = roll_sum(Xf, w)
    s2 = roll_sum(Xf * Xf, w)
    m = s / w
    var = s2 / w - m * m
    sd = np.sqrt(np.maximum(var, 1e-12))
    return m, sd, cnt


def xcorr_dot(X32, kern):
    """sum_i X[t-119+i]*kern[i],返回 (T-w+1,N) float32。分块做 FFT 卷积控内存。"""
    T, N = X32.shape
    out = np.empty((T - W + 1, N), np.float32)
    kr = kern[::-1].astype(np.float32).copy()
    step = 1200
    for a in range(0, N, step):
        b = min(a + step, N)
        out[:, a:b] = fftconvolve(X32[:, a:b], kr[:, None], mode="valid", axes=0)
    return out


def main():
    t0 = time.time()
    p = Panel()
    close = p.raw("close_px")
    amount = p.raw("amount")
    tw = p.raw("twap_open10")
    dates = close.index
    codes = close.columns
    T, N = close.shape
    log(f"panel {T}x{N} {dates[0].date()}..{dates[-1].date()}")

    valid = p["valid"].values                       # 当日真实成交
    univ = st.universe(p, "V1000").values           # 决策池

    # --- 前瞻收益 ---
    twv = tw.values.astype(np.float64)
    def fwd(h):
        f = np.full((T, N), np.nan, np.float64)
        f[: T - (1 + h)] = twv[1 + h:] / twv[1: T - h] - 1.0
        return f
    f60 = fwd(HZ)
    f20 = fwd(20)

    # --- 特征路径(只用 T 及之前收盘)---
    C = pd.DataFrame(close.values, index=dates, columns=codes).ffill().values.astype(np.float64)
    A = pd.DataFrame(amount.values, index=dates, columns=codes).ffill().values.astype(np.float64)
    logC = np.log(np.where(C > 0, C, np.nan))
    logA = np.log(np.where(A > 0, A, np.nan) + 1.0)
    del C, A

    mp, sdp, cntp = roll_mean_std_cnt(logC, W)
    ma, sda, cnta = roll_mean_std_cnt(logA, W)
    validcnt = roll_sum(valid.astype(np.float64), W)

    # 窗口完整 & 期间大部分交易日有成交
    winok = np.zeros((T, N), bool)
    winok[W - 1:] = (cntp >= W) & (cnta >= W) & (validcnt >= 100) & (sdp > 1e-4) & (sda > 1e-4)

    elig = winok & univ & np.isfinite(f60)          # 可评估样本
    log(f"eligible stock-days: {elig.sum():,}")

    seg_tr = np.asarray(dates <= pd.Timestamp(TRAIN_END))
    seg_te = ~seg_tr
    SEG = {"2020-23": seg_tr, "2024-25": seg_te}

    # --- 基准发生率 ---
    base = {}
    for name, sm in SEG.items():
        m = elig & sm[:, None]
        row = {"n": int(m.sum())}
        for th in THRESHOLDS:
            row[f"base_p{int(th*100)}"] = float((f60[m] >= th).mean())
        row["base_mean_f60"] = float(np.nanmean(f60[m]))
        row["base_mean_f20"] = float(np.nanmean(f20[m][np.isfinite(f20[m])]))
        base[name] = row
    log("BASE RATES: " + json.dumps(base, indent=1))

    # 全样本(用于聚类的赢家来自训练段)
    win50 = elig & (f60 >= 0.50)
    log(f"winners(>=50%) train={int((win50 & seg_tr[:,None]).sum()):,} "
        f"test={int((win50 & seg_te[:,None]).sum()):,}")

    # --- 抽取赢家启动前形状 ---
    demC = logC - np.nanmean(logC, axis=0, keepdims=True)
    demA = logA - np.nanmean(logA, axis=0, keepdims=True)
    C32 = np.nan_to_num(demC, nan=0.0).astype(np.float32)
    A32 = np.nan_to_num(demA, nan=0.0).astype(np.float32)
    del demC, demA
    p._cache.clear(); p._derived.clear()

    samp_t = [t for t in range(W - 1, T) if (t % SAMPLE_STEP == 0) and seg_tr[t]]
    vecs = []
    meta = []
    for t in samp_t:
        cols = np.flatnonzero(win50[t])
        if cols.size == 0:
            continue
        blkC = logC[t - W + 1: t + 1, cols].T            # (n,120)
        blkA = logA[t - W + 1: t + 1, cols].T
        zc = (blkC - blkC.mean(1, keepdims=True)) / blkC.std(1, keepdims=True)
        za = (blkA - blkA.mean(1, keepdims=True)) / blkA.std(1, keepdims=True)
        vecs.append(np.concatenate([zc, za], 1).astype(np.float32))
        meta.append(np.stack([np.full(cols.size, t), cols], 1))
    Z = np.concatenate(vecs, 0)
    MET = np.concatenate(meta, 0)
    del vecs, meta
    log(f"winner shape matrix {Z.shape}  ({time.time()-t0:.0f}s)")

    results = {"base": base, "n_winner_samples": int(Z.shape[0])}

    # --- 聚类 k=6..10 ---
    knots = [0, 20, 40, 60, 80, 100, 110, 119]
    cluster_desc = {}
    centers_all = {}
    for k in range(6, 11):
        km = KMeans(n_clusters=k, n_init=4, random_state=0, max_iter=200).fit(Z)
        lab = km.labels_
        cen = km.cluster_centers_.astype(np.float64)
        centers_all[k] = cen
        desc = []
        for j in range(k):
            cp = cen[j, :W]; ca = cen[j, W:]
            imax = int(np.argmax(cp))
            desc.append({
                "cluster": j,
                "n": int((lab == j).sum()),
                "share": float((lab == j).mean()),
                "px_knots": [round(float(cp[i]), 2) for i in knots],
                "amt_knots": [round(float(ca[i]), 2) for i in knots],
                "px_argmax": imax,
                "px_end_minus_max": round(float(cp[-1] - cp.max()), 2),
                "px_end_minus_start": round(float(cp[-1] - cp[0]), 2),
                "px_last25_range": round(float(cp[-25:].max() - cp[-25:].min()), 2),
                "px_last20_slope": round(float(np.polyfit(np.arange(20), cp[-20:], 1)[0]), 4),
                "px_first60_slope": round(float(np.polyfit(np.arange(60), cp[:60], 1)[0]), 4),
                "amt_last20_minus_first60": round(float(ca[-20:].mean() - ca[:60].mean()), 2),
            })
        cluster_desc[k] = desc
        log(f"--- k={k} ---")
        for d in desc:
            log(f"  c{d['cluster']} n={d['n']:6d} ({d['share']:.1%}) px={d['px_knots']} "
                f"end-max={d['px_end_minus_max']} last25rng={d['px_last25_range']} "
                f"slope20={d['px_last20_slope']} amtΔ={d['amt_last20_minus_first60']}")
    results["clusters"] = cluster_desc

    np.save(f"{OUT}_centers.npy", np.concatenate([centers_all[k] for k in range(6, 11)], 0))
    with open(f"{OUT}_desc.json", "w") as fh:
        json.dump(results, fh, indent=1)

    # ============ 全样本模板匹配 ============
    IDX308 = int(np.flatnonzero(codes == "300308.SZ")[0]) if (codes == "300308.SZ").any() else -1
    K_MAIN = 8
    cen = centers_all[K_MAIN]
    sdm = cen.std(1)                       # 240 维整体 std(每块 mean 已为 0)

    match_rows = []
    corr308 = {}
    percentile_hits = {}
    for j in range(K_MAIN):
        mpk = cen[j, :W]; mak = cen[j, W:]
        dp = xcorr_dot(C32, mpk)
        da = xcorr_dot(A32, mak)
        corr = np.zeros((T, N), np.float32)
        corr[W - 1:] = ((dp / sdp) + (da / sda)).astype(np.float32) / (2 * W * sdm[j])
        del dp, da
        if IDX308 >= 0:
            corr308[j] = corr[:, IDX308].copy()

        for thr in [0.5, 0.6, 0.7, 0.8]:
            Hm = elig & (corr >= thr)
            for sname, sm in SEG.items():
                match_rows.append(eval_hits(Hm & sm[:, None], elig & sm[:, None],
                                            f60, f20, dates, f"k8_c{j}", f"corr>={thr}", sname))
        # 每日 top 0.5% 相似度
        Hq = daily_top(corr, elig, 0.005)
        for sname, sm in SEG.items():
            match_rows.append(eval_hits(Hq & sm[:, None], elig & sm[:, None],
                                        f60, f20, dates, f"k8_c{j}", "daily_top0.5%", sname))
        del corr
        log(f"template c{j} done ({time.time()-t0:.0f}s)")

    # --- 基线对照:纯动量 / 放量,同样 daily top ---
    mom120 = np.full((T, N), np.nan, np.float64)
    mom120[W - 1:] = logC[W - 1:] - logC[: T - W + 1]
    mom20 = np.full((T, N), np.nan, np.float64)
    mom20[20:] = logC[20:] - logC[:-20]
    amtchg = np.full((T, N), np.nan, np.float64)
    amtchg[W - 1:] = (logA[W - 1:] - ma) / sda
    for bn, sig in [("mom120", mom120), ("mom20", mom20), ("amt_z120", amtchg),
                    ("mom120_neg", -mom120)]:
        Hq = daily_top(np.nan_to_num(sig, nan=-1e18).astype(np.float32), elig, 0.005)
        for sname, sm in SEG.items():
            match_rows.append(eval_hits(Hq & sm[:, None], elig & sm[:, None],
                                        f60, f20, dates, f"BASE_{bn}", "daily_top0.5%", sname))

    df = pd.DataFrame(match_rows)
    df.to_csv(f"{OUT}_match.csv", index=False)
    log("\n===== MATCH RESULTS =====")
    with pd.option_context("display.width", 250, "display.max_columns", 40,
                           "display.max_rows", 400, "display.float_format", lambda x: f"{x:.4f}"):
        log(df.to_string(index=False))

    # --- 300308 ---
    if IDX308 >= 0:
        cc = np.stack([corr308[j] for j in range(K_MAIN)], 1)
        d308 = pd.DataFrame(cc, index=dates, columns=[f"c{j}" for j in range(K_MAIN)])
        d308["f60"] = f60[:, IDX308]
        d308["elig"] = elig[:, IDX308]
        d308["best"] = np.where(np.isfinite(cc).all(1), cc.argmax(1), -1)
        d308["bestcorr"] = cc.max(1)
        sub = d308.loc["2024-09-01":]
        sub.to_csv(f"{OUT}_300308.csv")
        wdays = sub[(sub.elig) & (sub.f60 >= 0.50)]
        log(f"\n300308.SZ 2024-09 起 启动点(f60>=50%) 共 {len(wdays)} 天")
        if len(wdays):
            log(wdays["best"].value_counts().to_string())
            log(wdays[[f"c{j}" for j in range(K_MAIN)] + ["f60", "best", "bestcorr"]]
                .iloc[::5].round(3).to_string())
    log(f"DONE {time.time()-t0:.0f}s")


def daily_top(sig, elig, q):
    """每日在 elig 内取相似度最高的 q 比例。"""
    T, N = sig.shape
    H = np.zeros((T, N), bool)
    for t in range(T):
        m = elig[t]
        n = int(m.sum())
        if n < 200:
            continue
        k = max(1, int(round(n * q)))
        idx = np.flatnonzero(m)
        v = sig[t, idx]
        sel = idx[np.argpartition(-v, k - 1)[:k]]
        H[t, sel] = True
    return H


def eval_hits(H, E, f60, f20, dates, tmpl, rule, seg):
    n = int(H.sum())
    ne = int(E.sum())
    r = {"template": tmpl, "rule": rule, "seg": seg, "n_hits": n,
         "hit_rate": n / max(ne, 1)}
    if n < 30:
        r.update({"prec50": np.nan, "base50_daymatched": np.nan, "lift50": np.nan,
                  "exc_hit50": np.nan, "mean_f60": np.nan, "exc_f60": np.nan,
                  "exc_f20": np.nan, "t_f60": np.nan, "n_days": 0})
        return r
    w50 = (f60 >= 0.50)
    # 逐日匹配基准
    hits_d = H.sum(1)
    days = np.flatnonzero(hits_d > 0)
    ph, bh, pf, bf, p20, b20, wt = [], [], [], [], [], [], []
    for t in days:
        hm = H[t]; em = E[t]
        if em.sum() < 200:
            continue
        ph.append(w50[t, hm].mean()); bh.append(w50[t, em].mean())
        pf.append(np.nanmean(f60[t, hm])); bf.append(np.nanmean(f60[t, em]))
        v20h = f20[t, hm]; v20e = f20[t, em]
        p20.append(np.nanmean(v20h) if np.isfinite(v20h).any() else np.nan)
        b20.append(np.nanmean(v20e) if np.isfinite(v20e).any() else np.nan)
        wt.append(hm.sum())
    ph, bh, pf, bf = map(np.array, (ph, bh, pf, bf))
    p20, b20, wt = np.array(p20), np.array(b20), np.array(wt, float)
    ww = wt / wt.sum()
    prec = float((ph * ww).sum()); bmd = float((bh * ww).sum())
    ef60 = pf - bf
    m20 = np.isfinite(p20) & np.isfinite(b20)
    r.update({
        "prec50": prec,
        "base50_daymatched": bmd,
        "lift50": prec / bmd if bmd > 0 else np.nan,
        "exc_hit50": prec - bmd,
        "mean_f60": float((pf * ww).sum()),
        "exc_f60": float((ef60 * ww).sum()),
        "exc_f20": float(((p20 - b20)[m20] * (ww[m20] / ww[m20].sum())).sum()),
        "t_f60": float(ef60.mean() / (ef60.std(ddof=1) / np.sqrt(len(ef60)))) if len(ef60) > 5 else np.nan,
        "n_days": int(len(ef60)),
    })
    return r


if __name__ == "__main__":
    main()
