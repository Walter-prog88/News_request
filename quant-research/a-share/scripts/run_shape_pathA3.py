"""路径 A 第三阶段:硬分配版形状分类表(覆盖 100% 样本)。
每个 eligible 股票日按与 k=8 簇中心的相关系数取 argmax 分到一个形状簇,
报告:各簇在全样本的占比、在赢家(f60>=50%)中的占比、lift、日内中性化超额。
另加"用户口径形状"(300308 类)在两段的表现,以及簇的稳定性(2020-23 vs 2024-25 占比漂移)。
"""
from __future__ import annotations
import sys, time
import numpy as np
import pandas as pd
from scipy.signal import fftconvolve

sys.path.insert(0, "/path/to/user/Desktop/A stock market/src")
from afactor.panel import Panel
import afactor.strategy as st

W = 120
TRAIN_END = pd.Timestamp("2023-12-31")
log = lambda *a: print(*a, flush=True)


def roll_sum(X, w):
    cs = np.empty((X.shape[0] + 1, X.shape[1]), np.float64)
    cs[0] = 0.0
    np.cumsum(X, axis=0, out=cs[1:])
    return cs[w:] - cs[:-w]


def roll_ms_cnt(X, w):
    fin = np.isfinite(X)
    Xf = np.where(fin, X, 0.0)
    cnt = roll_sum(fin.astype(np.float64), w)
    m = roll_sum(Xf, w) / w
    var = roll_sum(Xf * Xf, w) / w - m * m
    return m, np.sqrt(np.maximum(var, 1e-12)), cnt


def xcorr(X32, kern):
    T, N = X32.shape
    out = np.empty((T - W + 1, N), np.float32)
    kr = np.ascontiguousarray(kern[::-1].astype(np.float32))
    for a in range(0, N, 1200):
        b = min(a + 1200, N)
        out[:, a:b] = fftconvolve(X32[:, a:b], kr[:, None], mode="valid", axes=0)
    return out


def nw_t(x, lag=60):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    n = len(x)
    if n < lag + 10:
        return np.nan
    e = x - x.mean(); s = (e * e).sum() / n
    for l in range(1, lag + 1):
        s += 2 * (1 - l / (lag + 1)) * (e[l:] * e[:-l]).sum() / n
    return float(x.mean() / np.sqrt(max(s, 1e-18) / n))


def main():
    t0 = time.time()
    p = Panel()
    close, amount, tw = p.raw("close_px"), p.raw("amount"), p.raw("twap_open10")
    dates, codes = close.index, close.columns
    T, N = close.shape
    valid = p["valid"].values
    univ = st.universe(p, "V1000").values
    twv = tw.values.astype(np.float64)
    f60 = np.full((T, N), np.nan); f60[: T - 61] = twv[61:] / twv[1: T - 60] - 1.0
    f20 = np.full((T, N), np.nan); f20[: T - 21] = twv[21:] / twv[1: T - 20] - 1.0

    C = pd.DataFrame(close.values, index=dates, columns=codes).ffill().values.astype(np.float64)
    A = pd.DataFrame(amount.values, index=dates, columns=codes).ffill().values.astype(np.float64)
    logC = np.log(np.where(C > 0, C, np.nan)); logA = np.log(np.where(A > 0, A, np.nan) + 1.0)
    del C, A
    mp, sdp, cp = roll_ms_cnt(logC, W)
    ma, sda, ca = roll_ms_cnt(logA, W)
    vc = roll_sum(valid.astype(np.float64), W)
    ok = np.zeros((T, N), bool)
    ok[W - 1:] = (cp >= W) & (ca >= W) & (vc >= 100) & (sdp > 1e-4) & (sda > 1e-4)
    elig = ok & univ & np.isfinite(f60)
    del cp, ca, vc, ok
    p._cache.clear(); p._derived.clear()
    gm_c = np.nanmean(np.where(np.isfinite(logC), logC, np.nan), 0, keepdims=True)
    gm_a = np.nanmean(np.where(np.isfinite(logA), logA, np.nan), 0, keepdims=True)
    C32 = np.nan_to_num(logC - gm_c, nan=0.0).astype(np.float32)
    A32 = np.nan_to_num(logA - gm_a, nan=0.0).astype(np.float32)
    del logC, logA

    def dot_block(Xb, kern, mroll, sdroll, gm):
        return (xcorr(Xb, kern) - (mroll - gm) * kern.sum()) / sdroll

    cen = np.load("/path/to/user/Desktop/A stock market/research/pathA/pathA_centers.npy")
    k8 = cen[13:21]
    K = 8
    CORR = np.full((K, T, N), -9.0, np.float32)
    for j in range(K):
        m = k8[j]
        CORR[j, W - 1:] = ((dot_block(C32, m[:W], mp, sdp, gm_c)
                            + dot_block(A32, m[W:], ma, sda, gm_a))
                           / (2 * W * m.std())).astype(np.float32)
    lab = CORR.argmax(0).astype(np.int8)
    best = CORR.max(0)
    del CORR
    log(f"assigned ({time.time()-t0:.0f}s)")

    seg = {"2020-23": np.asarray(dates <= TRAIN_END)}
    seg["2024-25"] = ~seg["2020-23"]
    win = f60 >= 0.50

    rows = []
    for sname, sm in seg.items():
        E = elig & sm[:, None]
        ne = E.sum()
        bw = win[E].mean()
        days = np.flatnonzero(sm & (elig.sum(1) >= 200))
        for j in range(K):
            M = E & (lab == j)
            n = int(M.sum())
            if n < 500:
                continue
            # 日内中性化
            ex60, ex20, exw, wts = [], [], [], []
            for t in days:
                hm = M[t]
                if hm.sum() < 3:
                    continue
                em = E[t]
                ex60.append(np.nanmean(f60[t, hm]) - np.nanmean(f60[t, em]))
                v1, v2 = f20[t, hm], f20[t, em]
                ex20.append((np.nanmean(v1) if np.isfinite(v1).any() else np.nan)
                            - (np.nanmean(v2) if np.isfinite(v2).any() else np.nan))
                exw.append(win[t, hm].mean() - win[t, em].mean())
                wts.append(hm.sum())
            ex60 = np.array(ex60); ex20 = np.array(ex20); exw = np.array(exw)
            wts = np.array(wts, float); ww = wts / wts.sum()
            rows.append({
                "seg": sname, "cluster": j, "n": n, "share_all": n / ne,
                "share_of_winners": float((win[M]).sum() / max(win[E].sum(), 1)),
                "P_win50": float(win[M].mean()), "base_win50": float(bw),
                "lift50_raw": float(win[M].mean() / bw),
                "exc_win50_dayneutral": float((exw * ww).sum()),
                "lift50_dayneutral": float(1 + (exw * ww).sum() / bw),
                "mean_f60": float(np.nanmean(f60[M])),
                "exc_f60": float((ex60 * ww).sum()), "t_f60_NW": nw_t(ex60, 60),
                "exc_f20": float(np.nansum(ex20 * ww)), "t_f20_NW": nw_t(ex20, 20),
                "n_days": len(ex60),
            })
    df = pd.DataFrame(rows)
    df.to_csv("/path/to/user/Desktop/A stock market/research/pathA/pathA3_hardassign.csv", index=False)
    with pd.option_context("display.width", 280, "display.max_columns", 40,
                           "display.max_rows", 60, "display.float_format", lambda x: f"{x:.4f}"):
        log("\n===== 硬分配:形状簇 -> 未来60日 =====")
        log(df.to_string(index=False))

    # 高相似度子集(corr>=0.75)重复一遍
    rows2 = []
    for sname, sm in seg.items():
        E = elig & sm[:, None]
        bw = win[E].mean()
        days = np.flatnonzero(sm & (elig.sum(1) >= 200))
        for j in range(K):
            M = E & (lab == j) & (best >= 0.75)
            n = int(M.sum())
            if n < 500:
                continue
            ex60, exw, wts = [], [], []
            for t in days:
                hm = M[t]
                if hm.sum() < 3:
                    continue
                em = E[t]
                ex60.append(np.nanmean(f60[t, hm]) - np.nanmean(f60[t, em]))
                exw.append(win[t, hm].mean() - win[t, em].mean())
                wts.append(hm.sum())
            ex60 = np.array(ex60); exw = np.array(exw); wts = np.array(wts, float)
            ww = wts / wts.sum()
            rows2.append({"seg": sname, "cluster": j, "n": n,
                          "P_win50": float(win[M].mean()), "base": float(bw),
                          "lift50_dayneutral": float(1 + (exw * ww).sum() / bw),
                          "exc_f60": float((ex60 * ww).sum()), "t_f60_NW": nw_t(ex60, 60)})
    df2 = pd.DataFrame(rows2)
    with pd.option_context("display.width", 280, "display.max_columns", 30,
                           "display.float_format", lambda x: f"{x:.4f}"):
        log("\n===== 高相似度子集 corr>=0.75 =====")
        log(df2.to_string(index=False))
    df2.to_csv("/path/to/user/Desktop/A stock market/research/pathA/pathA3_hi.csv", index=False)

    # 300308 硬分配轨迹
    if (codes == "300308.SZ").any():
        i = int(np.flatnonzero(codes == "300308.SZ")[0])
        s = pd.DataFrame({"lab": lab[:, i], "corr": best[:, i], "f60": f60[:, i],
                          "elig": elig[:, i]}, index=dates).loc["2024-06":]
        log("\n300308 2024-06 起 每月末所属簇:")
        log(s.resample("ME").last().round(3).to_string())
    log(f"DONE {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
