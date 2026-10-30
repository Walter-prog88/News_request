"""路径 A 第二阶段:
(1) 有监督形状搜索 —— 在 240 维形状空间做 PCA,再用日内百分位排名为目标拟合
    线性(=最优匹配滤波器)与 GBM,2020-23 训练 / 2024-25 检验。
    这是"形状空间里到底有没有可事前识别的结构"的上界估计,比 KMeans 模板强。
(2) 收益分布不对称性 —— 命中组的 P(f60>=50%) 抬升(lift)是否只是"更像彩票"。
    做路径依赖止损(-8%/-12%/-20%),看带止损后是否翻正。
(3) Newey-West(60 滞后)修正重叠窗口带来的 t 值虚高。
"""
from __future__ import annotations
import sys, json, time
import numpy as np
import pandas as pd
from scipy.signal import fftconvolve
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, "/path/to/user/Desktop/A stock market/src")
from afactor.panel import Panel
import afactor.strategy as st

W, HZ = 120, 60
TRAIN_END = pd.Timestamp("2023-12-31")
OUT = "/path/to/user/Desktop/A stock market/research/pathA/pathA2"
NPC = 20
EVAL_STEP = 3          # 评估日抽样
FIT_STEP = 4           # 训练样本日抽样
FIT_PER_DAY = 1200     # 每个训练日抽多少只
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
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < lag + 10:
        return np.nan
    e = x - x.mean()
    g0 = (e * e).sum() / n
    s = g0
    for l in range(1, lag + 1):
        g = (e[l:] * e[:-l]).sum() / n
        s += 2 * (1 - l / (lag + 1)) * g
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
    closev = close.values.astype(np.float64)

    f60 = np.full((T, N), np.nan)
    f60[: T - 61] = twv[61:] / twv[1: T - 60] - 1.0

    C = pd.DataFrame(close.values, index=dates, columns=codes).ffill().values.astype(np.float64)
    A = pd.DataFrame(amount.values, index=dates, columns=codes).ffill().values.astype(np.float64)
    logC = np.log(np.where(C > 0, C, np.nan))
    logA = np.log(np.where(A > 0, A, np.nan) + 1.0)
    del C, A
    mp, sdp, cntp = roll_ms_cnt(logC, W)
    ma, sda, cnta = roll_ms_cnt(logA, W)
    vc = roll_sum(valid.astype(np.float64), W)
    winok = np.zeros((T, N), bool)
    winok[W - 1:] = (cntp >= W) & (cnta >= W) & (vc >= 100) & (sdp > 1e-4) & (sda > 1e-4)
    elig = winok & univ & np.isfinite(f60)
    seg_tr = np.asarray(dates <= TRAIN_END)
    SEG = {"2020-23": seg_tr, "2024-25": ~seg_tr}
    del cntp, cnta, vc, winok
    p._cache.clear(); p._derived.clear()

    gm_c = np.nanmean(np.where(np.isfinite(logC), logC, np.nan), axis=0, keepdims=True)
    gm_a = np.nanmean(np.where(np.isfinite(logA), logA, np.nan), axis=0, keepdims=True)
    C32 = np.nan_to_num(logC - gm_c, nan=0.0).astype(np.float32)
    A32 = np.nan_to_num(logA - gm_a, nan=0.0).astype(np.float32)

    # ---------- 训练样本:全体 eligible(不只是赢家) ----------
    rng = np.random.default_rng(0)
    fit_t = [t for t in range(W - 1, T) if t % FIT_STEP == 0 and seg_tr[t]]
    Xl, yl, wl = [], [], []
    for t in fit_t:
        idx = np.flatnonzero(elig[t])
        if idx.size < 300:
            continue
        v = f60[t, idx]
        rk = (np.argsort(np.argsort(v)) + 0.5) / idx.size        # 日内百分位
        if idx.size > FIT_PER_DAY:
            sel = rng.choice(idx.size, FIT_PER_DAY, replace=False)
        else:
            sel = np.arange(idx.size)
        cols = idx[sel]
        bC = logC[t - W + 1: t + 1, cols].T
        bA = logA[t - W + 1: t + 1, cols].T
        zc = (bC - bC.mean(1, keepdims=True)) / bC.std(1, keepdims=True)
        za = (bA - bA.mean(1, keepdims=True)) / bA.std(1, keepdims=True)
        Xl.append(np.concatenate([zc, za], 1).astype(np.float32))
        yl.append(rk[sel].astype(np.float32))
        wl.append((v[sel] >= 0.5).astype(np.float32))
    X = np.concatenate(Xl); y = np.concatenate(yl); ywin = np.concatenate(wl)
    del Xl, yl, wl
    log(f"fit sample {X.shape}  win50 frac {ywin.mean():.4f}  ({time.time()-t0:.0f}s)")

    pca = PCA(n_components=NPC, random_state=0).fit(X)
    Xp = pca.transform(X).astype(np.float32)
    log("PCA evr top10: " + " ".join(f"{v:.3f}" for v in pca.explained_variance_ratio_[:10])
        + f"  cum{NPC}={pca.explained_variance_ratio_.sum():.3f}")

    ridge = Ridge(alpha=1.0).fit(Xp, y - 0.5)
    gbm = HistGradientBoostingRegressor(max_depth=4, max_iter=300, learning_rate=0.06,
                                        l2_regularization=1.0, random_state=0).fit(Xp, y - 0.5)
    log(f"models fit ({time.time()-t0:.0f}s)")

    # 线性模型折回 240 维 -> 单一匹配滤波器
    beta240 = pca.components_.T @ ridge.coef_          # (240,)
    const = -float(pca.mean_ @ beta240) + float(ridge.intercept_)
    bp, ba = beta240[:W].astype(np.float64), beta240[W:].astype(np.float64)
    log("LINEAR TEMPLATE px knots: " +
        str([round(float(bp[i]), 3) for i in [0, 20, 40, 60, 80, 100, 110, 119]]))
    log("LINEAR TEMPLATE amt knots: " +
        str([round(float(ba[i]), 3) for i in [0, 20, 40, 60, 80, 100, 110, 119]]))

    def dot_block(X32b, kern, mroll, sdroll, gm):
        d = xcorr(X32b, kern)
        return (d - (mroll - gm) * kern.sum()) / sdroll

    lin = np.full((T, N), np.nan)
    lin[W - 1:] = dot_block(C32, bp, mp, sdp, gm_c) + dot_block(A32, ba, ma, sda, gm_a) + const
    log(f"linear scored ({time.time()-t0:.0f}s)")

    # ---------- PC 分数(仅评估日) ----------
    eval_t = np.array([t for t in range(W - 1, T) if t % EVAL_STEP == 0 and np.isfinite(f60[t]).any()])
    eval_t = eval_t[elig[eval_t].sum(1) >= 300]
    row_of = {t: i for i, t in enumerate(eval_t)}
    PCS = np.zeros((NPC, len(eval_t), N), np.float32)
    for m in range(NPC):
        vp = pca.components_[m, :W].astype(np.float64)
        va = pca.components_[m, W:].astype(np.float64)
        full = np.full((T, N), np.nan)
        full[W - 1:] = dot_block(C32, vp, mp, sdp, gm_c) + dot_block(A32, va, ma, sda, gm_a)
        PCS[m] = full[eval_t].astype(np.float32)
        del full
    offs = (pca.mean_ @ pca.components_.T).astype(np.float32)      # (NPC,)
    PCS -= offs[:, None, None]
    log(f"PC scored on {len(eval_t)} dates ({time.time()-t0:.0f}s)")

    gbm_s = np.full((len(eval_t), N), np.nan, np.float32)
    for i, t in enumerate(eval_t):
        idx = np.flatnonzero(elig[t])
        if idx.size == 0:
            continue
        gbm_s[i, idx] = gbm.predict(PCS[:, i, idx].T)
    del PCS
    log(f"gbm scored ({time.time()-t0:.0f}s)")

    # ---------- 带止损的路径收益(评估日) ----------
    STOPS = [0.08, 0.12, 0.20]
    RS = {s: np.full((len(eval_t), N), np.nan, np.float32) for s in STOPS}
    RS[0.0] = np.full((len(eval_t), N), np.nan, np.float32)
    for i, t in enumerate(eval_t):
        if t + 61 > T - 1:
            continue
        e = twv[t + 1]
        exitp_final = twv[t + 61]
        base = exitp_final / e - 1.0
        RS[0.0][i] = base.astype(np.float32)
        cl = closev[t + 1: t + 60]              # close t+1..t+59
        twx = twv[t + 2: t + 61]                # 对应下一开盘 t+2..t+60
        for s in STOPS:
            thr = e * (1 - s)
            trig = cl <= thr
            any_t = trig.any(0)
            first = trig.argmax(0)
            ex = np.take_along_axis(twx, first[None, :], 0)[0]
            RS[s][i] = np.where(any_t, ex / e - 1.0, base).astype(np.float32)
    log(f"stop paths ({time.time()-t0:.0f}s)")

    # ---------- 评估 ----------
    elig_e = elig[eval_t]
    f60_e = f60[eval_t]
    win_e = f60_e >= 0.50
    dates_e = dates[eval_t]
    seg_e = {k: np.asarray(dates_e <= TRAIN_END) if k == "2020-23"
             else ~np.asarray(dates_e <= TRAIN_END) for k in SEG}

    def eval_sig(sig_e, name, q=0.005):
        rows = []
        H = np.zeros_like(elig_e)
        for i in range(len(eval_t)):
            m = elig_e[i] & np.isfinite(sig_e[i])
            n = int(m.sum())
            if n < 200:
                continue
            k = max(1, int(round(n * q)))
            idx = np.flatnonzero(m)
            H[i, idx[np.argpartition(-sig_e[i][idx], k - 1)[:k]]] = True
        for sname, sm in seg_e.items():
            d = np.flatnonzero(sm & (H.sum(1) > 0))
            if len(d) < 20:
                continue
            ph = np.array([win_e[i, H[i]].mean() for i in d])
            bh = np.array([win_e[i, elig_e[i]].mean() for i in d])
            rec = {"signal": name, "seg": sname, "n_days": len(d),
                   "n_hits": int(H[d].sum()),
                   "prec50": float(ph.mean()), "base50": float(bh.mean()),
                   "lift50": float(ph.mean() / bh.mean())}
            for s in [0.0] + STOPS:
                eh = np.array([np.nanmean(RS[s][i, H[i]]) for i in d])
                eb = np.array([np.nanmean(RS[s][i, elig_e[i]]) for i in d])
                ex = eh - eb
                lbl = "nostop" if s == 0.0 else f"stop{int(s*100)}"
                rec[f"exc_{lbl}"] = float(np.nanmean(ex))
                rec[f"t_{lbl}"] = nw_t(ex, 60)
                rec[f"abs_{lbl}"] = float(np.nanmean(eh))
            rows.append(rec)
        return rows

    all_rows = []
    all_rows += eval_sig(lin[eval_t], "LINEAR_shape(ridge)")
    all_rows += eval_sig(-lin[eval_t], "LINEAR_shape_NEG")
    all_rows += eval_sig(gbm_s, "GBM_shape")
    all_rows += eval_sig(-gbm_s, "GBM_shape_NEG")
    # KMeans 关键模板 + 基线对照
    cen = np.load("/path/to/user/Desktop/A stock market/research/pathA/pathA_centers.npy")
    k8 = cen[6 + 7: 6 + 7 + 8]                 # k=6(6) + k=7(7) 之后是 k=8 的 8 行
    for j in [0, 3, 4]:
        m240 = k8[j]
        sc = np.full((T, N), np.nan)
        sc[W - 1:] = (dot_block(C32, m240[:W], mp, sdp, gm_c)
                      + dot_block(A32, m240[W:], ma, sda, gm_a)) / (2 * W * m240.std())
        all_rows += eval_sig(sc[eval_t], f"KM_k8_c{j}")
        del sc
    mom120 = np.full((T, N), np.nan); mom120[W - 1:] = logC[W - 1:] - logC[: T - W + 1]
    all_rows += eval_sig(mom120[eval_t], "BASE_mom120")
    all_rows += eval_sig(-mom120[eval_t], "BASE_mom120_NEG")
    rng2 = np.random.default_rng(7)
    all_rows += eval_sig(rng2.standard_normal((len(eval_t), N)).astype(np.float32), "RANDOM")

    df = pd.DataFrame(all_rows)
    df.to_csv(f"{OUT}_supervised.csv", index=False)
    with pd.option_context("display.width", 260, "display.max_columns", 40,
                           "display.max_rows", 200, "display.float_format", lambda x: f"{x:.4f}"):
        log("\n===== SUPERVISED / STOP-LOSS RESULTS (daily top 0.5%) =====")
        log(df.to_string(index=False))

    # 秩相关(整体预测力,而非只看 top)
    log("\n===== 日内 Spearman IC of shape models =====")
    for nm, S in [("LINEAR", lin[eval_t]), ("GBM", gbm_s)]:
        for sname, sm in seg_e.items():
            ics = []
            for i in np.flatnonzero(sm):
                m = elig_e[i] & np.isfinite(S[i])
                if m.sum() < 200:
                    continue
                a = np.argsort(np.argsort(S[i][m])).astype(float)
                b = np.argsort(np.argsort(f60_e[i][m])).astype(float)
                ics.append(np.corrcoef(a, b)[0, 1])
            ics = np.array(ics)
            log(f"  {nm:7s} {sname}: meanIC={ics.mean():+.4f}  NW-t={nw_t(ics,60):+.2f}  n={len(ics)}")
    log(f"DONE {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
