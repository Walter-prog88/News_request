"""路径 A 第四阶段:形状模型到底是不是"动量的马甲"?
1) 重建 LINEAR(匹配滤波器)与 GBM 形状分数,输出归一化后的模板形状(可读描述)。
2) 十分位表(日内中性化超额 + NW t),看单调性与多空两端各贡献多少。
3) 正交化:每日横截面上把形状分数对 mom5/mom20/mom60/mom120/vol20/amt_z 回归取残差,
   残差还剩多少 IC 与十分位价差 —— 这是"形状 ≠ 动量"的关键检验。
4) 条件检验:只在"强势股池"(120 日涨幅前 30%)内部,形状还能不能区分。
   这是用户那位交易者的实际猎场。
"""
from __future__ import annotations
import sys, time
import numpy as np
import pandas as pd
from scipy.signal import fftconvolve
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, "/path/to/user/Desktop/A stock market/src")
from afactor.panel import Panel
import afactor.strategy as st

W = 120
TRAIN_END = pd.Timestamp("2023-12-31")
NPC, EVAL_STEP, FIT_STEP, FIT_PER_DAY = 20, 3, 4, 1200
log = lambda *a: print(*a, flush=True)


def roll_sum(X, w):
    cs = np.empty((X.shape[0] + 1, X.shape[1]), np.float64); cs[0] = 0.0
    np.cumsum(X, axis=0, out=cs[1:]); return cs[w:] - cs[:-w]


def roll_ms_cnt(X, w):
    fin = np.isfinite(X); Xf = np.where(fin, X, 0.0)
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
    x = np.asarray(x, float); x = x[np.isfinite(x)]; n = len(x)
    if n < lag + 10: return np.nan
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
    mp, sdp, cp = roll_ms_cnt(logC, W); ma, sda, ca = roll_ms_cnt(logA, W)
    vc = roll_sum(valid.astype(np.float64), W)
    ok = np.zeros((T, N), bool)
    ok[W - 1:] = (cp >= W) & (ca >= W) & (vc >= 100) & (sdp > 1e-4) & (sda > 1e-4)
    elig = ok & univ & np.isfinite(f60)
    seg_tr = np.asarray(dates <= TRAIN_END)
    del cp, ca, vc, ok
    p._cache.clear(); p._derived.clear()
    gm_c = np.nanmean(np.where(np.isfinite(logC), logC, np.nan), 0, keepdims=True)
    gm_a = np.nanmean(np.where(np.isfinite(logA), logA, np.nan), 0, keepdims=True)
    C32 = np.nan_to_num(logC - gm_c, nan=0.0).astype(np.float32)
    A32 = np.nan_to_num(logA - gm_a, nan=0.0).astype(np.float32)

    # 控制变量(全部 T 及之前)
    r = np.full((T, N), np.nan); r[1:] = logC[1:] - logC[:-1]
    ctl = {}
    for h in (5, 20, 60, 120):
        z = np.full((T, N), np.nan); z[h:] = logC[h:] - logC[:-h]; ctl[f"mom{h}"] = z
    v20 = np.full((T, N), np.nan)
    rr = np.nan_to_num(r); fin = np.isfinite(r).astype(float)
    s1 = roll_sum(rr, 20); s2 = roll_sum(rr * rr, 20); cn = roll_sum(fin, 20)
    v20[19:] = np.sqrt(np.maximum(s2 / 20 - (s1 / 20) ** 2, 0)); ctl["vol20"] = v20
    az = np.full((T, N), np.nan); az[W - 1:] = (logA[W - 1:] - ma) / sda; ctl["amt_z"] = az
    ctl["lnamt"] = logA
    CTLN = list(ctl.keys())

    # ---------- 拟合(与阶段2 相同设定)----------
    rng = np.random.default_rng(0)
    Xl, yl = [], []
    for t in [t for t in range(W - 1, T) if t % FIT_STEP == 0 and seg_tr[t]]:
        idx = np.flatnonzero(elig[t])
        if idx.size < 300: continue
        v = f60[t, idx]
        rk = (np.argsort(np.argsort(v)) + 0.5) / idx.size
        sel = rng.choice(idx.size, FIT_PER_DAY, replace=False) if idx.size > FIT_PER_DAY else np.arange(idx.size)
        cols = idx[sel]
        bC = logC[t - W + 1: t + 1, cols].T; bA = logA[t - W + 1: t + 1, cols].T
        zc = (bC - bC.mean(1, keepdims=True)) / bC.std(1, keepdims=True)
        za = (bA - bA.mean(1, keepdims=True)) / bA.std(1, keepdims=True)
        Xl.append(np.concatenate([zc, za], 1).astype(np.float32)); yl.append(rk[sel].astype(np.float32))
    X = np.concatenate(Xl); y = np.concatenate(yl); del Xl, yl
    pca = PCA(n_components=NPC, random_state=0).fit(X)
    Xp = pca.transform(X).astype(np.float32)
    ridge = Ridge(alpha=1.0).fit(Xp, y - 0.5)
    gbm = HistGradientBoostingRegressor(max_depth=4, max_iter=300, learning_rate=0.06,
                                        l2_regularization=1.0, random_state=0).fit(Xp, y - 0.5)
    beta = pca.components_.T @ ridge.coef_
    log(f"fit done {X.shape} ({time.time()-t0:.0f}s)")
    del X, Xp

    # 模板形状(归一化后可读)
    bn = beta / beta.std()
    kn = [0, 20, 40, 60, 80, 100, 110, 119]
    log("LINEAR 模板(单位方差)价格权重 knots: " + str([round(float(bn[i]), 2) for i in kn]))
    log("LINEAR 模板(单位方差)量能权重 knots: " + str([round(float(bn[W + i]), 2) for i in kn]))
    log(f"  价格块权重和={bn[:W].sum():.2f} 量能块权重和={bn[W:].sum():.2f} "
        f"|价格块|std={bn[:W].std():.2f} |量能块|std={bn[W:].std():.2f}")
    # 模板对应的"典型形状":与该滤波器最相关的形状就是 beta 本身
    zp = (bn[:W] - bn[:W].mean()); za_ = (bn[W:] - bn[W:].mean())
    log("  -> 该滤波器偏好的价格路径(累积形状,标准化): " +
        str([round(float(v), 2) for v in ((np.cumsum(zp) - np.cumsum(zp).mean()) / np.cumsum(zp).std())[kn]]))

    def dot_block(Xb, kern, mroll, sdroll, gm):
        return (xcorr(Xb, kern) - (mroll - gm) * kern.sum()) / sdroll

    lin = np.full((T, N), np.nan)
    lin[W - 1:] = dot_block(C32, beta[:W].astype(float), mp, sdp, gm_c) \
        + dot_block(A32, beta[W:].astype(float), ma, sda, gm_a)

    eval_t = np.array([t for t in range(W - 1, T) if t % EVAL_STEP == 0])
    eval_t = eval_t[elig[eval_t].sum(1) >= 300]
    PCS = np.zeros((NPC, len(eval_t), N), np.float32)
    for m in range(NPC):
        full = np.full((T, N), np.nan)
        full[W - 1:] = dot_block(C32, pca.components_[m, :W].astype(float), mp, sdp, gm_c) \
            + dot_block(A32, pca.components_[m, W:].astype(float), ma, sda, gm_a)
        PCS[m] = full[eval_t].astype(np.float32); del full
    PCS -= (pca.mean_ @ pca.components_.T).astype(np.float32)[:, None, None]
    gbm_s = np.full((len(eval_t), N), np.nan, np.float32)
    for i in range(len(eval_t)):
        idx = np.flatnonzero(elig[eval_t[i]])
        if idx.size: gbm_s[i, idx] = gbm.predict(PCS[:, i, idx].T)
    del PCS
    del C32, A32, logC, logA
    log(f"scored ({time.time()-t0:.0f}s)")

    E = elig[eval_t]; F60 = f60[eval_t]; F20 = f20[eval_t]
    CT = {k: v[eval_t] for k, v in ctl.items()}
    de = dates[eval_t]; tr_e = np.asarray(de <= TRAIN_END)
    SEGE = {"2020-23": tr_e, "2024-25": ~tr_e}
    LIN = lin[eval_t]

    def zs(a):
        a = np.asarray(a, float)
        s = a.std()
        return (a - a.mean()) / (s if s > 1e-12 else 1.0)

    def rank01(a):
        return (np.argsort(np.argsort(a)) + 0.5) / len(a)

    def analyze(score_e, name, pool=None, orth=False, ndec=10):
        """pool: 额外的股票池 mask(T_e,N)。orth: 是否对控制变量正交化。"""
        out = []
        ic_store = {}
        dec_store = {}
        for sname, sm in SEGE.items():
            ics, decs, days = [], [], np.flatnonzero(sm)
            for i in days:
                m = E[i] & np.isfinite(score_e[i])
                if pool is not None: m = m & pool[i]
                if m.sum() < 150: continue
                s = score_e[i][m].astype(float)
                if orth:
                    Xc = np.column_stack([rank01(CT[k][i][m]) for k in CTLN] + [np.ones(m.sum())])
                    bad = ~np.isfinite(Xc).all(1)
                    if bad.mean() > 0.3: continue
                    Xc = np.where(np.isfinite(Xc), Xc, 0.0)
                    sr = rank01(s)
                    coef, *_ = np.linalg.lstsq(Xc, sr, rcond=None)
                    s = sr - Xc @ coef
                fv = F60[i][m]; f2 = F20[i][m]
                a = np.argsort(np.argsort(s)).astype(float)
                b = np.argsort(np.argsort(fv)).astype(float)
                ics.append(np.corrcoef(a, b)[0, 1])
                q = np.minimum((a / len(a) * ndec).astype(int), ndec - 1)
                mu = fv.mean(); mu2 = np.nanmean(f2)
                decs.append([[np.nanmean(fv[q == d]) - mu, np.nanmean(f2[q == d]) - mu2]
                             for d in range(ndec)])
            ics = np.array(ics); D = np.array(decs)          # (days, ndec, 2)
            ic_store[sname] = ics
            dec_store[sname] = D
            out.append({"signal": name, "seg": sname, "n_days": len(ics),
                        "meanIC": ics.mean(), "IC_NWt": nw_t(ics, 60),
                        "D10_exc60": D[:, -1, 0].mean(), "D10_t": nw_t(D[:, -1, 0], 60),
                        "D1_exc60": D[:, 0, 0].mean(), "D1_t": nw_t(D[:, 0, 0], 60),
                        "LS_exc60": (D[:, -1, 0] - D[:, 0, 0]).mean(),
                        "LS_t": nw_t(D[:, -1, 0] - D[:, 0, 0], 60),
                        "D10_exc20": D[:, -1, 1].mean(), "D1_exc20": D[:, 0, 1].mean()})
        return out, dec_store

    rows = []
    store = {}
    for nm, sc in [("LINEAR", LIN), ("GBM", gbm_s), ("mom120", CT["mom120"]),
                   ("mom20", CT["mom20"])]:
        o, d = analyze(sc, nm); rows += o; store[nm] = d
    for nm, sc in [("LINEAR_orth", LIN), ("GBM_orth", gbm_s)]:
        o, d = analyze(sc, nm, orth=True); rows += o; store[nm] = d

    # 强势股池:mom120 前 30%
    pool = np.zeros_like(E)
    for i in range(len(eval_t)):
        m = E[i] & np.isfinite(CT["mom120"][i])
        if m.sum() < 150: continue
        idx = np.flatnonzero(m)
        k = int(round(idx.size * 0.30))
        pool[i, idx[np.argpartition(-CT["mom120"][i][idx], k - 1)[:k]]] = True
    for nm, sc in [("LINEAR@strong30", LIN), ("GBM@strong30", gbm_s)]:
        o, d = analyze(sc, nm, pool=pool, ndec=5); rows += o; store[nm] = d

    df = pd.DataFrame(rows)
    df.to_csv("/path/to/user/Desktop/A stock market/research/pathA/pathA4_decile.csv", index=False)
    with pd.option_context("display.width", 300, "display.max_columns", 40,
                           "display.max_rows", 100, "display.float_format", lambda x: f"{x:.4f}"):
        log("\n===== IC / 十分位(日内中性化,NW60 t) =====")
        log(df.to_string(index=False))

    log("\n===== 十分位曲线(f60 日内超额,%)=====")
    for nm in ["LINEAR", "GBM", "mom120", "LINEAR_orth", "GBM_orth"]:
        for sname in SEGE:
            D = store[nm][sname]
            log(f"  {nm:12s} {sname}: " + " ".join(f"{100*D[:,d,0].mean():+6.2f}" for d in range(D.shape[1])))
    log("\n===== 强势股池内 5 分位(f60 日内超额,%)=====")
    for nm in ["LINEAR@strong30", "GBM@strong30"]:
        for sname in SEGE:
            D = store[nm][sname]
            log(f"  {nm:16s} {sname}: " + " ".join(f"{100*D[:,d,0].mean():+6.2f}" for d in range(D.shape[1])))
    log(f"DONE {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
