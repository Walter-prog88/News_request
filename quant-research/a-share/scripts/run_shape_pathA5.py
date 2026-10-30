"""路径 A 第五阶段:把"形态"和"最近几天的短期反转"拆开。
三个变体,全部同一套训练/检验协议:
  FULL120   : 用完整 120 日形状(价格+量能)
  DROP_LAST20: 只用 T-119..T-20 这 100 天的形状(把最近 20 天整段挖掉)
  ONLY_LAST20: 只用最近 20 天
若 DROP_LAST20 的 OOS IC 崩塌,说明所谓"形态"其实就是末端短期反转,和路径形状无关。
同时输出线性模板的 12 个 10 日分箱权重(可读)。
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
    del cp, ca, vc, ok, mp, sdp, ma, sda
    p._cache.clear(); p._derived.clear()

    mom120 = np.full((T, N), np.nan); mom120[W - 1:] = logC[W - 1:] - logC[: T - W + 1]

    SLICES = {"FULL120": slice(0, 120), "DROP_LAST20": slice(0, 100), "ONLY_LAST20": slice(100, 120)}

    def build(tlist, sl):
        Xl, yl, keys = [], [], []
        for t in tlist:
            idx = np.flatnonzero(elig[t])
            if idx.size < 300: continue
            v = f60[t, idx]
            rk = (np.argsort(np.argsort(v)) + 0.5) / idx.size
            rng = np.random.default_rng(t)
            sel = rng.choice(idx.size, FIT_PER_DAY, replace=False) if idx.size > FIT_PER_DAY else np.arange(idx.size)
            cols = idx[sel]
            bC = logC[t - W + 1: t + 1, cols].T[:, sl]
            bA = logA[t - W + 1: t + 1, cols].T[:, sl]
            sc = bC.std(1, keepdims=True); sa = bA.std(1, keepdims=True)
            sc[sc < 1e-6] = 1e-6; sa[sa < 1e-6] = 1e-6
            zc = (bC - bC.mean(1, keepdims=True)) / sc
            za = (bA - bA.mean(1, keepdims=True)) / sa
            Xl.append(np.concatenate([zc, za], 1).astype(np.float32))
            yl.append(rk[sel].astype(np.float32))
            keys.append((t, cols))
        return np.concatenate(Xl), np.concatenate(yl), keys

    fit_t = [t for t in range(W - 1, T) if t % FIT_STEP == 0 and seg_tr[t]]
    eval_t = [t for t in range(W - 1, T) if t % EVAL_STEP == 0 and elig[t].sum() >= 300]

    rows = []
    curves = {}
    for sname, sl in SLICES.items():
        L = sl.stop - sl.start
        X, y, _ = build(fit_t, sl)
        pca = PCA(n_components=min(NPC, 2 * L), random_state=0).fit(X)
        Xp = pca.transform(X).astype(np.float32)
        ridge = Ridge(alpha=1.0).fit(Xp, y - 0.5)
        gbm = HistGradientBoostingRegressor(max_depth=4, max_iter=300, learning_rate=0.06,
                                            l2_regularization=1.0, random_state=0).fit(Xp, y - 0.5)
        beta = pca.components_.T @ ridge.coef_
        if sname == "FULL120":
            bn = beta / beta.std()
            bins = lambda a: [round(float(a[i * 10:(i + 1) * 10].mean()), 2) for i in range(len(a) // 10)]
            log("LINEAR 模板 价格块 12x10日均权重: " + str(bins(bn[:L])))
            log("LINEAR 模板 量能块 12x10日均权重: " + str(bins(bn[L:])))
            log("  最后 10 天逐日价格权重: " + str([round(float(v), 2) for v in bn[L - 10:L]]))
            log("  最后 10 天逐日量能权重: " + str([round(float(v), 2) for v in bn[2 * L - 10:2 * L]]))
        del X, Xp
        log(f"[{sname}] fit done ({time.time()-t0:.0f}s)")

        # 评估
        for mname, mdl in [("LINEAR", ridge), ("GBM", gbm)]:
            recs = {k: {"ic": [], "q": []} for k in ("2020-23", "2024-25")}
            recs_s = {k: {"ic": [], "q": []} for k in ("2020-23", "2024-25")}
            for t in eval_t:
                seg = "2020-23" if seg_tr[t] else "2024-25"
                idx = np.flatnonzero(elig[t])
                if idx.size < 200: continue
                bC = logC[t - W + 1: t + 1, idx].T[:, sl]
                bA = logA[t - W + 1: t + 1, idx].T[:, sl]
                sc = bC.std(1, keepdims=True); sa = bA.std(1, keepdims=True)
                sc[sc < 1e-6] = 1e-6; sa[sa < 1e-6] = 1e-6
                Z = np.concatenate([(bC - bC.mean(1, keepdims=True)) / sc,
                                    (bA - bA.mean(1, keepdims=True)) / sa], 1).astype(np.float32)
                s = mdl.predict(pca.transform(Z).astype(np.float32))
                fv = f60[t, idx]
                a = np.argsort(np.argsort(s)).astype(float)
                recs[seg]["ic"].append(np.corrcoef(a, np.argsort(np.argsort(fv)).astype(float))[0, 1])
                q = np.minimum((a / len(a) * 5).astype(int), 4)
                mu = fv.mean()
                recs[seg]["q"].append([fv[q == d].mean() - mu for d in range(5)])
                # 强势池
                mm = mom120[t, idx]
                thr = np.nanquantile(mm, 0.70)
                sm = np.isfinite(mm) & (mm >= thr)
                if sm.sum() >= 150:
                    ss = s[sm]; ff = fv[sm]
                    aa = np.argsort(np.argsort(ss)).astype(float)
                    recs_s[seg]["ic"].append(np.corrcoef(aa, np.argsort(np.argsort(ff)).astype(float))[0, 1])
                    qq = np.minimum((aa / len(aa) * 5).astype(int), 4)
                    m2 = ff.mean()
                    recs_s[seg]["q"].append([ff[qq == d].mean() - m2 for d in range(5)])
            for pool, R in [("ALL", recs), ("STRONG30", recs_s)]:
                for seg in ("2020-23", "2024-25"):
                    ic = np.array(R[seg]["ic"]); Q = np.array(R[seg]["q"])
                    if len(ic) < 30: continue
                    ls = Q[:, -1] - Q[:, 0]
                    rows.append({"window": sname, "model": mname, "pool": pool, "seg": seg,
                                 "n_days": len(ic), "meanIC": ic.mean(), "IC_NWt": nw_t(ic, 60),
                                 "Q1": Q[:, 0].mean(), "Q5": Q[:, -1].mean(),
                                 "LS": ls.mean(), "LS_NWt": nw_t(ls, 60)})
                    curves[(sname, mname, pool, seg)] = Q.mean(0)
        log(f"[{sname}] eval done ({time.time()-t0:.0f}s)")

    df = pd.DataFrame(rows)
    df.to_csv("/path/to/user/Desktop/A stock market/research/pathA/pathA5_windows.csv", index=False)
    with pd.option_context("display.width", 300, "display.max_columns", 30,
                           "display.max_rows", 100, "display.float_format", lambda x: f"{x:.4f}"):
        log("\n===== 窗口拆解:形态 vs 末端反转 =====")
        log(df.to_string(index=False))
    log("\n===== 五分位曲线(f60 日内超额,%)=====")
    for k, v in curves.items():
        log(f"  {k[0]:12s} {k[1]:6s} {k[2]:9s} {k[3]}: " + " ".join(f"{100*x:+6.2f}" for x in v))
    log(f"DONE {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
