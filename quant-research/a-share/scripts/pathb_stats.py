"""路径 B 第 3 步:配对显著性检验 + 追加对照 + 胜出形状画像。

在第 2 步基础上补:
  (a) 逐日 IC 序列落盘,做**配对差**的 Newey-West t —— "原始序列 vs 汇总统计量"、
      "原始序列 vs 时间打乱"这两个核心对照必须做配对检验,而不是各看各的 t。
  (b) 追加对照:vshf(只用成交额路径且打乱时间)、crse(把 120 维粗化成 12 维)。
  (c) Top-50 / Top-20 集中组合的超额(用户关心的是集中持仓,不是十分位价差)。
  (d) path 模型 D10 / D1 的平均 120 维特征画像 —— 模型真正偏好的"形状"长什么样。
内存:训练特征用 memmap 分块拼装,峰值 < 1GB。
"""
from __future__ import annotations
import sys, time, json, gc
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, "/path/to/user/Desktop/A stock market/src")
sys.path.insert(0, "/path/to/user/Desktop/A stock market/scripts")
from pathb_build import Builder, OUT, H, WIN
from pathb_train_eval import nw_t, spearman, ic_summary, to_path, shuffle_time, SEED

XPATH = OUT / "train_X.npy"
N_ITER = 400
KEYS = ["path", "summ", "shuf", "comb", "volu", "vshf", "crse"]
NCOL = {"path": 120, "summ": 4, "shuf": 120, "comb": 124, "volu": 60, "vshf": 60, "crse": 12}


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def coarse(Xp: np.ndarray) -> np.ndarray:
    """Xp 已 cumsum。价格路径取 6 个刻度点 + 成交额路径取 6 个 10 日均值。"""
    out = np.empty((len(Xp), 12), np.float32)
    out[:, :6] = Xp[:, [9, 19, 29, 39, 49, 59]]
    out[:, 6:] = Xp[:, WIN:].reshape(len(Xp), 6, 10).mean(2)
    return out


def feat(key, X, a, rng):
    """X: 原始 120 列(前 60 是归一化日收益,未 cumsum),会被就地修改。"""
    if key == "summ":
        return np.ascontiguousarray(a[:, :4])
    if key == "path":
        return to_path(X)
    if key == "shuf":
        return shuffle_time(X, rng)
    if key == "comb":
        return np.hstack([to_path(X), a[:, :4]])
    if key == "volu":
        return np.ascontiguousarray(X[:, WIN:])
    if key == "vshf":
        V = np.ascontiguousarray(X[:, WIN:])
        perm = np.argsort(rng.random(V.shape), axis=1)
        return np.take_along_axis(V, perm, axis=1)
    if key == "crse":
        return coarse(to_path(X))
    raise KeyError(key)


def train_feat(key, idx, AUX, rng, chunk=150_000):
    """分块从 memmap 拼训练特征。idx=None 表示全量。"""
    Xm = np.load(XPATH, mmap_mode="r")
    n = Xm.shape[0] if idx is None else len(idx)
    F = np.empty((n, NCOL[key]), np.float32)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        xb = np.array(Xm[s:e]) if idx is None else np.array(Xm[idx[s:e]])
        ab = AUX[s:e] if idx is None else AUX[idx[s:e]]
        F[s:e] = feat(key, xb, ab, rng)
        del xb
    del Xm
    gc.collect()
    return F


def make_model():
    return HistGradientBoostingRegressor(
        max_iter=N_ITER, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=500,
        l2_regularization=1.0, max_bins=128, early_stopping=False, random_state=SEED)


def stream(b, t_list, models, cond=False, profile_key=None):
    rng = np.random.default_rng(SEED + 7)
    res = {k: {"ic": [], "dec": [], "t50": [], "t20": []} for k in models}
    uni, ts = [], []
    prof = {"d10": np.zeros(120), "d1": np.zeros(120), "n": 0}
    for t in t_list:
        r = b.build(t)
        if r is None:
            continue
        Xraw, y, aux, idx = r
        fwd = b.FWD[t, idx]
        if cond:
            m = aux[:, 4] > np.median(aux[:, 4])
            if m.sum() < 200:
                continue
            Xraw, y, aux, fwd = Xraw[m], y[m], aux[m], fwd[m]
            y = pd.Series(y).rank().to_numpy() / (len(y) + 1.0)
        ts.append(t); uni.append(float(fwd.mean()))
        for k, mdl in models.items():
            pr = mdl.predict(feat(k, Xraw.copy(), aux, rng))
            res[k]["ic"].append(spearman(pr, y))
            rk = pd.Series(pr).rank(method="first").to_numpy()
            q = np.floor((rk - 1) / len(rk) * 10).astype(int)
            res[k]["dec"].append([float(fwd[q == g].mean()) for g in range(10)])
            o = np.argsort(-pr)
            res[k]["t50"].append(float(fwd[o[:50]].mean()))
            res[k]["t20"].append(float(fwd[o[:20]].mean()))
            if k == profile_key:
                P = to_path(Xraw.copy())
                prof["d10"] += P[q == 9].mean(0); prof["d1"] += P[q == 0].mean(0); prof["n"] += 1
    for k in res:
        for f in ["ic", "t50", "t20"]:
            res[k][f] = np.array(res[k][f], float)
        res[k]["dec"] = np.array(res[k]["dec"], float)
    return res, np.array(uni), np.array(ts), prof


BW = max(5, 2 * H)


def summarize(res, uni, ts, dates, k):
    d = res[k]
    dec = d["dec"]
    out = ic_summary(list(d["ic"]), k)
    for nm, s in [("LS", dec[:, 9] - dec[:, 0]), ("D10", dec[:, 9] - uni),
                  ("D1", dec[:, 0] - uni), ("T50", d["t50"] - uni), ("T20", d["t20"] - uni)]:
        m, t = nw_t(s, BW)
        out[f"h20_{nm}"] = round(m, 4)
        out[f"h20_{nm}_t"] = round(t, 2) if np.isfinite(t) else None
    out["dec_mean"] = [round(float(v), 4) for v in dec.mean(0)]
    yy = dates[ts].year.to_numpy()
    out["ic_by_year"] = {int(u): round(float(np.nanmean(d["ic"][yy == u])), 4) for u in np.unique(yy)}
    return out


def paired(res, a, bkey):
    """配对差 IC_a - IC_b 的 NW t。"""
    d = res[a]["ic"] - res[bkey]["ic"]
    m, t = nw_t(d, BW)
    return {"pair": f"{a}-{bkey}", "diff_mean": round(m, 5),
            "diff_t_nw": round(t, 2) if np.isfinite(t) else None}


def main():
    meta = json.loads((OUT / "meta.json").read_text())
    dates = pd.to_datetime(meta["dates"])
    DT = np.load(OUT / "train_DT.npy"); Y = np.load(OUT / "train_Y.npy")
    AUX = np.load(OUT / "train_AUX.npy")
    b = Builder()
    all_t = np.arange(len(dates))
    test_t = [int(t) for t in all_t if dates[t] >= pd.Timestamp("2024-01-01") and t + H + 1 < len(dates)]
    v23 = [int(t) for t in all_t if dates[t].year == 2023 and t >= 120 and t + H + 1 < len(dates)]
    cut = min(v23) - (H + 1)
    out = {}

    # ---- 搜索段内部留出:2020-2022 -> 2023 ----
    fidx = np.flatnonzero(DT <= cut)
    log("inner fit rows", len(fidx))
    mi = {}
    for k in KEYS:
        F = train_feat(k, fidx, AUX, np.random.default_rng(SEED))
        m = make_model(); m.fit(F, Y[fidx]); mi[k] = m
        log("  inner fit", k, F.shape)
        del F; gc.collect()
    r23, u23, ts23, _ = stream(b, v23, mi)
    out["inner_2023"] = {k: summarize(r23, u23, ts23, dates, k) for k in KEYS}
    out["inner_2023_paired"] = [paired(r23, "path", "summ"), paired(r23, "path", "shuf"),
                                paired(r23, "comb", "summ"), paired(r23, "volu", "vshf"),
                                paired(r23, "path", "crse")]
    for k, v in out["inner_2023"].items():
        log("  INNER23", k, v)
    log("  INNER23 paired", out["inner_2023_paired"])
    del mi, r23; gc.collect()

    # ---- 检验段:2020-2023 -> 2024-2025 ----
    mf = {}
    for k in KEYS:
        F = train_feat(k, None, AUX, np.random.default_rng(SEED))
        m = make_model(); m.fit(F, Y); mf[k] = m
        log("  full fit", k, F.shape)
        del F; gc.collect()
    rt, ut, tst, prof = stream(b, test_t, mf, profile_key="path")
    out["test_2024_25"] = {k: summarize(rt, ut, tst, dates, k) for k in KEYS}
    out["test_paired"] = [paired(rt, "path", "summ"), paired(rt, "path", "shuf"),
                          paired(rt, "comb", "summ"), paired(rt, "volu", "vshf"),
                          paired(rt, "path", "crse")]
    for k, v in out["test_2024_25"].items():
        log("  TEST", k, v)
    log("  TEST paired", out["test_paired"])
    out["profile_d10_price"] = [round(float(v), 3) for v in (prof["d10"][:WIN] / prof["n"])]
    out["profile_d1_price"] = [round(float(v), 3) for v in (prof["d1"][:WIN] / prof["n"])]
    out["profile_d10_vol"] = [round(float(v), 3) for v in (prof["d10"][WIN:] / prof["n"])]
    out["profile_d1_vol"] = [round(float(v), 3) for v in (prof["d1"][WIN:] / prof["n"])]

    # ---- 条件版(强势股)配对 ----
    rc, uc, tsc, _ = stream(b, test_t, mf, cond=True)
    out["test_strong_uncondmodel"] = {k: summarize(rc, uc, tsc, dates, k) for k in KEYS}
    out["test_strong_paired"] = [paired(rc, "path", "summ"), paired(rc, "path", "shuf"),
                                 paired(rc, "comb", "summ"), paired(rc, "volu", "vshf")]
    for k, v in out["test_strong_uncondmodel"].items():
        log("  TEST-STRONG", k, v)
    log("  TEST-STRONG paired", out["test_strong_paired"])

    (OUT / "stats.json").write_text(json.dumps(out, indent=1, default=float))
    log("written", OUT / "stats.json")


if __name__ == "__main__":
    main()
