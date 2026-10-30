"""路径 B 第 4 步:直接问"能不能事前认出未来的大涨股"。

用户关心的不是十分位价差,而是"这一根上升通道会不会走出下一段"。
所以做分类版:标签 = 未来 20 日收益进入当日截面前 5%(以及 fwd20 > +20% 的绝对口径)。
  模型 path(120 维原始序列) / summ(4 汇总统计量) / shuf(时间打乱)。
报 OOS AUC、Top-100 集中组合的命中率与实际收益、以及**事后**大涨股与其余股票
在 120 维特征上的可分性(Cohen's d),即"形状本身到底有没有区分度"。
"""
from __future__ import annotations
import sys, time, json, gc
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

sys.path.insert(0, "/path/to/user/Desktop/A stock market/src")
sys.path.insert(0, "/path/to/user/Desktop/A stock market/scripts")
from pathb_build import Builder, OUT, H, WIN
from pathb_train_eval import nw_t, to_path, shuffle_time, SEED
from pathb_stats import train_feat, NCOL, feat

KEYS = ["path", "summ", "shuf"]
BW = max(5, 2 * H)


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def main():
    meta = json.loads((OUT / "meta.json").read_text())
    dates = pd.to_datetime(meta["dates"])
    DT = np.load(OUT / "train_DT.npy"); Y = np.load(OUT / "train_Y.npy")
    AUX = np.load(OUT / "train_AUX.npy")
    ybin = (Y > 0.95).astype(np.int8)
    log("train rows", len(Y), "pos rate", ybin.mean().round(4))
    b = Builder()
    test_t = [int(t) for t in np.arange(len(dates))
              if dates[t] >= pd.Timestamp("2024-01-01") and t + H + 1 < len(dates)]

    models = {}
    for k in KEYS:
        F = train_feat(k, None, AUX, np.random.default_rng(SEED))
        m = HistGradientBoostingClassifier(
            max_iter=400, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=500,
            l2_regularization=1.0, max_bins=128, early_stopping=False, random_state=SEED)
        m.fit(F, ybin)
        models[k] = m
        log("  fit", k, F.shape)
        del F; gc.collect()

    rng = np.random.default_rng(SEED + 7)
    acc = {k: {"auc": [], "hit100": [], "ret100": [], "big100": []} for k in KEYS}
    base_hit, base_big, base_ret = [], [], []
    # 事后可分性:实际前 5% vs 其余,在 120 维路径上的均值差
    sw = np.zeros(120); so = np.zeros(120); s2w = np.zeros(120); s2o = np.zeros(120)
    nw_, no_ = 0, 0
    for t in test_t:
        r = b.build(t)
        if r is None:
            continue
        Xraw, y, aux, idx = r
        fwd = b.FWD[t, idx]
        lab = (y > 0.95).astype(int)
        big = (fwd > 0.20).astype(int)
        base_hit.append(lab.mean()); base_big.append(big.mean()); base_ret.append(fwd.mean())
        P = to_path(Xraw.copy())
        m = lab == 1
        sw += P[m].sum(0); s2w += (P[m] ** 2).sum(0); nw_ += int(m.sum())
        so += P[~m].sum(0); s2o += (P[~m] ** 2).sum(0); no_ += int((~m).sum())
        for k, mdl in models.items():
            pr = mdl.predict_proba(feat(k, Xraw.copy(), aux, rng))[:, 1]
            acc[k]["auc"].append(roc_auc_score(lab, pr) if lab.sum() > 5 else np.nan)
            o = np.argsort(-pr)[:100]
            acc[k]["hit100"].append(float(lab[o].mean()))
            acc[k]["big100"].append(float(big[o].mean()))
            acc[k]["ret100"].append(float(fwd[o].mean()))
    out = {"base_top5pct_rate": round(float(np.mean(base_hit)), 4),
           "base_fwd20_gt20pct_rate": round(float(np.mean(base_big)), 4),
           "base_fwd20_mean": round(float(np.mean(base_ret)), 4), "n_days": len(base_hit)}
    for k in KEYS:
        d = acc[k]
        mret, tret = nw_t(np.array(d["ret100"]) - np.array(base_ret), BW)
        mbig, tbig = nw_t(np.array(d["big100"]) - np.array(base_big), BW)
        out[k] = {"auc": round(float(np.nanmean(d["auc"])), 4),
                  "top100_top5pct_hit": round(float(np.mean(d["hit100"])), 4),
                  "top100_fwd20_gt20pct": round(float(np.mean(d["big100"])), 4),
                  "top100_gt20pct_excess": round(mbig, 4),
                  "top100_gt20pct_t": round(tbig, 2) if np.isfinite(tbig) else None,
                  "top100_fwd20_mean": round(float(np.mean(d["ret100"])), 4),
                  "top100_excess": round(mret, 4),
                  "top100_excess_t": round(tret, 2) if np.isfinite(tret) else None}
        log(" ", k, out[k])
    mw, mo = sw / nw_, so / no_
    vw = np.maximum(s2w / nw_ - mw ** 2, 1e-9); vo = np.maximum(s2o / no_ - mo ** 2, 1e-9)
    d = (mw - mo) / np.sqrt((vw + vo) / 2)
    out["expost_cohens_d_price"] = [round(float(v), 3) for v in d[:WIN]]
    out["expost_cohens_d_vol"] = [round(float(v), 3) for v in d[WIN:]]
    out["expost_max_abs_d"] = round(float(np.abs(d).max()), 3)
    out["expost_winner_mean_price_path"] = [round(float(v), 2) for v in mw[:WIN]]
    out["expost_other_mean_price_path"] = [round(float(v), 2) for v in mo[:WIN]]
    out["expost_winner_mean_vol_path"] = [round(float(v), 2) for v in mw[WIN:]]
    out["expost_other_mean_vol_path"] = [round(float(v), 2) for v in mo[WIN:]]
    log("expost max |Cohen d| over 120 dims =", out["expost_max_abs_d"])
    (OUT / "bigwin.json").write_text(json.dumps(out, indent=1, default=float))
    log("written", OUT / "bigwin.json")


if __name__ == "__main__":
    main()
