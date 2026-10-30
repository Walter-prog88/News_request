"""Walk-forward (zero look-ahead) comparison of model families.

Expanding window, retrain every 63 sessions, minimum 252 sessions of
history before the first trade. Same engine as the showcase: daily Top4,
t+1 open fills, 0.5bp per side.

  --config A   312-feature GBDT + MLP stack (the showcase model, rolled)
  --config B   flow-only (102 feature) GBDT
  --config C   linear trailing-252d IC-weighted composite (no ML)
  --config D   single factor prev_all_valid21_liq5m, no training at all
"""

import argparse
import json

import numpy as np
import pandas as pd

from sweep_dial import CACHE, load_env, run_topk, stats

UB = "/path/to/user/Desktop/ETF-Research/index-project-active-etf-upper-bound"
OUT = "/path/to/user/Desktop/ETF-Research/index project/research/showcase_v1/results"
RETRAIN = 63
MIN_TRAIN = 252


def win_stats(r):
    out = {}
    for wname, (lo, hi) in [("2020-2021", ("2020", "2021")),
                            ("2022-2023", ("2022", "2023")),
                            ("2024-2026", ("2024", "2027")),
                            ("full", ("2020", "2027"))]:
        rr = r.loc[lo:hi].dropna()
        out[wname] = stats(rr) if len(rr) > 60 else {}
    return out


def gbdt_fit_predict(Xtr, ytr, Xte, seeds=2):
    from sklearn.ensemble import HistGradientBoostingRegressor
    # long-window features are all-NaN in the earliest expanding windows and
    # crash the binner; 0.5 is the neutral value in rank space
    Xtr = np.nan_to_num(Xtr, nan=0.5)
    Xte = np.nan_to_num(Xte, nan=0.5)
    p = np.zeros(len(Xte))
    for s in range(seeds):
        m = HistGradientBoostingRegressor(
            max_iter=150, learning_rate=0.05, max_leaf_nodes=31,
            min_samples_leaf=500, l2_regularization=1.0, random_state=s)
        m.fit(Xtr, ytr)
        p += m.predict(Xte)
    return p / seeds


def mlp_fit_predict(Xtr, ytr, Xte, seeds=1, epochs=6):
    import torch
    import torch.nn as nn
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    Xtr = torch.tensor(np.nan_to_num(Xtr, nan=0.5))
    Xte_t = torch.tensor(np.nan_to_num(Xte, nan=0.5))
    yt = torch.tensor(ytr, dtype=torch.float32)
    n = len(Xtr)
    out = np.zeros(len(Xte))
    for s in range(seeds):
        torch.manual_seed(s)
        net = nn.Sequential(nn.Linear(Xtr.shape[1], 128), nn.GELU(),
                            nn.Dropout(0.1), nn.Linear(128, 32), nn.GELU(),
                            nn.Linear(32, 1)).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
        for _ in range(epochs):
            perm = torch.randperm(n)
            for i in range(0, n, 8192):
                b = perm[i:i + 8192]
                opt.zero_grad()
                nn.functional.mse_loss(
                    net(Xtr[b].to(dev)).squeeze(-1), yt[b].to(dev)).backward()
                opt.step()
        net.eval()
        with torch.no_grad():
            for i in range(0, len(Xte_t), 65536):
                out[i:i + 65536] += (net(Xte_t[i:i + 65536].to(dev))
                                     .squeeze(-1).cpu().numpy())
    return out / seeds


def walk_scores(X, y, day, fit_predict, feat_idx=None):
    if feat_idx is not None:
        X = X[:, feat_idx]
    n_days = day.max() + 1
    out = np.full(len(y), np.nan, dtype=np.float32)
    for t0 in range(MIN_TRAIN, n_days, RETRAIN):
        te = (day >= t0) & (day < t0 + RETRAIN)
        tr = day < t0
        if te.sum() == 0:
            continue
        out[te] = fit_predict(np.asarray(X[tr]), y[tr], np.asarray(X[te]))
        print(f"  t0={t0} train={tr.sum()} test={te.sum()}", flush=True)
    return out


def linear_ic_scores(X, y, day):
    """Trailing-252-session mean feature IC as weights, refreshed quarterly.
    X is already per-day cross-sectional rank in [0,1]."""
    n_days = day.max() + 1
    out = np.full(len(y), np.nan, dtype=np.float32)
    yr = y  # residual returns; rank-IC via corr of rank(X) with y
    for t0 in range(MIN_TRAIN, n_days, RETRAIN):
        tr = (day >= t0 - 252) & (day < t0)
        te = (day >= t0) & (day < t0 + RETRAIN)
        if te.sum() == 0:
            continue
        Xtr = np.asarray(X[tr])
        yc = yr[tr] - yr[tr].mean()
        ic = np.zeros(X.shape[1])
        for j in range(X.shape[1]):
            col = Xtr[:, j]
            m = np.isfinite(col)
            if m.sum() > 500:
                c = col[m] - col[m].mean()
                denom = np.sqrt((c**2).sum() * (yc[m]**2).sum())
                ic[j] = (c * yc[m]).sum() / denom if denom > 0 else 0.0
        w = ic / (np.abs(ic).sum() + 1e-12)
        Xte = np.nan_to_num(np.asarray(X[te]), nan=0.5)
        out[te] = Xte @ (w * X.shape[1])
        print(f"  t0={t0} |ic|max={np.abs(ic).max():.3f}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, choices=list("ABCD"))
    a = ap.parse_args()
    man, dates, G, O2O = load_env()
    day = np.load(f"{CACHE}/row_day.npy")
    sym = np.load(f"{CACHE}/row_sym.npy")
    y = np.load(f"{CACHE}/y_res.npy")
    X = np.load(f"{CACHE}/X.npy", mmap_mode="r")
    names = man["feature_names"]

    def panelize(s):
        P = np.full((len(dates), len(man["symbols"])), np.nan, np.float32)
        P[day, sym] = s
        return P

    if a.config == "A":
        g = walk_scores(X, y, day, gbdt_fit_predict)
        m = walk_scores(X, y, day, mlp_fit_predict)
        Pg, Pm = panelize(g), panelize(m)

        def z(P):
            mu = np.nanmean(P, axis=1, keepdims=True)
            sd = np.nanstd(P, axis=1, keepdims=True)
            return (P - mu) / np.where(sd > 0, sd, np.nan)
        P = np.nanmean(np.stack([z(Pg), z(Pm)]), axis=0)
    elif a.config == "B":
        idx = [i for i, nm in enumerate(names)
               if nm.startswith(("prev_", "same_"))]
        print("flow features:", len(idx))
        P = panelize(walk_scores(X, y, day, gbdt_fit_predict, feat_idx=idx))
    elif a.config == "C":
        P = panelize(linear_ic_scores(X, y, day))
    else:
        fl = np.load(f"{UB}/research/results/active_etf_upper_bound/"
                     "core_flow_main.npz", allow_pickle=True)
        fnames = list(fl["names"])
        j = fnames.index("prev_all_valid21_liq5m")
        fdt = {d: i for i, d in
               enumerate(fl["index"].astype("datetime64[D]"))}
        fsy = {s: i for i, s in enumerate(fl["cols"])}
        rows = np.array([fdt[d] for d in dates])
        cols = np.array([fsy[s] for s in man["symbols"]])
        P = fl["A"][j][rows][:, cols]
        # align with trained configs: no positions before the burn-in
        P[:MIN_TRAIN] = np.nan

    r = run_topk(P, G, O2O, dates)
    res = win_stats(r)
    print(json.dumps({"config": a.config, **res}, ensure_ascii=False))
    r.rename("ret").to_frame().to_parquet(f"{OUT}/rolling_{a.config}.parquet")
    with open(f"{OUT}/rolling_{a.config}.json", "w") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
