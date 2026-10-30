"""Position-sizing grid: does equal-weight Top4 stay optimal?

Grid: {five score panels} x {K=4, K=10} x {equal, rank-weighted,
inverse-vol}. House rules everywhere: 30% single-name cap, weights under
3% are dropped (the user's effective-holding definition) and the basket
renormalised, 0.5bp per side, t+1 open fills.

Rank weights are linear (K..1), so K=4 gives 40/30/20/10. Inverse-vol uses
the name's trailing 21-session O2O volatility, known at t.
"""

import json

import numpy as np
import pandas as pd

from sweep_dial import CACHE, COST_SIDE, load_env, stats

OUT = "/path/to/user/Desktop/ETF-Research/index project/research/showcase_v1/results"
CAP = 0.30
MIN_W = 0.03
KS = (4, 10)
SCHEMES = ("equal", "rank", "invvol")


def zpanel(P):
    mu = np.nanmean(P, axis=1, keepdims=True)
    sd = np.nanstd(P, axis=1, keepdims=True)
    return (P - mu) / np.where(sd > 0, sd, np.nan)


def load_panels():
    p = {}
    p["工厂662·GBDT"] = np.load(f"{CACHE}/factory_score_pre2024.npy")
    p["312·GBDT"] = np.load(f"{CACHE}/dial_iter10.npy")
    g = np.load(f"{CACHE}/insample/score_gbdt.npy")
    r = np.load(f"{CACHE}/insample/score_gbdt_rank.npy")
    m = np.load(f"{CACHE}/insample/score_mlp.npy")
    p["312·三模融合"] = np.nanmean(np.stack(
        [zpanel(np.load(f"{CACHE}/dial_iter10.npy")), zpanel(r), zpanel(m)]),
        axis=0)
    p["312·排序GBDT"] = r
    p["312·MLP"] = m
    return p


def basket_weights(scores, elig, k, scheme, vol_row):
    s = np.where(elig, scores, np.nan)
    idx = np.where(np.isfinite(s))[0]
    if len(idx) < k:
        return None
    top = idx[np.argsort(s[idx])][-k:][::-1]          # best first
    if scheme == "equal":
        raw = np.ones(k)
    elif scheme == "rank":
        raw = np.arange(k, 0, -1, dtype=float)         # K..1
    else:
        v = vol_row[top]
        v = np.where(np.isfinite(v) & (v > 1e-6), v, np.nanmedian(v))
        raw = 1.0 / v
    w = raw / raw.sum()
    for _ in range(3):                                 # cap then re-norm
        w = np.minimum(w, CAP)
        w = w / w.sum()
    keep = w >= MIN_W
    if keep.sum() == 0:
        return None
    w = w[keep] / w[keep].sum()
    for _ in range(2):
        w = np.minimum(w, CAP)
        w = w / w.sum()
    out = np.zeros(len(scores))
    out[top[keep]] = w
    return out


def run(P, G, O2O, dates, k, scheme, vol):
    T, N = P.shape
    w_prev = np.zeros(N)
    rets, n_eff = [], []
    for t in range(T - 2):
        w = basket_weights(P[t], G[t], k, scheme, vol[t])
        if w is None:
            w = w_prev
        turn = np.abs(w - w_prev).sum()
        rets.append(float((w * np.nan_to_num(O2O[t + 1])).sum())
                    - turn * COST_SIDE)
        n_eff.append(int((w > 0).sum()))
        w_prev = w
    r = pd.Series(rets, index=pd.DatetimeIndex(dates[2:]))
    return r, float(np.mean(n_eff))


def main():
    man, dates, G, O2O = load_env()
    # trailing 21-session O2O vol, known at t (uses returns up to open t)
    O2Ofull = np.vstack([np.full((1, O2O.shape[1]), np.nan), O2O])
    vol = pd.DataFrame(O2Ofull).rolling(21, min_periods=10).std().values

    rows = []
    for pname, P in load_panels().items():
        for k in KS:
            for scheme in SCHEMES:
                r, neff = run(P, G, O2O, dates, k, scheme, vol)
                st = stats(r.dropna())
                w22 = stats(r.loc["2022":"2023"].dropna())
                rows.append({"panel": pname, "K": k, "scheme": scheme,
                             "avg_eff_holdings": round(neff, 1), **st,
                             "sharpe_2223": w22.get("sharpe")})
                print(pname, k, scheme, st, "eff=", round(neff, 1),
                      flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/position_sizing_grid.csv", index=False)
    with open(f"{OUT}/position_sizing_grid.json", "w") as fh:
        json.dump(rows, fh, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
