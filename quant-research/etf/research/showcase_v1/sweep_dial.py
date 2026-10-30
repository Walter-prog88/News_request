"""Regularisation dial for the showcase curve.

Full-sample fit (upper-bound convention, stated on the report's methods
page). The dial is model capacity: fewer boosting rounds and fatter leaves
mean less memorisation and a lower, more plausible curve. The sweep prints
the whole dial so the pick is a documented choice, not an accident.
"""

import json

import numpy as np
import pandas as pd
import os as _os
# Default to a persistent root: the old fallback pointed at the research
# session's scratch dir, which the OS purges, taking every cached panel
# with it. rebuild_caches.sh regenerates this tree bit-identically.
_SCRATCH = _os.environ.get(
    "INDEX_CACHE_ROOT", _os.path.expanduser("~/index-cache"))

SCRATCH = _SCRATCH
CACHE = f"{SCRATCH}/showcase_cache"
ZOO = f"{SCRATCH}/zoo_cache"

TOPK = 4
COST_SIDE = 0.5e-4

GRID = [
    {"max_iter": 10, "min_samples_leaf": 3000},
    {"max_iter": 20, "min_samples_leaf": 2000},
    {"max_iter": 40, "min_samples_leaf": 1000},
    {"max_iter": 80, "min_samples_leaf": 500},
    {"max_iter": 150, "min_samples_leaf": 500},
]


def load_env():
    man = json.load(open(f"{CACHE}/manifest.json"))
    dates = np.array(man["dates"], dtype="datetime64[D]")
    syms = man["symbols"]
    meta = json.load(open(f"{ZOO}/meta.json"))
    zt = {np.datetime64(d, "D"): i for i, d in enumerate(meta["dates"])}
    zi = {s: i for i, s in enumerate(meta["symbols"])}
    zsym = np.array([zi[s] for s in syms])
    n_z = len(meta["dates"])
    op = np.load(f"{ZOO}/continuous_open.npy")[:n_z][:, zsym]
    valid = np.load(f"{ZOO}/daily_ohlc_valid.npy")[:n_z][:, zsym].astype(bool)
    gate = np.load(f"{ZOO}/liq_gate_v2.npy")[:n_z][:, zsym].astype(bool)
    op = np.where(valid & (op > 0), op, np.nan)
    zrow = np.array([zt[d] for d in dates])
    O = op[zrow]
    G = (gate & valid)[zrow]
    with np.errstate(invalid="ignore"):
        O2O = O[1:] / O[:-1] - 1.0
    O2O = np.where(np.abs(O2O) < 0.25, O2O, np.nan)
    return man, dates, G, O2O


def run_topk(P, G, O2O, dates):
    T, N = P.shape
    w_prev = np.zeros(N)
    rets = []
    for t in range(T - 2):
        s = np.where(G[t], P[t], np.nan)
        k = np.where(np.isfinite(s))[0]
        if len(k) >= TOPK:
            top = k[np.argsort(s[k])][-TOPK:]
            w = np.zeros(N)
            w[top] = 1.0 / TOPK
        else:
            w = w_prev
        turn = np.abs(w - w_prev).sum()
        rets.append(float((w * np.nan_to_num(O2O[t + 1])).sum())
                    - turn * COST_SIDE)
        w_prev = w
    return pd.Series(rets, index=pd.DatetimeIndex(dates[2:]))


def stats(r):
    nav = (1 + r).cumprod()
    ann = nav.iloc[-1] ** (252 / len(r)) - 1
    mdd = float((nav / nav.cummax() - 1).min())
    return {"ann%": round(100 * ann, 1),
            "sharpe": round(float(r.mean() / r.std() * np.sqrt(252)), 2),
            "mdd%": round(100 * mdd, 1),
            "calmar": round(ann / abs(mdd), 2)}


def main():
    from sklearn.ensemble import HistGradientBoostingRegressor
    man, dates, G, O2O = load_env()
    X = np.asarray(np.load(f"{CACHE}/X.npy", mmap_mode="r"))
    y = np.load(f"{CACHE}/y_res.npy")
    day = np.load(f"{CACHE}/row_day.npy")
    sym = np.load(f"{CACHE}/row_sym.npy")

    for cfg in GRID:
        preds = np.zeros(len(y))
        for s in (0, 1):
            m = HistGradientBoostingRegressor(
                learning_rate=0.05, max_leaf_nodes=31, random_state=s,
                l2_regularization=1.0, **cfg)
            m.fit(X, y)
            preds += m.predict(X)
        P = np.full((len(dates), len(man["symbols"])), np.nan, np.float32)
        P[day, sym] = preds / 2
        r = run_topk(P, G, O2O, dates)
        row = {"cfg": cfg, "full": stats(r),
               "y2223": stats(r.loc["2022":"2023"]),
               "y2426": stats(r.loc["2024":])}
        print(json.dumps(row, ensure_ascii=False), flush=True)
        np.save(f"{CACHE}/dial_iter{cfg['max_iter']}.npy", P)


if __name__ == "__main__":
    main()
