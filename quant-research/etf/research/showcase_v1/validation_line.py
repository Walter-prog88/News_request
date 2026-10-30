"""Model-based validation line: develop before 2024, walk forward after.

The existing validation line is a single frozen factor plus the frozen regime
gate. This asks whether the model stack — 312 factors plus the pre-2024
factor factory — earns its keep under the same discipline the user specified:
everything is fitted on data before 2024-01-02, and the 2024-2026 window is
executed forward only, with quarterly refits that never look past the day
being traded.

Variants, all on the same engine (daily Top4, t+1 open fill, 0.5bp per side):
  frozen      one fit on pre-2024 data, never refitted
  rolling     refit every 63 sessions on all data available at that point
  and each of those with the frozen regime gate applied.
"""

import json
import os

import numpy as np
import pandas as pd

from sweep_dial import CACHE, COST_SIDE, TOPK, load_env, stats

OUT = "/path/to/user/Desktop/ETF-Research/index project/research/showcase_v1/results"
REGIME = ("/path/to/user/Desktop/ETF-Research/index project/research/active_system_v1/"
          "regime_signal.parquet")
DEV_END = 1213                    # 2024-01-02, first validation session
RETRAIN = 63
GATE_TH = 0.30
DIAL = {"max_iter": 10, "min_samples_leaf": 3000}
TAG = os.environ.get("FACTORY_TAG", "_pre2024")


def fit_predict(Xtr, ytr, Xte, seeds=2):
    from sklearn.ensemble import HistGradientBoostingRegressor
    p = np.zeros(len(Xte))
    for s in range(seeds):
        m = HistGradientBoostingRegressor(
            learning_rate=0.05, max_leaf_nodes=31, random_state=s,
            l2_regularization=1.0, **DIAL)
        m.fit(np.nan_to_num(Xtr, nan=0.5), ytr)
        p += m.predict(np.nan_to_num(Xte, nan=0.5))
    return p / seeds


def topk_run(P, G, O2O, dates, gate_on=None):
    T, N = P.shape
    w_prev = np.zeros(N)
    rets = []
    for t in range(T - 2):
        if gate_on is not None and not gate_on[t]:
            w = np.zeros(N)
        else:
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


def main():
    man, dates, G, O2O = load_env()
    shape = (len(dates), len(man["symbols"]))
    day = np.load(f"{CACHE}/row_day.npy")
    sym = np.load(f"{CACHE}/row_sym.npy")
    y = np.load(f"{CACHE}/y_res.npy")

    X = np.asarray(np.load(f"{CACHE}/X.npy", mmap_mode="r"))
    x2 = f"{CACHE}/X2{TAG}.npy"
    if os.path.exists(x2):
        X = np.asarray(np.load(x2, mmap_mode="r"))
        print(f"using extended matrix {X.shape} from {TAG}", flush=True)
    else:
        print(f"extended matrix {x2} absent; 312 features only", flush=True)

    score = pd.read_parquet(REGIME)["regime_score"].reindex(
        pd.DatetimeIndex(dates)).ffill()
    gate_on = (score >= GATE_TH).values

    out = {}
    for mode in ("frozen", "rolling"):
        P = np.full(shape, np.nan, np.float32)
        if mode == "frozen":
            tr = day < DEV_END
            te = day >= DEV_END
            P[day[te], sym[te]] = fit_predict(X[tr], y[tr], X[te])
        else:
            for t0 in range(DEV_END, day.max() + 1, RETRAIN):
                tr = day < t0
                te = (day >= t0) & (day < t0 + RETRAIN)
                if te.sum() == 0:
                    continue
                P[day[te], sym[te]] = fit_predict(X[tr], y[tr], X[te])
                print(f"  {mode} t0={t0} train={tr.sum()} test={te.sum()}",
                      flush=True)
        for gated in (False, True):
            r = topk_run(P, G, O2O, dates, gate_on if gated else None)
            key = f"{mode}{'_gated' if gated else ''}"
            out[key] = {w: stats(r.loc[lo:hi].dropna())
                        for w, (lo, hi) in [("2024-2026", ("2024", "2027"))]}
            r.rename("ret").to_frame().to_parquet(
                f"{OUT}/validation_{key}.parquet")
            print(key, out[key], flush=True)

    out["_meta"] = {"dev_end_day": DEV_END, "dev_end_date": str(dates[DEV_END]),
                    "n_features": int(X.shape[1]), "retrain_sessions": RETRAIN,
                    "gate_threshold": GATE_TH,
                    "time_in_market": round(float(
                        gate_on[DEV_END:].mean()), 3)}
    with open(f"{OUT}/validation_line.json", "w") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
