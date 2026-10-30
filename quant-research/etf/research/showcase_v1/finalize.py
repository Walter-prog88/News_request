"""Final showcase assembly.

Locks the chosen dial (max_iter=10 / min_samples_leaf=3000, full-sample,
2 seeds), trains the rank twin at the same capacity, stacks the three
paradigms, adds a 15%-vol-targeted variant, and writes every report
ingredient: NAV series, window metrics, decile layers, permutation feature
importance, benchmark series, and the ritual audit.
"""

import json

import numpy as np
import pandas as pd

from sweep_dial import CACHE, ZOO, COST_SIDE, TOPK, load_env, run_topk, stats

OUT = "/path/to/user/Desktop/ETF-Research/index project/research/showcase_v1/results"
DIAL = {"max_iter": 10, "min_samples_leaf": 3000}
VOL_TARGET = 0.15
OLD = {"ann%": 64.35, "sharpe": 2.29, "mdd%": -18.21, "calmar": 3.53}
BENCH = "510500.SH"


def win_stats(r):
    return {"2019-2021": stats(r.loc["2019":"2021"]),
            "2022-2023": stats(r.loc["2022":"2023"]),
            "2024-2026": stats(r.loc["2024":]),
            "full": stats(r)}


def main():
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.inspection import permutation_importance
    man, dates, G, O2O = load_env()
    X = np.asarray(np.load(f"{CACHE}/X.npy", mmap_mode="r"))
    yres = np.load(f"{CACHE}/y_res.npy")
    yrank = np.load(f"{CACHE}/y_rank.npy")
    day = np.load(f"{CACHE}/row_day.npy")
    sym = np.load(f"{CACHE}/row_sym.npy")
    names = man["feature_names"]

    def fit_panel(y):
        preds = np.zeros(len(y))
        last = None
        for s in (0, 1):
            m = HistGradientBoostingRegressor(
                learning_rate=0.05, max_leaf_nodes=31, random_state=s,
                l2_regularization=1.0, **DIAL)
            m.fit(X, y)
            preds += m.predict(X)
            last = m
        P = np.full((len(dates), len(man["symbols"])), np.nan, np.float32)
        P[day, sym] = preds / 2
        return P, last

    P_gbdt = np.load(f"{CACHE}/dial_iter10.npy")
    P_rank, m_rank = fit_panel(yrank)
    P_mlp = np.load(f"{CACHE}/insample/score_mlp.npy")

    def z(P):
        mu = np.nanmean(P, axis=1, keepdims=True)
        sd = np.nanstd(P, axis=1, keepdims=True)
        return (P - mu) / np.where(sd > 0, sd, np.nan)

    P_stack = np.nanmean(np.stack([z(P_gbdt), z(P_rank), z(P_mlp)]), axis=0)

    curves = {}
    for name, P in [("gbdt", P_gbdt), ("rank", P_rank),
                    ("mlp", P_mlp), ("stack", P_stack)]:
        curves[name] = run_topk(P, G, O2O, dates)

    # vol-targeted variant of the stack
    r = curves["stack"]
    vol = r.rolling(21, min_periods=10).std() * np.sqrt(252)
    scale = (VOL_TARGET * np.sqrt(252) / 252 ** 0.5 / vol
             ).clip(upper=1.0).shift(1).fillna(1.0)
    scale = (VOL_TARGET / vol).clip(upper=1.0).shift(1).fillna(1.0)
    curves["stack_voltgt"] = (r * scale
                              - scale.diff().abs().fillna(0.0) * COST_SIDE)

    # benchmark: CSI500 ETF buy-and-hold on the same open-to-open marks
    if BENCH in man["symbols"]:
        j = man["symbols"].index(BENCH)
        rb = pd.Series(O2O[1:, j], index=pd.DatetimeIndex(dates[2:])).fillna(0)
        curves["bench_csi500"] = rb

    report = {"dial": DIAL, "cost_bp_per_side": 0.5, "topk": TOPK,
              "convention": "full-sample fit, signal t, fill t+1 open",
              "old_candidate_2024_2026": OLD, "curves": {}}
    navs = {}
    for name, r in curves.items():
        report["curves"][name] = win_stats(r.dropna())
        navs[name] = (1 + r.fillna(0)).cumprod()
        print(name, report["curves"][name]["full"], flush=True)
    pd.DataFrame(navs).to_parquet(f"{OUT}/nav_showcase.parquet")

    # decile layers on the stack
    T = len(dates)
    dec = np.full((T - 2, 10), np.nan, np.float32)
    for t in range(T - 2):
        s = np.where(G[t], P_stack[t], np.nan)
        k = np.where(np.isfinite(s))[0]
        if len(k) < 50:
            continue
        q = np.quantile(s[k], np.linspace(0, 1, 11))
        for d in range(10):
            mset = k[(s[k] >= q[d]) & (s[k] <= q[d + 1])]
            if len(mset):
                dec[t, d] = np.nanmean(O2O[t + 1][mset])
    dd = pd.DataFrame(dec, index=pd.DatetimeIndex(dates[2:]),
                      columns=[f"D{i+1}" for i in range(10)])
    dd.to_parquet(f"{OUT}/deciles_showcase.parquet")
    report["decile_ann%"] = {c: round(float(dd[c].mean() * 252 * 100), 1)
                             for c in dd.columns}

    # permutation importance on a subsample (report chart)
    rng = np.random.default_rng(0)
    sub = rng.choice(len(yres), 30000, replace=False)
    imp = permutation_importance(m_rank, X[sub], yrank[sub],
                                 n_repeats=3, random_state=0, n_jobs=4)
    top = np.argsort(-imp.importances_mean)[:30]
    report["feature_importance_top30"] = [
        {"factor": names[i], "importance": round(float(
            imp.importances_mean[i]), 6)} for i in top]

    def py(o):
        if isinstance(o, dict):
            return {k: py(v) for k, v in o.items()}
        if isinstance(o, list):
            return [py(v) for v in o]
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        return o

    with open(f"{OUT}/showcase_metrics.json", "w") as fh:
        json.dump(py(report), fh, ensure_ascii=False, indent=1)
    print("saved", OUT)


if __name__ == "__main__":
    main()
