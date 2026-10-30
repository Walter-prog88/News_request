"""Factory variant with the tail-reversal family removed.

130 of the 350 survivors are built on last30_ret. Prior work in this project
attributed most of that family's IC to bid-ask bounce rather than tradable
signal, and the full factory tilts hard toward 2024-2026 while giving back
2022-2023. This rebuilds the extended matrix without those formulas so the
chapter can show both variants and pick on window balance, not full-sample
headline.
"""

import json
import os

import numpy as np
import pandas as pd

from sweep_dial import CACHE, load_env, run_topk, stats

OUT = "/path/to/user/Desktop/ETF-Research/index project/research/showcase_v1/results"
DIAL = {"max_iter": 10, "min_samples_leaf": 3000}
DROP_TOKENS = ("last30_ret",)


def fit_panel(X, y, day, sym, shape):
    from sklearn.ensemble import HistGradientBoostingRegressor
    preds = np.zeros(len(y))
    for s in (0, 1):
        m = HistGradientBoostingRegressor(
            learning_rate=0.05, max_leaf_nodes=31, random_state=s,
            l2_regularization=1.0, **DIAL)
        m.fit(X, y)
        preds += m.predict(X)
    P = np.full(shape, np.nan, np.float32)
    P[day, sym] = preds / 2
    return P


def win(r):
    return {w: stats(r.loc[lo:hi].dropna())
            for w, (lo, hi) in [("2019-2021", ("2019", "2021")),
                                ("2022-2023", ("2022", "2023")),
                                ("2024-2026", ("2024", "2027")),
                                ("full", ("2019", "2027"))]}


def main():
    man, dates, G, O2O = load_env()
    day = np.load(f"{CACHE}/row_day.npy")
    sym = np.load(f"{CACHE}/row_sym.npy")
    y = np.load(f"{CACHE}/y_res.npy")
    shape = (len(dates), len(man["symbols"]))

    TAG = os.environ.get("FACTORY_TAG", "")
    X2 = np.load(f"{CACHE}/X2{TAG}.npy", mmap_mode="r")
    fac = np.load(f"{CACHE}/factory_panel{TAG}.npz", allow_pickle=True)
    fac_names = [str(n) for n in fac["names"]]
    n_base = X2.shape[1] - len(fac_names)
    assert n_base == 312, f"unexpected base width {n_base}"

    keep_fac = [i for i, nm in enumerate(fac_names)
                if not any(t in nm for t in DROP_TOKENS)]
    cols = list(range(n_base)) + [n_base + i for i in keep_fac]
    print(f"factory formulas: {len(fac_names)} -> kept {len(keep_fac)}",
          flush=True)

    Xc = np.asarray(X2[:, cols])
    r = run_topk(fit_panel(Xc, y, day, sym, shape), G, O2O, dates)
    res = {"n_features": len(cols), "n_factory_kept": len(keep_fac),
           "dropped_tokens": list(DROP_TOKENS),
           "windows": win(r)}
    print(json.dumps(res["windows"], ensure_ascii=False, indent=1))
    r.rename("ret").to_frame().to_parquet(f"{OUT}/factory_clean_nav{TAG}.parquet")
    with open(f"{OUT}/factory_clean_curve{TAG}.json", "w") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
