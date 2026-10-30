"""Does refitting more often recover the full-sample curve?

Walk-forward from 2024-01-02 with refit intervals of 63 / 21 / 5 sessions,
everything else identical. A flat ladder means daily refitting cannot help
either: staleness is not what separates the 134.6% full-sample number from
the 24.8% forward one — having seen the answers is.

One seed per fit (the ladder needs ~140 fits); the 63-session rung is also
run at two seeds so the seed effect can be read off directly.
"""

import json

import numpy as np
import pandas as pd

from sweep_dial import CACHE, load_env, stats
from validation_line import DEV_END, fit_predict, topk_run

OUT = "/path/to/user/Desktop/ETF-Research/index project/research/showcase_v1/results"
INTERVALS = (63, 21, 5)


def main():
    man, dates, G, O2O = load_env()
    shape = (len(dates), len(man["symbols"]))
    day = np.load(f"{CACHE}/row_day.npy")
    sym = np.load(f"{CACHE}/row_sym.npy")
    y = np.load(f"{CACHE}/y_res.npy")
    X = np.asarray(np.load(f"{CACHE}/X.npy", mmap_mode="r"))

    out = {}
    for interval in INTERVALS:
        for seeds in ((1, 2) if interval == 63 else (1,)):
            P = np.full(shape, np.nan, np.float32)
            n_fits = 0
            for t0 in range(DEV_END, day.max() + 1, interval):
                tr = day < t0
                te = (day >= t0) & (day < t0 + interval)
                if te.sum() == 0:
                    continue
                P[day[te], sym[te]] = fit_predict(X[tr], y[tr], X[te],
                                                  seeds=seeds)
                n_fits += 1
            r = topk_run(P, G, O2O, dates)
            key = f"every_{interval}d_seeds{seeds}"
            out[key] = {"n_fits": n_fits,
                        "2024-2026": stats(r.loc["2024":].dropna())}
            r.rename("ret").to_frame().to_parquet(
                f"{OUT}/ladder_{key}.parquet")
            print(key, out[key], flush=True)

    with open(f"{OUT}/retrain_ladder.json", "w") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
