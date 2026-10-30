"""Holding-count sweep for the forward validation window, at two cost levels.

Baseline is Top10 equal weight. Sweeps K over a wide range at 0.5bp and
1.0bp per side so the cost sensitivity of the concentration choice is
visible: a higher cost assumption should push the optimum toward larger K,
because a bigger basket turns over less.

The frozen score panel is cached on first use — it is one GBDT fit on
pre-2024 data and does not depend on K or on the cost level.
"""

import json
import os

import numpy as np
import pandas as pd

from sweep_dial import CACHE, load_env, stats
from validation_report_data import DEV_END, frozen_scores

OUT = "/path/to/user/Desktop/ETF-Research/index project/research/showcase_v1/results"
PANEL = f"{CACHE}/score_frozen_pre2024_panel.npy"
CAP, MIN_W = 0.30, 0.03
KS = (3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 25, 30)
COSTS = (0.5e-4, 1.0e-4)


def get_panel(shape):
    if os.path.exists(PANEL):
        return np.load(PANEL)
    P = frozen_scores(shape)
    np.save(PANEL, P)
    return P


def run(P, G, O2O, dates, k, cost_side):
    """Equal-weight Top-K, daily. Returns net series and mean one-way turnover."""
    T, N = P.shape
    w_prev = np.zeros(N)
    rets, turns = [], []
    for t in range(DEV_END, T - 2):
        s = np.where(G[t], P[t], np.nan)
        idx = np.where(np.isfinite(s))[0]
        if len(idx) >= k:
            top = idx[np.argsort(s[idx])][-k:]
            w = np.zeros(N)
            w[top] = 1.0 / k
            # cap then min-weight filter, same rules as the report engine
            if 1.0 / k > CAP:
                w[top] = CAP
                w /= w.sum()
            if 1.0 / k < MIN_W:
                raise ValueError(f"K={k} breaches the 3% floor")
        else:
            w = w_prev
        turn = np.abs(w - w_prev).sum()
        rets.append(float((w * np.nan_to_num(O2O[t + 1])).sum())
                    - turn * cost_side)
        turns.append(turn)
        w_prev = w
    r = pd.Series(rets, index=pd.DatetimeIndex(dates[DEV_END + 2:]))
    return r, float(np.mean(turns) / 2)


def main():
    man, dates, G, O2O = load_env()
    P = get_panel((len(dates), len(man["symbols"])))

    rows = []
    for k in KS:
        for cost in COSTS:
            r, tw = run(P, G, O2O, dates, k, cost)
            st = stats(r)
            rows.append({"K": k, "cost_bp": cost * 1e4,
                         "one_way_turnover": round(tw, 3), **st})
            print(f"K={k:2d} cost={cost*1e4:.1f}bp  ann={st['ann%']:6.1f}%  "
                  f"sharpe={st['sharpe']:.2f}  mdd={st['mdd%']:6.1f}%  "
                  f"calmar={st['calmar']:5.2f}  turn={tw:.3f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/sweep_n.csv", index=False)
    with open(f"{OUT}/sweep_n.json", "w") as fh:
        json.dump(rows, fh, ensure_ascii=False, indent=1)

    for cost in COSTS:
        sub = df[df.cost_bp == cost * 1e4]
        print(f"\n== {cost*1e4:.1f}bp 最优 ==")
        for metric in ("sharpe", "calmar", "ann%"):
            best = sub.loc[sub[metric].idxmax()]
            print(f"  按 {metric:7s}: K={int(best.K):2d} "
                  f"({best['ann%']:.1f}% / {best.sharpe:.2f} / "
                  f"{best['mdd%']:.1f}% / {best.calmar:.2f})")


if __name__ == "__main__":
    main()
