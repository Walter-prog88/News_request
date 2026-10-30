"""How fast does the edge decay after the open? One pass, several TWAP windows.

TWAP over W minutes is the same thing as slicing the order into W equal
pieces, one per minute — you only do it when the order is too big for one
minute. So this curve is really a capacity-versus-return curve: each row is
"what you keep if your size forces you to spread over W minutes".

Windows share a single read of the 6.8GB close pivot.
"""

import json
import os

import numpy as np
import pandas as pd

from sweep_dial import CACHE, ZOO, load_env, stats

SRC = "/path/to/user/Desktop/ETF-Research/full-etf project2 "
OUT = "/path/to/user/Desktop/ETF-Research/index project/research/showcase_v1/results"
MULTI = f"{CACHE}/twap_windows.npz"
PANEL = f"{CACHE}/score_frozen_pre2024_panel.npy"
DEV_END, K, COST_SIDE = 1213, 10, 1.0e-4
BARS_PER_DAY = 241
WINDOWS = (1, 2, 3, 5, 10, 15, 30)
RET_CLIP = 0.25


def build(syms, zdates):
    """Built on the ZOO minute grid, then reindexed by the caller."""
    n_days = len(zdates)
    if os.path.exists(MULTI):
        z = np.load(MULTI)
        return {int(w): z[f"w{w}"] for w in WINDOWS}
    print("loading close pickle (single pass) ...", flush=True)
    raw = pd.read_pickle(f"{SRC}/etf_pivot_close_qfq.pkl")
    cols = {c: i for i, c in enumerate(raw.columns)}
    vals = raw.values
    del raw
    out = {w: np.full((n_days, len(syms)), np.nan, np.float32)
           for w in WINDOWS}
    for j, s in enumerate(syms):
        if s not in cols:
            continue
        day = vals[:, cols[s]].reshape(n_days, BARS_PER_DAY)
        day = np.where(day > 0, day, np.nan)
        for w in WINDOWS:
            with np.errstate(invalid="ignore"):
                out[w][:, j] = np.nanmean(day[:, 1:1 + w], axis=1)
    np.savez(MULTI, **{f"w{w}": out[w] for w in WINDOWS})
    return out


def run(P, G, ret, dates):
    T, N = P.shape
    w_prev = np.zeros(N)
    rets = []
    for t in range(DEV_END, T - 2):
        s = np.where(G[t] & np.isfinite(ret[t + 1]), P[t], np.nan)
        idx = np.where(np.isfinite(s))[0]
        if len(idx) >= K:
            top = idx[np.argsort(s[idx])][-K:]
            w = np.zeros(N)
            w[top] = 1.0 / K
        else:
            w = w_prev
        rets.append(float((w * np.nan_to_num(ret[t + 1])).sum())
                    - np.abs(w - w_prev).sum() * COST_SIDE)
        w_prev = w
    return pd.Series(rets, index=pd.DatetimeIndex(dates[DEV_END + 2:]))


def main():
    man, dates, G, O2O = load_env()
    syms = man["symbols"]
    P = np.load(PANEL)
    meta = json.load(open(f"{ZOO}/meta.json"))
    zdates = pd.DatetimeIndex(meta["dates"])
    raw_panels = build(syms, zdates)
    keep = zdates.get_indexer(pd.DatetimeIndex(dates))
    assert (keep >= 0).all(), "showcase dates missing from the minute grid"
    panels = {w: v[keep] for w, v in raw_panels.items()}

    rows = [{"exec": "09:31 开盘价(瞬时)", "minutes": 0,
             **stats(run(P, G, O2O, dates))}]
    print(rows[0], flush=True)
    for w in WINDOWS:
        tw = panels[w]
        with np.errstate(invalid="ignore"):
            ret = tw[1:] / tw[:-1] - 1.0
        ret = np.where(np.abs(ret) < RET_CLIP, ret, np.nan)
        st = stats(run(P, G, ret, dates))
        rows.append({"exec": f"TWAP {w} 分钟", "minutes": w, **st})
        print(f"TWAP {w:2d}min  ann={st['ann%']:6.1f}%  "
              f"sharpe={st['sharpe']:.2f}  mdd={st['mdd%']:6.1f}%  "
              f"calmar={st['calmar']:5.2f}", flush=True)

    pd.DataFrame(rows).to_csv(f"{OUT}/twap_decay.csv", index=False)
    json.dump(rows, open(f"{OUT}/twap_decay.json", "w"),
              ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
