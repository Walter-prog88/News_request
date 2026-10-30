"""Closing-TWAP decay, minute by minute, base rule vs the B=30 rank buffer.

The matrix showed the close window loses 17.5pp between a 1-minute and a
5-minute TWAP, far more than the open window loses over the same stretch.
This resolves that stretch one minute at a time so the shape of the decay is
visible rather than inferred from two endpoints.

Window W ends at 15:00 and starts W minutes earlier: bars 241-W .. 240.
"""

import json
import os

import numpy as np
import pandas as pd

from sweep_dial import CACHE, ZOO, load_env, stats
from twap_matrix import BUFFER_B, DEV_END, K, run

SRC = "/path/to/user/Desktop/ETF-Research/full-etf project2 "
OUT = "/path/to/user/Desktop/ETF-Research/index project/research/showcase_v1/results"
BARS = f"{CACHE}/close_twap_windows.npz"
PANEL = f"{CACHE}/score_frozen_pre2024_panel.npy"
BARS_PER_DAY = 241
WINDOWS = tuple(range(1, 11))


def build(syms, zdates):
    if os.path.exists(BARS):
        return dict(np.load(BARS))
    print("loading close pickle (single pass) ...", flush=True)
    raw = pd.read_pickle(f"{SRC}/etf_pivot_close_qfq.pkl")
    cols = {c: i for i, c in enumerate(raw.columns)}
    vals = raw.values
    del raw
    n = len(zdates)
    out = {f"w{w}": np.full((n, len(syms)), np.nan, np.float32)
           for w in WINDOWS}
    for j, s in enumerate(syms):
        if s not in cols:
            continue
        day = vals[:, cols[s]].reshape(n, BARS_PER_DAY)
        day = np.where(day > 0, day, np.nan)
        for w in WINDOWS:
            with np.errstate(invalid="ignore"):
                out[f"w{w}"][:, j] = np.nanmean(day[:, 241 - w:241], axis=1)
    np.savez(BARS, **out)
    return out


def main():
    man, dates, G, O2O = load_env()
    syms = man["symbols"]
    P = np.load(PANEL)
    meta = json.load(open(f"{ZOO}/meta.json"))
    zdates = pd.DatetimeIndex(meta["dates"])
    keep = zdates.get_indexer(pd.DatetimeIndex(dates))
    px = {k: v[keep] for k, v in build(syms, zdates).items()}

    rows = []
    for w in WINDOWS:
        start = f"{14 + (60 - w) // 60}:{(60 - w) % 60:02d}"
        for rule, b in (("基线", None), (f"缓冲带B{BUFFER_B}", BUFFER_B)):
            r, turn = run(P, G, px[f"w{w}"], dates, 0, buffer_b=b)
            st = stats(r)
            rows.append({"minutes": w, "window": f"{start}-15:00",
                         "rule": rule, **st,
                         "one_way_turnover": round(turn, 3)})
            print(f"收盘TWAP {w:2d}分钟 ({start}-15:00)  {rule:10s} "
                  f"ann={st['ann%']:7.1f}%  sharpe={st['sharpe']:5.2f}  "
                  f"mdd={st['mdd%']:6.1f}%  calmar={st['calmar']:5.2f}",
                  flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/close_twap_sweep.csv", index=False)
    json.dump(rows, open(f"{OUT}/close_twap_sweep.json", "w"),
              ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
