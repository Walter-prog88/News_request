"""Trade the 14:45-14:55 window instead of the crowded closing auction.

Two timings, and the difference matters for what may be claimed:

  next-day tail   signal at t's close -> buy in t+1's window -> sell in t+2's.
                  The signal predates execution by ~24h, so nothing about it
                  needs to be computable at 14:45. Fully clean.
  same-day tail   signal at t -> buy in t's own window. This is the variant
                  that would require all 662 factors by 14:45, and 210 of
                  them are daily factors built on the session's close. Run as
                  a labelled upper bound, not as a deliverable.

Windows compared: 14:45-14:55 (ten bars, away from the closing crush), the
last five minutes, and the last minute.

Bar map: bar 1 = 09:31 ... bar 120 = 11:30, bar 121 = 13:01 ... bar 240 =
15:00, so 14:45-14:55 is bars 226..235.
"""

import json
import os

import numpy as np
import pandas as pd

from sweep_dial import CACHE, ZOO, load_env, stats
from twap_matrix import BUFFER_B, DEV_END, K, run

SRC = "/path/to/user/Desktop/ETF-Research/full-etf project2 "
OUT = "/path/to/user/Desktop/ETF-Research/index project/research/showcase_v1/results"
BARS = f"{CACHE}/tail_windows.npz"
PANEL = f"{CACHE}/score_frozen_pre2024_panel.npy"
BARS_PER_DAY = 241

SPECS = {
    "t1445_1455": slice(226, 236),   # 14:45-14:55, ten bars
    "t1450_1500": slice(231, 241),   # 14:50-15:00, for reference
    "last5": slice(236, 241),        # 14:55-15:00
    "last1": slice(240, 241),        # 14:59-15:00
}


def build(syms, zdates):
    if os.path.exists(BARS):
        return dict(np.load(BARS))
    print("loading close pickle (single pass) ...", flush=True)
    raw = pd.read_pickle(f"{SRC}/etf_pivot_close_qfq.pkl")
    cols = {c: i for i, c in enumerate(raw.columns)}
    vals = raw.values
    del raw
    n = len(zdates)
    out = {k: np.full((n, len(syms)), np.nan, np.float32) for k in SPECS}
    for j, s in enumerate(syms):
        if s not in cols:
            continue
        day = vals[:, cols[s]].reshape(n, BARS_PER_DAY)
        day = np.where(day > 0, day, np.nan)
        for k, sl in SPECS.items():
            with np.errstate(invalid="ignore"):
                out[k][:, j] = np.nanmean(day[:, sl], axis=1)
    np.savez(BARS, **out)
    return out


def main():
    man, dates, G, O2O = load_env()
    syms = man["symbols"]
    P = np.load(PANEL)
    meta = json.load(open(f"{ZOO}/meta.json"))
    zdates = pd.DatetimeIndex(meta["dates"])
    keep = zdates.get_indexer(pd.DatetimeIndex(dates))
    assert (keep >= 0).all()
    px = {k: v[keep] for k, v in build(syms, zdates).items()}

    labels = {"t1445_1455": "14:45-14:55 TWAP",
              "t1450_1500": "14:50-15:00 TWAP",
              "last5": "14:55-15:00 TWAP",
              "last1": "14:59-15:00 TWAP"}
    timings = [("次日尾盘(信号早24h,干净)", 1),
               ("同日尾盘(需14:45算出因子,上界)", 0)]

    rows = []
    for tname, off in timings:
        print(f"\n=== {tname} ===", flush=True)
        for pk, plabel in labels.items():
            for rule, b in (("基线", None), (f"缓冲带B{BUFFER_B}", BUFFER_B)):
                r, turn = run(P, G, px[pk], dates, off, buffer_b=b)
                st = stats(r)
                rows.append({"timing": tname, "entry_offset": off,
                             "window": plabel, "rule": rule, **st,
                             "one_way_turnover": round(turn, 3)})
                print(f"  {plabel:18s} {rule:10s} ann={st['ann%']:7.1f}%  "
                      f"sharpe={st['sharpe']:5.2f}  mdd={st['mdd%']:6.1f}%  "
                      f"calmar={st['calmar']:5.2f}", flush=True)

    pd.DataFrame(rows).to_csv(f"{OUT}/tail_1445.csv", index=False)
    json.dump(rows, open(f"{OUT}/tail_1445.json", "w"),
              ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
