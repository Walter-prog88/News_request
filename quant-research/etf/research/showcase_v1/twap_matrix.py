"""All-TWAP execution matrix: no single-print fills anywhere.

Single-print fills (the 09:31 open, the 15:00 close) are the prices a real
order is least likely to get, so this prices every leg on a TWAP average
instead, across two entry regimes and two portfolio rules.

Prices, from the 241-bar minute grid (bar 0 = auction, 1..240 = 09:31..15:00):
  open_twap1    bar 1            09:30-09:31
  open_twap5    bars 1..5        09:30-09:35
  close_twap1   bar 240          14:59-15:00
  close_twap5   bars 236..240    14:55-15:00

Cycles:
  open cycle    signal t -> buy at t+1 open window -> sell at t+2 open window
  close cycle   signal t -> buy at t   close window -> sell at t+1 close window
                (factors would be computed on ~14:55 data; a tight upper
                bound, disclosed, not hidden)

Rules: plain Top-K, and the B=30 rank buffer that holds a name until it
falls out of the top 30.
"""

import json
import os

import numpy as np
import pandas as pd

from sweep_dial import CACHE, ZOO, load_env, stats

SRC = "/path/to/user/Desktop/ETF-Research/full-etf project2 "
OUT = "/path/to/user/Desktop/ETF-Research/index project/research/showcase_v1/results"
BARS = f"{CACHE}/twap_matrix_bars.npz"
PANEL = f"{CACHE}/score_frozen_pre2024_panel.npy"
DEV_END, K, COST_SIDE = 1213, 10, 1.0e-4
BARS_PER_DAY = 241
RET_CLIP = 0.25
BUFFER_B = 30

SPECS = {
    "open_twap1": slice(1, 2),        # 09:30-09:31
    "open_twap5": slice(1, 6),        # 09:30-09:35
    "close_twap1": slice(240, 241),   # 14:59-15:00
    "close_twap5": slice(236, 241),   # 14:55-15:00
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


def run(P, G, price, dates, entry_offset, buffer_b=None):
    """Hold price[t+off] -> price[t+off+1]; optional rank-buffer holding."""
    T, N = P.shape
    w_prev = np.zeros(N)
    held = np.array([], dtype=int)
    rets, turns = [], []
    for t in range(DEV_END, T - 2):
        e, x = t + entry_offset, t + entry_offset + 1
        with np.errstate(invalid="ignore"):
            hold = price[x] / price[e] - 1.0
        hold = np.where(np.abs(hold) < RET_CLIP, hold, np.nan)
        s = np.where(G[t] & np.isfinite(hold), P[t], np.nan)
        idx = np.where(np.isfinite(s))[0]
        if len(idx) >= K:
            order = idx[np.argsort(-s[idx])]
            if buffer_b is None:
                chosen = order[:K]
            else:
                rank = {c: r for r, c in enumerate(order)}
                keep = [h for h in held if rank.get(h, 10**9) < buffer_b]
                fill = [c for c in order if c not in keep]
                chosen = np.array((keep + fill)[:K], dtype=int)
            w = np.zeros(N)
            w[chosen] = 1.0 / len(chosen)
            held = chosen
        else:
            w = w_prev
        turn = np.abs(w - w_prev).sum()
        rets.append(float((w * np.nan_to_num(hold)).sum()) - turn * COST_SIDE)
        turns.append(turn)
        w_prev = w
    return (pd.Series(rets, index=pd.DatetimeIndex(dates[DEV_END + 2:])),
            float(np.mean(turns) / 2))


def main():
    man, dates, G, O2O = load_env()
    syms = man["symbols"]
    P = np.load(PANEL)
    meta = json.load(open(f"{ZOO}/meta.json"))
    zdates = pd.DatetimeIndex(meta["dates"])
    keep = zdates.get_indexer(pd.DatetimeIndex(dates))
    assert (keep >= 0).all()
    px = {k: v[keep] for k, v in build(syms, zdates).items()}
    zi = {s: i for i, s in enumerate(meta["symbols"])}
    open_pt = np.load(f"{ZOO}/continuous_open.npy")[
        :, [zi[s] for s in syms]][keep]
    px["open_print"] = np.where(open_pt > 0, open_pt, np.nan)

    grid = [
        ("参照 开盘单点价(集合竞价)", "open_print", 1),
        ("开盘TWAP 1分钟", "open_twap1", 1),
        ("开盘TWAP 5分钟", "open_twap5", 1),
        ("收盘TWAP 1分钟", "close_twap1", 0),
        ("收盘TWAP 5分钟", "close_twap5", 0),
    ]
    rows = []
    for label, pk, off in grid:
        for rule, b in (("基线", None), (f"缓冲带B{BUFFER_B}", BUFFER_B)):
            r, turn = run(P, G, px[pk], dates, off, buffer_b=b)
            st = stats(r)
            rows.append({"exec": label, "rule": rule, "price": pk,
                         "entry_offset": off, **st,
                         "one_way_turnover": round(turn, 3),
                         "final_nav": round(
                             float((1 + r).cumprod().iloc[-1]), 2)})
            print(f"{label:22s} {rule:10s} ann={st['ann%']:7.1f}%  "
                  f"sharpe={st['sharpe']:5.2f}  mdd={st['mdd%']:6.1f}%  "
                  f"calmar={st['calmar']:5.2f}  turn={turn:.3f}", flush=True)
            r.rename("ret").to_frame().to_parquet(
                f"{OUT}/tm_{pk}_{'base' if b is None else 'buf'}.parquet")

    pd.DataFrame(rows).to_csv(f"{OUT}/twap_matrix.csv", index=False)
    json.dump(rows, open(f"{OUT}/twap_matrix.json", "w"),
              ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
