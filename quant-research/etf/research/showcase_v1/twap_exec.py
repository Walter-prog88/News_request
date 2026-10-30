"""Execute the frozen strategy at a 09:30-09:45 TWAP instead of the open print.

The reported curves fill at the 09:31 continuous open, which a real order of
any size cannot obtain. This replaces the fill price with the average close
of the first 15 continuous minutes — the price a simple 15-minute TWAP order
would realise — and reprices the whole validation window on that basis.

Minute grid: the pivots carry exactly 241 bars per session (ordinal 0 is the
opening auction, 1..240 are 09:31..15:00), so TWAP(09:30-09:45) is the mean
of bars 1..15.
"""

import json
import os

import numpy as np
import pandas as pd

from sweep_dial import CACHE, ZOO, load_env, stats

SRC = "/path/to/user/Desktop/ETF-Research/full-etf project2 "
OUT = "/path/to/user/Desktop/ETF-Research/index project/research/showcase_v1/results"
TWAP_PANEL = f"{CACHE}/twap_0930_0945.parquet"
PANEL = f"{CACHE}/score_frozen_pre2024_panel.npy"
DEV_END = 1213
BARS_PER_DAY = 241
TWAP_BARS = slice(1, 16)          # 09:31 .. 09:45 inclusive
K = 10
COST_SIDE = 1.0e-4
RET_CLIP = 0.25


def build_twap(syms):
    """Mean close over bars 1..15 of each session, for the showcase universe."""
    if os.path.exists(TWAP_PANEL):
        df = pd.read_parquet(TWAP_PANEL)
        df.index = pd.to_datetime(df.index)
        return df
    meta = json.load(open(f"{ZOO}/meta.json"))
    zdates = pd.DatetimeIndex(meta["dates"])
    zsyms = list(meta["symbols"])
    print("loading close pickle (6.8GB, one pass) ...", flush=True)
    raw = pd.read_pickle(f"{SRC}/etf_pivot_close_qfq.pkl")
    cols = list(raw.columns)
    n_rows = len(raw)
    assert n_rows == len(zdates) * BARS_PER_DAY, (n_rows, len(zdates))
    keep = [c for c in syms if c in set(cols)]
    pos = {c: i for i, c in enumerate(cols)}
    vals = raw.values
    del raw
    out = np.full((len(zdates), len(syms)), np.nan, dtype=np.float32)
    for j, s in enumerate(syms):
        if s not in pos:
            continue
        col = vals[:, pos[s]].reshape(len(zdates), BARS_PER_DAY)
        seg = col[:, TWAP_BARS]
        with np.errstate(invalid="ignore"):
            out[:, j] = np.nanmean(np.where(seg > 0, seg, np.nan), axis=1)
    df = pd.DataFrame(out, index=zdates, columns=syms)
    df.to_parquet(TWAP_PANEL)
    print(f"twap panel {df.shape}, coverage "
          f"{float(df.notna().mean().mean()):.3f}", flush=True)
    return df


def run(P, G, ret, dates, k, cost_side):
    T, N = P.shape
    w_prev = np.zeros(N)
    rets, turns = [], []
    for t in range(DEV_END, T - 2):
        s = np.where(G[t] & np.isfinite(ret[t + 1]), P[t], np.nan)
        idx = np.where(np.isfinite(s))[0]
        if len(idx) >= k:
            top = idx[np.argsort(s[idx])][-k:]
            w = np.zeros(N)
            w[top] = 1.0 / k
        else:
            w = w_prev
        turn = np.abs(w - w_prev).sum()
        rets.append(float((w * np.nan_to_num(ret[t + 1])).sum())
                    - turn * cost_side)
        turns.append(turn)
        w_prev = w
    return (pd.Series(rets, index=pd.DatetimeIndex(dates[DEV_END + 2:])),
            float(np.mean(turns) / 2))


def main():
    man, dates, G, O2O = load_env()
    syms = man["symbols"]
    P = np.load(PANEL)

    twap = build_twap(syms).reindex(index=pd.DatetimeIndex(dates))
    tw = twap.values
    with np.errstate(invalid="ignore"):
        T2T = tw[1:] / tw[:-1] - 1.0
    T2T = np.where(np.abs(T2T) < RET_CLIP, T2T, np.nan)

    out = {}
    for label, ret in (("开盘价成交", O2O), ("TWAP 09:30-09:45", T2T)):
        r, turn = run(P, G, ret, dates, K, COST_SIDE)
        st = stats(r)
        out[label] = {"stats": st, "one_way_turnover": round(turn, 3),
                      "final_nav": float((1 + r).cumprod().iloc[-1])}
        print(f"{label:18s} ann={st['ann%']:6.1f}%  sharpe={st['sharpe']:.2f}  "
              f"mdd={st['mdd%']:6.1f}%  calmar={st['calmar']:5.2f}  "
              f"turn={turn:.3f}", flush=True)
        r.rename("ret").to_frame().to_parquet(
            f"{OUT}/twap_{'open' if 'TWAP' not in label else 'twap'}.parquet")

    # how far the TWAP price sits from the open print, in bp
    op = np.load(f"{ZOO}/continuous_open.npy")
    meta = json.load(open(f"{ZOO}/meta.json"))
    zi = {s: i for i, s in enumerate(meta["symbols"])}
    opc = op[:len(dates)][:, [zi[s] for s in syms]]
    with np.errstate(invalid="ignore"):
        gap = (tw / np.where(opc > 0, opc, np.nan) - 1.0) * 1e4
    out["_twap_vs_open_bp"] = {
        "mean": round(float(np.nanmean(gap)), 2),
        "median": round(float(np.nanmedian(gap)), 2),
        "std": round(float(np.nanstd(gap)), 2)}
    print("TWAP 相对开盘价偏离(bp):", out["_twap_vs_open_bp"])

    out["_config"] = {"K": K, "cost_bp_per_side": COST_SIDE * 1e4,
                      "twap_bars": "1..15 (09:31-09:45)"}
    with open(f"{OUT}/twap_exec.json", "w") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
