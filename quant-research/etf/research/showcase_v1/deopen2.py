"""Buy at the signal day's own tail — does it capture the overnight jump?

Variants, basket chosen at day t:
  close-cycle      buy at close(t), sell at close(t+1)   (tail auction both)
  tail-twap cycle  same with the 14:51-14:55 TWAP        (realistic fills)
  50/50 blend      half the book on the close-cycle, half on the open-cycle

Timing caveat, disclosed rather than hidden: factors use day t through the
close, so tail execution assumes computing them on ~14:50 data and trading
the last minutes. For daily-horizon factors the last ten minutes change
almost nothing; the close-cycle row is therefore a tight upper bound.

Also decomposes the entry path: overnight jump close(t)->open(t+1) of the
chosen basket versus its intraday drift, which says WHERE the money is.
"""

import json

import numpy as np
import pandas as pd

from sweep_dial import CACHE, ZOO, load_env, stats

OUT = "/path/to/user/Desktop/ETF-Research/index project/research/showcase_v1/results"
PANEL = f"{CACHE}/score_frozen_pre2024_panel.npy"
BARS_NPZ = f"{CACHE}/exec_bars.npz"
DEV_END, K, COST_SIDE = 1213, 10, 1.0e-4
RET_CLIP = 0.25


def run_cycle(P, G, price, dates, entry_offset):
    """Hold from price[t+entry_offset] to price[t+entry_offset+1]."""
    T, N = P.shape
    w_prev = np.zeros(N)
    rets = []
    for t in range(DEV_END, T - 2):
        e, x = t + entry_offset, t + entry_offset + 1
        with np.errstate(invalid="ignore"):
            hold_ret = price[x] / price[e] - 1.0
        hold_ret = np.where(np.abs(hold_ret) < RET_CLIP, hold_ret, np.nan)
        s = np.where(G[t] & np.isfinite(hold_ret), P[t], np.nan)
        idx = np.where(np.isfinite(s))[0]
        if len(idx) >= K:
            top = idx[np.argsort(s[idx])][-K:]
            w = np.zeros(N)
            w[top] = 1.0 / K
        else:
            w = w_prev
        rets.append(float((w * np.nan_to_num(hold_ret)).sum())
                    - np.abs(w - w_prev).sum() * COST_SIDE)
        w_prev = w
    return pd.Series(rets, index=pd.DatetimeIndex(dates[DEV_END + 2:]))


def main():
    man, dates, G, O2O = load_env()
    syms = man["symbols"]
    P = np.load(PANEL)
    meta = json.load(open(f"{ZOO}/meta.json"))
    zi = {s: i for i, s in enumerate(meta["symbols"])}
    zdates = pd.DatetimeIndex(meta["dates"])
    keep = zdates.get_indexer(pd.DatetimeIndex(dates))
    cols = [zi[s] for s in syms]

    bars = dict(np.load(BARS_NPZ))
    close_p = bars["close15"][keep]
    tail_p = bars["twap_last5"][keep]
    open_p = np.load(f"{ZOO}/continuous_open.npy")[:, cols][keep]
    open_p = np.where(open_p > 0, open_p, np.nan)

    rows = []

    def add(label, r):
        st = stats(r)
        rows.append({"variant": label, **st})
        print(f"{label:30s} ann={st['ann%']:7.1f}%  sharpe={st['sharpe']:5.2f}"
              f"  mdd={st['mdd%']:6.1f}%  calmar={st['calmar']:5.2f}",
              flush=True)
        return r

    r_open = add("开盘循环(基线)", run_cycle(P, G, open_p, dates, 1))
    r_close = add("尾盘循环 买卖@收盘价", run_cycle(P, G, close_p, dates, 0))
    add("尾盘循环 买卖@尾盘TWAP5", run_cycle(P, G, tail_p, dates, 0))
    add("50/50 混合(半尾盘半开盘)",
        0.5 * r_close + 0.5 * r_open)

    # where is the money on the entry path?
    onr = np.load(f"{ZOO}/overnight_ret_continuous.npy")[:, cols][keep]
    gap, first15 = [], []
    z = np.load(f"{CACHE}/twap_windows.npz")
    tw15 = z["w15"][keep]
    with np.errstate(invalid="ignore"):
        fade = tw15 / open_p - 1.0
    for t in range(DEV_END, len(dates) - 2):
        s = np.where(G[t], P[t], np.nan)
        idx = np.where(np.isfinite(s))[0]
        if len(idx) < K:
            continue
        top = idx[np.argsort(s[idx])][-K:]
        gap.append(np.nanmean(onr[t + 1, top]))       # close(t)->open(t+1)
        first15.append(np.nanmean(fade[t + 1, top]))
    mech = {
        "篮子隔夜跳空 close(t)->open(t+1) bp/日":
            round(float(np.nanmean(gap)) * 1e4, 1),
        "篮子开盘后15分钟 bp/日":
            round(float(np.nanmean(first15)) * 1e4, 1),
    }
    print("\n入场路径分解:", json.dumps(mech, ensure_ascii=False))
    pd.DataFrame(rows).to_csv(f"{OUT}/deopen2.csv", index=False)
    json.dump({"rows": rows, "mech": mech},
              open(f"{OUT}/deopen2.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
