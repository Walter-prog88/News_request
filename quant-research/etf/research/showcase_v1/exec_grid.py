"""Execution grid: decouple the sell leg from the buy leg.

Diagnosis so far: the selected basket fades ~31bp in the first 15 minutes of
its ENTRY day (buying later is cheaper), while yesterday's basket peaks at
the open of its EXIT day (selling later gives the bounce back). TWAP on both
legs nets these against each other and loses. This grid prices each leg
separately.

Engine convention: basket chosen at t is bought on day t+1 at E, sold on day
t+2 at X. Requires exit-time <= entry-time within a session so the book is
never doubled; the gap is flat cash. Cost 1bp per side on |dw|; names kept
across days are treated as churned, consistent with the baseline engine.
"""

import json
import os

import numpy as np
import pandas as pd

from sweep_dial import CACHE, ZOO, load_env, stats

SRC = "/path/to/user/Desktop/ETF-Research/full-etf project2 "
OUT = "/path/to/user/Desktop/ETF-Research/index project/research/showcase_v1/results"
BARS_NPZ = f"{CACHE}/exec_bars.npz"
MULTI = f"{CACHE}/twap_windows.npz"
PANEL = f"{CACHE}/score_frozen_pre2024_panel.npy"
DEV_END, K, COST_SIDE = 1213, 10, 1.0e-4
BARS_PER_DAY = 241
RET_CLIP = 0.25
SINGLE_BARS = {"p1015": 45, "p1300": 121, "close15": 240}
LAST5 = slice(231, 236)              # 14:51..14:55


def build_bars(syms, zdates):
    if os.path.exists(BARS_NPZ):
        return dict(np.load(BARS_NPZ))
    print("loading close pickle (single pass) ...", flush=True)
    raw = pd.read_pickle(f"{SRC}/etf_pivot_close_qfq.pkl")
    cols = {c: i for i, c in enumerate(raw.columns)}
    vals = raw.values
    del raw
    n_days = len(zdates)
    out = {k: np.full((n_days, len(syms)), np.nan, np.float32)
           for k in list(SINGLE_BARS) + ["twap_last5"]}
    for j, s in enumerate(syms):
        if s not in cols:
            continue
        day = vals[:, cols[s]].reshape(n_days, BARS_PER_DAY)
        day = np.where(day > 0, day, np.nan)
        for k, b in SINGLE_BARS.items():
            out[k][:, j] = day[:, b]
        with np.errstate(invalid="ignore"):
            out["twap_last5"][:, j] = np.nanmean(day[:, LAST5], axis=1)
    np.savez(BARS_NPZ, **out)
    return out


def run(P, G, E, X, dates):
    """Buy at E[t+1], sell at X[t+2], for the basket chosen at t."""
    T, N = P.shape
    w_prev = np.zeros(N)
    rets = []
    for t in range(DEV_END, T - 2):
        with np.errstate(invalid="ignore"):
            hold_ret = X[t + 2] / E[t + 1] - 1.0
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
    zdates = pd.DatetimeIndex(meta["dates"])
    keep = zdates.get_indexer(pd.DatetimeIndex(dates))
    assert (keep >= 0).all()

    bars = {k: v[keep] for k, v in build_bars(syms, zdates).items()}
    z = np.load(MULTI)
    twap15 = z["w15"][keep]
    zi = {s: i for i, s in enumerate(meta["symbols"])}
    opz = np.load(f"{ZOO}/continuous_open.npy")
    open_p = opz[:, [zi[s] for s in syms]][keep]
    open_p = np.where(open_p > 0, open_p, np.nan)

    PRICES = {"open": open_p, "twap15": twap15, **bars}

    grid = [
        ("开盘/开盘(基线)", "open", "open"),
        ("双边 TWAP15", "twap15", "twap15"),
        ("卖@开盘 买@TWAP15", "twap15", "open"),
        ("卖@开盘 买@10:15", "p1015", "open"),
        ("卖@开盘 买@13:00", "p1300", "open"),
        ("双边 13:00", "p1300", "p1300"),
        ("双边 尾盘TWAP5", "twap_last5", "twap_last5"),
        ("卖@TWAP15 买@开盘(反向对照)", "open", "twap15"),
    ]
    rows = []
    for label, ek, xk in grid:
        r = run(P, G, PRICES[ek], PRICES[xk], dates)
        st = stats(r)
        rows.append({"exec": label, "entry": ek, "exit": xk, **st,
                     "final_nav": round(float((1 + r).cumprod().iloc[-1]), 2)})
        print(f"{label:24s} ann={st['ann%']:7.1f}%  sharpe={st['sharpe']:5.2f}"
              f"  mdd={st['mdd%']:6.1f}%  calmar={st['calmar']:6.2f}",
              flush=True)
        r.rename("ret").to_frame().to_parquet(
            f"{OUT}/exec_{ek}_{xk}.parquet")

    pd.DataFrame(rows).to_csv(f"{OUT}/exec_grid.csv", index=False)
    json.dump(rows, open(f"{OUT}/exec_grid.json", "w"),
              ensure_ascii=False, indent=1)

    # mechanism: the basket's fade on entry day vs exit day, and the
    # overnight/intraday split of the held return
    onr = np.load(f"{ZOO}/overnight_ret_continuous.npy")[
        :, [zi[s] for s in syms]][keep]
    ocr = np.load(f"{ZOO}/ret_oc_continuous.npy")[
        :, [zi[s] for s in syms]][keep]
    with np.errstate(invalid="ignore"):
        fade = twap15 / open_p - 1.0
    ge, gx, on_l, oc_l = [], [], [], []
    for t in range(DEV_END, len(dates) - 2):
        s = np.where(G[t], P[t], np.nan)
        idx = np.where(np.isfinite(s))[0]
        if len(idx) < K:
            continue
        top = idx[np.argsort(s[idx])][-K:]
        ge.append(np.nanmean(fade[t + 1, top]))
        gx.append(np.nanmean(fade[t + 2, top]))
        oc_l.append(np.nanmean(ocr[t + 1, top]))      # intraday of hold day
        on_l.append(np.nanmean(onr[t + 2, top]))      # overnight into exit
    mech = {
        "entry_day_fade_bp": round(float(np.nanmean(ge)) * 1e4, 1),
        "exit_day_fade_bp": round(float(np.nanmean(gx)) * 1e4, 1),
        "held_intraday_leg_bp_per_day": round(float(np.nanmean(oc_l)) * 1e4, 1),
        "held_overnight_leg_bp_per_day": round(float(np.nanmean(on_l)) * 1e4, 1),
    }
    print("\n机制分解:", json.dumps(mech, ensure_ascii=False))
    json.dump(mech, open(f"{OUT}/exec_mechanism.json", "w"),
              ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
