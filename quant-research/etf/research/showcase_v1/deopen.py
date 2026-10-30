"""Three answers to the open-auction dependency, all under the frozen protocol.

A. Label-matched retrain: same 662 features, same dial, but the label is
   TWAP15-to-TWAP15 instead of open-to-open. The model is asked "who is
   strong AFTER 09:45" — if the dependency is the training target's fault,
   this recovers TWAP-execution performance.
B. Rank buffer: keep a held name until it falls out of the top-B ranks.
   Cuts how much must be bought at any open at the price of staler holdings.
C. Five-day full-book rebalance: the blunt version of the same idea.

Everything trains on day < 2024-01-02 and walks forward after, as before.
"""

import json

import numpy as np
import pandas as pd

from sweep_dial import CACHE, ZOO, load_env, stats

OUT = "/path/to/user/Desktop/ETF-Research/index project/research/showcase_v1/results"
PANEL = f"{CACHE}/score_frozen_pre2024_panel.npy"
MULTI = f"{CACHE}/twap_windows.npz"
DEV_END, K, COST_SIDE = 1213, 10, 1.0e-4
DIAL = {"max_iter": 10, "min_samples_leaf": 3000}
RET_CLIP = 0.25
BUFFER_B = 30


def twap_grid(dates, syms):
    meta = json.load(open(f"{ZOO}/meta.json"))
    zdates = pd.DatetimeIndex(meta["dates"])
    keep = zdates.get_indexer(pd.DatetimeIndex(dates))
    assert (keep >= 0).all()
    return np.load(MULTI)["w15"][keep]


def retrain_twap_label(dates, syms):
    """Fit the same architecture on the TWAP15-to-TWAP15 residual label."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    tw = twap_grid(dates, syms)
    day = np.load(f"{CACHE}/row_day.npy")
    sym = np.load(f"{CACHE}/row_sym.npy")
    X = np.load(f"{CACHE}/X2_pre2024.npy", mmap_mode="r")

    with np.errstate(invalid="ignore"):
        fwd = tw[2:] / tw[1:-1] - 1.0            # entry t+1 -> exit t+2
    fwd = np.where(np.abs(fwd) < RET_CLIP, fwd, np.nan)
    y_raw = np.full(len(day), np.nan, dtype=np.float32)
    ok_day = day < len(fwd)
    y_raw[ok_day] = fwd[day[ok_day], sym[ok_day]]
    # residualise per day over the rows that have a label
    yr = pd.Series(y_raw)
    y_res = (yr - yr.groupby(pd.Series(day)).transform("mean")).values

    tr = (day < DEV_END) & np.isfinite(y_res)
    te = day >= DEV_END
    print(f"twap-label rows: train {tr.sum():,} / test {te.sum():,}",
          flush=True)
    preds = np.zeros(int(te.sum()))
    for s in (0, 1):
        m = HistGradientBoostingRegressor(
            learning_rate=0.05, max_leaf_nodes=31, random_state=s,
            l2_regularization=1.0, **DIAL)
        m.fit(np.nan_to_num(np.asarray(X[tr]), nan=0.5), y_res[tr])
        preds += m.predict(np.nan_to_num(np.asarray(X[te]), nan=0.5))
    P = np.full((len(dates), len(syms)), np.nan, np.float32)
    P[day[te], sym[te]] = preds / 2
    np.save(f"{CACHE}/score_frozen_twaplabel_panel.npy", P)
    return P, tw


def run_prices(P, G, E, X, dates, k=K, reb=1, buffer_b=None):
    """Generalised engine: entry price E, exit price X, optional rank buffer
    or n-day rebalance."""
    T, N = P.shape
    w_prev = np.zeros(N)
    held = np.array([], dtype=int)
    rets, turns = [], []
    for i, t in enumerate(range(DEV_END, T - 2)):
        with np.errstate(invalid="ignore"):
            hold_ret = X[t + 2] / E[t + 1] - 1.0
        hold_ret = np.where(np.abs(hold_ret) < RET_CLIP, hold_ret, np.nan)
        s = np.where(G[t] & np.isfinite(hold_ret), P[t], np.nan)
        idx = np.where(np.isfinite(s))[0]
        w = w_prev
        if len(idx) >= k and (i % reb == 0 or buffer_b is not None):
            order = idx[np.argsort(-s[idx])]
            if buffer_b is None:
                chosen = order[:k]
            else:
                rank_of = {sym_i: r for r, sym_i in enumerate(order)}
                keep_held = [h for h in held
                             if rank_of.get(h, 10**9) < buffer_b]
                fills = [c for c in order if c not in keep_held]
                chosen = np.array(
                    (keep_held + fills)[:k], dtype=int)
            w = np.zeros(N)
            w[chosen] = 1.0 / len(chosen)
            held = chosen
        turn = np.abs(w - w_prev).sum()
        rets.append(float((w * np.nan_to_num(hold_ret)).sum())
                    - turn * COST_SIDE)
        turns.append(turn)
        w_prev = w
    r = pd.Series(rets, index=pd.DatetimeIndex(dates[DEV_END + 2:]))
    return r, float(np.mean(turns) / 2)


def main():
    man, dates, G, O2O = load_env()
    syms = man["symbols"]
    P_open = np.load(PANEL)

    meta = json.load(open(f"{ZOO}/meta.json"))
    zi = {s: i for i, s in enumerate(meta["symbols"])}
    zdates = pd.DatetimeIndex(meta["dates"])
    keep = zdates.get_indexer(pd.DatetimeIndex(dates))
    open_p = np.load(f"{ZOO}/continuous_open.npy")[
        :, [zi[s] for s in syms]][keep]
    open_p = np.where(open_p > 0, open_p, np.nan)

    P_twap, tw = retrain_twap_label(dates, syms)

    rows = []

    def add(label, r, turn):
        st = stats(r)
        rows.append({"variant": label, **st,
                     "one_way_turnover": round(turn, 3)})
        print(f"{label:34s} ann={st['ann%']:7.1f}%  sharpe={st['sharpe']:5.2f}"
              f"  mdd={st['mdd%']:6.1f}%  calmar={st['calmar']:5.2f}"
              f"  turn={turn:.3f}", flush=True)

    # A. label-matched model at TWAP execution (its home turf)
    add("A 换标签模型 双边TWAP15",
        *run_prices(P_twap, G, tw, tw, dates))
    add("A 换标签模型 卖@开盘 买@TWAP15",
        *run_prices(P_twap, G, tw, open_p, dates))
    add("A 换标签模型 开盘/开盘(对照)",
        *run_prices(P_twap, G, open_p, open_p, dates))
    # original model reference rows
    add("原模型 双边TWAP15(参照)",
        *run_prices(P_open, G, tw, tw, dates))
    # B. rank buffer on the original model
    add(f"B 原模型+缓冲带B{BUFFER_B} 开盘/开盘",
        *run_prices(P_open, G, open_p, open_p, dates, buffer_b=BUFFER_B))
    add(f"B 原模型+缓冲带B{BUFFER_B} 双边TWAP15",
        *run_prices(P_open, G, tw, tw, dates, buffer_b=BUFFER_B))
    # C. 5-day rebalance
    add("C 原模型 5日调仓 开盘/开盘",
        *run_prices(P_open, G, open_p, open_p, dates, reb=5))
    add("C 原模型 5日调仓 双边TWAP15",
        *run_prices(P_open, G, tw, tw, dates, reb=5))

    pd.DataFrame(rows).to_csv(f"{OUT}/deopen_grid.csv", index=False)
    json.dump(rows, open(f"{OUT}/deopen_grid.json", "w"),
              ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
