"""Config D (raw prev_all factor, daily Top4) + the frozen regime gate.

Gate: frozen 2005-2019 state model's regime_score, known at t, applied to
the t+1 book — score < 0.30 sends the whole book to cash, same convention
as the old state030 candidate. Both components predate 2024.
"""

import json

import numpy as np
import pandas as pd

from sweep_dial import COST_SIDE, TOPK, load_env, run_topk, stats

UB = "/path/to/user/Desktop/ETF-Research/index-project-active-etf-upper-bound"
OUT = "/path/to/user/Desktop/ETF-Research/index project/research/showcase_v1/results"
GATE_TH = 0.30
MIN_TRAIN = 252


def main():
    man, dates, G, O2O = load_env()
    fl = np.load(f"{UB}/research/results/active_etf_upper_bound/"
                 "core_flow_main.npz", allow_pickle=True)
    j = list(fl["names"]).index("prev_all_valid21_liq5m")
    fdt = {d: i for i, d in enumerate(fl["index"].astype("datetime64[D]"))}
    fsy = {s: i for i, s in enumerate(fl["cols"])}
    P = fl["A"][j][np.array([fdt[d] for d in dates])][
        :, np.array([fsy[s] for s in man["symbols"]])]
    P[:MIN_TRAIN] = np.nan          # same burn-in as the rolling configs

    st = pd.read_parquet("/path/to/user/Desktop/ETF-Research/index project/research/"
                         "active_system_v1/regime_signal.parquet")
    score = st["regime_score"].reindex(pd.DatetimeIndex(dates)).ffill()
    gate_on = (score >= GATE_TH).values

    T, N = P.shape
    w_prev = np.zeros(N)
    rets = []
    for t in range(T - 2):
        if gate_on[t]:
            s = np.where(G[t], P[t], np.nan)
            k = np.where(np.isfinite(s))[0]
            if len(k) >= TOPK:
                top = k[np.argsort(s[k])][-TOPK:]
                w = np.zeros(N)
                w[top] = 1.0 / TOPK
            else:
                w = w_prev
        else:
            w = np.zeros(N)
        turn = np.abs(w - w_prev).sum()
        rets.append(float((w * np.nan_to_num(O2O[t + 1])).sum())
                    - turn * COST_SIDE)
        w_prev = w
    r = pd.Series(rets, index=pd.DatetimeIndex(dates[2:]))

    res = {}
    for wname, (lo, hi) in [("2020-2021", ("2020", "2021")),
                            ("2022-2023", ("2022", "2023")),
                            ("2024-2026", ("2024", "2027")),
                            ("full", ("2020", "2027"))]:
        rr = r.loc[lo:hi].dropna()
        res[wname] = stats(rr) if len(rr) > 60 else {}
    res["time_in_market"] = round(float(gate_on[MIN_TRAIN:].mean()), 3)
    print(json.dumps(res, ensure_ascii=False, indent=1))
    r.rename("ret").to_frame().to_parquet(f"{OUT}/rolling_Dgate.parquet")
    with open(f"{OUT}/rolling_Dgate.json", "w") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
