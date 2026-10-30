"""Daily Top-4 engine for the showcase stack.

Convention: score computed from day-t information, orders filled at t+1
continuous open, marked open-to-open, 0.5bp per executed side. T+1 is
satisfied by construction (positions are held at least one night); the 30%
single-name cap never binds at 25% targets; minute-capacity is reported as
the share of fills whose notional stays under 5% of that day's amount.

Backtests every score panel found in the cache plus their z-score stack,
writes NAV series, window metrics, decile layers, and the ritual audit.
"""

import glob
import json
import os

import numpy as np
import pandas as pd
import os as _os
# Default to a persistent root: the old fallback pointed at the research
# session's scratch dir, which the OS purges, taking every cached panel
# with it. rebuild_caches.sh regenerates this tree bit-identically.
_SCRATCH = _os.environ.get(
    "INDEX_CACHE_ROOT", _os.path.expanduser("~/index-cache"))

SCRATCH = _SCRATCH
CACHE = f"{SCRATCH}/showcase_cache"
ZOO = f"{SCRATCH}/zoo_cache"
UB = "/path/to/user/Desktop/ETF-Research/index-project-active-etf-upper-bound"
OUT = "/path/to/user/Desktop/ETF-Research/index project/research/showcase_v1/results"

TOPK = 4
COST_SIDE = 0.5e-4
CAPITAL = 1_000_000.0
WINDOWS = {"2019-2021": ("2019", "2021"), "2022-2023": ("2022", "2023"),
           "2024-2026": ("2024", "2027"), "full": ("2019", "2027")}
OLD = {"ann": 64.35, "sharpe": 2.29, "calmar": 3.53, "mdd": -18.2}


def metrics(r: pd.Series) -> dict:
    r = r.dropna()
    if len(r) < 60:
        return {}
    nav = (1 + r).cumprod()
    ann = nav.iloc[-1] ** (252 / len(r)) - 1
    mdd = float((nav / nav.cummax() - 1).min())
    sh = float(r.mean() / r.std() * np.sqrt(252))
    return {"ann%": round(100 * ann, 2), "sharpe": round(sh, 2),
            "mdd%": round(100 * mdd, 2),
            "calmar": round(ann / abs(mdd), 2) if mdd else None,
            "vol%": round(100 * r.std() * np.sqrt(252), 2), "n": len(r)}


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    man = json.load(open(f"{CACHE}/manifest.json"))
    dates = np.array(man["dates"], dtype="datetime64[D]")
    syms = man["symbols"]
    T, N = len(dates), len(syms)

    meta = json.load(open(f"{ZOO}/meta.json"))
    zdates = np.array(meta["dates"], dtype="datetime64[D]")
    zt = {d: i for i, d in enumerate(zdates)}
    zi = {s: i for i, s in enumerate(meta["symbols"])}
    zsym = np.array([zi[s] for s in syms])
    n_z = len(zdates)
    op = np.load(f"{ZOO}/continuous_open.npy")[:n_z][:, zsym]
    valid = np.load(f"{ZOO}/daily_ohlc_valid.npy")[:n_z][:, zsym].astype(bool)
    gate = np.load(f"{ZOO}/liq_gate_v2.npy")[:n_z][:, zsym].astype(bool)
    op = np.where(valid & (op > 0), op, np.nan)
    zrow = np.array([zt[d] for d in dates])
    O = op[zrow]                      # (T, N) opens on matrix dates
    G = (gate & valid)[zrow]
    with np.errstate(invalid="ignore"):
        O2O = O[1:] / O[:-1] - 1.0    # row t: open(t) -> open(t+1)
    O2O = np.where(np.abs(O2O) < 0.25, O2O, np.nan)

    fl = np.load(f"{UB}/research/results/active_etf_upper_bound/"
                 "core_flow_main.npz", allow_pickle=True)
    fdt = {d: i for i, d in enumerate(fl["index"].astype("datetime64[D]"))}
    fsy = {s: i for i, s in enumerate(fl["cols"])}
    amt = fl["daily_amount"][
        np.array([fdt[d] for d in dates])][:, [fsy[s] for s in syms]]

    panels = {os.path.basename(p)[6:-4]: np.load(p)
              for p in sorted(glob.glob(f"{CACHE}/score_*.npy"))}
    zs = []
    for name, P in panels.items():
        mu = np.nanmean(P, axis=1, keepdims=True)
        sd = np.nanstd(P, axis=1, keepdims=True)
        zs.append((P - mu) / np.where(sd > 0, sd, np.nan))
    if len(zs) > 1:
        panels["stack"] = np.nanmean(np.stack(zs), axis=0)

    report = {}
    for name, P in panels.items():
        w_prev = np.zeros(N)
        rets, tos, cap_ok, cap_all = [], [], 0, 0
        weights = np.zeros((T - 2, N), dtype=np.float32)
        for t in range(T - 2):
            s = np.where(G[t], P[t], np.nan)
            k = np.where(np.isfinite(s))[0]
            if len(k) >= TOPK:
                top = k[np.argsort(s[k])][-TOPK:]
                w = np.zeros(N)
                w[top] = 1.0 / TOPK
            else:
                w = w_prev
            fill_ret = np.nan_to_num(O2O[t + 1])
            turn = np.abs(w - w_prev).sum()
            rets.append(float((w * fill_ret).sum()) - turn * COST_SIDE)
            tos.append(turn)
            chg = np.where(np.abs(w - w_prev) > 1e-9)[0]
            day_amt = amt[t + 1, chg]
            cap_all += len(chg)
            cap_ok += int((CAPITAL * np.abs(w - w_prev)[chg]
                           <= 0.05 * np.where(day_amt > 0, day_amt, np.inf)
                           ).sum())
            weights[t] = w
            w_prev = w
        r = pd.Series(rets, index=pd.DatetimeIndex(dates[2:]))
        res = {w_: metrics(r.loc[lo:hi]) for w_, (lo, hi) in WINDOWS.items()}
        res["avg_daily_turnover"] = round(float(np.mean(tos)), 3)
        res["capacity_pass_rate"] = round(cap_ok / max(cap_all, 1), 4)
        report[name] = res
        r.rename("ret").to_frame().to_parquet(f"{OUT}/nav_{name}.parquet")
        print(name, "full:", res["full"], "| to:",
              res["avg_daily_turnover"], "| cap:",
              res["capacity_pass_rate"], flush=True)

    # decile layers on the stack (or single panel) for the report chart
    P = panels.get("stack", list(panels.values())[0])
    dec = np.full((T - 2, 10), np.nan, dtype=np.float32)
    for t in range(T - 2):
        s = np.where(G[t], P[t], np.nan)
        k = np.where(np.isfinite(s))[0]
        if len(k) < 50:
            continue
        q = np.quantile(s[k], np.linspace(0, 1, 11))
        fill_ret = O2O[t + 1]
        for d in range(10):
            m = k[(s[k] >= q[d]) & (s[k] <= q[d + 1])]
            if len(m):
                dec[t, d] = np.nanmean(fill_ret[m])
    pd.DataFrame(dec, index=pd.DatetimeIndex(dates[2:]),
                 columns=[f"D{i+1}" for i in range(10)]
                 ).to_parquet(f"{OUT}/deciles.parquet")

    report["_old_candidate_2024_2026"] = OLD
    with open(f"{OUT}/metrics.json", "w") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=1)
    print("saved", OUT)


if __name__ == "__main__":
    main()
