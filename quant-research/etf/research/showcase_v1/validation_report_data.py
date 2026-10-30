"""Data pack for the validation-line strategy report (汇报版).

The strategy: 662 features (312 engineered + 350 machine-mined, all selected
before 2024), one GBDT fitted once on pre-2024 data, then pure forward
execution over 2024-01-02 .. 2026-05-21. Three sizing configs share the same
frozen score panel:

  main       Top4  equal weight            (验证线·主)
  eq10       Top10 equal weight
  iv10       Top10 inverse-volatility

Per config: daily returns, additive per-ETF wealth contributions (reconciled
against final NAV), holding spells, and one-way turnover.
"""

import json

import numpy as np
import pandas as pd

from sweep_dial import CACHE, COST_SIDE, load_env, stats

OUT = "/path/to/user/Desktop/ETF-Research/index project/research/showcase_v1/results"
import sys as _sys
_sys.path.insert(0, "/path/to/user/Desktop/ETF-Research/index project/research/zoo_v1")
from harness import STORE as _STORE
MASTER = (f"{_STORE}/universe/"
          "classification_versions/classification-cb3a08d444b73aaa85ec727a/"
          "symbol_master.parquet")
DEV_END = 1213                       # 2024-01-02
DIAL = {"max_iter": 10, "min_samples_leaf": 3000}
CAP, MIN_W = 0.30, 0.03
BENCH = "510500.SH"


def frozen_scores(shape):
    from sklearn.ensemble import HistGradientBoostingRegressor
    X = np.asarray(np.load(f"{CACHE}/X2_pre2024.npy", mmap_mode="r"))
    y = np.load(f"{CACHE}/y_res.npy")
    day = np.load(f"{CACHE}/row_day.npy")
    sym = np.load(f"{CACHE}/row_sym.npy")
    tr, te = day < DEV_END, day >= DEV_END
    preds = np.zeros(int(te.sum()))
    for s in (0, 1):
        m = HistGradientBoostingRegressor(
            learning_rate=0.05, max_leaf_nodes=31, random_state=s,
            l2_regularization=1.0, **DIAL)
        m.fit(np.nan_to_num(X[tr], nan=0.5), y[tr])
        preds += m.predict(np.nan_to_num(X[te], nan=0.5))
    P = np.full(shape, np.nan, np.float32)
    P[day[te], sym[te]] = preds / 2
    return P


def weights_for(scores, elig, k, scheme, vol_row):
    s = np.where(elig, scores, np.nan)
    idx = np.where(np.isfinite(s))[0]
    if len(idx) < k:
        return None
    top = idx[np.argsort(s[idx])][-k:][::-1]
    if scheme == "equal":
        raw = np.ones(k)
    elif scheme == "rank":
        raw = np.arange(k, 0, -1, dtype=float)
    else:
        v = vol_row[top]
        v = np.where(np.isfinite(v) & (v > 1e-6), v, np.nanmedian(v))
        raw = 1.0 / v
    w = raw / raw.sum()
    for _ in range(3):
        w = np.minimum(w, CAP)
        w /= w.sum()
    keep = w >= MIN_W
    w = w[keep] / w[keep].sum()
    for _ in range(2):
        w = np.minimum(w, CAP)
        w /= w.sum()
    out = np.zeros(len(scores))
    out[top[keep]] = w
    return out


def run_config(P, G, O2O, dates, vol, k, scheme, t0):
    T, N = P.shape
    w_prev = np.zeros(N)
    nav = 1.0
    rets, turns = [], []
    contrib = np.zeros(N)
    cost_total = 0.0
    held_days = np.zeros(N, dtype=int)
    spells = np.zeros(N, dtype=int)
    in_spell = np.zeros(N, dtype=bool)
    W = np.zeros((T - 2 - t0, N), dtype=np.float32)
    for i, t in enumerate(range(t0, T - 2)):
        w = weights_for(P[t], G[t], k, scheme, vol[t])
        if w is None:
            w = w_prev
        held = w > 0
        held_days += held
        spells += held & ~in_spell
        in_spell = held
        r_t = np.nan_to_num(O2O[t + 1])
        pnl = w * r_t
        contrib += pnl * nav
        turn = np.abs(w - w_prev).sum()
        cost = turn * COST_SIDE
        cost_total += cost * nav
        day_ret = float(pnl.sum()) - cost
        rets.append(day_ret)
        turns.append(turn)
        nav *= 1 + day_ret
        W[i] = w
        w_prev = w
    r = pd.Series(rets, index=pd.DatetimeIndex(dates[t0 + 2:]))
    resid = (contrib.sum() - cost_total) - (nav - 1.0)
    assert abs(resid) < 1e-9, resid
    return {"r": r, "contrib": contrib, "cost_total": cost_total,
            "held_days": held_days, "spells": spells,
            "one_way_daily": float(np.mean(turns) / 2), "nav_end": nav}


def main():
    man, dates, G, O2O = load_env()
    syms = man["symbols"]
    shape = (len(dates), len(syms))
    P = frozen_scores(shape)
    O2Ofull = np.vstack([np.full((1, O2O.shape[1]), np.nan), O2O])
    vol = pd.DataFrame(O2Ofull).rolling(21, min_periods=10).std().values

    m = pd.read_parquet(MASTER).set_index("symbol")

    def label(s):
        nm = None
        if s in m.index:
            nm = m.at[s, "tracked_index_name"] or m.at[s, "name"]
        return f"{nm or s}  {s.split('.')[0]}"

    configs = {"main": (4, "equal"), "eq10": (10, "equal"),
               "iv10": (10, "invvol"), "rk10": (10, "rank")}
    pack = {"window": [str(dates[DEV_END]), str(dates[-1])],
            "configs": {}}
    navs = {}
    for key, (k, scheme) in configs.items():
        res = run_config(P, G, O2O, dates, vol, k, scheme, DEV_END)
        r = res["r"]
        st = stats(r)
        order = np.argsort(-np.abs(res["contrib"]))
        rows = []
        for j in order[:60]:
            if res["contrib"][j] == 0:
                continue
            rows.append({
                "symbol": syms[j], "label": label(syms[j]),
                "contrib_pp": round(float(res["contrib"][j] * 100), 2),
                "held_days": int(res["held_days"][j]),
                "spells": int(res["spells"][j]),
                "avg_spell": round(res["held_days"][j]
                                   / max(res["spells"][j], 1), 1)})
        pack["configs"][key] = {
            "k": k, "scheme": scheme, "stats": st,
            "one_way_daily_turnover": round(res["one_way_daily"], 3),
            "one_way_annual_turnover_x": round(
                res["one_way_daily"] * 252, 1),
            "cost_total_pp": round(float(res["cost_total"] * 100), 2),
            "n_names_traded": int((res["held_days"] > 0).sum()),
            "contrib": rows}
        navs[key] = (1 + r).cumprod()
        print(key, st, "| 单边日换手", round(res["one_way_daily"], 3),
              flush=True)

    j = syms.index(BENCH)
    rb = pd.Series(O2O[DEV_END + 1:, j],
                   index=pd.DatetimeIndex(dates[DEV_END + 2:])).fillna(0.0)
    navs["bench"] = (1 + rb).cumprod()
    pack["bench"] = {"symbol": BENCH, "stats": stats(rb)}

    pd.DataFrame(navs).to_parquet(f"{OUT}/validation_report_navs.parquet")

    def py(o):
        if isinstance(o, dict):
            return {k: py(v) for k, v in o.items()}
        if isinstance(o, list):
            return [py(v) for v in o]
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        return o

    with open(f"{OUT}/validation_report_pack.json", "w") as fh:
        json.dump(py(pack), fh, ensure_ascii=False, indent=1)
    print("saved")


if __name__ == "__main__":
    main()
