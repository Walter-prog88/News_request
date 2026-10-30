"""Mechanical factor zoo: signal constructors and the frozen axis grids.

The grids are enumerated HERE, in code, before any results exist. Search
agents run axes; they do not add, remove, or reorder specs. That is what
keeps "we tried everything on the list" honest.

Every constructor is causal: value at row t uses data from rows <= t only.
Direction conventions: sign=+1 means "high value -> long". Where the prior
literature direction is negative (reversal, vol, tail) the sign is baked in.
"""

import numpy as np
import pandas as pd

import harness as H


def _roll(a: np.ndarray, w: int, fn: str) -> np.ndarray:
    df = pd.DataFrame(a)
    r = getattr(df.rolling(w, min_periods=max(5, w // 2)), fn)()
    return r.values


def make_signal(P: H.Panels, spec: dict) -> np.ndarray:
    k = spec["kind"]
    sgn = float(spec.get("sign", 1.0))
    w = spec.get("w")
    if k == "mom":                       # close_{t-skip} / close_{t-w} - 1
        c = P.col("close")
        cs = pd.DataFrame(c)
        skip = spec.get("skip", 0)
        sig = (cs.shift(skip) / cs.shift(w) - 1).values
    elif k == "rollmean":
        sig = _roll(P.col(spec["col"]), w, "mean")
    elif k == "rollsum":
        sig = _roll(P.col(spec["col"]), w, "sum")
    elif k == "rsj":                     # (up - down) semivariance ratio
        rv = P.col("rv_closepath_continuous")
        up = P.col("up_rv_continuous")
        dn = P.col("down_rv_continuous")
        daily = np.where(rv > 0, (up - dn) / rv, np.nan)
        sig = _roll(daily, w, "mean")
    elif k == "updown_ratio":
        up = _roll(P.col("up_rv_continuous"), w, "mean")
        dn = _roll(P.col("down_rv_continuous"), w, "mean")
        sig = np.where(dn > 0, up / dn, np.nan)
    elif k == "pk":                      # Parkinson range vol
        hi, lo = P.col("high_continuous"), P.col("low_continuous")
        daily = np.log(np.where((hi > 0) & (lo > 0), hi / lo, np.nan)) ** 2
        sig = _roll(daily, w, "mean")
    elif k == "range_pos":               # close inside w-day hi-lo range
        c = pd.DataFrame(P.col("close"))
        hi = pd.DataFrame(P.col("high_continuous")).rolling(w, min_periods=w // 2).max()
        lo = pd.DataFrame(P.col("low_continuous")).rolling(w, min_periods=w // 2).min()
        sig = ((c - lo) / (hi - lo).replace(0, np.nan)).values
    elif k == "maxret":                  # lottery tail
        sig = _roll(P.col("ret_cc"), w, "max")
    elif k == "on_minus_id":             # overnight vs intraday tilt
        on = _roll(P.col("overnight_ret_continuous"), w, "sum")
        idr = _roll(P.col("ret_oc_continuous"), w, "sum")
        sig = on - idr
    elif k == "flow":                    # existing minute-flow factor store
        sig = P.flow(spec["name"])
    elif k == "flow_xa":                 # round B: all-asset signed flow
        sig = P.flow_xa(w)
    else:
        raise ValueError(k)
    return sgn * sig


# ------------------------------------------------------------- frozen grids

def _cs(axis_specs, domains=("dom", "all"), holds=(1, 5, 21), quantile=5):
    out = []
    for base in axis_specs:
        for d in domains:
            for h in holds:
                out.append({**base, "domain": d, "hold": h, "q": quantile})
    return out


AXES = {
    "anchor": [  # must reproduce module_alpha.py's ~0.55 full-window Sharpe
        {"kind": "flow", "name": "prev_all_valid21", "domain": "dom",
         "hold": 5, "q": 5},
    ],
    "mom": _cs(
        [{"kind": "mom", "w": w} for w in (5, 10, 21, 63, 126, 252)]
        + [{"kind": "mom", "w": 63, "skip": 5},
           {"kind": "mom", "w": 252, "skip": 21}]
        + [{"kind": "mom", "w": w, "sign": -1} for w in (1, 3, 5)]),
    "vol": _cs(
        [{"kind": "rollmean", "col": "rv_closepath_continuous", "w": w,
          "sign": -1} for w in (5, 21, 63)]
        + [{"kind": "rsj", "w": w, "sign": -1} for w in (5, 21, 63)]
        + [{"kind": "pk", "w": 21, "sign": -1},
           {"kind": "updown_ratio", "w": 21}]),
    "onid": _cs(
        [{"kind": "rollsum", "col": "overnight_ret_continuous", "w": w,
          "sign": -1} for w in (21, 63)]
        + [{"kind": "rollsum", "col": "ret_oc_continuous", "w": w}
           for w in (21, 63)]
        + [{"kind": "on_minus_id", "w": 21, "sign": -1}]),
    "sess": _cs(
        [{"kind": "rollsum", "col": "open30_ret_continuous", "w": w}
         for w in (5, 21)]
        + [{"kind": "rollsum", "col": "last30_ret", "w": w, "sign": -1}
           for w in (5, 21)]),
    "micro": _cs(
        [{"kind": "rollmean", "col": c, "w": w, "sign": s}
         for c in ("flat_bar_ratio", "zr_share_continuous")
         for w in (21, 63) for s in (-1,)]
        + [{"kind": "rollmean", "col": "n_price_moves", "w": w}
           for w in (21, 63)]
        + [{"kind": "rollmean", "col": "path_abs_return_continuous", "w": w,
            "sign": -1} for w in (21, 63)]),
    "shape": _cs(
        [{"kind": "rollmean", "col": "high_time_ordinal_continuous", "w": w}
         for w in (21, 63)]
        + [{"kind": "range_pos", "w": 21},
           {"kind": "maxret", "w": 21, "sign": -1}]),
}

_FLOW_NAMES = None


def flow_names():
    global _FLOW_NAMES
    if _FLOW_NAMES is None:
        z = np.load(H.FLOW_NPZ, allow_pickle=True)
        _FLOW_NAMES = list(z["names"])
    return _FLOW_NAMES


def flow_axis(hold: int, quantile: int = 5):
    return [{"kind": "flow", "name": n, "domain": "dom", "hold": hold,
             "q": quantile} for n in flow_names()]


def flow_xa_axis():
    """Round B grid: cross-asset flow, liquidity gate v2, mechanical."""
    return [{"kind": "flow_xa", "w": w, "domain": d, "hold": h, "q": q,
             "gate": "v2"}
            for w in (5, 21) for d in ("dom", "all", "nondom")
            for h in (5, 10) for q in (5, 10)]


# --------------------------------------------------- sleeve/allocation layer

def sleeve_specs(P: H.Panels, sl: pd.DataFrame = None):
    """Yield (spec, weights DataFrame) for the allocation layer, mechanically."""
    if sl is None:
        sl = P.sleeves
    cols = list(sl.columns)
    eq = pd.DataFrame(0.0, index=sl.index, columns=cols)

    def trail(w):
        return sl.rolling(w, min_periods=max(20, w // 2)).sum().shift(0)

    # 1. TSMOM long-flat, sign and vol-scaled, ladder of windows
    for w in (21, 63, 126, 252):
        pos = (trail(w) > 0).astype(float)
        wt = pos.div(pos.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
        yield {"kind": "tsmom_sign", "w": w}, wt
        vol = sl.rolling(21, min_periods=10).std() * np.sqrt(252)
        iv = pos / vol.replace(0, np.nan)
        wt = iv.div(iv.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
        yield {"kind": "tsmom_vol", "w": w}, wt
    # 2. cross-sleeve rank momentum: top-2 of 7
    for w in (21, 63, 126):
        r = trail(w).rank(axis=1, ascending=False)
        pos = (r <= 2).astype(float)
        wt = pos.div(pos.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
        yield {"kind": "rankmom_top2", "w": w}, wt
    # 3. static risk parity across the four frozen classes (the fixed X-leg)
    vol = sl.rolling(63, min_periods=30).std() * np.sqrt(252)
    iv = (1.0 / vol.replace(0, np.nan)).fillna(0.0)
    wt = iv.div(iv.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0).shift(1).fillna(0.0)
    yield {"kind": "risk_parity_all"}, wt
    static = eq.copy()
    static[:] = 1.0 / len(cols)
    yield {"kind": "equal_all"}, static
    # 4. lead-lag gates: bond / gold trailing sign gates the domestic sleeve.
    # Only meaningful on the class-level panel; generic panels (e.g. index
    # groups) lack these columns and skip the block.
    dom = "domestic_equity_etf"
    if dom in cols:
        for gater in ("fixed_income_etf", "gold_spot_etf"):
            if gater not in cols:
                continue
            for direction in (1, -1):
                g = (direction * trail(21)[gater] > 0).astype(float)
                wt = eq.copy()
                wt[dom] = g
                yield {"kind": "leadlag", "gater": gater, "dir": direction}, wt
    # 5. calendar: day-of-week / month-turn on the domestic sleeve
    if dom in cols:
        dow = pd.Series(sl.index.dayofweek, index=sl.index)
        for d in range(5):
            wt = eq.copy()
            wt[dom] = (dow == d).astype(float)
            yield {"kind": "dow", "d": d}, wt
        mt = sl.index.to_series().groupby(sl.index.to_period("M"))
        rank_in = mt.cumcount()
        size = mt.transform("size")
        turn_mask = ((rank_in < 3) | (rank_in >= size - 3)).astype(float)
        wt = eq.copy()
        wt[dom] = turn_mask.values
        yield {"kind": "month_turn"}, wt
        # 6. vol targeting on the domestic sleeve (10% ann, leverage cap 1)
        vol_dom = sl[dom].rolling(21, min_periods=10).std() * np.sqrt(252)
        scale = (0.10 / vol_dom).clip(upper=1.0).fillna(0.0)
        wt = eq.copy()
        wt[dom] = scale
        yield {"kind": "vol_target_dom"}, wt
