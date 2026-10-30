"""Zoo v1 shared measurement harness.

Every module in the study is measured by THIS engine and nothing else.
Search agents may not implement their own backtests; that rule is what makes
the trial ledger a complete record and keeps look-ahead out of the loop.

Conventions (frozen, see README.md):
- signal(t) uses info known by t 15:00; position earns ret_cc(t+1);
  H-day holding = H overlapping tranches; returns stamped on the EARN date.
- cross-sectional modules are long-short only; 10bp round-trip per unit turnover.
- SEAL: sessions after 2026-05-21 are invisible unless ZOO_UNSEAL=1.
"""

import glob
import hashlib
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



def _find_store():
    """The universe store moved once already; try both known locations."""
    import os
    for p in ("/path/to/user/Desktop/ETF-Research/full-etf project2 /full-etf-universe-store",
              "/path/to/user/Desktop/ETF-Research/full-etf project2 /full-etf-universe-store"):
        if os.path.isdir(p):
            return p
    raise FileNotFoundError("full-etf-universe-store not found in known locations")

STORE = _find_store()
MASTER = (
    f"{STORE}/universe/classification_versions/"
    "classification-cb3a08d444b73aaa85ec727a/symbol_master.parquet"
)
FLOW_NPZ = (
    "/path/to/user/Desktop/ETF-Research/index-project-active-etf-upper-bound/research/results/"
    "active_etf_upper_bound/core_flow_main.npz"
)
ROOT = os.path.dirname(os.path.abspath(__file__))
CACHE = os.environ.get(
    "ZOO_CACHE",
    _SCRATCH + "/zoo_cache",
)
LEDGER = f"{ROOT}/trial_ledger.jsonl"
MODDIR = f"{ROOT}/module_returns"

SEAL = "2026-05-21"
COST_BP = 10.0
MIN_AMT = 5e6

PANEL_COLS = [
    "ret_cc", "close", "continuous_open", "overnight_ret_continuous",
    "ret_oc_continuous", "rv_closepath_continuous", "up_rv_continuous",
    "down_rv_continuous", "open30_ret_continuous", "last30_ret",
    "high_continuous", "low_continuous", "flat_bar_ratio",
    "zr_share_continuous", "n_price_moves", "high_time_ordinal_continuous",
    "path_abs_return_continuous",
]

SLEEVE_CLASSES = [
    "domestic_equity_etf", "fixed_income_etf", "gold_spot_etf",
    "commodity_futures_etf", "hong_kong_equity_etf",
    "other_cross_border_equity_etf", "mainland_hong_kong_equity_etf",
]

WINDOWS = {
    "y19_21": ("2019-01-01", "2021-12-31"),
    "y22_23": ("2022-01-01", "2023-12-31"),
    "y24_26": ("2024-01-01", SEAL),   # contaminated: reference only, never a gate
    "pre24": ("2019-01-01", "2023-12-31"),
    "full": ("2019-01-01", SEAL),
}


# ---------------------------------------------------------------- cache build

def build_cache() -> None:
    os.makedirs(CACHE, exist_ok=True)
    files = sorted(glob.glob(
        f"{STORE}/daily/versions/daily-base-v2-*/batch=daily_base/**/*.parquet",
        recursive=True))
    cols = ["symbol", "session_date", "daily_ohlc_valid"] + PANEL_COLS
    df = pd.concat([pd.read_parquet(f, columns=cols) for f in files],
                   ignore_index=True)
    df["session_date"] = pd.to_datetime(df["session_date"])
    df = df.drop_duplicates(["symbol", "session_date"], keep="last")

    m = pd.read_parquet(MASTER)
    m = m[(m["instrument_type"] == "ETF")
          & m["research_class"].isin(SLEEVE_CLASSES)]
    classes = dict(zip(m["symbol"], m["research_class"]))
    symbols = sorted(set(classes) & set(df["symbol"].unique()))
    df = df[df["symbol"].isin(symbols)]
    dates = pd.DatetimeIndex(sorted(df["session_date"].unique()))

    for c in ["daily_ohlc_valid"] + PANEL_COLS:
        piv = (df.pivot(index="session_date", columns="symbol", values=c)
               .reindex(index=dates, columns=symbols))
        arr = (piv.fillna(False).astype(bool).values if c == "daily_ohlc_valid"
               else piv.values.astype(np.float32))
        np.save(f"{CACHE}/{c}.npy", arr)

    # liquidity gate: prior-known 252d median amount >= 5m where amount exists
    # (domestic pool only until the minute aggregation lands); True elsewhere.
    z = np.load(FLOW_NPZ, allow_pickle=True)
    amt = pd.DataFrame(z["daily_amount"], index=pd.DatetimeIndex(z["index"]),
                       columns=z["cols"])
    med = amt.rolling(252, min_periods=60).median().shift(1)
    liq = (med >= MIN_AMT).reindex(index=dates, columns=symbols)
    covered = [s for s in symbols if s in amt.columns]
    liq = liq.ffill()
    liq[[s for s in symbols if s not in covered]] = True
    liq[covered] = liq[covered].fillna(False)
    np.save(f"{CACHE}/liq_gate.npy", liq.values.astype(bool))

    meta = {"dates": [d.strftime("%Y-%m-%d") for d in dates],
            "symbols": symbols, "classes": classes,
            "amount_covered": covered}
    with open(f"{CACHE}/meta.json", "w") as fh:
        json.dump(meta, fh)

    # sleeve returns: equal weight inside research_class, 60-session maturity
    valid = np.load(f"{CACHE}/daily_ohlc_valid.npy")
    ret = np.load(f"{CACHE}/ret_cc.npy")
    ret = np.where(valid & (np.abs(ret) < 0.20), ret, np.nan)
    retdf = pd.DataFrame(ret, index=dates, columns=symbols)
    sleeves = {}
    for cls in SLEEVE_CLASSES:
        syms = [s for s in symbols if classes[s] == cls]
        sub = retdf[syms]
        mature = sub.notna() & (sub.notna().cumsum() >= 60)
        sleeves[cls] = sub.where(mature).mean(axis=1)
    pd.DataFrame(sleeves).to_parquet(f"{CACHE}/sleeves.parquet")
    print(f"cache built: {len(dates)} sessions x {len(symbols)} symbols "
          f"({len(covered)} with amount)")


# ------------------------------------------------------------------- loading

class Panels:
    def __init__(self):
        with open(f"{CACHE}/meta.json") as fh:
            meta = json.load(fh)
        dates = pd.DatetimeIndex(pd.to_datetime(meta["dates"]))
        self.sealed = os.environ.get("ZOO_UNSEAL") != "1"
        self._n = int((dates <= SEAL).sum()) if self.sealed else len(dates)
        self.dates = dates[: self._n]
        self.symbols = meta["symbols"]
        self.classes = meta["classes"]
        self.valid = np.load(f"{CACHE}/daily_ohlc_valid.npy")[: self._n]
        self.liq = np.load(f"{CACHE}/liq_gate.npy")[: self._n]
        raw_ret = np.load(f"{CACHE}/ret_cc.npy")[: self._n]
        self.ret = np.where(self.valid & (np.abs(raw_ret) < 0.20), raw_ret, np.nan)
        self.sleeves = pd.read_parquet(f"{CACHE}/sleeves.parquet").iloc[: self._n]
        self._cols = {}
        self._flow = None

    def col(self, name: str) -> np.ndarray:
        if name not in self._cols:
            a = np.load(f"{CACHE}/{name}.npy")[: self._n]
            self._cols[name] = np.where(self.valid, a, np.nan)
        return self._cols[name]

    # ---- round B additions: all-asset amount / flow from the minute pivots

    def liq2(self) -> np.ndarray:
        """Liquidity gate covering ALL symbols (round B closes the v1 hole)."""
        path = f"{CACHE}/liq_gate_v2.npy"
        if not os.path.exists(path):
            amt = pd.read_parquet(f"{CACHE}/daily_amount_all.parquet")
            amt.index = pd.to_datetime(amt.index)
            med = amt.rolling(252, min_periods=60).median().shift(1)
            full_dates = pd.DatetimeIndex(
                json.load(open(f"{CACHE}/meta.json"))["dates"])
            g = ((med >= MIN_AMT).reindex(index=full_dates,
                                          columns=self.symbols)
                 .ffill().fillna(False))
            np.save(path, g.values.astype(bool))
        return np.load(path)[: self._n]

    def flow_xa(self, w: int) -> np.ndarray:
        """Cross-asset signed-flow imbalance, w-session rolling mean."""
        f = pd.read_parquet(f"{CACHE}/flow_ratio_all.parquet")
        f.index = pd.to_datetime(f.index)
        f = f.reindex(index=self.dates, columns=self.symbols)
        return f.rolling(w, min_periods=max(3, w // 2)).mean().shift(1).values

    def sleeves_v2(self) -> pd.DataFrame:
        """Sleeve returns rebuilt on the liquidity-gated universe."""
        path = f"{CACHE}/sleeves_v2.parquet"
        if not os.path.exists(path):
            g = self.liq2()
            ret = pd.DataFrame(np.where(g, self.ret, np.nan),
                               index=self.dates, columns=self.symbols)
            out = {}
            for cls in SLEEVE_CLASSES:
                syms = [s for s in self.symbols if self.classes[s] == cls]
                sub = ret[syms]
                mature = sub.notna() & (sub.notna().cumsum() >= 60)
                out[cls] = sub.where(mature).mean(axis=1)
            pd.DataFrame(out).to_parquet(path)
        df = pd.read_parquet(path)
        df.index = pd.to_datetime(df.index)
        return df.iloc[: self._n]

    def sleeves_ix(self) -> pd.DataFrame:
        """Index-group returns: one column per tracked index, gated members.

        The group key is the product's own tracked_index_id — point-in-time
        by construction, never derived from returns (the clustering-leak
        red line). Groups need >= 2y of aggregate history to enter.
        """
        path = f"{CACHE}/sleeves_ix.parquet"
        if not os.path.exists(path):
            m = pd.read_parquet(MASTER)
            m = m[m["research_class"].isin(SLEEVE_CLASSES)
                  & m["tracked_index_id"].notna()]
            g = self.liq2()
            ret = pd.DataFrame(np.where(g, self.ret, np.nan),
                               index=self.dates, columns=self.symbols)
            out = {}
            for ix, grp in m.groupby("tracked_index_id"):
                syms = [s for s in grp["symbol"] if s in ret.columns]
                if not syms:
                    continue
                sub = ret[syms]
                mature = sub.notna() & (sub.notna().cumsum() >= 60)
                r = sub.where(mature).mean(axis=1)
                if r.notna().sum() >= 504:
                    out[ix] = r
            pd.DataFrame(out).to_parquet(path)
        df = pd.read_parquet(path)
        df.index = pd.to_datetime(df.index)
        return df.iloc[: self._n]

    def flow(self, name: str) -> np.ndarray:
        if self._flow is None:
            z = np.load(FLOW_NPZ, allow_pickle=True)
            self._flow = (list(z["names"]),
                          pd.DatetimeIndex(z["index"]), list(z["cols"]), z["A"])
        names, fdates, fcols, A = self._flow
        df = pd.DataFrame(A[names.index(name)], index=fdates, columns=fcols)
        return df.reindex(index=self.dates, columns=self.symbols).values

    def domain_mask(self, domain: str) -> np.ndarray:
        if domain == "dom":
            keep = {s for s, c in self.classes.items()
                    if c == "domestic_equity_etf"}
        elif domain == "all":
            keep = set(self.classes)
        elif domain == "nondom":
            keep = {s for s, c in self.classes.items()
                    if c != "domestic_equity_etf"}
        else:
            raise ValueError(domain)
        return np.array([s in keep for s in self.symbols])


# -------------------------------------------------------------------- engine

def backtest_cs(P: Panels, sig: np.ndarray, hold: int, quantile: int,
                domain: str, gate: np.ndarray = None) -> pd.Series:
    """Long-short cross-sectional module. Returns net series on earn dates."""
    min_names = 50 if quantile >= 10 else 20
    if gate is None:
        gate = P.liq
    dmask = P.domain_mask(domain)
    T, N = sig.shape
    fwd = np.vstack([P.ret[1:], np.full((1, N), np.nan)])
    fwd = np.nan_to_num(fwd)

    out = np.full(T, np.nan)
    turn = np.zeros(T)
    held = [None] * hold
    for t in range(T):
        s = np.where(dmask & gate[t] & P.valid[t], sig[t], np.nan)
        ok = np.isfinite(s)
        w = None
        if ok.sum() >= min_names:
            idx = np.where(ok)[0]
            order = idx[np.argsort(s[idx])]
            q = max(1, len(order) // quantile)
            w = np.zeros(N)
            w[order[-q:]] = 0.5 / q
            w[order[:q]] = -0.5 / q
        slot = t % hold
        prev = held[slot]
        if w is not None:
            turn[t] = (np.abs(w - prev).sum() if prev is not None
                       else np.abs(w).sum()) / hold
            held[slot] = w
        book = [h for h in held if h is not None]
        if book:
            out[t] = float(np.vstack(book).mean(axis=0) @ fwd[t])
    net = pd.Series(out, index=P.dates) - turn * COST_BP / 1e4
    return net.shift(1)  # stamp on earn date


def backtest_sleeve(P: Panels, weights: pd.DataFrame,
                    sl: pd.DataFrame = None) -> pd.Series:
    """Allocation-layer module: weights over sleeve columns, decided at t,
    earn sleeve ret at t+1. Net of 10bp x turnover, stamped on earn date.

    sl is the panel the weights were built on and the panel that pays —
    round-B correction: earnings previously always came from the ungated
    P.sleeves, so gated variants gated only the signal, not the P&L.
    """
    if sl is None:
        sl = P.sleeves
    w = weights.reindex(index=P.dates, columns=sl.columns).fillna(0.0)
    fwd = sl.reindex(index=P.dates).shift(-1).fillna(0.0)
    gross = (w * fwd).sum(axis=1)
    turn = w.diff().abs().sum(axis=1).fillna(w.abs().sum(axis=1))
    return (gross - turn * COST_BP / 1e4).shift(1)


# --------------------------------------------------------------------- stats

def stats(net: pd.Series) -> dict:
    res = {}
    for wname, (lo, hi) in WINDOWS.items():
        r = net.loc[lo:hi].dropna()
        if len(r) < 60 or r.std() == 0:
            res[wname] = None
            continue
        nav = (1 + r).cumprod()
        res[wname] = {
            "sharpe": round(float(r.mean() / r.std() * np.sqrt(252)), 3),
            "ann": round(float(nav.iloc[-1] ** (252 / len(r)) - 1) * 100, 2),
            "mdd": round(float((nav / nav.cummax() - 1).min()) * 100, 2),
            "n": len(r),
        }
    clean = [res[w]["sharpe"] for w in ("y19_21", "y22_23") if res.get(w)]
    res["worst_clean"] = min(clean) if clean else None
    return res


# -------------------------------------------------------------------- ledger

def record(axis: str, spec: dict, st: dict, net: pd.Series) -> str:
    mid = axis + "_" + hashlib.sha1(
        json.dumps(spec, sort_keys=True).encode()).hexdigest()[:10]
    os.makedirs(MODDIR, exist_ok=True)
    net.rename("net").to_frame().to_parquet(f"{MODDIR}/{mid}.parquet")
    line = json.dumps({"id": mid, "axis": axis, "spec": spec,
                       "worst_clean": st.get("worst_clean"),
                       "full": (st.get("full") or {}).get("sharpe"),
                       "stats": st}, ensure_ascii=False)
    with open(LEDGER, "a") as fh:
        fh.write(line + "\n")
    return mid
