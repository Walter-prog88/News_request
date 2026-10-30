"""因子检验引擎:Rank IC、分年份/滚动窗口一致性、分层单调性、换手、试验账本。"""
from __future__ import annotations
import json, time
import numpy as np
import pandas as pd

from afactor.config import LEDGER, N_QUANTILES, IC_THRESHOLD
from afactor.panel import Panel

MIN_STOCKS = 200
TRADING_DAYS = 242
IC_DIR = LEDGER.parent / "ic_series"
IC_DIR.mkdir(parents=True, exist_ok=True)
TRIALS = LEDGER / "trials.jsonl"


# ---------------- 基础统计 ----------------
def rank_ic(f: pd.DataFrame, r: pd.DataFrame, univ: pd.DataFrame) -> pd.Series:
    """逐日 Spearman Rank IC。"""
    m = univ.reindex_like(f).fillna(False) & f.notna() & r.reindex_like(f).notna()
    fr = f.where(m).rank(axis=1)
    rr = r.reindex_like(f).where(m).rank(axis=1)
    fm = fr.sub(fr.mean(axis=1), axis=0)
    rm = rr.sub(rr.mean(axis=1), axis=0)
    num = (fm * rm).sum(axis=1, min_count=1)
    den = np.sqrt((fm ** 2).sum(axis=1) * (rm ** 2).sum(axis=1))
    ic = num / den.where(den > 0)
    return ic.where(m.sum(axis=1) >= MIN_STOCKS)


def ic_stats(ic: pd.Series) -> dict:
    ic = ic.dropna()
    if len(ic) < 60:
        return {"ic_mean": np.nan, "ic_ir": np.nan, "ic_t": np.nan, "ic_pos": np.nan, "n_days": len(ic)}
    mu, sd = ic.mean(), ic.std()
    return {"ic_mean": mu, "ic_std": sd,
            "ic_ir": mu / sd if sd > 0 else np.nan,
            "ic_t": mu / sd * np.sqrt(len(ic)) if sd > 0 else np.nan,
            "ic_pos": float((ic > 0).mean()), "n_days": int(len(ic))}


def ic_by_year(ic: pd.Series) -> dict:
    g = ic.dropna().groupby(ic.dropna().index.year).mean()
    return {int(k): float(v) for k, v in g.items()}


def ic_rolling_windows(ic: pd.Series, years: int = 3) -> dict:
    """不重叠+滚动三年窗口的 IC 均值。"""
    ic = ic.dropna()
    if ic.empty:
        return {}
    out = {}
    y0, y1 = ic.index.year.min(), ic.index.year.max()
    for s in range(y0, y1 - years + 2):
        e = s + years - 1
        sub = ic[(ic.index.year >= s) & (ic.index.year <= e)]
        if len(sub) > 200:
            out[f"{s}-{e}"] = float(sub.mean())
    return out


# ---------------- 分层 ----------------
def layered(f: pd.DataFrame, r: pd.DataFrame, univ: pd.DataFrame,
            n: int = N_QUANTILES) -> tuple[pd.DataFrame, dict]:
    m = univ.reindex_like(f).fillna(False) & f.notna() & r.reindex_like(f).notna()
    m = m & (m.sum(axis=1) >= MIN_STOCKS).values[:, None]
    fq = f.where(m).rank(axis=1, pct=True)
    grp = np.ceil(fq * n).clip(1, n)
    rr = r.reindex_like(f).where(m)
    bench = rr.mean(axis=1)
    cols = {}
    members = {}
    for k in range(1, n + 1):
        sel = grp == k
        cols[k] = rr.where(sel).mean(axis=1) - bench
        members[k] = sel
    grp_ret = pd.DataFrame(cols)

    mean_exc = grp_ret.mean() * TRADING_DAYS
    ks = np.arange(1, n + 1)
    mono = pd.Series(mean_exc.values, index=ks).corr(pd.Series(ks, index=ks), method="spearman")
    mid = pd.Series(mean_exc.values[1:-1], index=ks[1:-1])
    mono_mid = mid.corr(pd.Series(ks[1:-1], index=ks[1:-1]), method="spearman")

    ic_sign = 1 if mean_exc.iloc[-1] >= mean_exc.iloc[0] else -1
    long_k = n if ic_sign > 0 else 1
    short_k = 1 if ic_sign > 0 else n
    long_ret = grp_ret[long_k].dropna()
    t_long = long_ret.mean() / long_ret.std() * np.sqrt(len(long_ret)) if long_ret.std() > 0 else np.nan

    sel = members[long_k]
    prev = sel.shift(1).fillna(False)
    inter = (sel & prev).sum(axis=1)
    tot = sel.sum(axis=1).replace(0, np.nan)
    turnover = (1 - inter / tot).mean()

    stats = {
        "q_excess": {int(k): float(v) for k, v in mean_exc.items()},
        "mono_spearman": float(mono), "mono_mid_spearman": float(mono_mid),
        "long_group": int(long_k),
        "long_excess_ann": float(mean_exc[long_k]), "short_excess_ann": float(mean_exc[short_k]),
        "long_t": float(t_long),
        "spread_ann": float(mean_exc[long_k] - mean_exc[short_k]),
        "long_turnover": float(turnover),
        "tail_concentration": float(
            (abs(mean_exc[long_k]) + abs(mean_exc[short_k])) /
            max(abs(mean_exc).sum(), 1e-12)),
    }
    return grp_ret, stats


# ---------------- 主入口 ----------------
def evaluate_factor(name: str, factor: pd.DataFrame, panel: Panel,
                    mode: str = "open", horizons=(1, 5, 10),
                    family: str = "", note: str = "", save_ic: bool = True) -> dict:
    t0 = time.time()
    univ = panel["universe_tradable"] if mode == "open" else panel["universe_tail"]
    factor = factor.reindex(index=panel.dates, columns=panel.codes)

    res = {"factor": name, "family": family, "mode": mode, "note": note,
           "ts": time.strftime("%Y-%m-%d %H:%M:%S")}
    ic_frame = {}
    for h in horizons:
        r = panel.fwd_ret(h, mode=mode)
        ic = rank_ic(factor, r, univ)
        ic_frame[f"h{h}"] = ic
        pre = f"h{h}_"
        st = ic_stats(ic)
        res.update({pre + k: (float(v) if v == v else None) for k, v in st.items()})
        if h == horizons[0]:
            res["ic_by_year"] = ic_by_year(ic)
            res["ic_windows"] = ic_rolling_windows(ic)
            yr = list(res["ic_by_year"].values())
            res["year_sign_consistency"] = (
                float(max(np.mean([v > 0 for v in yr]), np.mean([v < 0 for v in yr]))) if yr else None)
            wv = list(res["ic_windows"].values())
            res["window_sign_consistency"] = (
                float(max(np.mean([v > 0 for v in wv]), np.mean([v < 0 for v in wv]))) if wv else None)
            res["worst_window_ic"] = (
                float(min(wv, key=abs) if wv and np.mean(wv) > 0 else (max(wv, key=abs) if wv else np.nan))
                if wv else None)
            _, ls = layered(factor, r, univ)
            res.update({k: v for k, v in ls.items() if k != "q_excess"})
            res["q_excess"] = ls["q_excess"]

    res["coverage"] = float((factor.notna() & univ).sum(axis=1).mean())
    res["elapsed"] = round(time.time() - t0, 2)
    ic1 = ic_frame[f"h{horizons[0]}"]
    res["pass_ic"] = bool(abs(res.get(f"h{horizons[0]}_ic_mean") or 0) >= IC_THRESHOLD)

    if save_ic:
        pd.DataFrame(ic_frame).to_parquet(IC_DIR / f"{name}__{mode}.parquet")
    with open(TRIALS, "a") as fh:
        fh.write(json.dumps(res, ensure_ascii=False, default=float) + "\n")
    return res


def load_trials() -> pd.DataFrame:
    if not TRIALS.exists():
        return pd.DataFrame()
    rows = [json.loads(l) for l in open(TRIALS) if l.strip()]
    return pd.DataFrame(rows)
