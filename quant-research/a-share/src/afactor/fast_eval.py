"""高性能因子检验:一次行循环同时产出 Rank IC、分层收益、多头换手。

设计要点:前瞻收益与股票池只在 EvalContext 中准备一次,因子逐个流过。
"""
from __future__ import annotations
import json, time
import numpy as np
import pandas as pd
from scipy.stats import rankdata

from afactor.config import N_QUANTILES, IC_THRESHOLD, LEDGER
from afactor.panel import Panel

MIN_STOCKS = 200
TRADING_DAYS = 242
IC_DIR = LEDGER.parent / "ic_series"
IC_DIR.mkdir(parents=True, exist_ok=True)


class EvalContext:
    def __init__(self, panel: Panel, mode: str = "open", horizons=(1, 5, 10), tag: str = ""):
        # mode 决定股票池与成交口径;tag 只改变账本里的 mode 标签与 IC 文件名,
        # 用于同一口径下的变体运行(如 h5 主预测期),避免覆盖既有记录。
        self.panel, self.mode, self.horizons = panel, mode + (f"_{tag}" if tag else ""), tuple(horizons)
        self.dates = panel.dates
        self.codes = panel.codes
        univ = (panel["universe_tradable"] if mode == "open" else panel["universe_tail"])
        u = univ.to_numpy(dtype=bool)
        self.R, self.M = {}, {}
        for h in self.horizons:
            r = panel.fwd_ret(h, mode=mode).to_numpy(np.float32)
            self.R[h] = r
            self.M[h] = u & np.isfinite(r)
        self.years = np.asarray(self.dates.year)


def eval_one(F: np.ndarray, R: np.ndarray, M: np.ndarray,
             n_q: int = N_QUANTILES, want_layers: bool = True):
    """单因子单期限。F/R (T,N) float32,M (T,N) bool。"""
    T, N = F.shape
    ic = np.full(T, np.nan, np.float64)
    q_ret = np.full((T, n_q), np.nan, np.float64) if want_layers else None
    cover = np.zeros(T, np.int32)
    top_prev = None
    turn_num, turn_den = 0.0, 0
    bot_prev = None
    finF = np.isfinite(F)
    for t in range(T):
        m = M[t] & finF[t]
        n = int(m.sum())
        cover[t] = n
        if n < MIN_STOCKS:
            top_prev = bot_prev = None
            continue
        idx = np.flatnonzero(m)
        f = F[t, idx].astype(np.float64)
        r = R[t, idx].astype(np.float64)
        fr = rankdata(f)
        rr = rankdata(r)
        fm = fr - fr.mean(); rm = rr - rr.mean()
        den = np.sqrt((fm * fm).sum() * (rm * rm).sum())
        if den > 0:
            ic[t] = float((fm * rm).sum() / den)
        if want_layers:
            g = np.minimum((fr / n * n_q).astype(np.int64), n_q - 1)
            bench = r.mean()
            cnt = np.bincount(g, minlength=n_q).astype(np.float64)
            s = np.bincount(g, weights=r, minlength=n_q)
            with np.errstate(invalid="ignore", divide="ignore"):
                q_ret[t] = s / np.where(cnt > 0, cnt, np.nan) - bench
            top = idx[g == n_q - 1]
            bot = idx[g == 0]
            if top_prev is not None and len(top) > 0:
                turn_num += 1.0 - np.intersect1d(top, top_prev, assume_unique=True).size / len(top)
                turn_den += 1
            top_prev, bot_prev = top, bot
    turnover = turn_num / turn_den if turn_den else np.nan
    return ic, q_ret, cover, turnover


def _nw_t(v: np.ndarray, lag: int) -> float | None:
    """Newey-West t 值:h 日重叠前瞻收益使相邻 IC 自相关,普通 t 会虚高。"""
    n = v.size
    if n < 60:
        return None
    mu = v.mean(); d = v - mu
    var = float(d @ d) / n
    for l in range(1, min(lag, n - 1) + 1):
        w = 1.0 - l / (lag + 1.0)
        var += 2.0 * w * float(d[l:] @ d[:-l]) / n
    if var <= 0:
        return None
    return float(mu / np.sqrt(var / n))


def _ic_block(ic: np.ndarray, years: np.ndarray, nw_lag: int = 0) -> dict:
    v = ic[np.isfinite(ic)]
    if v.size < 60:
        return {"ic_mean": None, "n_days": int(v.size)}
    mu, sd = float(v.mean()), float(v.std(ddof=1))
    out = {"ic_mean": mu, "ic_std": sd,
           "ic_ir": mu / sd if sd > 0 else None,
           "ic_t": mu / sd * np.sqrt(v.size) if sd > 0 else None,
           "ic_pos": float((v > 0).mean()), "n_days": int(v.size)}
    if nw_lag > 0:
        out["ic_t_nw"] = _nw_t(v, nw_lag)
    by_year = {}
    for y in np.unique(years):
        s = ic[(years == y) & np.isfinite(ic)]
        if s.size > 40:
            by_year[int(y)] = float(s.mean())
    out["ic_by_year"] = by_year
    win = {}
    ys = sorted(by_year)
    for i in range(len(ys) - 2):
        a, b = ys[i], ys[i + 2]
        s = ic[(years >= a) & (years <= b) & np.isfinite(ic)]
        if s.size > 200:
            win[f"{a}-{b}"] = float(s.mean())
    out["ic_windows"] = win
    yv = list(by_year.values())
    out["year_sign_consistency"] = (
        float(max(np.mean([x > 0 for x in yv]), np.mean([x < 0 for x in yv]))) if yv else None)
    wv = list(win.values())
    out["window_sign_consistency"] = (
        float(max(np.mean([x > 0 for x in wv]), np.mean([x < 0 for x in wv]))) if wv else None)
    out["worst_window_abs_ic"] = float(min(abs(x) for x in wv)) if wv else None
    out["worst_year_abs_ic"] = float(min(abs(x) for x in yv)) if yv else None
    return out


def _layer_block(q_ret: np.ndarray, turnover: float, n_q: int = N_QUANTILES) -> dict:
    with np.errstate(invalid="ignore"):
        mean_exc = np.nanmean(q_ret, axis=0) * TRADING_DAYS
    ks = np.arange(1, n_q + 1)
    ok = np.isfinite(mean_exc)
    if ok.sum() < n_q:
        return {"mono_spearman": None}
    mono = float(pd.Series(mean_exc).corr(pd.Series(ks.astype(float)), method="spearman"))
    mono_mid = float(pd.Series(mean_exc[1:-1]).corr(pd.Series(ks[1:-1].astype(float)), method="spearman"))
    long_k = int(np.argmax(mean_exc))
    short_k = int(np.argmin(mean_exc))
    is_top_long = long_k >= n_q - 1 or short_k <= 0
    lr = q_ret[:, long_k]; lr = lr[np.isfinite(lr)]
    t_long = float(lr.mean() / lr.std(ddof=1) * np.sqrt(lr.size)) if lr.size > 60 and lr.std() > 0 else None
    denom = float(np.abs(mean_exc).sum()) or 1e-12
    return {
        "q_excess": [round(float(x), 5) for x in mean_exc],
        "mono_spearman": mono, "mono_mid_spearman": mono_mid,
        "long_group": long_k + 1, "short_group": short_k + 1,
        "extreme_long": bool(is_top_long),
        "long_excess_ann": float(mean_exc[long_k]), "short_excess_ann": float(mean_exc[short_k]),
        "long_t": t_long,
        "spread_ann": float(mean_exc[long_k] - mean_exc[short_k]),
        "long_turnover": None if turnover != turnover else float(turnover),
        "tail_concentration": float((abs(mean_exc[0]) + abs(mean_exc[-1])) / denom),
    }


def evaluate(name: str, F: np.ndarray, ctx: EvalContext, family: str = "",
             note: str = "", save_ic: bool = True, ledger_path=None) -> dict:
    t0 = time.time()
    res = {"factor": name, "family": family, "mode": ctx.mode, "note": note,
           "ts": time.strftime("%Y-%m-%d %H:%M:%S")}
    ic_store = {}
    for i, h in enumerate(ctx.horizons):
        ic, q_ret, cover, turn = eval_one(F, ctx.R[h], ctx.M[h], want_layers=(i == 0))
        ic_store[f"h{h}"] = ic
        blk = _ic_block(ic, ctx.years, nw_lag=max(5, 2 * h))   # 审查修正:h-1 带宽不足,t 虚高
        if i == 0:
            res.update(blk)
            res.update(_layer_block(q_ret, turn))
            res["coverage"] = float(cover[cover > 0].mean()) if (cover > 0).any() else 0.0
        else:
            res[f"h{h}_ic_mean"] = blk.get("ic_mean")
            res[f"h{h}_ic_ir"] = blk.get("ic_ir")
            res[f"h{h}_ic_t"] = blk.get("ic_t")
    res["elapsed"] = round(time.time() - t0, 2)
    m = res.get("ic_mean")
    res["pass_ic"] = bool(m is not None and abs(m) >= IC_THRESHOLD)
    if save_ic:
        pd.DataFrame(ic_store, index=ctx.dates).to_parquet(IC_DIR / f"{name}__{ctx.mode}.parquet")
    lp = ledger_path or (LEDGER / "trials.jsonl")
    with open(lp, "a") as fh:
        fh.write(json.dumps(res, ensure_ascii=False, default=float) + "\n")
    return res
