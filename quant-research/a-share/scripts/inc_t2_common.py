"""INC_T2 公共件:日历行号、事件分类、匹配变量、聚类统计(前缀 inc_t2)。"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
from afactor.config import TMP, LEDGER, REPORTS

IDX = ["000300", "000905", "000852"]
IDX_RANK = {"000300": 0, "000905": 1, "000852": 2}          # 越小 = 越大盘
IDX_NAME = {"000300": "沪深300", "000905": "中证500", "000852": "中证1000"}
WINDOWS = {"dev": ("2021-01-01", "2023-12-31"), "test": ("2024-01-01", "2025-12-31"),
           "all": ("2021-01-01", "2025-12-31")}
PREHEAT = ("2020H1", "2020H2")
K_CTRL = 3
SEED = 20261011
T0_KINDS = ("C", "A", "E")


def calendar(dates: pd.DatetimeIndex) -> pd.DataFrame:
    """调样日历 + 面板行号。C=截止日,A=公告日(盘后),A1=首个可交易日,Em1=生效前一日,E=生效日。"""
    cal = pd.read_csv(LEDGER / "inc_t2_review_calendar.csv",
                      parse_dates=["C_cutoff", "A_official_publish", "A_plus1_first_tradable",
                                   "E_minus1_passive_close", "E_effective"])
    out = cal.rename(columns={"C_cutoff": "C", "A_official_publish": "A", "A_plus1_first_tradable": "A1",
                              "E_minus1_passive_close": "Em1", "E_effective": "E"})
    for k in ("C", "A", "A1", "Em1", "E"):
        out[k + "_i"] = dates.get_indexer(out[k])
        assert (out[k + "_i"] >= 0).all(), f"{k} 不在交易日历上"
    assert (out["A1_i"] == out["A_i"] + 1).all()
    assert (out["Em1_i"] == out["E_i"] - 1).all()
    return out


def window_of(review: str) -> str:
    if review in PREHEAT:
        return "preheat"
    y = int(review[:4])
    return "dev" if y <= 2023 else "test"


def cluster_mean_se(x: np.ndarray, g: np.ndarray):
    """均值与按 g 聚类的稳健标准误(CR1);x 一维。返回 (mean, se, n_clusters)。"""
    x = np.asarray(x, np.float64)
    ok = np.isfinite(x)
    x, g = x[ok], np.asarray(g)[ok]
    n = len(x)
    if n == 0:
        return np.nan, np.nan, 0
    mu = x.mean()
    e = x - mu
    S = pd.Series(e).groupby(g).sum().to_numpy()
    nG = len(S)
    if nG < 2:
        return mu, np.nan, nG
    se = np.sqrt((S ** 2).sum() * nG / (nG - 1)) / n
    return float(mu), float(se), nG


def period_t(x: np.ndarray, g: np.ndarray):
    """逐期均值的 t(n−1 自由度):返回 (期均值的均值, t, 期数, 正期数)。"""
    s = pd.Series(np.asarray(x, np.float64)).groupby(np.asarray(g)).mean().dropna()
    n = len(s)
    if n < 2:
        return (float(s.mean()) if n else np.nan), np.nan, n, int((s > 0).sum())
    sd = s.std(ddof=1)
    return float(s.mean()), (float(s.mean() / (sd / np.sqrt(n))) if sd > 0 else np.nan), n, int((s > 0).sum())


def summarize(x: np.ndarray, g: np.ndarray) -> dict:
    """事件级统计:均值、中位数、按期聚类 SE/t(主)、期均值 t(次)、i.i.d. SE(乐观下界)、正期数。"""
    x = np.asarray(x, np.float64)
    ok = np.isfinite(x)
    xo, go = x[ok], np.asarray(g)[ok]
    n = len(xo)
    if n == 0:
        return dict(n=0)
    mu, se, nG = cluster_mean_se(xo, go)
    pm, pt, pn, ppos = period_t(xo, go)
    iid = float(xo.std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return dict(n=n, mean=mu, median=float(np.median(xo)), se_cl=se, t_cl=(mu / se if se and se > 0 else np.nan),
                lo95=(mu - 1.96 * se) if np.isfinite(se) else np.nan, hi95=(mu + 1.96 * se) if np.isfinite(se) else np.nan,
                n_periods=pn, n_pos_periods=ppos, period_mean=pm, t_period=pt,
                se_iid=iid, t_iid=(mu / iid if iid and iid > 0 else np.nan))


def to_ts(code6: str) -> str:
    c = str(code6).zfill(6)
    return c + (".SH" if c[0] == "6" else ".SZ")
