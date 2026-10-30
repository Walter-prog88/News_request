"""因子算子库:截面算子 + 时序算子 + 组合算子。
所有算子输入输出均为 DataFrame [date x ts_code],NaN 表示不可用。
"""
from __future__ import annotations
import numpy as np
import pandas as pd

EPS = 1e-12


# ---------------- 截面算子 ----------------
def cs_rank(x: pd.DataFrame) -> pd.DataFrame:
    """截面百分位排名,中心化到 [-0.5, 0.5]。"""
    return x.rank(axis=1, pct=True) - 0.5


def cs_zscore(x: pd.DataFrame) -> pd.DataFrame:
    m = x.mean(axis=1); s = x.std(axis=1)
    return x.sub(m, axis=0).div(s.replace(0, np.nan), axis=0)


def cs_winsor(x: pd.DataFrame, k: float = 5.0) -> pd.DataFrame:
    """MAD 去极值。"""
    med = x.median(axis=1)
    mad = (x.sub(med, axis=0)).abs().median(axis=1) * 1.4826
    lo = med - k * mad; hi = med + k * mad
    return x.clip(lower=lo, upper=hi, axis=0)


def cs_demean(x: pd.DataFrame) -> pd.DataFrame:
    return x.sub(x.mean(axis=1), axis=0)


def cs_neutralize(x: pd.DataFrame, *factors: pd.DataFrame) -> pd.DataFrame:
    """截面回归 x ~ 1 + factors,返回残差(逐日 OLS,向量化)。"""
    X = x.to_numpy(np.float64)
    Fs = [f.reindex_like(x).to_numpy(np.float64) for f in factors]
    out = np.full_like(X, np.nan)
    for i in range(X.shape[0]):
        y = X[i]
        cols = [np.ones_like(y)] + [f[i] for f in Fs]
        A = np.column_stack(cols)
        m = np.isfinite(y) & np.isfinite(A).all(axis=1)
        if m.sum() < 30:
            continue
        try:
            beta, *_ = np.linalg.lstsq(A[m], y[m], rcond=None)
        except np.linalg.LinAlgError:
            continue
        out[i, m] = y[m] - A[m] @ beta
    return pd.DataFrame(out, index=x.index, columns=x.columns)


def cs_neutralize_1d(x: pd.DataFrame, s: pd.DataFrame) -> pd.DataFrame:
    """对单个变量的截面正交化(向量化,等价于逐日 OLS 残差,比 lstsq 循环快百倍)。"""
    s = s.reindex_like(x)
    m = x.notna() & s.notna()
    xv = x.where(m); sv = s.where(m)
    xb = xv.mean(axis=1); sb = sv.mean(axis=1)
    xd = xv.sub(xb, axis=0); sd = sv.sub(sb, axis=0)
    num = (xd * sd).sum(axis=1, min_count=2)
    den = (sd ** 2).sum(axis=1, min_count=2)
    beta = num / den.where(den.abs() > EPS)
    return xd.sub(sd.mul(beta, axis=0))


def cs_group_demean(x: pd.DataFrame, g: pd.DataFrame, n_groups: int = 10) -> pd.DataFrame:
    """按 g 的截面分组做组内去均值(无行业数据时的替代分组中性化)。"""
    grp = g.rank(axis=1, pct=True).mul(n_groups).apply(np.ceil).clip(1, n_groups)
    out = x.copy()
    for k in range(1, n_groups + 1):
        m = grp == k
        sub = x.where(m)
        out = out.mask(m, sub.sub(sub.mean(axis=1), axis=0))
    return out.where(x.notna())


# ---------------- 时序算子 ----------------
def ts_mean(x, w): return x.rolling(w, min_periods=max(2, w // 2)).mean()
def ts_sum(x, w): return x.rolling(w, min_periods=max(2, w // 2)).sum()
def ts_std(x, w): return x.rolling(w, min_periods=max(3, w // 2)).std()
def ts_max(x, w): return x.rolling(w, min_periods=max(2, w // 2)).max()
def ts_min(x, w): return x.rolling(w, min_periods=max(2, w // 2)).min()
def ts_med(x, w): return x.rolling(w, min_periods=max(2, w // 2)).median()
def ts_skew(x, w): return x.rolling(w, min_periods=max(5, w // 2)).skew()
def ts_kurt(x, w): return x.rolling(w, min_periods=max(6, w // 2)).kurt()
def ts_delay(x, w): return x.shift(w)
def ts_delta(x, w): return x - x.shift(w)
def ts_ewm(x, w): return x.ewm(span=w, min_periods=max(2, w // 2)).mean()


def ts_pct(x, w):
    prev = x.shift(w)
    return x / prev.where(prev.abs() > EPS) - 1.0


def ts_zscore(x, w):
    m = ts_mean(x, w); s = ts_std(x, w)
    return (x - m) / s.where(s.abs() > EPS)


def ts_rank(x, w):
    """滚动窗口内当前值的百分位 (0~1)。"""
    return x.rolling(w, min_periods=max(3, w // 2)).rank(pct=True)


def ts_corr(x, y, w):
    return x.rolling(w, min_periods=max(5, w // 2)).corr(y)


def ts_cov(x, y, w):
    return x.rolling(w, min_periods=max(5, w // 2)).cov(y)


def ts_decay(x, w):
    """线性衰减加权平均(近期权重大)。"""
    wt = np.arange(1, w + 1, dtype=np.float64); wt /= wt.sum()
    return x.rolling(w, min_periods=max(2, w // 2)).apply(
        lambda a: np.dot(a, wt) if len(a) == w else np.nan, raw=True)


def ts_argmax(x, w):
    return x.rolling(w, min_periods=max(2, w // 2)).apply(np.argmax, raw=True) / max(w - 1, 1)


def ts_argmin(x, w):
    return x.rolling(w, min_periods=max(2, w // 2)).apply(np.argmin, raw=True) / max(w - 1, 1)


def ts_downside_std(x, w):
    return x.where(x < 0).rolling(w, min_periods=max(3, w // 2)).std()


def ts_upside_std(x, w):
    return x.where(x > 0).rolling(w, min_periods=max(3, w // 2)).std()


# ---------------- 二元 / 一元 ----------------
def safe_div(a, b):
    return a / b.where(b.abs() > EPS)


def signed_log(x):
    return np.sign(x) * np.log1p(np.abs(x))


def signed_sqrt(x):
    return np.sign(x) * np.sqrt(np.abs(x))


def prod_rank(a, b):
    return cs_rank(a) * cs_rank(b)


def diff_rank(a, b):
    return cs_rank(a) - cs_rank(b)


# ---------------- 标准化管线 ----------------
def standardize(x: pd.DataFrame, universe: pd.DataFrame | None = None) -> pd.DataFrame:
    """因子入库前的统一处理:限定股票池 -> 去极值 -> 截面 zscore。"""
    if universe is not None:
        x = x.where(universe.reindex_like(x).fillna(False))
    x = x.replace([np.inf, -np.inf], np.nan)
    return cs_zscore(cs_winsor(x))


def cs_neutralize_cap_ind(x: pd.DataFrame, log_cap: pd.DataFrame | None,
                          labels: pd.DataFrame | None = None,
                          universe: pd.DataFrame | None = None) -> pd.DataFrame:
    """市值 + 行业 双重中性化(逐日截面 OLS 残差)。

    用 FWL 定理分两步等价实现 y ~ 1 + log_cap + 行业哑变量:
      1) y 与 log_cap 各自在行业内去均值(含自身,即对哑变量的 OLS 投影);
      2) 去均值后的 y 对去均值后的 log_cap 做一元回归取残差。
    labels 为 None 时退化为纯市值中性化。
    """
    from afactor import statind as SI
    # 中性化必须限定在决策池内:池外股票会污染投影,导致池内残差仍与市值相关
    y = x.where(universe.reindex_like(x).fillna(False)) if universe is not None else x
    c = log_cap.reindex_like(x) if log_cap is not None else None
    if universe is not None and c is not None:
        c = c.where(universe.reindex_like(x).fillna(False))
    if labels is not None:
        L = labels.reindex_like(x)
        y = SI.label_demean(y, L, loo=False)
        if c is not None:
            c = SI.label_demean(c, L, loo=False)
    if log_cap is None:
        return y                     # 仅行业中性化(不做第二步回归)
    return cs_neutralize_1d(y, c)
