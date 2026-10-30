"""实验二:统计行业 —— 用滚动窗口日收益相关聚类替代缺失的行业分类。

每季度末用**截至当季末**的过去 120 个交易日日收益,对个股做 KMeans(K=30,
标准化后的欧氏距离 ≈ 相关距离),标签应用于**下一季度**(无前视)。
标签缓存为 cache/daily/statind_labels.parquet(-1 = 无标签)。
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from afactor.config import DAILY

K = 30
LOOKBACK = 120
MIN_VALID = 80
LABEL_FILE = DAILY / "statind_labels.parquet"


def _safe_std(X, mu, sd):
    with np.errstate(invalid="ignore", divide="ignore"):
        return (X - mu) / np.where(sd > 1e-9, sd, np.nan)


def build_labels(panel) -> pd.DataFrame:
    from sklearn.cluster import KMeans
    close = panel.raw("close_px")
    ret = (close / close.shift(1) - 1.0).clip(-0.25, 0.25)
    dates = ret.index
    q_ends = pd.Series(dates).groupby(dates.to_period("Q")).max().tolist()
    L = pd.DataFrame(-1, index=dates, columns=ret.columns, dtype=np.int16)
    for qi, qe in enumerate(q_ends):
        pos = dates.get_loc(qe)
        if pos < LOOKBACK:
            continue
        win = ret.iloc[pos - LOOKBACK + 1: pos + 1]
        valid = win.notna().sum() >= MIN_VALID
        cols = valid[valid].index
        X = win[cols].to_numpy(np.float64)
        mu = np.nanmean(X, axis=0); sd = np.nanstd(X, axis=0)
        ok = sd > 1e-9                                   # 先判定,再填充(长期停牌/一字股剔除)
        X = _safe_std(X, mu, sd)
        X = np.where(np.isfinite(X), X, 0.0).T          # (N,120) 标准化收益向量,缺日以 0(=均值)填
        km = KMeans(n_clusters=K, n_init=4, random_state=qi).fit(X[ok])
        lab = np.full(len(cols), -1, np.int16)
        lab[ok] = km.labels_
        # 应用到下一季度
        start = pos + 1
        end = dates.get_loc(q_ends[qi + 1]) if qi + 1 < len(q_ends) else len(dates) - 1
        if start > end:
            continue
        L.iloc[start:end + 1, [L.columns.get_loc(c) for c in cols]] = lab[None, :]
    L.to_parquet(LABEL_FILE)
    return L


def load_labels() -> pd.DataFrame:
    return pd.read_parquet(LABEL_FILE)


# ---------- 组运算(逐日 bincount,向量化) ----------
def label_mean(x: pd.DataFrame, L: pd.DataFrame, loo: bool = True) -> pd.DataFrame:
    """每只股票所属统计行业当日的 x **留一均值**(不含自身)。

    审查修正(2026-08-20):含自身的组均值会把个股自身信号混入"行业"因子
    (小簇尤甚),且使 label_demean 在单成员簇上恒为 0 产生伪中性信号。
    留一法:gm_loo_i = (S_g - x_i) / (n_g - 1);n_g < 2 处为 NaN。"""
    X = x.to_numpy(np.float64)
    Lab = L.reindex_like(x).fillna(-1).to_numpy(np.int64)
    out = np.full_like(X, np.nan)
    for t in range(X.shape[0]):
        lab = Lab[t]; v = X[t]
        m = (lab >= 0) & np.isfinite(v)
        if m.sum() < 50:
            continue
        s = np.bincount(lab[m], weights=v[m], minlength=K)
        c = np.bincount(lab[m], minlength=K).astype(np.float64)
        cm = c[lab[m]]
        with np.errstate(invalid="ignore", divide="ignore"):
            if loo:
                val = (s[lab[m]] - v[m]) / np.where(cm >= 2, cm - 1.0, np.nan)
            else:
                val = s[lab[m]] / np.where(cm >= 1, cm, np.nan)
        out[t, m] = val
    return pd.DataFrame(out, index=x.index, columns=x.columns)


def label_demean(x: pd.DataFrame, L: pd.DataFrame, loo: bool = True) -> pd.DataFrame:
    """loo=True(默认):留一去均值,用于**因子构造**,避免个股自身信号混入组均值。
    loo=False:含自身去均值,等价于对行业哑变量做 OLS 投影,用于**中性化**。"""
    return x - label_mean(x, L, loo=loo)


def label_std(x: pd.DataFrame, L: pd.DataFrame) -> pd.DataFrame:
    m = label_mean(x, L)
    return np.sqrt(label_mean((x - m) ** 2, L))
