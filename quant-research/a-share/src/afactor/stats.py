"""多重检验去偏:有效独立试验数、Bonferroni/BH-FDR、去偏后的 IC t 门槛。

搜索了 N 个因子后,"最好的那个"天然会好看。这里给出在本项目试验规模下,
一个因子的 t 值要达到多少才不能用运气解释。
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats as sps

from afactor.fast_eval import IC_DIR


def effective_n_trials(names: list[str], mode: str = "open", col: str = "h1") -> dict:
    """由 IC 序列相关矩阵估计有效独立试验数。

    N_eff = (sum λ)^2 / sum λ^2 (participation ratio),对高度相关的因子族给出远小于 N 的计数。
    """
    M = {}
    for n in names:
        f = IC_DIR / f"{n}__{mode}.parquet"
        if f.exists():
            M[n] = pd.read_parquet(f)[col]
    M = pd.DataFrame(M).dropna(how="all")
    C = M.corr(min_periods=200).fillna(0.0).to_numpy()
    C = (C + C.T) / 2
    np.fill_diagonal(C, 1.0)
    lam = np.linalg.eigvalsh(C)
    lam = np.clip(lam, 0, None)
    n_eff = float(lam.sum() ** 2 / (lam ** 2).sum())
    return {"n_trials": len(M.columns), "n_eff": n_eff,
            "top10_var_share": float(np.sort(lam)[::-1][:10].sum() / lam.sum())}


def expected_max_t(n_trials: float) -> float:
    """N 次独立试验下,标准正态最大值的期望(Gumbel 近似)——"运气门槛"。"""
    n = max(n_trials, 2.0)
    a = np.sqrt(2 * np.log(n))
    return float(a - (np.log(np.log(n)) + np.log(4 * np.pi)) / (2 * a) + 0.5772 / a)


def bh_fdr(pvals: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Benjamini-Hochberg,返回是否通过 FDR 控制。"""
    p = np.asarray(pvals, float)
    n = p.size
    order = np.argsort(p)
    thr = alpha * (np.arange(1, n + 1)) / n
    passed = p[order] <= thr
    k = np.max(np.flatnonzero(passed)) + 1 if passed.any() else 0
    out = np.zeros(n, bool)
    if k:
        out[order[:k]] = True
    return out


def deflate(df: pd.DataFrame, names_for_neff: list[str] | None = None,
            mode: str = "open") -> pd.DataFrame:
    """给账本加上去偏列。df 需含 factor / ic_t 列。"""
    d = df.copy()
    d["p_raw"] = 2 * sps.norm.sf(d["ic_t"].abs())
    n_all = len(d)
    eff = effective_n_trials(names_for_neff or d["factor"].tolist(), mode)
    d.attrs["effective"] = eff
    d["t_thr_bonferroni_all"] = expected_max_t(n_all)
    d["t_thr_bonferroni_eff"] = expected_max_t(eff["n_eff"])
    d["pass_bonf_all"] = d["ic_t"].abs() >= d["t_thr_bonferroni_all"]
    d["pass_bonf_eff"] = d["ic_t"].abs() >= d["t_thr_bonferroni_eff"]
    d["pass_fdr"] = bh_fdr(d["p_raw"].to_numpy())
    return d
