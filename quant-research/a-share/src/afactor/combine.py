"""组合因子构建:方向对齐 -> 标准化 -> 等权/滚动IC加权/聚类合成。
组合因子会作为独立因子重新走一遍完整检验。
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from afactor import ops
from afactor.panel import Panel
from afactor.fast_eval import IC_DIR


def prepare(fac: dict[str, pd.DataFrame], signs: dict[str, float],
            universe: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """方向对齐 + 去极值 + 截面 zscore(限定在决策池内)。"""
    out = {}
    for k, f in fac.items():
        z = ops.standardize(f, universe)
        out[k] = z * float(np.sign(signs.get(k, 1.0)))
    return out


def combine_equal(z: dict[str, pd.DataFrame]) -> pd.DataFrame:
    arrs = [v.to_numpy(np.float32) for v in z.values()]
    stack = np.stack(arrs)
    with np.errstate(invalid="ignore"):
        s = np.nanmean(stack, axis=0)
    n_ok = np.isfinite(stack).sum(axis=0)
    s[n_ok < max(1, len(arrs) // 2)] = np.nan
    ref = next(iter(z.values()))
    return pd.DataFrame(s, index=ref.index, columns=ref.columns)


def combine_ic_weighted(z: dict[str, pd.DataFrame], mode: str = "open",
                        halflife: int = 120, min_days: int = 250) -> pd.DataFrame:
    """用**截至前一日**的指数加权 IC 作权重(无前视)。"""
    names = list(z)
    ics = {}
    for n in names:
        f = IC_DIR / f"{n}__{mode}.parquet"
        s = pd.read_parquet(f)["h1"] if f.exists() else None
        ics[n] = s
    ref = next(iter(z.values()))
    W = pd.DataFrame({n: (ics[n].reindex(ref.index).ewm(halflife=halflife, min_periods=min_days)
                          .mean().shift(1) if ics[n] is not None else np.nan)
                      for n in names}, index=ref.index)
    W = W.abs()                      # 方向已对齐,权重取 IC 绝对强度
    W = W.div(W.sum(axis=1).replace(0, np.nan), axis=0)
    stack = np.stack([z[n].to_numpy(np.float32) for n in names])
    wt = np.stack([W[n].to_numpy(np.float32) for n in names])[:, :, None]
    num = np.nansum(np.where(np.isfinite(stack), stack * wt, 0.0), axis=0)
    den = np.nansum(np.where(np.isfinite(stack), wt, 0.0), axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        s = np.where(den > 0, num / den, np.nan)
    return pd.DataFrame(s, index=ref.index, columns=ref.columns)


def cluster_by_ic(names: list[str], mode: str = "open", thr: float = 0.5,
                  signs: dict[str, float] | None = None) -> dict[int, list[str]]:
    """按 IC 序列相关做层次聚类(相关系数 >= thr 视为同一信息簇)。"""
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform
    M = {}
    for n in names:
        f = IC_DIR / f"{n}__{mode}.parquet"
        if f.exists():
            M[n] = pd.read_parquet(f)["h1"]
    M = pd.DataFrame(M)
    if signs:
        M = M.mul(pd.Series({k: np.sign(v) for k, v in signs.items()}).reindex(M.columns), axis=1)
    C = M.corr(min_periods=200).fillna(0.0)
    D = 1.0 - C.abs().to_numpy()
    np.fill_diagonal(D, 0.0)
    D = (D + D.T) / 2
    Z = linkage(squareform(D, checks=False), method="average")
    lab = fcluster(Z, t=1.0 - thr, criterion="distance")
    out = {}
    for n, l in zip(C.columns, lab):
        out.setdefault(int(l), []).append(n)
    return out


def combine_cluster(z: dict[str, pd.DataFrame], clusters: dict[int, list[str]]) -> pd.DataFrame:
    """簇内等权 -> 簇间等权(降低同源信息的重复计权)。"""
    reps = {}
    for cid, members in clusters.items():
        mem = {m: z[m] for m in members if m in z}
        if mem:
            reps[f"c{cid}"] = combine_equal(mem)
    return combine_equal(reps)


def residual_ic_factor(f: pd.DataFrame, p: Panel) -> pd.DataFrame:
    """对规模/流动性代理(20日日均成交额)截面正交化后的残差因子。"""
    size = ops.cs_rank(ops.signed_log(p["adv20"]))
    return ops.cs_neutralize(f, size)


# ================= 维度层组合(2026-08-22) =================
def dimension_groups(names: list[str], signs: dict, mode: str = "open",
                     k: int = 5) -> dict[int, list[str]]:
    """把成员按 IC 序列相关聚成 k 个"真实维度"(方向已对齐后聚类)。

    动机:等权 N 个成员实际是按"挖掘密度"加权 —— 某方向挖得多、席位就多。
    实测 COMP_v5 的 25 个成员只有约 4.9 个有效独立维度(participation ratio),
    前 3 主成分解释 62% 方差。先聚成维度、维度内合成、维度间等权,可消除该失衡。
    """
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform
    from afactor.fast_eval import IC_DIR
    M = {}
    for n in names:
        f = IC_DIR / f"{n}__{mode}.parquet"
        if f.exists():
            M[n] = pd.read_parquet(f)["h1"] * np.sign(signs.get(n, 1.0))
    M = pd.DataFrame(M)
    C = M.corr(min_periods=200).fillna(0.0)
    D = 1.0 - C.abs().to_numpy()
    np.fill_diagonal(D, 0.0)
    D = (D + D.T) / 2
    lab = fcluster(linkage(squareform(D, checks=False), method="average"), t=k, criterion="maxclust")
    out = {}
    for n, l in zip(C.columns, lab):
        out.setdefault(int(l), []).append(n)
    return out


def effective_dimensions(names: list[str], signs: dict, mode: str = "open") -> float:
    """IC 序列相关矩阵的 participation ratio —— "这些成员实际代表几个独立观点"。"""
    from afactor.fast_eval import IC_DIR
    M = {}
    for n in names:
        f = IC_DIR / f"{n}__{mode}.parquet"
        if f.exists():
            M[n] = pd.read_parquet(f)["h1"] * np.sign(signs.get(n, 1.0))
    C = pd.DataFrame(M).corr(min_periods=200).fillna(0.0).to_numpy()
    np.fill_diagonal(C, 1.0)
    C = (C + C.T) / 2
    lam = np.clip(np.linalg.eigvalsh(C), 0, None)
    return float(lam.sum() ** 2 / (lam ** 2).sum())


def combine_by_dimension(z: dict[str, pd.DataFrame], groups: dict[int, list[str]],
                         scheme: str = "equal", ic_series: dict | None = None) -> pd.DataFrame:
    """维度内等权合成 -> 维度间加权。

    scheme='equal'      维度间等权(无拟合参数,推荐)
    scheme='riskparity' 维度间按 IC 序列波动的倒数加权(用全样本 IC 波动,属轻度样本内)
    """
    reps = {}
    for cid, mem in groups.items():
        mem = [m for m in mem if m in z]
        if mem:
            reps[f"dim{cid}"] = combine_equal({m: z[m] for m in mem})
    if scheme == "equal" or ic_series is None:
        return combine_equal(reps)
    w = {}
    for cid, mem in groups.items():
        key = f"dim{cid}"
        if key not in reps:
            continue
        vols = [ic_series[m].std() for m in mem if m in ic_series]
        w[key] = 1.0 / np.mean(vols) if vols else 1.0
    tot = sum(w.values())
    arrs = [reps[k].to_numpy(np.float32) * (w[k] / tot) for k in reps]
    stack = np.stack(arrs)
    with np.errstate(invalid="ignore"):
        s = np.nansum(stack, axis=0)
    n_ok = np.isfinite(np.stack([reps[k].to_numpy(np.float32) for k in reps])).sum(axis=0)
    s[n_ok < max(1, len(reps) // 2)] = np.nan
    ref = next(iter(reps.values()))
    return pd.DataFrame(s, index=ref.index, columns=ref.columns)
