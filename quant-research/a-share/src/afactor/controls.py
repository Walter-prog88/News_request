"""对照因子与置换检验:判断"高 IC 通过率"是真信号还是共同驱动/统计假象。

- ctrl_size_*   : 规模/流动性代理(无市值数据时用成交额),若它本身 IC 极高,
                  则大量因子的高 IC 只是它的马甲。
- ctrl_random_* : 纯噪声安慰剂,给出"零效应"在本检验框架下的 IC 分布。
- ctrl_shuffle_*: 真实因子做截面置换,保留边际分布但打乱与收益的对应关系。
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from afactor import ops
from afactor.panel import Panel


def build_controls(p: Panel, n_random: int = 8, seed: int = 20260820) -> dict[str, pd.DataFrame]:
    out = {}
    idx, col = p.dates, p.codes
    out["ctrl_size_adv20"] = -ops.cs_rank(p["adv20"])
    out["ctrl_size_amount"] = -ops.cs_rank(p.raw("amount"))
    out["ctrl_illiq_amihud"] = ops.cs_rank(ops.ts_mean(p.raw("amihud"), 20))
    out["ctrl_ntrade"] = -ops.cs_rank(ops.ts_mean(p.raw("n_trade_min"), 20))
    rng = np.random.default_rng(seed)
    for i in range(n_random):
        out[f"ctrl_random_{i}"] = pd.DataFrame(
            rng.standard_normal((len(idx), len(col))).astype(np.float32), index=idx, columns=col)
    # 慢变量噪声:随机数按 20 日平滑,模拟"低换手噪声因子"
    for i in range(3):
        base = pd.DataFrame(rng.standard_normal((len(idx), len(col))).astype(np.float32),
                            index=idx, columns=col)
        out[f"ctrl_slowrandom_{i}"] = ops.ts_mean(base, 20)
    return out


def shuffle_cs(f: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """逐日截面置换:保留每日因子值的边际分布,打乱与个股的对应。"""
    rng = np.random.default_rng(seed)
    A = f.to_numpy(np.float32).copy()
    for t in range(A.shape[0]):
        row = A[t]
        m = np.flatnonzero(np.isfinite(row))
        if m.size > 1:
            row[m] = row[rng.permutation(m)]
    return pd.DataFrame(A, index=f.index, columns=f.columns)
