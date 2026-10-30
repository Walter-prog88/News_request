"""INC-T4(弱因子组合 + 成本感知)公共设定与工具。预注册:research/INC_T4_PREREG.md(sha256 17c4276c…fedd3)。

本模块只放口径常量与无副作用的工具函数;所有参数在预注册里定死,不得按结果改动。
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from afactor.config import LEDGER, TMP, ROOT, RESEARCH_END  # noqa: E402

PFX = "inc_t4"
LOGS = ROOT / "logs"
LOGS.mkdir(exist_ok=True)
TD = 242
VERSION = "V1000"
MODE = "open10"
AUM = 1e7
REBALANCE = 5
NW_BW = 10                         # max(5, 2h),h = 5
SIM_START = "2020-07-01"           # 连续路径预热起点(与 capacity_grid 排雷基线相同)
PREREG = ROOT / "research" / "INC_T4_PREREG.md"
PREREG_SHA = "17c4276c76b899a8d1df566a0a1443080cd74cecafdfe6d972efa775156fedd3"

# 成员集(§3.2)
SPLITS = {"A": (2020, 2023), "main": (2020, 2024), "B": (2020, 2022)}
EXPECT_N = {"A": 518, "main": 475, "B": 577}
EXPECT_MED = {"A": 0.0166, "main": 0.0161, "B": 0.0161}
EXPECT_NOPEAD = {"A": 515, "main": 473, "B": 574}
MID_LO, MID_HI, MIN_DAYS = 0.010, 0.025, 400
SETS = ["A", "main", "B", "A_nopead", "main_nopead", "B_nopead"]

# 网格(§6)
HL_GRID = [0, 2, 5, 10, 20, 40]
TURN_RULE_MULT = 1.15
B_GRID = [(0.10, 0.25), (0.10, 0.35), (0.20, 0.35), (0.20, 0.45)]
B_MAIN = (0.10, 0.35)
DEEP_GRID = [(0.25, 0.40), (0.30, 0.45), (0.35, 0.50), (0.40, 0.55)]
BASE_BUF = (0.20, 0.35)

# 窗口(§3.4)
WINDOWS = {"full": (2021, 2025), "dev": (2021, 2023), "test": (2024, 2025), "y2025": (2025, 2025),
           "w2023_2025": (2023, 2025)}
YEARS = [2021, 2022, 2023, 2024, 2025]
CRASH = (pd.Timestamp("2024-01-02"), pd.Timestamp("2024-02-07"))

# 基线复现门槛(§3.5 ④),capacity_grid.csv COMP_v5_exclude|open10|Y1.0|1e7|pov
BASE_REF = {"full": 0.053175, "dev": 0.043395, "test": 0.069878}
BASE_TURN_REF = 0.019339


def check_prereg() -> str:
    h = hashlib.sha256(open(PREREG, "rb").read()).hexdigest()
    if h != PREREG_SHA:
        raise RuntimeError(f"预注册文件 sha256 变了:{h}(应为 {PREREG_SHA})—— 正文不得改写")
    return h


def sha256_file(path) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def members() -> dict:
    return json.load(open(TMP / f"{PFX}_members.json"))


def d_path(s: str):
    return TMP / f"{PFX}_D_{s}.parquet"


def ds_path(s: str, hl: int):
    return TMP / f"{PFX}_Ds_{s}_hl{hl}.parquet"


def ewm_smooth(D: pd.DataFrame, hl: int) -> pd.DataFrame:
    """§3.3:逐股票 EWM(halflife=HL 信号日, min_periods=1, ignore_na=True, adjust=True);HL=0 不平滑。
    只用 <= t 的值(ewm 是因果的)。"""
    if hl == 0:
        return D
    return D.ewm(halflife=hl, min_periods=1, ignore_na=True, adjust=True).mean().astype(np.float32)


def nw_t(x, bw: int = NW_BW) -> float:
    """均值的 Newey-West t(Bartlett 核),与 cap_capacity_curve._nw_t 同式。"""
    x = np.asarray(x, np.float64)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 10:
        return np.nan
    u = x - x.mean()
    lrv = u @ u / n
    for L in range(1, bw + 1):
        lrv += 2.0 * (1.0 - L / (bw + 1)) * (u[L:] @ u[:-L]) / n
    return float(x.mean() / np.sqrt(lrv / n)) if lrv > 0 else np.nan


def geo_ann(r: pd.Series) -> float:
    r = r.dropna()
    if len(r) == 0:
        return np.nan
    return float((1.0 + r).prod() ** (TD / len(r)) - 1.0)


def win_slice(s, w):
    a, b = WINDOWS[w] if isinstance(w, str) else w
    yrs = s.index.year
    return s[(yrs >= a) & (yrs <= b)]


def paired_stats(rv: pd.Series, rb: pd.Series, w) -> dict:
    """变体 − 基线 日收益差的算术年化 Δ、NW t(bw 10)、配对 TE、SE、MDE = 2·SE。"""
    a, b = rv.align(rb, join="inner")
    d = win_slice((a - b).dropna(), w)
    n = len(d)
    if n < 30:
        return dict(n_days=n)
    delta = float(d.mean() * TD)
    t = nw_t(d.to_numpy())
    se = delta / t if t and np.isfinite(t) and t != 0 else np.nan
    return dict(n_days=n, delta=delta, t_nw=t, te_pair=float(d.std() * np.sqrt(TD)), se=abs(se),
                mde=2 * abs(se), lo95=delta - 2 * abs(se), hi95=delta + 2 * abs(se))


def two_layer(W1: np.ndarray, W2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """§6.2:持仓 = (W1>0)&(W2>0) 等权;额外剔除集 E = (W1>0)&(W2=0) 等权(H2 分解用)。"""
    hold = (W1 > 0) & (W2 > 0)
    extra = (W1 > 0) & ~(W2 > 0)

    def ew(m):
        n = m.sum(axis=1, keepdims=True).astype(np.float32)
        return np.divide(m.astype(np.float32), n, out=np.zeros(m.shape, np.float32), where=n > 0)
    return ew(hold), ew(extra)
