"""INC-T1 公用:策略定义、打分、窗口价格、窗口冲击缩放 s_W、引擎模式注入、NW t。

复用(预注册 §4.3):afactor.strategy / impact、scripts/cap_capacity_curve.py 的 STRATS / load_scores /
strategy_weights / sim_mask / ConstAUMCost。不改任何库代码:新执行模式在进程内注入
ST.EXEC_MODES 并替换 ST.exec_prices(与 cap_capacity_curve.exante_sizing 同一模式)。
"""
from __future__ import annotations
import os, sys, warnings
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
warnings.filterwarnings("ignore")

from afactor.config import LEDGER, TMP, REPORTS
from afactor.panel import Panel
from afactor import strategy as ST, impact as IM
from afactor.strategy import TRADING_DAYS

import cap_capacity_curve as CAP                       # noqa: E402  (ConstAUMCost / STRATS / load_scores)
import inc_t1_build_windows as BW                      # noqa: E402

ConstAUMCost = CAP.ConstAUMCost
VERSION = "V1000"
REBALANCE = 5
AUMS = [1e7, 1e8, 1e9]
AUM_LABEL = {1e7: "1000万", 1e8: "1亿", 1e9: "10亿"}
STRAT_KEYS = ["COMP_v5_exclude", "COMP_v5_top20", "B_gbt_top20"]
STRAT_LABEL = {"COMP_v5_exclude": "排雷20/35(COMP_v5)", "COMP_v5_top20": "Top20(COMP_v5)",
               "B_gbt_top20": "Top20(B×GBT)"}
PERIODS = {   # strategy -> {name: (y0, y1, role)}
    "COMP_v5_exclude": {"2021-2025": (2021, 2025, "full"), "2021-2023": (2021, 2023, "dev"), "2024-2025": (2024, 2025, "test")},
    "COMP_v5_top20": {"2021-2025": (2021, 2025, "full"), "2021-2023": (2021, 2023, "dev"), "2024-2025": (2024, 2025, "test")},
    "B_gbt_top20": {"2023-2025": (2023, 2025, "full"), "2023": (2023, 2023, "dev"), "2024-2025": (2024, 2025, "test")},
}
WINDOWS = list(BW.WINDOWS)             # W0..W11
MULTI = [w for w in WINDOWS if w not in BW.SINGLE_BAR]
PRIMARY = ["W9", "W5", "W7", "S:W1|B:W7", "S:W7|B:W1", "S:W1|B:W9"]   # 预注册主候选(§5.2)
SPLIT_SET = ["W1", "W3", "W5", "W7", "W9"]
BATCHES = [("W1", "W5"), ("W1", "W7"), ("W2", "W5"), ("W3", "W7"), ("W10", "W11")]
WIN_DESC = {"W0": "09:30 开盘竞价", "W1": "09:31-09:40", "W2": "09:41-10:00", "W3": "10:01-10:30",
            "W4": "10:31-11:30", "W5": "13:01-14:00", "W6": "14:01-14:30", "W7": "14:31-14:57",
            "W8": "15:00 收盘竞价", "W9": "全日 VWAP", "W10": "上午 VWAP", "W11": "下午 VWAP"}


# ------------------------------------------------------------------ 统计
def nw_t(x, bw: int = 5) -> float:
    """均值的 Newey-West t(Bartlett 核,带宽 bw)。"""
    x = np.asarray(x, np.float64)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 2 * bw + 10:
        return np.nan
    e = x - x.mean()
    s = float(np.sum(e * e)) / n
    for k in range(1, bw + 1):
        s += 2.0 * (1.0 - k / (bw + 1.0)) * float(np.sum(e[k:] * e[:-k])) / n
    if s <= 0:
        return np.nan
    return float(x.mean() / np.sqrt(s / n))


def period_mask(index: pd.DatetimeIndex, y0: int, y1: int) -> np.ndarray:
    y = np.asarray(index.year)
    return (y >= y0) & (y <= y1)


# ------------------------------------------------------------------ 策略与打分
def load_scores(p: Panel) -> dict:
    return CAP.load_scores(p)


def strategy_weights(p: Panel, scores: dict, strat: str) -> np.ndarray:
    return CAP.strategy_weights(p, scores, strat)


def sim_mask(p: Panel, strat: str) -> np.ndarray:
    return CAP.sim_mask(p, strat)


# ------------------------------------------------------------------ 价格面板
def win_px(p: Panel, w: str, kind: str) -> pd.DataFrame:
    """窗口成交价(对齐面板)。kind = vwap / twap;单根 bar 窗口两者相同。"""
    return BW.window_price(w, kind).reindex(index=p.dates, columns=p.codes)


def win_amt(p: Panel, w: str) -> pd.DataFrame:
    return BW.load_win("amt", w).reindex(index=p.dates, columns=p.codes)


# ------------------------------------------------------------------ 窗口内封板(审查 P0 修复)
SEAL_TOL = 0.985        # 与 panel._LIMIT_TOL 同:窗口价达到涨跌停幅度的 98.5% 视为封板
SEAL_TIGHT_ABS = 0.001  # tight 口径:窗口价距涨跌停幅度不超过 0.1 个百分点(绝对收益差)
SEAL_MODES = ("dir", "any", "tight")
_SEAL_CACHE: dict = {}


def window_seal(p: Panel, w: str, seal: str = "any") -> tuple:
    """窗口 w 在成交日 te 的"整窗口封板"掩码 (up, dn),形状 (T, N) 的 bool(行 = te)。

    up[te,i] :窗口内所有 bar 的价格下沿 min(open,close) 相对前收盘 close_px[te-1] 的收益 ≥ 涨停幅度 × 98.5%
               (窗口全程贴在涨停价上,买单排不进去);
    dn[te,i] :窗口内所有 bar 的价格上沿 max(open,close) 相对前收盘的收益 ≤ −跌停幅度 × 98.5%(卖单卖不出去)。
    涨跌停幅度 = panel._limit_frame(p)(主板 10%、ST 5%、创业板 20% 等);98.5% 与引擎 panel._LIMIT_TOL 同。
    单 bar 窗口(W0、W8)hi = lo = 竞价成交价。
    seal 口径(只改后续如何使用 up / dn,这里返回的掩码对 dir / any 相同;tight 的阈值更严):
      dir   买单只被 up 挡、卖单只被 dn 挡(物理上正确的方向性口径,第一层主口径)
      any   up | dn 都挡买卖双方(与引擎 can 的一字板约定一致,第二层只能用这个)
      tight 方向性口径,阈值改为 幅度 − 0.1pp(只数贴死在涨跌停价上的窗口,计数下界)
    hi / lo 缺失(窗口无 bar)时为 False。"""
    from afactor.panel import _limit_frame
    tight = seal == "tight"
    key = (w, tight)
    if key not in _SEAL_CACHE:
        close = p.raw("close_px").to_numpy(np.float64)
        pc = np.vstack([np.full((1, close.shape[1]), np.nan), close[:-1]])
        L = _limit_frame(p).to_numpy(np.float64)
        hi = BW.load_win("hi", w).reindex(index=p.dates, columns=p.codes).to_numpy(np.float64)
        lo = BW.load_win("lo", w).reindex(index=p.dates, columns=p.codes).to_numpy(np.float64)
        thr = (L - SEAL_TIGHT_ABS) if tight else (L * SEAL_TOL)
        with np.errstate(divide="ignore", invalid="ignore"):
            rlo = lo / pc - 1.0
            rhi = hi / pc - 1.0
            up = np.isfinite(rlo) & (rlo >= thr)
            dn = np.isfinite(rhi) & (rhi <= -thr)
        _SEAL_CACHE[key] = (up, dn)
    return _SEAL_CACHE[key]


def blocked_mask(p: Panel, w: str, seal: str, side_buy: np.ndarray, te: np.ndarray, i: np.ndarray) -> np.ndarray:
    """逐笔被挡标志:成交日 te、股票 i、买(side_buy)/ 卖。seal = none 时全 False。"""
    if seal == "none":
        return np.zeros(len(te), bool)
    up, dn = window_seal(p, w, seal)
    if seal == "any":
        return (up | dn)[te, i]
    return np.where(side_buy, up[te, i], dn[te, i])


# ------------------------------------------------------------------ 窗口冲击缩放 s_W(§6)
def window_scales(p: Panel) -> pd.DataFrame:
    """逐年池内均值比 s_W = sqrt(f_var / f_vol)(与 capacity_diagnostics 同法):
       f_var = mean_pool ln(hi_W/lo_W)² / mean_pool ln(H/L)²,f_vol = Σ_pool amt_W / Σ_pool amount;
       单价窗口 W0 / W8:f_var = E[ln(P_auc/P_prev)²] / E[ln(H/L)²],P_prev = 前收盘(W0)/ 14:57 bar 中价(W8);
       W9 = 1。池 = 当日 V1000 决策池。返回 DataFrame(年 × 窗口),缓存 ledger。"""
    f = LEDGER / "inc_t1_window_scales.csv"
    if f.exists():
        return pd.read_csv(f, index_col=0)
    U = ST.universe(p, VERSION).to_numpy(bool)
    amt = p.raw("amount").to_numpy(np.float64)
    hi, lo = p.raw("high_px").to_numpy(np.float64), p.raw("low_px").to_numpy(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        pk_day = np.log(hi / lo) ** 2
    yrs = np.asarray(p.dates.year)
    close = p.raw("close_px").to_numpy(np.float64)
    prev_close = np.vstack([np.full((1, close.shape[1]), np.nan), close[:-1]])
    mid237 = BW.load_extra("mid237").reindex(index=p.dates, columns=p.codes).to_numpy(np.float64)
    out = {}
    for w in WINDOWS:
        a = win_amt(p, w).to_numpy(np.float64)
        if w == "W9":
            out[w] = {y: 1.0 for y in sorted(set(yrs))}
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            if w in BW.SINGLE_BAR:
                px = win_px(p, w, "twap").to_numpy(np.float64)
                ref = prev_close if w == "W0" else mid237
                pk_w = np.log(px / ref) ** 2
            else:
                h = BW.load_win("hi", w).reindex(index=p.dates, columns=p.codes).to_numpy(np.float64)
                l = BW.load_win("lo", w).reindex(index=p.dates, columns=p.codes).to_numpy(np.float64)
                pk_w = np.log(h / l) ** 2
            fvol = a / amt
        col = {}
        for y in sorted(set(yrs)):
            sel = yrs == y
            Uy = U[sel]
            ok = Uy & np.isfinite(pk_day[sel]) & np.isfinite(pk_w[sel]) & (pk_day[sel] > 0) & np.isfinite(fvol[sel]) \
                & (amt[sel] > 0) & (fvol[sel] > 0)
            fvar = pk_w[sel][ok].mean() / pk_day[sel][ok].mean()
            fv = np.nansum(np.where(Uy, a[sel], 0.0)) / np.nansum(np.where(Uy, amt[sel], 0.0))
            col[y] = float(np.sqrt(fvar / fv))
        out[w] = col
    df = pd.DataFrame(out)
    df.index.name = "year"
    df.to_csv(f)
    return df


def stockday_scale(p: Panel, w: str, s_year: pd.DataFrame) -> np.ndarray:
    """逐股票日 s_W(敏感性):sqrt(f_var_i,d / f_vol_i,d),非有限或 <=0 时退回当年池均值比,截到 [0.1, 10]。"""
    yrs = np.asarray(p.dates.year)
    fb = np.array([s_year.loc[y, w] for y in yrs])[:, None]
    if w == "W9":
        return np.ones((len(p.dates), len(p.codes)), np.float32)
    amt = p.raw("amount").to_numpy(np.float64)
    hi, lo = p.raw("high_px").to_numpy(np.float64), p.raw("low_px").to_numpy(np.float64)
    a = win_amt(p, w).to_numpy(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        pk_day = np.log(hi / lo) ** 2
        if w in BW.SINGLE_BAR:
            px = win_px(p, w, "twap").to_numpy(np.float64)
            close = p.raw("close_px").to_numpy(np.float64)
            ref = (np.vstack([np.full((1, close.shape[1]), np.nan), close[:-1]]) if w == "W0" else
                   BW.load_extra("mid237").reindex(index=p.dates, columns=p.codes).to_numpy(np.float64))
            pk_w = np.log(px / ref) ** 2
        else:
            h = BW.load_win("hi", w).reindex(index=p.dates, columns=p.codes).to_numpy(np.float64)
            l = BW.load_win("lo", w).reindex(index=p.dates, columns=p.codes).to_numpy(np.float64)
            pk_w = np.log(h / l) ** 2
        s = np.sqrt((pk_w / pk_day) / (a / amt))
    s = np.where(np.isfinite(s) & (s > 0), s, fb)
    return np.clip(s, 0.1, 10.0).astype(np.float32)


class WinScaledConstAUMCost:
    """恒定 AUM + 窗口冲击缩放:冲击 × s_W(成交日 te 所在年份的池均值比,或常数)。价差不变。"""

    def __init__(self, base: IM.CostModel, s_row: np.ndarray, label: str = ""):
        self.base = base
        self.s_row = np.asarray(s_row, np.float64)
        self.label = label

    def extra_cost(self, te, abs_delta, capital, nav):
        sp, im = self.base.extra_cost(te, abs_delta, capital, 1.0)
        return sp, im * float(self.s_row[int(te)])

    def describe(self) -> dict:
        d = self.base.describe()
        d.update(aum_mode="const", window_scale=self.label)
        return d


def s_row_for(p: Panel, s_year: pd.DataFrame, w: str, variant: str = "pool_year") -> np.ndarray:
    yrs = np.asarray(p.dates.year)
    if variant == "one":
        return np.ones(len(yrs))
    return np.array([s_year.loc[y, w] for y in yrs], np.float64)


# ------------------------------------------------------------------ 引擎模式注入(§4.2)
_ORIG_EXEC_PRICES = ST.exec_prices
_PRICES: dict = {}
MODE_SPECS = {}     # mode -> (window, kind, lag)


def mode_name(w: str, kind: str, lag: int = 1, seal: bool = False) -> str:
    """seal=True:窗口内封死涨 / 跌停的股票日按不可成交处理(审查 P0 修复,见 window_seal)。"""
    return f"inc_t1_{w}_{kind}" + ("" if lag == 1 else f"_lag{lag}") + ("_seal" if seal else "")


def parse_mode(m: str) -> dict:
    """mode 名 -> dict(window, kind, lag, seal)。"""
    seal = m.endswith("_seal")
    b = m[:-5] if seal else m
    parts = b.split("_")
    return dict(window=parts[2], kind=parts[3], lag=2 if b.endswith("_lag2") else 1, seal=seal)


def register_mode(p: Panel, w: str, kind: str, lag: int = 1, seal: bool = False) -> str:
    """注入 ST.EXEC_MODES[mode] 并缓存成交价 / 成交额。目标必须是 ndarray(禁止 callable,见预注册 §4.2)。

    seal=True(审查 P0 修复):窗口内全程封死在涨停或跌停价的股票日(window_seal 的 any 口径)把窗口成交额置 0,
    引擎 can = ... & (win_amt > 0) 即判不可成交 —— 与引擎对全日一字板(dead = one_word_up | one_word_down)
    的处理同一约定(方向无关;引擎 can 是单一的逐股票日掩码,无法区分买 / 卖)。价格面板不动,持有收益路径不变。"""
    m = mode_name(w, kind, lag, seal)
    lo, hi = BW.WINDOWS[w]
    ST.EXEC_MODES[m] = dict(desc=f"INC-T1 T收盘信号 -> T+{lag} {WIN_DESC[w]} {kind.upper()}" + (" 封板不可成交" if seal else ""),
                            win=(lo, hi), lag=lag, tail=False)
    if m not in _PRICES:
        px, amt = win_px(p, w, kind), win_amt(p, w)
        if seal:
            up, dn = window_seal(p, w, "any")
            blk = pd.DataFrame(up | dn, index=p.dates, columns=p.codes)
            amt = amt.mask(blk, 0.0)
        _PRICES[m] = (px, amt)
    MODE_SPECS[m] = (w, kind, lag)
    _install()
    return m


def release_mode(m: str):
    _PRICES.pop(m, None)


def _patched_exec_prices(p_, mode):
    if isinstance(mode, str) and mode.startswith("inc_t1_"):
        if mode not in _PRICES:
            raise KeyError(f"模式 {mode} 未注册价格")
        return _PRICES[mode]
    return _ORIG_EXEC_PRICES(p_, mode)


def _install():
    if ST.exec_prices is not _patched_exec_prices:
        ST.exec_prices = _patched_exec_prices
