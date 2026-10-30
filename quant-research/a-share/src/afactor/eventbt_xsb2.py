"""XSB2:小碎步二轮 —— 深回调低吸规格(26 笔取证校准)。协议见 research/XSB2_PROTOCOL.md。

信号层:共享门 G1-G5 + 三触发分支(T1 放量反转 / T2 强势浅回踩 / T3 缩量左侧)。
执行层复用 eventbt.run_event_bt(本模块只产 signals dict,新增 exit_extra 通道)。
信息时点:全日字段一律 shift(1);T 日仅用 open_auction_px、pre_ret_intraday、pre_amount。
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd

from afactor.panel import Panel
from afactor.eventbt import _ma


@dataclass
class XRules:
    # 共享门(取证依据见协议 §0)
    gates_on: bool = True
    rally_min: float = 1.0        # G1 前期涨幅 hi252/lo504-1 下限(24/26)
    rally_hi_win: int = 252
    rally_lo_win: int = 504
    ma_long: int = 250            # G2 年线
    ma_long_slope: int = 20       # 年线向上 = MA250 > 20 日前(24/26)
    pb_min: float = 0.20          # G3 深回调下限(22/26;T2 豁免)
    pb_deep: float = 0.65         # 回撤超过此深度不接
    dsh_min: int = 20             # G4 距 252 日高点天数(中位 42)
    chase_max: float = 0.08       # G5 追高限(实际观测上限,非口述的 1%)
    # 触发分支
    t1: bool = True               # 放量反转/首阳(用户分类 1)
    t2: bool = False              # 强势浅回踩(用户分类 2 = 一轮口径收编)
    t3: bool = True               # 缩量左侧企稳(用户分类 3)
    t1_look: int = 5
    t1_volr: float = 1.3
    t1_ret: float = 0.03
    t2_rp_min: float = 0.70
    t2_ma: int = 60
    t3_volr: float = 0.85
    t3_absret: float = 0.03
    t3_ret5: float = 0.02
    t3_intraday_scale: float = 0.95   # 14:49 时点成交额占全日典型比例(粗校准,入族键不入网格)
    # 退出
    exit_ma: int | None = 60      # 趋势退出均线;None=关。E1 教训:60 与深回调入场自相矛盾
                                  # (入场中位在 MA60 下方 15%,80% 交易开仓即被 MA60 踢出,持有 3.5 天)
    use_tp_hi: bool = True        # 前高不放量止盈
    tphi_band: float = 0.05
    tphi_volr: float = 1.0
    use_decay: bool = True        # 缩量阴跌退出
    decay_win: int = 20
    decay_amt: float = 0.8
    decay_ma: int = 20


def _days_since_high(c1: pd.DataFrame, hi: pd.DataFrame) -> pd.DataFrame:
    """距最近一次创 hi 窗口新高的交易日数(无新高记录处为 +inf 级大数)。"""
    is_hi = ((c1 >= hi * (1 - 1e-9)) & hi.notna()).to_numpy()
    t = np.arange(len(c1))[:, None]
    last = np.maximum.accumulate(np.where(is_hi, t, -1), axis=0)
    dsh = np.where(last < 0, 10 ** 6, t - last)
    return pd.DataFrame(dsh, index=c1.index, columns=c1.columns)


def _sector_rp(close, universe, labels, win):
    lab = labels.dropna()
    lab = lab[lab.index.isin(close.columns)]
    r = close.pct_change().where(universe)
    ir = r[lab.index].T.groupby(lab).mean().T
    idx = (1 + ir.fillna(0)).cumprod().shift(1)
    slo = idx.rolling(win, min_periods=120).min()
    shi = idx.rolling(win, min_periods=120).max()
    srp = (idx - slo) / (shi - slo)
    return pd.DataFrame({c: srp[lab[c]] for c in lab.index}).reindex(columns=close.columns)


def build_xsb2_signals(p: Panel, xr: XRules, universe: pd.DataFrame,
                       sector_labels: pd.Series | None) -> dict:
    close = p.raw("close_px")
    c1 = close.shift(1)
    opn = p.raw("open_auction_px")
    px = opn * (1.0 + p.raw("pre_ret_intraday"))          # T 日 14:49 价
    amount = p.raw("amount")
    a1 = amount.shift(1)
    adv20_1 = a1.rolling(20, min_periods=10).mean()
    volr_prev = a1 / adv20_1                               # T-1 日量比(全日,已知)
    pre_amt = p.raw("pre_amount")                          # T 日 09:30-14:49 累计额,已知
    ret49 = px / c1 - 1.0                                  # T 日 14:49 相对昨收

    hi252 = c1.rolling(xr.rally_hi_win, min_periods=200).max()
    lo504 = c1.rolling(xr.rally_lo_win, min_periods=252).min()
    rally = hi252 / lo504 - 1.0
    ma_long = _ma(c1, xr.ma_long)
    pb = px / hi252 - 1.0
    dsh = _days_since_high(c1, hi252)

    g1 = rally >= xr.rally_min
    g2 = ma_long > ma_long.shift(xr.ma_long_slope)
    g3 = (pb <= -xr.pb_min) & (pb >= -xr.pb_deep)
    g4 = dsh >= xr.dsh_min
    g5 = ret49 <= xr.chase_max
    if not xr.gates_on:
        g1 = g2 = g4 = pd.DataFrame(True, index=px.index, columns=px.columns)
        g3 = pd.DataFrame(True, index=px.index, columns=px.columns)

    trig = pd.DataFrame(False, index=px.index, columns=px.columns)
    if xr.t1:
        confirm = (volr_prev >= xr.t1_volr) & (c1 / c1.shift(1) - 1.0 >= xr.t1_ret)
        recent = confirm.rolling(xr.t1_look, min_periods=1).max() >= 1
        t1 = recent & (ret49 >= 0) & (px >= c1.rolling(5, min_periods=3).min()) & g3
        trig |= t1.fillna(False)
    if xr.t2:
        lo252 = c1.rolling(252, min_periods=200).min()
        rp = (c1 - lo252) / (hi252 - lo252)
        t2 = ((pb > -xr.pb_min) & (px > _ma(c1, xr.t2_ma)) & (rp >= xr.t2_rp_min))
        trig |= t2.fillna(False)          # T2 豁免 G3(浅回踩分支)
    if xr.t3:
        quiet = pre_amt / (xr.t3_intraday_scale * adv20_1) <= xr.t3_volr
        ret5 = px / c1.shift(4) - 1.0
        t3 = quiet & (ret49.abs() <= xr.t3_absret) & (ret5 <= xr.t3_ret5) & g3
        trig |= t3.fillna(False)

    gate = (g1 & g2 & g4 & g5).fillna(False) & universe
    entry = (gate & trig).fillna(False)

    sec_rp = None
    if sector_labels is not None:
        sec_rp = _sector_rp(close, universe, sector_labels, 252)
        lo252v = c1.rolling(252, min_periods=200).min()
        rp_all = (c1 - lo252v) / (hi252 - lo252v)
        score = sec_rp.fillna(0) + rp_all.fillna(0) * 0.01
    else:
        score = pd.DataFrame(0.0, index=px.index, columns=px.columns)

    exit_extra = {}
    if xr.use_tp_hi:
        exit_extra["tp_hi"] = ((px >= (1 - xr.tphi_band) * hi252)
                               & (volr_prev <= xr.tphi_volr)).fillna(False)
    if xr.use_decay:
        retw = px / c1.shift(xr.decay_win - 1) - 1.0
        amt_now = a1.rolling(xr.decay_win, min_periods=10).sum()
        amt_prev = a1.shift(xr.decay_win).rolling(xr.decay_win, min_periods=10).sum()
        exit_extra["decay"] = ((retw < 0) & (amt_now / amt_prev <= xr.decay_amt)
                               & (px < _ma(c1, xr.decay_ma))).fillna(False)

    ma_exit = (_ma(c1, xr.exit_ma) if xr.exit_ma
               else pd.DataFrame(np.nan, index=px.index, columns=px.columns))
    return dict(entry=entry, score=score.where(entry), score_all=score, px=px,
                ma_exit=ma_exit, exit_extra=exit_extra,
                gate=gate, trig=trig, pb=pb, rally=rally, sec_rp=sec_rp)
