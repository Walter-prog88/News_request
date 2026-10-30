"""日频面板:懒加载原料、派生量、股票池、可成交性、前瞻收益。

口径(2026-08-20 与用户确认):
  主口径  : T 日收盘后出信号 -> T+1 开盘 TWAP(09:31-09:40) 买入 -> T+2 开盘 TWAP 卖出
  尾盘口径: T 日 14:50 出信号(仅 pre_* 特征) -> T 日 14:50-14:57 TWAP 买入 -> T+1 同窗口卖出
"""
from __future__ import annotations
from functools import lru_cache
import numpy as np
import pandas as pd

from afactor.config import (DAILY, SEALED, MIN_ADV20, MIN_LIST_DAYS, price_limit,
                            SEAL_START, SEAL_LOCK)

_LIMIT_TOL = 0.985      # 收益达到涨跌幅上限的 98.5% 视为触板
_ONE_WORD_RNG = 0.004   # 日内振幅小于此值且触板 -> 一字板,不可成交


class Panel:
    """按需加载日频原料。sealed=True 时读取 2026 封存集(受 lock 保护)。"""

    _is_tail = False    # 主口径

    def __init__(self, sealed: bool = False):
        self.sealed = sealed
        self.dir = SEALED if sealed else DAILY
        if sealed and SEAL_LOCK.exists():
            raise PermissionError(f"封存样本已使用过一次(见 {SEAL_LOCK}),不得重跑。")
        self._cache: dict[str, pd.DataFrame] = {}
        self._derived: dict[str, pd.DataFrame] = {}

    # ---------- 原料 ----------
    def raw(self, name: str) -> pd.DataFrame:
        if name not in self._cache:
            f = self.dir / f"{name}.parquet"
            if not f.exists():
                raise KeyError(f"原料不存在: {name}")
            self._cache[name] = pd.read_parquet(f).astype(np.float32)
        return self._cache[name]

    def __getitem__(self, name: str) -> pd.DataFrame:
        if name in self._derived or name in _DERIVED_FN:
            return self.derived(name)
        return self.raw(name)

    @property
    def features(self) -> list[str]:
        return sorted(p.stem for p in self.dir.glob("*.parquet"))

    @property
    def dates(self) -> pd.DatetimeIndex:
        return self.raw("close_px").index

    @property
    def codes(self) -> pd.Index:
        return self.raw("close_px").columns

    # ---------- 派生量 ----------
    def derived(self, name: str) -> pd.DataFrame:
        if name not in self._derived:
            self._derived[name] = _DERIVED_FN[name](self)
        return self._derived[name]

    # ---------- 前瞻收益 ----------
    def fwd_ret(self, horizon: int = 1, mode: str = "open") -> pd.DataFrame:
        """理论前瞻收益(纯价格,不施加成交约束)——用于 IC 与分层,避免前视。

        mode='open' : twap_open10[T+1+h] / twap_open10[T+1] - 1
        mode='tail' : twap_tail8[T+h]    / twap_tail8[T]    - 1
        mode='c2c'  : close_px[T+h]      / close_px[T]      - 1   (诊断用)
        """
        if mode == "open":
            px = self.raw("twap_open10")
            return px.shift(-(1 + horizon)) / px.shift(-1) - 1.0
        if mode == "tail":
            px = self.raw("twap_tail8")
            return px.shift(-horizon) / px - 1.0
        if mode == "c2c":
            px = self.raw("close_px")
            return px.shift(-horizon) / px - 1.0
        raise ValueError(mode)

    def fwd_ret_exec(self, horizon: int = 1, mode: str = "open") -> pd.DataFrame:
        """可实现收益:T+1 买不进(一字板/无量)的标的收益记 0,而非剔除。
        用于回测与"执行摩擦后 IC",不用于因子初筛。"""
        theo = self.fwd_ret(horizon, mode)
        if mode == "open":
            ok = self["twap_open10_f"].notna().shift(-1)
        elif mode == "tail":
            ok = self["twap_tail8_f"].notna()
        else:
            return theo
        return theo.where(ok.fillna(False), 0.0).where(theo.notna())


# ================= 派生量定义 =================
def _prev_close(p): return p.raw("close_px").shift(1)


def _ret_c2c(p):
    c = p.raw("close_px")
    return c / c.shift(1) - 1.0


def _valid(p):
    """当日有真实成交。"""
    return (p.raw("amount") > 0) & p.raw("close_px").notna()


def _days_listed(p):
    v = _valid(p)
    return v.cumsum().where(v)


def _adv20(p):
    return p.raw("amount").rolling(20, min_periods=10).mean()


def _limit_arr(p):
    return pd.Series({c: price_limit(c, None) for c in p.codes}, dtype=np.float32)


def _limit_frame(p):
    """逐日逐股的涨跌幅上限。

    三条规则(2026-08-22 审查后修正):
      1. 创业板 300/301 在 2020-08-24 前是 10%,之后 20%(科创板全期 20%);
      2. 风险警示板 ST/*ST **只把 10% 降到 5%** —— 创业板/科创板 ST 仍是 20%、
         北交所 30%。此前一律置 5%,误判 23254 个股票日(77 只非主板 ST 股),
         虚增涨停 1382 个(一字板计数不受影响,因有 rng<0.004 兜底);
      3. ST 面板缺失时**报错而非静默降级** —— 裸 except 会让规则 2 无声失效。
    """
    from afactor.config import GEM_20PCT_START
    base = _limit_arr(p)
    lim = pd.DataFrame(np.repeat(base.to_numpy()[None, :], len(p.dates), 0),
                       index=p.dates, columns=p.codes)
    gem = [c for c in p.codes if c.startswith(("300", "301"))]
    if gem:
        lim.loc[lim.index < pd.Timestamp(GEM_20PCT_START), gem] = 0.10
    st = p["is_st"].reindex_like(lim).fillna(False)
    lim = lim.mask(st & (lim == 0.10), 0.05)
    return lim


def _limit_up(p):
    return _ret_c2c(p) >= _limit_frame(p) * _LIMIT_TOL


def _limit_down(p):
    return _ret_c2c(p) <= -_limit_frame(p) * _LIMIT_TOL


def _is_st(p):
    from afactor import fundamental as FU
    return FU.build_st_panel(p.dates, p.codes)


def _market_cap(p):
    from afactor import fundamental as FU
    return FU.build_market_cap(p.dates, p.codes, p.raw("close_px"))


def _log_cap(p):
    return ops_signed_log(p["market_cap"])


def ops_signed_log(x):
    return np.sign(x) * np.log1p(np.abs(x))


def _one_word_up(p):
    return _limit_up(p) & (p.raw("rng") < _ONE_WORD_RNG)


def _one_word_down(p):
    return _limit_down(p) & (p.raw("rng") < _ONE_WORD_RNG)


def _twap_open10_f(p):
    """开盘 TWAP 成交价;当日不可成交(一字涨停或开盘窗口无量)置 NaN。"""
    px = p.raw("twap_open10").copy()
    dead = _one_word_up(p) | _one_word_down(p) | (p.raw("amt_open10") <= 0)
    return px.mask(dead)


def _twap_tail8_f(p):
    px = p.raw("twap_tail8").copy()
    dead = _one_word_up(p) | _one_word_down(p) | (p.raw("amt_tail8") <= 0)
    return px.mask(dead)


def _universe(p):
    """决策池:**只用 T 日及之前的信息**,不含任何前视。
    上市满 N 日、20 日日均成交额达标、当日有成交。"""
    return (_valid(p)
            & (_days_listed(p) >= MIN_LIST_DAYS)
            & (_adv20(p) >= MIN_ADV20)).fillna(False)


def _universe_tradable(p):
    """兼容别名 = 决策池。真正的成交约束在 fwd_ret_exec / backtest 中体现,
    不得写进股票池(否则用 T+1 是否涨停筛 T 日的票 = 前视)。"""
    return _universe(p)


def _universe_tail(p):
    """尾盘口径决策池 = 基础决策池。

    审查修正(2026-08-20):此前额外叠加了 `_twap_tail8_f.notna()`,其中 amt_tail8
    是 T 日 14:50-14:57 的成交额——在 14:50 信号时点属于未来信息,等于用"事后
    是否成交得了"筛选样本(与"用 T+1 涨停筛 T 日票"同构)。尾盘窗口的可成交性
    只允许出现在 fwd_ret_exec 与回测层(未成交记 0),不得进入决策池。
    注:经 TailPanel 使用时,_universe 内的 amount/adv20 已被自动滞后一天,无日内前视。"""
    return _universe(p)


def _turnover_proxy(p):
    """换手代理:当日成交额 / 20 日日均成交额(无股本数据,只能用相对量)。"""
    return p.raw("amount") / _adv20(p)


def _overnight(p):
    return p.raw("open_auction_px") / p.raw("close_px").shift(1) - 1.0


def _px_1450(p):
    """14:49 收盘价(由开盘集合竞价价与 pre_ret_intraday 还原),尾盘口径下的"当前价"。"""
    return p.raw("open_auction_px") * (1.0 + p.raw("pre_ret_intraday"))


_DERIVED_FN = {
    "prev_close": _prev_close, "ret_c2c": _ret_c2c, "valid": _valid,
    "days_listed": _days_listed, "adv20": _adv20, "limit_up": _limit_up,
    "limit_down": _limit_down, "one_word_up": _one_word_up, "one_word_down": _one_word_down,
    "twap_open10_f": _twap_open10_f, "twap_tail8_f": _twap_tail8_f,
    "universe": _universe, "universe_tradable": _universe_tradable,
    "universe_tail": _universe_tail, "turnover_proxy": _turnover_proxy,
    "overnight": _overnight, "px_1450": _px_1450,
    "is_st": _is_st, "market_cap": _market_cap, "log_cap": _log_cap,
}


class TailPanel(Panel):
    _is_tail = True     # 口径判据:用属性而非 isinstance,使 SealedTailPanel 等子类也能正确分派

    """尾盘口径面板:**只暴露 T 日 14:50 前可得的信息**。

    - 存在 pre_X 的字段 -> 返回 pre_X(T 日 09:30-14:49 聚合)
    - 其余全日字段     -> 返回 X.shift(1)(T-1 日已收盘,可得)
    - close_px         -> px_1450(T 日 14:49 价)
    这样任何用全日口径写的因子代码,套在 TailPanel 上就自动变成无前视的尾盘版本。
    """

    _NO_SHIFT = {"open_auction_px", "twap_open10", "amt_open10", "amt_open_auction"}

    def raw(self, name: str) -> pd.DataFrame:
        if name == "close_px":
            return self.derived("px_1450")
        if name in ("twap_tail8", "amt_tail8"):
            return super().raw(name)               # T 日尾盘成交窗口本身,14:50 时点为未来但仅用于成交
        pre = f"pre_{name}"
        if (self.dir / f"{pre}.parquet").exists():
            return super().raw(pre)
        if name in self._NO_SHIFT:
            return super().raw(name)
        return super().raw(name).shift(1)

    def derived(self, name: str) -> pd.DataFrame:
        if name == "px_1450":
            if "px_1450" not in self._derived:
                self._derived["px_1450"] = (super().raw("open_auction_px") *
                                            (1.0 + super().raw("pre_ret_intraday")))
            return self._derived["px_1450"]
        return super().derived(name)
