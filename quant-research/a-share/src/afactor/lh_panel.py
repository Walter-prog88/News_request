"""2005-2025 含退市股日线面板(前缀 lh = long history)。构建见 scripts/lh_build_panel.py,验收见 research/LH_DATA_QA.md。

数据源(两源拼接,逐股主源见 code_info["src"]):
  价格 / 量额 = 新浪日线(不复权 OHLC + 新浪后复权因子;停牌日新浪无行,构建时补成停牌行),全部股票同一口径;
  换手率:现存股 = 新浪 成交量 / 流通股本;退市股 = baostock turn(新浪退市股无股本表);
  isST / 上市区间:退市股取 baostock;现存股 ST 见第 5 条。
  新浪无数据的少数股票整只退回 baostock(src = baostock / baostock_fallback)。两源逐日对比见 QA 报告。

接口尽量与 afactor.panel.Panel 一致,使 strategy.run_weights / simulate /
benchmark 可以直接跑:

    from afactor.lh_panel import LongPanel
    p = LongPanel()                          # 2005-01-04 .. 2025-12-31,约 5100 x 5475
    p.raw("close_px"); p["universe"]; p.dates; p.codes

与项目 Panel(2020-2025 分钟聚合)的**口径差异**(用之前必须知道):
  1. twap_open10 **不是开盘 10 分钟 TWAP,而是日开盘价**(日线没有分钟数据)。
     交易日 = 后复权开盘价(含开盘集合竞价);上市期间的停牌日 = 最近一个交易日的后复权收盘价(估值价),
     使引擎对停牌中的持仓按"停牌前收盘 -> 复牌开盘"记入跳空(项目 Panel 停牌日为 NaN,引擎把这段跳空记 0)。
     suspend_fill=False 时恢复 Panel 语义(停牌日 NaN)。日开盘价相对真实 open10 TWAP 的误差分布见 QA 报告。
  2. amt_open10 = 全日成交额 amount x 固定比例(只用于参与率上限,不是实测窗口成交额)。比例取项目 Panel
     2020-2025 的 amt_open10 / amount 中位数(_meta.json):open10_ratio="pool"(默认)= 决策池股票-日中位
     (预注册 §6.3 的 0.114 口径),"all" = 全部股票-日中位,也可直接给数值(如敏感性 0.061)。
  3. 价格是**后复权**(Panel 是前复权);收益率口径相同,但绝对价格水平不同,禁用绝对价格类因子。
     不复权价在 close_raw;涨跌停按不复权价精确判定(规则见 scripts/lh_build_panel.band_rules)。
  4. 一字板 one_word_up/down = 不复权 最高 == 最低 且 后复权日收益 > +4.5% / < -4.5%(**不依赖 ST 身份**,
     锁板即不可成交;两源同一规则)。精确口径(收盘 = 当日涨/跌停价且振幅 < 0.4%)保留为 ow_lim_up/ow_lim_down。
     另有 open_at_up/open_at_dn = 不复权开盘价即在涨/跌停价(预注册 §6.2 的不可成交主规则要用;涨跌停价依赖 ST 身份),
     引擎的 dead 默认只看 one_word_up/down,用 §6.2 规则时由策略层的适配器替换。
  5. is_st:baostock 源股票 = 当日 isST;新浪源股票 = baostock isST(已抓到的)> 项目 st_info_latest.csv
     (2016-08-09 起)> 视为非 ST。st_known 标记该股该日 ST 身份是否有来源(False = 按非 ST 处理的近似)。
  6. 退市:最后一行之后价格为 NaN。引擎对 NaN 价格的持仓记收益 0、永远卖不掉(按最后价格估值滞留)——
     退市股的处理(强制按最后价格出清 / 按损失计)要由策略层自行决定,本类只提供 last_trade / delisted 标记。
  7. vwap = amount / volume 还原到后复权(日均价),供 vwap1 口径;日频均价口径的自相关问题同 Panel。
  8. 市值:float_mv = 不复权收盘 x 成交量 / (换手率/100)(新浪源即 不复权收盘 x 新浪流通股本),是**流通市值估计**,
     不是 Panel 的 market_cap(总市值)。
不提供尾盘口径(无分钟数据)。只含 <= 2025-12-31 的数据(2026 封存不碰)。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from afactor.config import CACHE, MIN_ADV20, MIN_LIST_DAYS, RESEARCH_END

LH_DIR = CACHE / "long_daily" / "panel"
_STORED_FLOAT = ("open_px", "high_px", "low_px", "close_px", "preclose_px", "close_raw", "volume", "amount",
                 "turn", "pct_chg", "ret", "twap_open10", "days_listed", "lim_up_rate", "lim_dn_rate")
_STORED_BOOL = ("listed", "trade", "suspended", "is_st", "limit_up", "limit_down", "one_word_up",
                "one_word_down", "no_limit_day", "open_at_up", "open_at_dn", "ow_lim_up", "ow_lim_down",
                "out_band", "st_known")


class LongPanel:
    """2005-2025 日线面板(后复权,含退市股)。按需加载,字段见模块文档。

    start/end: 只截取部分日期(滚动量在截取起点重新起算;做分窗口统计请用全面板 + date_mask)。
    min_adv:   派生量 universe 的 20 日日均成交额门槛(默认与 Panel 相同 2000 万元;
               注意 strategy.universe(p, version) 用的是 VERSIONS[version]['min_adv'],不读这个参数)。
    suspend_fill: True(默认)停牌日 twap_open10 = 最近收盘估值价;False = NaN(Panel 语义)。
    open10_ratio: amt_open10 / amount 的比例,"pool"(默认,决策池中位)/ "all"(全部股票-日中位)/ 数值。
    """

    _is_tail = False
    sealed = False

    def __init__(self, start=None, end=None, min_adv: float = MIN_ADV20, suspend_fill: bool = True,
                 panel_dir: Path | str | None = None, open10_ratio: str | float = "pool"):
        self.dir = Path(panel_dir) if panel_dir is not None else LH_DIR
        if not (self.dir / "_meta.json").exists():
            raise FileNotFoundError(f"长历史面板不存在: {self.dir}(先跑 scripts/lh_fetch_daily.py 与 lh_build_panel.py)")
        self.meta = json.loads((self.dir / "_meta.json").read_text())
        self.start = pd.Timestamp(start) if start is not None else None
        end = pd.Timestamp(end) if end is not None else pd.Timestamp(RESEARCH_END)
        if end > pd.Timestamp(RESEARCH_END):
            raise PermissionError(f"长历史面板只含 <= {RESEARCH_END} 的数据(2026 封存)")
        self.end = end
        self.min_adv = float(min_adv)
        self.suspend_fill = bool(suspend_fill)
        self._open10_ratio = open10_ratio
        self._cache: dict[str, pd.DataFrame] = {}
        self._derived: dict[str, pd.DataFrame] = {}

    # ---------- 原料 ----------
    def _load(self, name: str) -> pd.DataFrame:
        f = self.dir / f"{name}.parquet"
        if not f.exists():
            raise KeyError(f"原料不存在: {name}")
        df = pd.read_parquet(f)
        df = df.loc[self.start: self.end] if self.start is not None else df.loc[: self.end]
        if name in _STORED_FLOAT:
            df = df.astype(np.float32)
        return df

    def raw(self, name: str) -> pd.DataFrame:
        if name in self._cache:
            return self._cache[name]
        if name in _STORED_FLOAT or name in _STORED_BOOL:
            if name == "twap_open10" and not self.suspend_fill:
                df = self._load("open_px")
            else:
                df = self._load(name)
        elif name == "amt_open10":
            df = (self.raw("amount") * np.float32(self.open10_amt_ratio)).astype(np.float32)
        elif name == "vwap":
            with np.errstate(divide="ignore", invalid="ignore"):
                v = self.raw("amount") / self.raw("volume").where(self.raw("volume") > 0)
                df = (v * self.raw("close_px") / self.raw("close_raw")).astype(np.float32)
        elif name == "rng":
            df = ((self.raw("high_px") - self.raw("low_px")) / self.raw("preclose_px")).astype(np.float32)
        else:
            raise KeyError(f"原料不存在: {name}")
        self._cache[name] = df
        return df

    def __getitem__(self, name: str) -> pd.DataFrame:
        if name in self._derived or name in _DERIVED_FN:
            return self.derived(name)
        return self.raw(name)

    @property
    def open10_amt_ratio(self) -> float:
        r = self._open10_ratio
        if isinstance(r, (int, float)) and not isinstance(r, bool):
            return float(r)
        if r == "pool" and "open10_amt_ratio_pool" in self.meta:
            return float(self.meta["open10_amt_ratio_pool"])
        if r in ("pool", "all"):
            return float(self.meta["open10_amt_ratio"])
        raise ValueError(f"open10_ratio={r!r}")

    @property
    def features(self) -> list[str]:
        stored = sorted(p.stem for p in self.dir.glob("*.parquet") if not p.stem.startswith("_"))
        return sorted(set(stored) | {"amt_open10", "vwap", "rng"})

    @property
    def dates(self) -> pd.DatetimeIndex:
        return self.raw("close_px").index

    @property
    def codes(self) -> pd.Index:
        return self.raw("close_px").columns

    @property
    def code_info(self) -> pd.DataFrame:
        """逐股元数据:bs_code / ipoDate / outDate / first_trade / last_trade / n_trade / n_no_limit 等,索引 = 代码。"""
        if "__code_info" not in self._derived:
            m = pd.read_parquet(self.dir / "_codes.parquet").set_index("code")
            self._derived["__code_info"] = m.reindex(self.codes)
        return self._derived["__code_info"]

    # ---------- 派生量 ----------
    def derived(self, name: str) -> pd.DataFrame:
        if name not in self._derived:
            self._derived[name] = _DERIVED_FN[name](self)
        return self._derived[name]

    # ---------- 前瞻收益(只用于 IC / 分层的理论收益;本类自身不计算任何策略收益)----------
    def fwd_ret(self, horizon: int = 1, mode: str = "open") -> pd.DataFrame:
        """mode='open': twap_open10[T+1+h] / twap_open10[T+1] - 1(这里 = 日开盘价,停牌估值价)
        mode='c2c' : close_px[T+h] / close_px[T] - 1。无尾盘口径。"""
        if mode == "open":
            px = self.raw("twap_open10")
            return px.shift(-(1 + horizon)) / px.shift(-1) - 1.0
        if mode == "c2c":
            px = self.raw("close_px")
            return px.shift(-horizon) / px - 1.0
        raise ValueError(f"LongPanel 不支持 mode={mode}(没有分钟数据)")

    def fwd_ret_exec(self, horizon: int = 1, mode: str = "open") -> pd.DataFrame:
        theo = self.fwd_ret(horizon, mode)
        if mode != "open":
            return theo
        ok = self["twap_open10_f"].notna().shift(-1)
        return theo.where(ok.fillna(False), 0.0).where(theo.notna())

    # ---------- 成本模型 ----------
    def cost_model(self, impact_y: float = 1.0, spread_mult: float = 1.0, window: int = 20,
                   min_periods: int = 10, tick: float | None = None, spread_method: str = "tick"):
        """返回 impact.CostModel。impact.tick_floor 读的是 newdata.raw_close(只覆盖 2020+),
        这里先按本面板的不复权收盘价 close_raw 算好 _base_panels 放进缓存,公式与 impact._base_panels 逐项相同。"""
        from afactor import impact as IM
        tick = IM.TICK if tick is None else float(tick)
        key = f"__eng_cost__{window}_{min_periods}_{tick}_{spread_method}"
        if key not in self._derived:
            close = self.raw("close_px").astype(np.float64)
            rawc = self.raw("close_raw").astype(np.float64)
            px = rawc.where(rawc > 0).fillna(close.where(close > 0))
            floor = (tick / px)
            if spread_method == "tick":
                S = floor.where(close.notna())
            else:
                prod = IM.ar_products(self)
                if spread_method == "ar_window":
                    S = np.sqrt(prod.rolling(window, min_periods=min_periods).mean().clip(lower=0.0))
                elif spread_method == "ar_2day":
                    S = np.sqrt(prod.clip(lower=0.0)).rolling(window, min_periods=min_periods).mean()
                else:
                    raise ValueError(spread_method)
                S = np.fmax(S, floor).where(S.notna())
            floor_use = floor.shift(1).to_numpy(np.float64)
            S_use = IM._fill_rows(S.shift(1).to_numpy(np.float64), "median", IM._FALLBACK_SPREAD)
            S_use = np.where(np.isfinite(floor_use), np.fmax(S_use, floor_use), S_use)
            sig = IM._fill_rows(IM.daily_sigma(self, window, min_periods).shift(1).to_numpy(np.float64),
                                "median", IM._FALLBACK_SIGMA)
            adv = IM._fill_rows(IM.daily_adv(self, window, min_periods).shift(1).to_numpy(np.float64),
                                "q05pos", IM._FALLBACK_ADV)
            self._derived[key] = dict(hs=(S_use / 2.0).astype(np.float32), sigma=sig.astype(np.float32),
                                      adv=adv.astype(np.float32))
        return IM.CostModel(self, impact_y=impact_y, spread_mult=spread_mult, window=window,
                            min_periods=min_periods, tick=tick, spread_method=spread_method)


# ================= 派生量 =================
def _prev_close(p):
    return p.raw("close_px").shift(1)


def _ret_c2c(p):
    """Panel 兼容:收盘对收盘,停牌日与复牌日为 NaN(复牌跳空不在内)。"""
    c = p.raw("close_px")
    return c / c.shift(1) - 1.0


def _ret_hold(p):
    """持有收益:停牌日 0、复牌日含跨停牌跳空(逐行计算;新浪源的停牌行由构建补出)。"""
    return p.raw("ret")


def _valid(p):
    return (p.raw("amount") > 0) & p.raw("close_px").notna()


def _days_listed(p):
    """自上市日起的交易日序号(上市首日 = 1),只在当日有成交时有值(与 Panel 一样 where(valid))。
    Panel 是"面板起点以来的有效交易日计数";这里按真实上市日,2005 年前上市的按工作日数近似。"""
    return p.raw("days_listed").where(_valid(p))


def _adv20(p):
    return p.raw("amount").rolling(20, min_periods=10).mean()


def _flag(name):
    def f(p):
        return p.raw(name).astype(bool)
    return f


def _twap_open10_f(p):
    px = p.raw("twap_open10").copy()
    dead = p["one_word_up"] | p["one_word_down"] | ~(p.raw("amt_open10") > 0)
    return px.mask(dead)


def _universe(p):
    """决策池:只用 T 日及之前的信息。上市满 MIN_LIST_DAYS、20 日日均成交额 >= min_adv、当日有成交。"""
    return (_valid(p) & (_days_listed(p) >= MIN_LIST_DAYS) & (_adv20(p) >= p.min_adv)).fillna(False)


def _turnover_proxy(p):
    return p.raw("amount") / _adv20(p)


def _overnight(p):
    """开盘价 / 前一交易日收盘 - 1(开盘价含开盘集合竞价;跨停牌时为 NaN)。"""
    return p.raw("open_px") / p.raw("close_px").shift(1) - 1.0


def _float_mv(p):
    """流通市值估计(元)= 不复权收盘 x 成交量 / (换手率%/100);换手率为 0/缺失的日子按前值填充(只在上市期间)。"""
    turn = p.raw("turn")
    with np.errstate(divide="ignore", invalid="ignore"):
        fs = p.raw("volume") / (turn.where(turn > 0) / 100.0)
    fs = fs.where(np.isfinite(fs)).ffill().where(p.raw("listed").astype(bool))
    return (p.raw("close_raw").ffill().where(p.raw("listed").astype(bool)) * fs).astype(np.float32)


def _log_float_mv(p):
    return np.log(p["float_mv"].where(p["float_mv"] > 0))


def _delisted(p):
    """该股已退市(在 2025-12-31 前有 outDate)且日期晚于最后一行(退市股都是 baostock 源)。"""
    info = p.code_info
    last = pd.to_datetime(info["last_row"])
    out = info["outDate"].fillna("").astype(str)
    is_del = (out != "") & (out <= RESEARCH_END)
    d = p.dates.to_numpy()[:, None]
    m = (d > last.to_numpy()[None, :]) & is_del.to_numpy()[None, :]
    return pd.DataFrame(m, index=p.dates, columns=p.codes)


def _last_trade_day(p):
    """该日是这只**退市股**的最后一个交易日(策略层做退市出清时用)。"""
    info = p.code_info
    out = info["outDate"].fillna("").astype(str)
    is_del = ((out != "") & (out <= RESEARCH_END)).to_numpy()
    lt = pd.to_datetime(info["last_trade"]).to_numpy()
    d = p.dates.to_numpy()[:, None]
    return pd.DataFrame((d == lt[None, :]) & is_del[None, :], index=p.dates, columns=p.codes)


_DERIVED_FN = {
    "prev_close": _prev_close, "ret_c2c": _ret_c2c, "ret_hold": _ret_hold, "valid": _valid,
    "days_listed": _days_listed, "adv20": _adv20,
    "limit_up": _flag("limit_up"), "limit_down": _flag("limit_down"),
    "one_word_up": _flag("one_word_up"), "one_word_down": _flag("one_word_down"),
    "is_st": _flag("is_st"), "suspended": _flag("suspended"), "listed": _flag("listed"),
    "no_limit_day": _flag("no_limit_day"), "open_at_up": _flag("open_at_up"), "open_at_dn": _flag("open_at_dn"),
    "ow_lim_up": _flag("ow_lim_up"), "ow_lim_down": _flag("ow_lim_down"), "out_band": _flag("out_band"),
    "twap_open10_f": _twap_open10_f, "universe": _universe, "universe_tradable": _universe,
    "turnover_proxy": _turnover_proxy, "overnight": _overnight,
    "float_mv": _float_mv, "log_float_mv": _log_float_mv,
    "delisted": _delisted, "last_trade_day": _last_trade_day,
}
