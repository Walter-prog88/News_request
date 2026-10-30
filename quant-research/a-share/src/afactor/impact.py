"""交易成本模型:佣金 + 印花税 + 半价差(一跳下限;Abdi-Ranaldo 2017 作上界)+ 平方根冲击(2026-10-10,前缀 eng)。

为什么要有它:旧引擎只扣"单边 1bp + 卖出印花税",对 1000 万以下、ADV>=2000 万的池子勉强说得过去,
但组合层要回答"容量多大""优化器该不该为了 2bp 的预期收益换手"时,必须有一个随资金量增长的成本项。
本模块给出**引擎与优化器共用的同一个成本模型**:

    单只股票单边成本(占组合净值的比例)
        = |Δw| * (佣金 1bp + 半价差 hs_i [+ 卖出印花税])  +  c_i * |Δw|^1.5
    其中 c_i = Y * σ_i * sqrt(capital * nav / ADV_i)
    (等价于冲击成本率 = Y * σ_i * sqrt(Q / ADV_i),Q = |Δw| * capital * nav 为成交金额,元)

## 参数与来源(**都是文献量级,没有用真实成交数据校准**)
- Y:平方根冲击系数。基准 Y=1.0,敏感性 0.5 / 1.5(预注册,不按回测结果挑)。
  量级参考:Almgren, Thum, Hauptmann & Li (2005, Risk) 美股大单 η≈0.3~1(3/5 次方形式);
  Tóth et al. (2011, PRX) 平方根定律 Y 为 O(1);Frazzini, Israel & Moskowitz (2018) 实盘
  成交数据的冲击明显低于多数学术模型。A 股小盘的盘口更薄,取 Y=1 偏保守,0.5~1.5 覆盖分歧。
- σ_i:20 日日收益(close_px 收盘对收盘,前复权)标准差,min_periods=10;
- ADV_i:20 日日均**全日**成交额 amount(元),min_periods=10(与决策池 adv20 同一公式);
- hs_i:半价差 = 全价差 / 2 × spread_mult。全价差默认 spread_method="tick" = 一跳价差
  0.01 元 / 未复权收盘价(**下界**);"ar_window" = Abdi-Ranaldo 逐股估计(与一跳取大,**偏高的上界**)。
  为什么不是任务书原定的 AR 逐股估计做默认,见下节"AR 有效性诊断";
- 佣金 COST_BPS_ONEWAY=1bp、印花税(2023-08-28 前 10bp、后 5bp,仅卖出)与 config 完全一致。

## 时序(无前视)
所有估计量在**成交日 te 只用到 te-1 收盘为止的数据**:面板行 te 存的是截至 te-1 的估计
(代码里统一 `.shift(1)`)。主口径 te=t+1,所以等价于信号日 t 收盘可得;尾盘口径 te=t,
用到 t-1 收盘(比可得信息还早半天,偏保守)。tests/test_engine_regression.py 用"截断未来数据后
估计量不变"的方式验证。

## Abdi-Ranaldo (2017, RFS) 估计量与 spread_method
    c_t = ln close_t,  η_t = (ln high_t + ln low_t) / 2
    两日乘积 s²_t = 4 (c_t - η_t)(c_t - η_{t+1})          —— 要到 t+1 收盘才可知
    "tick"     :**默认**。只用报价单位下限 tick / 未复权收盘价(一跳价差)
    "ar_window":逐股 20 日两日乘积均值 -> 负值截零 -> 开方 -> 与一跳取大(任务书原定口径)
    "ar_2day"  :逐个两日估计先截零开方再 20 日平均 -> 与一跳取大(AR 的 two-day corrected)
原料 high_px/low_px 是分钟 max(open,close)/min(open,close)(见 scripts/build_hilo.py),
比真实最高/最低价略窄;close_px 是 15:00 收盘集合竞价价;前复权价格的比例不受复权影响。

### AR 有效性诊断(2026-10-10;判据只看估计量本身,与任何策略收益无关)
在本数据上 AR **不可识别**(数字见 research/ledger/eng_spread_by_size.csv 与 eng_spread_ar_diagnostics.csv):
  1. 2021-2025 V1000 池按市值 / ADV 十分位合并的两日乘积均值**全部为负**(每天桶内截面均值的
     时间序列 Newey-West t 在 -2.1 ~ -4.7 之间)。AR 的期望值 = 价差²,负均值意味着价差信号被
     某种负偏差淹没,合并后按"负值截零"就是 0,只剩一跳下限;
  2. 逐股 20 日估计 49.7% 被截成 0(60 日 54%、120 日 60%,窗口越长越多);剩下的正值与 -ln ADV
     的截面秩相关日均 -0.016(方向还是反的)、与一跳价差 -0.045 —— 正值部分是截断后的噪声,不是价差;
  3. ADV 十分位桶内横截面合并也救不回来:市场共同成分在截面上不分散,20 日桶均值仍有 18%~39% 的天为正;
     拉长到 250 日后最大 ADV 桶 0% 的天为正、最小桶 11%。
可能原因:收盘价是集合竞价价(没有买卖价反弹,违反 AR 的识别假设)、high/low 由分钟 open/close
近似、日频收益有非鞅成分。逐股版本被"负值截零"抬高到半价差中位 13~18bp、均值 27~34bp(按市值
十分位),且与流动性无关;作为成本会系统性高估,作为优化器的线性系数还会制造随机的伪偏好。
因此默认用"AR 合并后为负 -> 截零 -> 一跳下限"的结果,即一跳价差(下界);ar_window 作为上界敏感性。
**建议下游把价差当区间报告**:spread_method="tick"(下界)/ spread_mult=2(两跳)/ "ar_window"(上界)。

## 缺失值兜底(都在 te 行内完成,不用未来数据)
σ 缺失 -> 当日截面中位数(再缺 -> 0.025);ADV 缺失或 <=0 -> 当日截面正值的 5% 分位
(再缺 -> 1e7 元,偏保守 = 高冲击);价差缺失 -> 当日截面中位数(再缺 -> 全价差 20bp)。
未复权收盘价缺失(约 0.6% 的股票日)-> 用前复权收盘价算下限(前复权价 <= 未复权价,下限偏高,保守)。

## 已知局限
- 平方根定律的 ADV 是全日成交额;open10 口径实际在开盘 10 分钟窗口内成交,窗口流动性只占全日一小部分,
  用全日 ADV 的平方根冲击**很可能低估**集中开盘成交的冲击。参与率上限(窗口成交额的 10%)是另一道闸。
- 冲击只算临时冲击,不建模永久冲击对后续交易的影响,也不建模买卖不对称。
- 数据不含已退市股(幸存者偏差),小盘低价股的价差/冲击估计样本偏向"活下来的"股票。
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from afactor.config import (COST_BPS_ONEWAY, STAMP_TAX_BPS_BEFORE, STAMP_TAX_BPS_AFTER,
                            STAMP_TAX_CHANGE_DATE, assert_not_sealed)

TICK = 0.01                 # A 股最小报价单位(元)
PREREG_Y = (0.5, 1.0, 1.5)  # 预注册冲击系数:基准 1.0,敏感性 0.5 / 1.5
_FALLBACK_SPREAD = 0.0020   # 全价差兜底 20bp(只在截面全缺时用,实际仅出现在 2020 年初预热段)
_FALLBACK_SIGMA = 0.025
_FALLBACK_ADV = 1e7

# 估计面板缓存挂在 Panel 对象自己的 _derived 上(随面板生灭);不用以 id(panel) 为键的模块字典,
# 否则面板被回收后 id 复用会拿到别的面板的估计。


# ------------------------------------------------------------------ 估计量
def ar_spread(p, window: int = 20, min_periods: int = 10, tick: float = TICK,
              method: str = "ar_window", floor: bool = True) -> pd.DataFrame:
    """Abdi-Ranaldo 有效**全价差**(小数),按"截至 d 日收盘可得"索引在 d 行。

    method="ar_window":逐股窗口内两日乘积取均值后截零开方;
    method="ar_2day"  :两日估计先截零开方再取均值(AR two-day corrected);
    method="tick"     :只用报价单位下限(CostModel 的默认,理由见模块文档)。
    floor=True 时与 tick / 未复权收盘价(d 日)取大。"""
    if getattr(p, "sealed", False):
        raise PermissionError("成本模型只在研究期面板上估计")
    assert_not_sealed(p.dates)
    if method == "tick":
        return tick_floor(p, tick).where(p.raw("close_px").notna())
    prod = ar_products(p)
    if method == "ar_window":
        m = prod.rolling(window, min_periods=min_periods).mean()
        S = np.sqrt(m.clip(lower=0.0))
    elif method == "ar_2day":
        S = np.sqrt(prod.clip(lower=0.0)).rolling(window, min_periods=min_periods).mean()
    else:
        raise ValueError(method)
    if floor:
        S = np.fmax(S, tick_floor(p, tick)).where(S.notna())
    return S


def ar_products(p) -> pd.DataFrame:
    """AR 两日乘积 4(c_{d-1}-η_{d-1})(c_{d-1}-η_d),在 d 日收盘才完整可得,故索引在 d 行。"""
    c = np.log(p.raw("close_px").astype(np.float64))
    eta = (np.log(p.raw("high_px").astype(np.float64)) + np.log(p.raw("low_px").astype(np.float64))) / 2.0
    c1 = c.shift(1)
    return 4.0 * (c1 - eta.shift(1)) * (c1 - eta)


def tick_floor(p, tick: float = TICK) -> pd.DataFrame:
    """最小报价单位对应的全价差下限 = tick / 未复权收盘价(d 日)。未复权价缺失时用前复权收盘价(偏保守)。"""
    from afactor import newdata as ND
    assert_not_sealed(p.dates)
    raw = ND.raw_close(p.dates, p.codes).astype(np.float64)        # reindex 到研究期日期,2026 行不进入计算
    adj = p.raw("close_px").astype(np.float64)
    px = raw.where(raw > 0).fillna(adj.where(adj > 0))
    return tick / px


def daily_sigma(p, window: int = 20, min_periods: int = 10) -> pd.DataFrame:
    """20 日日收益标准差(close_px 收盘对收盘,前复权),索引在 d 行 = 截至 d 收盘。"""
    c = p.raw("close_px").astype(np.float64)
    return (c / c.shift(1) - 1.0).rolling(window, min_periods=min_periods).std()


def daily_adv(p, window: int = 20, min_periods: int = 10) -> pd.DataFrame:
    """20 日日均全日成交额(元),索引在 d 行 = 截至 d 收盘。公式与 panel._adv20 相同。"""
    return p.raw("amount").astype(np.float64).rolling(window, min_periods=min_periods).mean()


def _fill_rows(a: np.ndarray, how: str, const: float) -> np.ndarray:
    """逐行兜底:how='median' 用当日截面中位数,'q05pos' 用当日正值 5% 分位;再缺用常数。"""
    out = a.copy()
    bad = ~np.isfinite(out) | ((out <= 0) if how == "q05pos" else False)
    rows = np.flatnonzero(bad.any(axis=1))
    for i in rows:
        row = a[i]
        ok = np.isfinite(row) & ((row > 0) if how == "q05pos" else True)
        if ok.sum() >= 20:
            v = float(np.median(row[ok])) if how == "median" else float(np.quantile(row[ok], 0.05))
        else:
            v = const
        out[i, bad[i]] = v
    return out


def _base_panels(p, window: int, min_periods: int, tick: float, method: str) -> dict:
    if getattr(p, "_is_tail", False):
        # TailPanel 的 raw() 会把全日字段滞后一天、把 close_px 换成 14:49 价,再 shift(1) 就成了截至 te-2。
        # 估计量已按成交日 te(截至 te-1)对齐,尾盘口径同样用主口径 Panel() 构建即可。
        raise ValueError("CostModel 请用主口径 Panel() 构建(尾盘口径同样适用,行 te = 截至 te-1 的估计)")
    key = f"__eng_cost__{window}_{min_periods}_{tick}_{method}"
    if key in p._derived:
        return p._derived[key]
    S = ar_spread(p, window, min_periods, tick, method, floor=True)
    floor_use = tick_floor(p, tick).shift(1).to_numpy(np.float64)
    # 行 te = 截至 te-1 的估计(shift(1)),再兜底
    S_use = _fill_rows(S.shift(1).to_numpy(np.float64), "median", _FALLBACK_SPREAD)
    S_use = np.where(np.isfinite(floor_use), np.fmax(S_use, floor_use), S_use)
    sig = _fill_rows(daily_sigma(p, window, min_periods).shift(1).to_numpy(np.float64), "median", _FALLBACK_SIGMA)
    adv = _fill_rows(daily_adv(p, window, min_periods).shift(1).to_numpy(np.float64), "q05pos", _FALLBACK_ADV)
    out = dict(hs=(S_use / 2.0).astype(np.float32), sigma=sig.astype(np.float32), adv=adv.astype(np.float32))
    p._derived[key] = out
    return out


# ------------------------------------------------------------------ 成本模型
class CostModel:
    """引擎与优化器共用的成本模型。

    用法:
        cm = CostModel(p, impact_y=1.0)                    # 基准:Y=1,一跳价差
        cm_hi = CostModel(p, impact_y=1.5, spread_method="ar_window")   # 悲观敏感性
        coef = cm.cost_coefficients(te, capital=1e8, nav=1.0)
        # 组合成本(占净值)= Σ coef['lin_buy']*buy_i + Σ coef['lin_sell']*sell_i + Σ coef['c15']*|Δw_i|^1.5
        run_weights(W, p, "open10", 1e8, cost_model=cm)    # 引擎内部调用 cm.extra_cost(...)

    impact_y=0 且 spread_mult=0 时 extra_cost 恒为 0.0,引擎结果与 cost_model=None 逐位一致。
    te 可以是 p.dates 的整数行号,也可以是成交日 Timestamp。
    """

    def __init__(self, p, impact_y: float = 1.0, spread_mult: float = 1.0, window: int = 20,
                 min_periods: int = 10, tick: float = TICK, spread_method: str = "tick"):
        self.p = p
        self.impact_y = float(impact_y)
        self.spread_mult = float(spread_mult)
        self.window, self.min_periods, self.tick, self.spread_method = window, min_periods, tick, spread_method
        base = _base_panels(p, window, min_periods, tick, spread_method)
        self._hs, self._sigma, self._adv = base["hs"], base["sigma"], base["adv"]
        self.dates = p.dates
        self.codes = p.codes
        self._stamp_switch = int(np.searchsorted(p.dates.values, np.datetime64(STAMP_TAX_CHANGE_DATE)))
        self.commission = COST_BPS_ONEWAY * 1e-4

    # ---- 索引 ----
    def _te(self, te) -> int:
        if isinstance(te, (int, np.integer)):
            return int(te)
        return int(self.dates.get_loc(pd.Timestamp(te)))

    def te_of(self, t: int, mode: str) -> int:
        """信号日行号 t -> 成交日行号 te = t + lag。"""
        from afactor.strategy import EXEC_MODES
        return int(t) + int(EXEC_MODES[mode]["lag"])

    # ---- 分项 ----
    def stamp(self, te) -> float:
        te = self._te(te)
        return (STAMP_TAX_BPS_BEFORE if te < self._stamp_switch else STAMP_TAX_BPS_AFTER) * 1e-4

    def half_spread(self, te) -> np.ndarray:
        """成交日 te 适用的半价差(小数),= 截至 te-1 的全价差估计 / 2 × spread_mult。"""
        return self._hs[self._te(te)].astype(np.float64) * self.spread_mult

    def sigma(self, te) -> np.ndarray:
        return self._sigma[self._te(te)].astype(np.float64)

    def adv(self, te) -> np.ndarray:
        return self._adv[self._te(te)].astype(np.float64)

    def impact_coef(self, te, capital: float, nav: float = 1.0) -> np.ndarray:
        """c_i,使冲击成本(占净值)= c_i * |Δw_i|^1.5。"""
        te = self._te(te)
        return self.impact_y * self.sigma(te) * np.sqrt(float(capital) * float(nav) / self.adv(te))

    def impact_rate(self, te, q_yuan) -> np.ndarray:
        """成交金额 Q(元,标量或 N 向量)对应的单边冲击成本率(小数)= Y σ sqrt(Q/ADV)。"""
        te = self._te(te)
        return self.impact_y * self.sigma(te) * np.sqrt(np.asarray(q_yuan, np.float64) / self.adv(te))

    # ---- 给优化器 ----
    def cost_coefficients(self, te, capital: float, nav: float = 1.0) -> dict:
        """返回成交日 te 每只股票的成本系数(都以占组合净值的比例计):
            lin_buy  = 佣金 + 半价差
            lin_sell = 佣金 + 半价差 + 印花税
            c15      = Y σ sqrt(capital*nav/ADV)   (冲击 = c15 * |Δw|^1.5)
        以及原料 half_spread / sigma / adv / stamp / commission。
        引擎在同一 te 上的成本与 Σ lin*|Δw| + Σ c15*|Δw|^1.5 数学恒等(测试 test_cost_coefficients_match_engine)。"""
        te = self._te(te)
        hs = self.half_spread(te)
        st = self.stamp(te)
        lin_buy = self.commission + hs
        return dict(lin_buy=lin_buy, lin_sell=lin_buy + st, c15=self.impact_coef(te, capital, nav),
                    half_spread=hs, sigma=self.sigma(te), adv=self.adv(te), stamp=st,
                    commission=self.commission, te=te)

    # ---- 给引擎 ----
    def extra_cost(self, te, abs_delta: np.ndarray, capital: float, nav: float) -> tuple[float, float]:
        """在旧成本(佣金+印花税)之上的增量:(半价差成本, 冲击成本),都占净值比例。"""
        te = self._te(te)
        nz = abs_delta > 0
        if not nz.any():
            return 0.0, 0.0
        ad = abs_delta[nz]
        spread = float((self.half_spread(te)[nz] * ad).sum()) if self.spread_mult != 0 else 0.0
        impact = float((self.impact_coef(te, capital, nav)[nz] * ad ** 1.5).sum()) if self.impact_y != 0 else 0.0
        return spread, impact

    def trade_cost(self, te, delta: np.ndarray, capital: float, nav: float = 1.0) -> dict:
        """一次调仓 Δw 的成本分解(占净值比例)。"""
        te = self._te(te)
        delta = np.asarray(delta, np.float64)
        buy = np.clip(delta, 0, None); sell = np.clip(-delta, 0, None)
        spread, impact = self.extra_cost(te, np.abs(delta), capital, nav)
        comm = float((buy.sum() + sell.sum()) * self.commission)
        stp = float(sell.sum() * self.stamp(te))
        return dict(commission=comm, stamp=stp, spread=spread, impact=impact,
                    total=comm + stp + spread + impact)

    def describe(self) -> dict:
        return dict(impact_y=self.impact_y, spread_mult=self.spread_mult, window=self.window,
                    min_periods=self.min_periods, tick=self.tick, spread_method=self.spread_method,
                    commission_bps=self.commission * 1e4)
