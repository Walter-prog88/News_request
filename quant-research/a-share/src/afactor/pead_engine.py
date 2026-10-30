"""PEAD 事件组合的执行层外挂(前缀 pead,2026-10-10)。

预注册 research/PEAD_PREREG_ADDENDUM.md 第 6 节要求的执行规则,有三条是项目引擎
`strategy.simulate` 默认没有的:整手、每笔最低 5 元佣金、卖不掉的持仓**每个交易日**同一窗口重试。
本模块不改引擎(硬纪律 2),而是通过引擎已公开的两个钩子实现:

  1. target_fn(t, h, ctx)(callable 目标):
       - 以 rebalance=1 调用 simulate,由本模块自己判断调仓日:成交日 te 在交易日历中的序号
         对 5 取余 == φ(预注册第 3 节,2020-01-02 记 0;te = 信号行 t + 1,与日历序号同号);
       - 调仓日:按批次目标权重 w(S 时点已知)做整手取整,返回目标向量;
       - 非调仓日:若有"应退出但还没卖掉"的持仓(停牌/一字跌停/参与率截量),只对这些股票给 0,
         其余给 NaN(= 保持实际持仓不交易),实现"之后每个交易日的同一窗口重试,直到卖出";
         没有待卖持仓时返回 None(不交易)。买单**不重试**(预注册:买不进的留现金)。
       - 决策只用 ctx['h_dec'] / ctx['aum_dec'](按 px[te-1] 计价,决策时可得)与 t_signal 未复权收盘价。
  2. cost_model.extra_cost(te, |Δw|, capital, nav)(成本钩子):
       在引擎旧成本(每边 1bp + 卖出印花税)之上**加**:
         - 最低佣金补差:每只股票每笔 max(0, 5 元 − 1bp × 成交额);
         - 费用压力:每笔实际买入、卖出各再加 stress_bps;
         - 可选 afactor.impact.CostModel(半价差 + 平方根冲击)。
       引擎把返回的第一项记在 record 的 cost_spread、第二项记在 cost_impact;本类另在 self.log
       按成交日逐项记录(最低佣金补差 / 费用压力 / 模型半价差 / 模型冲击),分解以 log 为准。

参与率:simulate(participation_on_nav=True),单笔成交额 ≤ 10% × te 日 amt_open10(按 capital × 当前净值折算)。
资金:normalize=False —— 目标权重按原值使用,剩余为现金;超预算时引擎等比压缩买单;卖不掉的残仓占用预算。

整手(预注册第 6 节):沪深主板、创业板 100 股一手向下取整;科创板(688/689)最低 200 股、超过后 1 股递增。
手数换算价格 = t_signal(信号行 t)未复权收盘价;实际成交按 TWAP 计价(引擎权重口径)。
局限(报告里写明):
  - 引擎按权重而非股数记账:整手只约束**下单目标**;参与率截量、超预算压缩会产生非整手成交;
  - 继续持有的股票(新一批仍入选):目标与现持仓之差不足一手的价值时不交易(NaN),避免为漂移付最低佣金;
  - 未复权收盘缺失(约 0.6% 股票日)时用前复权收盘代替(前复权价 ≤ 未复权价,股数偏多,影响可忽略),计数另报。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from afactor import strategy as ST
from afactor.config import COST_BPS_ONEWAY

MIN_COMMISSION_YUAN = 5.0
LOT_MAIN = 100
STAR_MIN_SHARES = 200
REBALANCE = 5
MODE = "open10"
H_TOL = 1e-10          # 持仓存在性阈值(权重)


def star_mask(codes) -> np.ndarray:
    """科创板(688/689):最低 200 股,超过后 1 股递增。"""
    return np.array([str(c).startswith(("688", "689")) for c in codes], bool)


def lot_shares(value_yuan: np.ndarray, px: np.ndarray, star: np.ndarray) -> np.ndarray:
    """目标金额 -> 可下单股数(向下取整)。px 非正或缺失 -> 0 股。"""
    value_yuan = np.asarray(value_yuan, np.float64)
    px = np.asarray(px, np.float64)
    ok = np.isfinite(px) & (px > 0) & np.isfinite(value_yuan) & (value_yuan > 0)
    raw = np.zeros_like(value_yuan)
    raw[ok] = np.floor(value_yuan[ok] / px[ok])
    main = np.floor(raw / LOT_MAIN) * LOT_MAIN
    st = np.where(raw >= STAR_MIN_SHARES, raw, 0.0)
    return np.where(star, st, main)


class EventFees:
    """引擎成本钩子:最低佣金补差 + 费用压力 + 可选冲击模型。见模块文档。"""

    def __init__(self, min_commission: float = MIN_COMMISSION_YUAN, stress_bps: float = 0.0,
                 impact_model=None, min_trade_yuan: float = 1.0):
        self.min_commission = float(min_commission)
        self.stress_bps = float(stress_bps)
        self.impact_model = impact_model
        # 成交额 < 1 元的 |Δw| 视为浮点噪声,不收最低佣金(引擎按权重记账,这类残差不对应真实委托)
        self.min_trade_yuan = float(min_trade_yuan)
        self.log: list[tuple] = []

    def extra_cost(self, te, abs_delta: np.ndarray, capital: float, nav: float) -> tuple[float, float]:
        aum = float(capital) * float(nav)
        v = abs_delta * aum
        tr = v >= self.min_trade_yuan
        mc, nb = 0.0, 0
        if self.min_commission > 0 and tr.any():
            top = np.maximum(0.0, self.min_commission - v[tr] * COST_BPS_ONEWAY * 1e-4)
            mc = float(top.sum()) / aum
            nb = int((top > 0).sum())
        stv = self.stress_bps * 1e-4 * float(abs_delta.sum())
        sp = im = 0.0
        if self.impact_model is not None:
            sp, im = self.impact_model.extra_cost(te, abs_delta, capital, nav)
        self.log.append((int(te), int(tr.sum()), nb, mc, stv, float(sp), float(im)))
        return mc + stv + sp, im

    def describe(self) -> dict:
        return dict(min_commission=self.min_commission, stress_bps=self.stress_bps,
                    impact=(self.impact_model.describe() if self.impact_model is not None else None))

    def log_frame(self, dates: pd.DatetimeIndex) -> pd.DataFrame:
        """按**信号日**(te-1)索引的成本分解(占净值比例)。"""
        if not self.log:
            return pd.DataFrame(columns=["n_trades", "n_min_bound", "cost_min", "cost_stress",
                                         "cost_spread_cm", "cost_impact_cm"])
        a = np.array(self.log, dtype=np.float64)
        idx = dates[a[:, 0].astype(int) - 1]
        return pd.DataFrame(dict(n_trades=a[:, 1], n_min_bound=a[:, 2], cost_min=a[:, 3],
                                 cost_stress=a[:, 4], cost_spread_cm=a[:, 5], cost_impact_cm=a[:, 6]),
                            index=pd.DatetimeIndex(idx, name="date"))


def make_batch_target(batches: dict, phase: int, raw_px: np.ndarray, adj_px: np.ndarray,
                      star: np.ndarray, lots: bool = True, rebalance: int = REBALANCE):
    """批次目标 -> (target_fn, stats)。

    batches: {te(成交日行号): (idx int ndarray, w float ndarray)},w 为 S 时点目标权重(整手前)。
    raw_px / adj_px: (T, N) 未复权 / 前复权收盘价,只取信号行 t(= te-1)。
    stats 记录每个调仓日的目标仓位(整手前/后)、te 不可成交的目标、待卖持仓的天数等。"""
    N = raw_px.shape[1]
    cur = np.zeros(N, bool)                       # 最近一批的目标集合(含"保持不动")
    reb_rows: list[tuple] = []
    st = dict(pending_stock_days=0, pending_days=0, px_fallback=0, cont_hold=0, cont_trade=0)

    def _px(t, idx):
        px = raw_px[t, idx].astype(np.float64)
        bad = ~(np.isfinite(px) & (px > 0))
        if bad.any():
            st["px_fallback"] += int(bad.sum())
            px = np.where(bad, adj_px[t, idx], px)
        return px

    def fn(t, h, ctx):
        nonlocal cur
        te = int(ctx["te"])
        hd = ctx["h_dec"]
        aum = float(ctx["aum_dec"])
        if te % rebalance == phase:
            w = np.zeros(N)
            b = batches.get(te)
            pre = post = unt = 0.0
            n_sel = n_zero = 0
            if b is not None:
                idx, wt = b
                if lots:
                    px = _px(t, idx)
                    sh = lot_shares(wt * aum, px, star[idx])
                    wl = sh * px / aum
                else:
                    wl = np.asarray(wt, np.float64)
                w[idx] = wl
                pre, post = float(np.sum(wt)), float(np.sum(wl))
                unt = float(wl[~ctx["can"][idx]].sum())
                n_sel, n_zero = len(idx), int((wl <= 0).sum())
            out = w
            held = hd > H_TOL
            cont = held & (w > 0)
            if lots and cont.any():
                j = np.flatnonzero(cont)
                one_lot = LOT_MAIN * _px(t, j) / aum
                same = np.abs(w[j] - hd[j]) < one_lot
                out = w.copy()
                out[j[same]] = np.nan                      # 不足一手的调整不做(NaN = 保持实际持仓)
                st["cont_hold"] += int(same.sum()); st["cont_trade"] += int((~same).sum())
            cur = (w > 0)
            reb_rows.append((t, te, pre, post, unt, n_sel, n_zero, float(hd.sum())))
            return out
        pend = (hd > H_TOL) & ~cur
        if not pend.any():
            return None
        st["pending_stock_days"] += int(pend.sum()); st["pending_days"] += 1
        out = np.full(N, np.nan)
        out[pend] = 0.0
        return out

    def reb_frame(dates: pd.DatetimeIndex) -> pd.DataFrame:
        if not reb_rows:
            return pd.DataFrame()
        a = np.array(reb_rows, dtype=np.float64)
        return pd.DataFrame(dict(te=a[:, 1].astype(int), tgt_pre=a[:, 2], tgt_post=a[:, 3],
                                 tgt_untradable=a[:, 4], n_sel=a[:, 5].astype(int),
                                 n_zero_lot=a[:, 6].astype(int), h_dec_before=a[:, 7]),
                            index=pd.DatetimeIndex(dates[a[:, 0].astype(int)], name="date"))

    return fn, st, reb_frame


_KEEP_COLS = ["ret", "gross_ret", "nav_prev", "aum", "trade_value", "buy_value", "sell_value", "cost_total",
              "cost_commission", "cost_stamp", "exposure", "part_cut", "stuck_w", "n_hold"]


def run_account(p, batches: dict, phase: int, capital: float, raw_px: np.ndarray, adj_px: np.ndarray,
                star: np.ndarray, lots: bool = True, min_commission: float = MIN_COMMISSION_YUAN,
                stress_bps: float = 0.0, impact_model=None, max_participation: float | None = None) -> dict:
    """单个独立账户(一个臂/复制组合 × 一个相位)跑完 2020-01-02 ~ 2025-12-31。

    返回 dict(daily=逐日 DataFrame[信号日], stats=dict)。daily 列:
      ret / gross_ret / exposure / tv(买+卖成交额/净值)/ cost_* 分解 / stuck_w / n_hold /
      is_reb(本相位调仓日)/ tgt_pre / tgt_post / tgt_untradable(仅调仓日行)。"""
    fees = EventFees(min_commission=min_commission, stress_bps=stress_bps, impact_model=impact_model)
    fn, st, reb_frame = make_batch_target(batches, phase, raw_px, adj_px, star, lots=lots)
    res = ST.simulate(fn, p, MODE, capital, rebalance=1, cost_model=fees, record=True,
                      normalize=False, participation_on_nav=True, max_participation=max_participation)
    dates = p.dates
    d = res["record"]["daily"][_KEEP_COLS].copy()
    del res["record"]                                              # 持仓矩阵 32MB,不留
    d["tv"] = d["trade_value"] / d["aum"]
    lf = fees.log_frame(dates).reindex(d.index).fillna(0.0)
    d = d.join(lf)
    rf = reb_frame(dates)
    d["is_reb"] = d.index.isin(rf.index)
    for c in ("tgt_pre", "tgt_post", "tgt_untradable", "n_sel", "n_zero_lot"):
        d[c] = rf[c].reindex(d.index) if len(rf) else np.nan
    return dict(daily=d, stats=st)


# ------------------------------------------------------------------ 下游信号面板
def load_signal_panel(p, mode: str = MODE, lag0: str = "raise") -> pd.DataFrame:
    """读 cache/_tmp/pead_signal_open10.parquet,并按成交口径做时序检查(审查 P2)。

    面板第 t 行 = "t+1 日 09:31 成交前已知",**不是** "t 日收盘已知":约 20% 的非空值
    在 t 日收盘后才公开(P == t 的盘后公告 16.2%,P 为 t 之后周末/节假日 4.0%;ledger pead_signal_panel_meta.csv)。
      - lag >= 1 的口径(open10 / open5 / vwap1):原样返回,与 run_weights 的 W[t] 同一索引;
      - lag == 0 的口径(tail1450 / tail1445 / close,W[t] 在 t 日成交):
          lag0="raise"(默认)报错;lag0="shift" 整体下移一行(第 t 行 = 原 t-1 行,t 日开盘前已公开);
          lag0="mask" 只保留 t 日收盘前已公开的格子(伴随文件 known_at_t_close,即 V <= t)。"""
    from afactor.config import TMP
    lag = int(ST.EXEC_MODES[mode]["lag"])
    S = pd.read_parquet(TMP / "pead_signal_open10.parquet").reindex(index=p.dates, columns=p.codes)
    if lag >= 1:
        return S
    if lag0 == "shift":
        return S.shift(1)
    if lag0 == "mask":
        L = pd.read_parquet(TMP / "pead_signal_open10_events.parquet",
                            columns=["t_signal", "stock_code", "known_at_t_close"])
        L = L[L["known_at_t_close"]]
        ok = np.zeros(S.shape, bool)
        i = p.dates.get_indexer(L["t_signal"]); j = p.codes.get_indexer(L["stock_code"])
        m = (i >= 0) & (j >= 0)
        ok[i[m], j[m]] = True
        return S.where(ok)
    raise ValueError(f"PEAD 信号面板只能直接用于 lag>=1 的口径;{mode} 的 lag=0,"
                     f"第 t 行约 20% 的值在 t 日收盘后才公开。请用 lag0='shift' 或 'mask'。")


# ------------------------------------------------------------------ 统计工具
def nw_se(x, L: int) -> float:
    """均值的 Newey-West(Bartlett)标准误。"""
    x = np.asarray(x, np.float64)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 10:
        return np.nan
    e = x - x.mean()
    s = float(e @ e) / n
    for k in range(1, int(L) + 1):
        s += 2.0 * (1.0 - k / (L + 1.0)) * float(e[k:] @ e[:-k]) / n
    return float(np.sqrt(max(s, 0.0) / n))


def mean_ci(x, L: int, ann: float = 1.0) -> dict:
    x = np.asarray(x, np.float64)
    x = x[np.isfinite(x)]
    if len(x) < 10:
        return dict(n=len(x), est=np.nan, se=np.nan, t=np.nan, lo=np.nan, hi=np.nan)
    m = float(x.mean()); se = nw_se(x, L)
    return dict(n=len(x), est=m * ann, se=se * ann, t=m / se if se > 0 else np.nan,
                lo=(m - 1.96 * se) * ann, hi=(m + 1.96 * se) * ann)


def quarterly_t(s: pd.Series) -> dict:
    """以日历季度为独立单位:每季度日均差为一个观测的 t 检验。"""
    q = s.groupby(s.index.to_period("Q")).mean()
    n = len(q)
    if n < 3 or q.std(ddof=1) <= 0:
        return dict(n_q=n, q_mean_ann=np.nan, q_t=np.nan)
    return dict(n_q=n, q_mean_ann=float(q.mean() * ST.TRADING_DAYS),
                q_t=float(q.mean() / (q.std(ddof=1) / np.sqrt(n))))


def ann_return(r: pd.Series) -> float:
    r = r.dropna()
    if len(r) == 0:
        return np.nan
    return float((1 + r).prod() ** (ST.TRADING_DAYS / len(r)) - 1)


def max_dd(r: pd.Series) -> float:
    nav = (1 + r.fillna(0)).cumprod()
    return float((nav / nav.cummax() - 1).min())
