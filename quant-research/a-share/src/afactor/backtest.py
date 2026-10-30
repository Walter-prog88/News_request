"""TWAP 成交模拟回测:日调仓、参与率上限、涨停不可成交、1bp 双边 + 卖出印花税。

口径:T 日收盘后确定目标持仓 -> T+1 开盘 09:31-09:40 TWAP 成交 -> 持有 -> 下一次调仓。
资金 1000 万;单只股票单日成交额不得超过成交窗口内市场成交额的 MAX_PARTICIPATION。
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from afactor.config import (CAPITAL, MAX_PARTICIPATION, COST_BPS_ONEWAY,
                            STAMP_TAX_BPS_BEFORE, STAMP_TAX_BPS_AFTER,
                            STAMP_TAX_CHANGE_DATE, MAX_TURNOVER)
from afactor.panel import Panel

TRADING_DAYS = 242


def target_weights(F: np.ndarray, M: np.ndarray, top_n: int | None = 100,
                   top_frac: float | None = None, sign: int = 1,
                   scheme: str = "top", band: tuple[float, float] | None = None) -> np.ndarray:
    """按因子值构建等权目标持仓。sign=+1 取因子值最大的一批。

    scheme='top'     : 取最好的 top_n 只(选优)
    scheme='exclude' : 排除最差的 top_frac 比例,其余全部等权(排雷)
    scheme='band'    : 取因子分位落在 band=(lo,hi) 区间的股票(分层测试指向的最优区段)
    """
    T, N = F.shape
    W = np.zeros((T, N), np.float32)
    for t in range(T):
        m = M[t] & np.isfinite(F[t])
        n = int(m.sum())
        if n < 50:
            continue
        idx = np.flatnonzero(m)
        v = F[t, idx] * sign
        if scheme == "top":
            k = min(top_n if top_n else max(int(round(n * (top_frac or 0.1))), 10), n)
            sel = idx[np.argpartition(-v, k - 1)[:k]]
        else:
            from scipy.stats import rankdata
            q = rankdata(v) / n
            if scheme == "exclude":
                thr = 0.1 if top_frac is None else top_frac   # 注意 0.0 是合法值,不能用 `or`
                sel = idx[q > thr]
            else:
                lo, hi = band or (0.6, 0.9)
                sel = idx[(q > lo) & (q <= hi)]
        if sel.size == 0:
            continue
        W[t, sel] = 1.0 / sel.size
    return W


def weights_exclude_buffer(F: np.ndarray, M: np.ndarray, sign: int = 1,
                           enter: float = 0.10, exit: float = 0.20) -> np.ndarray:
    """排雷 + 缓冲带:分位跌入最差 enter 才剔除,回升到 exit 以上才重新纳入。
    不对称阈值抑制在门槛附近来回进出造成的换手。"""
    from scipy.stats import rankdata
    T, N = F.shape
    W = np.zeros((T, N), np.float32)
    excluded = np.zeros(N, bool)
    for t in range(T):
        m = M[t] & np.isfinite(F[t])
        n = int(m.sum())
        if n < 50:
            W[t] = W[t - 1] if t else 0
            continue
        idx = np.flatnonzero(m)
        q = rankdata(F[t, idx] * sign) / n          # 低分位 = 最差
        ex_t = np.zeros(N, bool)
        ex_t[idx[q <= enter]] = True                # 新进雷区
        stay = excluded.copy()
        stay[idx[q > exit]] = False                 # 已出雷区
        stay[~m] = excluded[~m]                     # 当日无数据者维持原状态
        excluded = ex_t | stay
        sel = idx[~excluded[idx]]
        if sel.size:
            W[t, sel] = 1.0 / sel.size
    return W


def run_backtest(F: pd.DataFrame, panel: Panel, top_n: int = 100, sign: int = 1,
                 capital: float = CAPITAL, mode: str = "open",
                 scheme: str = "top", top_frac: float | None = None,
                 band: tuple[float, float] | None = None,
                 date_mask: np.ndarray | None = None,
                 buffer: tuple[float, float] | None = None,
                 rebalance_every: int = 1) -> dict:
    """date_mask 为 None 时回测全样本;给定布尔数组时只在该子区间建仓计价(用于封存期)。
    buffer=(enter,exit) 时使用排雷缓冲带方案;rebalance_every=k 时每 k 日才更新目标持仓。"""
    dates = panel.dates
    codes = panel.codes
    F = F.reindex(index=dates, columns=codes)
    M = panel["universe"].to_numpy(dtype=bool)
    if date_mask is not None:
        M = M & date_mask[:, None]
    if buffer is not None:
        W = weights_exclude_buffer(F.to_numpy(np.float32), M, sign=sign,
                                   enter=buffer[0], exit=buffer[1])
    else:
        W = target_weights(F.to_numpy(np.float32), M, top_n=top_n, sign=sign,
                           scheme=scheme, top_frac=top_frac, band=band)
    # rebalance_every 在主循环中处理:非调仓日不交易,持仓自然漂移
    # (若在此处强制 W[t]=旧目标,引擎会每天做"反漂移"交易,虚增换手)

    if mode == "open":
        px = panel.raw("twap_open10").to_numpy(np.float64)
        win_amt = panel.raw("amt_open10").to_numpy(np.float64)
        lag = 1                       # T 日信号 -> T+1 成交
    else:
        px = panel.raw("twap_tail8").to_numpy(np.float64)
        win_amt = panel.raw("amt_tail8").to_numpy(np.float64)
        lag = 0
    dead = (panel["one_word_up"] | panel["one_word_down"]).to_numpy(dtype=bool)
    ret_px = panel.raw("close_px").to_numpy(np.float64)

    T, N = W.shape
    stamp_switch = np.searchsorted(dates.values, np.datetime64(STAMP_TAX_CHANGE_DATE))
    h = np.zeros(N)                    # 持仓市值权重(按上期成交价计)
    rets, ridx, turns, costs, held, exps = [], [], [], [], [], []
    # 调仓相位锚在 date_mask 首个信号日,否则窗口开头会空仓 (-t0)%k 天(见 strategy.py 同注)
    t0 = int(np.flatnonzero(date_mask)[0]) if date_mask is not None and date_mask.any() else 0
    for t in range(T - lag - 1):
        te = t + lag                   # 成交日
        if te >= T - 1:
            break
        p_now = px[te]
        p_next = px[te + 1]
        valid_px = np.isfinite(p_now) & np.isfinite(p_next) & (p_now > 0)
        can_trade = valid_px & ~dead[te] & (win_amt[te] > 0)
        if rebalance_every > 1 and ((t - t0) % rebalance_every):
            delta = np.zeros(N)        # 非调仓日:不交易,持仓自然漂移
        else:
            # 卖不掉的旧仓先从预算里扣掉,剩下的才归一给可交易标的。
            # 旧写法会让 sum(newh) > 1(无成本杠杆),实测最大 1.5375。
            stuck = (~can_trade) & (h > 1e-12)
            budget = max(0.0, 1.0 - float(h[stuck].sum()))
            tgt = np.where(can_trade, W[t].astype(np.float64), 0.0)
            s = tgt.sum()
            if s > 0:
                tgt = tgt / s * budget
            tgt = tgt + np.where(stuck, h, 0.0)
            delta = tgt - h
        # 参与率上限:单只股票成交金额 <= MAX_PARTICIPATION * 窗口成交额
        cap_w = np.where(np.isfinite(win_amt[te]), MAX_PARTICIPATION * win_amt[te] / capital, 0.0)
        delta = np.clip(delta, -cap_w, cap_w)
        delta = np.where(can_trade, delta, 0.0)
        # 日换手上限
        tv = np.abs(delta).sum() / 2.0
        if tv > MAX_TURNOVER:
            delta *= MAX_TURNOVER / tv
            tv = MAX_TURNOVER
        # 卖出侧被参与率截断时残仓仍占资金,买单按剩余预算等比压缩(审查 E3,见 strategy.py)
        over = float(np.clip(h + delta, 0, None).sum()) - 1.0
        if over > 1e-12:
            buys = np.clip(delta, 0, None); bs = buys.sum()
            if bs > 0:
                delta = delta - buys * min(1.0, over / bs)
        newh = h + delta
        newh = np.clip(newh, 0, None)
        buy = np.clip(delta, 0, None).sum()
        sell = np.clip(-delta, 0, None).sum()
        stamp = (STAMP_TAX_BPS_BEFORE if te < stamp_switch else STAMP_TAX_BPS_AFTER) * 1e-4
        cost = (buy + sell) * COST_BPS_ONEWAY * 1e-4 + sell * stamp
        # 持有一日收益
        with np.errstate(invalid="ignore", divide="ignore"):
            r = np.where(valid_px, p_next / p_now - 1.0, 0.0)
        port_r = float(np.nansum(newh * r)) - cost
        # 权重随价格漂移。**保留现金腿**:参与率/换手上限没买进的部分留作现金,
        # 旧写法 grown/tot 会把未成交部分零成本补满,使两个上限实际失效。
        grown = newh * (1.0 + np.nan_to_num(r))
        h = grown / (1.0 + port_r) if (1.0 + port_r) > 1e-9 else grown
        hs = h.sum()
        if hs > 1.0 + 1e-12:          # 成本把现金扣成负 -> 实际是从持仓里出(审查 E3)
            h = h / hs
        rets.append(port_r); ridx.append(dates[t])
        turns.append(tv); costs.append(cost); held.append(int((newh > 1e-8).sum()))
        exps.append(float(newh.sum()))
        assert newh.sum() <= 1.0 + 1e-9, f"exposure>1: {newh.sum():.6f} at t={t}"

    # 改为日收益口径:旧写法 nav_s/nav_s.iloc[0] 会丢掉窗口首日的真实收益,
    # 且 pct_change().fillna(首日) 塞一个假 0 压低波动、虚高夏普;年化分母也少 1 天。
    # 全样本可忽略,但封存期窗口只有百余天时能造成 ~1.6pp 的年化偏差。
    r_s = pd.Series(rets, index=pd.DatetimeIndex(ridx))
    turns, costs = np.asarray(turns), np.asarray(costs)
    held, exps = np.asarray(held), np.asarray(exps)
    if date_mask is not None:
        keep = pd.Series(date_mask[:len(dates)], index=dates).reindex(
            r_s.index).fillna(False).to_numpy(bool)
        if keep.sum() < 5:
            return {"nav": (1 + r_s[keep]).cumprod(), "ann_return": np.nan, "sharpe": np.nan,
                    "max_dd": np.nan, "avg_turnover": np.nan, "avg_cost_bps": np.nan,
                    "avg_holdings": np.nan, "n_days": int(keep.sum())}
        r_s = r_s[keep]
        turns, costs, held, exps = turns[keep], costs[keep], held[keep], exps[keep]
    nav_s = (1.0 + r_s).cumprod()
    yrs = len(r_s) / TRADING_DAYS
    return {"nav": nav_s, "rets": r_s, "ann_return": float(nav_s.iloc[-1] ** (1 / yrs) - 1),
            "sharpe": float(r_s.mean() / r_s.std() * np.sqrt(TRADING_DAYS)) if r_s.std() > 0 else np.nan,
            "max_dd": float((nav_s / nav_s.cummax() - 1).min()),
            "avg_turnover": float(turns.mean()), "avg_cost_bps": float(costs.mean() * 1e4),
            "avg_holdings": float(held.mean()), "n_days": len(r_s),
            "exposure": float(exps.mean())}


def run_standard(F: pd.DataFrame, panel: Panel, sign: int = 1, mode: str = "open",
                 date_mask: np.ndarray | None = None) -> dict:
    """项目标准回测配置:排雷缓冲带 20/35 + 5 日调仓(见 config.py 依据注释)。"""
    from afactor.config import STANDARD_BUFFER, STANDARD_REBALANCE
    return run_backtest(F, panel, sign=sign, mode=mode, buffer=STANDARD_BUFFER,
                        rebalance_every=STANDARD_REBALANCE, date_mask=date_mask)


def benchmark_nav(panel: Panel, mode: str = "open") -> pd.Series:
    """决策池等权基准(同样按 TWAP 计价,不含成本)。

    收益打在**信号日 t**,与 run_backtest 的 navs 索引一致。旧版按成交日索引,
    主口径 lag=1 时整条基准相对策略错开一天,逐年超额带 1~10pp 伪信号。"""
    px = panel.raw("twap_open10" if mode == "open" else "twap_tail8")
    u = panel["universe"]
    lag = 0 if mode == "tail" else 1
    g = px.shift(-1) / px - 1                                   # 索引在成交日 te
    m = g.where(u.shift(lag).fillna(False)).mean(axis=1)        # te 日、用信号日 t 的池
    # 长度必须与 run_backtest 的 t ∈ [0, T-lag-2] 一致,否则年化分母差 lag+1 天。
    # 全期只差 0.016pp,但封存期百余天的窗口上同样 2 天能造成 ~1.6pp。
    daily = m.shift(-lag).iloc[:len(panel.dates) - 1 - lag].fillna(0.0)   # 搬回信号日 t
    return (1 + daily).cumprod()
