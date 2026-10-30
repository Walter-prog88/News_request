"""事件驱动回测引擎 —— 复现"小碎步交易法"(2026-08-23)。

与 strategy.py 的周频 Top-N 引擎不同:这里是**逐日扫触发条件、按槽位建仓、按退出规则平仓**。
对应用户提供文档的四层:
  等待层   : 守门(板块/个股接近 252 日高点 = "强势")不满足就不出手,现金可以很高
  标的层   : 强势方向里的强势股(板块 range_pos + 个股 range_pos + 60 日涨幅排序)
  入场层   : 相对回踩(距 20 日高点回撤 pb_min~pb_max,且未破 MA60)+ 盘中转强(低开收红 / 突破前高)
  仓位层   : n_slots 个槽,每槽 slot_w 净值;其余现金
  退出层   : 止盈 tp / 止损 sl / 破 MA20 / 最长持有 max_hold;可选"更优标的出现整块换"

执行口径:**尾盘 tail1450** —— 14:49 截断出信号,14:50-14:57 TWAP 成交(最接近交易者的盘中确认后买入)。
信息时点:全日字段一律 shift(1)(T-1 收盘已知);T 日只用 open_auction_px 与 pre_* 截断字段。
已知近似:一字板判定用全日 one_word_*(14:49 时封板≈收盘封板);无最高价字段,"前高"用收盘价 20 日高点。

评价纪律(README §10.7、ETF 项目教训):
  - 基线 = 同一守门集合的等权(用户硬门槛 A)**和** 不守门的同策略;只有两者都赢才算守门有价值
  - 2025H2 为用户定义的趋势市起点,在长历史到位前**不据此调参**(withhold=True 时不打印)
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd

from afactor.config import (COST_BPS_ONEWAY, STAMP_TAX_BPS_BEFORE, STAMP_TAX_BPS_AFTER,
                            STAMP_TAX_CHANGE_DATE, MAX_PARTICIPATION)
from afactor.panel import Panel

TRADING_DAYS = 242


@dataclass
class Rules:
    # 守门 / 标的
    rp_win: int = 252          # range_pos 回看
    stock_rp_min: float = 0.70 # 个股 range_pos 下限("强势")
    sector_rp_min: float = 0.70
    use_sector_gate: bool = True
    use_stock_gate: bool = True
    # 入场
    pb_min: float = 0.02       # 距 20 日高点回撤下限
    pb_max: float = 0.12       # 回撤上限(再深算趋势破坏)
    trend_ma: int = 60         # 未破 MA60
    gap_dn: float = 0.003      # 低开阈值(开盘 < 昨收 × (1-gap_dn))
    path_lowopen: bool = True  # 路径1(文档 3.46×):低开 >0.3% 后 14:49 收红(> 昨收),且处于 2-12% 回踩区
    path_breakout: bool = True # 路径2(文档 2.0×):盘中突破昨日最高价后回踩到前高 ±bo_band(用真实 high_px)
    bo_band: float = 0.003
    bo_require_retrace: bool = True   # 要求 14:49 价 < 当日(09:30-14:49)最高,即确已回落
    rank_by: str = "sector"    # 候选排序:momentum(板块强度+60日涨幅,审查判为最大破坏者) / sector(只按板块强度) / cap(板块内大市值=龙头代理) / none
    # 仓位
    n_slots: int = 5
    slot_w: float = 0.11
    min_fill: float = 0.5      # 成交不足槽位 50% 则放弃
    # 退出
    tp: float = 0.08
    sl: float = 0.05
    exit_ma: int = 20
    max_hold: int = 15
    swap_when_full: bool = False   # 槽满时,更优候选出现 -> 换掉最弱持仓(文档唯一可直接确认的退出机制)
    min_hold_before_swap: int = 2
    swap_margin: float = 0.1       # 候选 score 须高出最弱持仓 score 这么多(score 量纲见 build_signals)
    cooldown: int = 3
    # 早盘退出窗口(T+1 09:31-09:40,twap_open10 成交):文档"早盘有利润便落袋"/低开即走
    am_exit_gap: float | None = 0.03   # 开盘低开超过此比例 -> 早盘出
    am_tp: float | None = None         # 早盘价 >= 入场价×(1+am_tp) -> 早盘落袋;None 关闭          # 退出后同一只至少隔 N 天才可再进(否则止损当天原价买回,纯付成本)


def _ma(x, w): return x.rolling(w, min_periods=max(5, w // 2)).mean()


def build_signals(p: Panel, rules: Rules, universe: pd.DataFrame, sector_labels: pd.Series | None):
    """全部信号在 T 日 14:49 可得。返回 dict of DataFrame(bool/float)。"""
    close = p.raw("close_px")                      # 全日字段:用 shift(1)
    c1 = close.shift(1)
    opn = p.raw("open_auction_px")                 # T 日开盘,已知
    px = opn * (1.0 + p.raw("pre_ret_intraday"))   # T 日 14:49 价
    h1 = p.raw("high_px").shift(1)                 # 昨日最高(分钟 open/close 极值近似)
    pre_hi = p.raw("pre_high_px")                  # 今日 09:30-14:49 最高,已知
    lo = c1.rolling(rules.rp_win, min_periods=120).min()
    hi = c1.rolling(rules.rp_win, min_periods=120).max()
    rp = (c1 - lo) / (hi - lo)
    ma_t = _ma(c1, rules.trend_ma); ma_x = _ma(c1, rules.exit_ma)
    hi20 = c1.rolling(20, min_periods=10).max()
    pullback = px / hi20 - 1.0
    ret60 = c1 / c1.shift(60) - 1.0

    # 路径 1:低开 > gap_dn 后 14:49 收红(> 昨收,文档口径;审查 F4),且处于回踩区
    low_open_red = ((opn < c1 * (1 - rules.gap_dn)) & (px > c1)
                    & (pullback <= -rules.pb_min) & (pullback >= -rules.pb_max))
    # 路径 2:盘中突破昨日最高价后回踩到前高 ±bo_band(文档"突破前高后回踩";审查 F1)
    bo = (pre_hi >= h1) & ((px / h1 - 1.0).abs() <= rules.bo_band)
    if rules.bo_require_retrace:
        bo &= (px < pre_hi)
    paths = pd.DataFrame(False, index=px.index, columns=px.columns)
    if rules.path_lowopen: paths |= low_open_red.fillna(False)
    if rules.path_breakout: paths |= bo.fillna(False)

    strong = pd.DataFrame(True, index=px.index, columns=px.columns)
    if rules.use_stock_gate:
        strong &= (rp >= rules.stock_rp_min)
    sec_rp = None
    if rules.use_sector_gate and sector_labels is not None:
        lab = sector_labels.dropna(); lab = lab[lab.index.isin(close.columns)]
        r = close.pct_change().where(universe)
        ir = r[lab.index].T.groupby(lab).mean().T
        idx = (1 + ir.fillna(0)).cumprod().shift(1)
        slo = idx.rolling(rules.rp_win, min_periods=120).min(); shi = idx.rolling(rules.rp_win, min_periods=120).max()
        srp = (idx - slo) / (shi - slo)
        sec_rp = pd.DataFrame({c: srp[lab[c]] for c in lab.index}).reindex(columns=close.columns)
        strong &= (sec_rp >= rules.sector_rp_min).fillna(False)
    entry = (strong & universe & (px > ma_t) & paths).fillna(False)
    # 排序
    if rules.rank_by == "momentum":
        score = (sec_rp.fillna(0) * 10 if sec_rp is not None else 0) + ret60.fillna(-9)
    elif rules.rank_by == "sector":
        score = (sec_rp.fillna(0) if sec_rp is not None else 0) + rp.fillna(0) * 0.01
    elif rules.rank_by == "cap":
        from afactor import newdata as ND
        mv = np.log(ND.total_mv(close.index, close.columns)).shift(1)
        score = (sec_rp.fillna(0) * 100 if sec_rp is not None else 0) + mv.fillna(0)
    else:
        score = pd.DataFrame(0.0, index=px.index, columns=px.columns)
    return dict(entry=entry, score=score.where(entry), score_all=score, px=px, ma_exit=ma_x,
                strong=strong, sec_rp=sec_rp, rp=rp, path_lowopen=low_open_red.fillna(False), path_bo=bo.fillna(False))


def run_event_bt(p: Panel, rules: Rules, version: str = "V100", date_mask=None,
                 sector_labels: pd.Series | None = None, universe: pd.DataFrame | None = None,
                 signals: dict | None = None) -> dict:
    from afactor.strategy import VERSIONS, universe as _uni
    cfg = VERSIONS[version]
    u = universe if universe is not None else _uni(p, version)
    sig = signals or build_signals(p, rules, u, sector_labels)
    dates, codes = p.dates, p.codes
    N = len(codes)
    entry = sig["entry"].reindex(index=dates, columns=codes).to_numpy(bool)
    score = sig["score"].reindex(index=dates, columns=codes).to_numpy(np.float64)
    score_all = sig["score_all"].reindex(index=dates, columns=codes).to_numpy(np.float64)
    px49 = sig["px"].reindex(index=dates, columns=codes).to_numpy(np.float64)
    ma_x = sig["ma_exit"].reindex(index=dates, columns=codes).to_numpy(np.float64)
    extra_exit = {k: v.reindex(index=dates, columns=codes).fillna(False).to_numpy(bool)
                  for k, v in (sig.get("exit_extra") or {}).items()}   # XSB2 附加退出通道(默认空,行为不变)
    fill_px = p.raw("twap_tail8").to_numpy(np.float64)
    win_amt = p.raw("amt_tail8").to_numpy(np.float64)
    am_px = p.raw("twap_open10").to_numpy(np.float64)      # 早盘窗口成交价(T 日 09:31-09:40)
    am_amt = p.raw("amt_open10").to_numpy(np.float64)
    opn = p.raw("open_auction_px").to_numpy(np.float64)
    close = p.raw("close_px").to_numpy(np.float64)
    c1 = np.vstack([np.full((1, N), np.nan), close[:-1]])
    dead_up = p["one_word_up"].to_numpy(bool); dead_dn = p["one_word_down"].to_numpy(bool)
    stamp_switch = np.searchsorted(dates.values, np.datetime64(STAMP_TAX_CHANGE_DATE))
    mask = np.ones(len(dates), bool) if date_mask is None else np.asarray(date_mask, bool)

    cash = 1.0; pos = {}        # code_idx -> dict(shares, entry_px, entry_t, peak)
    last_exit = np.full(N, -10**6, dtype=np.int64)
    nav_hist, ridx, trades, expo = [], [], [], []
    stats = dict(cap_blocked_entries=0, partial_entries=0, partial_exits=0, cand_days=0, cand_total=0)
    nav_prev = 1.0

    def _sell(j, t, price, wamt, why, stamp):
        nonlocal cash
        q = pos[j]
        sell_val = q["shares"] * price
        cap = MAX_PARTICIPATION * wamt / cfg["capital"] * nav_prev
        frac = min(1.0, cap / sell_val) if sell_val > 0 else 1.0
        if frac < 1.0: stats["partial_exits"] += 1
        sold = q["shares"] * frac
        proceeds = sold * price
        cash += proceeds - proceeds * (COST_BPS_ONEWAY * 1e-4 + stamp)
        trades.append(dict(code=codes[j], side="sell", date=dates[t], px=price,
                           ret=price / q["entry_px"] - 1, held=t - q["entry_t"], why=why, frac=frac))
        q["shares"] -= sold
        if q["shares"] <= 1e-12:
            del pos[j]; last_exit[j] = t

    for t in range(len(dates)):
        if not mask[t]:
            continue
        stamp = (STAMP_TAX_BPS_BEFORE if t < stamp_switch else STAMP_TAX_BPS_AFTER) * 1e-4
        # ---- 早盘退出窗口(09:31-09:40,用开盘价判、twap_open10 成交)----
        if rules.am_exit_gap is not None or rules.am_tp is not None:
            for j in list(pos):
                if pos[j]["entry_t"] >= t:          # 当日才建的仓不在早盘处理
                    continue
                if not (np.isfinite(am_px[t, j]) and np.isfinite(opn[t, j]) and np.isfinite(c1[t, j])):
                    continue
                why = None
                if rules.am_exit_gap is not None and opn[t, j] < c1[t, j] * (1 - rules.am_exit_gap): why = "am_gap"
                elif rules.am_tp is not None and opn[t, j] >= pos[j]["entry_px"] * (1 + rules.am_tp): why = "am_tp"
                if why and not dead_dn[t, j] and am_amt[t, j] > 0:
                    _sell(j, t, am_px[t, j], am_amt[t, j], why, stamp)
        # ---- 尾盘退出(14:49 判,14:50-57 成交)----
        for j in list(pos):
            q = pos[j]; pxn = px49[t, j]
            if not np.isfinite(pxn) or q["entry_t"] >= t:
                continue
            r_un = pxn / q["entry_px"] - 1.0
            held = t - q["entry_t"]
            why = None
            if r_un >= rules.tp: why = "tp"
            elif r_un <= -rules.sl: why = "sl"
            elif np.isfinite(ma_x[t, j]) and pxn < ma_x[t, j]: why = "ma"
            elif held >= rules.max_hold: why = "time"
            if why is None:
                for _nm, _mat in extra_exit.items():
                    if _mat[t, j]:
                        why = _nm
                        break
            if why and np.isfinite(fill_px[t, j]) and not dead_dn[t, j] and win_amt[t, j] > 0:
                _sell(j, t, fill_px[t, j], win_amt[t, j], why, stamp)
        # ---- 入场(含槽满时轮换)----
        cand = np.flatnonzero(entry[t] & np.isfinite(score[t]))
        cand = [j for j in cand if j not in pos and np.isfinite(fill_px[t, j])
                and not dead_up[t, j] and win_amt[t, j] > 0 and t - last_exit[j] > rules.cooldown]
        cand.sort(key=lambda j: -score[t, j])
        if cand:
            stats["cand_days"] += 1; stats["cand_total"] += len(cand)
        if rules.swap_when_full and cand and len(pos) >= rules.n_slots:
            # 最弱持仓(按当前 score_all)且已持有够久 -> 换成更优候选
            worst = min(pos, key=lambda j: score_all[t, j] if np.isfinite(score_all[t, j]) else -1e9)
            if (t - pos[worst]["entry_t"] >= rules.min_hold_before_swap
                    and score[t, cand[0]] - (score_all[t, worst] if np.isfinite(score_all[t, worst]) else -1e9) >= rules.swap_margin
                    and np.isfinite(fill_px[t, worst]) and not dead_dn[t, worst] and win_amt[t, worst] > 0):
                _sell(worst, t, fill_px[t, worst], win_amt[t, worst], "swap", stamp)
        free = rules.n_slots - len(pos)
        for j in cand[:max(free, 0)]:
            target = rules.slot_w * nav_prev
            cap = MAX_PARTICIPATION * win_amt[t, j] / cfg["capital"] * nav_prev
            val = min(target, cap, cash)
            if val < rules.min_fill * target:
                stats["cap_blocked_entries"] += 1
                continue
            if val < target - 1e-12: stats["partial_entries"] += 1
            cost = val * COST_BPS_ONEWAY * 1e-4
            sh = val / fill_px[t, j]
            cash -= val + cost
            pos[j] = dict(shares=sh, entry_px=fill_px[t, j], entry_t=t, peak=fill_px[t, j])
            trades.append(dict(code=codes[j], side="buy", date=dates[t], px=fill_px[t, j],
                               ret=np.nan, held=0, why="entry", frac=val / target))
        # ---- 收盘计价 ----
        mv = sum(q["shares"] * (close[t, j] if np.isfinite(close[t, j]) else q["entry_px"]) for j, q in pos.items())
        nav = cash + mv
        nav_hist.append(nav); ridx.append(dates[t]); expo.append(mv / nav if nav > 0 else 0.0)
        nav_prev = nav
    nav_s = pd.Series(nav_hist, index=pd.DatetimeIndex(ridx))
    rets = nav_s.pct_change().fillna(nav_s.iloc[0] - 1.0) if len(nav_s) else pd.Series(dtype=float)
    tr = pd.DataFrame(trades)
    sells = tr[(tr.side == "sell") & (tr.frac >= 0.999)] if len(tr) else tr     # 完整退出才计入单笔统计
    yrs = max(len(nav_s), 1) / TRADING_DAYS
    out = dict(nav=nav_s, rets=rets, trades=tr,
               ann_return=float(nav_s.iloc[-1] ** (1 / yrs) - 1) if len(nav_s) else np.nan,
               sharpe=float(rets.mean() / rets.std() * np.sqrt(TRADING_DAYS)) if len(rets) > 5 and rets.std() > 0 else np.nan,
               max_dd=float((nav_s / nav_s.cummax() - 1).min()) if len(nav_s) else np.nan,
               exposure=float(np.mean(expo)) if expo else 0.0,
               n_trades=int(len(sells)), win_rate=float((sells.ret > 0).mean()) if len(sells) else np.nan,
               avg_ret=float(sells.ret.mean()) if len(sells) else np.nan,
               avg_hold=float(sells.held.mean()) if len(sells) else np.nan,
               exit_mix=sells.why.value_counts().to_dict() if len(sells) else {},
               n_days=int(len(nav_s)), rules=asdict(rules), stats=stats)
    return out
