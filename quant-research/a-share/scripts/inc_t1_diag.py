"""INC-T1 事后诊断(非预注册,只用于解释机制,不进入判定):
排雷 / Top20 的 1000 万成交清单按交易类别 × 隔夜跳空方向拆分"个股相对同窗池子"的已实现节省(W5、W9、W7 对 W1,VWAP)。

交易类别(按信号日 t 的目标权重 W[t,i] 与成交 Δw 判定):
  exit   卖出且目标为 0(调出:跌入排雷剔除区 / 掉出 Top20)
  trim   卖出且目标 > 0(等权再平衡:卖掉过去几天相对涨多的)
  entry  买入且 Δw > 0.5 × 目标(新建仓)
  add    其余买入(等权再平衡:补过去几天相对跌多的)
隔夜跳空(个股相对池子)= open_auction_px[t+1]/close[t] − 池子均值;按跳空方向 × 交易方向分"同向 / 反向"。
产出 research/ledger/inc_t1_diag_categories.csv。
审查 P0 修复:同时给 seal = none(修复前,整窗口封死的腿按成交价计)与 seal = dir(封死的腿顺延次日同窗口)两版,
列 seal 区分;封板腿多是"低开封跌停的调出股",它们不会日内回补,所以修复后"调出遇低开"的节省应缩小。
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import inc_t1_common as C                                   # noqa: E402
from inc_t1_common import ST, Panel, LEDGER, TRADING_DAYS   # noqa: E402
import inc_t1_tradelist as TL                               # noqa: E402


def main():
    p = Panel()
    ctx = TL.Ctx(p)
    scores = C.load_scores(p)
    T = len(p.dates)
    close = ctx.close
    oa = p.raw("open_auction_px").to_numpy(np.float64)
    U = ctx.U
    with np.errstate(divide="ignore", invalid="ignore"):
        G = oa[1:] / close[:-1] - 1.0
    Ug = U[:-1] & np.isfinite(G)
    gpool = np.full(T, np.nan)
    gpool[:-1] = np.where(Ug, G, 0).sum(1) / np.maximum(Ug.sum(1), 1)
    rows = []
    for strat in C.STRAT_KEYS:
        W = C.strategy_weights(p, scores, strat)
        mask = pd.Series(C.sim_mask(p, strat), index=p.dates)
        z = np.load(TL.trades_path(strat, 1e7))
        t = z["t"].astype(np.int64); i = z["i"].astype(np.int64); dw = z["dw"]
        s = np.sign(dw); w = np.abs(dw)
        tgt = W[t, i].astype(np.float64)
        cat = np.where(dw < 0, np.where(tgt <= 0, "exit", "trim"), np.where(dw > 0.5 * tgt, "entry", "add"))
        with np.errstate(divide="ignore", invalid="ignore"):
            gap = oa[np.minimum(t + 1, T - 1), i] / close[t, i] - 1.0 - gpool[t]
        gdir = np.where(~np.isfinite(gap), "na", np.where(gap * s > 0, "same", "opp"))   # 跳空方向与交易方向同向 = 买入的票高开 / 卖出的票低开
        def rel(wname, seal):
            P = ctx.px(wname, "vwap")
            with np.errstate(divide="ignore", invalid="ignore"):
                r = P[np.minimum(t + 1, T - 1), i].astype(np.float64) / close[t, i] - 1.0
                r2 = P[np.minimum(t + 2, T - 1), i].astype(np.float64) / close[t, i] - 1.0
            M = ctx.M(wname, "vwap", 1)[t]
            M2 = ctx.M(wname, "vwap", 2)[t]
            if seal != "none":
                blk = C.blocked_mask(p, wname, seal, dw > 0, np.minimum(t + 1, T - 1), i) & np.isfinite(r2) & np.isfinite(M2)
                r = np.where(blk, r2, r); M = np.where(blk, M2, M)
            return s * (r - M)
        for seal, wname in [(sl, wn) for sl in ("none", "dir") for wn in ["W5", "W9", "W7"]]:
            base = rel("W1", seal)
            sav = np.nan_to_num(base - rel(wname, seal))
            for per, (y0, y1, role) in C.PERIODS[strat].items():
                days = p.dates[(mask & (p.dates.year >= y0) & (p.dates.year <= y1)).to_numpy()]
                dsel = np.isin(t, p.dates.get_indexer(days))
                for c_ in ["exit", "trim", "entry", "add", "all"]:
                    for gd in ["same", "opp", "all"]:
                        m = dsel & ((cat == c_) if c_ != "all" else True) & ((gdir == gd) if gd != "all" else True)
                        daily = np.bincount(t[m], weights=w[m] * sav[m], minlength=T) / C.REBALANCE
                        wv = np.bincount(t[m], weights=w[m], minlength=T) / C.REBALANCE
                        d = pd.Series(daily, index=p.dates).reindex(days).fillna(0.0)
                        rows.append(dict(strategy=strat, window=wname, seal=seal, period=per, role=role, category=c_, gap_dir=gd,
                                         saving_pp=float(d.mean() * TRADING_DAYS * 100), t_nw=C.nw_t(d.to_numpy()),
                                         turnover_ann=float(pd.Series(wv, index=p.dates).reindex(days).fillna(0).mean() * TRADING_DAYS),
                                         bp_per_trade=float(daily[p.dates.get_indexer(days)].sum() / max(wv[p.dates.get_indexer(days)].sum(), 1e-12) * 1e4),
                                         n_trades=int(m.sum())))
        print(strat, "done", flush=True)
    pd.DataFrame(rows).to_csv(LEDGER / "inc_t1_diag_categories.csv", index=False)


if __name__ == "__main__":
    main()
