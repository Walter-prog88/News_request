"""INC-T1 审查 P0 的配套诊断(非预注册,只量化封板规则的口径影响,不进入判定)。

1. inc_t1_seal_counts.csv:排雷(COMP_v5 20/35)成交清单(1000 万,2021-2025)在各窗口被整窗口封板挡住的腿数、
   |Δw| 占比(买 / 卖分列,dir / any / tight 三口径),以及"顺延一天后同窗口、同方向仍封死"的比例(= 单次顺延口径的宽松程度)。
2. inc_t1_seal_multiroll.csv:已实现个股相对成本 (A = px_rel) 的节省(窗口 W 相对 W1,VWAP),三种封板处理并列:
   none(不处理)/ roll1(预注册规则:只顺延一次,第一层主口径)/ rollN(一直顺延到不封为止,最多 5 个交易日)。
   只算 px_rel(不含价差 / 冲击 / 截量项),口径与 d_px_rel 列一致,所以 roll1 行应复现 inc_t1_layer1_grid.csv 的 d_px_rel_pp。
   rollN 回答的问题:顺延日仍然封死的腿被当成"能成交"是不是在偏袒晚窗口。
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import inc_t1_common as C                                   # noqa: E402
from inc_t1_common import Panel, LEDGER, TRADING_DAYS       # noqa: E402
import inc_t1_tradelist as TL                               # noqa: E402

STRAT = "COMP_v5_exclude"
WINS = ["W2", "W3", "W4", "W5", "W6", "W7", "W8", "W9", "W10", "W11"]
MAXROLL = 5


def main():
    p = Panel()
    ctx = TL.Ctx(p)
    T = ctx.T
    mask = pd.Series(C.sim_mask(p, STRAT), index=p.dates)
    out_counts, out_roll = [], []
    for X in (1e7, 1e9):
        z = np.load(TL.trades_path(STRAT, X))
        t = z["t"].astype(np.int64); i = z["i"].astype(np.int64); dw = z["dw"]
        w = np.abs(dw); s = np.sign(dw); buy = dw > 0
        sel_y = np.asarray(p.dates.year)[t] >= 2021
        close = ctx.close

        def px_rel_multi(wn, mode):
            """逐腿 px_rel = w·s·(r_j − M_j),j = 实际成交日偏移;mode ∈ none / roll1 / rollN。不可成交(缺价 / 零成交额)沿用第一层规则顺延一次。"""
            P = ctx.px(wn, "vwap"); A = ctx.amt(wn)
            j = np.ones(len(t), np.int64)
            with np.errstate(all="ignore"):
                r = P[np.minimum(t + 1, T - 1), i].astype(np.float64) / close[t, i] - 1.0
            miss = ~np.isfinite(r) | ~(A[np.minimum(t + 1, T - 1), i] > 0)
            blk1 = C.blocked_mask(p, wn, "dir", buy, np.minimum(t + 1, T - 1), i)
            if mode == "none":
                need = miss
            else:
                need = miss | blk1
            j = np.where(need, 2, 1)
            if mode == "rollN":
                for jj in range(2, MAXROLL + 2):
                    still = (j == jj) & C.blocked_mask(p, wn, "dir", buy, np.minimum(t + jj, T - 1), i)
                    j = np.where(still, jj + 1, j)
            res = np.zeros(len(t))
            for jj in range(1, MAXROLL + 2):
                m = j == jj
                if not m.any():
                    continue
                with np.errstate(all="ignore"):
                    rj = P[np.minimum(t[m] + jj, T - 1), i[m]].astype(np.float64) / close[t[m], i[m]] - 1.0
                Mj = ctx.M(wn, "vwap", jj)[t[m]]
                v = w[m] * s[m] * (rj - Mj)
                res[m] = np.where(np.isfinite(v), v, 0.0)
            return res

        # 1 计数(只在 1000 万清单上)
        if X == 1e7:
            for wn in ["W0", "W1"] + WINS:
                te = np.minimum(t + 1, T - 1)
                for seal in ("dir", "any", "tight"):
                    b = C.blocked_mask(p, wn, seal, buy, te, i)
                    b2 = C.blocked_mask(p, wn, seal, buy, np.minimum(t + 2, T - 1), i)
                    for side, sm in (("buy", buy), ("sell", ~buy)):
                        m = sel_y & sm
                        out_counts.append(dict(window=wn, seal=seal, side=side, legs=int(m.sum()), blocked=int((b & m).sum()),
                                               blocked_share_legs=float((b & m).mean() / max(m.mean(), 1e-12)) if m.any() else np.nan,
                                               blocked_share_w=float(w[b & m].sum() / w[m].sum()),
                                               reblocked_next_day_share=float((b2 & b & m).sum() / max((b & m).sum(), 1))))
        # 2 多次顺延
        days_all = p.dates[(mask & (p.dates.year >= 2021)).to_numpy()]
        for mode in ("none", "roll1", "rollN"):
            base = px_rel_multi("W1", mode)
            for wn in WINS:
                sav = base - px_rel_multi(wn, mode)
                daily = np.bincount(t, weights=sav, minlength=T)[:T] / C.REBALANCE
                for pname, (y0, y1, role) in C.PERIODS[STRAT].items():
                    days = p.dates[(mask & (p.dates.year >= y0) & (p.dates.year <= y1)).to_numpy()]
                    d = pd.Series(daily, index=p.dates).reindex(days).fillna(0.0)
                    out_roll.append(dict(X=X, window=wn, mode=mode, period=pname, role=role,
                                         px_rel_saving_pp=float(d.mean() * TRADING_DAYS * 100), t_nw=C.nw_t(d.to_numpy())))
        print(f"X={X:.0e} done", flush=True)
    pd.DataFrame(out_counts).to_csv(LEDGER / "inc_t1_seal_counts.csv", index=False)
    pd.DataFrame(out_roll).to_csv(LEDGER / "inc_t1_seal_multiroll.csv", index=False)


if __name__ == "__main__":
    main()
