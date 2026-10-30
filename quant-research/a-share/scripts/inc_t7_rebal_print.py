"""INC-T7 事后机制诊断(D12,只解释,不进判定):"再平衡频率效应"在两面板符号相反,是不是价格口径(开盘集合竞价单点价 vs 开盘 10 分钟
TWAP)造成的?

同一个池(分钟面板 V1000 池,2021-2025),用三种价格做池等权基准(不含成本、不含可成交约束,与 benchmark(rebalance=k) 同一逻辑):
  twap_open10    开盘 10 分钟 TWAP(分钟面板现行口径)
  open_auction   开盘集合竞价单点价(长历史面板用的"日开盘价"同类)
  close_px       收盘价
报告每日重置(k=1)减 5 日重置(k=5,5 相位平均)的算术年化差,即 Δ 分解里的 "再平衡频率效应";长历史面板(q92 池,日开盘价)按同窗口给对照。
只读缓存,不跑策略、不算成本。产出 research/ledger/inc_t7_rebal_print.csv。
用法:PYTHONPATH=src python3 scripts/inc_t7_rebal_print.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import inc_t7_grid as G
from afactor.config import LEDGER

TD = 242


def b_series(P, U, k):
    T = P.shape[0]
    acc = None
    for ph in range(k):
        reset = ((np.arange(T - 2) - ph) % k) == 0
        r = G.pool_ew_custom(P, U, reset)
        acc = r if acc is None else acc + r
    return acc / k


def main():
    rows = []
    G._init("minute")
    c = G.ctx()
    dates = c.p.dates
    U = c.U.reindex(index=dates, columns=c.p.codes).fillna(False).to_numpy(bool)
    for name in ("twap_open10", "open_auction_px", "close_px"):
        P = c.p.raw(name).reindex(index=dates, columns=c.p.codes).to_numpy(np.float64)
        s = {k: pd.Series(b_series(P, U, k), index=dates[:len(dates) - 2]) for k in (1, 5)}
        d = (s[1] - s[5])
        for lab, a, b in (("2021-2025", "2021-01-01", "2025-12-31"), ("2021-2023", "2021-01-01", "2023-12-31"),
                          ("2024-2025", "2024-01-01", "2025-12-31")):
            dd = d.loc[a:b]
            rows.append(dict(panel="minute V1000", price=name, window=lab, k1_minus_k5_arith=float(dd.mean() * TD),
                             k1_ann_arith=float(s[1].loc[a:b].mean() * TD), k5_ann_arith=float(s[5].loc[a:b].mean() * TD),
                             n=len(dd)))
        print(name, "done", flush=True)
    G.__dict__["_CTX"] = None
    G._init("lh")
    c = G.ctx()
    dates = c.p.dates
    U = c.U.reindex(index=dates, columns=c.p.codes).fillna(False).to_numpy(bool)
    P = c.p.raw("twap_open10").reindex(index=dates, columns=c.p.codes).to_numpy(np.float64)
    s = {k: pd.Series(b_series(P, U, k), index=dates[:len(dates) - 2]) for k in (1, 5)}
    d = s[1] - s[5]
    for lab, a, b in (("2006-2025", "2006-01-01", "2025-12-31"), ("2006-2015", "2006-01-01", "2015-12-31"),
                      ("2016-2020", "2016-01-01", "2020-12-31"), ("2021-2025", "2021-01-01", "2025-12-31")):
        dd = d.loc[a:b]
        rows.append(dict(panel="长历史 q92", price="日开盘价", window=lab, k1_minus_k5_arith=float(dd.mean() * TD),
                         k1_ann_arith=float(s[1].loc[a:b].mean() * TD), k5_ann_arith=float(s[5].loc[a:b].mean() * TD),
                         n=len(dd)))
    out = pd.DataFrame(rows)
    out.to_csv(LEDGER / "inc_t7_rebal_print.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
