"""INC_T2 附:调入 / 调出篮子相对排雷基线的 Barra 风格暴露(预注册 §7.3"调入篮子相对基线的 Barra 风格暴露")。

每期在公告日 A(决策日)取:篮子 = 该格事件股等权(A 日合格池内);基线 = 排雷 COMP_v5 在 A 日的目标权重 W[A]。
active = RiskModel.active_exposure(篮子, 基线, A)(open10 风险模型,v3 缓存);覆盖池外权重忽略并记录。
输出 research/ledger/inc_t2_barra_exposure.csv(逐期 + 开发 / 检验 / 全期均值)。
"""
from __future__ import annotations
import os, sys, warnings
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
from afactor.config import TMP, LEDGER
from afactor.panel import Panel
from afactor.barra import RiskModel
from inc_t2_overlay import sim_mask, baseline_W

CELLS = [("O", "000905", "in", "ADD_800"), ("O", "000905", "out", "DEL_800"), ("O", "000300", "in", "ADD_800"),
         ("PA", "000852", "in", "ADD_pure"), ("PA", "000852", "out", "DEL_pure")]


def main():
    warnings.simplefilter("ignore")
    p = Panel()
    m = sim_mask(p)
    W = baseline_W(p, m)
    rm = RiskModel.load("open10")
    assert rm.codes.equals(p.codes)
    E = pd.read_parquet(TMP / "inc_t2_events.parquet")
    rows = []
    for src, idx, d, et in CELLS:
        g = E[(E["source"] == src) & (E["index"] == idx) & (E["direction"] == d) & (E["etype"] == et) & E["q_A"]
              & E["window"].isin(["dev", "test"])]
        for rv, gg in g.groupby("review"):
            t = int(gg["A_i"].iloc[0])
            w = np.zeros(len(p.codes))
            w[gg["j"].to_numpy()] = 1.0 / len(gg)
            wb = np.asarray(W[t], np.float64)
            try:
                e = rm.active_exposure(w, wb, p.dates[t], strict=False)
            except KeyError:
                continue
            rec = dict(source=src, index=idx, direction=d, etype=et, review=rv, window=gg["window"].iloc[0],
                       n=len(gg))
            rec.update({f: float(e[f]) for f in rm.style_names})
            rows.append(rec)
    R = pd.DataFrame(rows)
    agg = []
    for key, g in R.groupby(["source", "index", "direction", "etype"]):
        for wn, ws in (("dev", ["dev"]), ("test", ["test"]), ("all", ["dev", "test"])):
            gw = g[g["window"].isin(ws)]
            rec = dict(zip(["source", "index", "direction", "etype"], key), review=f"mean_{wn}", window=wn,
                       n=int(gw["n"].sum()))
            rec.update({f: float(gw[f].mean()) for f in rm.style_names})
            agg.append(rec)
    out = pd.concat([R, pd.DataFrame(agg)], ignore_index=True)
    out.to_csv(LEDGER / "inc_t2_barra_exposure.csv", index=False)
    print(pd.DataFrame(agg)[["source", "index", "direction", "etype", "window"] + rm.style_names].round(2).to_string())


if __name__ == "__main__":
    main()
