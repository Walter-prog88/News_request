"""T5 事后 Barra 暴露解释(只在 B-MIN,不进判定;预注册 §9 末条)。

月末(决策日)各篮子 / 组合相对池等权的主动风格暴露(barra.RiskModel.active_exposure,Country 暴露 = 主动权重和)。
篮子权重与 inc_t5_runs.stage_runs_min 同构(Panel,V1000,date_mask 2021-2025)。
用法: python3 scripts/inc_t5_barra.py
"""
from __future__ import annotations

import time

from inc_t5_common import *      # noqa: F401,F403
from inc_t5_common import LEDGER, WIN, log

import numpy as np
import pandas as pd

from afactor import inc_t5_stats as S5
from afactor.barra import RiskModel
from inc_t5_runs import MinCtx, STYLE6


def main():
    t0 = time.time()
    c = MinCtx()
    rm = RiskModel.load("open10")
    log(f"[barra] 载入 {time.time() - t0:.0f}s;风格因子 {[f for f in rm.factor_names if not f.startswith('行业')][:12]}")
    W = {}
    for b in STYLE6:
        W[b] = c.W(b, "20_35" if b == "LIQ" else "45_60")
    W["EW5"] = S5.combine_weights([W[m] for m in STYLE6[:5]], np.full(5, 0.2))
    W["CORE5"] = S5.combine_weights([W[m] for m in STYLE6[:5]], S5.core_weights(5, 0, 0.5))
    W["S4"] = S5.combine_weights([W[m] for m in STYLE6[1:5]], np.full(4, 0.25))
    U = c.U.to_numpy(bool) & c.mask[:, None]
    n = U.sum(1, keepdims=True)
    WB = np.divide(U.astype(np.float64), n, out=np.zeros(U.shape), where=n > 0)
    sel = pd.Series(np.arange(len(c.dates)), index=c.dates).loc[WIN["min"]["full"][0]:WIN["min"]["full"][1]]
    me = sel.groupby([sel.index.year, sel.index.month]).last().to_numpy()
    rows = []
    for name, w in W.items():
        ex = []
        for i in me:
            t = c.dates[i]
            if not rm.available(t) or WB[i].sum() == 0:
                continue
            e = rm.active_exposure(pd.Series(np.asarray(w[i], np.float64), index=c.codes), pd.Series(WB[i], index=c.codes), t,
                                   strict=False)
            ex.append(e)
        E = pd.DataFrame(ex)
        sty = [f for f in E.columns if not str(f).startswith("行业") and f != "Country"]
        r = E[sty].mean()
        rows.append(dict(config=name, n_months=len(E), **{f"exp_{k}": float(v) for k, v in r.items()}))
    out = pd.DataFrame(rows)
    out.to_csv(LEDGER / "inc_t5_barra_exposure.csv", index=False)
    log(f"[barra] 完成 {time.time() - t0:.0f}s")
    print(out.set_index("config").T.round(3).to_string())


if __name__ == "__main__":
    main()
