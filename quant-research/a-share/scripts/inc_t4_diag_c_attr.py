"""INC-T4 事后诊断(**不参与判定**):方法 (c) 里"α 叠加 D"相对 V1 的日收益差来自哪里。

做法:ΔH = H(S1_A_hl5, 相位 0) − H(V1, 相位 0)(基准在差里抵消),用 RiskModel.attribute(ΔH, R, by_factor=True)
把 ΔH 的日收益拆成 Country / 风格 / 行业 / 特质;另拆出风险模型缺失的 20 日反转(Rev20)暴露的贡献
(口径同 run_index_enhance.py:e20 = Σ z20·a,g20 = 风险模型残差对 z20 的 sqrt(市值) 加权截面回归斜率)。
并报告 ΔH 的平均风格暴露与 Rev20 暴露。产出 research/ledger/inc_t4_diag_c_attr.csv。
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import inc_t4_common as C  # noqa: E402
import inc_t4_run_c as RC  # noqa: E402
from afactor.config import LEDGER, TMP  # noqa: E402
from afactor import barra as BR, factors as FA, strategy as ST  # noqa: E402
from afactor.barra import RiskModel  # noqa: E402

warnings.filterwarnings("ignore")


def main(job: str):
    C.check_prereg()
    p = RC.P()
    dates, codes = p.dates, p.codes
    rm = RiskModel.load("open10")
    H1 = pd.read_parquet(TMP / f"{C.PFX}_ie_{job}_ph0_holdings.parquet")
    H0 = pd.read_parquet(TMP / "ie_V1_ph0_holdings.parquet")
    idx = H1.index.intersection(H0.index)
    idx = idx[idx >= RC.START]
    dH = H1.reindex(index=idx, columns=codes).fillna(0.0) - H0.reindex(index=idx, columns=codes).fillna(0.0)
    R = BR.fwd_returns(p, "open10")
    att, fac = rm.attribute(dH, R, by_factor=True)
    # 缺失因子 Rev20
    f = TMP / "ie_rev20_z.parquet"
    z20 = pd.read_parquet(f).reindex(index=dates, columns=codes).to_numpy(np.float64)
    U = ST.universe(p, "V1000").to_numpy(bool)
    mvv = FA._real(p, "mv").reindex(index=dates, columns=codes).to_numpy(np.float64)
    resid = rm.resid()
    Rs = np.asarray(resid[:len(R)], np.float64)
    g20 = np.full(len(dates), np.nan)
    first = int(np.searchsorted(dates, RC.START))
    for t in range(first, len(R)):
        m = U[t] & np.isfinite(Rs[t]) & np.isfinite(z20[t]) & np.isfinite(mvv[t]) & (mvv[t] > 0)
        if m.sum() < 200:
            continue
        ww = np.sqrt(np.sqrt(mvv[t, m]))
        A_ = np.column_stack([np.ones(m.sum()), z20[t, m]])
        g20[t] = np.linalg.lstsq(A_ * ww[:, None], Rs[t, m] * ww, rcond=None)[0][1]
    pos = dates.get_indexer(att.index)
    e20 = pd.Series([float(np.nansum(np.nan_to_num(z20[i]) * dH.loc[t].to_numpy())) for i, t in zip(pos, att.index)],
                    index=att.index)
    att["rev20"] = e20 * pd.Series(g20[pos], index=att.index)
    att["specific_ex_rev20"] = att["specific"] - att["rev20"].fillna(0.0)
    rows = []
    sty = list(rm.style_names)
    for wn, w in (("full", "full"), ("dev", "dev"), ("test", "test")):
        a, b = C.WINDOWS[w]
        sel = (att.index.year >= a) & (att.index.year <= b)
        if wn == "full":                                   # 与 inc_t4 其余口径一致:2021-2025,但 (c) 从 2021-04-20 起
            pass
        g = att[sel]
        for comp in ["total", "country", "style", "industry", "specific", "rev20", "specific_ex_rev20"]:
            x = g[comp].dropna()
            rows.append(dict(job=job, window=wn, component=comp, ann_pp=float(x.mean() * C.TD * 100),
                             t_nw=C.nw_t(x.to_numpy()), n_days=len(x)))
        gf = fac[sel]
        for s_ in sty:
            x = gf[s_]
            rows.append(dict(job=job, window=wn, component=f"style:{s_}", ann_pp=float(x.mean() * C.TD * 100),
                             t_nw=C.nw_t(x.to_numpy()), n_days=len(x)))
        # 平均暴露差
        Xd = []
        for i, t in zip(pos[sel], att.index[sel]):
            m = np.asarray(rm.mask[i]) if np.isfinite(rm.cov[i, 0, 0]) else np.isfinite(np.asarray(rm.X[i])[:, 0])
            X = np.nan_to_num(np.asarray(rm.X[i])[m].astype(np.float64))
            Xd.append(X.T @ dH.loc[t].to_numpy()[m])
        Xd = pd.DataFrame(Xd, columns=sty)
        for s_ in sty:
            rows.append(dict(job=job, window=wn, component=f"exposure_diff:{s_}", ann_pp=float(Xd[s_].mean()),
                             t_nw=C.nw_t(Xd[s_].to_numpy()), n_days=len(Xd)))
        rows.append(dict(job=job, window=wn, component="exposure_diff:Rev20", ann_pp=float(e20[sel].mean()),
                         t_nw=C.nw_t(e20[sel].to_numpy()), n_days=int(sel.sum())))
    df = pd.DataFrame(rows)
    df.to_csv(LEDGER / f"{C.PFX}_diag_c_attr.csv", index=False)
    print(df.round(4).to_string())


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else f"S1_A_hl{RC.hl_star()}")
