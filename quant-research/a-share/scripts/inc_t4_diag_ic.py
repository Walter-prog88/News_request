"""INC-T4 事后诊断(**不参与判定**,报告里单列于"偏离记录 / 事后诊断"):
在 V1000 决策池、5 日前向收益(open10,与指增 α 的 IC 预测同口径)上,S_λ = z(COMP_v5) + λ·z(D_HL) 的日度 rank IC
随 λ 的变化,以及与 λ=0(COMP_v5 单独)的配对 IC 差的 NW t。回答"D 在等权叠加里是增加还是稀释了 COMP_v5 的信息"。
产出 research/ledger/inc_t4_diag_ic.csv。
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import inc_t4_common as C  # noqa: E402
import inc_t4_run_ab as AB  # noqa: E402
from afactor.config import LEDGER  # noqa: E402
from afactor import optimizer as OPT, strategy as ST  # noqa: E402

warnings.filterwarnings("ignore")
LAMS = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, None]            # None = D 单独


def main():
    C.check_prereg()
    hs = AB.read_hl_star()
    p = AB.P()
    U = ST.universe(p, C.VERSION)
    fwd = p.fwd_ret(5, "open")
    yrs = np.asarray(p.dates.year)
    rows = []
    for s in ("A", "main"):
        for hl in sorted({0, hs}):
            zd, zc = AB.zD(s, hl), AB.zC(0)
            ic0 = None
            for lam in LAMS:
                S = zd if lam is None else zc + lam * zd
                ic = OPT.rank_ic_series(S, fwd, U)
                if lam == 0.0:
                    ic0 = ic
                for w, (a, b) in {"full": (2021, 2025), "dev": (2021, 2023), "test": (2024, 2025)}.items():
                    m = (yrs >= a) & (yrs <= b)
                    x = ic.to_numpy()[m]
                    x = x[np.isfinite(x)]
                    dx = (ic - ic0).to_numpy()[m] if ic0 is not None else np.array([np.nan])
                    dx = dx[np.isfinite(dx)]
                    rows.append(dict(set=s, hl=hl, lam="D_only" if lam is None else lam, window=w, n=len(x),
                                     ic_mean=float(x.mean()), ic_t_nw=C.nw_t(x, 10),
                                     d_ic_vs_comp=float(dx.mean()) if len(dx) else np.nan,
                                     d_ic_t_nw=C.nw_t(dx, 10) if len(dx) > 30 else np.nan))
    df = pd.DataFrame(rows)
    df.to_csv(LEDGER / f"{C.PFX}_diag_ic.csv", index=False)
    print(df[df.set == "A"].round(4).to_string())


if __name__ == "__main__":
    main()
