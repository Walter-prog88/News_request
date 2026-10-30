"""INC-T4 补充检查与诊断(预注册 §3.5、§7.1、§7.3、H2;不改变任何判定对象)。

  1. 引擎补充断言(写入 inc_t4_checks.csv):
     - eqv_weights:W_excl(z(COMP_v5)) 与 W_baseline(ST.build_weights 原始打分)的权重重叠 —— 保证 a_Csm 对照与基线
       走同一套排雷构造(两条打分路径只差 ops.standardize 的去极值 / z,对排名几乎无影响);
     - v1rep:V1rep(本轨道重算的 V1 相位 0)与 ie_V1_ph0 日收益逐日相同;
  2. 持仓重叠表 inc_t4_overlap.csv:各主配置与基线的权重重叠 Σmin(w_v, w_b)(解释配对 TE 小、MDE 大的原因);
  3. H2 分解表 inc_t4_h2.csv:被第二层额外剔除的股票等权组合(W1>0 且 W2=0)相对同频基准的**毛**超额与 NW t;
  4. 事后诊断(**不参与判定**,单列于报告"偏离记录"):持仓数匹配的 COMP_v5 单层 (0.21, 0.36) 对照 ——
     预注册深剔网格最浅为 (0.25, 0.40),比 (b) 主配置深得多,持仓数不匹配,所以补一个浅的。
用法:python3 scripts/inc_t4_extra.py
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
from afactor.config import LEDGER, TMP  # noqa: E402
from afactor import strategy as ST  # noqa: E402

warnings.filterwarnings("ignore")


def overlap(W1: np.ndarray, W2: np.ndarray, mask: np.ndarray) -> float:
    m = mask & (W1.sum(1) > 0) & (W2.sum(1) > 0)
    return float(np.minimum(W1[m], W2[m]).sum(1).mean())


def main():
    C.check_prereg()
    hs = AB.read_hl_star()
    p = AB.P()
    sm = AB.sim_mask()
    Wb = AB.W_baseline()
    rows = []
    # ---- 1. 等价性 / 复现
    Wc = AB.W_excl(AB.zC(0))
    ov = overlap(Wc, Wb, sm)
    rows.append(dict(check="eqv_weights", item="W_excl(z(COMP_v5)) vs W_baseline(原始打分) 权重重叠", value=ov,
                     reference=1.0, diff=ov - 1.0, passed=ov > 0.999, note="两条打分路径只差 standardize 去极值"))
    f0 = TMP / f"{C.PFX}_ie_V1rep_ph0_daily.parquet"
    if f0.exists():
        a = pd.read_parquet(f0)["ret"]
        b = pd.read_parquet(TMP / "ie_V1_ph0_daily.parquet")["ret"]
        a, b = a.align(b, join="inner")
        mx = float((a - b).abs().max())
        rows.append(dict(check="v1rep", item="V1rep 与 ie_V1_ph0 日收益最大绝对差", value=mx, reference=0.0, diff=mx,
                         passed=mx < 1e-9, note=f"n_days={len(a)}"))
    AB.write_check(rows)
    print(pd.DataFrame(rows).to_string(), flush=True)

    # ---- 2. 持仓重叠
    orows = []
    zd = AB.zD("A", hs)
    zdm = AB.zD("main", hs)
    for lab, W in {
        f"(a) S1 D_A HL{hs} 排雷 20/35": AB.W_excl(AB.zC(0) + zd),
        f"(a) COMP_v5 自身平滑 HL{hs}": AB.W_excl(AB.zC(hs)),
        f"(a) D_A HL{hs} 单独排雷": AB.W_excl(zd),
        f"(b) 两层 10/35 D_A HL{hs}": AB.W_two_layer(zd, *C.B_MAIN)[0],
        f"(b) 两层 10/35 D_main HL{hs}": AB.W_two_layer(zdm, *C.B_MAIN)[0],
    }.items():
        orows.append(dict(config=lab, overlap_full=overlap(W, Wb, sm & (p.dates.year >= 2021)),
                          overlap_dev=overlap(W, Wb, sm & np.asarray((p.dates.year >= 2021) & (p.dates.year <= 2023))),
                          overlap_test=overlap(W, Wb, sm & np.asarray(p.dates.year >= 2024))))
    for e, x in C.DEEP_GRID + [(0.21, 0.36)]:
        Wd = ST.build_weights(AB.comp_raw(), p, C.VERSION, scheme="exclude", buffer=(e, x), date_mask=sm)
        orows.append(dict(config=f"COMP_v5 单层 {e:.2f}/{x:.2f}", overlap_full=overlap(Wd, Wb, sm & (p.dates.year >= 2021)),
                          overlap_dev=overlap(Wd, Wb, sm & np.asarray((p.dates.year >= 2021) & (p.dates.year <= 2023))),
                          overlap_test=overlap(Wd, Wb, sm & np.asarray(p.dates.year >= 2024))))
    OV = pd.DataFrame(orows)
    OV.to_csv(LEDGER / f"{C.PFX}_overlap.csv", index=False)
    print(OV.round(4).to_string(), flush=True)

    # ---- 4. 持仓数匹配的浅对照(事后诊断,不参与判定)
    Wd = ST.build_weights(AB.comp_raw(), p, C.VERSION, scheme="exclude", buffer=(0.21, 0.36), date_mask=sm)
    AB.run_cell("deep_21_36", Wd, "Y1.0", dict(method="diag", kind="comp_deep_shallow", set="", hl=0, lam=0.0,
                                               e2=0.21, x2=0.36))

    # ---- 3. H2:额外剔除集的毛超额
    B = AB.bench()["pool"]
    hrows = []
    for s in ("A", "main"):
        cid = f"bE_{s}_hl{hs}_10_35"
        d = pd.read_parquet(AB.cell_file(cid))
        for w in ("full", "dev", "test", "y2025"):
            dw = C.win_slice(d, w)
            rb = B.reindex(dw.index)
            g = dw["gross_ret"] - rb
            n_ = dw["ret"] - rb
            hrows.append(dict(set=s, window=w, n_days=len(dw), holdings=float(dw["n_hold"].mean()),
                              turnover_daily=float(dw["turnover"].mean()),
                              gross_excess_arith=float(g.mean() * C.TD), gross_t_nw=C.nw_t(g.to_numpy()),
                              net_legacy_excess_arith=float(n_.mean() * C.TD), net_t_nw=C.nw_t(n_.to_numpy())))
    H2 = pd.DataFrame(hrows)
    H2.to_csv(LEDGER / f"{C.PFX}_h2.csv", index=False)
    print(H2.round(4).to_string(), flush=True)


if __name__ == "__main__":
    main()
