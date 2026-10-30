"""INC-T4 阶段 0:机制诊断(预注册 §5)。只用于解释,不得用于改动 §6 的网格或主配置。

  1. IC 期限结构:h ∈ {1,5,10,20} 的日度 Rank IC(EvalContext(p,"open",(1,5,10,20)),fast_eval.eval_one),
     D_A / D_main / COMP_v5,窗口 2021-2025(全期)、2021-2023(开发)、2024-2025(检验),另列 2020-2025;
     比值 IC(h5)/IC(h1)、IC(h20)/IC(h1);NW t 带宽 max(5, 2h)。
  2. 信号持续性:截面秩的 lag-{1,5,10,20} 秩相关(spearman(rank_t, rank_{t-k}),池内两期都有值)的时序均值。
  3. 平滑后的信息保留:各 HL 下 z(D_HL) 的 h1 / h5 IC 与 lag-5 秩自相关(D_A、D_main),COMP_v5 自身平滑同列。
  4. D 单独在标准排雷下的换手 / 成本 / 超额:取自方法 (a) 网格的 a_D_{set}_hl0 格(本脚本 summary 时并表)。
产出:research/ledger/inc_t4_stage0_diag.csv
"""
from __future__ import annotations

import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from scipy.stats import rankdata

sys.path.insert(0, os.path.dirname(__file__))
import inc_t4_common as C  # noqa: E402
from afactor.config import LEDGER, TMP  # noqa: E402
from afactor.panel import Panel  # noqa: E402
from afactor.fast_eval import EvalContext, eval_one  # noqa: E402
from afactor import ops  # noqa: E402

warnings.filterwarnings("ignore")
HS = (1, 5, 10, 20)
LAGS = (1, 5, 10, 20)
WINS = {"2021-2025": (2021, 2025), "2021-2023": (2021, 2023), "2024-2025": (2024, 2025), "2020-2025": (2020, 2025)}


def rank_autocorr(F: np.ndarray, M: np.ndarray, lags=LAGS) -> dict:
    """逐日 spearman(rank_t, rank_{t-k}):两期都在池内且有值的股票。返回 {k: 日度序列}。"""
    T = F.shape[0]
    ok = M & np.isfinite(F)
    out = {k: np.full(T, np.nan) for k in lags}
    for t in range(T):
        for k in lags:
            if t - k < 0:
                continue
            m = ok[t] & ok[t - k]
            if m.sum() < 200:
                continue
            a = rankdata(F[t, m]); b = rankdata(F[t - k, m])
            a -= a.mean(); b -= b.mean()
            den = np.sqrt((a * a).sum() * (b * b).sum())
            out[k][t] = float((a * b).sum() / den) if den > 0 else np.nan
    return out


def main():
    C.check_prereg()
    t0 = time.time()
    p = Panel()
    ctx = EvalContext(p, "open", HS)
    yrs = np.asarray(p.dates.year)
    u = p["universe"]
    Mstat = ctx.M[1]                                # 池(universe_tradable)× h1 前向收益可得
    comp = pd.read_parquet(TMP / "comp_v5_score_open10.parquet").reindex(index=p.dates, columns=p.codes)
    rows = []

    def diag(label, F_df, hl, do_terms=True, lags=LAGS):
        F = F_df.reindex(index=p.dates, columns=p.codes).to_numpy(np.float32)
        ics = {h: eval_one(F, ctx.R[h], ctx.M[h], want_layers=False)[0] for h in (HS if do_terms else (1, 5))}
        ac = rank_autocorr(F, Mstat, lags)
        for wn, (a, b) in WINS.items():
            m = (yrs >= a) & (yrs <= b)
            r = dict(series=label, hl=hl, window=wn)
            for h, ic in ics.items():
                x = ic[m]
                r[f"ic_h{h}"] = float(np.nanmean(x))
                r[f"ic_h{h}_t_nw"] = C.nw_t(x[np.isfinite(x)], max(5, 2 * h))
            r["ratio_h5_h1"] = r["ic_h5"] / r["ic_h1"] if r["ic_h1"] else np.nan
            if "ic_h20" in r:
                r["ratio_h10_h1"] = r["ic_h10"] / r["ic_h1"] if r["ic_h1"] else np.nan
                r["ratio_h20_h1"] = r["ic_h20"] / r["ic_h1"] if r["ic_h1"] else np.nan
            for k, s in ac.items():
                r[f"rank_ac_lag{k}"] = float(np.nanmean(s[m]))
            rows.append(r)
        print(f"  {label} hl={hl} {time.time()-t0:.0f}s  " +
              " ".join(f"{k}={v:.4f}" for k, v in rows[-4].items() if k.startswith(("ic_h", "ratio", "rank_ac")) and "t_nw" not in k),
              flush=True)

    # 1 + 2:期限结构与持续性(未平滑)
    diag("COMP_v5", comp, 0)
    for s in ("A", "main"):
        diag(f"D_{s}", pd.read_parquet(C.d_path(s)), 0)
    # 3:平滑后的信息保留(z(D_HL)、z(COMP_v5_HL)),只算 h1/h5 与 lag-5
    for hl in [h for h in C.HL_GRID if h > 0]:
        for s in ("A", "main"):
            Ds = C.ewm_smooth(pd.read_parquet(C.d_path(s)), hl)
            diag(f"z(D_{s}_HL)", ops.standardize(Ds, u), hl, do_terms=False, lags=(5,))
            del Ds
        diag("z(COMP_v5_HL)", ops.standardize(C.ewm_smooth(comp.astype(np.float32), hl), u), hl, do_terms=False, lags=(5,))
    df = pd.DataFrame(rows)
    df.to_csv(LEDGER / f"{C.PFX}_stage0_diag.csv", index=False)
    print(df.to_string(float_format=lambda x: f"{x:.4f}"))
    print(f"done {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
