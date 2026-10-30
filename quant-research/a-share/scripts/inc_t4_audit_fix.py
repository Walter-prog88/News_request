"""INC-T4 审查修复:对审查发现的两处机制性结论做实测(**不参与判定**,单列于报告"审查记录")。

审查 P1:报告原文说"D 做 HL5 平滑后总 IC 升高 -> 信息没有快速衰减、剩下的是逐日噪声"。这个推断从未在
**D 去掉 COMP_v5 已有部分之后的残差**上检验过。本脚本直接测:

  resid   对每个 HL,逐日截面把 z(D_A,HL) 对 z(COMP_v5) 回归取残差(与 run_midband_composite.py 的
          ops.cs_neutralize_1d 同法,即 2026-08-25 账本"D组对COMP_v5正交后"的口径),报告
          ① 残差 Rank IC 期限结构(h1/h5/h10/h20,全期 2021-2025 / 开发窗 / 检验窗,NW t);
          ② 先取 HL0 残差再做 EWM 平滑("平滑后的残差")的 IC;
          ③ z(D_HL) 与 z(COMP_v5) 的截面相关(Pearson / Spearman 日均);
          ④ HL0 残差与原始 D 的秩自相关(lag1/5/10/20);
          ⑤ S = z(COMP_v5) + κ·z(残差) 的 5 日 IC 随 κ 的变化(残差最多能给 COMP_v5 加多少 IC);
          ⑥ 与 2026-08-25 账本 midband_composite.csv 的核对(稳健A 持出窗 2024-2025 h1)。
  h2      被第二层额外剔除的股票 vs (b) 保留的股票:毛收益差、权重占比、"要达到 +1pp 所需的差"。

产出:research/ledger/inc_t4_diag_resid.csv、inc_t4_diag_resid_misc.csv、inc_t4_h2b.csv。
用法:python3 scripts/inc_t4_audit_fix.py [resid|h2|all]
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
from afactor import ops, optimizer as OPT, strategy as ST  # noqa: E402

warnings.filterwarnings("ignore")
HS = (1, 5, 10, 20)
HLS = (0, 2, 5, 10, 20)
WINS = {"full": (2021, 2025), "dev": (2021, 2023), "test": (2024, 2025)}


def ic_rows(tag: str, S: pd.DataFrame, fwds: dict, U: pd.DataFrame, yrs, extra: dict) -> list:
    rows = []
    for h in HS:
        ic = OPT.rank_ic_series(S, fwds[h], U)
        for w, (a, b) in WINS.items():
            x = ic.to_numpy()[(yrs >= a) & (yrs <= b)]
            x = x[np.isfinite(x)]
            rows.append(dict(series=tag, h=h, window=w, n=len(x), ic_mean=float(x.mean()),
                             ic_t_nw=C.nw_t(x, max(5, 2 * h)), **extra))
    return rows


def xs_corr(A: np.ndarray, B: np.ndarray, rank: bool, mask_rows: np.ndarray) -> float:
    from scipy.stats import rankdata
    cs = []
    for t in np.flatnonzero(mask_rows):
        m = np.isfinite(A[t]) & np.isfinite(B[t])
        if m.sum() < 200:
            continue
        a, b = A[t, m], B[t, m]
        if rank:
            a, b = rankdata(a), rankdata(b)
        cs.append(np.corrcoef(a, b)[0, 1])
    return float(np.mean(cs))


def rank_autocorr(X: np.ndarray, lag: int, mask_rows: np.ndarray) -> float:
    from scipy.stats import rankdata
    cs = []
    for t in np.flatnonzero(mask_rows):
        if t - lag < 0:
            continue
        m = np.isfinite(X[t]) & np.isfinite(X[t - lag])
        if m.sum() < 200:
            continue
        cs.append(np.corrcoef(rankdata(X[t, m]), rankdata(X[t - lag, m]))[0, 1])
    return float(np.mean(cs))


def run_resid() -> None:
    C.check_prereg()
    p = AB.P()
    U = ST.universe(p, C.VERSION)                       # = p["universe"](已核对 0 格不同)
    yrs = np.asarray(p.dates.year)
    fwds = {h: p.fwd_ret(h, "open") for h in HS}
    zc = AB.zC(0)
    rows, misc = [], []
    rows += ic_rows("COMP_v5", zc, fwds, U, yrs, dict(kind="ref", hl=np.nan))
    resid_hl0 = None
    for hl in HLS:
        zd = AB.zD("A", hl)
        rs = ops.cs_neutralize_1d(zd, zc)               # 逐日截面 OLS 残差(D_HL 对 COMP_v5)
        if hl == 0:
            resid_hl0 = rs
        rows += ic_rows(f"D_A_HL{hl}", zd, fwds, U, yrs, dict(kind="D_total", hl=hl))
        rows += ic_rows(f"resid(D_A_HL{hl}|COMP_v5)", rs, fwds, U, yrs, dict(kind="resid_of_smoothed", hl=hl))
        yy = yrs >= 2021
        A, B = zd.to_numpy(np.float64), zc.to_numpy(np.float64)
        misc.append(dict(item="xs_corr_pearson", hl=hl, value=xs_corr(A, B, False, yy)))
        misc.append(dict(item="xs_corr_spearman", hl=hl, value=xs_corr(A, B, True, yy)))
        print(f"HL{hl} done", flush=True)
    # 先取 HL0 残差、再做 EWM 平滑(回答:残差里的信息持续吗)
    for hl in (2, 5, 10, 20):
        sm = C.ewm_smooth(resid_hl0.astype(np.float32), hl)
        sm = ops.standardize(sm, p["universe"])
        rows += ic_rows(f"smooth(resid(D_A_HL0|COMP_v5),HL{hl})", sm, fwds, U, yrs,
                        dict(kind="smoothed_resid", hl=hl))
        print(f"smoothed resid HL{hl} done", flush=True)
    # 秩自相关:HL0 残差 / 原始 D / COMP_v5
    yy = yrs >= 2021
    sm_series = {f"smooth(resid_HL0,HL{h})": ops.standardize(C.ewm_smooth(resid_hl0.astype(np.float32), h), p["universe"])
                 for h in (2, 5, 10)}
    for lab, X in (("resid_HL0", resid_hl0), ("D_A_HL0", AB.zD("A", 0)), ("COMP_v5", zc), *sm_series.items()):
        Xn = X.to_numpy(np.float64)
        for lag in (1, 5, 10, 20):
            misc.append(dict(item=f"rank_autocorr_lag{lag}", series=lab, value=rank_autocorr(Xn, lag, yy)))
    # κ 曲线:S = z(COMP_v5) + κ·z(残差),5 日 IC,残差最多能给 COMP_v5 加多少
    fw5 = fwds[5]
    ic0 = OPT.rank_ic_series(zc, fw5, U)
    for hl in (0, 5):
        rs = ops.cs_neutralize_1d(AB.zD("A", hl), zc)
        zr = ops.standardize(rs, p["universe"])
        for kap in (0.1, 0.25, 0.5, 1.0, 2.0):
            ic = OPT.rank_ic_series(zc + kap * zr, fw5, U)
            for w, (a, b) in WINS.items():
                m = (yrs >= a) & (yrs <= b)
                x = ic.to_numpy()[m]
                d = (ic - ic0).to_numpy()[m]
                x, d = x[np.isfinite(x)], d[np.isfinite(d)]
                misc.append(dict(item="kappa_curve_h5", hl=hl, kappa=kap, window=w, ic_mean=float(x.mean()),
                                 d_ic_vs_comp=float(d.mean()), d_ic_t_nw=C.nw_t(d, 10)))
        print(f"kappa curve HL{hl} done", flush=True)
    R = pd.DataFrame(rows)
    R.to_csv(LEDGER / f"{C.PFX}_diag_resid.csv", index=False)
    M = pd.DataFrame(misc)
    # 与 2026-08-25 账本核对:稳健A 持出窗 2024-2025 的 h1 IC(原账本"D中段组"/"D组对COMP_v5正交后")
    old = pd.read_csv(LEDGER / "midband_composite.csv")
    oa = old[(old["切法"] == "稳健A")].set_index("组")
    mine_tot = R[(R.series == "D_A_HL0") & (R.h == 1) & (R.window == "test")].ic_mean.iloc[0]
    mine_res = R[(R.series == "resid(D_A_HL0|COMP_v5)") & (R.h == 1) & (R.window == "test")].ic_mean.iloc[0]
    M = pd.concat([M, pd.DataFrame([
        dict(item="check_vs_midband_composite", series="D_A 总 IC h1 2024-2025", value=mine_tot,
             reference=float(oa.loc["D中段组", "ic_mean"]), diff=mine_tot - float(oa.loc["D中段组", "ic_mean"])),
        dict(item="check_vs_midband_composite", series="D_A 残差 IC h1 2024-2025", value=mine_res,
             reference=float(oa.loc["D组对COMP_v5正交后", "ic_mean"]),
             diff=mine_res - float(oa.loc["D组对COMP_v5正交后", "ic_mean"])),
    ])], ignore_index=True)
    M.to_csv(LEDGER / f"{C.PFX}_diag_resid_misc.csv", index=False)
    pd.set_option("display.width", 250)
    pv = R[R.window.isin(["full", "test"])].pivot_table(index=["series", "window"], columns="h", values="ic_mean")
    pt = R[R.window.isin(["full", "test"])].pivot_table(index=["series", "window"], columns="h", values="ic_t_nw")
    print(pv.round(4).to_string())
    print(pt.round(2).to_string())
    print(M.round(4).to_string())


def run_h2() -> None:
    """被第二层额外剔除的股票 vs (b) 保留的股票。三个等权组合(同一套 5 日调仓 5 相位引擎、同日历)的毛收益:
       base = 基线(W1)、b = (b) 主配置(W1&W2,即"保留")、E = 额外剔除集(W1 且 非 W2)。"""
    C.check_prereg()
    hs = AB.read_hl_star()
    B = AB.bench()["pool"]
    base = pd.read_parquet(AB.cell_file("base_Y1.0"))
    rows = []
    for s in ("A", "main"):
        b = pd.read_parquet(AB.cell_file(f"b_{s}_hl{hs}_10_35"))
        e = pd.read_parquet(AB.cell_file(f"bE_{s}_hl{hs}_10_35"))
        for w in ("full", "dev", "test"):
            bw, ew, bsw = C.win_slice(b, w), C.win_slice(e, w), C.win_slice(base, w)
            rb = B.reindex(bw.index)
            gap = (bw["gross_ret"] - ew["gross_ret"].reindex(bw.index))           # 保留 − 额外剔除(毛)
            vs_pool = ew["gross_ret"].reindex(bw.index) - rb                       # 额外剔除 − 池基准(毛)
            keep_pool = bw["gross_ret"] - rb                                        # 保留 − 池基准(毛)
            d_gross = (bw["gross_ret"] - bsw["gross_ret"].reindex(bw.index))        # (b) − 基线(毛,实测)
            w_e = float((ew["n_hold"].reindex(bw.index) / bsw["n_hold"].reindex(bw.index)).mean())
            gap_ann = float(gap.mean() * C.TD)
            rows.append(dict(
                set=s, window=w, n_days=len(bw), n_extra=float(ew["n_hold"].mean()), n_base=float(bsw["n_hold"].mean()),
                weight_extra=w_e, extra_vs_pool_gross=float(vs_pool.mean() * C.TD), extra_vs_pool_t=C.nw_t(vs_pool.to_numpy()),
                kept_vs_pool_gross=float(keep_pool.mean() * C.TD), kept_vs_pool_t=C.nw_t(keep_pool.to_numpy()),
                gap_kept_minus_extra=gap_ann, gap_t_nw=C.nw_t(gap.to_numpy()),
                implied_gain_w_x_gap=w_e * gap_ann, measured_d_gross_b_minus_base=float(d_gross.mean() * C.TD),
                required_gap_for_1pp=0.01 / w_e))
    H = pd.DataFrame(rows)
    H.to_csv(LEDGER / f"{C.PFX}_h2b.csv", index=False)
    # (b) 各格:要达到 +1pp 毛增量,被多剔除的那部分股票需要比保留的差多少(w_e ≈ 1 − 持仓数/基线持仓数,开发窗)
    gb = pd.read_csv(LEDGER / f"{C.PFX}_grid_b.csv")
    hb = float(gb[(gb.cell == "base_Y1.0") & (gb.window == "dev")].holdings.iloc[0])
    cr = []
    for cid in [f"b_A_hl{h}_{e:02d}_{x:02d}" for h in (0, hs) for e, x in ((10, 25), (10, 35), (20, 35), (20, 45))]:
        r = gb[(gb.cell == cid) & (gb.window == "dev")]
        if len(r):
            we = 1 - float(r.holdings.iloc[0]) / hb
            cr.append(dict(cell=cid, dev_holdings=float(r.holdings.iloc[0]), weight_extra_approx=we,
                           required_gap_for_1pp=0.01 / we if we > 0 else np.nan))
    pd.DataFrame(cr).to_csv(LEDGER / f"{C.PFX}_h2b_cells.csv", index=False)
    pd.set_option("display.width", 250)
    print(H.round(4).to_string())
    print(pd.DataFrame(cr).round(4).to_string())


def run_ceiling() -> None:
    """毛收益天花板:所有 D 叠加格((a) S1/S0.5/S2、(b) 两层;D_A/D_main/D_B 及 D′;不含成本敏感性重复格、深剔/COMP 自平滑
    对照、D 单独、H2 的 bE 分解格)相对基线的**成本前**毛超额增量(算术,pp)。回答:就算交易成本为零,能不能过 +1pp。"""
    C.check_prereg()
    ga = pd.read_csv(LEDGER / f"{C.PFX}_grid_a.csv")
    gb = pd.read_csv(LEDGER / f"{C.PFX}_grid_b.csv")
    g = pd.concat([ga, gb]).drop_duplicates(["cell", "window"])
    base = g[g.cell == "base_Y1.0"].set_index("window")
    g = g[g.window.isin(["full", "dev", "test"])].copy()
    g["d_gross_pp"] = [(r.gross_excess_arith - base.loc[r.window, "gross_excess_arith"]) * 100 for r in g.itertuples()]
    g["d_cost_bp"] = [r.cost_total_bp - base.loc[r.window, "cost_total_bp"] for r in g.itertuples()]
    m = g[~g.cell.str.contains("legacy|Y1.5|base|Csm|deep|bE_|a_D_|diag")]
    rows = []
    for w in ("full", "dev", "test"):
        q = m[m.window == w].sort_values("d_gross_pp")
        rows.append(dict(window=w, n_cells=int(q.cell.nunique()), max_d_gross_pp=float(q.d_gross_pp.iloc[-1]),
                         max_cell=q.cell.iloc[-1], max_cell_d_cost_bp=float(q.d_cost_bp.iloc[-1]),
                         median_d_gross_pp=float(q.d_gross_pp.median()), min_d_gross_pp=float(q.d_gross_pp.iloc[0]),
                         min_cell=q.cell.iloc[0], n_cells_ge_1pp=int((q.d_gross_pp >= 1.0).sum())))
    out = pd.DataFrame(rows)
    out.to_csv(LEDGER / f"{C.PFX}_gross_ceiling.csv", index=False)
    print(out.round(3).to_string())


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("ceiling", "all"):
        run_ceiling()
    if which in ("h2", "all"):
        run_h2()
    if which in ("resid", "all"):
        run_resid()
