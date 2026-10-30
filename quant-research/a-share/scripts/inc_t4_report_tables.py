"""INC-T4:从 research/ledger/inc_t4_*.csv 生成报告用 markdown 表格(防手抄错)。
输出到 stdout 与 research/ledger/inc_t4_report_tables.md(供手写报告引用;不是报告本身)。
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import inc_t4_common as C  # noqa: E402
from afactor.config import LEDGER  # noqa: E402

warnings.filterwarnings("ignore")
L = LEDGER
HS = 5


def md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join("" if (isinstance(v, float) and not np.isfinite(v)) else str(v) for v in r.values) + " |")
    return "\n".join(out)


def pp(x, nd=2):
    return "" if x is None or not np.isfinite(x) else f"{x*100:+.{nd}f}"


def fmt_t(x):
    return "" if x is None or not np.isfinite(x) else f"{x:.2f}"


def main():
    hs = int(pd.read_csv(L / "inc_t4_checks.csv").query("check=='hl_star'")["value"].iloc[0])
    ga = pd.read_csv(L / "inc_t4_grid_a.csv")
    gb = pd.read_csv(L / "inc_t4_grid_b.csv")
    gc = pd.read_csv(L / "inc_t4_grid_c.csv")
    pr = pd.read_csv(L / "inc_t4_paired.csv")
    sd = pd.read_csv(L / "inc_t4_stage0_diag.csv")
    ck = pd.read_csv(L / "inc_t4_checks.csv")
    T = {}

    # ---------------- A:阶段 0
    x = sd[sd.hl == 0].copy()
    x = x[x.series.isin(["COMP_v5", "D_A", "D_main"]) & x.window.isin(["2021-2025", "2021-2023", "2024-2025"])]
    t = x[["series", "window", "ic_h1", "ic_h5", "ic_h10", "ic_h20", "ratio_h5_h1", "ratio_h20_h1", "rank_ac_lag1",
           "rank_ac_lag5", "rank_ac_lag10", "rank_ac_lag20"]].round(4)
    T["A_stage0_term"] = md(t)
    y = sd[(sd.window == "2021-2025") & (sd.hl > 0)][["series", "hl", "ic_h1", "ic_h5", "ratio_h5_h1", "rank_ac_lag5"]]
    y = y[y.series.isin(["z(D_A_HL)", "z(COMP_v5_HL)"])].sort_values(["series", "hl"]).round(4)
    T["A2_stage0_smooth"] = md(y)

    # ---------------- B:引擎断言
    b = ck[~ck.check.isin(["hl_rule_turnover", "hl_star"])][["check", "item", "value", "reference", "diff", "passed"]].copy()
    b[["value", "reference", "diff"]] = b[["value", "reference", "diff"]].astype(float).round(6)
    T["B_checks"] = md(b)
    hr = ck[ck.check == "hl_rule_turnover"][["item", "value", "reference", "passed"]].copy()
    hr[["value", "reference"]] = hr[["value", "reference"]].astype(float).round(5)
    T["B2_hl_rule"] = md(hr)

    # ---------------- C:方法 a 网格
    def P(var, base, w, method=None):
        q = pr[(pr.variant == var) & (pr.baseline == base) & (pr.window == w)]
        if method:
            q = q[q.method == method]
        return q.iloc[0] if len(q) else None

    def dstr(r):
        if r is None:
            return ""
        return f"{r.delta*100:+.2f} ({r.t_nw:+.2f})"

    def ga_get(cell, w, col):
        q = ga[(ga.cell == cell) & (ga.window == w)]
        return float(q[col].iloc[0]) if len(q) else np.nan

    rows = []
    for s in ("A", "main"):
        for hl in C.HL_GRID:
            cs = f"a_S1_{s}_hl{hl}"
            cd = f"a_D_{s}_hl{hl}"
            ccm = f"a_Csm_hl{hl}" if hl else "base_Y1.0"
            rows.append({"集": f"D_{s}", "HL": hl,
                         "S1 开发窗日换手": round(ga_get(cs, "dev", "turnover_daily"), 4),
                         "S1 全期年成本bp": round(ga_get(cs, "full", "cost_total_bp"), 0),
                         "S1 Δdev": dstr(P(cs, "base_Y1.0", "dev", "a")),
                         "S1 Δtest(t)": dstr(P(cs, "base_Y1.0", "test", "a")),
                         "S1 Δfull(t)": dstr(P(cs, "base_Y1.0", "full", "a")),
                         "D 单独 Δtest(t)": dstr(P(cd, "base_Y1.0", "test", "a")),
                         "COMP自身平滑 Δtest(t)": dstr(P(ccm, "base_Y1.0", "test", "a")) if hl else "0(基线)"})
    T["C_a_grid"] = md(pd.DataFrame(rows))
    # D_B
    rows = []
    for hl in (0, hs):
        cs = f"a_S1_B_hl{hl}"
        rows.append({"集": "D_B", "HL": hl, "S1 Δ2023-25(t)": dstr(P(cs, "base_Y1.0", "w2023_2025", "a")),
                     "S1 Δfull(t)": dstr(P(cs, "base_Y1.0", "full", "a")),
                     "D 单独 Δ2023-25(t)": dstr(P(f"a_D_B_hl{hl}", "base_Y1.0", "w2023_2025", "a"))})
    T["C2_a_B"] = md(pd.DataFrame(rows))

    # a 主配置 vs COMP 自身平滑对照(同 HL)
    rows = []
    for hl in C.HL_GRID:
        q = pr[(pr.method == "a_vs_Csmooth") & (pr.hl == hl)]
        if not len(q):
            continue
        d = q.set_index("window")
        rows.append({"HL": hl, "S1 − COMP自身平滑 Δdev(t)": dstr(d.loc["dev"]), "Δtest(t)": dstr(d.loc["test"]),
                     "Δfull(t)": dstr(d.loc["full"])})
    T["C3_a_vs_csm"] = md(pd.DataFrame(rows))

    # ---------------- D:方法 b
    base_dev_hold = ga_get("base_Y1.0", "dev", "holdings")
    rows = []
    cl = []
    for s in ("A", "main"):
        for hl in sorted({0, hs}):
            for e, x_ in C.B_GRID:
                cl.append((f"b_{s}_hl{hl}_{int(e*100)}_{int(x_*100)}", f"D_{s}", hl, f"{e:.2f}/{x_:.2f}"))
    for e, x_ in C.DEEP_GRID:
        cl.append((f"deep_{int(e*100)}_{int(x_*100)}", "COMP_v5 单层深剔", "-", f"{e:.2f}/{x_:.2f}"))
    cl.append(("deep_21_36", "COMP_v5 单层(事后诊断,持仓匹配)", "-", "0.21/0.36"))
    for cid, nm, hl, buf in cl:
        g = lambda w, c: ga_get(cid, w, c) if False else (lambda q: float(q[c].iloc[0]) if len(q) else np.nan)(
            gb[(gb.cell == cid) & (gb.window == w)])
        rows.append({"打分": nm, "HL": hl, "缓冲带": buf,
                     "开发窗持仓数": round(g("dev", "holdings"), 0), "开发窗日换手": round(g("dev", "turnover_daily"), 4),
                     "Δdev": dstr(P(cid, "base_Y1.0", "dev")), "Δtest(t)": dstr(P(cid, "base_Y1.0", "test")),
                     "Δfull(t)": dstr(P(cid, "base_Y1.0", "full")),
                     "MDE_test": f"{P(cid, 'base_Y1.0', 'test').mde*100:.2f}" if P(cid, "base_Y1.0", "test") is not None else ""})
    T["D_b_grid"] = md(pd.DataFrame(rows))
    rows = []
    for s in ("A", "main", "B", "A_nopead"):
        bm = f"b_{s}_hl{hs}_10_35"
        for nm, meth in (("b_vs_deep", "b_vs_deep"), ("b_vs_deep_diag", "b_vs_deep_diag")):
            q = pr[(pr.method == meth) & (pr.variant == bm)]
            if not len(q):
                continue
            d = q.set_index("window")
            rows.append({"集": s, "对照": q.baseline.iloc[0] + ("(预注册匹配)" if meth == "b_vs_deep" else "(事后诊断)"),
                         "b主配置 − 对照 Δdev(t)": dstr(d.loc["dev"]), "Δtest(t)": dstr(d.loc["test"]),
                         "Δfull(t)": dstr(d.loc["full"])})
    T["D2_b_vs_deep"] = md(pd.DataFrame(rows))

    # ---------------- E:方法 c
    rows = []
    for _, r in gc[gc.window == "full"].iterrows():
        nm = r.job
        d = {w: gc[(gc.job == nm) & (gc.window == w)].iloc[0] for w in ("dev", "test", "full")}
        base = "V1_ph0" if r.cost == "Y1.0" or nm.startswith("V1_ph0") else f"V1_{r.cost}"
        cc = lambda w: dstr(P(nm, base, w, "c")) if not nm.startswith("V1_ph0") and not nm.startswith("V1_") and nm != "V1rep" else ""
        rows.append({"作业": nm, "成本": r.cost if isinstance(r.cost, str) else "Y1.0", "绝对年化": f"{r.ann_return*100:.2f}",
                     "超额(中证1000代理)算术": pp(r.excess_arith_csi1000p),
                     "t": fmt_t(r.t_nw_csi1000p), "日换手": round(r.turnover_daily, 4), "年成本bp": round(r.cost_total_bp, 0),
                     "Δdev": cc("dev"), "Δtest(t)": cc("test"), "Δfull(t)": cc("full"),
                     "MDE_test": (f"{P(nm, base, 'test', 'c').mde*100:.2f}" if cc("test") else "")})
    T["E_c"] = md(pd.DataFrame(rows))

    # ---------------- G:判定表
    v = pd.read_csv(L / "inc_t4_verdict.csv")
    T["G_verdict"] = md(v.round(4))

    # ---------------- H:敏感性
    rows = []
    specs = [("a", f"a_S1_A_hl{hs}", "base_Y1.0", f"a_S1_main_hl{hs}", f"a_S1_B_hl{hs}", f"a_S1_A_hl{hs}_Y1.5", "base_Y1.5",
              f"a_S1_A_hl{hs}_legacy", "base_legacy", f"a_S1_A_nopead_hl{hs}"),
             ("b", f"b_A_hl{hs}_10_35", "base_Y1.0", f"b_main_hl{hs}_10_35", f"b_B_hl{hs}_10_35", f"b_A_hl{hs}_10_35_Y1.5",
              "base_Y1.5", f"b_A_hl{hs}_10_35_legacy", "base_legacy", f"b_A_nopead_hl{hs}_10_35"),
             ("c", f"S1_A_hl{hs}", "V1_ph0", f"S1_main_hl{hs}", f"S1_B_hl{hs}", f"S1_A_hl{hs}_Y1.5", "V1_Y1.5",
              f"S1_A_hl{hs}_legacy", "V1_legacy", f"S1_A_nopead_hl{hs}")]
    for m, main_, base, mn, bb, y15, b15, leg, bl, nop in specs:
        mm = m
        rows.append({"方法": m,
                     "D_A 主配置 Δtest(t)": dstr(P(main_, base, "test", mm)),
                     "D_main 2025": dstr(P(mn, base, "y2025", mm)),
                     "D_B 2023-25": dstr(P(bb, base, "w2023_2025", mm)),
                     "Y=1.5 Δtest(t)": dstr(P(y15, b15, "test", mm)),
                     "旧成本 Δtest(t)": dstr(P(leg, bl, "test", mm)),
                     "D′(无pead) Δtest(t)": dstr(P(nop, base, "test", mm)),
                     "D_main′ Δtest(t)": dstr(P(mn.replace(f"_hl{hs}", f"_nopead_hl{hs}") if m != "b" else f"b_main_nopead_hl{hs}_10_35",
                                               base, "test", mm))})
    T["H_sens"] = md(pd.DataFrame(rows))
    rows = []
    for lam, cid in ((0.5, f"a_S0.5_A_hl{hs}"), (1.0, f"a_S1_A_hl{hs}"), (2.0, f"a_S2_A_hl{hs}")):
        rows.append({"方法": "a", "λ": lam, "Δdev": dstr(P(cid, "base_Y1.0", "dev", "a")),
                     "Δtest(t)": dstr(P(cid, "base_Y1.0", "test", "a")), "Δfull(t)": dstr(P(cid, "base_Y1.0", "full", "a"))})
    for lam, cid in ((0.5, f"S0.5_A_hl{hs}"), (1.0, f"S1_A_hl{hs}")):
        rows.append({"方法": "c", "λ": lam, "Δdev": dstr(P(cid, "V1_ph0", "dev", "c")),
                     "Δtest(t)": dstr(P(cid, "V1_ph0", "test", "c")), "Δfull(t)": dstr(P(cid, "V1_ph0", "full", "c"))})
    T["H2_lambda"] = md(pd.DataFrame(rows))
    rows = []
    for cid, nm in ((f"S1_A_hl0", "c: λ=1,HL=0"), (f"D_A_hl{hs}", "c: D_HL5 单独"), (f"S1_A_hl{hs}", "c: λ=1,HL5(主)")):
        rows.append({"配置": nm, "Δdev": dstr(P(cid, "V1_ph0", "dev", "c")), "Δtest(t)": dstr(P(cid, "V1_ph0", "test", "c")),
                     "Δfull(t)": dstr(P(cid, "V1_ph0", "full", "c"))})
    T["H3_c_grid"] = md(pd.DataFrame(rows))

    # 5 相位(事后)
    q = pr[pr.method == "c5"]
    if len(q):
        d = q.set_index("window")
        T["H4_c5"] = md(pd.DataFrame([{"配置": "S1_A_hl5 5相位 vs V1 5相位(事后补充,不参与判定)",
                                      "Δdev(t)": dstr(d.loc["dev"]), "Δtest(t)": dstr(d.loc["test"]),
                                      "Δfull(t)": dstr(d.loc["full"]), "MDE_test": f"{d.loc['test'].mde*100:.2f}",
                                      "赢的年数": f"{int(d.loc['full'].years_won)}/{int(d.loc['full'].n_years)}"}]))

    # ---------------- I:逐年
    yr = pd.read_csv(L / "inc_t4_yearly.csv")
    yy = yr.pivot_table(index="series", columns="period", values="delta_vs_base_arith", aggfunc="first")
    yy = (yy * 100).round(2)
    T["I_yearly_delta"] = md(yy.reset_index())
    yx = yr.pivot_table(index="series", columns="period", values="excess_arith", aggfunc="first")
    yx = (yx * 100).round(2)
    T["I2_yearly_excess"] = md(yx.reset_index())
    cr = yr[yr.period == "2024-01-02~02-07"][["series", "n_days", "excess_cum", "delta_vs_base_cum"]].copy()
    cr[["excess_cum", "delta_vs_base_cum"]] = (cr[["excess_cum", "delta_vs_base_cum"]] * 100).round(2)
    T["I3_crash"] = md(cr)

    # ---------------- J:H2
    h2 = pd.read_csv(L / "inc_t4_h2.csv")
    h2 = h2.round(4)
    T["J_h2"] = md(h2)

    # ---------------- K:毛 / 成本 / 净
    rows = []
    for cid, nm in (("base_Y1.0", "COMP_v5 排雷 20/35(基线)"), (f"a_S1_A_hl{hs}", f"(a) S1 D_A HL{hs}"),
                    (f"a_Csm_hl{hs}", f"(a-对照) COMP_v5 自平滑 HL{hs}"), (f"a_D_A_hl0", "D_A 单独排雷 HL0"),
                    (f"a_D_A_hl20", "D_A 单独排雷 HL20"), (f"b_A_hl{hs}_10_35", f"(b) 两层 10/35"),
                    ("deep_25_40", "COMP_v5 深剔 25/40")):
        for w in ("dev", "test", "full"):
            q = ga[(ga.cell == cid) & (ga.window == w)]
            if not len(q):
                q = gb[(gb.cell == cid) & (gb.window == w)]
            r = q.iloc[0]
            rows.append({"配置": nm, "窗口": w, "持仓数": round(r.holdings, 0), "日换手": round(r.turnover_daily, 4),
                         "毛超额(算术)": pp(r.gross_excess_arith), "年成本bp": round(r.cost_total_bp, 0),
                         "净超额(算术)": pp(r.excess_arith), "净超额(几何)": pp(r.excess_geo), "NW t": fmt_t(r.t_nw)})
    T["K_triple"] = md(pd.DataFrame(rows))
    rows = []
    for nm in ("V1_ph0(ie 缓存)", f"S1_A_hl{hs}", f"S0.5_A_hl{hs}", f"D_A_hl{hs}", "S1_A_hl0"):
        for w in ("dev", "test", "full"):
            q = gc[(gc.job == nm) & (gc.window == w)]
            if not len(q):
                continue
            r = q.iloc[0]
            rows.append({"作业": nm, "窗口": w, "绝对年化(几何)": pp(r.ann_return), "超额vs中证1000代理(算术)": pp(r.excess_arith_csi1000p),
                         "t": fmt_t(r.t_nw_csi1000p), "日换手": round(r.turnover_daily, 4), "年成本bp": round(r.cost_total_bp, 0),
                         "持仓数": round(r.holdings, 0), "最大回撤": pp(r.max_dd, 1)})
    T["K2_c_triple"] = md(pd.DataFrame(rows))

    # ---------------- L:重叠与 MDE
    ov = pd.read_csv(L / "inc_t4_overlap.csv").round(4)
    T["L_overlap"] = md(ov)
    rows = []
    for m, cid, base in (("a", f"a_S1_A_hl{hs}", "base_Y1.0"), ("b", f"b_A_hl{hs}_10_35", "base_Y1.0"), ("c", f"S1_A_hl{hs}", "V1_ph0")):
        for w in ("dev", "test", "full"):
            r = P(cid, base, w, m)
            rows.append({"方法": m, "窗口": w, "Δ(pp)": f"{r.delta*100:+.2f}", "配对TE(%)": f"{r.te_pair*100:.2f}",
                         "SE(pp)": f"{r.se*100:.2f}", "MDE=2SE(pp)": f"{r.mde*100:.2f}",
                         "95%区间": f"[{r.lo95*100:+.2f}, {r.hi95*100:+.2f}]", "赢的年数": f"{int(r.years_won)}/{int(r.n_years)}"})
    T["L2_mde"] = md(pd.DataFrame(rows))

    # ---------------- M:诊断
    di = pd.read_csv(L / "inc_t4_diag_ic.csv")
    di = di[(di.set == "A") & (di.hl.isin([0, hs]))]
    di["ic_mean"] = di.ic_mean.round(4)
    di["Δ vs COMP(IC)"] = di.d_ic_vs_comp.round(4)
    di["Δt"] = di.d_ic_t_nw.round(2)
    T["M_diag_ic"] = md(di[["hl", "lam", "window", "ic_mean", "ic_t_nw", "Δ vs COMP(IC)", "Δt"]].assign(ic_t_nw=di.ic_t_nw.round(2)))
    ca = pd.read_csv(L / "inc_t4_diag_c_attr.csv")
    x = ca[ca.component.isin(["total", "country", "style", "industry", "specific", "rev20", "specific_ex_rev20"]) |
           ca.component.str.startswith("style:")].pivot_table(index="component", columns="window", values="ann_pp")[["dev", "test", "full"]]
    t_ = ca[ca.component.isin(["total", "style", "rev20"])].pivot_table(index="component", columns="window", values="t_nw")[["full"]]
    T["M2_attr"] = md(x.round(2).reset_index())
    xe = ca[ca.component.str.startswith("exposure_diff")].pivot_table(index="component", columns="window", values="ann_pp")[["dev", "test", "full"]]
    T["M3_expdiff"] = md(xe.round(3).reset_index())

    # ---------------- N:审查修复(残差 IC、κ 曲线、H2b)
    if (L / "inc_t4_diag_resid.csv").exists():
        rs = pd.read_csv(L / "inc_t4_diag_resid.csv")
        for w in ("full", "dev", "test"):
            q = rs[rs.window == w]
            ic = q.pivot_table(index=["kind", "series"], columns="h", values="ic_mean", sort=False).round(4)
            tt = q.pivot_table(index=["kind", "series"], columns="h", values="ic_t_nw", sort=False).round(2)
            z = pd.DataFrame({f"IC(h{h})": ic[h] for h in ic.columns})
            for h in tt.columns:
                z[f"t(h{h})"] = tt[h]
            z = z[[f"IC(h{h})" for h in ic.columns] + [f"t(h{h})" for h in ic.columns]]
            z["h5/h1"] = (ic[5] / ic[1]).round(2)
            z["h20/h1"] = (ic[20] / ic[1]).round(2)
            T[f"N_resid_{w}"] = md(z.reset_index())
        ms = pd.read_csv(L / "inc_t4_diag_resid_misc.csv")
        T["N2_resid_misc"] = md(ms.round(4))
    if (L / "inc_t4_h2b.csv").exists():
        T["O_h2b"] = md(pd.read_csv(L / "inc_t4_h2b.csv").round(4))
        T["O2_h2b_cells"] = md(pd.read_csv(L / "inc_t4_h2b_cells.csv").round(4))

    # ---------------- P:毛收益天花板(成本前,变体相对基线多出的毛超额 / 多付的成本 / 净差)
    rows = []
    allg = pd.concat([ga, gb]).drop_duplicates(["cell", "window"])
    cells_ = ([f"a_S1_A_hl{h}" for h in C.HL_GRID] + ["a_D_A_hl0", "a_D_A_hl20"]
              + [f"b_A_hl{h}_{e}_{x}" for h in (0, hs) for e, x in ((10, 25), (10, 35), (20, 35), (20, 45))]
              + ["deep_25_40", "deep_21_36"])
    for cid in cells_:
        for w in ("full", "dev", "test"):
            r = allg[(allg.cell == cid) & (allg.window == w)]
            b0 = allg[(allg.cell == "base_Y1.0") & (allg.window == w)]
            if not len(r):
                continue
            r, b0 = r.iloc[0], b0.iloc[0]
            rows.append({"配置": cid, "窗口": w, "Δ毛超额(pp)": pp(r.gross_excess_arith - b0.gross_excess_arith),
                         "Δ年成本(bp)": f"{r.cost_total_bp - b0.cost_total_bp:+.0f}",
                         "Δ净超额(算术,pp)": pp(r.excess_arith - b0.excess_arith),
                         "Δ日换手": f"{r.turnover_daily - b0.turnover_daily:+.4f}"})
    T["P_gross_ceiling"] = md(pd.DataFrame(rows))
    if (L / "inc_t4_gross_ceiling.csv").exists():
        T["P2_gross_ceiling_all"] = md(pd.read_csv(L / "inc_t4_gross_ceiling.csv").round(3))

    out = "\n\n".join(f"### {k}\n\n{v}" for k, v in T.items())
    (L / "inc_t4_report_tables.md").write_text(out)
    print(out)


if __name__ == "__main__":
    main()
