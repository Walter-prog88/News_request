"""INC-T4 汇总:账本、配对、逐年、判定、图(预注册 §7、§8、§9)。

产出 research/ledger/:inc_t4_grid_a.csv、inc_t4_grid_b.csv、inc_t4_grid_c.csv、inc_t4_paired.csv、
inc_t4_yearly.csv、inc_t4_verdict.csv;图 reports/inc_t4_paired_cum.png、reports/inc_t4_hl_grid.png。
只读各子脚本落盘的日频明细(cache/_tmp/inc_t4_cell_*.parquet、inc_t4_ie_*_daily.parquet),不重跑任何回测。
"""
from __future__ import annotations

import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import inc_t4_common as C  # noqa: E402
import inc_t4_run_ab as AB  # noqa: E402
from afactor.config import LEDGER, TMP, REPORTS  # noqa: E402

warnings.filterwarnings("ignore")
pd.set_option("display.width", 250)
WIN_ORDER = ["full", "dev", "test", "y2025", "w2023_2025"]
PAIR_WINS = ["full", "dev", "test", "y2025", "w2023_2025"]


def load_cells() -> dict:
    out = {}
    for f in sorted(TMP.glob(f"{C.PFX}_cell_*.parquet")):
        d = pd.read_parquet(f)
        out[d["cell"].iloc[0]] = d
    return out


def meta_of(d: pd.DataFrame) -> dict:
    m = {}
    for k in ("method", "kind", "set", "hl", "lam", "e2", "x2", "variant"):
        if k in d.columns:
            v = d[k].iloc[0]
            m[k] = v
    return m


def grid_rows(cells: dict, methods: tuple) -> pd.DataFrame:
    rows = []
    for cid, d in cells.items():
        m = meta_of(d)
        if m.get("method") not in methods:
            continue
        for w in WIN_ORDER:
            st = AB.window_stats(d, w)
            rows.append(dict(cell=cid, window=w, **m, **st, align_peak_k=int(d["align_peak_k"].iloc[0])))
    return pd.DataFrame(rows)


def pair_row(method, var, base, rv, rb, w, extra=None) -> dict:
    st = C.paired_stats(rv, rb, w)
    r = dict(method=method, variant=var, baseline=base, window=w, **st)
    a, b = rv.align(rb, join="inner")
    d = (a - b).dropna()
    yd = d.groupby(d.index.year).mean() * C.TD
    yd = yd[(yd.index >= (C.WINDOWS[w][0] if isinstance(w, str) else w[0]))
            & (yd.index <= (C.WINDOWS[w][1] if isinstance(w, str) else w[1]))]
    r["years_won"] = int((yd > 0).sum())
    r["n_years"] = int(len(yd))
    if extra:
        r.update(extra)
    return r


def ie_rets(name: str, phases=(0,)) -> pd.Series | None:
    ds = []
    for ph in phases:
        f = TMP / f"{C.PFX}_ie_{name}_ph{ph}_daily.parquet"
        if not f.exists():
            return None
        ds.append(pd.read_parquet(f))
    if len(ds) == 1:
        return ds[0]["ret"]
    return pd.concat([d["ret"] for d in ds], axis=1).mean(axis=1)


def ie_daily(name: str, phases=(0,)) -> pd.DataFrame | None:
    ds = []
    for ph in phases:
        f = TMP / f"{C.PFX}_ie_{name}_ph{ph}_daily.parquet"
        if not f.exists():
            return None
        ds.append(pd.read_parquet(f))
    return pd.concat(ds).groupby(level=0).mean(numeric_only=True)


def v1_rets(phases=(0,)) -> pd.Series:
    ds = [pd.read_parquet(TMP / f"ie_V1_ph{ph}_daily.parquet")["ret"] for ph in phases]
    return ds[0] if len(ds) == 1 else pd.concat(ds, axis=1).mean(axis=1)


def grid_c(hs: int) -> pd.DataFrame:
    from afactor import index_proxy as IP
    import inc_t4_run_c as RC
    p = AB.P()
    rb_csi = IP.exec_time_bench_returns(p, "open10")
    rb_pool = AB.bench()["pool"]
    rows = []
    names = ["V1_ph0(ie 缓存)"] + list(RC.jobs())
    for nm in names:
        if nm.startswith("V1_ph0"):
            d = pd.read_parquet(TMP / "ie_V1_ph0_daily.parquet")
            summ = {}
        else:
            d = ie_daily(nm)
            if d is None:
                continue
            summ = json.load(open(TMP / f"{C.PFX}_ie_{nm}_ph0_summary.json"))
        rs = d["ret"]
        for w in ("full", "dev", "test", "y2025", "w2023_2025"):
            dw = C.win_slice(d, w)
            r = C.win_slice(rs, w)
            row = dict(job=nm, phase=0, window=w, n_days=len(r), ann_return=C.geo_ann(r))
            for tag, rb in (("csi1000p", rb_csi), ("pool", rb_pool)):
                b = rb.reindex(r.index)
                row[f"bench_{tag}_ann"] = C.geo_ann(b)
                row[f"excess_geo_{tag}"] = row["ann_return"] - row[f"bench_{tag}_ann"]
                row[f"excess_arith_{tag}"] = float((r - b).mean() * C.TD)
                row[f"t_nw_{tag}"] = C.nw_t((r - b).to_numpy())
            row.update(turnover_daily=float(dw["turnover"].mean()), cost_total_bp=float(dw["cost_total"].mean() * C.TD * 1e4),
                       cost_spread_bp=float(dw["cost_spread"].mean() * C.TD * 1e4),
                       cost_impact_bp=float(dw["cost_impact"].mean() * C.TD * 1e4),
                       holdings=float(dw["n_hold"].mean()), exposure_max=float(dw["exposure"].max()),
                       max_dd=float(((1 + r).cumprod() / (1 + r).cumprod().cummax() - 1).min()),
                       n_failed=summ.get("n_failed"), cost=summ.get("cost"), score=summ.get("score"),
                       desc=summ.get("desc"))
            rows.append(row)
    return pd.DataFrame(rows)


def verdict_for(pr: pd.DataFrame, method: str, var: str, base: str) -> dict:
    """§8 条件 1、2 与"无增量"规则在一个 (变体, 基线) 上的核心判定。"""
    q = pr[(pr.method == method) & (pr.variant == var) & (pr.baseline == base)].set_index("window")
    dv, te, fu = q.loc["dev"], q.loc["test"], q.loc["full"]
    c1 = dv.delta > 0 and te.delta > 0
    c2 = te.delta >= 0.01 and te.t_nw >= 2.0 and fu.t_nw >= 2.0
    none_ = (dv.delta <= 0 and te.delta <= 0) or (te.delta < 0 and abs(te.t_nw) >= 2)
    core = "有增量候选" if (c1 and c2) else ("无增量" if none_ else "证据不足")
    return dict(c1=c1, c2=c2, none_rule=none_, core=core, d_dev=dv.delta, d_test=te.delta, t_test=te.t_nw,
                d_full=fu.delta, t_full=fu.t_nw, mde_test=te.mde)


def main():
    C.check_prereg()
    hs = AB.read_hl_star()
    cells = load_cells()
    base = cells["base_Y1.0"]
    # ---------------- grid a / b
    ga = grid_rows(cells, ("a", "base"))
    ga.to_csv(LEDGER / f"{C.PFX}_grid_a.csv", index=False)
    gb = grid_rows(cells, ("b", "deep", "b_extra", "base", "diag"))
    gb.to_csv(LEDGER / f"{C.PFX}_grid_b.csv", index=False)
    gc = grid_c(hs)
    gc.to_csv(LEDGER / f"{C.PFX}_grid_c.csv", index=False)

    # ---------------- paired
    prs = []
    for cid, d in cells.items():
        m = meta_of(d)
        if m.get("method") not in ("a", "b", "deep"):
            continue
        v = m.get("variant", "Y1.0")
        bname = "base_Y1.0" if v == "Y1.0" else f"base_{v}"
        if bname not in cells:
            continue
        for w in PAIR_WINS:
            prs.append(pair_row(m["method"], cid, bname, d["ret"], cells[bname]["ret"], w,
                                dict(set=m.get("set"), hl=m.get("hl"), kind=m.get("kind"), cost=v)))
    # (b) 主配置 vs 深剔对照(开发窗持仓数最接近者)
    bmain = f"b_A_hl{hs}_{int(C.B_MAIN[0]*100)}_{int(C.B_MAIN[1]*100)}"
    deep_choice = None
    if bmain in cells:
        hb = AB.window_stats(cells[bmain], "dev")["holdings"]
        deep_ids = [f"deep_{int(e*100)}_{int(x*100)}" for e, x in C.DEEP_GRID]            # 只在预注册 §6.2 的四个深剔格里选
        cand = {c: abs(AB.window_stats(cells[c], "dev")["holdings"] - hb) for c in deep_ids if c in cells}
        deep_choice = min(cand, key=cand.get)
        for s in ("A", "main", "B", "A_nopead", "main_nopead", "B_nopead"):
            bm = f"b_{s}_hl{hs}_{int(C.B_MAIN[0]*100)}_{int(C.B_MAIN[1]*100)}"
            if bm in cells:
                for w in PAIR_WINS:
                    prs.append(pair_row("b_vs_deep", bm, deep_choice, cells[bm]["ret"], cells[deep_choice]["ret"], w,
                                        dict(set=s, hl=hs, kind="two_layer", cost="Y1.0")))
        # 事后诊断(不参与判定):持仓数匹配的浅对照 deep_21_36
        if "deep_21_36" in cells:
            for s in ("A", "main", "B", "A_nopead", "main_nopead", "B_nopead"):
                bm = f"b_{s}_hl{hs}_{int(C.B_MAIN[0]*100)}_{int(C.B_MAIN[1]*100)}"
                if bm in cells:
                    for w in PAIR_WINS:
                        prs.append(pair_row("b_vs_deep_diag", bm, "deep_21_36", cells[bm]["ret"], cells["deep_21_36"]["ret"],
                                            w, dict(set=s, hl=hs, kind="two_layer", cost="Y1.0")))
            for w in PAIR_WINS:
                prs.append(pair_row("diag", "deep_21_36", "base_Y1.0", cells["deep_21_36"]["ret"], base["ret"], w,
                                    dict(set="", hl=0, kind="comp_deep_shallow", cost="Y1.0")))
        # COMP_v5 自身平滑对照 vs 叠加:a 的 S1 与 C_smooth 同 HL 比
    for hl in C.HL_GRID:
        cs = f"a_Csm_hl{hl}" if hl else "base_Y1.0"
        s1 = f"a_S1_A_hl{hl}"
        if cs in cells and s1 in cells:
            for w in PAIR_WINS:
                prs.append(pair_row("a_vs_Csmooth", s1, cs, cells[s1]["ret"], cells[cs]["ret"], w,
                                    dict(set="A", hl=hl, kind="S1", cost="Y1.0")))
    # (c):相位 0 vs V1_ph0
    import inc_t4_run_c as RC
    r0 = v1_rets((0,))
    for nm, sp in RC.jobs().items():
        rv = ie_rets(nm)
        if rv is None:
            continue
        if sp["cost"] == "Y1.0":
            bn, rb = "V1_ph0", r0
        else:
            rb = ie_rets(f"V1_{sp['cost']}")
            bn = f"V1_{sp['cost']}"
            if rb is None or nm.startswith("V1_"):
                continue
        for w in PAIR_WINS:
            prs.append(pair_row("c", nm, bn, rv, rb, w, dict(set=(sp["score"] or ["comp"])[0],
                                                              hl=(sp["score"] or [None, 0])[1], kind="ie",
                                                              cost=sp["cost"])))
    # (c) 5 相位确认(若已跑)
    mainc = f"S1_A_hl{hs}"
    r5 = ie_rets(mainc, (0, 1, 2, 3, 4))
    if r5 is not None:
        v5 = v1_rets((0, 1, 2, 3, 4))
        for w in PAIR_WINS:
            prs.append(pair_row("c5", mainc + "(5相位)", "V1(5相位)", r5, v5, w,
                                dict(set="A", hl=hs, kind="ie", cost="Y1.0")))
    PR = pd.DataFrame(prs)
    PR.to_csv(LEDGER / f"{C.PFX}_paired.csv", index=False)

    # ---------------- yearly(主配置 + 基线 + 深剔对照)
    yrows = []
    amain, b_main = f"a_S1_A_hl{hs}", bmain
    show = {"base_Y1.0": "COMP_v5 排雷 20/35", amain: f"(a) S1 D_A HL{hs}", b_main: f"(b) 两层 10/35 D_A HL{hs}"}
    if deep_choice:
        show[deep_choice] = f"深剔对照 {deep_choice}"
    for s in ("main", "B"):
        show[f"a_S1_{s}_hl{hs}"] = f"(a) S1 D_{s} HL{hs}"
        show[f"b_{s}_hl{hs}_{int(C.B_MAIN[0]*100)}_{int(C.B_MAIN[1]*100)}"] = f"(b) 两层 D_{s} HL{hs}"
    for s in ("A", "main"):
        show[f"a_D_{s}_hl0"] = f"D_{s} 单独排雷 HL0"
    periods = [(str(y), (pd.Timestamp(f"{y}-01-01"), pd.Timestamp(f"{y}-12-31"))) for y in C.YEARS] + \
              [("2024-01-02~02-07", C.CRASH)]
    B = AB.bench()["pool"]
    for cid, lab in show.items():
        if cid not in cells:
            continue
        d = cells[cid]
        for pn, (a, b) in periods:
            r = d["ret"][(d.index >= a) & (d.index <= b)]
            rb = B.reindex(r.index)
            r0_ = base["ret"].reindex(r.index)
            yrows.append(dict(series=lab, cell=cid, period=pn, n_days=len(r),
                              excess_arith=float((r - rb).mean() * C.TD) if pn[0] == "2" and len(pn) == 4 else np.nan,
                              excess_cum=float((1 + r).prod() / (1 + rb).prod() - 1),
                              delta_vs_base_arith=float((r - r0_).mean() * C.TD) if len(pn) == 4 else np.nan,
                              delta_vs_base_cum=float((1 + r).prod() / (1 + r0_).prod() - 1),
                              turnover=float(d["turnover"][(d.index >= a) & (d.index <= b)].mean())))
    # (c) 逐年
    rc = ie_rets(mainc)
    if rc is not None:
        for pn, (a, b) in periods:
            r = rc[(rc.index >= a) & (rc.index <= b)]
            r0_ = r0.reindex(r.index)
            yrows.append(dict(series=f"(c) 指增 S1 D_A HL{hs} 相位0", cell=mainc, period=pn, n_days=len(r),
                              delta_vs_base_arith=float((r - r0_).mean() * C.TD) if len(pn) == 4 else np.nan,
                              delta_vs_base_cum=float((1 + r).prod() / (1 + r0_).prod() - 1)))
    YR = pd.DataFrame(yrows)
    YR.to_csv(LEDGER / f"{C.PFX}_yearly.csv", index=False)

    # ---------------- 判定(§8)
    ck = pd.read_csv(LEDGER / f"{C.PFX}_checks.csv")
    eng_ok = bool(ck[ck.check.isin(["baseline_reproduce", "exposure_le_1", "align_peak_k0", "fullpool_resid_pp"])]
                  ["passed"].astype(bool).all())
    allk0 = all(int(d["align_peak_k"].iloc[0]) == 0 for d in cells.values())
    allexp = all(float(d["exp_max_phase"].max()) <= 1 + 1e-9 for d in cells.values())
    vrows = []
    specs = {
        "a": dict(method="a", main=f"a_S1_A_hl{hs}", base="base_Y1.0", main_s=f"a_S1_main_hl{hs}",
                  B=f"a_S1_B_hl{hs}", y15=f"a_S1_A_hl{hs}_Y1.5", y15b="base_Y1.5", nop=f"a_S1_A_nopead_hl{hs}"),
        "b": dict(method="b", main=bmain, base="base_Y1.0", main_s=f"b_main_hl{hs}_10_35", B=f"b_B_hl{hs}_10_35",
                  y15=f"{bmain}_Y1.5", y15b="base_Y1.5", nop=f"b_A_nopead_hl{hs}_10_35"),
        "c": dict(method="c", main=f"S1_A_hl{hs}", base="V1_ph0", main_s=f"S1_main_hl{hs}", B=f"S1_B_hl{hs}",
                  y15=f"S1_A_hl{hs}_Y1.5", y15b="V1_Y1.5", nop=f"S1_A_nopead_hl{hs}"),
    }
    for mth, sp in specs.items():
        try:
            core = verdict_for(PR, sp["method"], sp["main"], sp["base"])
        except KeyError:
            continue
        g = lambda var, bas, w: PR[(PR.method == sp["method"]) & (PR.variant == var) & (PR.baseline == bas)
                                   & (PR.window == w)]
        q = g(sp["main_s"], sp["base"], "y2025"); c3a = bool(len(q) and q.delta.iloc[0] > 0)
        q = g(sp["B"], sp["base"], "w2023_2025"); c3b = bool(len(q) and q.delta.iloc[0] > 0)
        q = g(sp["y15"], sp["y15b"], "test"); c5a = bool(len(q) and q.delta.iloc[0] > 0)
        try:
            core_nop = verdict_for(PR, sp["method"], sp["nop"], sp["base"])["core"]
        except KeyError:
            core_nop = "缺"
        c5b = core_nop == core["core"]
        c4 = True
        if mth == "b" and deep_choice:
            qd = PR[(PR.method == "b_vs_deep") & (PR.variant == bmain)].set_index("window")
            c4 = bool(qd.loc["dev"].delta > 0 and qd.loc["test"].delta > 0)
        c6 = eng_ok and allk0 and allexp
        if core["core"] == "有增量候选":
            final = "有增量(待归因)" if (c3a and c3b and c4 and c5a and c5b and c6) else "证据不足"
        elif core["core"] == "无增量" or (mth == "b" and not c4):
            final = "无增量"
        else:
            final = "证据不足"
        if not c5b and final != "无增量":
            final = "证据不足"
        # ---- 审查修复(2026-10-11):预注册 §7.1 "MDE_test > 2pp 则结论上限为证据不足(除非 |Δ| 超过 MDE)"。
        # 原版只按 §8 规则出标签;现在两列并存:verdict_rule8 = 纯 §8 规则,verdict = 再套 §7.1 功效上限(正式判定)。
        verdict_rule8 = final
        cap71 = bool(core["mde_test"] > 0.02 and not (abs(core["d_test"]) > core["mde_test"]))
        if cap71 and final in ("无增量", "有增量(待归因)"):
            final = "证据不足"
        # (b) 事后诊断:若深剔对照换成持仓数匹配的 deep_21_36(不参与判定),条件 4 与规则标签会怎样
        c4_diag, v_diag = np.nan, ""
        if mth == "b" and "deep_21_36" in cells:
            qd2 = PR[(PR.method == "b_vs_deep_diag") & (PR.variant == bmain)].set_index("window")
            c4_diag = bool(qd2.loc["dev"].delta > 0 and qd2.loc["test"].delta > 0)
            v_diag = core["core"] if c4_diag else "无增量"
        vrows.append(dict(method=mth, main=sp["main"], baseline=sp["base"], **core,
                          c3_main_2025=c3a, c3_B_2023_2025=c3b, c4_b_vs_deep=c4 if mth == "b" else np.nan,
                          deep_control=deep_choice if mth == "b" else "", c5_y15_test_pos=c5a, core_nopead=core_nop,
                          c5_nopead_same=c5b, c6_engine=c6, cap_7_1_low_power=cap71, verdict_rule8=verdict_rule8,
                          c4_diag_deep_21_36=c4_diag, verdict_if_diag_control=v_diag, verdict=final))
    V = pd.DataFrame(vrows)
    V.to_csv(LEDGER / f"{C.PFX}_verdict.csv", index=False)
    print(V.T.to_string())
    plots(cells, hs, bmain, deep_choice)


def plots(cells, hs, bmain, deep_choice):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Heiti SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
    base = cells["base_Y1.0"]["ret"]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw=dict(width_ratios=[1, 1]))
    series = [(f"a_S1_A_hl{hs}", f"(a) 平滑叠加 HL{hs}", BLUE), (bmain, "(b) 两层排雷 10/35", ORANGE)]
    for cid, lab, col in series:
        if cid in cells:
            r, b = cells[cid]["ret"].align(base, join="inner")
            d = (r - b)[r.index.year >= 2021]
            ax.plot(d.index, d.cumsum() * 100, color=col, lw=2, label=lab)
    if deep_choice:
        r, b = cells[deep_choice]["ret"].align(base, join="inner")
        d = (r - b)[r.index.year >= 2021]
        ax.plot(d.index, d.cumsum() * 100, color=GRAY, lw=1.5, ls="--", label=f"深剔对照 {deep_choice[5:].replace('_', '/')}")
    ax.set_title("(a)(b):相对 COMP_v5 排雷 20/35 的累计配对差(pp,成本后 Y=1)", fontsize=10)
    ax.set_ylim(-3, 3)
    rc = ie_rets(f"S1_A_hl{hs}")
    if rc is not None:
        r, b = rc.align(v1_rets((0,)), join="inner")
        ax2.plot(r.index, (r - b).cumsum() * 100, color=AQUA, lw=2, label="S1 相位0 − V1 相位0(判定口径)")
    r5 = ie_rets(f"S1_A_hl{hs}", (0, 1, 2, 3, 4))
    if r5 is not None:
        r, b = r5.align(v1_rets((0, 1, 2, 3, 4)), join="inner")
        ax2.plot(r.index, (r - b).cumsum() * 100, color=ORANGE, lw=1.6, ls="--", label="5 相位 − V1 5 相位(事后补充)")
    ax2.set_title("(c) 指增优化器:α 叠加 D 相对 V1 的累计配对差(pp)", fontsize=10)
    for a_ in (ax, ax2):
        a_.axhline(0, color="#c3c2b7", lw=1)
        a_.axvline(pd.Timestamp("2024-01-01"), color="#c3c2b7", lw=1, ls=":")
        for s_ in ("top", "right"):
            a_.spines[s_].set_visible(False)
        a_.grid(axis="y", color="#e6e5e0", lw=0.8)
        a_.legend(frameon=False, fontsize=9, loc="lower left")
    ax.set_ylabel("日差求和(pp)")
    ax.text(pd.Timestamp("2024-01-15"), 2.5, "检验窗 →", color="#52514e", fontsize=9)
    fig.tight_layout()
    fig.savefig(REPORTS / f"{C.PFX}_paired_cum.png", dpi=150)
    plt.close(fig)

    # HL 网格:换手 vs 检验窗 Δ
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    tb = AB.window_stats(cells["base_Y1.0"], "dev")["turnover_daily"]
    for kind, lab, col in (("S1", "z(COMP)+z(D_A,HL)", BLUE), ("Csm", "z(COMP_HL) 自身平滑对照", GRAY),
                           ("D", "z(D_A,HL) 单独", ORANGE)):
        xs, tv, dt = [], [], []
        for hl in C.HL_GRID:
            cid = {"S1": f"a_S1_A_hl{hl}", "Csm": f"a_Csm_hl{hl}" if hl else "base_Y1.0", "D": f"a_D_A_hl{hl}"}[kind]
            if cid not in cells:
                continue
            xs.append(hl)
            tv.append(AB.window_stats(cells[cid], "dev")["turnover_daily"])
            st = C.paired_stats(cells[cid]["ret"], cells["base_Y1.0"]["ret"], "test")
            dt.append(st.get("delta", np.nan) * 100)
        axes[0].plot(xs, tv, marker="o", ms=5, color=col, lw=2, label=lab)
        axes[1].plot(xs, dt, marker="o", ms=5, color=col, lw=2, label=lab)
    axes[0].axhline(tb * C.TURN_RULE_MULT, color="#e34948", lw=1, ls="--")
    axes[0].text(0.5, tb * C.TURN_RULE_MULT * 1.03, "1.15×基线(HL* 规则)", color="#52514e", fontsize=8)
    axes[0].set_title("开发窗日换手", fontsize=10)
    axes[1].axhline(0, color="#c3c2b7", lw=1)
    axes[1].set_title("检验窗 Δ 对 COMP_v5 排雷(pp/年)", fontsize=10)
    for ax in axes:
        ax.set_xlabel("EWM 半衰期 HL(信号日)")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.grid(axis="y", color="#e6e5e0", lw=0.8)
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(REPORTS / f"{C.PFX}_hl_grid.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
