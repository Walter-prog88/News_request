"""INC-T3 S4:叠加剔除的汇总与判定(预注册 §7.3、§7.4、§8.2)。

配对日差 d_t = r_over,t − r_base,t(5 相位平均、扣成本后的日收益,同一成本 / 资金设定的基线臂),
年化 = 日均 × 242;Newey-West(Bartlett)带宽 10;窗口按收益索引日(信号日)的年份切:
2021-2025 / 2021-2023 / 2024-2025;逐年符号;5 个相位各自的年化差与极差;20 个日历季度为独立单位的 t。
两臂各自对同频基准:算术年化超额与 NW t(ST.excess_stats,带宽 10)、几何年化超额、换手、成本分解、持仓、回撤。
安慰剂:placebo_mean = 10 套伪剔除日收益均值;报告 overlay − placebo_mean 与 placebo_mean − base。

用法  python3 scripts/inc_t3_analyze.py
产出  research/ledger/inc_t3_overlay_grid.csv、inc_t3_overlay_yearly.csv、inc_t3_overlay_phase.csv、
     inc_t3_placebo.csv、inc_t3_verdict.csv;reports/inc_t3_overlay_diff.png
"""
from __future__ import annotations
import os, sys, glob
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import inc_t3_common as C                                   # noqa: E402
import inc_t3_overlay as OV                                 # noqa: E402
from afactor import strategy as ST                          # noqa: E402
import cap_capacity_curve as CAP                            # noqa: E402

TD = 242
BW = 10
YW = {"2021-2025": (2021, 2025), "2021-2023": (2021, 2023), "2024-2025": (2024, 2025)}
GRID18 = [(x, n, m) for x in OV.XS for n in (5, 10, 20) for m in OV.MS]


def wsl(s: pd.Series, w) -> pd.Series:
    y0, y1 = YW[w] if isinstance(w, str) else w
    return s[(s.index.year >= y0) & (s.index.year <= y1)]


def diff_stats(d: pd.Series, prefix: str = "") -> dict:
    out = {}
    for w in YW:
        x = wsl(d, w).dropna()
        se = C.nw_se(x.to_numpy(), BW)
        out[f"{prefix}ann_{w}"] = float(x.mean() * TD)
        out[f"{prefix}se_{w}"] = float(se * TD)
        out[f"{prefix}t_{w}"] = float(x.mean() / se) if se and se > 0 else np.nan
    return out


def arm_stats(r: pd.DataFrame, B: pd.DataFrame, tag: str) -> dict:
    out = {}
    for w in YW:
        x = wsl(r, w)
        rs = x["ret"]
        es = ST.excess_stats(rs, B["pool"].reindex(rs.index), bandwidth=20, t_bandwidth=CAP.NW_T_BW)
        out[f"{tag}_excess_arith_{w}"] = es.get("excess_arith", np.nan)
        out[f"{tag}_t_nw_{w}"] = es.get("t_nw", np.nan)
        out[f"{tag}_excess_geo_{w}"] = CAP.geo_ann(rs) - CAP.geo_ann(B["pool"].reindex(rs.index))
    x = wsl(r, "2021-2025")
    out[f"{tag}_turnover_ann"] = float(x["turnover"].mean() * TD)
    for c in ("total", "spread", "impact", "commission", "stamp"):
        out[f"{tag}_cost_{c}_bp"] = float(x[f"cost_{c}"].mean() * TD * 1e4)
    out[f"{tag}_holdings"] = float(x["n_hold"].mean())
    nav = (1 + x["ret"]).cumprod()
    out[f"{tag}_max_dd"] = float((nav / nav.cummax() - 1).min())
    out[f"{tag}_exposure_max"] = float(x["exp_max_phase"].max())
    return out


def quarterly_t(d: pd.Series) -> tuple[float, int]:
    x = wsl(d, "2021-2025")
    q = x.groupby(x.index.to_period("Q")).sum()
    return float(q.mean() / q.std(ddof=1) * np.sqrt(len(q))), len(q)


def main():
    B = pd.read_parquet(CAP.BENCH_CACHE.format(OV.MODE))
    base = pd.read_parquet(C.TMP / "inc_t3_base_daily.parquet")
    ov = pd.read_parquet(OV.DAILY)
    bases = {(v, a): g.set_index("date").sort_index() for (v, a), g in base.groupby(["variant", "aum"])}
    rows, yrows, prows = [], [], []
    for (cell, v, a), g in ov.groupby(["cell", "variant", "aum"], sort=False):
        o = g.set_index("date").sort_index()
        b = bases[(v, a)]
        d = (o["ret"] - b["ret"].reindex(o.index)).dropna()
        rec = dict(cell=cell, variant=v, aum=a, X=g["X"].iloc[0], N_pre=g["N_pre"].iloc[0], M=g["M"].iloc[0],
                   tier=g["tier"].iloc[0], src=g["src"].iloc[0],
                   is_main=(cell == OV.cell_name(*OV.MAIN)), in_grid18=(g["N_pre"].iloc[0] in (5, 10, 20) and g["tier"].iloc[0] == "main"))
        rec.update(diff_stats(d))
        rec["quarterly_t"], rec["n_quarters"] = quarterly_t(d)
        gd = (o["gross_ret"] - b["gross_ret"].reindex(o.index)).dropna()
        rec.update(diff_stats(gd, "gross_"))
        rec.update(arm_stats(o, B, "over"))
        rec.update(arm_stats(b, B, "base"))
        rec["d_turnover_ann"] = rec["over_turnover_ann"] - rec["base_turnover_ann"]
        rec["d_cost_total_bp"] = rec["over_cost_total_bp"] - rec["base_cost_total_bp"]
        # 逐年
        for y in range(2021, 2026):
            x = d[d.index.year == y]
            yrows.append(dict(cell=cell, variant=v, aum=a, year=y, ann_diff=float(x.mean() * TD), n=len(x)))
        rec["n_years_pos"] = int(sum(1 for r in yrows[-5:] if r["ann_diff"] > 0))
        # 相位
        pcols = [c for c in o.columns if c.startswith("ret_p")]
        pv = []
        for c in pcols:
            x = wsl((o[c] - b[c].reindex(o.index)).dropna(), "2021-2025")
            pv.append(float(x.mean() * TD))
            prows.append(dict(cell=cell, variant=v, aum=a, phase=c, ann_diff_2021_2025=pv[-1]))
        rec["phase_min"], rec["phase_max"] = (min(pv), max(pv)) if pv else (np.nan, np.nan)
        rec["phase_range"] = rec["phase_max"] - rec["phase_min"]
        rows.append(rec)
    G = pd.DataFrame(rows)
    G.to_csv(C.LEDGER / "inc_t3_overlay_grid.csv", index=False)
    pd.DataFrame(yrows).to_csv(C.LEDGER / "inc_t3_overlay_yearly.csv", index=False)
    pd.DataFrame(prows).to_csv(C.LEDGER / "inc_t3_overlay_phase.csv", index=False)

    # ---- 安慰剂 ----
    fs = sorted(glob.glob(str(C.TMP / "inc_t3_placebo_daily_*.parquet")))
    PL = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True) if fs else pd.DataFrame()
    meta = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(C.TMP / "inc_t3_placebo_meta_*.csv")))],
                     ignore_index=True) if fs else pd.DataFrame()
    prow = []
    b = bases[("Y1.0", 1e7)]
    for cell, g in (PL.groupby("cell", sort=False) if len(PL) else []):
        o = ov[(ov.cell == cell) & (ov.variant == "Y1.0") & (ov.aum == 1e7)].set_index("date").sort_index()
        reps = g.pivot_table(index="date", columns="rep", values="ret").sort_index()
        pm = reps.mean(axis=1)
        rec = dict(cell=cell, n_reps=reps.shape[1])
        rec.update(diff_stats((o["ret"] - pm.reindex(o.index)).dropna(), "over_minus_pl_"))
        rec.update(diff_stats((pm - b["ret"].reindex(pm.index)).dropna(), "pl_minus_base_"))
        per = [(wsl(reps[r] - b["ret"].reindex(reps.index), "2021-2025").mean() * TD) for r in reps.columns]
        rec["pl_minus_base_rep_min"], rec["pl_minus_base_rep_max"] = float(np.min(per)), float(np.max(per))
        od = wsl(o["ret"] - b["ret"].reindex(o.index), "2021-2025").mean() * TD
        rec["over_minus_base_ann"] = float(od)
        rec["rank_over_among_reps"] = int(np.sum(np.array(per) < od))          # 0..10:伪剔除有几套不如真实剔除
        pt = g.groupby("date")["turnover"].mean()
        rec["pl_turnover_ann"] = float(wsl(pt, "2021-2025").mean() * TD)
        if len(meta):
            mm = meta[meta.cell == cell]
            rec["count_ratio_mean"] = float(mm["count_ratio"].mean())
            rec["fallback_any_mean"] = float(mm["n_fallback_any"].mean())
            rec["fallback_age_only_mean"] = float(mm["n_fallback_age_only"].mean())
        prow.append(rec)
    P = pd.DataFrame(prow)
    P.to_csv(C.LEDGER / "inc_t3_placebo.csv", index=False)

    # ---- 判定 ----
    main_ = G[(G.cell == OV.cell_name(*OV.MAIN)) & (G.variant == "Y1.0") & (G.aum == 1e7)].iloc[0]
    g18 = G[(G.variant == "Y1.0") & (G.aum == 1e7) & G.in_grid18]
    n_pos18 = int((g18["ann_2021-2025"] > 0).sum())
    y15 = G[(G.cell == OV.cell_name(*OV.MAIN)) & (G.variant == "Y1.5") & (G.aum == 1e7)]
    pm_ = P[P.cell == OV.cell_name(*OV.MAIN)]
    c1 = (main_["ann_2021-2023"] > 0) and (main_["ann_2024-2025"] > 0)
    c2 = main_["t_2021-2025"] >= 2.0
    c3 = main_["ann_2021-2025"] >= 0.0020
    c4 = n_pos18 >= 12
    c5 = bool(len(pm_) and pm_["over_minus_pl_ann_2021-2025"].iloc[0] > 0)
    no_ = (main_["ann_2021-2025"] <= 0.0010) or ((main_["ann_2021-2023"] <= 0) and (main_["ann_2024-2025"] <= 0)) \
        or (main_["t_2021-2025"] <= 0)
    if c1 and c2 and c3 and c4:
        verdict = "有增量" + ("(解禁信息增量)" if c5 else "(增量来自风格:次新 / 规模,不是解禁信息)")
        if len(y15) and y15["ann_2021-2025"].iloc[0] <= 0:
            verdict += "(成本敏感)"
    elif no_:
        verdict = "无增量"
    else:
        verdict = "证据不足"
    V = pd.DataFrame([
        dict(criterion="①两窗配对日差年化 > 0", value=f"dev {main_['ann_2021-2023']*100:+.3f}pp / test {main_['ann_2024-2025']*100:+.3f}pp", passed=bool(c1)),
        dict(criterion="②全期 NW t(带宽 10)>= 2.0", value=f"{main_['t_2021-2025']:.2f}", passed=bool(c2)),
        dict(criterion="③全期年化 >= +0.20pp", value=f"{main_['ann_2021-2025']*100:+.3f}pp(NW se {main_['se_2021-2025']*100:.3f}pp)", passed=bool(c3)),
        dict(criterion="④18 格中全期为正 >= 12", value=f"{n_pos18}/18", passed=bool(c4)),
        dict(criterion="⑤overlay − 安慰剂均值 全期 > 0(归因标注)",
             value=(f"{pm_['over_minus_pl_ann_2021-2025'].iloc[0]*100:+.3f}pp(t {pm_['over_minus_pl_t_2021-2025'].iloc[0]:.2f})" if len(pm_) else "缺"),
             passed=bool(c5)),
        dict(criterion="Y=1.5 主格全期", value=(f"{y15['ann_2021-2025'].iloc[0]*100:+.3f}pp" if len(y15) else "缺"),
             passed=bool(len(y15) and y15["ann_2021-2025"].iloc[0] > 0)),
        dict(criterion="无增量条件(任一):全期 <= +0.10pp / 两窗都 <= 0 / 全期 t <= 0", value="", passed=bool(no_)),
        dict(criterion="判定", value=verdict, passed=True),
    ])
    V.to_csv(C.LEDGER / "inc_t3_verdict.csv", index=False)
    pd.set_option("display.width", 250)
    show = ["cell", "variant", "aum", "ann_2021-2025", "t_2021-2025", "ann_2021-2023", "ann_2024-2025", "quarterly_t",
            "n_years_pos", "phase_range", "d_turnover_ann", "d_cost_total_bp", "gross_ann_2021-2025"]
    print(G[show].round(5).to_string())
    if len(P):
        print(P.round(5).to_string())
    print(V.to_string())
    plot(G, ov, bases, P, PL)


def plot(G, ov, bases, P, PL):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    blue, orange, aqua, yellow = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw=dict(width_ratios=[1.35, 1]))
    ax = axes[0]
    b = bases[("Y1.0", 1e7)]
    main = OV.cell_name(*OV.MAIN)
    for cell, col, lab in ((main, blue, f"主格 {main}"), ("X10_N20_M0", orange, "X10_N20_M0"),
                           (OV.cell_name(*OV.MAIN, "Ctier"), "#8a8986", "C 层事后上界(不可交易)")):
        o = ov[(ov.cell == cell) & (ov.variant == "Y1.0") & (ov.aum == 1e7)].set_index("date").sort_index()
        if o.empty:
            continue
        d = wsl(o["ret"] - b["ret"].reindex(o.index), "2021-2025")
        ax.plot(d.index, d.cumsum() * 100, color=col, lw=2.0 if cell == main else 1.6,
                ls="--" if "Ctier" in cell else "-", label=lab)
    if len(PL):
        g = PL[PL.cell == main]
        if len(g):
            reps = g.pivot_table(index="date", columns="rep", values="ret").sort_index()
            cs = wsl(reps.sub(b["ret"].reindex(reps.index), axis=0), "2021-2025").cumsum() * 100
            ax.fill_between(cs.index, cs.min(axis=1), cs.max(axis=1), color=aqua, alpha=0.18, lw=0,
                            label="主格安慰剂 10 套范围")
            ax.plot(cs.index, cs.mean(axis=1), color=aqua, lw=1.4, label="主格安慰剂均值")
    ax.axhline(0, color="#888", lw=0.8)
    ax.axvline(pd.Timestamp("2024-01-01"), color="#bbb", lw=0.8, ls=":")
    ax.set_ylabel("累计配对日差(叠加 − 基线,pp)")
    ax.set_title("排雷叠加剔除:成本后配对日差累计(Y=1,1000 万,5 相位)", fontsize=11)
    ax.legend(fontsize=8.5, frameon=False, loc="upper left")
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    # 右:网格热图(全期年化,pp)
    ax = axes[1]
    g = G[(G.variant == "Y1.0") & (G.aum == 1e7) & (G.tier == "main")]
    rowsl = [(n, m) for n in OV.NPRES for m in OV.MS]
    M = np.array([[g[(g.X == x) & (g.N_pre == n) & (g.M == m)]["ann_2021-2025"].iloc[0] * 100
                   for x in OV.XS] for (n, m) in rowsl])
    lim = max(0.3, np.nanmax(np.abs(M)))
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("div", ["#c2410c", "#f2f1ee", "#2a78d6"])
    im = ax.imshow(M, cmap=cmap, vmin=-lim, vmax=lim, aspect="auto")
    for i in range(M.shape[0]):
        for jx in range(M.shape[1]):
            n, m = rowsl[i]
            bold = (OV.XS[jx], n, m) == OV.MAIN
            ax.text(jx, i, f"{M[i, jx]:+.2f}", ha="center", va="center", fontsize=9,
                    fontweight="bold" if bold else "normal", color="#0b0b0b")
    ax.set_xticks(range(len(OV.XS)), [f"X={int(x*100)}%" for x in OV.XS])
    ax.set_yticks(range(len(rowsl)), [f"N_pre={n}, M={m}" for n, m in rowsl], fontsize=8.5)
    ax.set_title("24 格全期年化配对日差(pp,2021-2025;粗体 = 主格)", fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    fig.tight_layout()
    fig.savefig(C.REPORTS / "inc_t3_overlay_diff.png", dpi=140)
    plt.close(fig)


def extras():
    """描述性补充(不进判定):H7 首发原股东两窗差、追加承诺占主格剔除的比重(§11.1 偏乐观上界)、主格事件构成。"""
    out = []
    L = pd.read_csv(C.LEDGER / "inc_t3_event_level.csv.gz", parse_dates=["D"])
    for grp, m in (("首发原股东 ratio_plan>=10%", L["first_type"].str.startswith("首发原股东") & (L["ratio_plan"] >= 0.10)),
                   ("定增 ratio_plan>=10%", (L["tg"] == "定增") & (L["ratio_plan"] >= 0.10)),
                   ("全部 ratio_plan>=10%", L["ratio_plan"] >= 0.10)):
        for col in ("car_B1_[0,20]", "car_B1_[-20,-1]", "car_B2_[0,20]"):
            a = L[m & (L["D"] < "2024-01-01") & (L["D"] >= "2021-01-01") & L[col].notna()]
            b = L[m & (L["D"] >= "2024-01-01") & L[col].notna()]
            dif, se, nG = ES_cluster_diff(b[col].to_numpy()[:, None], b["d_pos"].to_numpy(),
                                          a[col].to_numpy()[:, None], a["d_pos"].to_numpy())
            out.append(dict(item=f"H7 {grp} {col}:检验窗 − 开发窗", dev=a[col].mean(), test=b[col].mean(),
                            diff=float(dif[0]), se=float(se[0]), t=float(dif[0] / se[0]), n_dev=len(a), n_test=len(b)))
    ev = C.load_events()
    T = int(ev["d_pos"].max()) + 1
    x, n, mm = OV.MAIN
    win = OV.exclusion_windows(ev, x, n, mm, 10 ** 6, "main")
    e2 = ev[ev["ratio_plan"] >= x].reset_index(drop=True)
    days = (win["end"] - win["start"] + 1)
    out.append(dict(item="主格剔除窗口中 追加承诺 事件占比(窗口数 / 股票日)",
                    dev=float((e2["tg"] == "追加承诺").mean()),
                    test=float(days[(e2["tg"] == "追加承诺").to_numpy()].sum() / days.sum())))
    out.append(dict(item="主格剔除窗口中 A 层占比", dev=float((e2["tier"] == "A").mean())))
    pd.DataFrame(out).to_csv(C.LEDGER / "inc_t3_extras.csv", index=False)
    print(pd.DataFrame(out).round(5).to_string())


def ES_cluster_diff(x1, g1, x0, g0):
    n1, n0 = len(x1), len(x0)
    m1, m0 = x1.mean(axis=0), x0.mean(axis=0)
    infl = np.concatenate([(x1 - m1) / n1, -(x0 - m0) / n0], axis=0)
    g = np.concatenate([g1, g0])
    G = pd.Series(np.arange(len(g))).groupby(g).indices
    S = np.vstack([infl[ix].sum(axis=0) for ix in G.values()])
    nG = len(G)
    return m1 - m0, np.sqrt((S ** 2).sum(axis=0) * nG / max(nG - 1, 1)), nG


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "extras":
        extras()
    else:
        main()
