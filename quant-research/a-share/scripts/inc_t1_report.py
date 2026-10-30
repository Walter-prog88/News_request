"""INC-T1 判定与图(预注册 §8):读第一层 / 第二层账本,逐策略 × 资金 × 方案给出判定。

产出 research/ledger/inc_t1_summary.csv(全部方案的判定与判据明细)、reports/inc_t1_*.png。
判据(对某策略某资金档的方案 S;ΔC = 可信增量,> 0 = 比现行 W1 同口径省):
  c1 ΔC_full > 0 且开发窗、检验窗均 > 0
  c2 全期配对日度差 NW t ≥ 2(预注册主候选)/ ≥ 2.8(其余)
  c3 ΔC_full ≥ 0.2pp/年(排雷)/ 1.0pp/年(Top20)
  c4 Y ∈ {0.5, 1.5} 与 VWAP/TWAP 口径下 ΔC_full 符号不变
  c5 单窗口方案:第二层终点校正总收益差在开发窗、检验窗均 > 0,且该模式通过引擎断言(非单窗口方案不适用)
  c6 剔除池子级部分(严格口径:再剔除未成交成本中的池子级部分)后仍满足 c1~c3
  降级:增量全部来自模型项 (B) 而已实现部分 (A) 为负或 t < 2 -> 有增量降为证据不足
判定:满足 c1~c6 -> 有增量;c1~c3 满足但 c6 不满足 -> 无增量(择时 beta);两窗异号或 ΔC_full ≤ 0 -> 无增量;
     其余(两窗同为正但 t / 实质性 / 稳健性 / 第二层不达)-> 证据不足。

审查修复(2026-10-11):
  P0  第一层 main = seal dir(整窗口封板的腿按不可成交顺延次日),第二层用 _seal 模式(any 口径);
      seal_none(修复前口径)/ seal_any / seal_tight 作为敏感性列并列(l1_* 列),不进 c4。
  P1  第二层 c5 与"Δ同窗超额"对同价格口径的 W1 参照(total_diff_vs_ref_pp / d_excess_vs_ref_pp)。
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import inc_t1_common as C                                   # noqa: E402
from inc_t1_common import LEDGER, REPORTS                   # noqa: E402
import inc_t1_tradelist as TL                               # noqa: E402

SUBST = {"COMP_v5_exclude": 0.2, "COMP_v5_top20": 1.0, "B_gbt_top20": 1.0}


def main_kind_by_window() -> dict:
    """外部核对规则:预注册窗口 |偏离| 中位 > 10bp 则该窗口主口径改 TWAP。返回 {窗口: 'vwap'/'twap'}。"""
    out = {f"W{k}": "vwap" for k in range(12)}
    f = LEDGER / "inc_t1_ext_vwap_check.csv"
    if f.exists():
        e = pd.read_csv(f)
        for _, r in e[e.prereg & e.rule_switch_to_twap].iterrows():
            for w in str(r.maps_to).split(","):
                out[w] = "twap"
    return out


def scheme_windows(sc: str) -> set:
    legs = TL.scheme_legs(sc)
    return {lg.split("|")[0] for v in legs.values() for lg in v}


def verdicts() -> pd.DataFrame:
    G = pd.read_csv(LEDGER / "inc_t1_layer1_grid.csv")
    L2f = LEDGER / "inc_t1_layer2_grid.csv"
    L2 = pd.read_csv(L2f) if L2f.exists() else pd.DataFrame()
    E = pd.read_csv(LEDGER / "inc_t1_engine_checks.csv")
    mode_ok = E.groupby("mode").passed.all().to_dict()
    mk = main_kind_by_window()
    rows = []
    for (strat, X, sc), g in G.groupby(["strategy", "X", "scheme"]):
        main_var = "twap" if any(mk[w] == "twap" for w in scheme_windows(sc)) else "main"
        alt_var = "main" if main_var == "twap" else "twap"
        per = C.PERIODS[strat]
        role = {v[2]: k for k, v in per.items()}
        gm = g[g.variant == main_var].set_index("period")
        if gm.empty:
            continue
        full, dev, test = gm.loc[role["full"]], gm.loc[role["dev"]], gm.loc[role["test"]]
        thr = 2.0 if sc in C.PRIMARY else 2.8
        sub = SUBST[strat]
        c1 = (full.d_credible_pp > 0) and (dev.d_credible_pp > 0) and (test.d_credible_pp > 0)
        c2 = full.t_credible >= thr
        c3 = full.d_credible_pp >= sub
        sens = {}
        for v in (["Y0.5", "Y1.5"] if main_var == "main" else []) + [alt_var]:
            gv = g[(g.variant == v) & (g.period == role["full"])]
            if len(gv):
                sens[v] = float(gv.d_credible_pp.iloc[0])
        c4 = all(np.sign(x) == np.sign(full.d_credible_pp) for x in sens.values())
        # 第二层(单窗口)
        c5, l2_dev, l2_test, l2_full, l2_mode = np.nan, np.nan, np.nan, np.nan, ""
        l2x_full = l2x_dev = l2x_test = np.nan
        if TL.SCHEME_TYPE[sc] == "single" and sc != "W1" and len(L2):
            kind = "twap" if sc in ("W0", "W8") else ("twap" if main_var == "twap" else "vwap")
            l2_mode = C.mode_name(sc, kind, 1, True)
            h = L2[(L2.strategy == strat) & (L2.X == X) & (L2["mode"] == l2_mode) & (L2.Y == 1.0)].set_index("period")
            if len(h):
                l2_full = float(h.loc[role["full"], "total_diff_vs_ref_pp"])
                l2_dev = float(h.loc[role["dev"], "total_diff_vs_ref_pp"])
                l2_test = float(h.loc[role["test"], "total_diff_vs_ref_pp"])
                l2x_full = float(h.loc[role["full"], "d_excess_vs_ref_pp"])
                l2x_dev = float(h.loc[role["dev"], "d_excess_vs_ref_pp"])
                l2x_test = float(h.loc[role["test"], "d_excess_vs_ref_pp"])
                c5 = bool(l2_dev > 0 and l2_test > 0 and mode_ok.get(l2_mode, False))
        elif TL.SCHEME_TYPE[sc] == "delay" and sc == "D1" and len(L2):
            l2_mode = C.mode_name("W1", "twap", 2, True)
            h = L2[(L2.strategy == strat) & (L2.X == X) & (L2["mode"] == l2_mode) & (L2.Y == 1.0)].set_index("period")
            if len(h):
                l2_full = float(h.loc[role["full"], "total_diff_vs_ref_pp"])
                l2_dev = float(h.loc[role["dev"], "total_diff_vs_ref_pp"])
                l2_test = float(h.loc[role["test"], "total_diff_vs_ref_pp"])
                l2x_full = float(h.loc[role["full"], "d_excess_vs_ref_pp"])
                l2x_dev = float(h.loc[role["dev"], "d_excess_vs_ref_pp"])
                l2x_test = float(h.loc[role["test"], "d_excess_vs_ref_pp"])
        c5_ok = True if (isinstance(c5, float) and np.isnan(c5)) else bool(c5)
        c6 = (full.d_strict_pp > 0 and dev.d_strict_pp > 0 and test.d_strict_pp > 0
              and full.t_strict >= thr and full.d_strict_pp >= sub)
        A_ok = (full.d_realized_A_pp > 0) and (full.t_realized_A >= 2)
        reasons = []
        if not ((dev.d_credible_pp > 0 and test.d_credible_pp > 0) and full.d_credible_pp > 0):
            verdict = "无增量"
            if np.sign(dev.d_credible_pp) != np.sign(test.d_credible_pp):
                reasons.append("两窗异号")
            if full.d_credible_pp <= 0:
                reasons.append("ΔC≤0")
            if dev.d_credible_pp > 0 and test.d_credible_pp > 0 and full.d_credible_pp <= 0:
                reasons.append("全期≤0")
            if dev.d_credible_pp < 0 and test.d_credible_pp < 0:
                reasons.append("两窗同为负")
        elif c2 and c3 and not c6 and not (dev.d_strict_pp > 0 and test.d_strict_pp > 0 and full.d_strict_pp >= sub):
            verdict = "无增量(择时 beta)"
            reasons.append("剔除池子级后点估计不满足 1 或 3")
        elif c2 and c3 and not c6:
            verdict = "证据不足"
            reasons.append(f"严格口径(再剔除未成交中的池子级部分)t={full.t_strict:.2f}<{thr},点估计仍为正(偏离记录第 6 条)")
        elif c2 and c3 and c4 and c5_ok and c6:
            verdict = "有增量"
            if not A_ok:
                verdict = "证据不足"
                reasons.append("降级:已实现部分为负或 t<2,增量来自模型冲击项")
        else:
            verdict = "证据不足"
            if not c2:
                reasons.append(f"t={full.t_credible:.2f}<{thr}")
            if not c3:
                reasons.append(f"ΔC={full.d_credible_pp:.3f}<{sub}")
            if not c4:
                reasons.append("Y/口径敏感性变号")
            if not c5_ok:
                reasons.append("第二层未确认")
        if sc in ("W1", "S:W1|B:W1"):
            verdict, reasons = "基线", ["现行口径(W1 同价格口径)本身"]
        elif sc.startswith("S:") and sc.split("|")[0][2:] == sc.split("|")[1][2:]:
            reasons.append(f"对角线 = 单窗口 {sc.split('|')[1][2:]}(自洽核对,不单独计数)")
        seal_cols = {}
        for sv in ("seal_none", "seal_any", "seal_tight"):
            gv = g[(g.variant == sv)].set_index("period")
            if len(gv):
                seal_cols[f"{sv}_dC_full"] = float(gv.loc[role["full"], "d_credible_pp"])
                seal_cols[f"{sv}_dC_dev"] = float(gv.loc[role["dev"], "d_credible_pp"])
                seal_cols[f"{sv}_dC_test"] = float(gv.loc[role["test"], "d_credible_pp"])
                seal_cols[f"{sv}_t"] = float(gv.loc[role["full"], "t_credible"])
                seal_cols[f"{sv}_realized_A"] = float(gv.loc[role["full"], "d_realized_A_pp"])
                seal_cols[f"{sv}_t_realized_A"] = float(gv.loc[role["full"], "t_realized_A"])
        rows.append(dict(
            strategy=strat, X=X, scheme=sc, type=TL.SCHEME_TYPE[sc], primary=sc in C.PRIMARY, main_variant=main_var,
            dC_full=full.d_credible_pp, dC_dev=dev.d_credible_pp, dC_test=test.d_credible_pp, t_full=full.t_credible,
            t_thr=thr, subst_thr=sub,
            d_px_rel=full.d_px_rel_pp, d_spread=full.d_spread_pp, d_impact=full.d_impact_pp, d_unf=full.d_unf_tot_pp,
            d_unf_mkt=full.d_unf_mkt_pp, d_px_mkt_excluded=full.d_px_mkt_pp,
            d_realized_A=full.d_realized_A_pp, t_realized_A=full.t_realized_A, d_model_B=full.d_model_B_pp,
            t_model_B=full.t_model_B, d_strict_full=full.d_strict_pp, d_strict_dev=dev.d_strict_pp,
            d_strict_test=test.d_strict_pp, t_strict=full.t_strict,
            d_total_incl_mkt=full.d_total_incl_mkt_pp, t_total_incl_mkt=full.t_total_incl_mkt,
            sens_Y05=sens.get("Y0.5", np.nan), sens_Y15=sens.get("Y1.5", np.nan),
            sens_altkind=sens.get(alt_var, np.nan),
            l2_mode=l2_mode, l2_total_diff_full=l2_full, l2_total_diff_dev=l2_dev, l2_total_diff_test=l2_test,
            l2_d_excess_full=l2x_full, l2_d_excess_dev=l2x_dev, l2_d_excess_test=l2x_test,
            c1=c1, c2=c2, c3=c3, c4=c4, c5=c5, c6=c6, A_ok=A_ok,
            capped_share=full.capped_share, unfilled_share=full.unfilled_share, sealed_share=full.sealed_share,
            sealed_legs_ann=full.sealed_legs_ann, **seal_cols,
            base_credible_cost_pp=full.base_credible_cost_pp, verdict=verdict, reasons=";".join(reasons)))
    S = pd.DataFrame(rows)
    # H5:买卖分离的可信增量上限 = 卖出侧与买入侧各自最优单窗口(同一 5 窗口集合)的节省之和
    SD = pd.read_csv(LEDGER / "inc_t1_layer1_sides.csv")
    ub = {}
    for (strat, X), g in SD.groupby(["strategy", "X"]):
        full = [k for k, v in C.PERIODS[strat].items() if v[2] == "full"][0]
        g = g[(g.period == full) & g.scheme.isin(C.SPLIT_SET)]
        best = {sd: g[g.side == sd].sort_values("d_credible_pp").iloc[-1] for sd in ("buy", "sell")}
        ub[(strat, X)] = (float(max(best["sell"].d_credible_pp, 0) + max(best["buy"].d_credible_pp, 0)),
                          f"卖 {best['sell'].scheme} {best['sell'].d_credible_pp:+.2f} + 买 {best['buy'].scheme} {best['buy'].d_credible_pp:+.2f}")
    S["h5_upper_bound"] = [ub.get((r.strategy, r.X), (np.nan, ""))[0] if r.type == "split" else np.nan for r in S.itertuples()]
    S["h5_best_sides"] = [ub.get((r.strategy, r.X), (np.nan, ""))[1] if r.type == "split" else "" for r in S.itertuples()]
    # Preserve the original mechanical criteria; persistent seals are a post-review guard.
    S["verdict_mechanical"] = S["verdict"]
    S["rollN_px_rel_pp"] = np.nan
    S["rollN_t"] = np.nan
    diag = LEDGER / "inc_t1_seal_multiroll.csv"
    if diag.exists():
        R = pd.read_csv(diag)
        for i, r in S.iterrows():
            h = R[(R.X == r.X) & (R.window == r.scheme) &
                  (R["mode"] == "rollN") & (R.role == "full")]
            if r.strategy == "COMP_v5_exclude" and len(h):
                S.loc[i, "rollN_px_rel_pp"] = h.px_rel_saving_pp.iloc[0]
                S.loc[i, "rollN_t"] = h.t_nw.iloc[0]
                if r.verdict == "有增量" and (h.px_rel_saving_pp.iloc[0] <= 0 or h.t_nw.iloc[0] < 2):
                    S.loc[i, "verdict"] = "证据不足"
                    S.loc[i, "reasons"] = str(r.reasons) + ";审查后降级:连续封板顺延诊断已实现 t<2(不改预注册机械判据)"
    S.to_csv(LEDGER / "inc_t1_summary.csv", index=False)
    return S


# ====================================================================== 图
def figures():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Heiti SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    G = pd.read_csv(LEDGER / "inc_t1_layer1_grid.csv")
    ink, muted, grid = "#0b0b0b", "#52514e", "#e4e3df"
    c_dev, c_test, c_full = "#2a78d6", "#eb6834", "#52514e"   # 参考调色板 slot1 / slot2 / 中性灰

    # 1 第一层:单窗口 ΔC(三策略 × 三资金)
    wins = [f"W{k}" for k in range(12) if k != 1]
    fig, axes = plt.subplots(3, 3, figsize=(15, 10), sharex=True)
    for r, strat in enumerate(C.STRAT_KEYS):
        per = C.PERIODS[strat]
        role = {v[2]: k for k, v in per.items()}
        for c, X in enumerate(C.AUMS):
            ax = axes[r, c]
            g = G[(G.strategy == strat) & (G.X == X) & (G.variant == "main") & G.scheme.isin(wins)]
            x = np.arange(len(wins))
            for off, (rl, col) in zip((-0.25, 0, 0.25), (("dev", c_dev), ("full", c_full), ("test", c_test))):
                v = g[g.period == role[rl]].set_index("scheme").reindex(wins).d_credible_pp
                ax.bar(x + off, v, width=0.23, color=col, label={"dev": "开发窗", "full": "全期", "test": "检验窗"}[rl])
            ax.axhline(0, color=muted, lw=0.8)
            ax.set_xticks(x); ax.set_xticklabels(wins, fontsize=8)
            ax.grid(axis="y", color=grid, lw=0.5)
            ax.set_title(f"{C.STRAT_LABEL[strat]} · {C.AUM_LABEL[X]}", fontsize=10, color=ink)
            if c == 0:
                ax.set_ylabel("可信增量 ΔC(pp/年,>0 = 比 W1 省)", fontsize=8)
            for s_ in ("top", "right"):
                ax.spines[s_].set_visible(False)
    axes[0, 0].legend(fontsize=8, frameon=False)
    fig.suptitle("第一层:固定成交清单逐笔重定价(整窗口封板的腿顺延次日),单窗口相对 09:31-09:40 VWAP 的可信增量(Y=1)", fontsize=12)
    fig.tight_layout()
    fig.savefig(REPORTS / "inc_t1_layer1_single.png", dpi=130)
    plt.close(fig)

    # 2 日内路径(1000 万清单):个股相对同窗池子的成本(bp)
    P = pd.read_csv(LEDGER / "inc_t1_layer1_paths.csv")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=False)
    for c, strat in enumerate(C.STRAT_KEYS):
        ax = axes[c]
        per = C.PERIODS[strat]
        role = {v[2]: k for k, v in per.items()}
        for side, col in (("buy", "#2a78d6"), ("sell", "#eb6834")):
            for rl, ls in (("dev", "-"), ("test", "--")):
                g = P[(P.strategy == strat) & (P.X == 1e7) & (P.side == side) & (P.period == role[rl])
                      & (P.variant == "rolled")].sort_values("idx")
                ax.plot(g.point, g.rel_cost_bp, ls=ls, color=col, marker="o", ms=3,
                        label=f"{'买' if side == 'buy' else '卖'}·{'开发窗' if rl == 'dev' else '检验窗'}")
        ax.axhline(0, color=muted, lw=0.8)
        ax.set_title(C.STRAT_LABEL[strat], fontsize=10)
        ax.tick_params(axis="x", labelsize=8, rotation=45)
        ax.grid(axis="y", color=grid, lw=0.5)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
        if c == 0:
            ax.set_ylabel("个股相对同时点池子的成本(bp,决策价 = T 收盘)", fontsize=8)
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle("交易清单在 T+1 九个时点的相对路径(1000 万清单,整窗口封板的腿顺延 T+2 同时点;正 = 在该时点成交比池子平均更贵)", fontsize=11)
    fig.tight_layout()
    fig.savefig(REPORTS / "inc_t1_paths.png", dpi=130)
    plt.close(fig)

    # 3 第二层:单窗口模式对现行 open10 的终点校正总收益差
    L2f = LEDGER / "inc_t1_layer2_grid.csv"
    if L2f.exists():
        L2 = pd.read_csv(L2f)
        modes = [m for m in L2[L2.sealed]["mode"].unique()
                 if m not in (C.mode_name("W1", "twap", 1, True), C.mode_name("W1", "vwap", 1, True))]
        modes = sorted(modes, key=lambda m: (int(m.split("_")[2][1:]), m))
        fig, axes = plt.subplots(3, 3, figsize=(15, 10), sharex=True)
        for r, strat in enumerate(C.STRAT_KEYS):
            per = C.PERIODS[strat]
            role = {v[2]: k for k, v in per.items()}
            for c, X in enumerate(C.AUMS):
                ax = axes[r, c]
                g = L2[(L2.strategy == strat) & (L2.X == X) & (L2.Y == 1.0) & L2.sealed]
                x = np.arange(len(modes))
                for off, (rl, col) in zip((-0.25, 0, 0.25), (("dev", c_dev), ("full", c_full), ("test", c_test))):
                    v = g[g.period == role[rl]].set_index("mode").reindex(modes).total_diff_vs_ref_pp
                    ax.bar(x + off, v, width=0.23, color=col, label={"dev": "开发窗", "full": "全期", "test": "检验窗"}[rl])
                ax.axhline(0, color=muted, lw=0.8)
                ax.set_xticks(x)
                ax.set_xticklabels([m.replace("inc_t1_", "").replace("_seal", "").replace("_vwap", "").replace("_twap", "T") for m in modes],
                                   fontsize=7, rotation=60)
                ax.grid(axis="y", color=grid, lw=0.5)
                ax.set_title(f"{C.STRAT_LABEL[strat]} · {C.AUM_LABEL[X]}", fontsize=10)
                if c == 0:
                    ax.set_ylabel("总收益差 vs 同价格口径 W1(pp/年,收盘统一计价)", fontsize=8)
                for s_ in ("top", "right"):
                    ax.spines[s_].set_visible(False)
        axes[0, 0].legend(fontsize=8, frameon=False)
        fig.suptitle("第二层(封板按不可成交,any 口径):引擎按各窗口成交,终点按收盘统一计价后对同价格口径 W1 的年化总收益差\n"
                     "W1T_lag2 = 延迟一天成交(对 W1 TWAP);W0 / W8 为开盘 / 收盘集合竞价(对 W1 VWAP);其余对 W1 VWAP", fontsize=11)
        fig.tight_layout()
        fig.savefig(REPORTS / "inc_t1_layer2.png", dpi=130)
        plt.close(fig)


if __name__ == "__main__":
    S = verdicts()
    pd.set_option("display.width", 250)
    print(S[S.primary | (S.verdict != "无增量")][["strategy", "X", "scheme", "dC_full", "dC_dev", "dC_test", "t_full",
                                                  "d_realized_A", "d_model_B", "l2_total_diff_dev", "l2_total_diff_test",
                                                  "verdict", "reasons"]].round(3).to_string())
    figures()
