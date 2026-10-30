"""把容量账本渲染成 research/CAPACITY_REPORT.md(前缀 cap,2026-10-10;审查修复后重写)。

为什么用脚本渲染:报告里的每个表格数字都直接从 research/ledger/capacity_*.csv 读出、格式化后填进模板,
不经手抄,保证"报告数字 = 账本数字"。正文里的叙述性数字一部分同样由本脚本从账本算出后填入
({{N:...}} 占位符,见 numbers()),其余手写数字出自主网格(sizing=pov,与审查前逐位一致)并在段落后注明列名。
先跑 scripts/cap_capacity_curve.py 的 run(两个口径 × 两种截量)/ checks / summarize,再跑本脚本。
2026-10-10 第二轮修订(fixa):基准换成同频池等权 / 成交时点代理后,原先手写在模板里的叙述性数字改由
numbers_fixa() 从账本算出;文末"第二轮修订"对照表读 research/ledger/fixa_capacity_before_after.csv
(scripts/fixa_benchmark_fix.py capdiff,修订前账本快照 vs 修订后账本)。"审查记录"一节保留第一轮审查时的数字(定格文本)。
同日第二轮修订的复核审查(fixa review2):numbers_r2() 读 capacity_grid / summary(100 亿格的 NW t 与 ±1SE 盈亏平衡)、
research/ledger/fixa_r2_mech.csv(再平衡频率差的机制)与 fixa_r2_barra_delta.csv(BARRA 归因换同频基准的变化)。

    python3 scripts/cap_render_report.py
"""
from __future__ import annotations
import os, sys, re
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from afactor.config import LEDGER, ROOT

NAME = {"COMP_v5_exclude": "COMP_v5 排雷 20/35", "COMP_v5_top20": "COMP_v5 Top20",
        "COMP_v5_top50": "COMP_v5 Top50", "B_gbt_top20": "B×GBT Top20(OOS)"}
STR = list(NAME)
TOPN = STR[1:]
AUMS = [1e6, 1e7, 3e7, 1e8, 3e8, 1e9, 3e9, 1e10]
LAB = ["100万", "1000万", "3000万", "1亿", "3亿", "10亿", "30亿", "100亿"]
FLAG = {"gt_max": ">100亿", "lt_min": "<100万", "ref_nonpositive": "—(1000万已≤0)", "nan": "—"}
SZ = {"pov": "POV", "exante": "事前"}


def money(a, flag=None) -> str:
    if a is None or not np.isfinite(a):
        return FLAG.get(flag, str(flag))
    return f"{a/1e8:.1f}亿" if a >= 1e8 else f"{a/1e4:.0f}万"


def pp(x) -> str:
    return f"{x*100:+.2f}"


def load():
    return dict(G=pd.read_csv(LEDGER / "capacity_grid.csv"), S=pd.read_csv(LEDGER / "capacity_summary.csv"),
                SC=pd.read_csv(LEDGER / "capacity_scaling.csv"), D=pd.read_csv(LEDGER / "capacity_decomp.csv"),
                DG=pd.read_csv(LEDGER / "capacity_diagnostics.csv"), CK=pd.read_csv(LEDGER / "capacity_checks.csv"),
                PL=pd.read_csv(LEDGER / "capacity_placebo.csv"))


def q(df, **kw) -> pd.DataFrame:
    """按列等值筛选。"""
    m = np.ones(len(df), bool)
    for k, v in kw.items():
        m &= (df[k] == v).to_numpy()
    return df[m]


def one(df, **kw) -> pd.Series:
    r = q(df, **kw)
    if len(r) != 1:
        raise KeyError(f"{kw} 匹配 {len(r)} 行")
    return r.iloc[0]


def cap(r, c) -> str:
    return money(r[c], r[f"{c}_flag"])


# ====================================================================== 表
def t_headline(G, S) -> str:
    out = ["| 策略 | 口径 | 窗口 | 截量 | 100万 | 1000万 | 1亿 | 10亿 | 100亿 | 1000万 算术 | 100亿 算术 | 半衰容量 | 盈亏平衡 | "
           "盈亏平衡(算术) | 盈亏平衡 ±1SE | 分批×5 盈亏平衡 |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for m in ("open10", "vwap1"):
        for s in STR:
            for sz in ("pov", "exante"):
                g = q(G, strategy=s, mode=m, variant="Y1.0", sizing=sz, window_role="full").set_index("aum")
                r = one(S, strategy=s, mode=m, variant="Y1.0", sizing=sz, window_role="full")
                vals = " | ".join(pp(g.loc[a, "excess_pool"]) for a in (1e6, 1e7, 1e8, 1e9, 1e10))
                ar = f"{pp(g.loc[1e7, 'excess_pool_arith'])} | {pp(g.loc[1e10, 'excess_pool_arith'])}"
                if np.isfinite(r.cap_be_pool):
                    t5 = money(r.cap_be_pool_tranche5)
                else:
                    t5 = {"gt_max": ">500亿", "lt_min": "<500万"}.get(r.cap_be_pool_flag, "—")
                out.append(f"| {NAME[s]} | {m} | {r.window} | {SZ[sz]} | {vals} | {ar} | {cap(r, 'cap_half_pool')} | "
                           f"{cap(r, 'cap_be_pool')} | {cap(r, 'cap_be_arith')} | {cap(r, 'cap_be_arith_lo')} ~ "
                           f"{cap(r, 'cap_be_arith_hi')} | {t5} |")
    return "\n".join(out)


def t_curves(G) -> str:
    out = []
    for m in ("open10", "vwap1"):
        out += [f"**{m}**(POV 主网格;括号内为事前截量)", "", "| 策略 | 成本设定 | " + " | ".join(LAB) + " |",
                "|---|---|" + "---|" * 8]
        for s in STR:
            for v in ("legacy", "Y0.5", "Y1.0", "Y1.5", "Y1.0_2tick", "Y1.0_ride"):
                g = q(G, strategy=s, mode=m, variant=v, sizing="pov", window_role="full").set_index("aum")
                e = q(G, strategy=s, mode=m, variant=v, sizing="exante", window_role="full").set_index("aum")
                cells = []
                for a in AUMS:
                    x, y = g.loc[a, "excess_pool"] * 100, e.loc[a, "excess_pool"] * 100
                    cells.append(f"{x:+.1f}" if abs(x - y) < 0.05 else f"{x:+.1f} ({y:+.1f})")
                out.append(f"| {NAME[s]} | {v} | " + " | ".join(cells) + " |")
        out.append("")
    return "\n".join(out)


def t_capacity(S) -> str:
    out = ["| 策略 | 口径 | 窗口 | Y | 1000万净超额 | NW t | 半衰 POV | 半衰 事前 | 盈亏平衡 POV | 盈亏平衡 事前 | "
           "盈亏平衡 ±1SE(POV) |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for m in ("open10", "vwap1"):
        for s in STR:
            for role in ("full", "dev", "test"):
                for v in ("Y0.5", "Y1.0", "Y1.5"):
                    r = one(S, strategy=s, mode=m, variant=v, sizing="pov", window_role=role)
                    e = one(S, strategy=s, mode=m, variant=v, sizing="exante", window_role=role)
                    out.append(f"| {NAME[s]} | {m} | {r.window} | {v[1:]} | {pp(r.excess_pool_1e7)} | {r.t_nw_1e7:.2f} | "
                               f"{cap(r, 'cap_half_pool')} | {cap(e, 'cap_half_pool')} | {cap(r, 'cap_be_pool')} | "
                               f"{cap(e, 'cap_be_pool')} | {cap(r, 'cap_be_arith_lo')} ~ {cap(r, 'cap_be_arith_hi')} |")
    return "\n".join(out)


def t_extras(S) -> str:
    out = ["| 策略 | 口径 | 设定 | 1000万净超额 POV | 1000万净超额 事前 | 半衰 POV / 事前 | 盈亏平衡 POV / 事前 |",
           "|---|---|---|---|---|---|---|"]
    for m in ("open10", "vwap1"):
        for s in STR:
            for v in ("legacy", "Y1.0_2tick", "Y1.0_ride"):
                r = one(S, strategy=s, mode=m, variant=v, sizing="pov", window_role="full")
                e = one(S, strategy=s, mode=m, variant=v, sizing="exante", window_role="full")
                out.append(f"| {NAME[s]} | {m} | {v} | {pp(r.excess_pool_1e7)} | {pp(e.excess_pool_1e7)} | "
                           f"{cap(r, 'cap_half_pool')} / {cap(e, 'cap_half_pool')} | "
                           f"{cap(r, 'cap_be_pool')} / {cap(e, 'cap_be_pool')} |")
    return "\n".join(out)


def t_sizing(G, S) -> str:
    """截量口径对比(Y=1 与 legacy,全期 + 检验窗):净超额与容量。"""
    out = ["| 策略 | 口径 | 成本 | 窗口 | 截量 | 1000万 | 3000万 | 1亿 | 10亿 | 100亿 | 100亿 毛超额(算术) | 100亿 现金 | 半衰容量 | 盈亏平衡 |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for m in ("open10", "vwap1"):
        for s in STR:
            for v in ("Y1.0", "legacy"):
                for role in ("full", "test"):
                    if v == "legacy" and role == "test":
                        continue
                    for sz in ("pov", "exante"):
                        g = q(G, strategy=s, mode=m, variant=v, sizing=sz, window_role=role).set_index("aum")
                        r = one(S, strategy=s, mode=m, variant=v, sizing=sz, window_role=role)
                        vals = " | ".join(pp(g.loc[a, "excess_pool"]) for a in (1e7, 3e7, 1e8, 1e9, 1e10))
                        out.append(f"| {NAME[s]} | {m} | {v} | {r.window} | {SZ[sz]} | {vals} | "
                                   f"{pp(g.loc[1e10, 'gross_excess_pool_arith'])} | {g.loc[1e10, 'cash_mean']*100:.1f}% | "
                                   f"{cap(r, 'cap_half_pool')} | {cap(r, 'cap_be_pool')} |")
    return "\n".join(out)


def t_cash(G, S) -> str:
    """现金计价敏感性:盈亏平衡(现金零收益 / 按池收益补仓 / 按中证1000 代理补仓)与盈亏平衡处的现金比例。"""
    out = ["| 策略 | 口径 | 成本 | 窗口 | 截量 | 现金=0(主口径) | 现金按池收益 | 现金按中证1000代理 | 主口径盈亏平衡处现金 | "
           "现金 1000万 / 3000万 / 1亿 |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    rows = []
    for m in ("open10", "vwap1"):
        for s in TOPN:
            for v in ("legacy", "Y0.5", "Y1.0"):
                rows.append((s, m, v, "full"))
    rows += [("B_gbt_top20", "open10", "Y0.5", "dev"), ("B_gbt_top20", "open10", "Y1.0", "dev"),
             ("COMP_v5_top50", "open10", "Y0.5", "test"), ("COMP_v5_top50", "vwap1", "Y0.5", "test")]
    for s, m, v, role in rows:
        for sz in ("pov", "exante"):
            r = one(S, strategy=s, mode=m, variant=v, sizing=sz, window_role=role)
            g = q(G, strategy=s, mode=m, variant=v, sizing=sz, window_role=role).set_index("aum")
            c = " / ".join(f"{g.loc[a, 'cash_mean']*100:.1f}%" for a in (1e7, 3e7, 1e8))
            ca = r.cash_at_cap_be_pool
            out.append(f"| {NAME[s]} | {m} | {v} | {r.window} | {SZ[sz]} | {cap(r, 'cap_be_pool')} | {cap(r, 'cap_be_cashpool')} | "
                       f"{cap(r, 'cap_be_cashcsi')} | {('—' if not np.isfinite(ca) else f'{ca*100:.1f}%')} | {c} |")
    return "\n".join(out)


def t_yuan(S) -> str:
    """元超额峰值(X × 算术净超额,网格点上取最大;非预注册):现金零收益 vs 现金按池收益补仓。"""
    out = ["| 策略 | 口径 | 成本 | 截量 | 现金=0:最大元超额 @ 规模 | 现金按池收益:最大元超额 @ 规模 |", "|---|---|---|---|---|---|"]
    for m in ("open10", "vwap1"):
        for s in TOPN:
            for v in ("legacy", "Y0.5", "Y1.0"):
                for sz in ("pov", "exante"):
                    r = one(S, strategy=s, mode=m, variant=v, sizing=sz, window_role="full")
                    c = [f"{r[f'yuan_alpha_max_{t}']/1e4:+,.0f} 万元/年 @ {money(r[f'aum_at_yuan_alpha_max_{t}'])}"
                         for t in ("cash0", "cashpool")]
                    out.append(f"| {NAME[s]} | {m} | {v} | {SZ[sz]} | {c[0]} | {c[1]} |")
    return "\n".join(out)


def t_csi(G, S) -> str:
    out = ["| 策略 | 口径 | 窗口 | 同频池等权年化 | 中证1000代理(成交时点)年化 | 真实000852年化(价格) | 策略年化 | vs池 | vs代理 | vs真实指数 | vs代理 NW t | 盈亏平衡(vs代理) |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for m in ("open10", "vwap1"):
        for s in STR:
            g = one(G, strategy=s, mode=m, variant="Y1.0", sizing="pov", window_role="full", aum=1e7)
            r = one(S, strategy=s, mode=m, variant="Y1.0", sizing="pov", window_role="full")
            out.append(f"| {NAME[s]} | {m} | {g.window} | {g.bench_pool_ann*100:.2f}% | {g.bench_csi1000p_ann*100:.2f}% | "
                       f"{g.bench_csi1000r_ann*100:.2f}% | {g.ann_return*100:.2f}% | {pp(g.excess_pool)} | {pp(g.excess_csi1000p)} | "
                       f"{pp(g.excess_csi1000r)} | {g.t_nw_csi1000p:.2f} | {money(r.cap_be_csi1000p, r.cap_be_csi1000p_flag)} |")
    return "\n".join(out)


def t_costs(G) -> str:
    out = ["| 策略 | 口径 | AUM | 年单边换手 | 佣金 bp | 印花税 bp | 半价差 bp | 冲击 bp | 合计 pp/年 | 平均现金 | "
           "调仓日意图成交额被截比例 | 调仓日被截股票数 | 平均持仓数 |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for m in ("open10", "vwap1"):
        for s in STR:
            for a in (1e6, 1e7, 1e8, 1e9, 1e10):
                g = one(G, strategy=s, mode=m, variant="Y1.0", sizing="pov", window_role="full", aum=a)
                out.append(f"| {NAME[s]} | {m} | {LAB[AUMS.index(a)]} | {g.turnover_ann:.1f} | {g.cost_commission_bp:.0f} | "
                           f"{g.cost_stamp_bp:.0f} | {g.cost_spread_bp:.0f} | {g.cost_impact_bp:.0f} | {g.cost_total_bp/100:.2f} | "
                           f"{g.cash_mean*100:.1f}% | {g.part_bound_frac_rebal*100:.1f}% | {g.n_part_bound_rebal:.1f} | {g.holdings:.0f} |")
    return "\n".join(out)


def t_decomp(D) -> str:
    out = ["| 策略 | 口径 | AUM | Δ净超额 | Δ毛超额 | 其中现金拖累 | Δ成本 | 其中Δ半价差 | 其中Δ冲击 | Δ现金比例 |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for m in ("open10", "vwap1"):
        for s in STR:
            for a in (1e8, 1e9, 1e10):
                r = one(D, strategy=s, mode=m, variant="Y1.0", sizing="pov", window_role="full", aum=a)
                out.append(f"| {NAME[s]} | {m} | {LAB[AUMS.index(a)]} | {r.d_net_pp:+.2f} | {r.d_gross_pp:+.2f} | "
                           f"{-r.d_cash_drag_pp:+.2f} | {r.d_cost_pp:+.2f} | {r.d_spread_pp:+.2f} | {r.d_impact_pp:+.2f} | "
                           f"{r.d_cash*100:+.1f}pp |")
    return "\n".join(out)


def t_scaling(SC) -> str:
    out = ["| 策略 | 口径 | 窗口 | 年单边换手 | 持仓数 | 每元毛超额 α_tr | 每元线性成本 c_lin | 其中半价差 | 毛利 α_tr−c_lin | "
           "每元冲击 100万 | 每元冲击 1000万 | 外推盈亏平衡 | 每元冲击因子 | 毛利因子 |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for m in ("open10", "vwap1"):
        for s in STR:
            for w in (["2021-2025", "2024-2025"] if s != "B_gbt_top20" else ["2023-2025", "2024-2025"]):
                r = one(SC, strategy=s, mode=m, variant="Y1.0", sizing="pov", window=w)
                xb = money(r.x_be_sqrt_extrap) if np.isfinite(r.x_be_sqrt_extrap) else "无(毛利≤0)"
                ok = s != "COMP_v5_exclude" and np.isfinite(r.impact_penalty_vs_exclude)
                cf = f"{r.impact_penalty_vs_exclude:.0f}×" if ok else "—"
                mf = ("—" if not ok else ("毛利≤0" if not np.isfinite(r.margin_penalty_vs_exclude)
                                          else f"{r.margin_penalty_vs_exclude:.0f}×"))
                out.append(f"| {NAME[s]} | {m} | {w} | {r.turnover_ann_1e6:.1f} | {r.holdings_1e6:.0f} | {r.alpha_per_traded_bp:.1f}bp | "
                           f"{r.lin_cost_per_traded_bp:.1f}bp | {r.spread_per_traded_bp:.1f}bp | {r.margin_bp:.1f}bp | "
                           f"{r.impact_per_traded_bp_1e6:.2f}bp | {r.impact_per_traded_bp_1e7:.2f}bp | {xb} | {cf} | {mf} |")
    return "\n".join(out)


def t_extrap(SC) -> str:
    """排雷:平方根外推(交易路径不变)vs 引擎网格的算术净超额。"""
    out = ["| 口径 | 窗口 | 截量 | 外推 10亿 | 网格 10亿 | 外推 30亿 | 网格 30亿 | 外推 100亿 | 网格 100亿 | 年单边换手 100万 → 100亿 | 外推盈亏平衡 |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for m in ("open10", "vwap1"):
        for w in ("2021-2025", "2024-2025"):
            for sz in ("pov", "exante"):
                r = one(SC, strategy="COMP_v5_exclude", mode=m, variant="Y1.0", sizing=sz, window=w)
                out.append(f"| {m} | {w} | {SZ[sz]} | {r['pred_net_arith_1e+09']:+.2f} | {r['grid_net_arith_1e+09']:+.2f} | "
                           f"{r['pred_net_arith_3e+09']:+.2f} | {r['grid_net_arith_3e+09']:+.2f} | "
                           f"{r['pred_net_arith_1e+10']:+.2f} | {r['grid_net_arith_1e+10']:+.2f} | "
                           f"{r.turnover_ann_1e6:.2f} → {r['turnover_ann_1e+10']:.2f} | {money(r.x_be_sqrt_extrap)} |")
    return "\n".join(out)


def _note_num(note: str, key: str) -> str:
    m = re.search(re.escape(key) + r"\s*([+\-]?[0-9.]+)", note)
    return m.group(1) if m else "?"


def t_placebo(CK) -> str:
    out = ["| 口径 | AUM | pov − 事前(几何,均值) | 种子间 sd | min / max | >0 的种子 | 日配对差 NW t | pov 超额均值 | 事前超额均值 | "
           "仓位 pov / 事前 | 现金按池收益补仓后的差 |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in q(CK, check="placebo_random_top20_pov_minus_exante").iterrows():
        n = r.note
        mm = re.search(r"min/max ([+\-][0-9.]+)/([+\-][0-9.]+)pp", n)
        npos = re.search(r">0 的种子 (\d+/\d+)", n).group(1)
        out.append(f"| {r['mode']} | {money(float(r['item'].split()[1]))} | {r.value*100:+.2f}pp | {_note_num(n, '种子间 sd')}pp | "
                   f"{mm.group(1)} / {mm.group(2)}pp | {npos} | {_note_num(n, 'NW t=')} | {_note_num(n, 'pov 超额均值')}pp | "
                   f"{_note_num(n, 'exante')}pp | {_note_num(n, '仓位 pov')} / {_note_num(n, '/ exante')} | "
                   f"{_note_num(n, '现金按池收益补仓后差 均值')}pp |")
    return "\n".join(out)


def t_mech(CK) -> str:
    out = ["| 检验 | 口径 | 对象 | 数值 | 说明 |", "|---|---|---|---|---|"]
    for chk in ("exante_fallback_share", "exante_actual_over_est", "exante_mechanism_rankcorr", "exante_realized_participation"):
        for _, r in q(CK, check=chk).iterrows():
            val = f"{r.value:.2e}" if chk == "exante_fallback_share" else (
                f"{r.value*100:.1f}%" if chk == "exante_realized_participation" else f"{r.value:.4f}" if
                chk == "exante_mechanism_rankcorr" else f"{r.value:.3f}")
            out.append(f"| `{chk}` | {r['mode']} | {r['item']} | {val} | {r.note} |")
    return "\n".join(out)


def t_cold(CK) -> str:
    out = ["| 检验 | 口径 | 对象 | 数值(pp) | 参照(pp) | 说明 |", "|---|---|---|---|---|---|"]
    for chk in ("bgbt_coldstart_shortfall", "bgbt_coldstart_proxy"):
        for _, r in q(CK, check=chk).iterrows():
            ref = f"{r.reference*100:+.2f}" if chk == "bgbt_coldstart_shortfall" else "—"
            out.append(f"| `{chk}` | {r['mode']} | {r['item']} | {r.value*100:+.2f} | {ref} | {r.note} |")
    return "\n".join(out)


def diag_fill(DG) -> dict:
    p = DG.pivot(index="item", columns="year", values="value")
    out = {}
    for y in range(2021, 2026):
        out[f"fvol_{y}"] = f"{p.loc['fvol_open10_ratio_of_sums', y]*100:.1f}%"
        out[f"fvar_{y}"] = f"{p.loc['fvar_open10_parkinson_ratio_of_means', y]*100:.1f}%"
        out[f"scale_{y}"] = f"{p.loc['window_impact_scale_sqrt_fvar_over_fvol', y]:.2f}"
        out[f"smed_{y}"] = f"{p.loc['window_impact_scale_median_stockday', y]:.2f}"
        out[f"pamt_{y}"] = f"{p.loc['pool_amount_sum_median_yuan', y]/1e12:.2f}"
        out[f"pa10_{y}"] = f"{p.loc['pool_amt_open10_sum_median_yuan', y]/1e8:,.0f}"
        out[f"mkt_{y}"] = f"{p.loc['market_amount_sum_median_yuan', y]/1e12:.2f}"
    return out


# ====================================================================== 叙述性数字(全部从账本算)
def numbers(d) -> dict:
    G, S, SC, CK = d["G"], d["S"], d["SC"], d["CK"]
    N = {}

    def g(s, m, v, sz, role, a, col="excess_pool"):
        return one(G, strategy=s, mode=m, variant=v, sizing=sz, window_role=role, aum=a)[col]

    def s_(s, m, v, sz, role):
        return one(S, strategy=s, mode=m, variant=v, sizing=sz, window_role=role)

    # 排雷:两种截量
    for m in ("open10", "vwap1"):
        for sz in ("pov", "exante"):
            for a, tag in ((1e7, "1e7"), (1e10, "1e10")):
                N[f"ex_{m}_{sz}_{tag}"] = pp(g("COMP_v5_exclude", m, "Y1.0", sz, "full", a))
                N[f"exa_{m}_{sz}_{tag}"] = pp(g("COMP_v5_exclude", m, "Y1.0", sz, "full", a, "excess_pool_arith"))
                N[f"ext_{m}_{sz}_{tag}"] = pp(g("COMP_v5_exclude", m, "Y1.0", sz, "test", a))
            N[f"exg_{m}_{sz}_1e10"] = pp(g("COMP_v5_exclude", m, "Y1.0", sz, "full", 1e10, "gross_excess_pool_arith"))
            N[f"excash_{m}_{sz}_1e10"] = f"{g('COMP_v5_exclude', m, 'Y1.0', sz, 'full', 1e10, 'cash_mean')*100:.1f}%"
            for v in ("Y1.0", "Y1.5"):
                for role in ("full", "dev", "test"):
                    r = s_("COMP_v5_exclude", m, v, sz, role)
                    N[f"exhalf_{m}_{v}_{sz}_{role}"] = cap(r, "cap_half_pool")
                    N[f"exbe_{m}_{v}_{sz}_{role}"] = cap(r, "cap_be_pool")
    # 排雷盈亏平衡在全部预注册格(Y、窗口)里的最小值(两种截量)
    for sz in ("pov", "exante"):
        rr = q(S, strategy="COMP_v5_exclude", sizing=sz)
        rr = rr[rr.variant.isin(["Y0.5", "Y1.0", "Y1.5"])]
        fin = rr[np.isfinite(rr.cap_be_pool)]
        N[f"exbe_min_{sz}"] = ("全部 >100亿" if len(fin) == 0 else
                               "; ".join(f"{r_['mode']} {r_['variant']} {r_['window']} {money(r_['cap_be_pool'])}"
                                         for _, r_ in fin.iterrows()))
    # Top-N:Y=1 全期 1000 万格范围与盈亏平衡上限
    for sz in ("pov", "exante"):
        vals = [g(s, m, "Y1.0", sz, "full", 1e7) for s in TOPN for m in ("open10", "vwap1")]
        N[f"topn_1e7_range_{sz}"] = f"{min(vals)*100:+.2f} ~ {max(vals)*100:+.2f}"
        bes = [s_(s, m, "Y1.0", sz, "full") for s in TOPN for m in ("open10", "vwap1")]
        fin = [r_.cap_be_pool for r_ in bes if np.isfinite(r_.cap_be_pool)]
        N[f"topn_be_max_{sz}"] = money(max(fin)) if fin else "<100万"
        for s in TOPN:
            for m in ("open10", "vwap1"):
                for v in ("legacy", "Y0.5", "Y1.0"):
                    r = s_(s, m, v, sz, "full")
                    N[f"be_{s}_{m}_{v}_{sz}"] = cap(r, "cap_be_pool")
                    N[f"becp_{s}_{m}_{v}_{sz}"] = cap(r, "cap_be_cashpool")
                    N[f"becc_{s}_{m}_{v}_{sz}"] = cap(r, "cap_be_cashcsi")
                    ca = r.cash_at_cap_be_pool
                    N[f"cashbe_{s}_{m}_{v}_{sz}"] = "—" if not np.isfinite(ca) else f"{ca*100:.1f}%"
                N[f"e1e7_{s}_{m}_{sz}"] = pp(g(s, m, "Y1.0", sz, "full", 1e7))
                N[f"e1e8_{s}_{m}_{sz}"] = pp(g(s, m, "Y1.0", sz, "full", 1e8))
    # B×GBT 细节
    for sz in ("pov", "exante"):
        for v in ("legacy", "Y0.5", "Y1.0"):
            N[f"bg_open10_{v}_{sz}_1e7"] = pp(g("B_gbt_top20", "open10", v, sz, "full", 1e7))
        r = s_("B_gbt_top20", "open10", "Y0.5", sz, "full")
        N[f"bg_be_y05_{sz}"] = cap(r, "cap_be_pool")
        N[f"bg_becp_y05_{sz}"] = cap(r, "cap_be_cashpool")
        N[f"bg_be_y1_2023_{sz}"] = cap(s_("B_gbt_top20", "open10", "Y1.0", sz, "dev"), "cap_be_pool")
        N[f"bg_becp_y1_2023_{sz}"] = cap(s_("B_gbt_top20", "open10", "Y1.0", sz, "dev"), "cap_be_cashpool")
    for v in ("Y0.5", "Y1.0"):
        r = one(G, strategy="B_gbt_top20", mode="open10", variant=v, sizing="pov", window_role="test", aum=1e6)
        N[f"bg_test_1e6_{v}"] = f"几何 {pp(r.excess_pool)} / 算术 {pp(r.excess_pool_arith)}pp(NW t {r.t_nw:.2f})"
    # 现金计价:Top50 2024-2025 Y0.5
    for m in ("open10", "vwap1"):
        r = s_("COMP_v5_top50", m, "Y0.5", "pov", "test")
        N[f"t50_test_y05_{m}"] = f"{cap(r, 'cap_be_pool')} → {cap(r, 'cap_be_cashpool')}"
    # 排雷 Y1.0_ride vs 主口径
    for m in ("open10", "vwap1"):
        N[f"ride_ex_{m}"] = (f"{pp(g('COMP_v5_exclude', m, 'Y1.0_ride', 'pov', 'full', 1e10))} vs "
                             f"{pp(g('COMP_v5_exclude', m, 'Y1.0', 'pov', 'full', 1e10))}")
    rd = []
    for s in TOPN:
        for m in ("open10", "vwap1"):
            for a in AUMS:
                rd.append((g(s, m, "Y1.0_ride", "pov", "full", a) - g(s, m, "Y1.0", "pov", "full", a), s, m, a))
    rd.sort(key=lambda x: -abs(x[0]))
    N["ride_topn_max"] = "、".join(f"{NAME[s]} {m} {money(a)} {x*100:+.2f}pp" for x, s, m, a in rd[:3])
    for s, m in (("COMP_v5_top50", "vwap1"), ("B_gbt_top20", "vwap1")):
        N[f"ride_be_{s}_{m}"] = (f"{cap(s_(s, m, 'Y1.0', 'pov', 'full'), 'cap_be_pool')} → "
                                 f"{cap(s_(s, m, 'Y1.0_ride', 'pov', 'full'), 'cap_be_pool')}")
    N["ride_half_ex_vwap1"] = (f"{cap(s_('COMP_v5_exclude', 'vwap1', 'Y1.0_ride', 'pov', 'full'), 'cap_half_pool')} vs "
                               f"{cap(s_('COMP_v5_exclude', 'vwap1', 'Y1.0', 'pov', 'full'), 'cap_half_pool')}")
    # 外推 vs 网格(排雷 open10 全期)
    r = one(SC, strategy="COMP_v5_exclude", mode="open10", variant="Y1.0", sizing="pov", window="2021-2025")
    N["extrap_open10"] = (f"10亿 外推 {r['pred_net_arith_1e+09']:+.2f} / 网格 {r['grid_net_arith_1e+09']:+.2f};"
                          f"30亿 {r['pred_net_arith_3e+09']:+.2f} / {r['grid_net_arith_3e+09']:+.2f};"
                          f"100亿 {r['pred_net_arith_1e+10']:+.2f} / {r['grid_net_arith_1e+10']:+.2f}pp")
    N["extrap_turn_open10"] = f"{r.turnover_ann_1e6:.2f} → {r['turnover_ann_1e+10']:.2f}"
    r2 = one(SC, strategy="COMP_v5_exclude", mode="vwap1", variant="Y1.0", sizing="pov", window="2021-2025")
    N["extrap_vwap1"] = f"100亿 外推 {r2['pred_net_arith_1e+10']:+.2f} / 网格 {r2['grid_net_arith_1e+10']:+.2f}pp"
    b = one(SC, strategy="B_gbt_top20", mode="open10", variant="Y1.0", sizing="pov", window="2024-2025")
    c = one(SC, strategy="COMP_v5_top20", mode="open10", variant="Y1.0", sizing="pov", window="2024-2025")
    N["bg_vs_t20_impact"] = (f"{b.impact_per_traded_bp_1e6:.2f} vs {c.impact_per_traded_bp_1e6:.2f}bp,比值 "
                             f"{b.impact_vs_comp_top20:.2f}(容量倍数 {b.impact_vs_comp_top20**2:.1f}×);每元冲击因子 "
                             f"{b.impact_penalty_vs_exclude:.1f}× vs {c.impact_penalty_vs_exclude:.1f}×;年单边换手 "
                             f"{b.turnover_ann_1e6:.1f} vs {c.turnover_ann_1e6:.1f}")
    # 截量口径:part_bound 价值口径 vs 股票数
    for s, m, a in (("COMP_v5_exclude", "open10", 1e10), ("COMP_v5_exclude", "open10", 1e9), ("COMP_v5_top20", "open10", 1e8)):
        r = one(G, strategy=s, mode=m, variant="Y1.0", sizing="pov", window_role="full", aum=a)
        N[f"pb_{s}_{m}_{a:.0e}"] = f"{r.part_bound_frac_rebal*100:.1f}%"
        N[f"npb_{s}_{m}_{a:.0e}"] = f"{r.n_part_bound_rebal:.0f}"
        N[f"hold_{s}_{m}_{a:.0e}"] = f"{r.holdings:.0f}"
    # 安慰剂
    pl = q(CK, check="placebo_random_top20_pov_minus_exante")
    for _, r in pl.iterrows():
        a = float(r["item"].split()[1])
        npos = re.search(r">0 的种子 (\d+/\d+)", r.note).group(1)
        N[f"pl_{r['mode']}_{a:.0e}"] = f"{r.value*100:+.2f}pp(NW t {_note_num(r.note, 'NW t=')},{npos} 个种子为正)"
        N[f"plc_{r['mode']}_{a:.0e}"] = _note_num(r.note, "现金按池收益补仓后差 均值") + "pp"
        N[f"plexp_{r['mode']}_{a:.0e}"] = f"{_note_num(r.note, '仓位 pov')} vs {_note_num(r.note, '/ exante')}"
    # 截量口径:排雷在现金补仓后的 pov / exante 缺口(说明缺口不是仓位效应)
    for role in ("full", "test"):
        for col, tag in (("excess_pool", "c0"), ("excess_pool_cashpool", "cp"), ("gross_excess_pool_arith", "gr"),
                         ("cash_mean", "cash"), ("cost_total_bp", "cost")):
            for sz in ("pov", "exante"):
                v_ = g("COMP_v5_exclude", "open10", "Y1.0", sz, role, 1e10, col)
                N[f"ex1e10_{role}_{tag}_{sz}"] = (f"{v_*100:.1f}%" if tag == "cash" else f"{v_/100:.2f}" if tag == "cost"
                                                  else pp(v_))
    # legacy 盈亏平衡在 exante 下缩小的幅度(Top-N,全期,两口径)
    rat = []
    for s in TOPN:
        for m in ("open10", "vwap1"):
            a_, b_ = s_(s, m, "legacy", "pov", "full").cap_be_pool, s_(s, m, "legacy", "exante", "full").cap_be_pool
            if np.isfinite(a_) and np.isfinite(b_):
                rat.append(1 - b_ / a_)
    N["legacy_shrink"] = f"{min(rat)*100:.0f}%~{max(rat)*100:.0f}%"
    # Top-N 在 1000 万 ~ 3 亿之间 pov 与 exante 的最大差(全期,legacy + 三档 Y)
    f_ = G[(G.window_role == "full") & (G.strategy != "COMP_v5_exclude") & G.variant.isin(["legacy", "Y0.5", "Y1.0", "Y1.5"])
           & (G.aum >= 1e7) & (G.aum <= 3e8)]
    kk = ["strategy", "mode", "variant", "aum"]
    gap = (f_[f_.sizing == "pov"].set_index(kk).excess_pool - f_[f_.sizing == "exante"].set_index(kk).excess_pool).abs()
    i_ = gap.idxmax()
    N["topn_gap_max"] = f"{gap.max()*100:.1f}pp({NAME[i_[0]]} {i_[1]} {i_[2]} {money(i_[3])})"
    N["topn_gap_median"] = f"{gap.median()*100:.1f}pp"
    # 几何 vs 算术:盈亏平衡之比(POV,全部有限值)与排雷的差
    s_pov = S[(S.sizing == "pov") & S.variant.isin(["legacy", "Y0.5", "Y1.0", "Y1.5"])]
    ratio = (s_pov.cap_be_arith / s_pov.cap_be_pool).dropna()
    N["be_arith_ratio"] = f"{ratio.min():.2f}~{ratio.max():.2f}"
    e_ = G[(G.sizing == "pov") & (G.strategy == "COMP_v5_exclude") & (G.variant == "Y1.0") & (G.window_role == "full")]
    dga = (e_.excess_pool - e_.excess_pool_arith) * 100
    d7 = dga[e_.aum == 1e7]
    N["ex_geo_arith_gap"] = f"{dga.min():.1f}~{dga.max():.1f}pp(1000 万格 {d7.min():.2f}~{d7.max():.2f}pp)"
    # 随机组合本身相对池子的超额(安慰剂 1e6,pov)
    PL = d["PL"]
    pr = PL[(PL.aum == 1e6) & (PL.sizing == "pov")].groupby("mode").excess_pool.mean() * 100
    N["placebo_base"] = f"{pr.max():+.1f}~{pr.min():+.1f}pp"
    # 元超额峰值(算术 × X;网格点)
    for s in TOPN:
        for m in ("open10", "vwap1"):
            for v in ("legacy", "Y0.5", "Y1.0"):
                for sz in ("pov", "exante"):
                    r = s_(s, m, v, sz, "full")
                    for tag in ("cash0", "cashpool"):
                        y, x = r[f"yuan_alpha_max_{tag}"], r[f"aum_at_yuan_alpha_max_{tag}"]
                        N[f"ya_{s}_{m}_{v}_{sz}_{tag}"] = f"{y/1e4:.0f} 万元/年(规模 {money(x)} 处)"
    # 机制
    for _, r in q(CK, check="exante_mechanism_rankcorr").iterrows():
        N[f"mech_{r['mode']}"] = f"{r.value:.4f}(NW t {_note_num(r.note, 'NW t(带宽 5)=')})"
    for _, r in q(CK, check="exante_fallback_share").iterrows():
        N[f"fb_{r['mode']}"] = f"{r.value:.1e}"
    for _, r in q(CK, check="exante_realized_participation").iterrows():
        key = r["item"].split()[0] + "_" + r["mode"]
        N[f"rp_{key}"] = f"{r.value*100:.0f}%"
    # 冷启动
    for _, r in q(CK, check="bgbt_coldstart_shortfall").iterrows():
        v = r["item"].split()[2]
        N[f"cs_{r['mode']}_{v}"] = f"{r.value*100:+.2f}pp"
        N[f"csp_{r['mode']}_{v}"] = f"{r.reference*100:+.2f}pp"
    for _, r in q(CK, check="bgbt_coldstart_proxy").iterrows():
        parts = r["item"].split()
        N[f"cpx_{r['mode']}_{parts[3]}_{parts[5].split(':')[0]}"] = f"{r.value*100:+.2f}pp"
    return N


# ====================================================================== 第二轮修订(fixa,2026-10-10)新增的叙述性数字
def _ord(a, flag) -> float:
    """容量排序用:右删失记 +inf、左删失记 0。"""
    if np.isfinite(a):
        return float(a)
    return np.inf if flag == "gt_max" else 0.0 if flag == "lt_min" else np.nan


def _rng_money(rows, col) -> str:
    vals = [(_ord(r[col], r[f"{col}_flag"]), r[col], r[f"{col}_flag"]) for r in rows]
    vals = [v for v in vals if np.isfinite(v[0]) or v[0] == np.inf]
    lo, hi = min(vals, key=lambda x: x[0]), max(vals, key=lambda x: x[0])
    return f"{money(lo[1], lo[2])} ~ {money(hi[1], hi[2])}"


def _rng(vals, fmt="{:.2f}", unit="") -> str:
    v = [x for x in vals if np.isfinite(x)]
    return f"{fmt.format(min(v))}~{fmt.format(max(v))}{unit}"


def _corr_note(note: str) -> dict:
    return {int(k): float(v) for k, v in re.findall(r"k=([+\-]\d+):([+\-]?[0-9.]+)", note)}


def numbers_fixa(d) -> dict:
    G, S, SC, CK, D = d["G"], d["S"], d["SC"], d["CK"], d["D"]
    N = {}

    def gg(s, m, v, sz, role, a):
        return one(G, strategy=s, mode=m, variant=v, sizing=sz, window_role=role, aum=a)

    def ss(s, m, v, sz, role):
        return one(S, strategy=s, mode=m, variant=v, sizing=sz, window_role=role)

    # 基准年化(2021-2025 与 B×GBT 2023-2025)
    for m in ("open10", "vwap1"):
        r = gg("COMP_v5_exclude", m, "Y1.0", "pov", "full", 1e7)
        N[f"bpool_{m}"] = f"{r.bench_pool_ann*100:.2f}%"
        N[f"bpool_d1_{m}"] = f"{r.bench_pool_d1_ann*100:.2f}%"
        N[f"bcsi_{m}"] = f"{r.bench_csi1000p_ann*100:.2f}%"
        N[f"bcsi_legacy_{m}"] = f"{r.bench_csi1000p_legacy_ann*100:.2f}%"
        N[f"pool_minus_csi_{m}"] = f"{(r.bench_pool_ann - r.bench_csi1000p_ann)*100:.2f}"
        N[f"pool_r5_minus_d1_{m}"] = f"{(r.bench_pool_ann - r.bench_pool_d1_ann)*100:+.2f}"
        N[f"csi_exec_minus_legacy_{m}"] = f"{(r.bench_csi1000p_ann - r.bench_csi1000p_legacy_ann)*100:+.2f}"
        N[f"pool_ann_comp_{m}"] = f"{r.bench_pool_ann*100:.1f}"
        N[f"pool_ann_bg_{m}"] = f"{gg('B_gbt_top20', m, 'Y1.0', 'pov', 'full', 1e7).bench_pool_ann*100:.1f}"
    # 排雷:外推、盈亏平衡、t、半衰
    for m in ("open10", "vwap1"):
        for w in ("2021-2025", "2024-2025"):
            r = one(SC, strategy="COMP_v5_exclude", mode=m, variant="Y1.0", sizing="pov", window=w)
            N[f"xbe_ex_{m}_{w}"] = money(r.x_be_sqrt_extrap)
    ex = S[(S.strategy == "COMP_v5_exclude") & S.variant.isin(["Y0.5", "Y1.0"])]
    fin = ex[np.isfinite(ex.cap_be_pool)]
    N["exbe_y01"] = ("全部 > 100 亿" if len(fin) == 0 else
                     "; ".join(f"{r_['mode']} {r_['variant']} {r_['sizing']} {r_['window']} {money(r_['cap_be_pool'])}"
                               for _, r_ in fin.iterrows()))
    exp_ = S[(S.strategy == "COMP_v5_exclude") & S.variant.isin(["Y0.5", "Y1.0", "Y1.5"]) & (S.sizing == "pov")]
    N["ex_t_rng"] = _rng(exp_.t_nw_1e7)
    hl = S[(S.strategy == "COMP_v5_exclude") & S.variant.isin(["Y0.5", "Y1.0", "Y1.5"])]
    i_ = hl.cap_half_pool.idxmin()
    N["exhalf_min"] = (f"{money(hl.loc[i_, 'cap_half_pool'])}({hl.loc[i_, 'mode']} {hl.loc[i_, 'variant']} "
                       f"{hl.loc[i_, 'window']} {SZ[hl.loc[i_, 'sizing']]})")
    # Top-N(Y=1 全期)盈亏平衡描述与盈亏平衡处现金
    for sz in ("pov", "exante"):
        rr = [ss(s, m, "Y1.0", sz, "full") for s in TOPN for m in ("open10", "vwap1")]
        fins = [r_.cap_be_pool for r_ in rr if np.isfinite(r_.cap_be_pool)]
        N[f"topn_be_desc_{sz}"] = "全部 < 100 万" if not fins else f"最高 {money(max(fins))}"
        N[f"topn_be_cash_max_{sz}"] = f"{max(r_.cash_at_cap_be_pool for r_ in rr)*100:.1f}%"
    # Y=0.5 Top-N 全期盈亏平衡最大值(全部 / 去掉 B×GBT open10)
    for sz in ("pov", "exante"):
        rr = [(ss(s, m, "Y0.5", sz, "full"), s, m) for s in TOPN for m in ("open10", "vwap1")]
        best = max(rr, key=lambda x: _ord(x[0].cap_be_pool, x[0].cap_be_pool_flag))
        N[f"topn_y05_be_max_{sz}"] = f"{cap(best[0], 'cap_be_pool')}({NAME[best[1]]} {best[2]})"
        rx = [x for x in rr if not (x[1] == "B_gbt_top20" and x[2] == "open10")]
        bx = max(rx, key=lambda x: _ord(x[0].cap_be_pool, x[0].cap_be_pool_flag))
        N[f"topn_y05_be_max_exbg_{sz}"] = f"{cap(bx[0], 'cap_be_pool')}({NAME[bx[1]]} {bx[2]})"
        N[f"topn_y05_cash_max_exbg_{sz}"] = f"{max(x[0].cash_at_cap_be_pool for x in rx)*100:.1f}%"
    # 旧成本 Top-N 全期盈亏平衡的范围(POV)及盈亏平衡处现金、补仓后
    for sz in ("pov", "exante"):
        rr = [ss(s, m, "legacy", sz, "full") for s in TOPN for m in ("open10", "vwap1")]
        N[f"leg_be_rng_{sz}"] = _rng_money(rr, "cap_be_pool")
        N[f"leg_becp_rng_{sz}"] = _rng_money(rr, "cap_be_cashpool")
        N[f"leg_becc_rng_{sz}"] = _rng_money(rr, "cap_be_cashcsi")
        N[f"leg_cash_rng_{sz}"] = _rng([r_.cash_at_cap_be_pool * 100 for r_ in rr], "{:.0f}", "%")
    rat = [ss(s, "vwap1", "legacy", "pov", "full").cap_be_pool / ss(s, "open10", "legacy", "pov", "full").cap_be_pool
           for s in TOPN]
    N["leg_ratio_rng"] = _rng(rat, "{:.1f}", " 倍")
    # 现金拖累决定的格子:盈亏平衡处现金的范围
    cells = [ss(s, m, "legacy", "pov", "full") for s in TOPN for m in ("open10", "vwap1")]
    cells += [ss("B_gbt_top20", "open10", "Y0.5", "pov", "full"), ss("B_gbt_top20", "open10", "Y0.5", "pov", "dev"),
              ss("B_gbt_top20", "open10", "Y1.0", "pov", "dev"), ss("COMP_v5_top50", "open10", "Y0.5", "pov", "test")]
    N["cashdrag_rng"] = _rng([r_.cash_at_cap_be_pool * 100 for r_ in cells], "{:.0f}", "%")
    # B×GBT 每元成交
    b = one(SC, strategy="B_gbt_top20", mode="open10", variant="Y1.0", sizing="pov", window="2023-2025")
    N.update(bg_atr=f"{b.alpha_per_traded_bp:.1f}", bg_clin=f"{b.lin_cost_per_traded_bp:.1f}",
             bg_spr=f"{b.spread_per_traded_bp:.1f}", bg_px=f"{0.005 / (b.spread_per_traded_bp * 1e-4):.1f}",
             bg_margin=f"{b.margin_bp:.1f}", bg_xbe=money(b.x_be_sqrt_extrap))
    bb = [one(SC, strategy="B_gbt_top20", mode=m, variant="Y1.0", sizing="pov", window="2023-2025").alpha_per_traded_bp
          for m in ("open10", "vwap1")]
    N["bg_atr_rng"] = _rng(bb, "{:.1f}", "bp")
    # 毛利之比(排雷 / Top-N,同口径全期,Y=1 POV)
    ratios, nonpos = [], []
    for s in TOPN:
        for m in ("open10", "vwap1"):
            w = "2023-2025" if s == "B_gbt_top20" else "2021-2025"
            r = one(SC, strategy=s, mode=m, variant="Y1.0", sizing="pov", window=w)
            e = one(SC, strategy="COMP_v5_exclude", mode=m, variant="Y1.0", sizing="pov", window="2021-2025")
            if r.margin_bp > 0:
                ratios.append(e.margin_bp / r.margin_bp)
            else:
                nonpos.append(f"{NAME[s]} {m} {r.margin_bp:.1f}bp")
    N["margin_ratio_rng"] = f"1/{min(ratios):.0f} ~ 1/{max(ratios):.0f}"
    N["margin_nonpos"] = "、".join(nonpos) if nonpos else "无"
    tab = SC[(SC.variant == "Y1.0") & (SC.sizing == "pov")]
    tab = tab[((tab.strategy == "B_gbt_top20") & tab.window.isin(["2023-2025", "2024-2025"]))
              | ((tab.strategy != "B_gbt_top20") & tab.window.isin(["2021-2025", "2024-2025"]))]
    tn, te_ = tab[tab.strategy != "COMP_v5_exclude"], tab[tab.strategy == "COMP_v5_exclude"]
    N["atr_topn_rng"] = _rng(tn.alpha_per_traded_bp, "{:.0f}")
    N["atr_ex_rng"] = _rng(te_.alpha_per_traded_bp, "{:.0f}")
    N["atr_ratio_rng"] = f"1/{te_.alpha_per_traded_bp.min()/tn.alpha_per_traded_bp.max():.0f} ~ 1/{te_.alpha_per_traded_bp.max()/tn.alpha_per_traded_bp.min():.0f}"
    pos = tn[tn.margin_bp > 0].margin_bp
    N["margin_topn_rng"] = _rng(pos, "{:.1f}")
    N["margin_t20_open10_full"] = f"{one(tn, strategy='COMP_v5_top20', mode='open10', window='2021-2025').margin_bp:.1f}"
    # §3 判据
    for m in ("open10", "vwap1"):
        for chk, tag in (("fullpool_resid_pp", "d1"), ("fullpool_resid_pp_samefreq", "sf"), ("fullpool_resid_pp_r5_vs_d1", "r5d1")):
            v = q(CK, check=chk)
            v = v[v["mode"] == m].value
            N[f"fp_{tag}_{m}"] = _rng(v, "{:.2f}", "pp")
        for item in ("cap_real_index_alignment", "csi1000_proxy_exec_time", "csi1000_proxy_same_exec",
                     "index_returns_on_signal_days"):
            r = one(CK, check="index_alignment_peak", mode=m, item=item)
            c = _corr_note(r.note)
            N[f"al_{m}_{item}_k0"] = f"{c[0]:.3f}"
            N[f"al_{m}_{item}_km1"] = f"{c[-1]:.3f}"
    sf = one(CK, check="grid_align_peak_k0_samefreq")
    N["grid_align_sf"] = f"{sf.value*100:.0f}%"
    # §5.1 Top50 检验窗 Y=0.5、B×GBT 2023 Y=1、B×GBT 的 t
    for m in ("open10", "vwap1"):
        for sz in ("pov", "exante"):
            r = ss("COMP_v5_top50", m, "Y0.5", sz, "test")
            N[f"t50t_{m}_{sz}_be"] = cap(r, "cap_be_pool")
            N[f"t50t_{m}_{sz}_becp"] = cap(r, "cap_be_cashpool")
            N[f"t50t_{m}_{sz}_cash"] = f"{r.cash_at_cap_be_pool*100:.1f}%"
        r = ss("COMP_v5_top50", m, "Y0.5", "pov", "test")
        N[f"t50t_{m}_ex1e7"] = pp(r.excess_pool_1e7)
        N[f"t50t_{m}_t"] = f"{r.t_nw_1e7:.2f}"
    r = ss("B_gbt_top20", "open10", "Y1.0", "pov", "dev")
    N["bg_dev_cash_y1"] = f"{r.cash_at_cap_be_pool*100:.0f}%"
    bgs = S[(S.strategy == "B_gbt_top20") & S.variant.isin(["Y0.5", "Y1.0", "Y1.5"]) & (S.sizing == "pov")]
    N["bg_t_rng"] = _rng(bgs.t_nw_1e7, "{:+.2f}")
    # §5.4 B×GBT open10 Y=0.5 全期:盈亏平衡处现金与补仓曲线
    r = ss("B_gbt_top20", "open10", "Y0.5", "pov", "full")
    N["bg_y05_cash"] = f"{r.cash_at_cap_be_pool*100:.1f}%"
    gcp = q(G, strategy="B_gbt_top20", mode="open10", variant="Y0.5", sizing="pov", window_role="full").set_index("aum")
    N["bg_y05_cp_1e6"], N["bg_y05_cp_1e7"], N["bg_y05_cp_3e7"] = (pp(gcp.loc[a_, "excess_pool_cashpool"]) for a_ in (1e6, 1e7, 3e7))
    N["bg_y05_cp_min_ge3e7"] = pp(gcp.loc[gcp.index >= 3e7, "excess_pool_cashpool"].min())
    # §6 分解
    for m in ("open10", "vwap1"):
        r = one(D, strategy="COMP_v5_exclude", mode=m, variant="Y1.0", sizing="pov", window_role="full", aum=1e10)
        N[f"dec_ex_{m}_imp"] = f"{r.d_impact_pp:.2f}"
        N[f"dec_ex_{m}_gross"] = f"{-r.d_gross_pp:.2f}"
        N[f"dec_ex_{m}_cash"] = f"{r.d_cash_drag_pp:.2f}"
    dd = D[(D.strategy != "COMP_v5_exclude") & (D["mode"] == "open10") & (D.variant == "Y1.0") & (D.sizing == "pov")
           & (D.window_role == "full")]
    N["dec_topn_cash_1e8"] = _rng(dd[dd.aum == 1e8].d_cash_drag_pp, "{:.1f}", "pp")
    N["dec_topn_cash_ge1e9"] = _rng(dd[dd.aum >= 1e9].d_cash_drag_pp, "{:.1f}", "pp")
    N["dec_t50_vwap1_1e8_imp"] = f"{one(D, strategy='COMP_v5_top50', mode='vwap1', variant='Y1.0', sizing='pov', window_role='full', aum=1e8).d_impact_pp:.2f}"
    # §7.3 / §7.4
    for m in ("open10", "vwap1"):
        N[f"ex_cost_1e10_{m}"] = f"{gg('COMP_v5_exclude', m, 'Y1.0', 'pov', 'full', 1e10).cost_total_bp/100:.2f}"
    rows = [gg(s, m, "Y1.0", "pov", "full", 1e7) for s in STR for m in ("open10", "vwap1")]
    N["csi_gap_rng"] = _rng([(r_.excess_csi1000p - r_.excess_pool) * 100 for r_ in rows], "{:.1f}", "pp")
    r = gg("COMP_v5_exclude", "open10", "Y1.0", "pov", "full", 1e7)
    N["ex_vs_csi_open10"] = f"{pp(r.excess_csi1000p)}pp(open10,NW t {r.t_nw_csi1000p:.2f})"
    N["ex_vs_real_open10"] = pp(r.excess_csi1000r)
    N["csi_dev_ann_open10"] = f"{gg('COMP_v5_exclude', 'open10', 'Y1.0', 'pov', 'dev', 1e7).bench_csi1000p_ann*100:+.2f}%"
    N["csi_2023_ann_open10"] = f"{gg('B_gbt_top20', 'open10', 'Y1.0', 'pov', 'dev', 1e7).bench_csi1000p_ann*100:+.2f}%"
    rr = [gg(s, m, "Y1.0", "pov", "full", 1e7) for s in ("COMP_v5_top50", "B_gbt_top20") for m in ("open10", "vwap1")]
    N["idx_topn_csi_rng"] = _rng([r_.excess_csi1000p * 100 for r_ in rr], "{:+.2f}", "pp")
    N["idx_topn_csi_tmax"] = f"{max(r_.t_nw_csi1000p for r_ in rr):.2f}"
    N["idx_topn_csi_be_open10_rng"] = _rng_money([ss(s, "open10", "Y1.0", "pov", "full")
                                                  for s in ("COMP_v5_top50", "B_gbt_top20")], "cap_be_csi1000p")
    # §7.5
    rr = [gg("B_gbt_top20", m, "legacy", "pov", "full", 1e7) for m in ("open10", "vwap1")]
    N["bg_legacy_rng"] = _rng([r_.excess_pool * 100 for r_ in rr], "{:+.1f}", "pp")
    N["bg_legacy_t_rng"] = _rng([r_.t_nw for r_ in rr], "{:.2f}")
    tt = G[(G.strategy == "B_gbt_top20") & (G["mode"] == "open10") & G.variant.isin(["Y0.5", "Y1.0"]) & (G.sizing == "pov")
           & (G.window_role == "test") & (G.aum >= 1e7)]
    N["bg_test_t_rng"] = _rng(tt.t_nw, "{:+.1f}")
    N["bg_test_all_neg"] = "都为负" if bool(((tt.excess_pool < 0) & (tt.excess_pool_arith < 0)).all()) else "并不都为负"
    # §7.6
    N["ex_2tick_open10_1e7"] = pp(gg("COMP_v5_exclude", "open10", "Y1.0_2tick", "pov", "full", 1e7).excess_pool)
    drops = [(gg(s, m, "Y1.0", "pov", "full", 1e7).excess_pool - gg(s, m, "Y1.0_2tick", "pov", "full", 1e7).excess_pool) * 100
             for s in TOPN for m in ("open10", "vwap1")]
    N["topn_2tick_drop_rng"] = _rng(drops, "{:.1f}", "pp")
    tr = [ss(s, m, "Y1.0", "pov", "full") for s in TOPN for m in ("open10", "vwap1")]
    fin = [(NAME[r_.strategy], r_["mode"], r_.cap_be_pool_tranche5) for r_ in tr if np.isfinite(r_.cap_be_pool_tranche5)]
    fx = pd.read_csv(LEDGER / "fixa_capacity_before_after.csv")
    for m in ("open10", "vwap1"):
        N[f"dec_cashdrag_change_{m}"] = fx[fx.metric == f"损失分解 d_cash_drag_pp(现金 × 池基准收益){m}"].after.iloc[0]
    N["tranche5_desc"] = ("Top-N 全部 < 500 万" if not fin else
                          "、".join(f"{a_} {b_} {money(c_)}" for a_, b_, c_ in fin) + ";其余 Top-N < 500 万")
    return N


def numbers_r2(d) -> dict:
    """第二轮修订的复核审查(同日)新增的叙述性数字:vwap1 / open10 排雷 100 亿格的不确定性(capacity_grid / summary)、
    再平衡频率差的机制(research/ledger/fixa_r2_mech.csv)、BARRA_ATTRIBUTION 换同频基准后的变化(fixa_r2_barra_delta.csv)。"""
    G, S = d["G"], d["S"]
    N = {}
    for m in ("open10", "vwap1"):
        for sz in ("pov", "exante"):
            for role in ("full", "dev", "test"):
                r = one(G, strategy="COMP_v5_exclude", mode=m, variant="Y1.0", sizing=sz, window_role=role, aum=1e10)
                N[f"r2_ex_{m}_{sz}_{role}_1e10"] = pp(r.excess_pool)
                N[f"r2_t_{m}_{sz}_{role}_1e10"] = f"{r.t_nw:.2f}"
                s = one(S, strategy="COMP_v5_exclude", mode=m, variant="Y1.0", sizing=sz, window_role=role)
                N[f"r2_be_lo_{m}_{sz}_{role}"] = cap(s, "cap_be_arith_lo")
                N[f"r2_be_hi_{m}_{sz}_{role}"] = cap(s, "cap_be_arith_hi")
    mech = pd.read_csv(LEDGER / "fixa_r2_mech.csv")
    pm = mech[(mech.panel == "Panel")]
    for m in ("open10", "vwap1", "tail1450"):
        r = one(pm, mode=m, window="2021-2025")
        N[f"r2_xs1_{m}"] = f"{r.xs_corr_l1:.3f}"
        N[f"r2_pac1_{m}"] = f"{r.pool_ac1_d1:.3f}"
        N[f"r2_pred_{m}"] = f"{r.pred_gap_arith*100:.2f}"
        N[f"r2_act_{m}"] = f"{r.arith_r5_minus_d1*100:.2f}"
    N["r2_pred_ratio_rng"] = _rng(pm.pred_over_actual * 100, "{:.0f}", "%")
    r = one(pm, mode="open10", window="2021-2023")
    N["r2_pac1_open10_dev"] = f"{r.pool_ac1_d1:.3f}"
    N["r2_gap_open10_dev"] = f"{r.d_r5_minus_d1_geo*100:+.2f}"
    for pn in ("Panel", "TailPanel"):
        for w in ("2020-2025", "2021-2025", "2024-2025"):
            r = one(mech, panel=pn, mode="tail1450", window=w)
            N[f"r2_tail_{pn}_{w}"] = f"{r.d_r5_minus_d1_geo*100:+.2f}"
    bd = pd.read_csv(LEDGER / "fixa_r2_barra_delta.csv")

    def bt(mode, term):
        return one(bd, mode=mode, term=term)
    for mode in ("open10", "tail1450"):
        for term, key in (("__总超额", "tot"), ("__alpha", "alpha"), ("Liquidity 流动性", "liq")):
            r = bt(mode, term)
            N[f"r2_barra_{mode}_{key}_old"] = f"{r.old_pp:+.2f}"
            N[f"r2_barra_{mode}_{key}_new"] = f"{r.new_pp:+.2f}"
        r = bt(mode, "流动性占总超额(old → new)")
        N[f"r2_barra_{mode}_share_old"] = f"{r.old_pp:.0f}%"
        N[f"r2_barra_{mode}_share_new"] = f"{r.new_pp:.0f}%"
    N["r2_barra_open10_t_old_rng"] = _rng([bt("open10", f"复算 bench=d1:alpha t(NW q{q})").new_pp for q in (0, 5, 10, 20)], "{:.2f}")
    tt = [bt("open10", f"复算 bench=same:alpha t(NW q{q})").new_pp for q in (0, 5, 10, 20)]
    N["r2_barra_open10_t_new_rng"] = _rng(tt, "{:.2f}")
    return N


def t_fixa() -> str:
    """第二轮修订(fixa)前后对照表:读 research/ledger/fixa_capacity_before_after.csv(scripts/fixa_benchmark_fix.py capdiff)。"""
    f = LEDGER / "fixa_capacity_before_after.csv"
    df = pd.read_csv(f)
    out = ["| 组 | 项 | 修订前(k=1 池等权 / 收盘权重代理) | 修订后(同频池等权 / 成交时点代理) | 出处 |", "|---|---|---|---|---|"]
    for _, r in df.iterrows():
        out.append(f"| {r.group} | {r.metric} | {r.before} | {r.after} | `{r.source}` |")
    return "\n".join(out)


TEMPLATE_FILE = os.path.join(os.path.dirname(__file__), "cap_report_template.txt")


def main():
    d = load()
    fills = dict(T_A=t_headline(d["G"], d["S"]), T_B=t_curves(d["G"]), T_C=t_capacity(d["S"]),
                 T_D=t_extras(d["S"]), T_E=t_csi(d["G"], d["S"]), T_F=t_costs(d["G"]), T_H=t_decomp(d["D"]),
                 T_G=t_scaling(d["SC"]), T_X=t_extrap(d["SC"]), T_S=t_sizing(d["G"], d["S"]), T_K=t_cash(d["G"], d["S"]),
                 T_P=t_placebo(d["CK"]), T_M=t_mech(d["CK"]), T_CS=t_cold(d["CK"]), T_Y=t_yuan(d["S"]),
                 T_FIXA=t_fixa())
    fills.update(diag_fill(d["DG"]))
    N = numbers(d)
    N.update(numbers_fixa(d))                    # 第二轮修订(fixa)新增的叙述性数字
    N.update(numbers_r2(d))                      # 第二轮修订的复核审查(fixa review2)新增的叙述性数字
    txt = open(TEMPLATE_FILE, encoding="utf-8").read()
    for k, v in fills.items():
        txt = txt.replace("{{" + k + "}}", v)
    for k, v in N.items():
        txt = txt.replace("{{N:" + k + "}}", v)
    left = [s for s in txt.split("{{")[1:]]
    if left:
        raise RuntimeError(f"模板有未填的占位符: {[x.split('}}')[0] for x in left]}")
    out = ROOT / "research" / "CAPACITY_REPORT.md"
    with open(out, "w", encoding="utf-8") as f:
        f.write(txt)
    print(f"已写 {out}({len(txt)} 字符)")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "numbers":
        d_ = load()
        for k, v in {**numbers(d_), **numbers_fixa(d_), **numbers_r2(d_)}.items():
            print(f"{k:45s} {v}")
    else:
        main()
