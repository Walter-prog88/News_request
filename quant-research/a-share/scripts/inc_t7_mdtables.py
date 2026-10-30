"""INC-T7:从 research/ledger/inc_t7_*.csv 生成报告里的 markdown 表(只读账本,不算任何新数字)。

用法:PYTHONPATH=src python3 scripts/inc_t7_mdtables.py <输出目录>
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
L = ROOT / "research" / "ledger"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
OUT.mkdir(parents=True, exist_ok=True)


def pp(x, d=2, sign=True):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "–"
    return f"{x * 100:+.{d}f}" if sign else f"{x * 100:.{d}f}"


def md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(out)


def order_key(cell: str):
    if cell.startswith("E1_"):
        return (2, int(cell.split("_")[1]), 0)
    if cell.startswith("E2b_"):
        return (3, int(cell.split("_")[1]), 0)
    k, band = cell.split("|")
    e, x = [int(v) for v in band.split("/")]
    return (0, e * 1000 + x, int(k[1:]))


def table_main(panel: str) -> str:
    gm = pd.read_csv(L / f"inc_t7_grid_{panel}.csv")
    ev = pd.read_csv(L / "inc_t7_event.csv")
    ev = ev[ev.panel == panel]
    g = pd.concat([gm[gm.type == "grid"], ev], ignore_index=True)
    g = g[(g.cost == "Y1") & (g.window == "full")].set_index("cell")
    P = pd.read_csv(L / "inc_t7_paired.csv")
    P = P[P.panel == panel]
    rows = []
    cells = sorted(g.index, key=order_key)
    for c in cells:
        r = g.loc[c]
        row = {"格子": c.replace("|", " ") if "|" in c else c, "净超额": pp(r.excess_geo), "毛超额": pp(r.gross_excess),
               "年成本": pp(r.cost_total_ann, 2, False), "年单边换手": f"{r.turnover_ann:.2f}"}
        if c == "k5|20/35":
            row.update({"Δ全期 ±1se": "(现行)", "NW t": "–", "Δdev": "–", "Δtest": "–", "Δold": "–", "Δpess": "–"})
        else:
            pf = P[(P.cell == c) & (P.cost == "Y1") & (P.window == "full")].iloc[0]
            pdv = P[(P.cell == c) & (P.cost == "Y1") & (P.window == "dev")].iloc[0]
            pts = P[(P.cell == c) & (P.cost == "Y1") & (P.window == "test")].iloc[0]
            po = P[(P.cell == c) & (P.cost == "old") & (P.window == "full")].iloc[0]
            pe = P[(P.cell == c) & (P.cost == "pess") & (P.window == "full")].iloc[0]
            row.update({"Δ全期 ±1se": f"{pf.d_geo * 100:+.2f} ± {pf.se_nw20 * 100:.2f}", "NW t": f"{pf.t_nw20:.2f}",
                        "Δdev": pp(pdv.d_geo), "Δtest": pp(pts.d_geo), "Δold": pp(po.d_geo), "Δpess": pp(pe.d_geo)})
        rows.append(row)
    return md(pd.DataFrame(rows))


def table_crit() -> str:
    C = pd.read_csv(L / "inc_t7_criteria.csv")
    rows = []
    for _, r in C.sort_values("cell", key=lambda s: s.map(order_key)).iterrows():
        def tf(x):
            return "Y" if bool(x) else "n"
        rows.append({"格子": r.cell, "类别": r.primary if isinstance(r.primary, str) and r.primary else "探索",
                     "分钟 C1/C2/C3/C4/C6": "/".join(tf(r[f"minute_{c}"]) for c in ("C1", "C2", "C3", "C4", "C6")),
                     "长历史 C1/C2/C3/C4/C6": "/".join(tf(r[f"lh_{c}"]) for c in ("C1", "C2", "C3", "C4", "C6")),
                     "C5": tf(r.C5), "通过": tf(r.pass_all)})
    return md(pd.DataFrame(rows))


def table_costs(panel: str, cells) -> str:
    gm = pd.read_csv(L / f"inc_t7_grid_{panel}.csv")
    ev = pd.read_csv(L / "inc_t7_event.csv")
    g = pd.concat([gm[gm.type == "grid"], ev[ev.panel == panel]], ignore_index=True)
    g = g[(g.panel == panel) & (g.window == "full")]
    rows = []
    for c in cells:
        for cost in ("Y1", "old", "pess"):
            q = g[(g.cell == c) & (g.cost == cost)]
            if q.empty:
                continue
            r = q.iloc[0]
            rows.append({"格子": c, "成本口径": cost, "净超额": pp(r.excess_geo), "毛超额": pp(r.gross_excess),
                         "佣金": pp(r.cost_commission_ann, 2, False), "印花税": pp(r.cost_stamp_ann, 2, False),
                         "价差": pp(r.cost_spread_ann, 2, False), "冲击": pp(r.cost_impact_ann, 2, False),
                         "合计": pp(r.cost_total_ann, 2, False), "换手": f"{r.turnover_ann:.2f}",
                         "持仓": f"{r.holdings:.0f}", "仓位": f"{r.exposure:.3f}", "最大回撤": pp(r.max_dd, 1)})
    return md(pd.DataFrame(rows))


def table_cap(panel: str) -> str:
    S = pd.read_csv(L / "inc_t7_sensitivity.csv")
    S = S[(S.panel == panel) & S.test.astype(str).str.startswith("资金")]
    P = pd.read_csv(L / "inc_t7_paired.csv")
    rows = []
    cells = ["k5|20/35", "k1|20/35", "k1|20/30", "k1|20/45", "E2b_10", "E2b_20", "E1_3", "E1_6", "k5|30/45", "k10|20/35", "k20|20/35"]
    caps = [(1e7, "1e7"), (1e8, "1e8"), (1e9, "1e9")]
    for c in cells:
        row = {"格子": c}
        for cap, lab in caps:
            q = S[(S.cell == c) & (S.capital == cap) & (S.window == "full")]
            qd = S[(S.cell == c) & (S.capital == cap) & (S.window == "dev")]
            qt = S[(S.cell == c) & (S.capital == cap) & (S.window == "test")]
            if panel == "minute" and cap == 1e7:
                # 主网格即 1e7 恒定 AUM
                if c == "k5|20/35":
                    row[lab + " Δ(dev/test)"] = "(现行)"
                else:
                    pf = P[(P.panel == panel) & (P.cell == c) & (P.cost == "Y1")]
                    f, d, t = (pf[pf.window == w].iloc[0] for w in ("full", "dev", "test"))
                    row[lab + " Δ(dev/test)"] = f"{f.d_geo * 100:+.2f} ({d.d_geo * 100:+.2f}/{t.d_geo * 100:+.2f}), t {f.t_nw20:.1f}"
                continue
            if q.empty:
                row[lab + " Δ(dev/test)"] = "–"
                continue
            if c == "k5|20/35":
                row[lab + " Δ(dev/test)"] = f"(现行;年成本 {q.iloc[0].cost_ann * 100:.2f})"
            else:
                r = q.iloc[0]
                row[lab + " Δ(dev/test)"] = (f"{r.d_vs_cur * 100:+.2f} ({qd.iloc[0].d_vs_cur * 100:+.2f}/{qt.iloc[0].d_vs_cur * 100:+.2f}), "
                                             f"t {r.t20_vs_cur:.1f}")
        rows.append(row)
    return md(pd.DataFrame(rows))


# ---------------------------------------------------------------- 审查修复(2026-10-11)表
def _cell_pp(df, w):
    r = df[df.window == w]
    return "–" if r.empty else f"{r.iloc[0].d_geo * 100:+.2f}"


def table_fix_price() -> str:
    P = pd.read_csv(L / "inc_t7_fix_price.csv")
    P = P[(P.panel == "minute") & (P.cost == "Y1")]
    rows = []
    for sig, lab in (("S1_turn20", "S1=-turn20(长历史信号映射到分钟面板)"), ("COMP_v5", "COMP_v5"), ("pool_EW", "全池等权(零信号)")):
        for price, plab in (("twap", "开盘 10 分钟 TWAP"), ("auction", "开盘集合竞价单点价(窗口额 amt_open10)"),
                            ("auction_real", "开盘集合竞价单点价(真实集合竞价额)"), ("close", "收盘价")):
            q = P[(P.signal == sig) & (P.price == price)]
            f = q[q.window == "full"]
            if f.empty:
                continue
            rows.append({"信号": lab, "成交价": plab, "k1−k5 全期": _cell_pp(q, "full"), "NW t": f"{f.iloc[0].t_nw20:.2f}",
                         "开发 2021-23": _cell_pp(q, "dev"), "检验 2024-25": _cell_pp(q, "test")})
    L2 = pd.read_csv(L / "inc_t7_fix_price.csv")
    L2 = L2[L2.panel == "lh_engine"]
    for w in ("2021-2025", "2021-2023", "2024-2025", "2006-2019", "2020-2025"):
        r = L2[L2.window == w].iloc[0]
        rows.append({"信号": "S1(长历史引擎,q92 池)", "成交价": "日开盘价", "k1−k5 全期": f"{r.d_geo * 100:+.2f}",
                     "NW t": f"{r.t_nw20:.2f}", "开发 2021-23": w, "检验 2024-25": ""})
    return md(pd.DataFrame(rows))


def table_fix_lock() -> str:
    P = pd.read_csv(L / "inc_t7_fix_lock.csv")
    C = pd.read_csv(L / "inc_t7_fix_lock_cross.csv")
    P = P[P.cost == "Y1"]
    names = {"sym": "引擎(双向不可成交,标记用 te 日收盘信息)", "sym_ol": "双向不交易,开盘窗口可观察标记", "dir": "方向性(封涨停可卖不可买),被挡买单留现金",
             "dir2": "方向性,被挡名字冻结、其余重新归一", "dir_ol": "方向性,开盘窗口可观察标记", "dir2_ol": "方向性+重新归一,开盘窗口可观察标记"}
    rows = []
    for rule, lab in names.items():
        row = {"规则": lab}
        for cell in ("k1", "E2b_10", "E2b_20", "pool_EW_k1"):
            q = P[(P.rule == rule) & (P.cell == cell)]
            f = q[q.window == "full"]
            if f.empty:
                row[cell] = "–"
                continue
            row[cell] = f"{f.iloc[0].d_geo * 100:+.2f} (t {f.iloc[0].t_nw20:.1f}; {_cell_pp(q, 'dev')}/{_cell_pp(q, 'test')})"
        cx = C[(C.rule == rule) & (C.window == "full") & (C.test == "cell_under_rule_minus_k5_under_sym")]
        row["对引擎规则下 k5:k1"] = "–" if cx[cx.cell == "k1"].empty else f"{cx[cx.cell == 'k1'].iloc[0].d_geo * 100:+.2f}"
        row["对引擎规则下 k5:E2b_10"] = "–" if cx[cx.cell == "E2b_10"].empty else f"{cx[cx.cell == 'E2b_10'].iloc[0].d_geo * 100:+.2f}"
        rows.append(row)
    return md(pd.DataFrame(rows))


def table_fix_lots() -> str:
    S = pd.read_csv(L / "inc_t7_fix_lots_stats.csv")
    rows = [{"规模": f"{r.capital:.0e}", "格子": r.cell, "年订单数": f"{r.orders_per_year:,.0f}", "不足一手(笔数)": f"{r.sublot_share_orders:.1%}",
             "不足一手(金额)": f"{r.sublot_share_value:.1%}", "平均单笔(元)": f"{r.mean_order_yuan:,.0f}",
             "5 元最低佣金额外成本(占 AUM/年)": f"{r.floor5_extra_cost_pct_per_year:.2f}%"} for _, r in S.iterrows()]
    return md(pd.DataFrame(rows))


def table_fix_lots_sim() -> str:
    M = pd.read_csv(L / "inc_t7_fix_lots_sim.csv")
    rows = []
    for (cap, rule, lots, mc), q in M.groupby(["capital", "rule", "lots", "min_comm"], sort=False):
        row = {"规模": f"{cap:.0e}", "规则": rule, "取整到手": "是" if lots else "否", "最低佣金": f"{mc:.0f} 元" if mc else "无"}
        for cell in ("k1", "E2b_10", "E2b_20"):
            qq = q[q.cell == cell]
            f = qq[qq.window == "full"]
            row[cell] = f"{f.iloc[0].d_geo * 100:+.2f} (t {f.iloc[0].t_nw20:.1f}; {_cell_pp(qq, 'dev')}/{_cell_pp(qq, 'test')})"
        rows.append(row)
    return md(pd.DataFrame(rows))


if __name__ == "__main__":
    for panel in ("minute", "lh"):
        (OUT / f"t_main_{panel}.md").write_text(table_main(panel))
        (OUT / f"t_cost_{panel}.md").write_text(table_costs(panel, ["k5|20/35", "k1|20/35", "k10|20/35", "k20|20/35", "k5|30/45", "E1_6", "E2b_10"]))
    (OUT / "t_crit.md").write_text(table_crit())
    if (L / "inc_t7_fix_price.csv").exists():
        (OUT / "t_fix_price.md").write_text(table_fix_price())
        (OUT / "t_fix_lock.md").write_text(table_fix_lock())
        (OUT / "t_fix_lots.md").write_text(table_fix_lots())
        (OUT / "t_fix_lots_sim.md").write_text(table_fix_lots_sim())
    for panel in ("minute", "lh"):
        (OUT / f"t_cap_{panel}.md").write_text(table_cap(panel))
    print("ok", OUT)
