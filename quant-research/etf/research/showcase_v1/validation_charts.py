"""Four matplotlib figures for the validation-line strategy report.

Style follows the user's reference charts: white ground, bold CJK title with
a gray methodology subtitle, steel-blue strategy lines, dashed gray bench,
filled drawdown panel, signed horizontal contribution bars with value labels,
and a gray source footnote.
"""

import json

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.rcParams.update({
    "font.sans-serif": ["PingFang SC", "Hiragino Sans GB",
                        "Arial Unicode MS"],
    "axes.unicode_minus": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.dpi": 160,
})

RES = "/path/to/user/Desktop/ETF-Research/index project/research/showcase_v1/results"
FIG = f"{RES}/figs"
BLUE, GOLD, GREEN, GRAY = "#2E6FA8", "#C08A2D", "#2E8A5F", "#555555"
NEG = "#D2822D"

LABELS = {"main": "Top4·等权", "eq10": "Top10·等权",
          "iv10": "Top10·逆波动", "bench": "中证500ETF 510500"}
COLORS = {"main": BLUE, "eq10": GREEN, "iv10": GOLD}


def fig_nav(navs, pack):
    fig, (ax, axd) = plt.subplots(
        2, 1, figsize=(12.8, 9.2), sharex=True,
        gridspec_kw={"height_ratios": [3, 1.15]}, constrained_layout=True)
    for k in ("main", "eq10"):
        st = pack["configs"][k]["stats"]
        ax.plot(navs.index, navs[k], color=COLORS[k], lw=2.0 if k == "main"
                else 1.5,
                label=f"{LABELS[k]}(CAGR {st['ann%']:.1f}%,"
                      f"Sharpe {st['sharpe']:.2f})")
    bst = pack["bench"]["stats"]
    ax.plot(navs.index, navs["bench"], color=GRAY, ls="--", lw=1.4,
            label=f"{LABELS['bench']}(CAGR {bst['ann%']:.1f}%)")
    ax.set_title("前向验证:Top4等权与Top10等权 vs 中证500ETF",
                 fontsize=17, fontweight="bold", loc="left", pad=30)
    ax.text(0, 1.012,
            f"验证区间 {pack['window'][0]}—{pack['window'][1]};模型与因子于"
            " 2024-01-02 前全部冻结,验证期纯前向执行;T+1开盘成交,单边成本"
            "0.5bp,单票上限30%,权重<3%剔除。",
            transform=ax.transAxes, fontsize=10.5, color="#666666")
    ax.set_ylabel("累计净值(起始资金=1)")
    ax.legend(loc="upper left", fontsize=11, frameon=False)
    ax.grid(axis="y", color="#E8E8E8", lw=0.7)
    for k in ("main", "eq10"):
        dd = navs[k] / navs[k].cummax() - 1
        if k == "main":
            axd.fill_between(navs.index, dd, 0, color=BLUE, alpha=0.18)
        axd.plot(navs.index, dd, color=COLORS[k],
                 lw=1.6 if k == "main" else 1.1)
    ddb = navs["bench"] / navs["bench"].cummax() - 1
    axd.plot(navs.index, ddb, color=GRAY, ls="--", lw=1.1)
    axd.axhline(-0.15, color="#D08030", ls=":", lw=1.2)
    mdds = {k: pack["configs"][k]["stats"]["mdd%"]
            for k in ("main", "eq10")}
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=COLORS["main"], lw=1.6),
               Line2D([], [], color=COLORS["eq10"], lw=1.1),
               Line2D([], [], color=GRAY, ls="--", lw=1.1),
               Line2D([], [], color="#D08030", ls=":", lw=1.2)]
    axd.legend(handles,
               [f"Top4等权最大回撤 {mdds['main']:.1f}%",
                f"Top10等权 {mdds['eq10']:.1f}%",
                f"510500 {pack['bench']['stats']['mdd%']:.1f}%",
                "15%回撤线"],
               loc="lower left", fontsize=9.5, frameon=False, ncol=4)
    axd.set_ylabel("回撤")
    axd.yaxis.set_major_formatter(
        matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    axd.grid(axis="y", color="#E8E8E8", lw=0.7)
    fig.text(0.01, -0.015,
             "来源:showcase_v1 冻结模型前向回放与 510500 日频数据;"
             "开盘-开盘计价,回测/模拟结果,不构成收益承诺。",
             fontsize=9, color="#888888")
    fig.savefig(f"{FIG}/fig_nav.png", bbox_inches="tight")
    plt.close(fig)


def fig_contrib(rows, title, sub, path, top_n=30):
    rows = [r for r in rows if abs(r["contrib_pp"]) > 0.005][:top_n]
    rows = sorted(rows, key=lambda r: r["contrib_pp"], reverse=True)
    n = len(rows)
    fig, ax = plt.subplots(figsize=(12.8, 0.34 * n + 2.2),
                           constrained_layout=True)
    ys = np.arange(n)[::-1]
    vals = [r["contrib_pp"] for r in rows]
    cols = [BLUE if v >= 0 else NEG for v in vals]
    ax.barh(ys, vals, color=cols, height=0.62)
    span = max(vals) - min(min(vals), 0)
    for y, r in zip(ys, rows):
        v = r["contrib_pp"]
        ax.text(v + (0.012 * span if v >= 0 else -0.012 * span), y,
                f"{v:+.2f}", va="center",
                ha="left" if v >= 0 else "right", fontsize=9,
                color="#444444")
    ax.set_yticks(ys)
    ax.set_yticklabels([r["label"] for r in rows], fontsize=10)
    ax.axvline(0, color="#333333", lw=0.9)
    ax.set_title(title, fontsize=16, fontweight="bold", loc="left",
                 pad=26)
    ax.text(0, 1.004, sub, transform=ax.transAxes, fontsize=10,
            color="#666666")
    ax.set_xlabel("对最终财富的贡献(初始资金百分点)")
    ax.grid(axis="x", color="#ECECEC", lw=0.7)
    fig.text(0.01, -0.02,
             "贡献为逐日持仓财富贡献的可加总结果,与期末净值精确对账;"
             "不是单只ETF自身买入持有收益率。",
             fontsize=9, color="#888888")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_holding(rows, pack, top_n=20):
    rows = [r for r in rows if r["contrib_pp"] > 0][:top_n]
    rows = sorted(rows, key=lambda r: r["held_days"], reverse=True)
    n = len(rows)
    fig, ax = plt.subplots(figsize=(12.8, 0.36 * n + 2.4),
                           constrained_layout=True)
    ys = np.arange(n)[::-1]
    days = [r["held_days"] for r in rows]
    ax.barh(ys, days, color=BLUE, height=0.6, alpha=0.85,
            label="累计持有交易日")
    for y, r in zip(ys, rows):
        ax.text(r["held_days"] + max(days) * 0.012, y,
                f"{r['held_days']}日 / {r['spells']}次 / 平均{r['avg_spell']}日",
                va="center", fontsize=9, color="#444444")
    ax.set_yticks(ys)
    ax.set_yticklabels([r["label"] for r in rows], fontsize=10)
    ax.set_title("Top4等权主要ETF的持有结构(验证期 2024-01 → 2026-05)",
                 fontsize=16, fontweight="bold", loc="left", pad=26)
    to = pack["configs"]["main"]
    ax.text(0, 1.005,
            f"标注 = 累计持有交易日 / 进出次数 / 平均单次持有;策略日均单边换手 "
            f"{to['one_way_daily_turnover']*100:.0f}%,全期交易成本合计 "
            f"{to['cost_total_pp']:.1f} 个百分点(单边0.5bp)。",
            transform=ax.transAxes, fontsize=10, color="#666666")
    ax.set_xlabel("累计持有交易日")
    ax.grid(axis="x", color="#ECECEC", lw=0.7)
    fig.savefig(f"{FIG}/fig_holding.png", bbox_inches="tight")
    plt.close(fig)


def main():
    import os
    os.makedirs(FIG, exist_ok=True)
    pack = json.load(open(f"{RES}/validation_report_pack.json"))
    navs = pd.read_parquet(f"{RES}/validation_report_navs.parquet")
    fig_nav(navs, pack)
    c = pack["configs"]
    fig_contrib(
        c["main"]["contrib"],
        "Top4·等权:各ETF累计财富贡献",
        f"验证区间 {pack['window'][0]}—{pack['window'][1]};共交易 "
        f"{c['main']['n_names_traded']} 只ETF;ETF贡献与交易成本合计对应期末净收益 "
        f"{(c['main']['stats']['ann%'] and (navs['main'].iloc[-1]-1)*100):.1f}%;"
        f"交易成本贡献 -{c['main']['cost_total_pp']:.1f} 个百分点。",
        f"{FIG}/fig_contrib_main.png")
    fig_contrib(
        c["eq10"]["contrib"],
        "Top10·等权:各ETF累计财富贡献",
        f"同一冻结模型、同一验证区间;持仓分散至10只等权;"
        f"期末净收益 {(navs['eq10'].iloc[-1]-1)*100:.1f}%;交易成本贡献 "
        f"-{c['eq10']['cost_total_pp']:.1f} 个百分点。",
        f"{FIG}/fig_contrib_eq10.png")
    fig_holding(c["main"]["contrib"], pack)
    print("figures written to", FIG)


if __name__ == "__main__":
    main()
