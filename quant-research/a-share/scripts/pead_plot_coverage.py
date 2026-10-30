"""PEAD 事件数据验收图(只画事件数与事件池规模,不含任何收益)。

输入:cache/_tmp/pead_events.parquet、research/ledger/pead_event_pool_size.csv
输出:reports/pead_event_coverage.png
  上图:主事件按 V 所在月份计数,按"新鲜度"分色堆叠
  下图:逐成交日 T 的合格事件池规模(过 ADV/上市天数/ST 池后),虚线 = 建仓下限 5 只
两幅图量纲不同,分成上下两张,不用双轴。
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from afactor.config import LEDGER, REPORTS
from afactor import earnings_events as EE

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
CATS = [("预告在先", "#2a78d6"), ("快报在先", "#eb6834"),
        ("在先未知(快报日期不早于正式财报)", "#1baf7a"), ("无在先记录", "#eda100")]


def main():
    ev = EE.load_events()
    m = ev[ev["is_main"]].copy()
    m["ym"] = m["V"].dt.to_period("M")
    tab = m.groupby(["ym", "freshness"]).size().unstack(fill_value=0)
    idx = pd.period_range("2020-01", "2025-12", freq="M")
    tab = tab.reindex(idx, fill_value=0)
    pool = pd.read_csv(LEDGER / "pead_event_pool_size.csv", parse_dates=["T"])

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(12, 7.2), facecolor=SURF,
                                 gridspec_kw=dict(height_ratios=[1.15, 1]))
    x = np.arange(len(idx))
    bottom = np.zeros(len(idx))
    for name, col in CATS:
        v = tab[name].to_numpy() if name in tab else np.zeros(len(idx))
        a1.bar(x, v, bottom=bottom, color=col, width=0.8, edgecolor=SURF, linewidth=1.0,
               label=name)
        bottom += v
    a1.set_xticks(x[::6])
    a1.set_xticklabels([str(p) for p in idx[::6]], color=INK2, fontsize=9)
    a1.set_ylabel("主事件数(按 V 所在月)", color=INK2, fontsize=10)
    a1.set_title("主事件(定期报告·营业利润单季同比·首次披露·信号可用·最新报告期)按新鲜度",
                 color=INK, fontsize=11, loc="left")
    a1.legend(frameon=False, fontsize=9, ncol=4, loc="upper left", labelcolor=INK2)

    a2.plot(pool["T"], pool["n_pool"], color="#2a78d6", linewidth=1.2)
    a2.axhline(5, color=INK2, linewidth=1.0, linestyle=(0, (3, 3)))
    a2.text(pool["T"].iloc[5], 6.2, "建仓下限 5 只", color=INK2, fontsize=9)
    a2.set_yscale("symlog", linthresh=10)
    a2.set_ylim(0, 6000)
    a2.set_ylabel("成交日 T 的合格事件池(只)", color=INK2, fontsize=10)
    a2.set_title("事件池规模:V ∈ [T-4, T] 且过 ADV20≥2000 万 / 上市满 60 日 / 非 ST 池(对数刻度)",
                 color=INK, fontsize=11, loc="left")
    for ax in (a1, a2):
        ax.set_facecolor(SURF)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.tick_params(colors=INK2, labelsize=9)
    fig.text(0.01, 0.005, "数据:tushare 利润表 + 东财业绩预告/快报,截至 2025-12-31;样本缺已退市股。"
             "来源:research/ledger/pead_event_pool_size.csv、cache/_tmp/pead_events.parquet",
             color=INK2, fontsize=8)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    out = REPORTS / "pead_event_coverage.png"
    fig.savefig(out, dpi=150, facecolor=SURF)
    print(out)


if __name__ == "__main__":
    main()
