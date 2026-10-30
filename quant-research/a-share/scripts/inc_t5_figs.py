"""T5 图(前缀 inc_t5;配色沿用 dataviz 参考调色板的前三个类别槽:蓝 / 橙 / 青绿,第四条线用灰色虚线,不靠颜色单独区分)。

reports/inc_t5_rel_nav.png      相对净值(策略 / 同频池等权 B1):B-LH 2017-2025 与 B-MIN 2021-2025
reports/inc_t5_corr.png         篮子日超额相关矩阵(B-LH 全期、B-MIN 全期)
reports/inc_t5_ir_windows.png   IR 分窗口:排雷 vs 组合
reports/inc_t5_placebo.png      ΔIR(相对排雷):真实组合 vs 随机篮子安慰剂
"""
from __future__ import annotations

import sys

from inc_t5_common import *      # noqa: F401,F403
from inc_t5_common import ROOT, LEDGER, WIN, LE, log

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from inc_t5_report import TrackData, ir_of

matplotlib.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "Arial Unicode MS", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
SURF, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8985", "#e6e5e0"
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8985"
OUT = ROOT / "reports"


def style(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9, length=0)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def fig_nav():
    fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.6), facecolor=SURF)
    spec = [("S|LIQ|20_35", "排雷(LIQ)", BLUE, "-"), ("C|EW5|45_60", "5 篮子等权", ORANGE, "-"),
            ("C|CORE5|45_60", "核心-卫星", AQUA, "-"), ("C|S4|45_60", "不含排雷的 4 风格", GRAY, "--")]
    for ax, tr, ttl in ((axs[0], "lh", "B-LH:长历史面板,2017-2025(含退市股)"), (axs[1], "min", "B-MIN:分钟面板,2021-2025")):
        td = TrackData(tr)
        a, b = WIN[tr]["full"]
        for n, lab, col, ls in spec:
            r = LE.slice_eval(td.r(n), a, b)
            bb = LE.slice_eval(td.b1, a, b)
            rel = (1 + r).cumprod() / (1 + bb).cumprod()
            ax.plot(rel.index, rel.values, color=col, lw=1.8, ls=ls, label=lab)
            ax.text(rel.index[-1], rel.values[-1], f" {lab}", color=INK2, fontsize=8.5, va="center")
        ax.axhline(1.0, color=MUTED, lw=0.8)
        ax.set_title(ttl, color=INK, fontsize=11, loc="left")
        style(ax)
        ax.set_xlim(None, rel.index[-1] + pd.Timedelta(days=330))
        ax.set_ylabel("相对净值 = 策略 / 同频池等权(成本后)", color=INK2, fontsize=9)
    axs[0].legend(frameon=False, fontsize=8.5, loc="upper left", labelcolor=INK2)
    fig.suptitle("风格篮子叠加没有抬高相对净值:与排雷同向、跟踪误差不降", color=INK, fontsize=12.5, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / "inc_t5_rel_nav.png", dpi=150, facecolor=SURF)
    plt.close(fig)


def fig_corr():
    c = pd.read_csv(LEDGER / "inc_t5_corr.csv")
    order = ["LIQ", "LOWVOL", "VALUE", "DIV", "QUAL", "REV"]
    lab = {"LIQ": "排雷", "LOWVOL": "低波动", "VALUE": "价值", "DIV": "红利", "QUAL": "质量", "REV": "反转"}
    cmap = LinearSegmentedColormap.from_list("seq", ["#f4f8fd", "#2a78d6"])
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.9), facecolor=SURF)
    for ax, tr, ttl in ((axs[0], "lh", "B-LH 2017-2025"), (axs[1], "min", "B-MIN 2021-2025")):
        m = c[(c.track == tr) & (c.kind == "pearson_daily") & (c.window == "full")].pivot(index="a", columns="b", values="rho").loc[order, order]
        ax.imshow(m.values, cmap=cmap, vmin=-0.2, vmax=1.0)
        for i in range(len(order)):
            for j in range(len(order)):
                v = m.values[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9, color="#ffffff" if v > 0.6 else INK)
        ax.set_xticks(range(len(order)), [lab[o] for o in order], fontsize=9, color=INK2)
        ax.set_yticks(range(len(order)), [lab[o] for o in order], fontsize=9, color=INK2)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(ttl, color=INK, fontsize=11, loc="left")
    fig.suptitle("各篮子相对同频池等权的日超额相关:低波动 / 价值 / 红利与排雷是同一条腿(ρ 0.5-0.9)", color=INK, fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT / "inc_t5_corr.png", dpi=150, facecolor=SURF)
    plt.close(fig)


def fig_ir():
    fig, axs = plt.subplots(1, 2, figsize=(12.0, 4.5), facecolor=SURF)
    cfgs = [("S|LIQ|20_35", "排雷", BLUE), ("C|EW5|45_60", "5 篮子等权", ORANGE), ("C|CORE5|45_60", "核心-卫星", AQUA),
            ("C|ERC5|45_60", "逆 TE 等风险", GRAY)]
    wl = {"lh": ["全期 2017-25", "开发 2017-19", "检验 2020-25"], "min": ["全期 2021-25", "开发 2021-23", "检验 2024-25"]}
    for ax, tr, ttl in ((axs[0], "lh", "B-LH"), (axs[1], "min", "B-MIN")):
        td = TrackData(tr)
        x = np.arange(3)
        wdt = 0.19
        for k, (n, lab, col) in enumerate(cfgs):
            vals = [ir_of(td, n, w) for w in ("full", "dev", "test")]
            ax.bar(x + (k - 1.5) * wdt, vals, wdt - 0.02, color=col, label=lab)
            for xi, v in zip(x + (k - 1.5) * wdt, vals):
                ax.text(xi, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=7.5, color=INK2)
        ax.set_xticks(x, wl[tr], fontsize=9, color=INK2)
        ax.set_title(f"{ttl}:IR = 算术年化超额 / 跟踪误差(成本后,对同频池等权)", color=INK, fontsize=10.5, loc="left")
        style(ax)
    axs[0].legend(frameon=False, fontsize=8.5, loc="upper right", labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(OUT / "inc_t5_ir_windows.png", dpi=150, facecolor=SURF)
    plt.close(fig)


def fig_placebo():
    d = pd.read_csv(LEDGER / "inc_t5_dir.csv")
    fig, axs = plt.subplots(1, 3, figsize=(13.0, 4.2), facecolor=SURF, sharey=False)
    for ax, tr, main, ttl in ((axs[0], "lh", ["C|EW5|45_60", "C|CORE5|45_60", "C|EW5|20_35"], "B-LH"),
                              (axs[1], "min", ["C|EW5|45_60", "C|CORE5|45_60"], "B-MIN"),
                              (axs[2], "a", ["C|A2_120", "C|COREA2_120"], "A 轨(排雷 + 低波动)")):
        x = d[(d.track == tr) & (d.window == "full")]
        pl = x[x.config.str.startswith("P|")]
        ax.scatter(np.zeros(len(pl)) + np.random.default_rng(0).normal(0, 0.03, len(pl)), pl["d_ir"], s=36, color=GRAY, alpha=0.85,
                   label="随机篮子安慰剂", zorder=3)
        for k, n in enumerate(main):
            r = x[x.config == n].iloc[0]
            col = [ORANGE, AQUA, BLUE][k % 3]
            xk = 1 + 0.45 * k
            ax.scatter([xk], [r["d_ir"]], s=60, color=col, zorder=4)
            ax.text(xk + 0.07, r["d_ir"], n.replace("C|", "").replace("|", " "), fontsize=8, color=INK2, va="center", ha="left")
            if np.isfinite(r.get("boot_lo90", np.nan)):
                ax.plot([xk, xk], [r["boot_lo90"], r["boot_hi90"]], color=col, lw=1.4, zorder=2)
        ax.axhline(0, color=INK, lw=0.9)
        ax.set_xticks([0, 1 + 0.45 * (len(main) - 1) / 2], ["安慰剂", "真实组合"], fontsize=9, color=INK2)
        ax.set_xlim(-0.5, 2.0 + 0.45 * (len(main) - 1))
        ax.set_title(ttl + ":ΔIR = IR(组合) − IR(排雷)", color=INK, fontsize=10.5, loc="left")
        style(ax)
    fig.suptitle("ΔIR 全部为负(竖线 = 20 日块自助 90% 区间);真实风格篮子在 B-MIN 甚至不如随机篮子", color=INK, fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT / "inc_t5_placebo.png", dpi=150, facecolor=SURF)
    plt.close(fig)


if __name__ == "__main__":
    for fn in (fig_nav, fig_corr, fig_ir, fig_placebo):
        fn()
        log(f"[figs] {fn.__name__}")
