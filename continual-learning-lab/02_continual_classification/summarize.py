"""
汇总实验 02：用系统 python3 运行
    /usr/bin/python3 summarize.py
输出 results/figs/ 下的两张图和 summary.md：
  - curves.png    ：随任务数变化的 当前任务准确率 / 上一任务准确率 / 死亡比例 / 有效秩
  - tradeoff.png  ：稳定性—可塑性平面（横轴记得住多少，纵轴学得进多少），每种方法一个点
"""

import csv
import glob
import os
import re
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "results", "figs")
LAST = 200          # "最后阶段" = 最后 200 个任务


def load():
    groups = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(HERE, "results", "*.csv"))):
        name = os.path.basename(path)[:-4]
        if "_dev" in name or "_tune" in name:      # 调参结果由 summarize_tuning.py 单独汇总
            continue
        config = re.sub(r"_seed\d+", "", name)
        with open(path) as f:
            rows = list(csv.DictReader(f))
        groups[config].append({k: np.array([float(r[k]) for r in rows]) for k in rows[0]})
    return groups


def smooth(y, k=25):
    return np.convolve(y, np.ones(k) / k, mode="valid")


def mean_se(vals):
    vals = np.array(vals)
    return vals.mean(), (vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0)


def main():
    os.makedirs(FIGS, exist_ok=True)
    g = load()
    main_methods = [c for c in sorted(g) if not c.startswith("ewc")] + \
                   [c for c in ["ewc_lam10_lr0.05", "ewc_lam1000_lr0.05"] if c in g]

    metrics = [("acc_current", "当前任务测试准确率（可塑性）"), ("acc_prev", "上一个任务的准确率（短期记忆）"),
               ("dead_frac", "死亡 ReLU 单元比例"), ("eff_rank", "最后隐藏层的有效秩")]
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.3))
    for config in main_methods:
        runs = g[config]
        for ax, (key, _) in zip(axes, metrics):
            Y = np.stack([smooth(np.nan_to_num(r[key][1:], nan=0.0)) for r in runs])
            x = np.arange(Y.shape[1]) + 13
            m = Y.mean(0)
            line, = ax.plot(x, m, linewidth=1.3, label=f"{config.replace('_lr0.05', '')}（{len(runs)}）")
            if len(runs) > 1:
                se = Y.std(0, ddof=1) / np.sqrt(len(runs))
                ax.fill_between(x, m - se, m + se, alpha=0.2, color=line.get_color())
    for ax, (_, t) in zip(axes, metrics):
        ax.set_title(t, fontsize=11)
        ax.set_xlabel("任务编号")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle("实验 02：打乱像素的手写数字，1000 个任务（25 任务滑动平均，阴影 = 种子间标准误）")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "curves.png"), dpi=120)
    plt.close(fig)

    # 稳定性—可塑性平面
    lines = ["| 配置 | 种子数 | 前 50 任务 当前准确率 | 最后 200 任务 当前准确率（可塑性） | 最后 200 任务 上一任务准确率（记忆） | 最后 死亡比例 | 最后 有效秩 |",
             "|---|---|---|---|---|---|---|"]
    fig, ax = plt.subplots(figsize=(7.5, 6))
    for config in sorted(g):
        runs = g[config]
        plast = mean_se([r["acc_current"][-LAST:].mean() for r in runs])
        stab = mean_se([np.nanmean(r["acc_prev"][-LAST:]) for r in runs])
        early = mean_se([r["acc_current"][:50].mean() for r in runs])
        dead = mean_se([r["dead_frac"][-LAST:].mean() for r in runs])
        rank = mean_se([np.nanmean(r["eff_rank"][-LAST:]) for r in runs])
        label = config.replace("_lr0.05", "")
        lines.append(f"| {label} | {len(runs)} | {early[0]:.3f} | {plast[0]:.3f} ± {plast[1]:.3f} | "
                     f"{stab[0]:.3f} ± {stab[1]:.3f} | {dead[0]:.2f} | {rank[0]:.1f} |")
        is_ewc = config.startswith("ewc")
        ax.errorbar(stab[0], plast[0], xerr=stab[1], yerr=plast[1], fmt="s" if is_ewc else "o",
                    color="gray" if is_ewc else None, capsize=3)
        ax.annotate(label, (stab[0], plast[0]), textcoords="offset points", xytext=(6, 4), fontsize=9)
    ax.set_xlabel("记得住：学完新任务后，上一个任务的准确率（最后 200 个任务）")
    ax.set_ylabel("学得进：当前任务的准确率（最后 200 个任务）")
    ax.set_title("稳定性—可塑性平面（右上角最好；灰色方块 = 不同 λ 的 EWC）")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "tradeoff.png"), dpi=120)
    plt.close(fig)

    with open(os.path.join(FIGS, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
