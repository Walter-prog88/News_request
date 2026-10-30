"""
汇总实验 06（持续 LoRA）：用系统 python3 运行
    /usr/bin/python3 summarize.py
输出 results/continual_lora.png 和 results/summary.md
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


def main():
    groups = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(HERE, "results", "*.csv"))):
        config = re.sub(r"_seed\d+", "", os.path.basename(path)[:-4])
        with open(path) as f:
            rows = list(csv.DictReader(f))
        groups[config].append({k: np.array([float(r[k]) for r in rows]) for k in rows[0]})

    order = ["full_r8", "lora_r8", "lora_merge_r8", "lora_fresh_r8", "lora_r2", "lora_merge_r2", "lora_r32", "lora_merge_r32"]
    configs = [c for c in order if c in groups]
    metrics = [("acc_current", "当前任务准确率（可塑性）"), ("acc_original", "原始数字任务的准确率"),
               ("dead_frac", "死亡 ReLU 比例"), ("adapter_norm", "LoRA 适配器范数 ‖B·A‖（对数）")]
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.3))
    lines = ["| 配置 | 种子数 | 前 20 任务 当前准确率 | 最后 50 任务 当前准确率 | 最后 50 任务 死亡比例 | 最后 适配器范数 |",
             "|---|---|---|---|---|---|"]
    for config in configs:
        runs = groups[config]
        dashed = "_r2" in config or "_r32" in config
        for ax, (key, _) in zip(axes, metrics):
            Y = np.stack([r[key] for r in runs])   # 全是 NaN 也照常画：画不出线，但保证各图颜色一致
            k = 10
            Ys = np.stack([np.convolve(y, np.ones(k) / k, mode="valid") for y in Y])
            x = np.arange(Ys.shape[1]) + k // 2
            ax.plot(x, Ys.mean(0), "--" if dashed else "-", linewidth=1.3, label=f"{config}（{len(runs)}）")
        lines.append(f"| {config} | {len(runs)} | {np.mean([r['acc_current'][:20].mean() for r in runs]):.3f} | "
                     f"{np.mean([r['acc_current'][-50:].mean() for r in runs]):.3f} ± "
                     f"{np.std([r['acc_current'][-50:].mean() for r in runs], ddof=1) / np.sqrt(len(runs)):.3f} | "
                     f"{np.mean([r['dead_frac'][-50:].mean() for r in runs]):.2f} | "
                     f"{np.nanmean([r['adapter_norm'][-1] for r in runs]):.1f} |")
    axes[3].set_yscale("log")
    for ax, (_, t) in zip(axes, metrics):
        ax.set_title(t, fontsize=11)
        ax.set_xlabel("任务编号")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle("实验 06：持续用 LoRA 适应 300 个新任务（10 任务滑动平均；实线秩 8，虚线秩 2/32）")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "results", "continual_lora.png"), dpi=120)
    with open(os.path.join(HERE, "results", "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
