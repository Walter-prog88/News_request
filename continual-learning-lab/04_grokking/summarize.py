"""
汇总实验 04（Grokking）：用系统 python3 运行
    /usr/bin/python3 summarize.py
输出 results/grokking.png：每个权重衰减设置一行，四列分别是 准确率 / 损失 / 权重范数 / 傅里叶集中度。
"""

import csv
import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    paths = sorted(glob.glob(os.path.join(HERE, "results", "*.csv")))
    configs = sorted({os.path.basename(p).rsplit("_seed", 1)[0] for p in paths})
    fig, axes = plt.subplots(len(configs), 4, figsize=(18, 3.2 * len(configs)), squeeze=False)
    lines = ["| 配置 | 种子 | 训练准确率首次 ≥99% 的步数 | 测试准确率首次 ≥95% 的步数 | 最终测试准确率 | 最终权重范数 |",
             "|---|---|---|---|---|---|"]
    for row, config in zip(axes, configs):
        for path in sorted(glob.glob(os.path.join(HERE, "results", config + "_seed*.csv"))):
            with open(path) as f:
                d = {k: np.array([float(r[k]) for r in rows]) for rows in [list(csv.DictReader(f))] for k in rows[0]}
            seed = path.rsplit("_seed", 1)[1][:-4]
            s = d["step"]
            row[0].plot(s, d["train_acc"], "--", label=f"训练 seed{seed}")
            row[0].plot(s, d["test_acc"], label=f"测试 seed{seed}")
            row[1].semilogy(s, d["train_loss"], "--")
            row[1].semilogy(s, d["test_loss"])
            row[2].plot(s, d["weight_norm"])
            row[3].plot(s, d["fourier_conc"])
            first = lambda arr, th: int(s[np.argmax(arr >= th)]) if (arr >= th).any() else "未达到"
            lines.append(f"| {config} | {seed} | {first(d['train_acc'], 0.99)} | {first(d['test_acc'], 0.95)} | "
                         f"{d['test_acc'][-1]:.3f} | {d['weight_norm'][-1]:.1f} |")
        for ax, t in zip(row, ["准确率（虚线训练，实线测试）", "损失（对数坐标）", "权重 L2 范数", "嵌入的傅里叶集中度"]):
            ax.set_title(f"{config}：{t}", fontsize=10)
            ax.set_xlabel("步数")
            ax.grid(alpha=0.3)
        row[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "results", "grokking.png"), dpi=110)
    with open(os.path.join(HERE, "results", "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
