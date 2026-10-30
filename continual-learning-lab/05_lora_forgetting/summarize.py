"""
汇总实验 05：用系统 python3 运行
    /usr/bin/python3 summarize.py
图：横轴 = 在任务 B 上学到的准确率，纵轴 = 任务 A 剩下的准确率。
    每种方法取"最终任务 B 准确率最高"的学习率，画出它在微调过程中（第 1→40 轮）的轨迹，5 个种子平均。
    越往右 = 学得越多；越往上 = 忘得越少。
"""

import csv
import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    with open(os.path.join(HERE, "results", "lora_vs_full.csv")) as f:
        rows = list(csv.DictReader(f))
    # (task_b, method, lr, epoch) -> 种子列表
    agg = defaultdict(list)
    params = {}
    for r in rows:
        agg[(r["task_b"], r["method"], float(r["lr"]), int(r["epoch"]))].append((float(r["acc_b"]), float(r["acc_a"])))
        params[r["method"]] = int(r["trainable_params"])
    before = defaultdict(list)
    for r in rows:
        before[r["task_b"]].append((float(r["acc_b_before"]), float(r["acc_a_before"])))

    methods = ["lora_1", "lora_2", "lora_4", "lora_8", "lora_32", "full"]
    epochs = [1, 2, 5, 10, 20, 40]
    lines = ["| 任务 B | 方法 | 可训练参数 | 最佳学习率 | 任务 B 准确率（学到） | 任务 A 准确率（剩下） | 任务 A 掉了多少 |",
             "|---|---|---|---|---|---|---|"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, kind in zip(axes, ["flip", "permute"]):
        b0, a0 = np.mean(before[kind], axis=0)
        ax.plot(b0, a0, "k*", markersize=14, label="微调前")
        for method in methods:
            lrs = sorted({k[2] for k in agg if k[0] == kind and k[1] == method})
            best = max(lrs, key=lambda lr: np.mean([v[0] for v in agg[(kind, method, lr, 40)]]))
            traj = np.array([np.mean(agg[(kind, method, best, ep)], axis=0) for ep in epochs])
            style = dict(color="black", linewidth=2.2) if method == "full" else dict(linewidth=1.5)
            ax.plot(traj[:, 0], traj[:, 1], "o-", markersize=4, label=f"{method}（{params[method]:,} 参数，lr={best}）", **style)
            fb, fa = traj[-1]
            lines.append(f"| {kind} | {method} | {params[method]:,} | {best} | {fb:.3f} | {fa:.3f} | {a0 - fa:+.3f} |")
        ax.set_title({"flip": "任务 B = 左右翻转（相关的新领域）", "permute": "任务 B = 像素打乱（无关的新任务）"}[kind])
        ax.set_xlabel("任务 B 测试准确率（学到多少）→")
        ax.set_ylabel("任务 A 测试准确率（剩下多少）↑")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("实验 05：LoRA 秩 vs 全参数微调（每条线是微调第 1→40 轮的轨迹，5 个种子平均）")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "results", "lora_vs_full.png"), dpi=120)
    with open(os.path.join(HERE, "results", "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
