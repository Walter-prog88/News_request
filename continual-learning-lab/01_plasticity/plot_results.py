"""
把一个或多个结果 CSV 画成 3 联图：误差 / 死亡单元比例 / 平均权重大小。
多个 CSV 会画在同一张图上，方便比较（例如 bp vs cbp）。

只依赖 matplotlib（不需要 torch），所以可以用任何装了 matplotlib 的 Python 运行：
    python3 plot_results.py results/a.csv results/b.csv -o results/compare.png
"""

import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")   # 不弹窗口，直接保存图片
import matplotlib.pyplot as plt


def read_csv(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return {k: [float(r[k]) for r in rows] for k in rows[0]}


def plot_csvs(csv_paths, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    panels = [
        ("mse", "每 1 万样本的均方误差 (MSE)"),
        ("dead_frac", "死亡单元比例 (tanh/sigmoid 为饱和比例)"),
        ("mean_abs_weight", "平均 |权重|"),
    ]
    # 尝试使用 macOS 自带的中文字体，避免中文显示成方块
    plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    for path in csv_paths:
        d = read_csv(path)
        label = os.path.splitext(os.path.basename(path))[0]
        for ax, (key, _) in zip(axes, panels):
            ax.plot(d["examples"], d[key], label=label, linewidth=1.2)
    for ax, (_, title) in zip(axes, panels):
        ax.set_title(title)
        ax.set_xlabel("样本数")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print("图已保存：", out_path)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("csvs", nargs="+")
    p.add_argument("-o", "--out", default=None, help="输出 PNG 路径（默认与第一个 CSV 同名）")
    a = p.parse_args()
    plot_csvs(a.csvs, a.out or os.path.splitext(a.csvs[0])[0] + ".png")
