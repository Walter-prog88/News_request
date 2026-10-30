"""
汇总 results/sweep/ 里的批量实验：同一配置的多个种子取平均，画"均值 ± 标准误"三联图，并输出汇总表。

只依赖 numpy + matplotlib，用系统 python3 运行：
    /usr/bin/python3 summarize.py
输出：results/sweep_figs/exp*.png 和 results/sweep_figs/summary.md
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

HERE = os.path.dirname(os.path.abspath(__file__))
SWEEP = os.path.join(HERE, "results", "sweep")
FIGS = os.path.join(HERE, "results", "sweep_figs")
METRICS = [("mse", "均方误差（5 点滑动平均）"), ("dead_frac", "死亡/饱和单元比例"), ("mean_abs_weight", "平均 |权重|")]

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def load_groups():
    """把文件名里的 _seedN 去掉作为"配置名"，同一配置的不同种子归为一组。"""
    groups = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(SWEEP, "*.csv"))):
        name = os.path.splitext(os.path.basename(path))[0]
        config = re.sub(r"_seed\d+", "", name)
        with open(path) as f:
            rows = list(csv.DictReader(f))
        groups[config].append({k: np.array([float(r[k]) for r in rows]) for k in rows[0]})
    return groups


def smooth(y, k=5):
    return np.convolve(y, np.ones(k) / k, mode="valid") if len(y) >= k else y


def plot_experiment(groups, configs, title, out_name):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    for config in configs:
        runs = groups.get(config)
        if not runs:
            continue
        x = runs[0]["examples"]
        for ax, (key, _) in zip(axes, METRICS):
            Y = np.stack([r[key] for r in runs])            # [种子数, 分箱数]
            if key == "mse":
                Y = np.stack([smooth(y) for y in Y])
                xs = x[len(x) - Y.shape[1]:]
            else:
                xs = x
            mean = Y.mean(0)
            se = Y.std(0, ddof=1) / np.sqrt(len(Y)) if len(Y) > 1 else np.zeros_like(mean)
            line, = ax.plot(xs, mean, linewidth=1.3, label=f"{config}（{len(runs)} 个种子）")
            ax.fill_between(xs, mean - se, mean + se, alpha=0.2, color=line.get_color())
    for ax, (_, t) in zip(axes, METRICS):
        ax.set_title(t)
        ax.set_xlabel("样本数")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.suptitle(title)
    fig.tight_layout()
    path = os.path.join(FIGS, out_name)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print("图已保存：", path)


def summary_table(groups):
    """每个配置：前 20% 与后 20% 分箱的平均值（先对每个种子求，再在种子间平均）。"""
    lines = ["| 配置 | 种子数 | MSE 前20% → 后20% | 死亡比例 前 → 后 | 平均\\|w\\| 前 → 后 | CBP 累计重置 |",
             "|---|---|---|---|---|---|"]
    for config in sorted(groups):
        runs = groups[config]
        n = len(runs[0]["mse"])
        q = max(1, n // 5)
        cells = []
        for key in ["mse", "dead_frac", "mean_abs_weight"]:
            a = np.mean([r[key][:q].mean() for r in runs])
            b = np.mean([r[key][-q:].mean() for r in runs])
            cells.append(f"{a:.2f} → {b:.2f}")
        replaced = np.mean([r["num_replaced"][-1] for r in runs])
        lines.append(f"| {config} | {len(runs)} | " + " | ".join(cells) + f" | {replaced:.0f} |")
    return "\n".join(lines)


def plot_probe():
    """实验 F：横轴 = 存档时已训练的样本数，纵轴 = 存档网络学新任务的误差（越高 = 可塑性越差）。"""
    paths = sorted(glob.glob(os.path.join(HERE, "results", "probe", "*.csv")))
    if not paths:
        return
    by_method = defaultdict(list)
    for path in paths:
        method = os.path.basename(path).split("_seed")[0]
        with open(path) as f:
            rows = list(csv.DictReader(f))
        by_method[method].append({k: np.array([float(r[k]) for r in rows]) for k in rows[0]})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    lines = ["\n## 实验 F：可塑性探针（存档网络学新任务的平均误差，种子间均值 ± 标准误）\n",
             "| 方法 | 种子数 | " + " | ".join(f"{int(c):,} 处存档" for c in by_method[next(iter(by_method))][0]["checkpoint"]) + " |",
             "|---|---|" + "---|" * len(by_method[next(iter(by_method))][0]["checkpoint"])]
    for method, runs in sorted(by_method.items()):
        x = runs[0]["checkpoint"]
        for ax, key in zip(axes, ["probe_mse_mean", "dead_after_probe"]):
            Y = np.stack([r[key] for r in runs])
            mean = Y.mean(0)
            se = Y.std(0, ddof=1) / np.sqrt(len(Y)) if len(Y) > 1 else np.zeros_like(mean)
            ax.errorbar(x, mean, yerr=se, marker="o", capsize=3, label=f"{method}（{len(runs)} 个种子）")
        Y = np.stack([r["probe_mse_mean"] for r in runs])
        se = Y.std(0, ddof=1) / np.sqrt(len(Y)) if len(Y) > 1 else np.zeros(Y.shape[1])
        lines.append(f"| {method} | {len(runs)} | " + " | ".join(f"{m:.3f} ± {s:.3f}" for m, s in zip(Y.mean(0), se)) + " |")
    # 配对分析：不同种子的新任务难度不同，直接平均误差带很宽。
    # 用"同一种子下、新网络（0 处存档）的探针误差"作分母，只看相对变化，噪声小得多。
    lines += ["", "配对版本（每个种子各自除以自己 0 处存档的探针误差；1.00 = 和新网络一样好学）：", "",
              lines[1], lines[2]]
    for method, runs in sorted(by_method.items()):
        R = np.stack([r["probe_mse_mean"] / r["probe_mse_mean"][0] for r in runs])
        se = R.std(0, ddof=1) / np.sqrt(len(R)) if len(R) > 1 else np.zeros(R.shape[1])
        lines.append(f"| {method} | {len(runs)} | " + " | ".join(f"{m:.2f} ± {s:.2f}" for m, s in zip(R.mean(0), se)) + " |")
    axes[0].set_title("存档网络学新任务的平均误差（越高 = 越难学）")
    axes[1].set_title("学完新任务后的死亡单元比例")
    for ax in axes:
        ax.set_xlabel("存档时已训练的样本数")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("实验 F：可塑性探针（新任务统一用普通 SGD 学 10 万个样本）")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "expF_probe.png"), dpi=120)
    plt.close(fig)
    with open(os.path.join(FIGS, "summary_probe.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    os.makedirs(FIGS, exist_ok=True)
    g = load_groups()
    plot_experiment(g, [f"relu_{m}_lr0.01_n1000000" for m in ["bp", "l2", "cbp"]],
                    "实验 A：三种方法对比（relu，lr=0.01）", "expA_methods.png")
    plot_experiment(g, [f"relu_bp_lr{lr}_n1000000" for lr in ["0.01", "0.003", "0.001"]],
                    "实验 B：普通反传的学习率扫描（relu）", "expB_stepsize.png")
    plot_experiment(g, [f"{a}_{m}_lr0.01_n1000000" for a in ["tanh", "sigmoid"] for m in ["bp", "cbp"]],
                    "实验 C：tanh / sigmoid 激活（lr=0.01）", "expC_activation.png")
    plot_experiment(g, [f"relu_{m}_lr0.01_n3000000" for m in ["bp", "cbp"]],
                    "实验 D：300 万样本长程运行（relu，lr=0.01）", "expD_long.png")
    plot_experiment(g, [f"relu_{m}_lr0.01_n1000000" for m in ["bp", "l2", "l2init", "snp", "cbp"]],
                    "实验 E：把权重拉回参考点的几种做法（relu，lr=0.01）", "expE_anchor.png")
    plot_experiment(g, ["relu_bp_lr0.01_n1000000", "relu_cbp_lr0.01_n1000000", "relu_cbp_lr0.01_n1000000_urandom"],
                    "实验 G：CBP 消融——按效用挑单元 vs 随机挑单元", "expG_utility_ablation.png")
    plot_experiment(g, [f"relu_{m}_lr0.01_n1000000{h}" for h in ["", "_h20", "_h100"] for m in ["bp", "cbp"]],
                    "实验 H：网络宽度 5 / 20 / 100", "expH_width.png")
    plot_probe()
    table = summary_table(g)
    with open(os.path.join(FIGS, "summary.md"), "w") as f:
        f.write(table + "\n")
    print(table)


if __name__ == "__main__":
    main()
