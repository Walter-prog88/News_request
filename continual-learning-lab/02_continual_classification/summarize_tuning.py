"""
汇总实验 07（公平调参）：用系统 python3 运行
    /usr/bin/python3 summarize_tuning.py
对每种方法的每个超参数取值，算最后 200 个任务的 当前任务准确率（可塑性）和 上一任务准确率（记忆），
3 个种子的均值 ± 标准误；再按可塑性挑出每种方法的最佳取值，做最终对比。
输出 results/figs/tuning.md
"""

import csv
import glob
import os
import re
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LAST = 200


def main():
    groups = defaultdict(list)
    for path in glob.glob(os.path.join(HERE, "results", "*_tune*.csv")):
        name = os.path.basename(path)[:-4]
        config = re.sub(r"_seed\d+", "", name)
        with open(path) as f:
            rows = list(csv.DictReader(f))
        cur = np.mean([float(r["acc_current"]) for r in rows[-LAST:]])
        prev = np.nanmean([float(r["acc_prev"]) for r in rows[-LAST:]])
        groups[config].append((cur, prev))

    def ms(v):
        v = np.array(v)
        return v.mean(), (v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0)

    lines = ["## 实验 07：每个超参数取值的结果（最后 200 个任务，3 个种子）", "",
             "| 方法 | 配置 | 种子数 | 可塑性（当前任务准确率） | 记忆（上一任务准确率） |", "|---|---|---|---|---|"]
    best = {}
    for config in sorted(groups):
        method = config.split("_")[0]
        p, s = ms([c for c, _ in groups[config]]), ms([q for _, q in groups[config]])
        lines.append(f"| {method} | {config} | {len(groups[config])} | {p[0]:.3f} ± {p[1]:.3f} | {s[0]:.3f} ± {s[1]:.3f} |")
        if method not in best or p[0] > best[method][1][0]:
            best[method] = (config, p, s)
    lines += ["", "## 每种方法在其最佳超参数下（按可塑性挑选）", "",
              "| 方法 | 最佳配置 | 可塑性 | 记忆 |", "|---|---|---|---|"]
    for method, (config, p, s) in sorted(best.items(), key=lambda kv: -kv[1][1][0]):
        lines.append(f"| {method} | {config} | {p[0]:.3f} ± {p[1]:.3f} | {s[0]:.3f} ± {s[1]:.3f} |")
    os.makedirs(os.path.join(HERE, "results", "figs"), exist_ok=True)
    with open(os.path.join(HERE, "results", "figs", "tuning.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
