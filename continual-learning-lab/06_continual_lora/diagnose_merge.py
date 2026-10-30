"""
实验 06 的追问：为什么"合并后重置"反而崩得最彻底？
================================================

观察：lora_merge 在第 50–100 个任务之间崩到接近随机（0.14），死亡 ReLU 升到 65%，
     每个任务结束时的适配器范数从 13 一路降到 1 左右，说明它后来几乎学不动了。

假设：每次合并都把一个"又大又集中"的低秩改动（05 里看到的）永久加进主干。
     几十次以后主干权重被推大、推偏，大量 ReLU 单元的输入长期为负而"死掉"；
     主干是冻结的，低秩适配器没法把这么多单元同时救回来；
     死掉的单元又不传梯度，适配器能学的空间越来越小。

诊断：逐任务记录 主干各层权重范数、各层死亡比例、各层偏置均值、第一个任务后的梯度范数，
     对比 full（全参数）和 lora_merge 两种方式。

用法（torch 环境）：python diagnose_merge.py
"""

import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "05_lora_forgetting"))
from lora_vs_full import MLP, LoRALinear, acc, load, train   # noqa: E402
from continual_lora import merge_and_reset                     # noqa: E402


def base_weight(layer):
    return layer.base.weight if isinstance(layer, LoRALinear) else layer.weight


@torch.no_grad()
def per_layer_dead(net, X):
    x, out = X, []
    for layer in net.layers[:-1]:
        x = F.relu(layer(x))
        out.append((x.max(0).values <= 0).float().mean().item())
    return out


def grad_norm(net, X, y):
    net.zero_grad()
    F.cross_entropy(net(X), y).backward()
    return np.sqrt(sum((p.grad ** 2).sum().item() for p in net.parameters() if p.grad is not None))


def main():
    torch.set_num_threads(1)
    lines = ["| 方法 | 任务 | 当前准确率 | 各层主干权重范数 | 各层死亡比例 | 新任务开始时梯度范数 |", "|---|---|---|---|---|---|"]
    for method in ["full", "lora_merge"]:
        torch.manual_seed(0)
        Xtr, ytr, Xte, yte = load(0)
        net = MLP()
        train(net, Xtr, ytr, lr=0.05, epochs=30)
        if method != "full":
            net.layers = nn.ModuleList(LoRALinear(l, 8) for l in net.layers)
        lr = 0.03 if method == "full" else 0.1
        g = torch.Generator().manual_seed(777)
        for t in range(150):
            perm = torch.randperm(64, generator=g)
            gn = grad_norm(net, Xtr[:, perm], ytr)
            train(net, Xtr[:, perm], ytr, lr=lr, epochs=3)
            if t in (0, 10, 25, 50, 75, 100, 149):
                wn = "/".join(f"{base_weight(l).norm():.1f}" for l in net.layers)
                dead = "/".join(f"{d:.2f}" for d in per_layer_dead(net, Xte[:, perm]))
                lines.append(f"| {method} | {t} | {acc(net, Xte[:, perm], yte):.3f} | {wn} | {dead} | {gn:.3f} |")
                print(lines[-1], flush=True)
            if method == "lora_merge":
                merge_and_reset(net)
    with open(os.path.join(HERE, "results", "diagnose_merge.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
