"""
实验 06（自己的问题）：持续用 LoRA 适应新环境，适配器本身会不会失去可塑性？
=========================================================================

动机（把 01、02、05 和你的 VLA 项目接起来）：
  - 机器人部署后要不断适应新环境。大模型（如 π0）通常冻结主干、用 LoRA 微调。
  - 01/02 在特定小型设置中观察到：持续 SGD 训练伴随新任务适应能力下降。
  - 05 的小 MLP 诊断显示：某些 LoRA 配置的改动更集中；不能外推到任意模型。
  那么：如果一直用同一个 LoRA 适配器学一个接一个的新任务，它会不会也"学不动"？
       论文里 CBP 的思路是"定期重置没用的单元"。对 LoRA 的对应做法是
       "把适配器合并进主干，再重新初始化"（类似 ReLoRA，Lialin et al. 2023）。这能恢复可塑性吗？代价是什么？

设置：
  - 预训练：64→100→100→10 的 MLP 在原始 8×8 数字上训练（和 05 相同）。
  - 之后依次学 T 个"像素打乱"任务，每个任务训练几轮。比较 4 种适应方式：
      full         ：全部参数持续训练
      lora         ：冻结主干，同一个秩 r 的 LoRA 一直训练下去
      lora_merge   ：每个任务结束后，把 B·A 加进主干权重，再重置 A（随机）、B（0）—— 主干会慢慢漂移
      lora_fresh   ：每个任务都换一个全新的 LoRA，主干永远不变 —— 可塑性的上限参照（但完全不积累）
  - 每个任务结束时记录：当前任务准确率（可塑性）、原始任务 A 的准确率（主干是否被破坏）、
    适配器的范数、死亡 ReLU 比例。

用法（torch 环境）：python continual_lora.py --method lora --seed 0
"""

import argparse
import csv
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "05_lora_forgetting"))
from lora_vs_full import MLP, LoRALinear, acc, load, train   # noqa: E402  复用 05 的网络和 LoRA 实现


@torch.no_grad()
def dead_fraction(net, X):
    x, dead = X, []
    for i, layer in enumerate(net.layers[:-1]):
        x = F.relu(layer(x))
        dead.append(x.max(0).values <= 0)
    return torch.cat(dead).float().mean().item()


def lora_norm(net):
    return sum((l.scale * l.B @ l.A).norm().item() for l in net.layers if isinstance(l, LoRALinear))


@torch.no_grad()
def merge_and_reset(net, fresh_only=False):
    """把每个 LoRA 的 B·A 合并进主干（fresh_only=True 时丢掉不合并），然后重新初始化 A、B。"""
    for l in net.layers:
        if not fresh_only:
            l.base.weight += l.scale * l.B @ l.A
        l.A.copy_(torch.randn_like(l.A) / np.sqrt(l.base.in_features))
        l.B.zero_()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["full", "lora", "lora_merge", "lora_fresh"], default="lora")
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--tasks", type=int, default=300)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr-full", type=float, default=0.03)
    ap.add_argument("--lr-lora", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)

    Xtr, ytr, Xte, yte = load(args.seed)
    net = MLP()
    train(net, Xtr, ytr, lr=0.05, epochs=30)                    # 预训练（任务 A）
    if args.method != "full":
        net.layers = nn.ModuleList(LoRALinear(l, args.rank) for l in net.layers)
    lr = args.lr_full if args.method == "full" else args.lr_lora
    g = torch.Generator().manual_seed(args.seed + 777)

    rows, t0 = [], time.time()
    for t in range(args.tasks):
        perm = torch.randperm(64, generator=g)
        train(net, Xtr[:, perm], ytr, lr=lr, epochs=args.epochs)
        row = {"task": t, "acc_current": acc(net, Xte[:, perm], yte), "acc_original": acc(net, Xte, yte),
               "dead_frac": dead_fraction(net, Xte[:, perm]),
               "adapter_norm": lora_norm(net) if args.method != "full" else float("nan")}
        if args.method == "lora_merge":
            merge_and_reset(net)
        elif args.method == "lora_fresh":
            merge_and_reset(net, fresh_only=True)
        rows.append(row)
        if t % 50 == 0 or t == args.tasks - 1:
            print(f"[任务 {t:>3d}] 当前={row['acc_current']:.3f}  原始任务={row['acc_original']:.3f}  "
                  f"死亡={row['dead_frac']:.2f}  适配器范数={row['adapter_norm']:.2f}  ({time.time() - t0:.0f}s)", flush=True)

    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    path = os.path.join(HERE, "results", f"{args.method}_r{args.rank}_seed{args.seed}.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("CSV 已保存：", path)


if __name__ == "__main__":
    main()
