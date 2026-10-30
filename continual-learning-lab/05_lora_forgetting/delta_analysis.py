"""
实验 05 的追问：为什么这里全参数微调反而忘得更少？
====================================================

假设：要把网络改到"同样会做任务 B"，LoRA 只能沿少数几个方向改，所以每个方向都得改得很大；
     全参数微调可以把改动摊到很多方向上，每个方向都改得很小。
     大而集中的改动，更可能碰到任务 A 依赖的那些方向。

检验：每种方法都训练到任务 B 准确率 ≥ 0.93 就停（保证"学到的一样多"），然后比较每一层的权重改动 ΔW：
  - Frobenius 范数 ‖ΔW‖_F（总改动量）
  - 谱范数 ‖ΔW‖_2（最大那个方向的改动量）
  - 有效秩（改动摊在多少个方向上）
  - 任务 A 剩下的准确率

用法（torch 环境）：python delta_analysis.py
"""

import os

import torch
import torch.nn as nn

from lora_vs_full import MLP, LoRALinear, acc, load, make_task_b, train

HERE = os.path.dirname(os.path.abspath(__file__))


def eff_rank(M):
    s = torch.linalg.svdvals(M)
    p = s / s.sum()
    return torch.exp(-(p * torch.log(p.clamp_min(1e-12))).sum()).item()


def delta_weights(net, pre):
    """返回每一层的 ΔW（相对预训练权重）。"""
    out = []
    for i, layer in enumerate(net.layers):
        if isinstance(layer, LoRALinear):
            out.append((layer.scale * layer.B @ layer.A).detach())
        else:
            out.append((layer.weight - pre.layers[i].weight).detach())
    return out


def main():
    torch.set_num_threads(1)
    lines = ["| 任务B | 方法 | 种子 | 轮数 | 任务B准确率 | 任务A剩下 | 各层 ‖ΔW‖_F | 各层 ‖ΔW‖_2 | 各层 ΔW 有效秩 |",
             "|---|---|---|---|---|---|---|---|---|"]
    for kind in ["flip", "permute"]:
        for seed in range(5):
            torch.manual_seed(seed)
            Xtr, ytr, Xte, yte = load(seed)
            pre = MLP()
            train(pre, Xtr, ytr, lr=0.05, epochs=30)
            to_b = make_task_b(kind, seed)
            for method, lr in [("full", 0.03), ("lora_8", 0.1), ("lora_2", 0.1)]:
                net = MLP()
                net.load_state_dict(pre.state_dict())
                if method != "full":
                    net.layers = nn.ModuleList(LoRALinear(l, int(method.split("_")[1])) for l in net.layers)
                torch.manual_seed(seed + 1)
                ep = 0
                while acc(net, to_b(Xte), yte) < 0.93 and ep < 100:   # 训练到"学得一样多"为止
                    train(net, to_b(Xtr), ytr, lr=lr, epochs=1)
                    ep += 1
                dws = delta_weights(net, pre)
                fro = "/".join(f"{d.norm():.2f}" for d in dws)
                spec = "/".join(f"{torch.linalg.matrix_norm(d, 2):.2f}" for d in dws)
                rk = "/".join(f"{eff_rank(d):.1f}" for d in dws)
                lines.append(f"| {kind} | {method} | {seed} | {ep} | {acc(net, to_b(Xte), yte):.3f} | "
                             f"{acc(net, Xte, yte):.3f} | {fro} | {spec} | {rk} |")
                print(lines[-1], flush=True)
    with open(os.path.join(HERE, "results", "delta_analysis.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
