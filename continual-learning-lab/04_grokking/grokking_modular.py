"""
实验 04：Grokking（"顿悟"）—— 先死记，很久以后才突然泛化
==========================================================

论文思想来源：
  - Power et al. (2022), "Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets"
  - Nanda et al. (2023), "Progress measures for grokking via mechanistic interpretability"：
    网络最后学到的是用傅里叶特征（cos/sin）做模加法的"算法"
  - Gromov (2023)：一个两层 MLP 就能出现 grokking

研究现象：某些设置下，训练准确率先升高，测试泛化延迟出现。
      已有结果只有单个种子的三个权重衰减配置，不能证明跨种子的稳定规律。
研究假设：权重衰减可能改变训练动力学并促进泛化；权重范数和傅里叶集中度只作诊断，
          本小实验没有识别网络的完整算法，也没有证明权重范数变化就是泛化的唯一原因。

任务：(a + b) mod p，p = 97。一共 97×97 = 9409 个样本，随机取 train_frac 做训练集。
网络：a、b 各查一个可学习的嵌入 → 拼接 → 一层隐藏层（ReLU）→ 输出 97 类。全批量 AdamW。

每 log_every 步记录：训练/测试准确率、训练/测试损失、权重的 L2 范数、
                    嵌入的"傅里叶集中度"（嵌入能量集中在少数几个频率上的程度，越高越像"算法解"）。

用法（torch 环境）：
    python grokking_modular.py --weight-decay 1.0 --seed 0
"""

import argparse
import csv
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))


class ModMLP(nn.Module):
    def __init__(self, p, d_embed=128, hidden=512):
        super().__init__()
        self.embed = nn.Embedding(p, d_embed)
        self.fc1 = nn.Linear(2 * d_embed, hidden)
        self.fc2 = nn.Linear(hidden, p)

    def forward(self, a, b):
        x = torch.cat([self.embed(a), self.embed(b)], dim=1)
        return self.fc2(F.relu(self.fc1(x)))


@torch.no_grad()
def fourier_concentration(embed, top_k=5):
    """把嵌入矩阵（p × d）沿"数字"那一维做离散傅里叶变换，算前 top_k 个频率占的能量比例。
    随机初始化时能量平均分布在约 p/2 个频率上（比例约 top_k / 48 ≈ 0.1）；
    学到"三角函数算法"后，能量集中在少数几个关键频率上，比例接近 1。"""
    W = embed.weight                                    # [p, d]
    spec = torch.fft.rfft(W - W.mean(0), dim=0).abs() ** 2
    power = spec[1:].sum(1)                             # 去掉频率 0，按频率求总能量
    return (power.sort(descending=True).values[:top_k].sum() / power.sum()).item()


def main():
    ap = argparse.ArgumentParser(description="Grokking：模加法")
    ap.add_argument("--p", type=int, default=97)
    ap.add_argument("--train-frac", type=float, default=0.3)
    ap.add_argument("--weight-decay", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--steps", type=int, default=20_000)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    torch.set_num_threads(2)

    p = args.p
    a, b = torch.meshgrid(torch.arange(p), torch.arange(p), indexing="ij")
    a, b = a.flatten(), b.flatten()
    y = (a + b) % p
    perm = torch.randperm(p * p)
    n_tr = int(args.train_frac * p * p)
    tr, te = perm[:n_tr], perm[n_tr:]

    net = ModMLP(p)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.98))

    rows, t0 = [], time.time()
    for step in range(args.steps + 1):
        logits = net(a[tr], b[tr])
        loss = F.cross_entropy(logits, y[tr])
        if step % args.log_every == 0:
            with torch.no_grad():
                te_logits = net(a[te], b[te])
                row = {
                    "step": step,
                    "train_acc": (logits.argmax(1) == y[tr]).float().mean().item(),
                    "test_acc": (te_logits.argmax(1) == y[te]).float().mean().item(),
                    "train_loss": loss.item(),
                    "test_loss": F.cross_entropy(te_logits, y[te]).item(),
                    "weight_norm": torch.sqrt(sum((q ** 2).sum() for q in net.parameters())).item(),
                    "fourier_conc": fourier_concentration(net.embed),
                }
            rows.append(row)
            if step % (args.log_every * 20) == 0:
                print(f"[{step:>6d}] 训练准确率={row['train_acc']:.3f}  测试准确率={row['test_acc']:.3f}  "
                      f"权重范数={row['weight_norm']:.1f}  傅里叶集中度={row['fourier_conc']:.2f}  "
                      f"({time.time() - t0:.0f}s)", flush=True)
        opt.zero_grad()
        loss.backward()
        opt.step()

    out = os.path.join(HERE, "results")
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, f"wd{args.weight_decay}_frac{args.train_frac}_seed{args.seed}.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("CSV 已保存：", path)


if __name__ == "__main__":
    main()
