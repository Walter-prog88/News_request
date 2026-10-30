"""Small-MLP LoRA versus full-update study, adapted for a public portfolio.
This is not a language-model or robotics benchmark. Archived metrics are historical.
Training is opt-in: python lora_vs_full.py --run-training
Public digits data are loaded through scikit-learn, not bundled in this repository.
See ../README.md for settings, provenance, and selection limitations.
"""
import argparse
import csv
import itertools
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "runs")


class LoRALinear(nn.Module):
    """y = x·Wᵀ + b + α·x·Aᵀ·Bᵀ，其中 W、b 冻结，只训练 A（r×in）和 B（out×r）。"""

    def __init__(self, base: nn.Linear, r: int, alpha: float = 1.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.A = nn.Parameter(torch.randn(r, base.in_features) / np.sqrt(base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))   # 初始为 0 → 微调开始时网络和预训练完全相同
        self.scale = alpha

    def forward(self, x):
        return self.base(x) + self.scale * (x @ self.A.T) @ self.B.T


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(64, 100), nn.Linear(100, 100), nn.Linear(100, 10)])

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = F.relu(x)
        return x


def load(seed):
    from sklearn.datasets import load_digits
    pixels, targets = load_digits(return_X_y=True)
    X = torch.tensor((pixels / 16.0).astype(np.float32))
    y = torch.tensor(targets.astype(np.int64))
    idx = torch.randperm(len(X), generator=torch.Generator().manual_seed(seed))
    return X[idx[360:]], y[idx[360:]], X[idx[:360]], y[idx[:360]]


def make_task_b(kind, seed):
    if kind == "flip":
        cols = torch.arange(64).view(8, 8).flip(1).flatten()     # 每行左右翻转
        return lambda X: X[:, cols]
    perm = torch.randperm(64, generator=torch.Generator().manual_seed(seed + 999))
    return lambda X: X[:, perm]


@torch.no_grad()
def acc(net, X, y):
    return (net(X).argmax(1) == y).float().mean().item()


def train(net, X, y, lr, epochs, batch=16, log=None):
    params = [p for p in net.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=lr)
    for ep in range(epochs):
        for b in torch.randperm(len(X)).split(batch):
            loss = F.cross_entropy(net(X[b]), y[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
        if log:
            log(ep + 1)


def main():
    parser = argparse.ArgumentParser(description='Opt-in small-MLP training; archived results are not overwritten.')
    parser.add_argument('--run-training', action='store_true')
    args = parser.parse_args()
    if not args.run_training:
        parser.error('Training is disabled by default; pass --run-training deliberately.')
    torch.set_num_threads(1)
    methods = ["full", "lora_1", "lora_2", "lora_4", "lora_8", "lora_32"]
    lrs = {"full": [0.003, 0.01, 0.03], "lora": [0.03, 0.1, 0.3]}
    rows, t0 = [], time.time()
    for seed, kind in itertools.product(range(5), ["flip", "permute"]):
        torch.manual_seed(seed)
        Xtr, ytr, Xte, yte = load(seed)
        pre = MLP()
        train(pre, Xtr, ytr, lr=0.05, epochs=30)                 # 预训练（任务 A）
        to_b = make_task_b(kind, seed)
        a0, b0 = acc(pre, Xte, yte), acc(pre, to_b(Xte), yte)
        for method in methods:
            for lr in lrs["full" if method == "full" else "lora"]:
                net = MLP()
                net.load_state_dict(pre.state_dict())
                if method != "full":
                    r = int(method.split("_")[1])
                    net.layers = nn.ModuleList(LoRALinear(l, r) for l in net.layers)
                n_train = sum(p.numel() for p in net.parameters() if p.requires_grad)
                torch.manual_seed(seed + 1)

                def log(ep):
                    if ep in (1, 2, 5, 10, 20, 40):
                        rows.append({"seed": seed, "task_b": kind, "method": method, "lr": lr, "epoch": ep,
                                     "trainable_params": n_train, "acc_b": acc(net, to_b(Xte), yte),
                                     "acc_a": acc(net, Xte, yte), "acc_a_before": a0, "acc_b_before": b0})
                train(net, to_b(Xtr), ytr, lr=lr, epochs=40, log=log)
        print(f"seed={seed} task_b={kind} 完成（{time.time() - t0:.0f}s）", flush=True)

    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "lora_vs_full.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("CSV 已保存：", path)


if __name__ == "__main__":
    main()
