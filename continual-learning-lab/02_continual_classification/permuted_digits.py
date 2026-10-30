"""
实验 02：打乱像素的手写数字（Permuted Digits）—— 在同一个网络上同时测"可塑性"和"遗忘"
=====================================================================================

论文思想来源：
  - Dohare et al., Nature 2024 的 Permuted MNIST 实验：每个任务把像素按一个新的随机顺序打乱，
    网络依次学几百个任务。普通反向传播在新任务上的准确率会越来越低（可塑性丧失）。
  - 思想地图 n32「灾难性遗忘与持续学习」：EWC（重要权重拉回原位）和经验回放（混入旧样本）。
这两篇其实在讲"稳定性—可塑性"两难的两端：
  - 遗忘 = 太可塑，旧知识被冲掉；
  - 可塑性丧失 = 太僵，新知识学不进去。
本实验在同一个设置里同时测两边，看每种方法落在两难的哪个位置。

数据：scikit-learn 自带的 8×8 手写数字（1797 张，已导出到 ../data/digits_8x8.npz），不用下载。
      MNIST 是 28×28、6 万张，这里是它的迷你版，CPU 上几分钟跑完。

每个任务结束时记录：
  - acc_current ：当前任务的测试准确率          → 可塑性（越来越低 = 可塑性丧失）
  - acc_task0   ：第 0 个任务的测试准确率        → 长期遗忘（掉得越快 = 遗忘越严重）
  - acc_prev    ：学完当前任务后，上一个任务的准确率 → 短期遗忘
  - dead_frac   ：对当前任务所有测试样本都输出 0 的 ReLU 单元比例
  - eff_rank    ：最后一个隐藏层表示的"有效秩"（Roy & Vetterli 2007；论文用它衡量表示的多样性）
  - mean_abs_w  ：平均 |权重|

用法（torch 环境）：
    python permuted_digits.py --method bp --tasks 200 --seed 0
"""

import argparse
import csv
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data", "digits_8x8.npz")


# ----------------------------------------------------------------------------
# 数据
# ----------------------------------------------------------------------------

def load_data(seed=0):
    d = np.load(DATA)
    X, y = torch.tensor(d["X"]), torch.tensor(d["y"])
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(X), generator=g)
    n_test = 360
    te, tr = idx[:n_test], idx[n_test:]
    return X[tr], y[tr], X[te], y[te]


# ----------------------------------------------------------------------------
# 网络：64 → H → H → 10 的多层感知机
# ----------------------------------------------------------------------------

class MLP(nn.Module):
    def __init__(self, hidden=100, depth=2):
        super().__init__()
        sizes = [64] + [hidden] * depth
        self.hidden = nn.ModuleList(nn.Linear(a, b) for a, b in zip(sizes[:-1], sizes[1:]))
        self.out = nn.Linear(hidden, 10)
        for layer in self.hidden:
            nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        acts = []
        for layer in self.hidden:
            x = F.relu(layer(x))
            acts.append(x)
        return self.out(x), acts

    def next_layer(self, i):
        """第 i 个隐藏层的"下游"线性层（CBP 需要它的权重来算效用、置零输出权重）。"""
        return self.hidden[i + 1] if i + 1 < len(self.hidden) else self.out


# ----------------------------------------------------------------------------
# 各种方法
# ----------------------------------------------------------------------------

class CBP:
    """多层版的简化持续反向传播：对每个隐藏层，按替换率把效用最低的成熟单元重新初始化。
    效用 = |激活| × Σ|输出权重|（指数滑动平均）。详细解释见 01_plasticity 里的单层版本。"""

    def __init__(self, net, rate=1e-4, decay=0.99, maturity=100):
        self.net, self.rate, self.decay, self.maturity = net, rate, decay, maturity
        self.util = [torch.zeros(l.out_features) for l in net.hidden]
        self.age = [torch.zeros(l.out_features) for l in net.hidden]
        self.acc = [0.0 for _ in net.hidden]
        self.replaced = 0

    @torch.no_grad()
    def step(self, acts):
        for i, (layer, h) in enumerate(zip(self.net.hidden, acts)):
            nxt = self.net.next_layer(i)
            u = h.abs().mean(0) * nxt.weight.abs().sum(0)
            self.util[i] = self.decay * self.util[i] + (1 - self.decay) * u
            self.age[i] += 1
            mature = self.age[i] > self.maturity
            self.acc[i] += self.rate * mature.sum().item()
            while self.acc[i] >= 1 and mature.any():
                self.acc[i] -= 1
                util = self.util[i].clone()
                util[~mature] = float("inf")
                j = int(torch.argmin(util))
                w = torch.empty(1, layer.in_features)
                nn.init.kaiming_uniform_(w, nonlinearity="relu")
                layer.weight[j] = w[0]
                layer.bias[j] = 0.0
                nxt.weight[:, j] = 0.0           # 新单元的输出权重置 0，不扰动当前输出
                self.util[i][j], self.age[i][j] = 0.0, 0
                mature[j] = False
                self.replaced += 1


class EWC:
    """在线 EWC（Kirkpatrick et al. 2017；在线版见 Schwarz et al. 2018）：
    每学完一个任务，用该任务的数据估计对角费希尔信息 F，累加起来；
    之后的损失加上 (λ/2)·Σ F_i (θ_i − θ*_i)²，θ* = 上一个任务结束时的参数。"""

    def __init__(self, net, lam, lr, gamma=0.9):
        self.net, self.lam, self.gamma = net, lam, gamma
        # 刚度上限：SGD 对二次惩罚 (k/2)(θ−a)² 的更新是 θ ← θ − lr·k·(θ−a)，lr·k > 2 时会振荡发散。
        # 所以把每个参数的刚度 λ·F 截断在 0.5/lr（数值保护，不改变"重要的权重拉得更紧"的思想）。
        self.max_stiff = 0.5 / lr
        self.F = [torch.zeros_like(p) for p in net.parameters()]
        self.anchor = None

    def penalty(self):
        if self.anchor is None:
            return 0.0
        return 0.5 * sum(((self.lam * f).clamp(max=self.max_stiff) * (p - a) ** 2).sum()
                         for f, p, a in zip(self.F, self.net.parameters(), self.anchor))

    def end_task(self, X, y, n=200):
        new_F = [torch.zeros_like(p) for p in self.net.parameters()]
        for i in torch.randperm(len(X))[:n]:            # 经验费希尔：逐样本梯度的平方
            self.net.zero_grad()
            logits, _ = self.net(X[i:i + 1])
            F.cross_entropy(logits, y[i:i + 1]).backward()
            for f, p in zip(new_F, self.net.parameters()):
                f += p.grad.detach() ** 2 / n
        # 在线 EWC：旧的重要性按 γ 衰减，再加上新任务的（否则累加几百个任务后刚度无限变大）
        self.F = [self.gamma * f + nf for f, nf in zip(self.F, new_F)]
        self.anchor = [p.detach().clone() for p in self.net.parameters()]


class Replay:
    """经验回放：用"蓄水池抽样"保留一个固定大小的旧样本缓冲区，每个批次混入同样多的旧样本。
    对应思想地图里的互补学习系统：海马体存经历，睡眠时回放给新皮层。"""

    def __init__(self, size=500):
        self.size, self.X, self.y, self.seen = size, [], [], 0

    def add(self, X, y):
        for xi, yi in zip(X, y):
            self.seen += 1
            if len(self.X) < self.size:
                self.X.append(xi), self.y.append(yi)
            else:
                j = np.random.randint(self.seen)
                if j < self.size:
                    self.X[j], self.y[j] = xi, yi

    def sample(self, n):
        if not self.X:
            return None
        idx = np.random.randint(len(self.X), size=n)
        return torch.stack([self.X[i] for i in idx]), torch.stack([self.y[i] for i in idx])


# ----------------------------------------------------------------------------
# 评估
# ----------------------------------------------------------------------------

@torch.no_grad()
def evaluate(net, X, y):
    logits, acts = net(X)
    acc = (logits.argmax(1) == y).float().mean().item()
    dead = torch.cat([(a.max(0).values <= 0) for a in acts]).float().mean().item()
    s = torch.linalg.svdvals(acts[-1])                   # 有效秩 = exp(奇异值分布的熵)
    p = s / s.sum().clamp_min(1e-12)
    eff_rank = torch.exp(-(p * torch.log(p.clamp_min(1e-12))).sum()).item()
    return acc, dead, eff_rank


# ----------------------------------------------------------------------------
# 主循环
# ----------------------------------------------------------------------------

def run(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(1)
    Xtr, ytr, Xte, yte = load_data(args.seed)
    g = torch.Generator().manual_seed(args.seed + 123)
    perms = [torch.randperm(64, generator=g) for _ in range(args.tasks)]

    net = MLP(args.hidden, args.depth)
    wd = args.weight_decay if args.method == "l2" else 0.0
    opt = torch.optim.SGD(net.parameters(), lr=args.lr, weight_decay=wd)
    cbp = CBP(net, rate=args.replacement_rate) if args.method == "cbp" else None
    ewc = EWC(net, args.ewc_lambda, args.lr) if args.method == "ewc" else None
    replay = Replay(args.buffer) if args.method == "replay" else None

    rows, t0 = [], time.time()
    for t, perm in enumerate(perms):
        Xt, Xt_te = Xtr[:, perm], Xte[:, perm]
        for _ in range(args.epochs):
            order = torch.randperm(len(Xt))
            for s in range(0, len(Xt), args.batch):
                b = order[s:s + args.batch]
                xb, yb = Xt[b], ytr[b]
                if replay is not None:
                    old = replay.sample(len(b))
                    if old is not None:
                        xb, yb = torch.cat([xb, old[0]]), torch.cat([yb, old[1]])
                logits, acts = net(xb)
                loss = F.cross_entropy(logits, yb)
                if ewc is not None:
                    loss = loss + ewc.penalty()
                opt.zero_grad()
                loss.backward()
                opt.step()
                if args.method == "snp":
                    with torch.no_grad():
                        for p in net.parameters():
                            p.mul_(1 - args.lr * args.weight_decay).add_(torch.randn_like(p) * args.noise_std)
                if cbp is not None:
                    cbp.step([a.detach() for a in acts])
        if ewc is not None:
            ewc.end_task(Xt, ytr)
        if replay is not None:
            keep = torch.randperm(len(Xt))[:100]
            replay.add(Xt[keep], ytr[keep])

        acc_cur, dead, rank = evaluate(net, Xt_te, yte)
        acc0, _, _ = evaluate(net, Xte[:, perms[0]], yte)
        acc_prev = evaluate(net, Xte[:, perms[t - 1]], yte)[0] if t > 0 else float("nan")  # 短期遗忘
        w = torch.cat([p.detach().abs().flatten() for n, p in net.named_parameters() if "weight" in n]).mean().item()
        rows.append({"task": t, "acc_current": acc_cur, "acc_task0": acc0, "acc_prev": acc_prev, "dead_frac": dead,
                     "eff_rank": rank, "mean_abs_w": w})
        if t % max(1, args.tasks // 10) == 0 or t == args.tasks - 1:
            print(f"[任务 {t:>4d}] 当前任务准确率={acc_cur:.3f}  任务0准确率={acc0:.3f}  死亡={dead:.2f}  "
                  f"有效秩={rank:.1f}  |w|={w:.3f}  ({time.time() - t0:.0f}s)", flush=True)
    return rows


def main():
    p = argparse.ArgumentParser(description="打乱像素的手写数字：可塑性 + 遗忘")
    p.add_argument("--method", choices=["bp", "l2", "snp", "cbp", "ewc", "replay"], default="bp")
    p.add_argument("--tasks", type=int, default=200, help="任务个数（每个任务是一种新的像素打乱方式）")
    p.add_argument("--epochs", type=int, default=3, help="每个任务训练几轮")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--hidden", type=int, default=100)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--weight-decay", type=float, default=1e-3, help="l2 / snp 的衰减强度")
    p.add_argument("--noise-std", type=float, default=1e-3, help="snp 的噪声")
    p.add_argument("--replacement-rate", type=float, default=1e-4, help="cbp 的替换率")
    p.add_argument("--ewc-lambda", type=float, default=100.0, help="EWC 惩罚强度")
    p.add_argument("--buffer", type=int, default=500, help="回放缓冲区大小")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default="", help="附加到文件名的标签（做参数扫描时用）")
    args = p.parse_args()

    print(f"method={args.method} tasks={args.tasks} lr={args.lr} hidden={args.hidden} seed={args.seed}")
    rows = run(args)
    out = os.path.join(HERE, "results")
    os.makedirs(out, exist_ok=True)
    name = f"{args.method}{args.tag}_lr{args.lr}_seed{args.seed}.csv"
    with open(os.path.join(out, name), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("CSV 已保存：", os.path.join(out, name))


if __name__ == "__main__":
    main()
