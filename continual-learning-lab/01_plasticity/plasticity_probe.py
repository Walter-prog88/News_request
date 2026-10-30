"""
实验 F：可塑性探针 —— 直接测"这个网络还能不能学新东西"
=========================================================

主脚本记录的是在线误差，它把"学得多快"和"任务本身多难"混在一起。这里换一个更干净的测法：
    1) 第一阶段：照常在缓变回归任务上训练（bp 或 cbp），在几个时间点把网络存一份。
    2) 第二阶段（探针）：对每个存档，换一个全新的目标网络和输入流，
       用普通 SGD（不管第一阶段用的是什么方法）训练 probe_examples 个样本，看误差。
    同一个种子下，所有存档面对的是同一个新任务，所以误差可以直接比较。
    如果存档越"老"，在新任务上的误差越高，就说明网络本身失去了可塑性。

用法（用 torch 环境运行）：
    python plasticity_probe.py --method bp --seed 0
输出：results/probe/<method>_seed<N>.csv，每行是一个存档点
"""

import argparse
import copy
import csv
import os

import torch

from slowly_changing_regression import BIN_SIZE, ContinualBackprop, InputStream, Learner, TargetNetwork


def train(net, opt, stream, target, num_examples, cbp=None):
    """在线训练 num_examples 个样本，返回每个分箱的平均平方误差和最后一个分箱的死亡单元比例。"""
    bin_mse, dead = [], 0.0
    for _ in range(num_examples // BIN_SIZE):
        x_block = stream.next_block(BIN_SIZE)
        y_block = target(x_block)
        err, ever_active = 0.0, torch.zeros(net.fc1.out_features, dtype=torch.bool)
        for t in range(BIN_SIZE):
            pred, h = net(x_block[t])
            loss = (pred - y_block[t]) ** 2
            opt.zero_grad()
            loss.backward()
            opt.step()
            if cbp is not None:
                cbp.step(h)
            err += loss.item()
            ever_active |= h.detach() > 0
        bin_mse.append(err / BIN_SIZE)
        dead = 1.0 - ever_active.float().mean().item()
    return bin_mse, dead


def probe(snapshot, args):
    """探针：拷贝存档，换一个新任务（固定种子 10000+seed），用普通 SGD 学 probe_examples 个样本。"""
    net = copy.deepcopy(snapshot)
    g = torch.Generator().manual_seed(10_000 + args.seed)
    target, stream = TargetNetwork(generator=g), InputStream(generator=g)
    torch.manual_seed(10_000 + args.seed)
    opt = torch.optim.SGD(net.parameters(), lr=args.step_size)
    bin_mse, dead = train(net, opt, stream, target, args.probe_examples)
    return sum(bin_mse) / len(bin_mse), sum(bin_mse[-3:]) / 3, dead


def main():
    p = argparse.ArgumentParser(description="可塑性探针")
    p.add_argument("--method", choices=["bp", "cbp"], default="bp")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--step-size", type=float, default=0.01)
    p.add_argument("--checkpoints", default="0,100000,300000,600000,1000000", help="在第几个样本处存档")
    p.add_argument("--probe-examples", type=int, default=100_000, help="探针阶段学新任务的样本数")
    args = p.parse_args()
    torch.set_num_threads(1)

    torch.manual_seed(args.seed)
    g = torch.Generator().manual_seed(args.seed)
    target, stream = TargetNetwork(generator=g), InputStream(generator=g)
    net = Learner("relu")
    opt = torch.optim.SGD(net.parameters(), lr=args.step_size)
    cbp = ContinualBackprop(net) if args.method == "cbp" else None

    checkpoints = [int(c) for c in args.checkpoints.split(",")]
    rows, done = [], 0
    for ck in checkpoints:
        _, dead_train = train(net, opt, stream, target, ck - done, cbp) if ck > done else ([], float("nan"))
        done = ck
        probe_mean, probe_end, probe_dead = probe(net, args)
        w = torch.cat([net.fc1.weight.abs().flatten(), net.fc2.weight.abs().flatten()]).mean().item()
        rows.append({"checkpoint": ck, "probe_mse_mean": probe_mean, "probe_mse_end": probe_end,
                     "dead_at_checkpoint": dead_train, "dead_after_probe": probe_dead, "mean_abs_weight": w})
        print(f"[{ck:>9,d}] 探针误差（平均）={probe_mean:.3f}  （最后 3 万）={probe_end:.3f}  "
              f"存档时死亡={dead_train:.2f}  |w|={w:.3f}", flush=True)

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "probe")
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, f"{args.method}_seed{args.seed}.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("CSV 已保存：", path)


if __name__ == "__main__":
    main()
