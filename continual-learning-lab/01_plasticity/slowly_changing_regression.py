"""
缓慢变化回归（Slowly-Changing Regression）—— 可塑性丧失（loss of plasticity）的最小复现
=====================================================================================

参考论文：Dohare, Hernandez-Garcia, Lan, Rahman, Mahmood & Sutton,
         "Loss of plasticity in deep continual learning", Nature 2024.
官方代码（仅外部参考，不随本仓库分发）：https://github.com/shibhansh/loss-of-plasticity

这个实验想回答的问题：
    一个神经网络如果一直"在线"学习一个缓慢变化的任务，它会不会越学越差？
    论文的答案是：会。普通反向传播（backprop）训练久了，网络会逐渐失去学习新东西的能力
    （"可塑性丧失"）。典型症状：ReLU 单元"死亡"（对所有输入都输出 0）、权重越来越大。
    论文提出的"持续反向传播"（continual backprop, CBP）通过不断把"没用"的隐藏单元
    重新初始化，来保持网络的可塑性。

实验设置（与论文一致，个别细节做了简化，见注释）：
    - 输入 x：20 个比特（0/1）
        * 前 15 个是"慢比特"：每隔 T=10,000 个样本，随机挑其中 1 个翻转（0↔1）
        * 后 5 个是"快比特"：每个样本都重新随机
        * 另外还有一个恒为 1 的偏置输入（在 PyTorch 里就是 nn.Linear 的 bias）
    - 目标 y：由一个固定的随机"目标网络"产生
        * 一层隐藏层，100 个线性阈值单元（LTU，输出 0 或 1），权重随机取 ±1
        * 输出层权重也随机取 ±1，输出一个标量
    - 学习者：一层隐藏层，只有 5 个隐藏单元；在线 SGD（每次只看 1 个样本），平方误差
    因为学习者只有 5 个单元、目标网络有 100 个，学习者不可能完美拟合，
    它必须"追踪"慢比特变化后目标函数的变化 —— 这就是在考验持续学习能力。

用法示例（在 README.md 里有更详细的说明）：
    python slowly_changing_regression.py --activation relu --method bp --examples 200000
    python slowly_changing_regression.py --activation relu --method cbp --examples 200000
    python slowly_changing_regression.py --activation relu --method l2 --weight-decay 1e-3
"""

import argparse
import csv
import os
import time

import torch
import torch.nn as nn


# ----------------------------------------------------------------------------
# 1. 数据：输入流 + 固定的目标网络
# ----------------------------------------------------------------------------

NUM_INPUTS = 20        # 输入比特总数
NUM_SLOW_BITS = 15     # 其中"慢比特"的个数
FLIP_EVERY = 10_000    # 每隔多少个样本翻转一个慢比特（论文中的 T）
BIN_SIZE = 10_000      # 统计日志的"分箱"大小：每 1 万个样本记录一次


class TargetNetwork:
    """固定不变的目标网络：20 输入 -> 100 个 LTU -> 1 输出。

    LTU（linear threshold unit，线性阈值单元）：当加权和 > 阈值 tau 时输出 1，否则输出 0。
    论文里阈值的设置：tau_i = beta * (m + 1) - S_i
        m = 输入个数（20），+1 是偏置；beta = 0.7
        S_i = 第 i 个单元的输入权重（含偏置）中 -1 的个数
    直观理解：输入与权重"越匹配"（权重 +1 的位置输入为 1、权重 -1 的位置输入为 0），
    加权和越大；只有匹配程度足够高时这个单元才会被"激活"。
    （官方代码里 S 的计算方式略有不同，这里用了更直观的定义，属于合理简化。）
    """

    def __init__(self, num_hidden=100, beta=0.7, generator=None):
        g = generator
        # 权重和偏置都是随机的 ±1
        self.w1 = torch.randint(0, 2, (num_hidden, NUM_INPUTS), generator=g).float() * 2 - 1
        self.b1 = torch.randint(0, 2, (num_hidden,), generator=g).float() * 2 - 1
        self.w2 = torch.randint(0, 2, (1, num_hidden), generator=g).float() * 2 - 1
        self.b2 = torch.randint(0, 2, (1,), generator=g).float() * 2 - 1
        # S = 每个单元的（输入权重 + 偏置）中 -1 的个数
        S = (self.w1 < 0).sum(dim=1).float() + (self.b1 < 0).float()
        self.tau = beta * (NUM_INPUTS + 1) - S

    @torch.no_grad()
    def __call__(self, x):
        # x: [N, 20] -> y: [N]
        pre = x @ self.w1.T + self.b1
        h = (pre > self.tau).float()          # LTU：0/1 输出
        return (h @ self.w2.T + self.b2).squeeze(1)


class InputStream:
    """按"块"生成输入。为了速度，一次生成一个分箱（1 万个样本）的数据，再逐个喂给学习者。"""

    def __init__(self, generator):
        self.g = generator
        # 慢比特的初始值随机
        self.slow_bits = torch.randint(0, 2, (NUM_SLOW_BITS,), generator=self.g).float()

    def next_block(self, n):
        # 这一块里慢比特保持不变（因为 n == FLIP_EVERY），快比特每个样本都不同
        slow = self.slow_bits.expand(n, NUM_SLOW_BITS)
        fast = torch.randint(0, 2, (n, NUM_INPUTS - NUM_SLOW_BITS), generator=self.g).float()
        x = torch.cat([slow, fast], dim=1)
        # 这一块用完后，随机挑一个慢比特翻转，供下一块使用
        i = torch.randint(0, NUM_SLOW_BITS, (1,), generator=self.g).item()
        self.slow_bits = self.slow_bits.clone()
        self.slow_bits[i] = 1 - self.slow_bits[i]
        return x


# ----------------------------------------------------------------------------
# 2. 学习者：20 -> 5 -> 1 的小网络
# ----------------------------------------------------------------------------

ACTIVATIONS = {
    "relu": torch.relu,
    "tanh": torch.tanh,
    "sigmoid": torch.sigmoid,
}


class Learner(nn.Module):
    def __init__(self, activation, num_hidden=5):
        super().__init__()
        self.fc1 = nn.Linear(NUM_INPUTS, num_hidden)   # 输入层 -> 隐藏层（bias 就是恒为 1 的偏置输入）
        self.fc2 = nn.Linear(num_hidden, 1)            # 隐藏层 -> 输出
        self.act = ACTIVATIONS[activation]
        self.activation_name = activation
        self.init_input_weights(torch.arange(num_hidden))

    def init_input_weights(self, idx):
        """（重新）初始化第 idx 个隐藏单元的输入权重。
        使用 Kaiming（He）均匀初始化，ReLU 网络的常用选择；CBP 重置单元时也调用这个函数。"""
        with torch.no_grad():
            bound = (6.0 / NUM_INPUTS) ** 0.5 if self.activation_name == "relu" else (3.0 / NUM_INPUTS) ** 0.5
            self.fc1.weight[idx] = torch.empty(len(idx), NUM_INPUTS).uniform_(-bound, bound)
            self.fc1.bias[idx] = 0.0

    def forward(self, x):
        h = self.act(self.fc1(x))      # 隐藏层输出（激活值）
        y = self.fc2(h).squeeze(-1)    # 标量预测
        return y, h


# ----------------------------------------------------------------------------
# 3. 简化版"持续反向传播"（Continual Backprop, CBP）
# ----------------------------------------------------------------------------

class ContinualBackprop:
    """每一步：
        1) 更新每个隐藏单元的"效用"（utility）：
               贡献效用 = |该单元的激活值| × |该单元的输出权重|
           用指数滑动平均（衰减率 0.99）平滑。效用低 = 这个单元对输出几乎没贡献。
        2) 只考虑"成熟"单元（年龄 > maturity_threshold，即至少活了 100 步，
           避免新单元还没来得及学就被替换）。
        3) 按替换率 rho 累积"应替换单元数"；累计到 >= 1 时，
           把效用最低的成熟单元重新初始化：
               - 输入权重：重新随机（和初始化时同一分布）
               - 输出权重：置 0（这样新单元刚加入时不会扰乱网络当前的输出）
               - 把旧单元的平均贡献加到输出偏置上，补偿移除它带来的输出变化
    论文版本还用了"可适应贡献"（除以输入权重大小）等细节，这里做了简化。
    """

    def __init__(self, net, replacement_rate=1e-4, decay=0.99, maturity_threshold=100, utility="contrib"):
        self.net = net
        self.utility_type = utility   # "contrib"=按效用挑最差的；"random"=随机挑一个成熟单元（消融对照）
        n = net.fc1.out_features
        self.rho = replacement_rate
        self.decay = decay
        self.maturity = maturity_threshold
        self.util = torch.zeros(n)
        self.mean_act = torch.zeros(n)   # 每个单元激活值的滑动平均（用于偏置补偿）
        self.age = torch.zeros(n)
        self.to_replace = 0.0
        self.num_replaced = 0

    @torch.no_grad()
    def step(self, h):
        h = h.detach()
        w_out = self.net.fc2.weight[0]                     # [num_hidden]
        self.age += 1
        self.util = self.decay * self.util + (1 - self.decay) * (h.abs() * w_out.abs())
        self.mean_act = self.decay * self.mean_act + (1 - self.decay) * h

        mature = self.age > self.maturity
        if mature.sum() == 0:
            return
        self.to_replace += self.rho * mature.sum().item()
        while self.to_replace >= 1:
            self.to_replace -= 1
            # 在成熟单元中找效用最低的那个
            util = self.util.clone()
            util[~mature] = float("inf")
            if self.utility_type == "random":
                i = int(torch.nonzero(mature)[torch.randint(int(mature.sum()), (1,))])
            else:
                i = int(torch.argmin(util))
            # 偏置补偿：旧单元平均贡献 = mean_act * w_out
            self.net.fc2.bias += self.mean_act[i] * w_out[i]
            # 重新初始化这个单元
            self.net.init_input_weights(torch.tensor([i]))
            self.net.fc2.weight[0, i] = 0.0
            self.util[i] = 0.0
            self.mean_act[i] = 0.0
            self.age[i] = 0
            mature[i] = False
            self.num_replaced += 1


# ----------------------------------------------------------------------------
# 4. 训练主循环
# ----------------------------------------------------------------------------

def run(args):
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)   # 网络极小，单线程反而最快
    g = torch.Generator().manual_seed(args.seed)

    target = TargetNetwork(generator=g)
    stream = InputStream(generator=g)
    net = Learner(args.activation, num_hidden=args.hidden)

    # L2 正则化：在 SGD 中设置 weight_decay，相当于每步把权重往 0 拉一点
    wd = args.weight_decay if args.method == "l2" else 0.0
    opt = torch.optim.SGD(net.parameters(), lr=args.step_size, weight_decay=wd)
    cbp = ContinualBackprop(net, replacement_rate=args.replacement_rate, utility=args.utility) \
        if args.method == "cbp" else None
    # l2init：把权重往"初始值"拉，而不是往 0 拉（类似 EWC 的弹簧，只是锚点换成了初始化）
    init_params = [p.detach().clone() for p in net.parameters()] if args.method == "l2init" else None

    num_bins = args.examples // BIN_SIZE
    rows = []
    t0 = time.time()
    for b in range(num_bins):
        x_block = stream.next_block(BIN_SIZE)
        y_block = target(x_block)

        sq_err_sum = 0.0
        # ever_active[i] = 这个分箱中，第 i 个单元是否至少有一次"活着"
        ever_active = torch.zeros(net.fc1.out_features, dtype=torch.bool)
        for t in range(BIN_SIZE):
            x, y = x_block[t], y_block[t]
            pred, h = net(x)
            loss = (pred - y) ** 2           # 平方误差（先预测、再更新：记录的是"在线"误差）
            opt.zero_grad()
            loss.backward()
            opt.step()
            if args.method == "l2init":
                with torch.no_grad():   # w ← w − lr·λ·(w − w_init)
                    for p, p0 in zip(net.parameters(), init_params):
                        p.add_(p0 - p, alpha=args.step_size * args.weight_decay)
            elif args.method == "snp":
                with torch.no_grad():   # Shrink & Perturb：w ← (1 − lr·λ)·w + 噪声
                    for p in net.parameters():
                        p.mul_(1 - args.step_size * args.weight_decay).add_(torch.randn_like(p) * args.noise_std)
            if cbp is not None:
                cbp.step(h)

            sq_err_sum += loss.item()
            with torch.no_grad():
                if args.activation == "relu":
                    ever_active |= h > 0                          # ReLU：输出 > 0 才算活着
                else:
                    ever_active |= (h > 0.01) & (h < 0.99) if args.activation == "sigmoid" \
                        else h.abs() < 0.99                       # tanh/sigmoid：没饱和才算"活着"

        with torch.no_grad():
            w_mag = torch.cat([p.abs().flatten() for p in [net.fc1.weight, net.fc2.weight]]).mean().item()
        row = {
            "examples": (b + 1) * BIN_SIZE,
            "mse": sq_err_sum / BIN_SIZE,
            "dead_frac": 1.0 - ever_active.float().mean().item(),
            "mean_abs_weight": w_mag,
            "num_replaced": cbp.num_replaced if cbp else 0,
        }
        rows.append(row)
        if (b + 1) % max(1, num_bins // 20) == 0 or b == 0:
            print(f"[{row['examples']:>9,d}] mse={row['mse']:.4f}  dead={row['dead_frac']:.2f}  "
                  f"|w|={row['mean_abs_weight']:.3f}  replaced={row['num_replaced']}  "
                  f"({time.time() - t0:.0f}s)", flush=True)
    return rows


def save_results(rows, args):
    os.makedirs(args.out_dir, exist_ok=True)
    tag = f"{args.activation}_{args.method}_lr{args.step_size}_seed{args.seed}_n{args.examples}"
    # 只有偏离默认值时才把这些参数写进文件名，这样旧结果的文件名不变
    if args.hidden != 5:
        tag += f"_h{args.hidden}"
    if args.method == "cbp" and args.utility != "contrib":
        tag += f"_u{args.utility}"
    csv_path = os.path.join(args.out_dir, tag + ".csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("CSV 已保存：", csv_path, flush=True)
    if args.no_plot:      # 批量实验时不给每次运行单独画图，最后用 summarize.py 统一汇总
        return

    # 在同一虚拟环境中绘图；依赖由仓库 requirements.txt 提供。
    png_path = os.path.join(args.out_dir, tag + ".png")
    try:
        from plot_results import plot_csvs
    except ImportError:
        print("请在当前环境安装 matplotlib 后运行 plot_results.py。CSV 已保存。")
        return
    plot_csvs([csv_path], png_path)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description="缓慢变化回归：可塑性丧失最小复现")
    p.add_argument("--activation", choices=list(ACTIVATIONS), default="relu", help="隐藏层激活函数")
    p.add_argument("--method", choices=["bp", "l2", "cbp", "l2init", "snp"], default="bp",
                   help="bp=普通反向传播；l2=L2 正则化（拉向 0）；cbp=简化版持续反向传播；"
                        "l2init=L2 拉向初始权重；snp=Shrink & Perturb（衰减 + 噪声）")
    p.add_argument("--step-size", type=float, default=0.01, help="SGD 学习率（步长）")
    p.add_argument("--examples", type=int, default=1_000_000, help="总样本数（会向下取整到 1 万的倍数）")
    p.add_argument("--seed", type=int, default=0, help="随机种子（同时决定目标网络和数据流）")
    p.add_argument("--weight-decay", type=float, default=1e-3, help="L2 / 衰减强度（l2、l2init、snp 时生效）")
    p.add_argument("--noise-std", type=float, default=1e-3, help="Shrink & Perturb 每步噪声的标准差（仅 snp）")
    p.add_argument("--hidden", type=int, default=5, help="学习者隐藏单元个数（论文为 5）")
    p.add_argument("--utility", choices=["contrib", "random"], default="contrib",
                   help="CBP 挑哪个单元重置：contrib=效用最低的；random=随机（消融对照，仅 cbp）")
    p.add_argument("--replacement-rate", type=float, default=1e-4, help="CBP 替换率（仅 --method cbp 时生效）")
    p.add_argument("--out-dir", default=os.path.join(here, "results"), help="结果保存目录")
    p.add_argument("--no-plot", action="store_true", help="只保存 CSV，不画单次运行的图（批量实验用）")
    args = p.parse_args()

    print(f"torch {torch.__version__} | activation={args.activation} method={args.method} "
          f"lr={args.step_size} examples={args.examples:,} seed={args.seed}")
    rows = run(args)
    save_results(rows, args)


if __name__ == "__main__":
    main()
