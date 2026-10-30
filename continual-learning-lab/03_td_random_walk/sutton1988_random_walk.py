"""
实验 03：按 Sutton (1988)《Learning to Predict by the Methods of Temporal Differences》实现随机游走教学实验
=====================================================================================================

为什么做这个：
  - 思想地图里「TD 学习」节点的原始论文。Dohare 的可塑性论文出自同一个组（Sutton），
    TD 是整个强化学习的地基。
  - 论文的核心主张：用"下一步的预测"当目标（TD），比等到最终结果才更新（蒙特卡洛，λ=1）
    在有限数据下学得更好。这和直觉相反：监督学习不是一直用"真实结果"当标签吗？

环境：5 个非终止状态 B C D E F，从中间的 D 出发，每步等概率向左或向右，
      走到左端 A 结果为 0，走到右端 G 结果为 1。
      真实值（走到右端的概率）= 1/6, 2/6, 3/6, 4/6, 5/6。

学习：线性 TD(λ)，状态用独热编码，所以 P_t = w[当前状态]。论文的更新式（式 4）：
      Δw = Σ_t α (P_{t+1} − P_t) Σ_{k≤t} λ^{t−k} ∇P_k
      λ=0 → 纯 TD（只用下一步预测当目标）；λ=1 → 等价于蒙特卡洛（用最终结果当目标）。

生成对应论文三类比较的教学图（使用本实现的随机样本与收敛容差，不承诺逐项数值匹配）：
  图 3：每个训练集（10 条序列）反复呈现直到收敛，权重在整个训练集之后才更新。比较不同 λ 收敛后的误差。
  图 4：每个训练集只呈现一次，每条序列后更新，初值 0.5。看误差随 α、λ 的变化。
  图 5：对每个 λ 取最好的 α，误差随 λ 的变化。
所有误差 = 5 个状态预测与真实值之间的均方根误差（RMS），在 100 个训练集上平均。

只依赖 numpy + matplotlib，用系统 python3 运行：
    /usr/bin/python3 sutton1988_random_walk.py
"""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
TRUE_V = np.arange(1, 6) / 6.0
N_SETS, SEQ_PER_SET = 100, 10


def generate_sequence(rng):
    """返回 (访问过的非终止状态列表（0..4）, 最终结果 z)。起点 D = 下标 2。"""
    s, states = 2, [2]
    while True:
        s += 1 if rng.random() < 0.5 else -1
        if s < 0:
            return states, 0.0
        if s > 4:
            return states, 1.0
        states.append(s)


def sequence_delta(w, states, z, lam, alpha):
    """一条序列带来的权重增量 Δw（式 4）。序列内权重固定，所以预测都用同一个 w。"""
    dw, trace = np.zeros(5), np.zeros(5)
    for t, s in enumerate(states):
        trace = lam * trace
        trace[s] += 1.0                                   # 独热编码下 ∇P_t 就是 e_s
        p_next = z if t == len(states) - 1 else w[states[t + 1]]
        dw += alpha * (p_next - w[s]) * trace
    return dw


def rms(w):
    return np.sqrt(np.mean((w - TRUE_V) ** 2))


def make_training_sets(seed=0):
    rng = np.random.default_rng(seed)
    return [[generate_sequence(rng) for _ in range(SEQ_PER_SET)] for _ in range(N_SETS)]


def figure3(train_sets, lams, alpha=0.01, tol=1e-4):
    """反复呈现：每轮把 10 条序列的 Δw 累加起来，轮末一次性更新，直到权重不再变化。"""
    errs = []
    for lam in lams:
        e = []
        for seqs in train_sets:
            w = np.full(5, 0.5)
            for _ in range(10_000):
                dw = sum(sequence_delta(w, s, z, lam, alpha) for s, z in seqs)
                w += dw
                if np.abs(dw).max() < tol:
                    break
            e.append(rms(w))
        errs.append(np.mean(e))
        print(f"图3  λ={lam:.1f}  收敛后 RMS = {errs[-1]:.3f}")
    return np.array(errs)


def single_presentation(train_sets, lam, alpha):
    e = []
    for seqs in train_sets:
        w = np.full(5, 0.5)
        for s, z in seqs:
            w += sequence_delta(w, s, z, lam, alpha)
        e.append(rms(w))
    return np.mean(e)


def main():
    os.makedirs(OUT, exist_ok=True)
    train_sets = make_training_sets()

    # ---------- 图 3 ----------
    lams3 = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    err3 = figure3(train_sets, lams3)

    # ---------- 图 4 ----------
    alphas = np.round(np.arange(0, 0.65, 0.05), 2)
    lams4 = [0.0, 0.3, 0.8, 1.0]
    err4 = {lam: [single_presentation(train_sets, lam, a) for a in alphas] for lam in lams4}

    # ---------- 图 5 ----------
    lams5 = np.round(np.arange(0, 1.01, 0.1), 1)
    best5, best_alpha = [], []
    for lam in lams5:
        e = [single_presentation(train_sets, lam, a) for a in alphas]
        best5.append(min(e))
        best_alpha.append(alphas[int(np.argmin(e))])
        print(f"图5  λ={lam:.1f}  最佳 α={best_alpha[-1]:.2f}  RMS = {best5[-1]:.3f}")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    axes[0].plot(lams3, err3, "o-")
    axes[0].set(title="图 3：反复呈现直到收敛", xlabel="λ", ylabel="RMS 误差")
    for lam in lams4:
        e = np.array(err4[lam])
        keep = e < 0.7                                    # 和原图一样，太大的误差不画
        axes[1].plot(alphas[keep], e[keep], "o-", label=f"λ={lam}")
    axes[1].set(title="图 4：每个训练集只呈现一次", xlabel="α", ylabel="RMS 误差")
    axes[1].legend()
    axes[2].plot(lams5, best5, "o-")
    axes[2].set(title="图 5：每个 λ 取最佳 α", xlabel="λ", ylabel="RMS 误差")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.suptitle("Sutton (1988) 随机游走复现：100 个训练集 × 10 条序列")
    fig.tight_layout()
    path = os.path.join(OUT, "sutton1988_fig3_4_5.png")
    fig.savefig(path, dpi=120)
    print("图已保存：", path)

    with open(os.path.join(OUT, "numbers.md"), "w") as f:
        f.write("| λ | 图3 收敛后 RMS |\n|---|---|\n")
        f.writelines(f"| {l} | {e:.3f} |\n" for l, e in zip(lams3, err3))
        f.write("\n| λ | 图5 最佳 α | 图5 RMS |\n|---|---|---|\n")
        f.writelines(f"| {l} | {a} | {e:.3f} |\n" for l, a, e in zip(lams5, best_alpha, best5))


if __name__ == "__main__":
    main()
