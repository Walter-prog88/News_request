"""噪音门槛:在我这个搜索规模下,纯运气能刷出多高的 IR?

判据来自评审提示词任务一第 1 条:threshold = E[max of N] / sqrt(年数)。
本脚本在此基础上补三件复核指出的事:

1. N 从 search_space_manifest.json 读,不再手填(先前手数成 656 是把 baselines.csv
   里未截断干净的 rets 字段的内嵌换行也当成了数据行,实际 304)。
2. E[max] 的一阶渐近式在小 N 下系统性低估(N=11 时低 10.7%),方向是把门槛压低、
   对自己有利。这里同时给蒙特卡洛真值,判定一律用真值。
3. E[max] 只是"纯运气下最好那个约有 50% 概率达到"的水位,不是显著性门槛。
   怀疑立场的评审要的是族错误率 5%,即 max 分布的 95 分位
   Phi^-1(0.95 ** (1/N))。两条水位都报。

产出:research/ledger/noise_threshold.csv、noise_threshold_verdict.csv
"""
from __future__ import annotations
import os, sys, json, math
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from afactor.config import LEDGER

YEARS = 6.0
MC_DRAWS = 400_000
SEED = 20260901


def e_max_analytic(n: float) -> float:
    """N 个独立标准正态最大值期望的一阶渐近近似。"""
    if n <= 1:
        return 0.0
    a = np.sqrt(2.0 * np.log(n))
    return float(a - (np.log(np.log(n)) + np.log(4.0 * np.pi)) / (2.0 * a))


def e_max_mc(n: float, rng) -> float:
    """蒙特卡洛真值。用逆 CDF 采样:max 的 CDF 是 Phi(x)**n,故 M = Phi^-1(U**(1/n))。"""
    from scipy.stats import norm
    u = rng.random(MC_DRAWS)
    return float(norm.ppf(u ** (1.0 / n)).mean())


def q95_max(n: float) -> float:
    """max 分布的 95 分位:族错误率 5% 的门槛。"""
    from scipy.stats import norm
    return float(norm.ppf(0.95 ** (1.0 / n)))


def e_max_exact(n: float) -> float:
    """E[max of N] 的数值积分值(比蒙特卡洛稳,用于反解临界 N)。

    E[max] = ∫ x · n · Phi(x)^(n-1) · phi(x) dx
    """
    from scipy.stats import norm
    from scipy.integrate import quad
    f = lambda x: x * n * norm.cdf(x) ** (n - 1.0) * norm.pdf(x)
    v, _ = quad(f, -8.0, 12.0, limit=400)
    return float(v)


def critical_n(t_obs: float, lo: float = 1.5, hi: float = 1e9) -> float:
    """反解:族规模要多大,纯运气的 E[max] 才追平这个观测到的 t?

    这是比"选哪一档 N"更好的呈现方式——把 N 的选择权交给评审:
    "这一格要在多大的搜索族里才会失效",而不是"我认为 N 该取多少"。
    """
    if not np.isfinite(t_obs) or t_obs <= e_max_exact(lo):
        return float("nan")
    a, b = lo, hi
    for _ in range(80):
        m = np.sqrt(a * b)                       # 在对数尺度上二分
        if e_max_exact(m) < t_obs:
            a = m
        else:
            b = m
    return float(np.sqrt(a * b))


def main():
    man = json.load(open(LEDGER / "search_space_manifest.json"))
    cands = [
        ("N_eff    有效独立试验数(仅作下界)", float(man["N_eff"]["value"])),
        ("N_strategy 策略层回测配置数", float(man["N_strategy"]["value"])),
        ("N_trials   因子层试验行数", float(man["N_trials"]["value"])),
    ]
    rng = np.random.default_rng(SEED)

    rows = []
    for label, n in cands:
        ea, em, q95 = e_max_analytic(n), e_max_mc(n, rng), q95_max(n)
        rows.append(dict(
            scope=label, N=n,
            E_max_analytic=round(ea, 4), E_max_mc=round(em, 4),
            approx_bias_pct=round((ea / em - 1) * 100, 2),
            q95_max=round(q95, 4),
            thr_IR_50pct=round(em / np.sqrt(YEARS), 4),      # 判定用这一列
            thr_IR_fwer5=round(q95 / np.sqrt(YEARS), 4),     # 更严的那一列
        ))
    th = pd.DataFrame(rows)

    sm = pd.read_csv(LEDGER / "ir_analysis_summary.csv")
    sm = sm[sm.strategy == "排雷缓冲20/35"]

    out = []
    for _, r in sm.iterrows():
        d = dict(口径=r["mode"], 基准=r["benchmark"],
                 IR=round(r.IR, 3), t_naive=round(r.t_naive, 2),
                 t_NW=round(r.t_newey_west, 2))
        for label, n in cands:
            key = label.split()[0]
            d[f"{key}_50%"] = "过" if r.IR > e_max_mc(n, np.random.default_rng(SEED)) / np.sqrt(YEARS) else "不过"
            d[f"{key}_FWER5%"] = "过" if r.IR > q95_max(n) / np.sqrt(YEARS) else "不过"
        out.append(d)
    res = pd.DataFrame(out)

    pd.set_option("display.width", 200)
    print("=" * 110)
    print(f"噪音门槛(年数 = {YEARS}, 蒙特卡洛 {MC_DRAWS:,} 次, seed={SEED})")
    print("=" * 110)
    print(th.to_string(index=False))
    print("\n注:判定一律用 E_max_mc(真值);analytic 列只用来显示渐近近似低估了多少。")
    print("\n" + "=" * 110)
    print("实测 IR vs 门槛 —— 排雷缓冲20/35")
    print("=" * 110)
    print(res.to_string(index=False))

    # ---- 临界 N:每一格要在多大的族里才会失效 ----
    crit = []
    for _, r in sm.iterrows():
        crit.append(dict(
            口径=r["mode"], 基准=r["benchmark"],
            t_naive=round(r.t_naive, 3), t_NW_q10=round(r.t_newey_west, 3),
            临界N_按t_naive=round(critical_n(r.t_naive), 0),
            临界N_按t_NW=round(critical_n(r.t_newey_west), 0),
        ))
    cn = pd.DataFrame(crit)
    print("\n" + "=" * 110)
    print("临界 N —— 族规模要大到多少,纯运气的 E[max] 才追平这一格的 t")
    print("=" * 110)
    print(cn.to_string(index=False))
    print(f"\n对照:实测族规模 N_strategy={int(man['N_strategy']['value'])}、"
          f"N_trials={int(man['N_trials']['value'])}。")
    print("门槛按 sqrt(2 ln N) 走,N 放大 10 倍只把门槛推高约 0.6——'N 取哪一档'远没有看上去承重。")

    th.to_csv(LEDGER / "noise_threshold.csv", index=False)
    res.to_csv(LEDGER / "noise_threshold_verdict.csv", index=False)
    cn.to_csv(LEDGER / "noise_threshold_critical_n.csv", index=False)
    print(f"\n已写 {LEDGER/'noise_threshold.csv'}")
    print(f"已写 {LEDGER/'noise_threshold_verdict.csv'}")
    print(f"已写 {LEDGER/'noise_threshold_critical_n.csv'}")


if __name__ == "__main__":
    main()
