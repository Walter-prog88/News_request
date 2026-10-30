"""特征集消融的配对定性:中段 IC 因子在**选优框架**下到底赢没赢。

背景:
  归因显示 Spearman(模型置换重要性, 原始|IC|) = 0.079,扛 70% 权重的两个特征
  IC 只有 0.0223 / 0.0241,都在 0.025 门槛下面 —— 门槛可能对模型层是错的标准。
  据此跑过 C(高IC)/ D(中段IC)/ E(多样性)三组消融,D 组三年均超额 +14.74%
  vs C 组 +6.74%,看起来 D 赢 8pp。**但相位极差最大 28.6pp,比差距还大三倍,未能定性。**

  另注:2026-08-25 的中段因子组合实验(README §13)证明中段因子确有真实且独立的信息
  (对 COMP_v5 正交后残差 IC t=3.0~3.4,三切法全显著),但在**排雷框架**下变不成钱。
  本实验测的是**选优框架**(Top20 周频),两者赚的钱来源不同,必须分开定性。

为什么之前定不了性:
  周频调仓有 5 个合法相位。同一份打分,只是换个起始日,逐年超额能摆 23~42pp。
  "D 组均值比 C 组高 8pp"是从两个各自 20~29pp 宽的分布里各抽一个点来比,没有判据意义。

本实验怎么定性:
  **配对比较。** 相位锚在 date_mask 起点,所以同一年、同一相位对 C/D/E 是**同一套调仓日历**,
  市场涨跌、踩没踩到暴跌这些共同因素在配对差里抵消。
  3 个 OOS 年 × 5 个相位 = **15 对**,对 (D−C) 和 (E−C) 做配对 t 检验。

  另加一个**宽分散对照**(排雷缓冲 20/35,持仓约 2900 只):相位噪声在这里被稀释到可忽略,
  若 D 在宽分散上也赢,那才是稳的。

  模型打分面板已由 `run_featureset_ablation.py` 存盘(scores/AB{C,D,E}_gbt_open_h5_{年}.parquet),
  **本实验不重训模型**,只重跑回测取逐相位结果。

产出:research/ledger/ablation_paired.csv
"""
from __future__ import annotations
import os, sys, warnings
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

from afactor.config import LEDGER
from afactor.panel import Panel
from afactor import factors as FA, strategy as ST

SETS = {"C_高IC": "ABC", "D_中段IC": "ABD", "E_多样性": "ABE"}
YEARS = [2023, 2024, 2025]
MODE, HORIZON, EXEC = "open", 5, "open10"
VERSION = "V1000"
SCORE_DIR = LEDGER / "scores"


def _load(tag, year, p):
    f = SCORE_DIR / f"{tag}_gbt_{MODE}_h{HORIZON}_{year}.parquet"
    if not f.exists():
        return None
    sc = pd.read_parquet(f)
    full = pd.DataFrame(np.nan, index=p.dates, columns=p.codes, dtype=np.float32)
    idx = sc.index.intersection(p.dates)
    full.loc[idx, :] = sc.reindex(index=idx, columns=p.codes).to_numpy(np.float32)
    return full


def main():
    p = Panel(); FA.build_all()
    years = np.asarray(p.dates.year)

    rows, wide = [], []
    for label, tag in SETS.items():
        for y in YEARS:
            sc = _load(tag, y, p)
            if sc is None:
                print(f"  [缺] {tag} {y}")
                continue
            te = years == y
            bm = ST.benchmark(p, VERSION, EXEC, date_mask=te)["ann_return"]

            # 选优:Top20 周频,取逐相位
            r = ST.run_strategy(sc, p, VERSION, EXEC, date_mask=te,
                                scheme="top", n=20, rebalance=5, avg_phases=True)
            pa = r.get("phase_ann") or [r["ann_return"]]
            for i, a in enumerate(pa):
                rows.append({"组": label, "年": y, "相位": i,
                             "年化": a, "基准": bm, "超额": a - bm})

            # 宽分散对照:排雷缓冲 20/35(持仓约 2900 只,相位噪声被稀释)
            rw = ST.run_strategy(sc, p, VERSION, EXEC, date_mask=te,
                                 scheme="exclude", exclude_frac=0.30,
                                 buffer=(0.20, 0.35), rebalance=5, avg_phases=True)
            wide.append({"组": label, "年": y, "年化": rw["ann_return"], "基准": bm,
                         "超额": rw["ann_return"] - bm, "夏普": rw["sharpe"],
                         "换手": rw.get("avg_turnover"), "持仓": rw.get("avg_holdings")})
            print(f"  {label:10s} {y}  Top20 相位极差 {max(pa)-min(pa):.3f} | "
                  f"均值超额 {np.mean(pa)-bm:+.4f} | 宽分散超额 {wide[-1]['超额']:+.4f}",
                  flush=True)

    df = pd.DataFrame(rows); dw = pd.DataFrame(wide)
    df.to_csv(LEDGER / "ablation_paired.csv", index=False)
    dw.to_csv(LEDGER / "ablation_paired_wide.csv", index=False)
    pd.set_option("display.width", 220)

    print("\n" + "=" * 104)
    print("=== 选优 Top20 周频:逐(年, 相位)超额 ===")
    piv = df.pivot_table(index=["年", "相位"], columns="组", values="超额")
    print((piv * 100).to_string(float_format=lambda x: f"{x:+7.2f}"))

    print("\n--- 相位噪声有多大 ---")
    for y in YEARS:
        s = df[df.年 == y]
        for g in SETS:
            v = s[s.组 == g].超额
            if len(v):
                print(f"  {y} {g:10s} 5 个相位: "
                      + " ".join(f"{x*100:+6.2f}" for x in v)
                      + f"  极差 {(v.max()-v.min())*100:5.2f}pp")

    print("\n" + "=" * 104)
    print("=== 配对检验(同年同相位一对一,15 对)===")
    base = piv["C_高IC"]
    for g in ["D_中段IC", "E_多样性"]:
        if g not in piv:
            continue
        d = (piv[g] - base).dropna()
        n = len(d); mu = d.mean(); sd = d.std(ddof=1)
        t = mu / sd * np.sqrt(n) if sd > 0 else np.nan
        wins = int((d > 0).sum())
        print(f"\n  {g} − C_高IC:")
        print(f"    配对均值差 {mu*100:+.2f}pp | 配对标准差 {sd*100:.2f}pp | "
              f"t = {t:+.2f} (n={n}) | 赢 {wins}/{n} 对")
        for y in YEARS:
            dy = (piv[g] - base).loc[y].dropna()
            print(f"      {y}: 均值差 {dy.mean()*100:+6.2f}pp  赢 {int((dy>0).sum())}/{len(dy)}")
        verdict = ("D 显著赢" if t > 2.13 else "C 显著赢" if t < -2.13 else "**分不出**")
        print(f"    判定(|t|>2.13 为 95% 显著,df=14):{verdict}")

    print("\n" + "=" * 104)
    print("=== 宽分散对照(排雷缓冲 20/35,持仓约 2900 只,相位噪声可忽略)===")
    print(dw.pivot_table(index="年", columns="组", values="超额")
          .mul(100).to_string(float_format=lambda x: f"{x:+7.2f}"))
    print("\n三年均值:")
    print(dw.groupby("组").agg(超额=("超额", "mean"), 夏普=("夏普", "mean"),
                               换手=("换手", "mean"), 持仓=("持仓", "mean"))
          .to_string(float_format=lambda x: f"{x:.4f}"))
    print("\n产出:ablation_paired.csv / ablation_paired_wide.csv")


if __name__ == "__main__":
    main()
