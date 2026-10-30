"""阶段2:基线策略 —— 多因子打分 -> TopN 等权满仓,两个资金版本 × 多执行口径 × 多频率。

同时报排雷基线做对照(防止"基线太弱"造成模型看起来好的错觉)。
按滚动年份分别报告,便于后续与模型逐年对比。
"""
from __future__ import annotations
import os, sys, json, warnings
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

from afactor.config import LEDGER
from afactor.panel import Panel, TailPanel
from afactor import factors as FA, combine, strategy as ST

pd.set_option("display.width", 250)


def build_score(p, members, signs):
    fac = {n: FA.REGISTRY[n][1](p).reindex(index=p.dates, columns=p.codes)
           .replace([np.inf, -np.inf], np.nan) for n in members if n in FA.REGISTRY}
    return combine.combine_equal(combine.prepare(fac, signs, p["universe"]))


def main():
    FA.build_all()
    man = json.load(open(LEDGER / "comp_v5_members.json"))
    members, signs = man["members"], man["signs"]
    rows = []
    for mode in ["open10", "tail1450"]:
        p = TailPanel() if ST.EXEC_MODES[mode]["tail"] else Panel()
        score = build_score(p, members, signs)
        for version in ["V1000", "V100"]:
            bm = ST.benchmark(p, version, mode)
            print(f"\n{'='*128}\n{version} × {mode}({ST.EXEC_MODES[mode]['desc']})"
                  f"  池等权基准年化 {bm['ann_return']*100:.2f}%  波动 {bm['vol']*100:.1f}%\n{'='*128}")
            print("%-28s %9s %9s %8s %8s %9s %8s %8s %7s" % (
                "策略", "年化", "超额", "夏普", "波动", "最大回撤", "换手", "成本bp", "持仓"))
            plans = []
            for n in [10, 20, 50]:
                for reb in [1, 5]:
                    plans.append((f"基线 Top{n} {'日频' if reb==1 else '周频'}",
                                  dict(scheme="top", n=n, rebalance=reb)))
            plans.append(("对照 排雷剔30%", dict(scheme="exclude", exclude_frac=0.30, rebalance=5)))
            plans.append(("对照 排雷缓冲20/35", dict(scheme="exclude", buffer=(0.20, 0.35), rebalance=5)))
            for label, kw in plans:
                r = ST.run_strategy(score, p, version, mode, **kw)
                if not np.isfinite(r.get("ann_return", np.nan)):
                    continue
                rows.append({"version": version, "mode": mode, "plan": label,
                             **{k: v for k, v in r.items() if k != "nav"},
                             "excess": r["ann_return"] - bm["ann_return"],
                             "bench_ann": bm["ann_return"]})
                print("%-28s %8.2f%% %8.2f%% %8.2f %7.1f%% %8.1f%% %8.3f %8.2f %7.0f" % (
                    label, r["ann_return"]*100, (r["ann_return"]-bm["ann_return"])*100,
                    r["sharpe"], r["vol"]*100, r["max_dd"]*100, r["turnover"],
                    r["cost_bps"], r["holdings"]))
    df = pd.DataFrame(rows)
    df.to_csv(LEDGER / "baselines.csv", index=False)
    print(f"\n已写 {LEDGER/'baselines.csv'}")
    print("\n=== 每个版本的最佳基线(按夏普)===")
    for v in ["V1000", "V100"]:
        d = df[(df.version == v) & df.plan.str.startswith("基线")]
        if len(d):
            b = d.loc[d.sharpe.idxmax()]
            print(f"  {v}: {b['plan']} @ {b['mode']} -> 年化 {b['ann_return']*100:.2f}% "
                  f"超额 {b['excess']*100:+.2f}% 夏普 {b['sharpe']:.2f} 回撤 {b['max_dd']*100:.1f}%")


if __name__ == "__main__":
    main()
