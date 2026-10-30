"""对最终短名单与组合因子做 TWAP 成交模拟回测(1000万资金、10%参与率、1bp双边+印花税)。"""
from __future__ import annotations
import os, sys, json, warnings
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

from afactor.config import LEDGER
from afactor.panel import Panel, TailPanel
from afactor import factors as FA, combine, backtest as BT, screen

MANIFEST = LEDGER / "final_manifest.json"


def main():
    man = json.load(open(MANIFEST))
    FA.build_all()
    rows = []
    for mode in man["modes"]:
        p = TailPanel() if mode == "tail" else Panel()
        bench = BT.benchmark_nav(p, mode)
        bench_ann = bench.iloc[-1] ** (242 / len(bench)) - 1
        print(f"\n=== {mode} 口径 ===  决策池等权基准年化 {bench_ann*100:.2f}%")
        targets = {}
        for n in man["single_factors"]:
            if n in FA.REGISTRY:
                targets[n] = FA.REGISTRY[n][1](p).reindex(index=p.dates, columns=p.codes) \
                    .replace([np.inf, -np.inf], np.nan)
        mem = [n for n in man["composite_members"] if n in FA.REGISTRY]
        if mem:
            fac = {n: FA.REGISTRY[n][1](p).reindex(index=p.dates, columns=p.codes)
                   .replace([np.inf, -np.inf], np.nan) for n in mem}
            z = combine.prepare(fac, man["composite_signs"], p["universe"])
            targets["COMP_final"] = combine.combine_equal(z)

        led = screen.load_ledger("trials*.jsonl")
        led = led[led["mode"] == mode].set_index("factor")
        print("%-30s %4s %9s %9s %8s %8s %9s %8s %7s" % (
            "factor", "N", "年化", "超额", "夏普", "最大回撤", "日换手", "成本bp", "持仓数"))
        for name, f in targets.items():
            sgn = 1
            if name in led.index:
                sgn = int(np.sign(led.loc[name, "ic_mean"])) or 1
            for top_n in (50, 100, 200):
                r = BT.run_backtest(f, p, top_n=top_n, sign=sgn, mode=mode)
                b = bench.iloc[:r["n_days"]]
                b_ann = b.iloc[-1] ** (242 / len(b)) - 1
                rows.append({"factor": name, "mode": mode, "top_n": top_n,
                             "ann": r["ann_return"], "excess": r["ann_return"] - b_ann,
                             "sharpe": r["sharpe"], "max_dd": r["max_dd"],
                             "turnover": r["avg_turnover"], "cost_bps": r["avg_cost_bps"],
                             "holdings": r["avg_holdings"]})
                print("%-30s %4d %8.2f%% %8.2f%% %8.2f %7.1f%% %8.2f %8.2f %7.0f" % (
                    name if top_n == 50 else "", top_n, r["ann_return"] * 100,
                    (r["ann_return"] - b_ann) * 100, r["sharpe"], r["max_dd"] * 100,
                    r["avg_turnover"], r["avg_cost_bps"], r["avg_holdings"]))
    out = pd.DataFrame(rows)
    out.to_csv(LEDGER / "backtest.csv", index=False)
    print(f"\n已写入 {LEDGER/'backtest.csv'}")


if __name__ == "__main__":
    main()
