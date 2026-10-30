"""因子用法对比:选优(top-N) vs 排雷(剔除最差) vs 分层最优区段(band)。

分层测试显示多数因子的预测力集中在最差组,本脚本用真实成交与成本口径验证
"排雷"是否才是这些因子的正确用法。
"""
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
        led = screen.load_ledger("trials*.jsonl")
        led = led[led["mode"] == mode].set_index("factor")

        mem = [n for n in man["composite_members"] if n in FA.REGISTRY]
        fac = {n: FA.REGISTRY[n][1](p).reindex(index=p.dates, columns=p.codes)
               .replace([np.inf, -np.inf], np.nan) for n in mem}
        z = combine.prepare(fac, man["composite_signs"], p["universe"])
        targets = {"COMP_final": combine.combine_equal(z)}
        for n in ["id_up_amt_share_m60", "id_ret_intraday_m60", "ivol_20", "amt_vol_20"]:
            if n in FA.REGISTRY:
                targets[n] = FA.REGISTRY[n][1](p).reindex(index=p.dates, columns=p.codes) \
                    .replace([np.inf, -np.inf], np.nan)

        schemes = [
            ("选优 top100", dict(scheme="top", top_n=100)),
            ("选优 top10%", dict(scheme="top", top_n=None, top_frac=0.10)),
            ("排雷 剔最差10%", dict(scheme="exclude", top_frac=0.10)),
            ("排雷 剔最差20%", dict(scheme="exclude", top_frac=0.20)),
            ("排雷 剔最差30%", dict(scheme="exclude", top_frac=0.30)),
            ("区段 60-90%", dict(scheme="band", band=(0.60, 0.90))),
            ("区段 50-80%", dict(scheme="band", band=(0.50, 0.80))),
        ]
        print(f"\n{'='*112}\n{mode} 口径   决策池等权基准年化 "
              f"{(bench.iloc[-1]**(242/len(bench))-1)*100:.2f}%\n{'='*112}")
        print("%-22s %-16s %9s %9s %8s %9s %8s %8s %7s" % (
            "factor", "用法", "年化", "超额", "夏普", "最大回撤", "日换手", "成本bp", "持仓"))
        for name, f in targets.items():
            sgn = int(np.sign(led.loc[name, "ic_mean"])) if name in led.index else 1
            sgn = sgn or 1
            for label, kw in schemes:
                r = BT.run_backtest(f, p, sign=sgn, mode=mode, **kw)
                b = bench.iloc[:r["n_days"]]
                b_ann = b.iloc[-1] ** (242 / len(b)) - 1
                rows.append({"factor": name, "mode": mode, "scheme": label,
                             "ann": r["ann_return"], "excess": r["ann_return"] - b_ann,
                             "sharpe": r["sharpe"], "max_dd": r["max_dd"],
                             "turnover": r["avg_turnover"], "cost_bps": r["avg_cost_bps"],
                             "holdings": r["avg_holdings"]})
                print("%-22s %-16s %8.2f%% %8.2f%% %8.2f %8.1f%% %8.2f %8.2f %7.0f" % (
                    name if label == schemes[0][0] else "", label, r["ann_return"] * 100,
                    (r["ann_return"] - b_ann) * 100, r["sharpe"], r["max_dd"] * 100,
                    r["avg_turnover"], r["avg_cost_bps"], r["avg_holdings"]))
    pd.DataFrame(rows).to_csv(LEDGER / "usage_modes.csv", index=False)
    print(f"\n已写入 {LEDGER/'usage_modes.csv'}")


if __name__ == "__main__":
    main()
