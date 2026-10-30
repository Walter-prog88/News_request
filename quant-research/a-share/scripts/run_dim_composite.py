"""维度层组合:把 COMP_v5 的 25 个成员聚成 k 个真实维度后再合成。

对照:等权基线 vs 维度层(k=4,5,6,8) × (维度间等权 | 风险平价)。
所有 k 一并报告,不做"挑最好的 k"以免过拟合。
"""
from __future__ import annotations
import os, sys, json, warnings
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

from afactor.config import LEDGER
from afactor.panel import Panel, TailPanel
from afactor.fast_eval import EvalContext, evaluate, IC_DIR
from afactor import factors as FA, combine, backtest as BT


def main():
    FA.build_all()
    man = json.load(open(LEDGER / "comp_v5_members.json"))
    members, signs = man["members"], man["signs"]
    eff = combine.effective_dimensions(members, signs, "open")
    print(f"COMP_v5 成员 {len(members)} 个 -> 有效独立维度 {eff:.1f}")

    ics = {}
    for n in members:
        f = IC_DIR / f"{n}__open.parquet"
        if f.exists():
            ics[n] = pd.read_parquet(f)["h1"] * np.sign(signs.get(n, 1))

    rows = []
    for mode in ["open", "tail"]:
        p = TailPanel() if mode == "tail" else Panel()
        ctx = EvalContext(p, mode, (1, 5))
        u = p["universe"]
        fac = {n: FA.REGISTRY[n][1](p).reindex(index=p.dates, columns=p.codes)
               .replace([np.inf, -np.inf], np.nan) for n in members if n in FA.REGISTRY}
        z = combine.prepare(fac, signs, u)
        bench = BT.benchmark_nav(p, mode)
        b_ann = bench.iloc[-1] ** (242 / len(bench)) - 1

        print(f"\n{'='*118}\n{mode} 口径(基准年化 {b_ann*100:.2f}%)\n{'='*118}")
        print("%-26s %8s %7s %9s %8s %8s %9s %8s %8s" % (
            "方案", "IC", "t", "最差窗", "单调", "多头组", "回测超额", "夏普", "换手"))
        variants = [("基线 等权25成员", combine.combine_equal(z), None)]
        for k in (4, 5, 6, 8):
            g = combine.dimension_groups(members, signs, "open", k=k)
            variants.append((f"维度层 k={k} 等权",
                             combine.combine_by_dimension(z, g, "equal"), g))
            variants.append((f"维度层 k={k} 风险平价",
                             combine.combine_by_dimension(z, g, "riskparity", ics), g))
        for nm, F, g in variants:
            r = evaluate(f"DIM_{nm.replace(' ', '_')}__{mode}", F.to_numpy(np.float32), ctx,
                         family="composite", note="维度层组合实验",
                         ledger_path=LEDGER / "trials_dim.jsonl")
            bt = BT.run_standard(F, p, sign=1, mode=mode)
            rows.append({"mode": mode, "scheme": nm, "ic": r["ic_mean"], "t": r["ic_t"],
                         "worst_win": r["worst_window_abs_ic"], "mono": r["mono_spearman"],
                         "long_group": r["long_group"], "excess": bt["ann_return"] - b_ann,
                         "sharpe": bt["sharpe"], "turnover": bt["avg_turnover"],
                         "max_dd": bt["max_dd"]})
            print("%-26s %8.4f %7.1f %9.4f %8.2f %8d %8.2f%% %8.2f %8.3f" % (
                nm, r["ic_mean"], r["ic_t"], r["worst_window_abs_ic"], r["mono_spearman"],
                r["long_group"], (bt["ann_return"] - b_ann) * 100, bt["sharpe"], bt["avg_turnover"]))
        if mode == "open":
            g5 = combine.dimension_groups(members, signs, "open", k=5)
            print("\n  === k=5 的维度构成 ===")
            for cid, mem in sorted(g5.items(), key=lambda x: -len(x[1])):
                print(f"   维度{cid} ({len(mem)}个): {', '.join(mem[:6])}{' ...' if len(mem) > 6 else ''}")
            json.dump({str(k): v for k, v in g5.items()},
                      open(LEDGER / "dim_groups_k5.json", "w"), ensure_ascii=False, indent=1)
    pd.DataFrame(rows).to_csv(LEDGER / "dim_composite.csv", index=False)
    print(f"\n已写 {LEDGER/'dim_composite.csv'}")


if __name__ == "__main__":
    main()
