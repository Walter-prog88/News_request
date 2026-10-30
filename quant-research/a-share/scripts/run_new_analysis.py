"""六个实验的终局分析。

A. 新族(market/condvp/statind)漏斗 + 新维度判定:过关新因子的 IC 序列与既有
   19 个簇代表逐一求相关,max|ρ|<0.5 者标记为"候选新维度"。
B. h5 中周期重筛:open_h5 账本走同一漏斗(用 NW t),与 h1 短名单对比;
   对 h5 头部因子做 5 日调仓回测(对比 1 日调仓的换手与净超额)。
"""
from __future__ import annotations
import os, sys, json, warnings
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

from afactor.config import LEDGER
from afactor.panel import Panel
from afactor.fast_eval import IC_DIR
from afactor import factors as FA, screen, backtest as BT

pd.set_option("display.width", 240)


def ic_corr_vs_reps(names: list[str], mode: str, reps: list[str], rep_mode: str = "open"):
    """每个新因子 IC 序列与既有簇代表 IC 序列的最大|相关|及对应代表。"""
    repM = {}
    for r in reps:
        f = IC_DIR / f"{r}__{rep_mode}.parquet"
        if f.exists():
            repM[r] = pd.read_parquet(f)["h1"]
    repDF = pd.DataFrame(repM)
    out = []
    for n in names:
        f = IC_DIR / f"{n}__{mode}.parquet"
        if not f.exists():
            out.append((n, np.nan, ""));  continue
        s = pd.read_parquet(f)["h1"]
        c = repDF.corrwith(s).abs()
        out.append((n, float(c.max()), str(c.idxmax())))
    return pd.DataFrame(out, columns=["factor", "max_corr_vs_reps", "nearest_rep"])


def analyse_new_families():
    print("=" * 100)
    print("A. 新因子族漏斗与新维度判定")
    print("=" * 100)
    df = screen.load_ledger("trials_[0-9]*.jsonl")
    ct = pd.read_csv(LEDGER / "clusters_open.csv")
    reps = ct["rep"].tolist()
    results = {}
    for mode in ["open", "tail"]:
        d = df[(df["mode"] == mode) & (df["family"].isin(["market", "condvp", "statind"]))]
        g1 = screen.gate_stats(d)
        g2 = screen.gate_consistency(g1)
        print(f"\n[{mode}] 新族试验 {len(d)} -> IC/t 达标 {len(g1)} -> 方向一致 {len(g2)}")
        print(g2.groupby("family").size().to_dict())
        if g2.empty:
            continue
        cc = ic_corr_vs_reps(g2["factor"].tolist(), mode, reps)
        g2 = g2.merge(cc, on="factor")
        novel = g2[g2["max_corr_vs_reps"] < 0.5].sort_values("worst_window_abs_ic", ascending=False)
        absorbed = g2[g2["max_corr_vs_reps"] >= 0.5]
        print(f"与既有 19 个维度代表 max|ρ|<0.5 的『候选新维度』: {len(novel)} 个;被既有维度吸收: {len(absorbed)} 个")
        cols = ["factor", "family", "ic_mean", "ic_t", "worst_window_abs_ic",
                "mono_spearman", "long_group", "long_excess_ann", "long_turnover",
                "max_corr_vs_reps", "nearest_rep"]
        if len(novel):
            print("\n候选新维度(前 15):")
            print(novel[cols].head(15).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        g2.to_csv(LEDGER / f"new_families_{mode}.csv", index=False)
        results[mode] = g2
    return results


def analyse_h5():
    print("\n" + "=" * 100)
    print("B. h5 中周期重筛(NW 修正 t)")
    print("=" * 100)
    df = screen.load_ledger("trials_[0-9]*.jsonl")
    d = df[df["mode"] == "open_h5"].copy()
    if d.empty:
        print("open_h5 账本为空,跳过");  return
    # 用 NW t 替换普通 t 作为门槛(h5 重叠收益)
    d["ic_t_plain"] = d["ic_t"]
    d["ic_t"] = d["ic_t_nw"].where(d["ic_t_nw"].notna(), d["ic_t"])
    g1 = screen.gate_stats(d, ic_thr=0.04, t_thr=3.0)     # h5 IC 天然更高,门槛相应提高
    g2 = screen.gate_consistency(g1)
    g3, _ = screen.dedupe_by_ic_corr(g2, "open_h5", thr=0.7)
    print(f"试验 {len(d)} -> |IC|>=0.04 且 NW|t|>=3: {len(g1)} -> 方向一致 {len(g2)} -> 去重 {len(g3)}")
    cols = ["factor", "family", "ic_mean", "ic_t", "ic_t_plain", "worst_window_abs_ic",
            "mono_spearman", "long_group", "long_excess_ann", "long_turnover"]
    top = g3.sort_values("worst_window_abs_ic", ascending=False)
    print("\nh5 去重后前 15(ic_mean 为 5 日 Rank IC,ic_t 为 NW 修正):")
    print(top[cols].head(15).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    top.to_csv(LEDGER / "shortlist_h5.csv", index=False)

    # h1 短名单对比
    h1 = pd.read_csv(LEDGER / "shortlist_open.csv")["factor"].head(15).tolist()
    h5names = top["factor"].head(15).tolist()
    print(f"\n与 h1 短名单前15的交集: {sorted(set(h1) & set(h5names))}")
    print(f"h5 独有: {sorted(set(h5names) - set(h1))[:10]}")

    # 5日调仓回测:h5 头部 3 个 + COMP_final
    print("\n5 日调仓 vs 1 日调仓(排雷剔最差30%):")
    p = Panel()
    FA.build_all()
    man = json.load(open(LEDGER / "final_manifest.json"))
    from afactor import combine
    fac = {n: FA.REGISTRY[n][1](p).reindex(index=p.dates, columns=p.codes)
           .replace([np.inf, -np.inf], np.nan) for n in man["composite_members"] if n in FA.REGISTRY}
    comp = combine.combine_equal(combine.prepare(fac, man["composite_signs"], p["universe"]))
    bench = BT.benchmark_nav(p, "open")
    b_ann = bench.iloc[-1] ** (242 / len(bench)) - 1
    targets = {"COMP_final": (comp, 1)}
    led = d.set_index("factor")
    for n in top["factor"].head(3):
        if n in FA.REGISTRY:
            targets[n] = (FA.REGISTRY[n][1](p).reindex(index=p.dates, columns=p.codes)
                          .replace([np.inf, -np.inf], np.nan),
                          int(np.sign(led.loc[n, "ic_mean"])) or 1)
    rows = []
    print("%-30s %6s %9s %9s %8s %8s" % ("factor", "调仓", "年化", "超额", "夏普", "日换手"))
    for name, (f, sgn) in targets.items():
        for k in (1, 5):
            r = BT.run_backtest(f, p, sign=sgn, mode="open", scheme="exclude",
                                top_frac=0.30, rebalance_every=k)
            rows.append({"factor": name, "rebalance": k, **{x: v for x, v in r.items() if x != "nav"}})
            print("%-30s %6d %8.2f%% %8.2f%% %8.2f %8.3f" % (
                name if k == 1 else "", k, r["ann_return"] * 100,
                (r["ann_return"] - b_ann) * 100, r["sharpe"], r["avg_turnover"]))
    pd.DataFrame(rows).to_csv(LEDGER / "h5_rebalance.csv", index=False)


if __name__ == "__main__":
    analyse_new_families()
    analyse_h5()
