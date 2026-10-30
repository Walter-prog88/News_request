"""聚类 -> 选簇代表 -> 构建组合因子 -> 组合因子作为独立因子重新检验。"""
from __future__ import annotations
import os, sys, warnings, json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

from afactor.config import LEDGER
from afactor.panel import Panel
from afactor.fast_eval import EvalContext, evaluate
from afactor import factors as FA, ops, screen, combine

CLUSTER_THR = 0.6
MIN_RETENTION = 0.5


def main():
    p = Panel()
    ctx = EvalContext(p, "open", (1, 5, 10))
    FA.build_all()

    df = screen.load_ledger("trials_[0-9]*.jsonl")
    g2 = screen.gate_consistency(screen.gate_stats(df))
    rid = pd.read_csv(LEDGER / "residual_ic.csv")
    g2 = g2.merge(rid, on="factor", how="left")
    g2["retention"] = (g2["resid_ic"].abs() / g2["ic_mean"].abs())
    names = [n for n in g2["factor"] if n in FA.REGISTRY]
    signs = dict(zip(g2["factor"], np.sign(g2["ic_mean"])))
    print(f"候选 {len(names)} 个,开始聚类(IC序列相关 >= {CLUSTER_THR} 视为同簇)")

    clusters = combine.cluster_by_ic(names, "open", thr=CLUSTER_THR, signs=signs)
    print(f"共 {len(clusters)} 个信息簇;簇规模分布 {sorted((len(v) for v in clusters.values()), reverse=True)[:15]}")

    # 每簇代表:最差滚动窗口 |IC| 最高者
    info = g2.set_index("factor")
    reps, cluster_tbl = {}, []
    for cid, mem in clusters.items():
        sub = info.loc[[m for m in mem if m in info.index]]
        if sub.empty:
            continue
        best = sub["worst_window_abs_ic"].idxmax()
        reps[cid] = best
        cluster_tbl.append({"cluster": cid, "size": len(mem), "rep": best,
                            "family": sub.loc[best, "family"],
                            "ic": sub.loc[best, "ic_mean"], "resid_ic": sub.loc[best, "resid_ic"],
                            "retention": sub.loc[best, "retention"],
                            "members": ",".join(mem[:12])})
    ct = pd.DataFrame(cluster_tbl).sort_values("size", ascending=False)
    ct.to_csv(LEDGER / "clusters.csv", index=False)
    print("\n=== 信息簇代表(前20)===")
    print(ct[["cluster", "size", "rep", "family", "ic", "resid_ic", "retention"]].head(20)
          .to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # 构建因子值
    rep_names = list(reps.values())
    fac = {}
    for n in rep_names:
        fam, fn, note = FA.REGISTRY[n]
        fac[n] = fn(p).reindex(index=p.dates, columns=p.codes).replace([np.inf, -np.inf], np.nan)
    univ = p["universe"]
    z = combine.prepare(fac, {n: signs[n] for n in rep_names}, univ)

    indep = [n for n in rep_names if (info.loc[n, "retention"] or 0) >= MIN_RETENTION]
    strong = info.loc[rep_names].sort_values("worst_window_abs_ic", ascending=False)
    top8 = strong.head(8).index.tolist()
    top8_indep = strong.loc[[n for n in strong.index if n in indep]].head(8).index.tolist()

    comps = {
        "COMP_equal_allclusters": combine.combine_equal(z),
        "COMP_equal_independent": combine.combine_equal({n: z[n] for n in indep}),
        "COMP_equal_top8": combine.combine_equal({n: z[n] for n in top8}),
        "COMP_equal_top8_indep": combine.combine_equal({n: z[n] for n in top8_indep}),
        "COMP_icweighted": combine.combine_ic_weighted(z, "open"),
        "COMP_icweighted_indep": combine.combine_ic_weighted({n: z[n] for n in indep}, "open"),
    }
    # 组合因子的规模正交化版本
    size = ops.cs_rank(ops.signed_log(p["adv20"]))
    for k in list(comps):
        comps[k + "_resid"] = ops.cs_neutralize_1d(comps[k], size)

    print(f"\n=== 组合因子检验({len(comps)} 个,作为独立因子)===")
    rows = []
    for k, f in comps.items():
        r = evaluate(k, f.reindex(index=p.dates, columns=p.codes).to_numpy(np.float32), ctx,
                     family="composite", note=f"组合因子", ledger_path=LEDGER / "trials_composite.jsonl")
        rows.append(r)
        print("%-30s IC=%7.4f t=%6.1f 年一致=%.2f 窗一致=%.2f 最差窗=%.4f 单调=%6.2f 多头组=%2d 多头超额=%.3f 换手=%.2f h5IC=%.4f" % (
            k, r["ic_mean"], r["ic_t"], r["year_sign_consistency"] or 0, r["window_sign_consistency"] or 0,
            r["worst_window_abs_ic"] or 0, r["mono_spearman"] or 0, r["long_group"],
            r["long_excess_ann"], r["long_turnover"] or 0, r["h5_ic_mean"] or 0))
    pd.DataFrame(rows).to_csv(LEDGER / "composites.csv", index=False)
    best = max(rows, key=lambda r: abs(r["ic_mean"]))
    print(f"\n最强组合:{best['factor']}  分层超额 {best['q_excess']}")
    print(f"  分年 IC {json.dumps({k: round(v,4) for k,v in best['ic_by_year'].items()})}")
    print(f"  滚动窗口 {json.dumps({k: round(v,4) for k,v in best['ic_windows'].items()})}")
    json.dump({"reps": reps, "indep": indep, "top8": top8, "top8_indep": top8_indep},
              open(LEDGER / "composite_members.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
