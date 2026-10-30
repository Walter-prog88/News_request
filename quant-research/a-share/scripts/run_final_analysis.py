"""终局分析:双口径漏斗 -> 残差IC -> 聚类 -> 组合因子 -> 多重检验去偏 -> 冻结清单。

产出(research/ledger/):
  funnel.csv              各口径漏斗各关存活数
  factor_summary_all.csv  全部试验的关键指标
  shortlist_{mode}.csv    去重后候选(含残差IC、分层形态、去偏判定)
  clusters_{mode}.csv     信息簇与代表
  composites_final.csv    组合因子检验结果
  final_manifest.json     冻结的最终因子清单(封存测试只认它)
"""
from __future__ import annotations
import os, sys, json, warnings, time
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

from afactor.config import LEDGER, IC_THRESHOLD
from afactor.panel import Panel, TailPanel
from afactor.fast_eval import EvalContext, evaluate, eval_one
from afactor import factors as FA, ops, screen, combine, stats as ST

CLUSTER_THR = 0.6
_G = {}


def _init(mode):
    p = TailPanel() if mode == "tail" else Panel()
    _G["p"] = p
    _G["ctx"] = EvalContext(p, mode, (1,))
    _G["size"] = ops.cs_rank(ops.signed_log(p["adv20"]))
    _G["mode"] = mode
    FA.build_all()


def _resid(names):
    p, ctx, size = _G["p"], _G["ctx"], _G["size"]
    out = []
    for n in names:
        try:
            f = FA.REGISTRY[n][1](p).reindex(index=p.dates, columns=p.codes) \
                .replace([np.inf, -np.inf], np.nan)
            r = ops.cs_neutralize_1d(f, size)
            ic, _, _, _ = eval_one(r.to_numpy(np.float32), ctx.R[1], ctx.M[1], want_layers=False)
            v = ic[np.isfinite(ic)]
            out.append((n, float(v.mean()) if v.size > 60 else None,
                        float(v.mean() / v.std(ddof=1) * np.sqrt(v.size)) if v.size > 60 and v.std() > 0 else None))
        except Exception:
            out.append((n, None, None))
    return out


def residual_ic(names, mode) -> pd.DataFrame:
    chunks = [names[i:i + 12] for i in range(0, len(names), 12)]
    rows = []
    with ProcessPoolExecutor(max_workers=5, initializer=_init, initargs=(mode,)) as ex:
        for fu in as_completed([ex.submit(_resid, c) for c in chunks]):
            rows += fu.result()
    return pd.DataFrame(rows, columns=["factor", "resid_ic", "resid_t"])


def analyse(mode: str) -> dict:
    print(f"\n{'='*90}\n口径 = {mode}\n{'='*90}")
    df = screen.load_ledger("trials_[0-9]*.jsonl")
    df = df[df["mode"] == mode]
    FA.build_all()
    df = df[df["factor"].isin(FA.REGISTRY)]
    stages = {"0_全部试验": len(df)}
    if "error" in df.columns:
        stages["0b_成功计算"] = int(df["error"].isna().sum())
    g1 = screen.gate_stats(df); stages["1_IC>=0.025且|t|>=3"] = len(g1)
    g2 = screen.gate_consistency(g1); stages["2_分年+滚动窗口方向全一致"] = len(g2)

    print(f"  计算残差 IC({len(g2)} 个)...", flush=True)
    rid = residual_ic(g2["factor"].tolist(), mode)
    g2 = g2.merge(rid, on="factor", how="left")
    g2["retention"] = g2["resid_ic"].abs() / g2["ic_mean"].abs()

    g3, dropped = screen.dedupe_by_ic_corr(g2, mode, thr=0.7)
    stages["3_IC序列去重(|ρ|<0.7)"] = len(g3)
    g4 = screen.gate_layers(g3, min_mono=0.5, require_extreme=True)
    stages["4_分层单调且极端组为多头"] = len(g4)
    g5 = g4[g4["retention"] >= 0.5]
    stages["5_非规模马甲(残差IC保留>=50%)"] = len(g5)

    for k, v in stages.items():
        print(f"  {k}: {v}")

    # 去偏
    eff = ST.effective_n_trials(df["factor"].tolist(), mode)
    thr_all = ST.expected_max_t(eff["n_trials"])
    thr_eff = ST.expected_max_t(eff["n_eff"])
    print(f"  试验数 {eff['n_trials']} -> 有效独立试验数 {eff['n_eff']:.1f} "
          f"(前10主成分解释 {eff['top10_var_share']*100:.0f}% 方差)")
    print(f"  运气门槛:按全部试验 |t|>{thr_all:.2f};按有效独立试验 |t|>{thr_eff:.2f}")
    for g in (g3, g4, g5):
        g["pass_deflated"] = g["ic_t"].abs() >= thr_all

    # 聚类与组合
    signs = dict(zip(g2["factor"], np.sign(g2["ic_mean"])))
    clusters = combine.cluster_by_ic(g2["factor"].tolist(), mode, thr=CLUSTER_THR, signs=signs)
    info = g2.set_index("factor")
    reps, ct = {}, []
    for cid, mem in clusters.items():
        sub = info.loc[[m for m in mem if m in info.index]]
        if sub.empty:
            continue
        best = sub["worst_window_abs_ic"].idxmax()
        reps[cid] = best
        ct.append({"cluster": cid, "size": len(mem), "rep": best, "family": sub.loc[best, "family"],
                   "ic": sub.loc[best, "ic_mean"], "resid_ic": sub.loc[best, "resid_ic"],
                   "retention": sub.loc[best, "retention"], "members": ",".join(mem[:20])})
    ct = pd.DataFrame(ct).sort_values("size", ascending=False)
    ct.to_csv(LEDGER / f"clusters_{mode}.csv", index=False)
    print(f"  信息簇 {len(clusters)} 个;规模 {sorted((len(v) for v in clusters.values()), reverse=True)[:12]}")

    screen.summary_table(g3).assign(
        resid_ic=g3.set_index("factor")["resid_ic"].reindex(
            screen.summary_table(g3)["factor"]).values,
        retention=g3.set_index("factor")["retention"].reindex(
            screen.summary_table(g3)["factor"]).values,
    ).to_csv(LEDGER / f"shortlist_{mode}.csv", index=False)

    return {"mode": mode, "stages": stages, "g2": g2, "g3": g3, "g4": g4, "g5": g5,
            "clusters": clusters, "reps": reps, "signs": signs, "eff": eff,
            "thr_all": thr_all, "cluster_tbl": ct}


def build_composites(res: dict) -> pd.DataFrame:
    mode = res["mode"]
    p = TailPanel() if mode == "tail" else Panel()
    ctx = EvalContext(p, mode, (1, 5, 10))
    FA.build_all()
    info = res["g2"].set_index("factor")
    rep_names = list(res["reps"].values())
    fac = {n: FA.REGISTRY[n][1](p).reindex(index=p.dates, columns=p.codes)
           .replace([np.inf, -np.inf], np.nan) for n in rep_names}
    z = combine.prepare(fac, {n: res["signs"][n] for n in rep_names}, p["universe"])
    indep = [n for n in rep_names if (info.loc[n, "retention"] or 0) >= 0.5]
    strong = info.loc[rep_names].sort_values("worst_window_abs_ic", ascending=False)
    sets = {
        "COMP_all": rep_names,
        "COMP_indep": indep,
        "COMP_top8": strong.head(8).index.tolist(),
        "COMP_top8_indep": [n for n in strong.index if n in indep][:8],
    }
    rows = []
    for nm, mem in sets.items():
        if len(mem) < 2:
            continue
        f = combine.combine_equal({n: z[n] for n in mem})
        r = evaluate(f"{nm}__{mode}", f.reindex(index=p.dates, columns=p.codes).to_numpy(np.float32),
                     ctx, family="composite", note=f"{mode}口径 等权组合 {len(mem)} 成员",
                     ledger_path=LEDGER / "trials_composite_final.jsonl")
        r["members"] = ",".join(mem)
        rows.append(r)
        print("  %-22s 成员=%2d IC=%7.4f t=%6.1f 年一致=%.2f 最差窗=%.4f 单调=%6.2f "
              "多头组=%2d 多头超额=%.3f 换手=%.2f" % (
                  nm, len(mem), r["ic_mean"], r["ic_t"], r["year_sign_consistency"] or 0,
                  r["worst_window_abs_ic"] or 0, r["mono_spearman"] or 0, r["long_group"],
                  r["long_excess_ann"], r["long_turnover"] or 0))
    return pd.DataFrame(rows)


def main():
    out = {}
    comps = []
    for mode in ("open", "tail"):
        r = analyse(mode)
        out[mode] = r
        print(f"\n  --- {mode} 口径组合因子 ---")
        c = build_composites(r)
        c["mode"] = mode
        comps.append(c)
    comps = pd.concat(comps, ignore_index=True)
    comps.to_csv(LEDGER / "composites_final.csv", index=False)

    pd.DataFrame({m: out[m]["stages"] for m in out}).to_csv(LEDGER / "funnel.csv")
    allt = screen.load_ledger("trials*.jsonl")
    allt.to_csv(LEDGER / "factor_summary_all.csv", index=False)

    # 冻结清单:两口径各取去重+分层+非马甲后的存活者(按最差窗口 IC 排序取前 8)
    man = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "modes": ["open", "tail"],
           "single_factors": [], "composite_members": [], "composite_signs": {}, "is_ic": {}}
    for mode in ("open", "tail"):
        g5 = out[mode]["g5"].sort_values("worst_window_abs_ic", ascending=False)
        picks = g5["factor"].head(8).tolist()
        for n in picks:
            if n not in man["single_factors"]:
                man["single_factors"].append(n)
            man["is_ic"].setdefault(n, {})[mode] = float(
                out[mode]["g2"].set_index("factor").loc[n, "ic_mean"])
    reps_open = list(out["open"]["reps"].values())
    info_o = out["open"]["g2"].set_index("factor")
    man["composite_members"] = [n for n in reps_open if (info_o.loc[n, "retention"] or 0) >= 0.5]
    man["composite_signs"] = {n: float(np.sign(info_o.loc[n, "ic_mean"])) for n in man["composite_members"]}
    for mode in ("open", "tail"):
        row = comps[(comps["mode"] == mode) & (comps["factor"].str.startswith("COMP_indep"))]
        if len(row):
            man["is_ic"].setdefault("COMP_final", {})[mode] = float(row.iloc[0]["ic_mean"])
    json.dump(man, open(LEDGER / "final_manifest.json", "w"), ensure_ascii=False, indent=1)
    print(f"\n冻结清单已写入 {LEDGER/'final_manifest.json'}:"
          f"单因子 {len(man['single_factors'])} 个,组合成员 {len(man['composite_members'])} 个")


if __name__ == "__main__":
    main()
