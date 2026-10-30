"""实验五:从"被去重吸收"的因子中榨取相对现有组合的残余信息。

对全部通过一致性关的因子:对 COMP_final(及规模代理)截面正交化,计算残差 IC。
残差 |IC|>=0.015、|t|>=4、分年方向一致的因子视为"有增量",按簇去重后并入组合
形成 COMP_v2,与 COMP_final 同口径对比。
"""
from __future__ import annotations
import os, sys, json, warnings
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

from afactor.config import LEDGER
from afactor.panel import Panel, TailPanel
from afactor.fast_eval import EvalContext, evaluate, eval_one
from afactor import factors as FA, ops, screen, combine

_G = {}


TRAIN_END = "2023-12-31"     # 选择只用 2020-2023;2024-2025 留作增量评估(防 winner's curse)


def _init():
    p = Panel()
    _G["p"] = p
    _G["ctx"] = EvalContext(p, "open", (1,))
    FA.build_all()
    man = json.load(open(LEDGER / "final_manifest.json"))
    fac = {n: FA.REGISTRY[n][1](p).reindex(index=p.dates, columns=p.codes)
           .replace([np.inf, -np.inf], np.nan) for n in man["composite_members"] if n in FA.REGISTRY}
    z = combine.prepare(fac, man["composite_signs"], p["universe"])
    comp = combine.combine_equal(z)
    size = ops.cs_rank(ops.signed_log(p["adv20"]))
    _G["comp"] = comp
    # FWL 合规:第二个正交化目标先对 comp 正交,等价于对 (comp, size) 联合回归取残差
    _G["size_perp"] = ops.cs_neutralize_1d(size, comp)
    _G["train_mask"] = np.asarray(p.dates <= pd.Timestamp(TRAIN_END))


def _nw_t_local(v: np.ndarray, lag: int = 5) -> float:
    n = v.size
    mu = v.mean(); d = v - mu
    var = float(d @ d) / n
    for l in range(1, min(lag, n - 1) + 1):
        var += 2.0 * (1.0 - l / (lag + 1.0)) * float(d[l:] @ d[:-l]) / n
    return float(mu / np.sqrt(var / n)) if var > 0 else np.nan


def _run(names):
    p, ctx, comp, size_perp = _G["p"], _G["ctx"], _G["comp"], _G["size_perp"]
    tm = _G["train_mask"]
    out = []
    for n in names:
        try:
            f = FA.REGISTRY[n][1](p).reindex(index=p.dates, columns=p.codes) \
                .replace([np.inf, -np.inf], np.nan)
            r1 = ops.cs_neutralize_1d(f, comp)
            r2 = ops.cs_neutralize_1d(r1, size_perp)
            ic, _, _, _ = eval_one(r2.to_numpy(np.float32), ctx.R[1], ctx.M[1], want_layers=False)
            s = pd.Series(ic, index=p.dates)
            v = s[tm].dropna()                       # 选择统计只看训练窗
            if len(v) < 300:
                out.append((n, None, None, None, None)); continue
            yearly = v.groupby(v.index.year).mean()
            cons = float(max((yearly > 0).mean(), (yearly < 0).mean()))
            t = _nw_t_local(v.to_numpy())
            hold = s[~tm].dropna()                    # 2024-2025 持出窗,仅记录
            out.append((n, float(v.mean()), t, cons,
                        float(hold.mean()) if len(hold) > 100 else None))
        except Exception:
            out.append((n, None, None, None, None))
    return out


def main():
    df = screen.load_ledger("trials_[0-9]*.jsonl")
    df = df[df["mode"] == "open"]
    FA.build_all()
    g2 = screen.gate_consistency(screen.gate_stats(df))
    man = json.load(open(LEDGER / "final_manifest.json"))
    cand = [n for n in g2["factor"] if n in FA.REGISTRY and n not in man["composite_members"]]
    print(f"候选 {len(cand)} 个(过一致性关且不在现有组合中)", flush=True)

    chunks = [cand[i:i + 12] for i in range(0, len(cand), 12)]
    rows = []
    with ProcessPoolExecutor(max_workers=5, initializer=_init) as ex:
        for i, fu in enumerate(as_completed([ex.submit(_run, c) for c in chunks]), 1):
            rows += fu.result()
            if i % 5 == 0:
                print(f"  [{len(rows)}/{len(cand)}]", flush=True)
    res = pd.DataFrame(rows, columns=["factor", "residc_ic", "residc_t", "residc_cons",
                                      "residc_ic_holdout"])
    res = res.dropna(subset=["residc_ic"])
    res = res.merge(g2[["factor", "family", "ic_mean", "worst_window_abs_ic"]].drop_duplicates("factor"),
                    on="factor")
    res.to_csv(LEDGER / "residual_vs_comp.csv", index=False)

    # 选择:训练窗(2020-2023)统计;持出窗 IC 只展示,不参与选择
    hits = res[(res["residc_ic"].abs() >= 0.015) & (res["residc_t"].abs() >= 4) &
               (res["residc_cons"] >= 1.0)].sort_values("residc_ic", key=abs, ascending=False)
    print(f"\n训练窗内有残余信息的因子:{len(hits)} 个(|残差IC|>=0.015, NW|t|>=4, 分年一致)")
    print(hits.head(20).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    ok_hold = hits["residc_ic_holdout"].notna() & (np.sign(hits["residc_ic_holdout"]) == np.sign(hits["residc_ic"]))
    print(f"其中持出窗(2024-2025)方向一致: {int(ok_hold.sum())}/{len(hits)}")

    if len(hits) == 0:
        print("无增量因子,COMP_v2 不构建。")
        return
    # 按簇去重:每簇只取残差 IC 最强者,最多补 8 个
    ct = pd.read_csv(LEDGER / "clusters_open.csv")
    fam_cluster = {}
    for _, r in ct.iterrows():
        for m in str(r["members"]).split(","):
            fam_cluster[m] = r["cluster"]
    hits["cluster"] = hits["factor"].map(fam_cluster).fillna(-1)
    picks_df = hits.sort_values("residc_ic", key=abs, ascending=False) \
                   .drop_duplicates("cluster").head(8)
    picks = picks_df["factor"].tolist()
    # 符号取自残差 IC(正交化后方向可能翻转,取原始 ic_mean 会反向加入)
    pick_signs = {r["factor"]: float(np.sign(r["residc_ic"])) for _, r in picks_df.iterrows()}
    print(f"\n并入组合:{picks}")

    for mode in ["open", "tail"]:
        p = TailPanel() if mode == "tail" else Panel()
        ctx = EvalContext(p, mode, (1, 5))
        members = [n for n in man["composite_members"] if n in FA.REGISTRY] + picks
        signs = dict(man["composite_signs"])
        signs.update(pick_signs)
        fac = {n: FA.REGISTRY[n][1](p).reindex(index=p.dates, columns=p.codes)
               .replace([np.inf, -np.inf], np.nan) for n in members}
        z = combine.prepare(fac, signs, p["universe"])
        v2 = combine.combine_equal(z)
        r = evaluate(f"COMP_v2__{mode}", v2.to_numpy(np.float32), ctx, family="composite",
                     note=f"COMP_final + {len(picks)}个残余信息因子(训练窗选择)",
                     ledger_path=LEDGER / "trials_engineering.jsonl")
        base = screen.load_ledger("trials*.jsonl")
        base = base[(base["factor"] == f"COMP_indep__{mode}") & (base["mode"] == mode)]
        b_ic = float(base["ic_mean"].iloc[0]) if len(base) else np.nan
        b_yr = base["ic_by_year"].iloc[0] if len(base) else {}
        if isinstance(b_yr, str):
            b_yr = json.loads(b_yr.replace("'", '"'))
        v_yr = r["ic_by_year"]
        # 决定性对比:选择未用过的 2024-2025(注意成员因子与符号仍来自全样本流程,
        # 该对比只对"picks 的增量"是持出的,结论标注为增量检验而非完全 OOS)
        inc24 = (v_yr.get(2024, np.nan) or np.nan) - float(b_yr.get("2024", b_yr.get(2024, np.nan)) or np.nan)
        inc25 = (v_yr.get(2025, np.nan) or np.nan) - float(b_yr.get("2025", b_yr.get(2025, np.nan)) or np.nan)
        print(f"[{mode}] COMP_v2 IC={r['ic_mean']:.4f} t={r['ic_t']:.1f} "
              f"最差窗={r['worst_window_abs_ic']:.4f} 单调={r['mono_spearman']:.2f} "
              f"(对照 IC={b_ic:.4f};持出窗增量: 2024 {inc24:+.4f}, 2025 {inc25:+.4f})")
    json.dump({"picks": picks}, open(LEDGER / "comp_v2_members.json", "w"), ensure_ascii=False)


if __name__ == "__main__":
    main()
