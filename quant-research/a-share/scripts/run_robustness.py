"""稳健性四关:滞后一天、截面置换、口径敏感性(股票池阈值/尾盘口径)、分层形态诊断。"""
from __future__ import annotations
import os, sys, json, warnings
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

from afactor.config import LEDGER, MIN_ADV20
from afactor.panel import Panel
from afactor.fast_eval import EvalContext, evaluate, eval_one, _ic_block, _layer_block
from afactor import factors as FA, ops, screen, combine, controls

N_PERM = 20


def shortlist(k=10):
    df = screen.load_ledger("trials_[0-9]*.jsonl")
    g2 = screen.gate_consistency(screen.gate_stats(df))
    rid = pd.read_csv(LEDGER / "residual_ic.csv")
    g2 = g2.merge(rid, on="factor", how="left")
    g2["retention"] = g2["resid_ic"].abs() / g2["ic_mean"].abs()
    g3, _ = screen.dedupe_by_ic_corr(g2, "open")
    g3 = g3.sort_values("worst_window_abs_ic", ascending=False)
    return g3.head(k)


def main():
    p = Panel()
    FA.build_all()
    ctx = EvalContext(p, "open", (1,))
    ctx_tail = EvalContext(p, "tail", (1,))
    sl = shortlist(10)
    names = sl["factor"].tolist()
    print(f"短名单 {len(names)} 个 + 最优组合\n")

    # 组合因子重建
    mem = json.load(open(LEDGER / "composite_members.json"))
    df_all = screen.load_ledger("trials_[0-9]*.jsonl")
    g2 = screen.gate_consistency(screen.gate_stats(df_all))
    signs = dict(zip(g2["factor"], np.sign(g2["ic_mean"])))
    fac = {n: FA.REGISTRY[n][1](p).reindex(index=p.dates, columns=p.codes)
           .replace([np.inf, -np.inf], np.nan) for n in mem["reps"].values()}
    z = combine.prepare(fac, {n: signs.get(n, 1) for n in fac}, p["universe"])
    comp = combine.combine_equal(z)

    targets = {n: FA.REGISTRY[n][1](p).reindex(index=p.dates, columns=p.codes)
               .replace([np.inf, -np.inf], np.nan) for n in names}
    targets["COMP_equal_allclusters"] = comp

    # ---- 备用股票池(ADV 门槛敏感性)----
    def univ_adv(thr):
        return (p["valid"] & (p["days_listed"] >= 6) & (p["adv20"] >= thr)).fillna(False)

    ctx_loose, ctx_tight = EvalContext(p, "open", (1,)), EvalContext(p, "open", (1,))
    ctx_loose.M = {1: univ_adv(1e7).to_numpy(bool) & np.isfinite(ctx.R[1])}
    ctx_tight.M = {1: univ_adv(5e7).to_numpy(bool) & np.isfinite(ctx.R[1])}

    rows = []
    print("%-30s %8s %8s %8s %8s %8s %8s %8s %8s" % (
        "factor", "基准IC", "滞后1日", "尾盘口径", "ADV1000万", "ADV5000万", "置换均值", "置换最大", "置换p"))
    for name, f in targets.items():
        A = f.to_numpy(np.float32)
        base = eval_one(A, ctx.R[1], ctx.M[1], want_layers=False)[0]
        base_ic = np.nanmean(base)
        lag = eval_one(f.shift(1).to_numpy(np.float32), ctx.R[1], ctx.M[1], want_layers=False)[0]
        tail = eval_one(A, ctx_tail.R[1], ctx_tail.M[1], want_layers=False)[0]
        loose = eval_one(A, ctx.R[1], ctx_loose.M[1], want_layers=False)[0]
        tight = eval_one(A, ctx.R[1], ctx_tight.M[1], want_layers=False)[0]
        perms = []
        for s in range(N_PERM):
            sh = controls.shuffle_cs(f, seed=1000 + s).to_numpy(np.float32)
            perms.append(abs(np.nanmean(eval_one(sh, ctx.R[1], ctx.M[1], want_layers=False)[0])))
        perms = np.array(perms)
        pval = float((perms >= abs(base_ic)).mean())
        r = {"factor": name, "ic": base_ic, "ic_lag1": np.nanmean(lag),
             "ic_tail": np.nanmean(tail), "ic_adv1e7": np.nanmean(loose),
             "ic_adv5e7": np.nanmean(tight), "perm_mean": perms.mean(),
             "perm_max": perms.max(), "perm_p": pval,
             "lag_retention": abs(np.nanmean(lag)) / max(abs(base_ic), 1e-9)}
        rows.append(r)
        print("%-30s %8.4f %8.4f %8.4f %8.4f %8.4f %8.4f %8.4f %8.3f" % (
            name, r["ic"], r["ic_lag1"], r["ic_tail"], r["ic_adv1e7"], r["ic_adv5e7"],
            r["perm_mean"], r["perm_max"], r["perm_p"]))
    out = pd.DataFrame(rows)
    out.to_csv(LEDGER / "robustness.csv", index=False)
    print("\n滞后1日保留率(<0.5 视为微观毛刺):")
    print(out[["factor", "lag_retention"]].sort_values("lag_retention").to_string(index=False, float_format=lambda x: f"{x:.2f}"))


if __name__ == "__main__":
    main()
