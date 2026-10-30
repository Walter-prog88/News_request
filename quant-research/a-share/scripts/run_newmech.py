"""新机制族检验(2026-08-22,用户挖掘系统提供的研报方向)。

三个机制,查证后确认现有 1951 个因子里**完全没有**,不是算子变体:
  nm_lead_*  / nm_dlead_*  量价滞后互相关(中金 corr_pvl「量领先价」)
  nm_chip_*                筹码分布(中金 doc_vol_pdf90_std「筹码分位波动」)
  nm_qrs_*                 QRS 阻力支撑相对强度

关键判据**不是** IC 高低,而是**是否新增独立维度**:
本项目已确认 1459 个因子只承载约 11 个独立维度,第二/三轮新增的 PEAD、TA 各加了 0 个。
故除常规 gate 外,额外报告新因子 IC 序列与既有维度代表因子的相关性。

账本写到 trials_newmech.jsonl —— 注意 **不匹配** `trials_[0-9]*.jsonl` 这个 glob,
因此不会污染正在跑的 A/B/C 与消融实验的因子选取。
"""
from __future__ import annotations
import os, sys, json, time, warnings
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

from afactor.config import LEDGER, IC_THRESHOLD
from afactor.panel import Panel, TailPanel
from afactor.fast_eval import EvalContext, eval_one, _nw_t, _ic_block, _layer_block
from afactor import factors as FA, ops, screen

pd.set_option("display.width", 240)

# 既有 11 个独立维度的代表因子(第一轮主成分分析结论,见 FINDINGS.md)
DIM_REPS = ["rt_turn_z20", "ivol_20", "rev_20", "id_ret_intraday_m60", "id_amt_gini_m5",
            "id_v_ac1_m5", "amihud", "mom_overnight_20", "id_smart_money_m20",
            "ret_skew_20", "lit_vola_downvol_ratio_w20"]


def main():
    mode = os.environ.get("NM_MODE", "open")
    horizon = 1
    FA.build_all()
    p = TailPanel() if mode == "tail" else Panel()
    u = p["universe"]
    names = [n for n, v in FA.REGISTRY.items() if v[0] == "newmech"]
    print(f"新机制族 {len(names)} 个因子 | mode={mode}", flush=True)

    ctx = EvalContext(p, mode, (1, 5))
    years = np.asarray(p.dates.year)
    rows, ic_series = [], {}
    t0 = time.time()
    for i, n in enumerate(names):
        try:
            f = FA.REGISTRY[n][1](p).reindex(index=p.dates, columns=p.codes)
            arr = ops.cs_rank(f.replace([np.inf, -np.inf], np.nan).where(u)).to_numpy(np.float32)
            ic, q, cov, turn = eval_one(arr, ctx.R[1], ctx.M[1], want_layers=True)
            v = ic[np.isfinite(ic)]
            if v.size < 200:
                continue
            ic5, _, _, _ = eval_one(arr, ctx.R[5], ctx.M[5], want_layers=False)
            v5 = ic5[np.isfinite(ic5)]
            blk = _ic_block(ic, years)
            lay = _layer_block(q, turn) or {}
            ic_series[n] = pd.Series(ic, index=p.dates)
            rows.append(dict(factor=n, family="newmech", mode=mode,
                             ic_mean=float(v.mean()), ic_t=float(v.mean() / v.std(ddof=1) * np.sqrt(v.size)),
                             ic_t_nw=_nw_t(v, 5), h5_ic_mean=float(v5.mean()) if v5.size else np.nan,
                             year_sign_consistency=blk.get("year_sign_consistency"),
                             window_sign_consistency=blk.get("window_sign_consistency"),
                             worst_window_abs_ic=blk.get("worst_window_abs_ic"),
                             long_excess_ann=lay.get("long_excess_ann"),
                             mono_spearman=lay.get("mono_spearman"),
                             long_turnover=lay.get("long_turnover"),
                             n_days=int(v.size)))
        except Exception as e:
            print(f"  [失败] {n}: {type(e).__name__}: {e}", flush=True)
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(names)}  {time.time()-t0:.0f}s", flush=True)

    d = pd.DataFrame(rows)
    d["abs_ic"] = d["ic_mean"].abs()
    with open(LEDGER / "trials_newmech.jsonl", "a") as fh:
        for r in rows:
            fh.write(json.dumps({**r, "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                                 "error": None}, ensure_ascii=False, default=float) + "\n")

    print(f"\n{'='*110}\n=== 常规 gate(|IC|>={IC_THRESHOLD} 且 |t|>=3 且方向一致)===")
    g = d[(d["abs_ic"] >= IC_THRESHOLD) & (d["ic_t"].abs() >= 3.0)]
    print(f"过 IC 关卡 {len(g)}/{len(d)}")
    print(d.sort_values("abs_ic", ascending=False).head(20)[
        ["factor", "ic_mean", "ic_t", "ic_t_nw", "h5_ic_mean", "year_sign_consistency",
         "long_excess_ann", "mono_spearman"]].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # ---- 真正的判据:是否新增独立维度 ----
    print(f"\n{'='*110}\n=== 独立性检验:新因子 IC 序列 vs 既有 11 个维度代表 ===")
    reps = {}
    for r in DIM_REPS:
        if r not in FA.REGISTRY:
            continue
        fr = FA.REGISTRY[r][1](p).reindex(index=p.dates, columns=p.codes)
        ar = ops.cs_rank(fr.replace([np.inf, -np.inf], np.nan).where(u)).to_numpy(np.float32)
        icr, _, _, _ = eval_one(ar, ctx.R[1], ctx.M[1], want_layers=False)
        reps[r] = pd.Series(icr, index=p.dates)
    RP = pd.DataFrame(reps)
    out = []
    for n, s in ic_series.items():
        c = RP.corrwith(s).abs()
        out.append((n, float(d.loc[d["factor"] == n, "abs_ic"].iloc[0]),
                    float(c.max()), c.idxmax()))
    o = pd.DataFrame(out, columns=["factor", "abs_ic", "与既有维度最大|相关|", "最像哪个维度"])
    o = o.sort_values("与既有维度最大|相关|")
    print("\n最正交的 15 个(相关性越低 = 越可能是新维度):")
    print(o.head(15).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\n按族汇总:")
    o["族"] = o["factor"].str.extract(r"^nm_(dlead|lead|chip|qrs)")
    print(o.groupby("族").agg(个数=("factor", "size"), 平均绝对IC=("abs_ic", "mean"),
                             最高绝对IC=("abs_ic", "max"),
                             与既有维度中位相关=("与既有维度最大|相关|", "median"),
                             最正交的相关=("与既有维度最大|相关|", "min")).round(4).to_string())
    o.to_csv(LEDGER / f"newmech_orthogonality_{mode}.csv", index=False)
    d.to_csv(LEDGER / f"newmech_gate_{mode}.csv", index=False)
    print(f"\n结果已写入 newmech_gate_{mode}.csv / newmech_orthogonality_{mode}.csv")


if __name__ == "__main__":
    main()
