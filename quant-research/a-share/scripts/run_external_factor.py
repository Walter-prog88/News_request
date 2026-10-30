"""第四步:外部因子归位 —— 老师给的 momentum_factor_total 走同一套检验。

回答三个问题:(1) 它有没有预测力?(2) 它落在我们已知的哪个信息维度里,还是新维度?
(3) 它相对现有组合 COMP_v3 有没有增量?
"""
from __future__ import annotations
import os, sys, json, warnings
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

from afactor.config import LEDGER, ROOT
from afactor.panel import Panel
from afactor.fast_eval import EvalContext, evaluate, eval_one, IC_DIR
from afactor import factors as FA, ops, screen, combine, statind as SI

pd.set_option("display.width", 240)


def load_external(p: Panel) -> pd.DataFrame:
    df = pd.read_pickle(ROOT / "data_8:21" / "momentum_factor_total.pkl")
    if df.columns.nlevels > 1:
        df = df.droplevel(0, axis=1)
    df.index = pd.to_datetime(df.index)
    return df.reindex(index=p.dates, columns=p.codes).astype(np.float32)


def main():
    p = Panel(); FA.build_all()
    ctx = EvalContext(p, "open", (1, 5, 10))
    ext = load_external(p)
    u = p["universe"]
    print(f"外部因子面板 {ext.shape} | 决策池内覆盖率 {ext.where(u).notna().sum().sum()/u.sum().sum():.3f}")
    print(f"值域: {np.nanpercentile(ext.where(u), [1,50,99]).round(3)}")

    # 1) 基础检验
    r = evaluate("EXT_teacher_momentum", ext.to_numpy(np.float32), ctx, family="external",
                 note="老师提供的 momentum_factor_total(来源未知)",
                 ledger_path=LEDGER / "trials_external.jsonl")
    print(f"\n=== 基础检验 ===")
    print(f"IC={r['ic_mean']:.4f} t={r['ic_t']:.1f} IR={r['ic_ir']:.3f} "
          f"年一致={r['year_sign_consistency']:.2f} 窗一致={r['window_sign_consistency']:.2f}")
    print(f"最差窗={r['worst_window_abs_ic']:.4f} 单调={r['mono_spearman']:.2f} "
          f"多头组={r['long_group']} 多头超额={r['long_excess_ann']:.3f} 换手={r['long_turnover']:.2f}")
    print(f"h5 IC={r['h5_ic_mean']:.4f}  h10 IC={r['h10_ic_mean']:.4f}")
    print(f"分年 IC: " + " ".join(f"{k}:{v:+.4f}" for k, v in r["ic_by_year"].items()))
    print(f"分层超额: {r['q_excess']}")

    # 2) 归位:与 19 个簇代表 + COMP_v3 成员的 IC 序列相关
    ct = pd.read_csv(LEDGER / "clusters_open.csv")
    reps = ct["rep"].tolist()
    s_ext = pd.read_parquet(IC_DIR / "EXT_teacher_momentum__open.parquet")["h1"]
    cors = {}
    for n in reps:
        f = IC_DIR / f"{n}__open.parquet"
        if f.exists():
            cors[n] = s_ext.corr(pd.read_parquet(f)["h1"])
    cs = pd.Series(cors).sort_values(key=abs, ascending=False)
    print(f"\n=== 归位:与既有维度代表的 IC 序列相关(前8)===")
    print(cs.head(8).round(3).to_string())
    print(f"最大 |相关| = {cs.abs().max():.3f} -> " +
          ("**候选新维度**(<0.5)" if cs.abs().max() < 0.5 else f"归属于「{cs.abs().idxmax()}」维度"))

    # 3) 中性化与增量
    lc = p["log_cap"]; L = SI.load_labels().reindex(index=p.dates, columns=p.codes)
    man = json.load(open(LEDGER / "comp_v3_members.json"))
    fac = {n: FA.REGISTRY[n][1](p).reindex(index=p.dates, columns=p.codes)
           .replace([np.inf, -np.inf], np.nan) for n in man["members"] if n in FA.REGISTRY}
    comp = combine.combine_equal(combine.prepare(fac, man["signs"], u))

    def ic_of(F):
        ic, _, _, _ = eval_one(F.reindex(index=p.dates, columns=p.codes).to_numpy(np.float32),
                               ctx.R[1], ctx.M[1], want_layers=False)
        v = ic[np.isfinite(ic)]
        return float(v.mean()), float(v.mean() / v.std(ddof=1) * np.sqrt(v.size))

    print(f"\n=== 中性化与增量 ===")
    base = r["ic_mean"]
    for nm, F in [("对市值+行业中性化", ops.cs_neutralize_cap_ind(ext, lc, L, universe=u)),
                  ("对流动性中性化", ops.cs_neutralize_1d(ext.where(u), ops.signed_log(p["adv20"]).where(u))),
                  ("对 COMP_v3 正交化", ops.cs_neutralize_1d(ext.where(u), comp.where(u)))]:
        ic, t = ic_of(F)
        print(f"  {nm:22s} IC={ic:+.4f} t={t:+6.1f}  保留率 {abs(ic)/abs(base):.2f}")
    # 反向:COMP_v3 对它正交后还剩多少
    ic, t = ic_of(ops.cs_neutralize_1d(comp.where(u), ext.where(u)))
    ic_c, _ = ic_of(comp)
    print(f"  {'COMP_v3 对它正交后':22s} IC={ic:+.4f} t={t:+6.1f}  保留率 {abs(ic)/abs(ic_c):.2f} (原 {ic_c:.4f})")

    # 4) 若有增量,合成 COMP_v4 对比
    z = combine.prepare({**fac, "EXT": ext}, {**man["signs"], "EXT": float(np.sign(base))}, u)
    v4 = combine.combine_equal(z)
    r4 = evaluate("COMP_v4_with_ext", v4.to_numpy(np.float32), ctx, family="composite",
                  note="COMP_v3 + 外部动量因子", ledger_path=LEDGER / "trials_external.jsonl")
    print(f"\n=== 加入组合的效果 ===")
    print(f"  COMP_v3      IC={ic_c:.4f}")
    print(f"  COMP_v4(+ext) IC={r4['ic_mean']:.4f} t={r4['ic_t']:.1f} 单调={r4['mono_spearman']:.2f} "
          f"最差窗={r4['worst_window_abs_ic']:.4f}")
    print(f"  分年 IC: " + " ".join(f"{k}:{v:+.4f}" for k, v in r4["ic_by_year"].items()))


if __name__ == "__main__":
    main()
