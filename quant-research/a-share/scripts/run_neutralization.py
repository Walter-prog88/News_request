"""第三步:正规中性化重跑 —— 市值 / 行业 / 流动性 三个维度分别与联合。

对 COMP_v3 成员 + 第一轮短名单,对比原始 IC 与四种中性化后的残差 IC:
  cap     : 仅对数市值
  ind     : 仅统计行业(收益相关聚类,已用 comp_type 验证:银行83%/券商100%聚类正确)
  cap+ind : 用户要求的双重中性化(FWL 两步实现)
  cap+ind+liq: 再加流动性(20日成交额),用于识别"流动性马甲"

同时跑"剔除 ST"股票池对照,量化 ST 股对结论的贡献。
"""
from __future__ import annotations
import os, sys, json, warnings
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

from afactor.config import LEDGER
from afactor.panel import Panel
from afactor.fast_eval import EvalContext, eval_one, _layer_block
from afactor import factors as FA, ops, screen, statind as SI

_G = {}


def _init():
    p = Panel()
    FA.build_all()
    u = p["universe"]
    _G.update(p=p, u=u, ctx=EvalContext(p, "open", (1,)),
              lc=p["log_cap"], ladv=ops.signed_log(p["adv20"]),
              L=SI.load_labels().reindex(index=p.dates, columns=p.codes))
    # 剔除 ST 的股票池
    _G["u_nost"] = u & ~p["is_st"].reindex_like(u).fillna(False)
    ctx2 = EvalContext(p, "open", (1,))
    ctx2.M = {1: _G["u_nost"].to_numpy(bool) & np.isfinite(_G["ctx"].R[1])}
    _G["ctx_nost"] = ctx2


def _ic(F, ctx):
    ic, q, cov, turn = eval_one(F.reindex(index=_G["p"].dates, columns=_G["p"].codes)
                                .to_numpy(np.float32), ctx.R[1], ctx.M[1], want_layers=True)
    v = ic[np.isfinite(ic)]
    if v.size < 100:
        return None, None, None
    lay = _layer_block(q, turn)
    return float(v.mean()), float(v.mean() / v.std(ddof=1) * np.sqrt(v.size)), lay


def _run(names):
    p, u, ctx, lc, ladv, L = (_G[k] for k in ("p", "u", "ctx", "lc", "ladv", "L"))
    out = []
    for n in names:
        try:
            f = FA.REGISTRY[n][1](p).reindex(index=p.dates, columns=p.codes) \
                .replace([np.inf, -np.inf], np.nan)
            row = {"factor": n, "family": FA.REGISTRY[n][0]}
            base_ic, base_t, lay = _ic(f, ctx)
            row.update(ic=base_ic, t=base_t, mono=lay["mono_spearman"] if lay else None,
                       long_excess=lay["long_excess_ann"] if lay else None)
            variants = {
                "cap": ops.cs_neutralize_cap_ind(f, lc, None, universe=u),
                "ind": ops.cs_neutralize_cap_ind(f, lc * 0.0, L, universe=u),
                "cap_ind": ops.cs_neutralize_cap_ind(f, lc, L, universe=u),
            }
            # cap+ind+liq:在 cap_ind 残差基础上再对(同样中性化过的)流动性正交
            liq_ci = ops.cs_neutralize_cap_ind(ladv, lc, L, universe=u)
            variants["cap_ind_liq"] = ops.cs_neutralize_1d(variants["cap_ind"], liq_ci)
            for k, v in variants.items():
                ic, t, _ = _ic(v, ctx)
                row[f"ic_{k}"] = ic
                row[f"ret_{k}"] = (abs(ic) / abs(base_ic)) if (ic and base_ic) else None
            ic_ns, t_ns, _ = _ic(f, _G["ctx_nost"])
            row["ic_no_st"] = ic_ns
            row["ret_no_st"] = (abs(ic_ns) / abs(base_ic)) if (ic_ns and base_ic) else None
            out.append(row)
        except Exception as e:
            out.append({"factor": n, "error": f"{type(e).__name__}: {e}"})
    return out


def main():
    FA.build_all()
    man = json.load(open(LEDGER / "comp_v3_members.json"))
    targets = [n for n in man["members"] if n in FA.REGISTRY]
    # 补入基本面过关因子
    led = screen.load_ledger("trials_[0-9]*.jsonl")
    fu = screen.gate_consistency(screen.gate_stats(
        led[(led["family"] == "fundamental") & (led["mode"] == "open")]))
    targets += [n for n in fu["factor"] if n in FA.REGISTRY and n not in targets]
    print(f"待中性化检验 {len(targets)} 个因子(COMP_v3 成员 {len(man['members'])} + 基本面过关 {len(fu)})",
          flush=True)

    chunks = [targets[i:i + 6] for i in range(0, len(targets), 6)]
    rows = []
    with ProcessPoolExecutor(max_workers=5, initializer=_init) as ex:
        for i, fu_ in enumerate(as_completed([ex.submit(_run, c) for c in chunks]), 1):
            rows += fu_.result()
            print(f"  [{len(rows)}/{len(targets)}]", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(LEDGER / "neutralization.csv", index=False)
    ok = df[df.get("error").isna()] if "error" in df.columns else df
    pd.set_option("display.width", 240)
    cols = ["factor", "family", "ic", "t", "ic_cap", "ret_cap", "ic_ind", "ret_ind",
            "ic_cap_ind", "ret_cap_ind", "ic_cap_ind_liq", "ret_cap_ind_liq",
            "ic_no_st", "ret_no_st"]
    print("\n=== 中性化前后 IC(ret_* = 残差IC/原始IC 的保留率)===")
    print(ok[cols].sort_values("ic", key=abs, ascending=False)
          .to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n=== 各中性化维度的中位保留率 ===")
    for k in ["cap", "ind", "cap_ind", "cap_ind_liq", "no_st"]:
        c = f"ret_{k}"
        if c in ok:
            print(f"  {k:14s} 中位 {ok[c].median():.3f}  最低 {ok[c].min():.3f}  <50%的因子数 {(ok[c] < 0.5).sum()}")


if __name__ == "__main__":
    main()
