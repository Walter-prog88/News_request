"""把"开关"做成因子:个股独立性(idio_share / mkt_corr)的截面加权与时序切割。

动机(2026-08-25,开关全扫描 `run_switch_sweep.py` 之后):
  103 个候选开关去重后只有 5 个维度,排名第 2–3 的是"个股独立性"
  (mkt_corr 形状 −0.976 / idio_share +0.988,35 个核心因子无一例外单调,极差 3.5 倍),
  **而它从来没被当作条件变量用过**(历史上只用过 amt / 量比 / rvol / 尾盘占比 / amtz / ep / roe / size)。

两种构造,分别对应两种机制,必须分开测:

  A 族(截面加权,**对应扫描发现**)—— 扫描说的是"独立性高的**股票**上因子更强",
     这是截面现象,所以构造也必须是截面的:
       sw_{F}__W__{S} = cs_rank(F) × (cs_rank(S) + 0.5)     连续加权,低开关档信号被压扁
       sw_{F}__T__{S} = cs_rank(F) 只在开关前 70% 保留,其余置 NaN   硬门限
     注意现有的 `ix_a__X__b` = cs_rank(a)×cs_rank(b) **不是**这个 ——
     cs_rank 已中心化到 [-0.5,0.5],两个中心化量相乘表达的是"两头都高",
     会把低开关档的信号**翻转**而不是压扁。所以这个构造确实是新的。

  B 族(时序切割,老机制的自然延伸,作对照)—— 现有 `cv_*_by_*` 用的
     `_rolling_split_by` 是**时序**切割(过去 w 天里按条件排序),
     和扫描发现的截面现象不是一回事,但机制本身独立成立,顺带一起测。

判据(逐层,任一层不过就不值得并入):
  1. IC 与一致性是否比基础因子更好
  2. 与基础因子的 IC 序列相关 —— 是不是只是老信息的放大
  3. 对 COMP_v5 正交后的残差 IC —— 能不能给现有组合**加**东西

产出:research/ledger/switch_factors.csv
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
from afactor.fast_eval import EvalContext, eval_one, _ic_block, _layer_block
from afactor import factors as FA, ops, combine

# 被加权的基础因子:覆盖核心 32 的主要维度,各取一个代表
BASES = [
    "ivol_20", "id_ret_intraday_m60", "id_gap_share_m5", "max_ret_5",
    "ix_rev_20__P__rvol_20", "lit_cn_b_apb_5d_short_w5_w20", "mi_semibeta_mp_m20",
    "lit_intr_intraday_cum_rev_w40", "cv_ret_intraday_by_rvol_20", "amt_vol_5",
    "lit_volu_amt_ordering_trend", "si_gd_amt_vol_20", "mx_amt_gini_std60",
    "lit_volu_time_centroid_shift",
]
SWITCHES = [("idio", "idio_share"), ("mktcorr", "mkt_corr")]
# B 族的值变量(照抄现有 _COND_VALUES)
COND_VALUES = ["ret_c2c", "ret_intraday", "overnight", "ret_last30",
               "smart_money", "vwap_dev", "rsj"]
TOP_FRAC = 0.70          # 硬门限:只在开关最高的 70% 里保留信号


def _switch_rank(p, field, sign):
    """开关的截面秩,方向统一为"值大 = 因子更强"。
    idio_share 正向(特质性高 -> 因子强);mkt_corr 负向(与大盘相关低 -> 因子强)。"""
    return ops.cs_rank(p.raw(field)) * sign


def _comp_v5(p):
    """COMP_v5:25 成员方向对齐 + 去极值 + 截面 zscore + 等权。"""
    man = json.load(open(LEDGER / "comp_v5_members.json"))
    members = [n for n in man["members"] if n in FA.REGISTRY]
    signs = {n: float(man["signs"][n]) for n in members}
    fac = {n: FA.REGISTRY[n][1](p).reindex(index=p.dates, columns=p.codes)
           for n in members}
    z = combine.prepare(fac, signs, p["universe"])
    return combine.combine_equal(z)


def _eval(F, ctx, nw=0):
    A = F.reindex(index=ctx.panel.dates, columns=ctx.panel.codes) \
         .replace([np.inf, -np.inf], np.nan).to_numpy(np.float32)
    ic, q, cov, turn = eval_one(A, ctx.R[1], ctx.M[1], want_layers=True)
    b = _ic_block(ic, ctx.years, nw_lag=nw)
    if b.get("ic_mean") is None:
        return None, None
    b.update(_layer_block(q, turn))
    return b, pd.Series(ic, index=ctx.panel.dates)


def _run(mode):
    P = Panel() if mode == "open" else TailPanel()
    FA.build_all()
    ctx = EvalContext(P, mode, (1,))
    u = P["universe"]

    comp = _comp_v5(P)
    base_f = {"COMP_v5": comp}
    for n in BASES:
        if n in FA.REGISTRY:
            base_f[n] = FA.REGISTRY[n][1](P).reindex(index=P.dates, columns=P.codes)

    sw = {tag: _switch_rank(P, fld, +1.0 if fld == "idio_share" else -1.0)
          for tag, fld in SWITCHES}

    rows, ics = [], {}
    for bn, bf in base_f.items():
        b0, ic0 = _eval(bf, ctx)
        if b0 is None:
            continue
        ics[bn] = ic0
        rows.append({"factor": bn, "kind": "基础", "base": bn, **b0})
        r = ops.cs_rank(bf)
        for tag, s in sw.items():
            # A1 连续加权
            nm = f"sw_{bn}__W__{tag}"
            bb, ic = _eval(r * (s + 0.5), ctx)
            if bb: rows.append({"factor": nm, "kind": "A截面加权", "base": bn, **bb}); ics[nm] = ic
            # A2 硬门限:只在开关最高的 TOP_FRAC 里保留信号
            nm = f"sw_{bn}__T__{tag}"
            keep = s > (0.5 - TOP_FRAC)
            bb, ic = _eval(r.where(keep), ctx)
            if bb: rows.append({"factor": nm, "kind": "A硬门限", "base": bn, **bb}); ics[nm] = ic

    # B 族:时序切割(老机制,作对照)
    from afactor.factors import _rolling_split_by
    getv = {"ret_c2c": lambda p: p["ret_c2c"], "overnight": lambda p: p["overnight"]}
    for vn in COND_VALUES:
        vf = getv.get(vn, lambda p, vn=vn: p.raw(vn))(P)
        for tag, fld in SWITCHES:
            for w in (20, 40):
                nm = f"cv_{vn}_by_{tag}_{w}"
                bb, ic = _eval(_rolling_split_by(vf, P.raw(fld), w), ctx)
                if bb: rows.append({"factor": nm, "kind": "B时序切割", "base": vn, **bb}); ics[nm] = ic

    # 与基础因子的 IC 序列相关 + 对 COMP_v5 正交后的残差 IC
    comp_z = combine.prepare({"c": comp}, {"c": 1.0}, u)["c"]
    for r in rows:
        b = r["base"]
        if b in ics and r["factor"] in ics and r["kind"] != "基础":
            a, c = ics[r["factor"]].align(ics[b], join="inner")
            r["corr_vs_base"] = float(a.corr(c))
        if r["kind"] == "基础" and r["factor"] == "COMP_v5":
            r["resid_ic_vs_comp"] = None
    return rows, {"mode": mode}, comp_z, ics


def main():
    print("A 族(截面加权/硬门限)+ B 族(时序切割),开关 = 个股独立性", flush=True)
    out = []
    for mode in ("open", "tail"):
        print(f"\n>>> {mode} 口径", flush=True)
        rows, meta, _, _ = _run(mode)
        for r in rows:
            r["mode"] = mode
        out += rows
        print(f"    完成 {len(rows)} 个因子", flush=True)

    df = pd.DataFrame(out)
    df.to_csv(LEDGER / "switch_factors.csv", index=False)
    pd.set_option("display.width", 250)

    for mode in ("open", "tail"):
        d = df[df["mode"] == mode].copy()
        base = d[d.kind == "基础"].set_index("factor")["ic_mean"]
        d["base_ic"] = d["base"].map(base)
        d["提升"] = d.ic_mean.abs() / d.base_ic.abs()
        print("\n" + "=" * 118)
        print(f"=== {mode} 口径:开关加权 vs 基础因子 ===")
        cols = ["factor", "kind", "ic_mean", "ic_t", "base_ic", "提升", "corr_vs_base",
                "year_sign_consistency", "mono_spearman", "long_excess_ann", "long_turnover"]
        show = d[d.kind != "B时序切割"].sort_values("提升", ascending=False)
        print(show[cols].head(24).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
        print("\n--- 按构造汇总(提升 = |新IC| / |基础IC|)---")
        g = d[d.kind != "基础"].groupby("kind").agg(
            n=("factor", "size"), 提升中位=("提升", "median"), 提升最大=("提升", "max"),
            与基础相关中位=("corr_vs_base", "median"), IC中位=("ic_mean", lambda s: s.abs().median()))
        print(g.to_string(float_format=lambda x: f"{x:.3f}"))
        b = d[d.kind == "B时序切割"]
        if len(b):
            print("\n--- B 族(时序切割)前 8 ---")
            print(b.sort_values("ic_mean", key=abs, ascending=False)[
                ["factor", "ic_mean", "ic_t", "year_sign_consistency",
                 "mono_spearman", "long_turnover"]].head(8)
                .to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        cv5 = d[d.factor == "COMP_v5"]
        cw = d[d.base == "COMP_v5"]
        if len(cv5) and len(cw):
            print(f"\n--- 组合层:COMP_v5 基础 IC {cv5.ic_mean.iloc[0]:.4f} ---")
            print(cw[["factor", "kind", "ic_mean", "ic_t", "提升", "corr_vs_base",
                      "mono_spearman", "long_excess_ann", "long_turnover"]]
                  .to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n产出:switch_factors.csv")


if __name__ == "__main__":
    main()
