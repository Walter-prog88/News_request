"""COMP_v5 按成员"头部表现"拆成 排雷 / 选优 两个子组合,各自回测 + Barra 归因。

## 问题
COMP_v5(25 个分钟因子等权)+ 排雷缓冲 20/35 的超额 98% 是流动性风格、残差 alpha≈0。
分钟因子的预测信息集中在坏股票那一头;**好股票那一头有没有信息,从未被正确地测过**
(评价流水线只看 1 日全区间 IC、"极端组"门槛实际是 IC 符号检查——因子原值不做方向对齐、
Top20 归因分辨率只有 ±9pp)。见 research/SELECTION_ABILITY_AUDIT.md §4 第 2/4/5 条。

## 做法
1. 账本 trials_*.jsonl 取 25 个成员(mode=open, ts 最新)的 q_excess(10 分位、h1、毛收益、
   第 1 组=因子值最低),**按 comp_v5_members.json 的 signs 做方向对齐**:
   sign=+1 时头部=第 10 组、尾部=第 1 组;sign=−1 反之。核对 sign 与 ic_mean 符号。
   head_t 用 fast_eval.eval_one 在研究期面板上重算头部组逐日超额序列得到
   (账本只存最好组的 long_t;头部不是最好组时账本没有 t)。
2. 拆分规则(主):选优组 = head_is_best 且 head_excess ≥ 0.5×|tail_excess|;排雷组 = 其余。
   备选:head_excess > 0.03。
3. FA.REGISTRY 造 25 个因子,combine.prepare + combine_equal 合成 全 / 选优 / 排雷 三份打分。
4. 每份打分四种用法(V1000, open10, rebalance=5, avg_phases=True, 全研究期):
   top n=100、top n=400(≈10%)、exclude 缓冲 20/35、exclude 30%。
5. 选优子组合 top n=400 做 Barra 归因(与 run_barra_attribution_top20.py 同一套 OLS / NW t /
   对齐断言 / 逐年);对照:排雷子组合 exclude 20/35 的归因。

只读研究期面板 Panel(),不触碰封存样本;不改动任何既有文件。

## 产出
    research/ledger/comp_split_members.csv
    research/ledger/comp_split_backtests.csv
    research/ledger/comp_split_attribution.csv
    research/r_comp_split.log(调用方 tee)
"""
from __future__ import annotations
import os, sys, json, warnings, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

from afactor.config import LEDGER
from afactor.panel import Panel
from afactor import factors as FA, combine, strategy as ST
from afactor.strategy import TRADING_DAYS
from afactor.fast_eval import EvalContext, eval_one
from run_barra_attribution import nw_t, assert_aligned
from run_attribution import nw_se_ols

pd.set_option("display.width", 240)
pd.set_option("display.max_columns", 40)

VERSION, EXEC, MODE = "V1000", "open10", "open"
REB = 5
QS = (0, 5, 10, 20)
PLANS = {
    "top100":        dict(scheme="top", n=100, rebalance=REB),
    "top400":        dict(scheme="top", n=400, rebalance=REB),
    "exclude_b20_35": dict(scheme="exclude", buffer=(0.20, 0.35), rebalance=REB),
    "exclude30":     dict(scheme="exclude", exclude_frac=0.30, rebalance=REB),
}


# ---------------------------------------------------------------- 工具
def load_ledger_latest(names: list[str], mode: str) -> pd.DataFrame:
    """所有 trials_*.jsonl,按 ts 取每个 (factor, mode) 最新一行。"""
    rows = []
    for f in sorted(LEDGER.glob("trials_*.jsonl")):
        for line in open(f):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if d.get("mode") == mode and d.get("factor") in names:
                rows.append(d)
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["ts"])
    df = df.sort_values("ts").drop_duplicates(subset=["factor", "mode"], keep="last")
    return df.set_index("factor")


def geo_ann(r: pd.Series) -> float:
    nav = (1.0 + r).cumprod()
    return float(nav.iloc[-1] ** (TRADING_DAYS / len(r)) - 1.0)


def regress(y: pd.Series, F: pd.DataFrame, styles: list, indus: list) -> dict:
    """excess_t = alpha + Σ β_k f_kt + ε_t(OLS)。与 run_barra_attribution_top20.regress 同一套。"""
    terms = styles + indus
    A = np.column_stack([np.ones(len(y)), F[terms].to_numpy(float)])
    yv = y.to_numpy(float)
    b, *_ = np.linalg.lstsq(A, yv, rcond=None)
    fitted = A @ b
    resid = yv - fitted
    r2 = 1.0 - resid.var() / yv.var()
    fann = {t: float(F[t].mean() * TRADING_DAYS) for t in terms}
    contrib = {t: float(b[i + 1] * fann[t]) for i, t in enumerate(terms)}
    alpha_series = resid + b[0]
    t_nw = {q: nw_t(alpha_series, q) for q in QS}
    se_ols = {q: float(nw_se_ols(A, resid, q)[0]) for q in QS}
    return dict(b=b, terms=terms, alpha=float(b[0] * TRADING_DAYS), r2=float(r2),
                tot=float(yv.mean() * TRADING_DAYS),
                style_sum=float(sum(contrib[t] for t in styles)),
                ind_sum=float(sum(contrib[t] for t in indus)),
                contrib=contrib, fann=fann, t_nw=t_nw,
                se_ols={q: se_ols[q] * TRADING_DAYS for q in QS},
                t_ols={q: float(b[0] / se_ols[q]) if se_ols[q] > 0 else np.nan for q in QS},
                fitted=pd.Series(fitted, index=y.index), resid=pd.Series(resid, index=y.index),
                n=len(y))


def emit(rows: list, section: str, res: dict, styles: list, indus: list, note: str = ""):
    liq = "Liquidity 流动性"
    for i, t in enumerate(res["terms"]):
        rows.append(dict(section=section, term=t, beta=res["b"][i + 1], factor_ann=res["fann"][t],
                         contrib_ann=res["contrib"][t], t=np.nan,
                         note="风格" if t in styles else "行业"))
    rows.append(dict(section=section, term="__总超额", beta=np.nan, factor_ann=np.nan,
                     contrib_ann=res["tot"], t=np.nan, note=f"算术年化, n={res['n']}"))
    rows.append(dict(section=section, term="__风格合计", beta=np.nan, factor_ann=np.nan,
                     contrib_ann=res["style_sum"], t=np.nan, note=""))
    rows.append(dict(section=section, term="__行业合计", beta=np.nan, factor_ann=np.nan,
                     contrib_ann=res["ind_sum"], t=np.nan, note=""))
    rows.append(dict(section=section, term="__alpha", beta=np.nan, factor_ann=res["se_ols"][5],
                     contrib_ann=res["alpha"], t=res["t_nw"][5],
                     note="t 为 NW q=5; factor_ann 列放 alpha 年化 SE(nw_se_ols q=5)"))
    for q in QS:
        rows.append(dict(section=section, term=f"__alpha_t_q{q}", beta=np.nan, factor_ann=np.nan,
                         contrib_ann=np.nan, t=res["t_nw"][q], note="nw_t(resid+alpha)"))
        rows.append(dict(section=section, term=f"__alpha_t_ols_q{q}", beta=np.nan,
                         factor_ann=res["se_ols"][q], contrib_ann=np.nan, t=res["t_ols"][q],
                         note="nw_se_ols 截距; factor_ann=年化 SE"))
    rows.append(dict(section=section, term="__R2", beta=np.nan, factor_ann=np.nan,
                     contrib_ann=res["r2"], t=np.nan, note=""))
    rows.append(dict(section=section, term="__Liquidity占比", beta=np.nan, factor_ann=np.nan,
                     contrib_ann=res["contrib"][liq] / res["tot"] if res["tot"] != 0 else np.nan,
                     t=np.nan, note="Liquidity 贡献 / 总超额" + (f"; {note}" if note else "")))


def print_res(tag: str, res: dict):
    liq = res["contrib"]["Liquidity 流动性"]
    print(f"  [{tag}] n={res['n']} 总超额 {res['tot']*100:+.2f}pp | 风格 {res['style_sum']*100:+.2f} "
          f"| 行业 {res['ind_sum']*100:+.2f} | alpha {res['alpha']*100:+.2f}pp "
          f"(SE {res['se_ols'][5]*100:.2f}pp) | R2 {res['r2']:.3f} "
          f"| Liquidity {liq*100:+.2f}pp ({liq/res['tot']*100 if res['tot'] else np.nan:+.0f}%)")
    print("         alpha t(nw_t) " + " ".join(f"q{q}={res['t_nw'][q]:+.2f}" for q in QS)
          + " | t(nw_se_ols) " + " ".join(f"q{q}={res['t_ols'][q]:+.2f}" for q in QS))


def run_plan(score, p, plan_name):
    kw = dict(PLANS[plan_name])
    return ST.run_strategy(score, p, VERSION, EXEC, avg_phases=True, **kw)


# ---------------------------------------------------------------- 主流程
def main():
    t_start = time.time()
    man = json.load(open(LEDGER / "comp_v5_members.json"))
    members, signs = man["members"], {k: float(v) for k, v in man["signs"].items()}
    assert len(members) == 25, len(members)

    # ================= 1. 成员头部/尾部(方向对齐) =================
    print("=" * 110 + "\n1. 成员头部 / 尾部表现(账本 q_excess,按 signs 方向对齐;h1 毛收益,年化,vs 池等权)\n" + "=" * 110)
    led = load_ledger_latest(members, MODE)
    missing = [m for m in members if m not in led.index]
    assert not missing, f"账本缺成员: {missing}"

    p = Panel(); FA.build_all()
    print(f"面板 {p.dates[0].date()} → {p.dates[-1].date()}, {len(p.dates)} 天 × {len(p.codes)} 只")
    # lit_mome_ideal_rev_bigmove:账本记录的编译表达式是裸字段 `bigmove_amt_share`(params {}),
    # 现版 DSL 不再接受规格里的 1{...} 语法,该名不在 REGISTRY 里;此前所有 COMP_v5 脚本用
    # `if n in FA.REGISTRY` 把它**静默丢掉**(实际只跑了 24 个成员)。这里按账本表达式重建,
    # 并用存档 IC 序列核对同一性。
    def _build(n):
        if n in FA.REGISTRY:
            return FA.REGISTRY[n][1](p)
        L = led.loc[n]
        expr = str(L["note"]).split(" | ")[-2].strip()
        print(f"  ⚠ {n} 不在 FA.REGISTRY;按账本编译表达式 `{expr}` 重建(裸面板字段)")
        assert expr == "bigmove_amt_share", expr
        return p[expr]
    fac = {n: _build(n).reindex(index=p.dates, columns=p.codes)
           .replace([np.inf, -np.inf], np.nan) for n in members}
    ctx = EvalContext(p, MODE, horizons=(1,))
    years = np.asarray(p.dates.year)
    rebuilt = [n for n in members if n not in FA.REGISTRY]
    from afactor.fast_eval import IC_DIR
    for n in rebuilt:
        ic, *_ = eval_one(fac[n].to_numpy(np.float32), ctx.R[1], ctx.M[1])
        ic_s = pd.Series(ic, index=p.dates)
        ref = pd.read_parquet(IC_DIR / f"{n}__{MODE}.parquet")["h1"].reindex(p.dates)
        c = float(ic_s.corr(ref))
        print(f"    {n}: 重建 IC 序列 vs 存档 IC 序列 corr={c:.4f}, 均值 {np.nanmean(ic):+.4f} vs {ref.mean():+.4f}")
        assert c > 0.99, "重建因子与账本记录不同一"

    mrows = []
    mismatches = []
    for n in members:
        L = led.loc[n]
        q = np.asarray(L["q_excess"], float)
        sgn = int(np.sign(signs[n]))
        ic_sign = int(np.sign(L["ic_mean"]))
        head_k, tail_k = (9, 0) if sgn > 0 else (0, 9)
        head_ex, tail_ex = float(q[head_k]), float(q[tail_k])
        best_k = int(np.argmax(q)); worst_k = int(np.argmin(q))
        head_is_best = best_k == head_k
        tail_is_worst = worst_k == tail_k
        # 头部组逐日超额序列 -> t(账本只有最好组的 long_t)
        ic, q_ret, cover, turn = eval_one(fac[n].to_numpy(np.float32), ctx.R[1], ctx.M[1])
        recomputed = np.nanmean(q_ret, axis=0) * TRADING_DAYS
        d_recon = float(np.nanmax(np.abs(recomputed - q)))
        hr = q_ret[:, head_k]; hr = hr[np.isfinite(hr)]
        head_t = float(hr.mean() / hr.std(ddof=1) * np.sqrt(hr.size))
        head_t_nw5 = nw_t(hr, 5)
        tr = q_ret[:, tail_k]; tr = tr[np.isfinite(tr)]
        tail_t = float(tr.mean() / tr.std(ddof=1) * np.sqrt(tr.size))
        # 头部逐年(方向对齐后)
        head_by_year = {}
        for y in sorted(set(years)):
            m = (years == y) & np.isfinite(q_ret[:, head_k])
            if m.sum() > 40:
                head_by_year[int(y)] = float(np.nanmean(q_ret[m, head_k]) * TRADING_DAYS)
        head_year_pos = float(np.mean([v > 0 for v in head_by_year.values()]))
        ic_al = float(L["ic_mean"]) * sgn
        if sgn != ic_sign:
            mismatches.append((n, sgn, float(L["ic_mean"])))
        # 拆分规则
        primary = bool(head_is_best and head_ex >= 0.5 * abs(tail_ex))
        alt = bool(head_ex > 0.03)
        mrows.append(dict(
            factor=n, family=L.get("family", ""), sign=sgn, ic_mean=float(L["ic_mean"]),
            ic_sign_match=(sgn == ic_sign), ic_aligned=ic_al,
            h5_ic_aligned=float(L.get("h5_ic_mean", np.nan)) * sgn,
            head_group=head_k + 1, tail_group=tail_k + 1,
            head_excess=head_ex, tail_excess=tail_ex, head_t=head_t, head_t_nw5=head_t_nw5,
            tail_t=tail_t, head_is_best=head_is_best, tail_is_worst=tail_is_worst,
            best_group=best_k + 1, worst_group=worst_k + 1,
            best_excess=float(q[best_k]), worst_excess=float(q[worst_k]),
            ledger_long_t=float(L.get("long_t", np.nan) or np.nan),
            head_ratio=head_ex / abs(tail_ex) if tail_ex != 0 else np.nan,
            head_year_pos=head_year_pos, head_by_year=json.dumps(head_by_year),
            q_excess_aligned=json.dumps([round(float(x), 4) for x in (q if sgn > 0 else q[::-1])]),
            q_recon_maxdiff=d_recon, ledger_ts=str(L["ts"]),
            group_primary="select" if primary else "exclude",
            group_alt="select" if alt else "exclude"))
    M = pd.DataFrame(mrows)
    print(M[["factor", "sign", "ic_mean", "ic_sign_match", "head_excess", "tail_excess", "head_t",
             "head_is_best", "tail_is_worst", "best_group", "head_ratio", "head_year_pos",
             "q_recon_maxdiff", "group_primary", "group_alt"]].to_string(
        index=False, float_format=lambda x: f"{x:+.4f}"))
    print(f"\n  q_excess 重算 vs 账本最大差(全体成员): {M.q_recon_maxdiff.max():.5f}")
    if mismatches:
        print("  ⚠ sign 与 ic_mean 符号不一致的成员:")
        for n, s, ic in mismatches:
            print(f"     {n}: sign={s:+d}, ic_mean={ic:+.4f}")
    else:
        print("  sign 与 ic_mean 符号:25/25 一致")

    sel_p = M[M.group_primary == "select"].factor.tolist()
    exc_p = M[M.group_primary == "exclude"].factor.tolist()
    sel_a = M[M.group_alt == "select"].factor.tolist()
    exc_a = M[M.group_alt == "exclude"].factor.tolist()
    print(f"\n  主规则(head_is_best 且 head ≥ 0.5|tail|): 选优 {len(sel_p)} 个 = {sel_p}")
    print(f"                                             排雷 {len(exc_p)} 个")
    print(f"  备选规则(head_excess > 0.03):            选优 {len(sel_a)} 个 = {sel_a}")
    print(f"                                             排雷 {len(exc_a)} 个")
    print(f"  两规则选优名单差异: 仅主={sorted(set(sel_p)-set(sel_a))}  仅备={sorted(set(sel_a)-set(sel_p))}")
    M.to_csv(LEDGER / "comp_split_members.csv", index=False)

    # ================= 2. 合成三份(+备选)打分 =================
    print("\n" + "=" * 110 + "\n2. 合成打分(combine.prepare 方向对齐 + 截面 zscore, combine_equal)\n" + "=" * 110)
    z = combine.prepare(fac, signs, p["universe"])
    scores = {"COMP_v5_all": combine.combine_equal(z)}
    if sel_p:
        scores["select_primary"] = combine.combine_equal({k: z[k] for k in sel_p})
    if exc_p:
        scores["exclude_primary"] = combine.combine_equal({k: z[k] for k in exc_p})
    if sel_a and set(sel_a) != set(sel_p):
        scores["select_alt"] = combine.combine_equal({k: z[k] for k in sel_a})
    if exc_a and set(exc_a) != set(exc_p):
        scores["exclude_alt"] = combine.combine_equal({k: z[k] for k in exc_a})
    # 决策池内三份打分的逐日截面 Spearman(选优 vs 排雷 子组合相关多高)
    U = ST.universe(p, VERSION).to_numpy(bool)
    def cs_spearman(a, b):
        A, B = a.to_numpy(np.float32), b.to_numpy(np.float32)
        out = []
        for ti in range(0, len(p.dates), 5):
            m = U[ti] & np.isfinite(A[ti]) & np.isfinite(B[ti])
            if m.sum() < 200:
                continue
            out.append(pd.Series(A[ti, m]).corr(pd.Series(B[ti, m]), method="spearman"))
        return float(np.nanmean(out))
    names = list(scores)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            print(f"  截面 Spearman {names[i]:16s} vs {names[j]:16s}: {cs_spearman(scores[names[i]], scores[names[j]]):+.3f}")

    # ================= 3. 回测 =================
    print("\n" + "=" * 110 + f"\n3. 回测({VERSION}, {EXEC}, rebalance={REB}, 5 相位平均, 全研究期)\n" + "=" * 110)
    bm = ST.benchmark(p, VERSION, EXEC)
    bench = bm["rets"]
    print(f"  池等权基准 几何年化 {bm['ann_return']*100:+.2f}%  n={len(bench)}")
    ir = pd.read_parquet(LEDGER / "ir_daily_series.parquet")
    ir_b = ir["open10|基准 池等权"].reindex(bench.index)
    print(f"  ir_daily_series 池等权 vs ST.benchmark 最大差 {float((ir_b - bench).abs().max()):.2e}")

    brows, series = [], {}
    print("%-18s %-16s %8s %8s %7s %8s %8s %8s %8s %6s" % (
        "score", "plan", "ann%", "excess%", "sharpe", "maxdd%", "turn/d", "cost_bp/d", "cost%/y", "hold"))
    for sname, sc in scores.items():
        for pname in PLANS:
            r = run_plan(sc, p, pname)
            rets = r["rets"]
            b = bench.reindex(rets.index)
            ex = r["ann_return"] - geo_ann(b)
            ex_arith = float((rets - b).mean() * TRADING_DAYS)
            cost_ann = r["cost_bps"] * TRADING_DAYS / 1e4
            series[(sname, pname)] = rets
            row = dict(score=sname, plan=pname, n_members=(len(members) if sname == "COMP_v5_all"
                       else len(sel_p) if sname == "select_primary" else len(exc_p) if sname == "exclude_primary"
                       else len(sel_a) if sname == "select_alt" else len(exc_a)),
                       ann=r["ann_return"], bench_ann=geo_ann(b), excess_ann=ex, excess_arith=ex_arith,
                       sharpe=r["sharpe"], max_dd=r["max_dd"], vol=r["vol"],
                       turnover=r["turnover"], cost_bps=r["cost_bps"], cost_ann=cost_ann,
                       holdings=r["holdings"], exposure=r["exposure"], n_days=r["n_days"],
                       phase_ann=json.dumps([round(a, 4) for a in r["phase_ann"]]),
                       phase_range=float(max(r["phase_ann"]) - min(r["phase_ann"])))
            # 逐年超额
            for y in sorted(set(rets.index.year)):
                m = rets.index.year == y
                row[f"excess_{y}"] = geo_ann(rets[m]) - geo_ann(b[m])
            brows.append(row)
            print("%-18s %-16s %+8.2f %+8.2f %7.2f %8.1f %8.4f %8.2f %8.2f %6.0f" % (
                sname, pname, r["ann_return"] * 100, ex * 100, r["sharpe"], r["max_dd"] * 100,
                r["turnover"], r["cost_bps"], cost_ann * 100, r["holdings"]))
    B = pd.DataFrame(brows)
    B.to_csv(LEDGER / "comp_split_backtests.csv", index=False)

    # 与 ir_daily_series 的排雷 20/35 对账(同一打分、同一用法应同源)
    ir_c = ir["open10|排雷缓冲20/35"]
    s_all = series[("COMP_v5_all", "exclude_b20_35")]
    d = float((ir_c.reindex(s_all.index) - s_all).abs().max())
    print(f"\n  全 COMP_v5(25 员) exclude 20/35 vs ir_daily_series 排雷缓冲20/35 最大日差 {d:.2e} "
          f"({'同源' if d < 1e-8 else '不同源'})")
    if d >= 1e-8 and rebuilt:
        z24 = {k: v for k, v in z.items() if k not in rebuilt}
        r24 = run_plan(combine.combine_equal(z24), p, "exclude_b20_35")
        s24 = r24["rets"]
        d24 = float((ir_c.reindex(s24.index) - s24).abs().max())
        print(f"  去掉 {rebuilt} 的 24 员版本 exclude 20/35 vs ir_daily_series 最大日差 {d24:.2e} "
              f"({'同源 → 此前账本里的 COMP_v5 排雷实际是 24 员' if d24 < 1e-8 else '仍不同源'});"
              f" 24 员 excess {r24['ann_return'] - geo_ann(bench.reindex(s24.index)):+.4f}"
              f" vs 25 员 {B[(B.score=='COMP_v5_all') & (B.plan=='exclude_b20_35')].excess_ann.iloc[0]:+.4f}")
        brows.append(dict(score="COMP_v5_24_no_bigmove", plan="exclude_b20_35", n_members=24,
                          ann=r24["ann_return"], bench_ann=geo_ann(bench.reindex(s24.index)),
                          excess_ann=r24["ann_return"] - geo_ann(bench.reindex(s24.index)),
                          sharpe=r24["sharpe"], max_dd=r24["max_dd"], vol=r24["vol"],
                          turnover=r24["turnover"], cost_bps=r24["cost_bps"],
                          cost_ann=r24["cost_bps"] * TRADING_DAYS / 1e4, holdings=r24["holdings"],
                          exposure=r24["exposure"], n_days=r24["n_days"]))
        B = pd.DataFrame(brows); B.to_csv(LEDGER / "comp_split_backtests.csv", index=False)

    # 与 usage_modes 的"选优 top10%(日频)"并列
    print("\n  --- 5 日调仓 vs 日频(usage_modes.csv 里 COMP_final 选优 top10%,BT.run_backtest,universe_tradable 全池,日频)---")
    try:
        um = pd.read_csv(LEDGER / "usage_modes.csv")
        u1 = um[(um.factor == "COMP_final") & (um["mode"] == "open") & (um.scheme == "选优 top10%")].iloc[0]
        u2 = um[(um.factor == "COMP_final") & (um["mode"] == "open") & (um.scheme == "选优 top100")].iloc[0]
        r400 = B[(B.score == "COMP_v5_all") & (B.plan == "top400")].iloc[0]
        r100 = B[(B.score == "COMP_v5_all") & (B.plan == "top100")].iloc[0]
        for lab, u, r in (("top10%/top400", u1, r400), ("top100", u2, r100)):
            print(f"  {lab:14s} 日频: excess {u.excess*100:+.2f}%  turn {u.turnover:.3f}/d  cost {u.cost_bps:.2f}bp/d "
                  f"= {u.cost_bps*TRADING_DAYS/1e4*100:.2f}%/y  hold {u.holdings:.0f}")
            print(f"  {'':14s} 5日 : excess {r.excess_ann*100:+.2f}%  turn {r.turnover:.3f}/d  cost {r.cost_bps:.2f}bp/d "
                  f"= {r.cost_ann*100:.2f}%/y  hold {r.holdings:.0f}   → 成本省 "
                  f"{(u.cost_bps - r.cost_bps)*TRADING_DAYS/1e4*100:.2f}pp/y, 超额变化 {(r.excess_ann - u.excess)*100:+.2f}pp")
        print("  注:usage_modes 的池是 universe_tradable(~4000 只、无 ADV 门槛)、日频、老引擎 BT.run_backtest;"
              "本脚本是 V1000 决策池(ADV≥2000 万)、5 日调仓、ST.run_strategy。二者不完全同池,成本差是主要可比项。")
    except Exception as e:  # noqa
        print(f"  (usage_modes 对照失败: {e})")

    # ================= 4. Barra 归因 =================
    print("\n" + "=" * 110 + "\n4. Barra 归因(open10 纯因子收益,OLS,alpha 年化×242,NW t)\n" + "=" * 110)
    F = pd.read_csv(LEDGER / "barra_factor_returns.csv", header=[0, 1], index_col=0,
                    parse_dates=True)[EXEC].astype(float)
    styles = [c for c in F.columns if not c.startswith("行业:") and c != "Country"]
    indus = [c for c in F.columns if c.startswith("行业:")]
    Fok = F.dropna(how="any")
    print(f"  纯因子收益: {len(styles)} 风格 + {len(indus)} 行业, 覆盖 {Fok.index[0].date()} → {Fok.index[-1].date()}, {len(Fok)} 天")

    arows = []
    attr_targets = [("select_primary", "top400", "select_top400"),
                    ("select_primary", "top100", "select_top100"),
                    ("exclude_primary", "exclude_b20_35", "exclude_control_b20_35"),
                    ("COMP_v5_all", "top400", "all_top400"),
                    ("COMP_v5_all", "exclude_b20_35", "all_b20_35")]
    if "select_alt" in scores:
        attr_targets.append(("select_alt", "top400", "select_alt_top400"))
    first_idx = None
    daily = {}
    for sname, pname, tag in attr_targets:
        if (sname, pname) not in series:
            continue
        strat = series[(sname, pname)]
        idx = strat.index.intersection(bench.index).intersection(Fok.index)
        if first_idx is None:
            first_idx = idx
            print(f"  回归样本 {len(idx)} 天({idx[0].date()} → {idx[-1].date()});"
                  f"策略天数 {len(strat)},被因子收益缺失剔除 {len(strat) - len(idx)} 天")
            assert_aligned(bench.reindex(idx), F.reindex(idx)["Country"], "池等权 vs Country 纯因子")
        y = (strat.reindex(idx) - bench.reindex(idx)).astype(float)
        Fi = F.reindex(idx)
        print(f"\n  ---- {tag} ({sname} × {pname}) ----")
        assert_aligned(strat.reindex(idx), bench.reindex(idx), f"{tag} vs 池等权")
        res = regress(y, Fi, styles, indus)
        print_res(tag + " 全期", res)
        emit(arows, f"{tag}_full", res, styles, indus, note=f"{sname} × {pname}")
        print("    十风格逐项(β / 纯因子年化 / 贡献 pp):")
        for t in styles:
            i = res["terms"].index(t)
            print(f"      {t:22s} β {res['b'][i+1]:+8.4f}  f_ann {res['fann'][t]*100:+7.2f}  "
                  f"contrib {res['contrib'][t]*100:+7.2f}")
        top_ind = sorted(indus, key=lambda t: -abs(res["contrib"][t]))[:5]
        print("    行业贡献绝对值前 5: " + ", ".join(f"{t[3:]} {res['contrib'][t]*100:+.2f}" for t in top_ind))
        for yv in sorted(set(idx.year)):
            m = idx.year == yv
            if m.sum() < 30:
                print(f"    {yv}: 仅 {int(m.sum())} 天,跳过逐年回归")
                continue
            ry = regress(y[m], Fi[m], styles, indus)
            print_res(f"{tag} {yv}", ry)
            emit(arows, f"{tag}_year_{yv}", ry, styles, indus)
        daily[f"{tag}_strategy"] = strat.reindex(idx)
        daily[f"{tag}_excess"] = y
        daily[f"{tag}_resid"] = res["resid"]

    A = pd.DataFrame(arows, columns=["section", "term", "beta", "factor_ann", "contrib_ann", "t", "note"])
    A.to_csv(LEDGER / "comp_split_attribution.csv", index=False)
    D = pd.DataFrame(daily); D["bench"] = bench.reindex(D.index); D.index.name = "date"
    D.to_csv(LEDGER / "comp_split_attribution_daily.csv", float_format="%.8g")

    # ================= 5. 摘要 =================
    print("\n" + "=" * 110 + "\n5. 摘要\n" + "=" * 110)
    def get(sec, term, col="contrib_ann"):
        s = A[(A.section == sec) & (A.term == term)]
        return float(s[col].iloc[0]) if len(s) else np.nan
    for tag in ["select_top400", "select_top100", "exclude_control_b20_35", "all_top400", "all_b20_35", "select_alt_top400"]:
        sec = f"{tag}_full"
        if not (A.section == sec).any():
            continue
        print(f"  {tag:24s} 总超额 {get(sec,'__总超额')*100:+.2f}pp | alpha {get(sec,'__alpha')*100:+.2f}pp "
              f"± {get(sec,'__alpha','factor_ann')*100:.2f} (t_q5 {get(sec,'__alpha','t'):+.2f}) "
              f"| R2 {get(sec,'__R2'):.3f} | Liquidity {get(sec,'Liquidity 流动性')*100:+.2f}pp "
              f"({get(sec,'__Liquidity占比')*100:+.0f}%) | 风格合计 {get(sec,'__风格合计')*100:+.2f}pp")
    print(f"\n已写 {LEDGER/'comp_split_members.csv'}, {LEDGER/'comp_split_backtests.csv'}, "
          f"{LEDGER/'comp_split_attribution.csv'}, {LEDGER/'comp_split_attribution_daily.csv'}")
    print(f"总耗时 {(time.time()-t_start)/60:.1f} 分钟")


if __name__ == "__main__":
    main()
