"""Barra 式超额归因 —— B×GBT 截面模型、Top20 等权、周频(5 日)、open10、V1000,OOS 2023–25。

## 目的

`research/BARRA_ATTRIBUTION.md` 把 COMP_v5 排雷缓冲 20/35 的六年超额几乎全部归到流动性暴露
(残差 alpha ≈ 0)。本脚本把**同一把尺子**套到模型层的选优组合上——
`research/ledger/ml_results_open_h5.csv` 里 feature_set=B, model=gbt, version=V1000, plan=Top20周频 的三行:

    2023 ann 0.1886 excess 0.1547 | 2024 ann 0.0868 excess 0.0711 | 2025 ann 0.6623 excess 0.2498

回答:Top20 的 15~25pp 超额里,剔除 Barra 十风格 + 申万一级行业之后,还剩多少 alpha?
以及同窗(2023–25)的排雷对照剩多少,二者是否只是"流动性暴露的两种浓度"。

## 方法(与 BARRA_ATTRIBUTION.md §2.2 完全同一套)

1. **重建策略逐日收益**。读 `research/ledger/scores/B_gbt_open_h5_{year}.parquet`,
   逐年 `ST.run_strategy(score, p, "V1000", "open10", date_mask=(year==y), scheme="top", n=20,
   rebalance=5, avg_phases=True)`——与 `run_ml_models.py` 写账本时的调用一字不差
   (逐年掩码、相位锚在当年首个信号日、5 相位日收益平均)。
   重建出的逐年 ann / bench / excess 必须与账本对上(±1pp),否则 raise,不出归因数。
2. **基准** = `ST.benchmark(p, "V1000", "open10", date_mask)` 池等权(与账本 bench 同源)。
3. **对齐断言**:corr(strategy_t, bench_{t+k}),k∈{−2..2} 峰值必须在 k=0。
4. **回归**:excess_t = alpha + Σ_k β_k f_{k,t} + ε_t,X = `barra_factor_returns.csv` 的 open10 口径
   (Country 之外)10 风格 + 28 行业纯因子收益,OLS;alpha 年化 ×242,t 用 nw_t(resid+alpha, q)
   (q=0/5/10/20,与原脚本同一函数);贡献 = β × 纯因子年化。
5. 逐年、逐相位(直接调 `ST._run_one_phase(..., phase=ph)` 拿 5 条相位序列)、
   同窗排雷对照(`ir_daily_series.parquet` 的 `open10|排雷缓冲20/35` 截到 2023–25)各跑一遍同一回归。
6. 打分与十个 Barra 暴露(`build_exposures(p, p["universe"])`)在 V1000 决策池内的逐日截面 Spearman。

只读研究期面板 Panel(),不触碰封存样本。

## 产出
    research/ledger/barra_top20_summary.csv   长表:section, term, beta, factor_ann, contrib_ann, t, note
    research/ledger/barra_top20_daily.csv     strategy, bench, excess, fitted, resid(+5 相位)
    research/r_barra_top20.log                由调用方 tee;脚本自身也 print 全部对账
"""
from __future__ import annotations
import os, sys, warnings
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

from afactor.config import LEDGER
from afactor.panel import Panel
from afactor import factors as FA, strategy as ST
from afactor.strategy import TRADING_DAYS
from run_barra_attribution import nw_t, assert_aligned, build_exposures
from run_attribution import nw_se_ols

pd.set_option("display.width", 240)

YEARS = [2023, 2024, 2025]
VERSION, EXEC = "V1000", "open10"
PLAN = dict(scheme="top", n=20, rebalance=5)
SCORE_DIR = LEDGER / "scores"
TOL = 0.01          # 重建对账容差(绝对,年化)
QS = (0, 5, 10, 20)


# ---------------------------------------------------------------- 工具
def load_score(year: int, p: Panel) -> pd.DataFrame:
    f = SCORE_DIR / f"B_gbt_open_h5_{year}.parquet"
    sc = pd.read_parquet(f)
    full = pd.DataFrame(np.nan, index=p.dates, columns=p.codes, dtype=np.float32)
    idx = sc.index.intersection(p.dates)
    full.loc[idx, :] = sc.reindex(index=idx, columns=p.codes).to_numpy(np.float32)
    return full


def geo_ann(r: pd.Series) -> float:
    nav = (1.0 + r).cumprod()
    return float(nav.iloc[-1] ** (TRADING_DAYS / len(r)) - 1.0)


def regress(y: pd.Series, F: pd.DataFrame, styles: list, indus: list) -> dict:
    """excess_t = alpha + Σ β_k f_kt + ε_t(OLS)。返回系数、贡献、t、拟合与残差序列。"""
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
    # 交叉核对:完整 OLS 协方差的 NW 标准误(把回归量的抽样误差一并算进去)
    se_ols = {q: float(b[0] / nw_se_ols(A, resid, q)[0]) for q in QS}
    return dict(b=b, terms=terms, alpha=float(b[0] * TRADING_DAYS), r2=float(r2),
                tot=float(yv.mean() * TRADING_DAYS),
                style_sum=float(sum(contrib[t] for t in styles)),
                ind_sum=float(sum(contrib[t] for t in indus)),
                contrib=contrib, fann=fann, t_nw=t_nw, t_ols=se_ols,
                fitted=pd.Series(fitted, index=y.index), resid=pd.Series(resid, index=y.index),
                n=len(y))


def emit(rows: list, section: str, res: dict, styles: list, indus: list, note: str = ""):
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
    rows.append(dict(section=section, term="__alpha", beta=np.nan, factor_ann=np.nan,
                     contrib_ann=res["alpha"], t=res["t_nw"][5], note="t 为 NW q=5"))
    for q in QS:
        rows.append(dict(section=section, term=f"__alpha_t_q{q}", beta=np.nan, factor_ann=np.nan,
                         contrib_ann=np.nan, t=res["t_nw"][q], note="nw_t(resid+alpha)"))
        rows.append(dict(section=section, term=f"__alpha_t_ols_q{q}", beta=np.nan, factor_ann=np.nan,
                         contrib_ann=np.nan, t=res["t_ols"][q], note="nw_se_ols 截距"))
    rows.append(dict(section=section, term="__R2", beta=np.nan, factor_ann=np.nan,
                     contrib_ann=res["r2"], t=np.nan, note=""))
    liq = "Liquidity 流动性"
    rows.append(dict(section=section, term="__Liquidity占比", beta=np.nan, factor_ann=np.nan,
                     contrib_ann=res["contrib"][liq] / res["tot"] if res["tot"] != 0 else np.nan,
                     t=np.nan, note="Liquidity 贡献 / 总超额" + (f"; {note}" if note else "")))


def print_res(tag: str, res: dict):
    liq = res["contrib"]["Liquidity 流动性"]
    print(f"  [{tag}] n={res['n']} 总超额 {res['tot']*100:+.2f}pp | 风格 {res['style_sum']*100:+.2f} "
          f"| 行业 {res['ind_sum']*100:+.2f} | alpha {res['alpha']*100:+.2f}pp | R2 {res['r2']:.3f} "
          f"| Liquidity {liq*100:+.2f}pp ({liq/res['tot']*100 if res['tot'] else np.nan:+.0f}%)")
    print("         alpha t(nw_t) " + " ".join(f"q{q}={res['t_nw'][q]:+.2f}" for q in QS)
          + " | t(nw_se_ols) " + " ".join(f"q{q}={res['t_ols'][q]:+.2f}" for q in QS))


# ---------------------------------------------------------------- 主流程
def main():
    p = Panel(); FA.build_all()
    years = np.asarray(p.dates.year)
    ledger = pd.read_csv(LEDGER / "ml_results_open_h5.csv")
    ledger = ledger[(ledger.feature_set == "B") & (ledger.model == "gbt") &
                    (ledger.version == VERSION) & (ledger.plan == "Top20周频")].set_index("oos_year")

    # ---- 1. 逐年重建策略与基准(与 run_ml_models.py 同一调用)----
    print("=" * 100 + "\n1. 重建对账(逐年掩码、5 相位平均、open10、V1000、Top20 周频)\n" + "=" * 100)
    strat_parts, bench_parts, phase_parts = [], [], {ph: [] for ph in range(5)}
    recon_rows, recon_ok = [], True
    for y in YEARS:
        sc = load_score(y, p)
        te = years == y
        bm = ST.benchmark(p, VERSION, EXEC, date_mask=te)
        r = ST.run_strategy(sc, p, VERSION, EXEC, date_mask=te, avg_phases=True, **PLAN)
        ann, bann = r["ann_return"], bm["ann_return"]
        L = ledger.loc[y]
        d_ann, d_b, d_ex = ann - L.ann, bann - L.bench, (ann - bann) - L.excess
        ok = (abs(d_ann) <= TOL) and (abs(d_b) <= TOL) and (abs(d_ex) <= TOL)
        recon_ok &= ok
        print(f"  {y}: 重建 ann {ann:+.4f} bench {bann:+.4f} excess {ann-bann:+.4f} | "
              f"账本 ann {L.ann:+.4f} bench {L.bench:+.4f} excess {L.excess:+.4f} | "
              f"Δ ann {d_ann*100:+.2f}pp Δ bench {d_b*100:+.2f}pp Δ excess {d_ex*100:+.2f}pp "
              f"| 相位 ann {[round(a, 4) for a in r['phase_ann']]} | {'OK' if ok else 'MISMATCH'}")
        recon_rows.append(dict(year=y, ann=ann, bench=bann, excess=ann - bann,
                               ledger_ann=L.ann, ledger_bench=L.bench, ledger_excess=L.excess,
                               d_ann=d_ann, d_bench=d_b, d_excess=d_ex, ok=ok))
        strat_parts.append(r["rets"]); bench_parts.append(bm["rets"])
        for ph in range(5):
            one = ST._run_one_phase(sc, p, VERSION, EXEC, PLAN["scheme"], PLAN["n"], PLAN["rebalance"],
                                    0.3, None, te, 1, ph)
            phase_parts[ph].append(one["rets"])
    if not recon_ok:
        raise AssertionError("重建与 ml_results 账本不符(>1pp),不出归因数。")
    print("  重建对账通过(逐年 ann / bench / excess 均在 ±1pp 内)。")

    strat = pd.concat(strat_parts).sort_index()
    bench = pd.concat(bench_parts).sort_index()
    phases = {ph: pd.concat(v).sort_index() for ph, v in phase_parts.items()}
    assert strat.index.is_unique and bench.index.is_unique
    # 相位平均 == run_strategy 的 rets(自检)
    ph_mean = pd.concat(phases, axis=1).mean(axis=1).reindex(strat.index)
    assert float((ph_mean - strat).abs().max()) < 1e-10, "相位平均与 run_strategy rets 不一致"

    # 与 ir_daily_series 的池等权对账(排雷对照的基准同源性)
    ir = pd.read_parquet(LEDGER / "ir_daily_series.parquet")
    ir_b = ir["open10|基准 池等权"].reindex(bench.index)
    dmax = float((ir_b - bench).abs().max())
    print(f"  ir_daily_series 池等权 vs ST.benchmark 同窗最大差 {dmax:.2e} "
          f"({'同源' if dmax < 1e-8 else '不同源!'})")

    # ---- 2. 对齐断言 ----
    print("\n" + "=" * 100 + "\n2. 对齐断言 corr(strategy_t, bench_{t+k})\n" + "=" * 100)
    assert_aligned(strat, bench, "Top20 vs 池等权(全期)")
    for y in YEARS:
        assert_aligned(strat[strat.index.year == y], bench[bench.index.year == y], f"Top20 vs 池等权 {y}")

    # ---- 3. 因子收益 ----
    F = pd.read_csv(LEDGER / "barra_factor_returns.csv", header=[0, 1], index_col=0,
                    parse_dates=True)[EXEC].astype(float)
    styles = [c for c in F.columns if not c.startswith("行业:") and c != "Country"]
    indus = [c for c in F.columns if c.startswith("行业:")]
    print(f"\n纯因子收益: {len(styles)} 风格 + {len(indus)} 行业, 覆盖 {F.index[0].date()} → {F.index[-1].date()}")
    idx = strat.index.intersection(bench.index).intersection(F.dropna(how="any").index)
    print(f"回归样本 {len(idx)} 天({idx[0].date()} → {idx[-1].date()});"
          f"策略天数 {len(strat)},被因子收益缺失剔除 {len(strat) - len(idx)} 天")
    y_all = (strat.reindex(idx) - bench.reindex(idx)).astype(float)
    Fi = F.reindex(idx)
    # 因子收益索引是否与策略/基准同在"信号日 t":池等权 vs Country 纯因子必须在 k=0 达峰
    assert_aligned(bench.reindex(idx), Fi["Country"], "池等权 vs Country 纯因子")
    ks = (-2, -1, 0, 1, 2)
    liq = Fi["Liquidity 流动性"]
    print("  [诊断] 超额 vs Liquidity 纯因子 corr: "
          + " ".join(f"{k:+d}:{float(y_all.corr(liq.shift(k))):+.3f}" for k in ks)
          + "(|corr| 峰值应在 k=0;符号为负属正常,不作断言)")

    rows = []
    # ---- (a)(e) 全期 ----
    print("\n" + "=" * 100 + "\n3. 全期 2023–25 归因\n" + "=" * 100)
    full = regress(y_all, Fi, styles, indus)
    print_res("全期", full)
    emit(rows, "full_2023_2025", full, styles, indus)
    print("\n  十风格逐项(β / 纯因子年化 / 贡献 pp):")
    for t in styles:
        i = full["terms"].index(t)
        print(f"    {t:22s} β {full['b'][i+1]:+8.4f}  f_ann {full['fann'][t]*100:+7.2f}  "
              f"contrib {full['contrib'][t]*100:+7.2f}")
    top_ind = sorted(indus, key=lambda t: -abs(full["contrib"][t]))[:5]
    print("  行业贡献绝对值前 5: " + ", ".join(f"{t[3:]} {full['contrib'][t]*100:+.2f}" for t in top_ind))

    # ---- (b) 逐年 ----
    print("\n" + "=" * 100 + "\n4. 逐年归因\n" + "=" * 100)
    for y in YEARS:
        m = idx.year == y
        res = regress(y_all[m], Fi[m], styles, indus)
        print_res(str(y), res)
        emit(rows, f"year_{y}", res, styles, indus)

    # ---- (c) 逐相位 ----
    print("\n" + "=" * 100 + "\n5. 逐相位归因(全期,5 条单相位序列)\n" + "=" * 100)
    for ph in range(5):
        yp = (phases[ph].reindex(idx) - bench.reindex(idx)).astype(float)
        res = regress(yp, Fi, styles, indus)
        print_res(f"相位{ph}", res)
        emit(rows, f"phase_{ph}", res, styles, indus)

    # ---- (d) 同窗排雷对照 ----
    print("\n" + "=" * 100 + "\n6. 同窗排雷对照 COMP_v5 排雷缓冲20/35(ir_daily_series, 2023–25)\n" + "=" * 100)
    ctl = ir["open10|排雷缓冲20/35"].reindex(idx).astype(float)
    ctl_b = ir["open10|基准 池等权"].reindex(idx).astype(float)
    ok_c = ctl.notna() & ctl_b.notna()
    print(f"  排雷序列有效 {int(ok_c.sum())} / {len(idx)} 天;"
          f"几何年化 策略 {geo_ann(ctl[ok_c]):+.4f} 池等权 {geo_ann(ctl_b[ok_c]):+.4f}")
    assert_aligned(ctl[ok_c], ctl_b[ok_c], "排雷 vs 池等权(同窗)")
    y_c = ctl[ok_c] - ctl_b[ok_c]
    ctl_res = regress(y_c, Fi[ok_c], styles, indus)
    print_res("排雷同窗", ctl_res)
    emit(rows, "control_exclude_2023_2025", ctl_res, styles, indus, note="COMP_v5 排雷缓冲20/35")
    for y in YEARS:
        m = (y_c.index.year == y)
        res = regress(y_c[m], Fi[ok_c][m], styles, indus)
        print_res(f"排雷 {y}", res)
        emit(rows, f"control_exclude_{y}", res, styles, indus)

    # ---- (f) 打分 vs Barra 暴露的截面 Spearman ----
    print("\n" + "=" * 100 + "\n7. 打分 vs 十个 Barra 暴露:逐日截面 Spearman(V1000 决策池内)\n" + "=" * 100)
    u_all = p["universe"]
    X = build_exposures(p, u_all)
    U1000 = ST.universe(p, VERSION)
    oos = np.isin(years, YEARS)
    score_full = pd.concat([load_score(y, p).loc[years == y] for y in YEARS]).reindex(p.dates)
    S = score_full.to_numpy(np.float32)
    Um = U1000.to_numpy(bool) & np.isfinite(S)
    sp_rows = []
    for k, xdf in X.items():
        Xv = xdf.reindex(index=p.dates, columns=p.codes).to_numpy(np.float64)
        cs = []
        for ti in np.flatnonzero(oos):
            m = Um[ti] & np.isfinite(Xv[ti])
            if m.sum() < 100:
                continue
            a = pd.Series(S[ti, m]).rank().to_numpy(); b = pd.Series(Xv[ti, m]).rank().to_numpy()
            cs.append(np.corrcoef(a, b)[0, 1])
        cs = np.asarray(cs)
        sp = float(np.nanmean(cs))
        by_year = {}
        # 逐年均值
        dts = p.dates[oos][:len(cs)] if len(cs) == oos.sum() else None
        print(f"    {k:22s} Spearman 均值 {sp:+.4f}  (std {np.nanstd(cs):.3f}, n={len(cs)}, "
              f"t={sp/np.nanstd(cs)*np.sqrt(len(cs)):+.1f})")
        sp_rows.append(dict(section="score_exposure_spearman", term=k, beta=np.nan, factor_ann=np.nan,
                            contrib_ann=np.nan, t=sp, note=f"逐日截面 Spearman 均值, n={len(cs)}"))
    rows.extend(sp_rows)

    # ---- 重建对账行 ----
    for rr in recon_rows:
        rows.append(dict(section="reconstruction", term=f"ann_{rr['year']}", beta=np.nan,
                         factor_ann=rr["ledger_ann"], contrib_ann=rr["ann"], t=np.nan,
                         note=f"factor_ann=账本, contrib_ann=重建, Δ={rr['d_ann']*100:+.2f}pp"))
        rows.append(dict(section="reconstruction", term=f"bench_{rr['year']}", beta=np.nan,
                         factor_ann=rr["ledger_bench"], contrib_ann=rr["bench"], t=np.nan,
                         note=f"Δ={rr['d_bench']*100:+.2f}pp"))
        rows.append(dict(section="reconstruction", term=f"excess_{rr['year']}", beta=np.nan,
                         factor_ann=rr["ledger_excess"], contrib_ann=rr["excess"], t=np.nan,
                         note=f"Δ={rr['d_excess']*100:+.2f}pp; {'OK' if rr['ok'] else 'MISMATCH'}"))

    # ---- 落盘 ----
    out = pd.DataFrame(rows, columns=["section", "term", "beta", "factor_ann", "contrib_ann", "t", "note"])
    out.to_csv(LEDGER / "barra_top20_summary.csv", index=False)
    daily = pd.DataFrame({"strategy": strat.reindex(idx), "bench": bench.reindex(idx),
                          "excess": y_all, "fitted": full["fitted"], "resid": full["resid"],
                          "control_exclude": ctl, "control_bench": ctl_b})
    for ph in range(5):
        daily[f"phase{ph}"] = phases[ph].reindex(idx)
    daily.index.name = "date"
    daily.to_csv(LEDGER / "barra_top20_daily.csv", float_format="%.8g")
    print(f"\n已写 {LEDGER/'barra_top20_summary.csv'}, {LEDGER/'barra_top20_daily.csv'}")


if __name__ == "__main__":
    main()
