"""Barra 归因的真实市值修复重跑 + 同频基准(前缀 barra2,2026-10-10)。

## 为什么要重跑
`research/TOTAL_MV_FIX.md` 修复了 `newdata.total_mv`(旧口径把限售解禁当成股本增加,研究期 13.6% 的格子市值
高估 1.5 倍以上)。`BARRA_ATTRIBUTION.md`(2026-09-02)里经 `FA._real(p, "mv")` 用到市值的地方有五处:
Size / NonlinSize 暴露、全部风格的市值加权去中心化、Beta / ResidVol 用的市值加权市场收益、截面 WLS 的
sqrt(市值) 权重、行业"市值加权和为零"约束。策略打分(COMP_v5 25 个成员)、决策池、池等权基准、成本模型都不用
`ND.total_mv`,所以超额序列 y 本身与市值口径无关,变的只是回归元(纯因子收益)。

同时把基准从"每日重置池等权"换成"同频(5 日)再平衡池等权"`ST.benchmark(..., rebalance=5)`
(fixa 审计:频率不同会把截面反转的再平衡收益记成超额,见 ENGINE_IMPACT_MODEL.md §3.1)。

## 与 run_barra_attribution.py 的关系
复用其函数(fwd_returns / build_exposures / pure_factor_returns / nw_t / assert_aligned,均来自 afactor.barra,
经 run_barra_attribution re-export),策略与回归部分与 `run_barra_attribution.main` 逐行同构(见 run_mode / attribute)。
`scripts/run_barra_attribution.py` 本身不改,默认行为不变;正式账本 barra_attribution_summary.csv /
barra_factor_returns.csv 由它在新口径下直接重跑产出(日志 logs/barra2_run_barra_attribution.log),
本脚本 compare 阶段核对"本脚本 new + 每日重置"与正式账本逐位一致。

## 阶段
  --stage selfcheck         回归函数核对:原账本纯因子收益(research/ledger/_pre_mvfix/barra_factor_returns.csv)
                            + 冻结打分 cache/_tmp/comp_v5_score_open10.parquet + 每日重置基准,
                            必须复现原 barra_attribution_summary.csv(_pre_mvfix 备份)open10 的每一行。
  --stage run --mv new      当前代码(新市值)。两个口径 open10 / tail1450,两种基准 d1 / same。
  --stage run --mv legacy   当前代码 + newdata.TOTAL_SHARE_METHOD="legacy"(诊断开关,只差 total_mv)。
                            中间结果 cache/_tmp/barra2_{mv}_{mode}_{F,rets}.parquet;
                            账本 research/ledger/barra2_attribution_{mv}.csv。
  --stage compare           汇总 -> research/ledger/barra2_attribution_summary.csv(长表,mv × bench × mode)、
                            barra2_compare.csv(原账本 / 四列对照)、barra2_checks.csv(全部核对)。
  --stage bpep              审查 P2 的敏感性(不改正式口径):BookToPrice / EarningsYield 的分母仍是近似市值
                            p["market_cap"](fundamental.build_market_cap,含分红泄漏),只把这两个暴露的分母换成
                            真实市值 FA._real(p, "mv"),y 取 run --mv new 缓存的策略与两种基准日收益,重做纯因子收益与归因。
                            账本 research/ledger/barra2_bpep_truecap.csv(长表,bpep_cap = approx / true)、
                            barra2_bpep_checks.csv(基线复现、其余暴露不变、BP/EP 暴露新旧秩相关与覆盖)。

纪律:只读研究期面板(Panel / TailPanel,<= 2025-12-31),不触碰 cache/sealed;COMP_v5 成员在 2020-2025 选出,
本文所有收益数字属信号样本内,面板不含已退市股(幸存者偏差)。不改任何预注册参数。
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)
warnings.filterwarnings("ignore")

from afactor.config import LEDGER, TMP, RESEARCH_END          # noqa: E402
from afactor.panel import Panel, TailPanel                     # noqa: E402
from afactor import factors as FA, combine, strategy as ST, newdata as ND   # noqa: E402
from afactor.strategy import TRADING_DAYS                      # noqa: E402
import run_barra_attribution as RBA                            # noqa: E402  (只 import,不执行 main)
from run_barra_attribution import (nw_t, fwd_returns, assert_aligned,      # noqa: E402
                                   build_exposures, pure_factor_returns)

MODES = RBA.MODES                  # ["open10", "tail1450"]
VERSION = RBA.VERSION              # "V1000"
PLAN = dict(RBA.PLAN)              # exclude, buffer (0.20, 0.35), rebalance 5
K_SAME = PLAN["rebalance"]         # 同频基准的再平衡周期 = 策略调仓周期
BENCHES = ("d1", "same")
BENCH_DESC = {"d1": "每日重置池等权(原口径)", "same": f"同频 rebalance={K_SAME} 池等权(推荐)"}
PRE = LEDGER / "_pre_mvfix"
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)


# ---------------------------------------------------------------- 回归(与 run_barra_attribution.main 同构)
def attribute(y: pd.Series, Fi: pd.DataFrame, mode: str) -> tuple[list[dict], dict]:
    """y = 策略 − 基准 的日超额;Fi = 同索引的纯因子收益。行格式与 barra_attribution_summary.csv 完全相同。"""
    styles = [c for c in Fi.columns if not c.startswith("行业:") and c != "Country"]
    indus = [c for c in Fi.columns if c.startswith("行业:")]
    A = np.column_stack([np.ones(len(y)), Fi[styles + indus].to_numpy()])
    b, *_ = np.linalg.lstsq(A, y.to_numpy(), rcond=None)
    resid = y.to_numpy() - A @ b
    r2 = 1.0 - resid.var() / y.to_numpy().var()

    terms = styles + indus
    contrib = {t: b[i + 1] * Fi[t].mean() * TRADING_DAYS for i, t in enumerate(terms)}
    sty_sum = sum(contrib[t] for t in styles)
    ind_sum = sum(contrib[t] for t in indus)
    alpha = b[0] * TRADING_DAYS
    tot = y.mean() * TRADING_DAYS

    rows = []
    for i, t in enumerate(terms):
        rows.append(dict(mode=mode, term=t, kind="风格" if t in styles else "行业",
                         beta=b[i + 1], factor_ann=Fi[t].mean() * TRADING_DAYS,
                         contrib_ann=contrib[t]))
    rows.append(dict(mode=mode, term="__alpha", kind="残差", beta=np.nan,
                     factor_ann=np.nan, contrib_ann=alpha))
    rows.append(dict(mode=mode, term="__总超额", kind="合计", beta=np.nan,
                     factor_ann=np.nan, contrib_ann=tot))
    rows.append(dict(mode=mode, term="__风格合计", kind="合计", beta=np.nan,
                     factor_ann=np.nan, contrib_ann=sty_sum))
    rows.append(dict(mode=mode, term="__行业合计", kind="合计", beta=np.nan,
                     factor_ann=np.nan, contrib_ann=ind_sum))
    rows.append(dict(mode=mode, term="__R2", kind="合计", beta=np.nan,
                     factor_ann=np.nan, contrib_ann=r2))
    for q in (0, 5, 10, 20):
        rows.append(dict(mode=mode, term=f"__alpha_t_q{q}", kind="合计", beta=np.nan,
                         factor_ann=np.nan, contrib_ann=nw_t(resid + b[0], q)))
    return rows, dict(tot=tot, sty=sty_sum, ind=ind_sum, alpha=alpha, r2=r2,
                      t5=nw_t(resid + b[0], 5))


# ---------------------------------------------------------------- 阶段 selfcheck
def stage_selfcheck() -> pd.DataFrame:
    """原纯因子收益 + 冻结打分 + 每日重置基准 -> 必须复现原账本 open10(验证 attribute 与原 main 同构)。"""
    F_all = pd.read_csv(PRE / "barra_factor_returns.csv", header=[0, 1], index_col=0, parse_dates=True)
    old = pd.read_csv(PRE / "barra_attribution_summary.csv")
    mode = "open10"
    p = Panel()
    score = pd.read_parquet(TMP / "comp_v5_score_open10.parquet")
    strat = ST.run_strategy(score, p, VERSION, mode, **PLAN)["rets"]
    bench = ST.benchmark(p, VERSION, mode)["rets"]
    Fi = F_all[mode].dropna(how="any").astype(float)
    idx = strat.index.intersection(bench.index).intersection(Fi.index)
    assert len(idx) == len(Fi), (len(idx), len(Fi))
    y = (strat.reindex(idx) - bench.reindex(idx)).astype(float)
    rows, _ = attribute(y, Fi.reindex(idx), mode)
    new = pd.DataFrame(rows).set_index("term")
    o = old[old["mode"] == mode].set_index("term")
    assert list(new.index) == list(o.index), "行顺序 / 项名与原账本不一致"
    d = (new["contrib_ann"] - o["contrib_ann"]).abs()
    db = (new["beta"] - o["beta"]).abs()
    out = pd.DataFrame([dict(check="selfcheck:attribute 复现原账本 open10(原 F + 冻结打分 + 每日重置)",
                             n=len(o), max_abs_diff=float(np.nanmax(np.r_[d.to_numpy(), db.to_numpy()])),
                             value=float(new.loc["__alpha", "contrib_ann"]),
                             note=f"原账本 alpha {o.loc['__alpha', 'contrib_ann']:.6f};{len(idx)} 天")])
    log(out.to_string())
    assert out["max_abs_diff"].iloc[0] < 1e-10, "attribute() 未能复现原账本"
    return out


# ---------------------------------------------------------------- 阶段 run
def run_mode(mode: str, mv: str) -> tuple[list[dict], pd.DataFrame, pd.DataFrame]:
    """一个执行口径:暴露度 -> 纯因子收益 -> 策略 -> 两种基准 -> 归因。与 run_barra_attribution.main 同序。"""
    p = TailPanel() if ST.EXEC_MODES[mode]["tail"] else Panel()
    assert p.dates.max() <= pd.Timestamp(RESEARCH_END), "面板越过研究期"
    R = fwd_returns(p, mode)
    u = p["universe"].reindex_like(R).fillna(False)

    log(f"{mv} {mode}: 构建暴露度…")
    X = build_exposures(p, u)
    log(f"  风格 {len(X)} 个: {list(X)}")
    log(f"{mv} {mode}: 逐日截面 WLS 求纯因子收益…")
    F = pure_factor_returns(R, X, u, p)
    del X
    log(f"  得到 {F.shape[0]} 天 x {F.shape[1]} 个因子")

    man = json.load(open(LEDGER / "comp_v5_members.json"))
    members, signs = man["members"], man["signs"]
    fac = {n: FA.REGISTRY[n][1](p).reindex(index=p.dates, columns=p.codes)
           .replace([np.inf, -np.inf], np.nan) for n in members if n in FA.REGISTRY}
    # 与 run_barra_attribution.main 一样静默跳过注册表里没有的成员(lit_mome_ideal_rev_bigmove 不在注册表,
    # COMP_v5 实际 24 员 —— 原账本即如此,冻结打分缓存也是 24 员;这里只记录,不改配方)
    log(f"  COMP_v5 成员 {len(members)} 个,注册表命中 {len(fac)} 个,缺 {sorted(set(members) - set(fac))}")
    score = combine.combine_equal(combine.prepare(fac, signs, p["universe"]))
    del fac
    if mode == "open10":
        score.astype(np.float64).to_parquet(TMP / f"barra2_{mv}_open10_score.parquet")
    strat = ST.run_strategy(score, p, VERSION, mode, **PLAN)["rets"]
    del score
    bench = {"d1": ST.benchmark(p, VERSION, mode)["rets"],
             "same": ST.benchmark(p, VERSION, mode, rebalance=K_SAME)["rets"]}
    for bn, bb in bench.items():
        assert_aligned(strat, bb, f"策略 vs 池等权[{bn}]")

    # 样本 = run_barra_attribution 的 idx(策略 ∩ 每日重置基准 ∩ 纯因子收益完整日);同频基准必须覆盖同一组日子
    idx = strat.index.intersection(bench["d1"].index).intersection(F.dropna(how="any").index)
    assert bench["same"].reindex(idx).notna().all(), "同频基准在回归样本内有缺失"
    Fi = F.reindex(idx).astype(float)
    rows = []
    for bn in BENCHES:
        y = (strat.reindex(idx) - bench[bn].reindex(idx)).astype(float)
        r, s = attribute(y, Fi, mode)
        for x in r:
            x.update(mv=mv, bench=bn, n_days=len(idx))
        rows += r
        log(f"  [{mv} {mode} {bn:4s}] 总超额 {s['tot']*100:+.3f}pp | 风格 {s['sty']*100:+.3f} | "
            f"行业 {s['ind']*100:+.3f} | alpha {s['alpha']*100:+.3f} (NW q5 t {s['t5']:+.2f}) | R2 {s['r2']:.4f}")
    rets = pd.DataFrame({"strat": strat, "bench_d1": bench["d1"], "bench_same": bench["same"]})
    del p
    return rows, Fi, rets


def stage_run(mv: str) -> None:
    if mv == "legacy":
        ND.TOTAL_SHARE_METHOD = "legacy"
    assert ND.TOTAL_SHARE_METHOD == {"new": "songzhuan", "legacy": "legacy"}[mv]
    log(f"newdata.TOTAL_SHARE_METHOD = {ND.TOTAL_SHARE_METHOD}")
    FA.build_all()
    rows = []
    for mode in MODES:
        r, Fi, rets = run_mode(mode, mv)
        rows += r
        Fi.to_parquet(TMP / f"barra2_{mv}_{mode}_F.parquet")
        rets.to_parquet(TMP / f"barra2_{mv}_{mode}_rets.parquet")
    cols = ["mv", "bench", "mode", "term", "kind", "beta", "factor_ann", "contrib_ann", "n_days"]
    pd.DataFrame(rows)[cols].to_csv(LEDGER / f"barra2_attribution_{mv}.csv", index=False)
    log(f"已写 {LEDGER / f'barra2_attribution_{mv}.csv'}")


# ---------------------------------------------------------------- 阶段 compare
HEAD_TERMS = ["__总超额", "__风格合计", "__行业合计", "Liquidity 流动性", "Size 市值", "NonlinSize 非线性市值",
              "__alpha", "__alpha_t_q0", "__alpha_t_q5", "__alpha_t_q10", "__alpha_t_q20", "__R2"]


def _maxdiff(a: pd.DataFrame, b: pd.DataFrame) -> float:
    return float(np.nanmax(np.abs(a.to_numpy(np.float64) - b.to_numpy(np.float64))))


def stage_compare() -> None:
    new = pd.read_csv(LEDGER / "barra2_attribution_new.csv")
    leg = pd.read_csv(LEDGER / "barra2_attribution_legacy.csv")
    allv = pd.concat([leg, new], ignore_index=True)
    allv.to_csv(LEDGER / "barra2_attribution_summary.csv", index=False)

    orig = pd.read_csv(PRE / "barra_attribution_summary.csv")
    fixa = pd.read_csv(LEDGER / "fixa_r2_barra_delta.csv")
    off = pd.read_csv(LEDGER / "barra_attribution_summary.csv")          # run_barra_attribution.py 新口径重跑
    checks = []

    # 1) 正式账本(run_barra_attribution.py 重跑)== 本脚本 new + d1
    for mode in MODES:
        a = off[off["mode"] == mode].set_index("term")
        b = new[(new["mode"] == mode) & (new["bench"] == "d1")].set_index("term")
        same_terms = list(a.index) == list(b.index)
        checks.append(dict(check=f"正式账本 barra_attribution_summary.csv == barra2 new/d1 [{mode}]",
                           n=len(a), max_abs_diff=_maxdiff(a[["beta", "contrib_ann"]].fillna(0),
                                                           b[["beta", "contrib_ann"]].fillna(0)) if same_terms else np.nan,
                           value=np.nan, note=f"项名与顺序一致={same_terms}"))

    # 2) 正式纯因子收益 == 本脚本 new 的 Fi;列名 / 日期与旧账本一致
    F_off = pd.read_csv(LEDGER / "barra_factor_returns.csv", header=[0, 1], index_col=0, parse_dates=True)
    F_old = pd.read_csv(PRE / "barra_factor_returns.csv", header=[0, 1], index_col=0, parse_dates=True)
    checks.append(dict(check="barra_factor_returns.csv 列(MultiIndex)与旧账本完全一致", n=F_off.shape[1],
                       max_abs_diff=np.nan, value=float(list(F_off.columns) == list(F_old.columns)),
                       note="1=一致"))
    checks.append(dict(check="barra_factor_returns.csv 日期索引与旧账本完全一致", n=len(F_off),
                       max_abs_diff=np.nan, value=float(F_off.index.equals(F_old.index)),
                       note=f"新 {len(F_off)} 行 [{F_off.index[0].date()}~{F_off.index[-1].date()}],旧 {len(F_old)} 行"))
    for mv in ("new", "legacy"):
        for mode in MODES:
            Fb = pd.read_parquet(TMP / f"barra2_{mv}_{mode}_F.parquet")
            if mv == "new":
                Fo = F_off[mode].dropna(how="all").reindex(Fb.index)
                checks.append(dict(check=f"正式 barra_factor_returns.csv[{mode}] == barra2 new Fi", n=len(Fb),
                                   max_abs_diff=_maxdiff(Fo[Fb.columns], Fb), value=np.nan, note=""))
            Fo = F_old[mode].dropna(how="all")
            com = Fo.index.intersection(Fb.index)
            d = (Fb.loc[com, Fo.columns].astype(float) - Fo.loc[com].astype(float))
            for c in ["Country", "Size 市值", "NonlinSize 非线性市值", "Beta 市场敏感", "ResidVol 残差波动",
                      "Liquidity 流动性", "BookToPrice 价值"]:
                checks.append(dict(check=f"纯因子收益 {mv} vs 原账本(09-02)[{mode}] {c}", n=len(com),
                                   max_abs_diff=float(d[c].abs().max()),
                                   value=float(Fb.loc[com, c].corr(Fo.loc[com, c].astype(float))),
                                   note="value=日序列相关"))

    # 3) 新口径 open10 纯因子收益 == 风险模型影子重跑的原版逻辑缓存 risk_open10_F_orig.parquet(promote 后为新市值)
    Fr = pd.read_parquet(TMP / "risk_open10_F_orig.parquet")
    Fb = pd.read_parquet(TMP / "barra2_new_open10_F.parquet")
    com = Fb.index.intersection(Fr.index)
    checks.append(dict(check="barra2 new open10 Fi == cache/_tmp/risk_open10_F_orig.parquet(run_risk_model 影子新口径)",
                       n=len(com), max_abs_diff=_maxdiff(Fr.loc[com, Fb.columns], Fb.loc[com]), value=np.nan,
                       note="同一库函数 pure_factor_returns + 新市值"))

    # 4) 策略 / 基准与市值口径无关(new vs legacy 的 y 逐位相同)
    for mode in MODES:
        a = pd.read_parquet(TMP / f"barra2_new_{mode}_rets.parquet")
        b = pd.read_parquet(TMP / f"barra2_legacy_{mode}_rets.parquet")
        checks.append(dict(check=f"策略与两种基准日收益 new == legacy [{mode}]", n=len(a),
                           max_abs_diff=_maxdiff(a, b.reindex(a.index)), value=np.nan, note="y 与市值口径无关"))
    # 5) open10 打分 == fixb 打分缓存;与冻结打分的差(fixb 财报 PIT)
    s_new = pd.read_parquet(TMP / "barra2_new_open10_score.parquet")
    s_leg = pd.read_parquet(TMP / "barra2_legacy_open10_score.parquet")
    checks.append(dict(check="open10 COMP_v5 打分 new == legacy", n=int(s_new.notna().sum().sum()),
                       max_abs_diff=_maxdiff(s_new, s_leg.reindex_like(s_new)), value=np.nan, note=""))
    for nm in ("comp_v5_score_open10_fixb.parquet", "comp_v5_score_open10.parquet"):
        s_c = pd.read_parquet(TMP / nm).reindex_like(s_new)
        checks.append(dict(check=f"open10 COMP_v5 打分(当前代码)vs cache/_tmp/{nm}", n=int(s_new.notna().sum().sum()),
                           max_abs_diff=_maxdiff(s_new, s_c), value=np.nan,
                           note="NaN 位置一致" if s_new.isna().equals(s_c.isna()) else "NaN 位置不一致"))
    # 6) 同频基准复核 fixa_r2_barra_delta.csv:原 F + 冻结打分下的 same 结果(已由 fixa 记录)只作引用,不重算

    ck = pd.DataFrame(checks)
    ck.to_csv(LEDGER / "barra2_checks.csv", index=False)
    print(ck.to_string(max_colwidth=90))

    # ---- 对照表:原账本(09-02,旧市值 + 每日重置)、fixa(旧市值 + 同频,原 F)、当前代码四列 ----
    def pick(df, mode, term, col="contrib_ann"):
        s = df[(df["mode"] == mode) & (df["term"] == term)][col]
        return float(s.iloc[0]) if len(s) else np.nan

    def fixa_val(mode, term):
        f = fixa[fixa["mode"] == mode]
        if term in ("__总超额", "__alpha", "__风格合计", "__行业合计", "Liquidity 流动性"):
            return float(f[f["term"] == term]["new_pp"].iloc[0]) / 100
        if term.startswith("__alpha_t_q") and mode == "open10":
            q = term.replace("__alpha_t_q", "")
            return float(f[f["term"] == f"复算 bench=same:alpha t(NW q{q})"]["new_pp"].iloc[0])
        return np.nan

    rows = []
    for mode in MODES:
        for term in HEAD_TERMS:
            r = dict(mode=mode, term=term,
                     orig_d1=pick(orig, mode, term), orig_same_fixa=fixa_val(mode, term))
            for mv in ("legacy", "new"):
                for bn in BENCHES:
                    d = allv[(allv["mv"] == mv) & (allv["bench"] == bn)]
                    r[f"{mv}_{bn}"] = pick(d, mode, term)
            rows.append(r)
        # 流动性占总超额
        r = dict(mode=mode, term="流动性占总超额")
        r["orig_d1"] = pick(orig, mode, "Liquidity 流动性") / pick(orig, mode, "__总超额")
        r["orig_same_fixa"] = fixa_val(mode, "Liquidity 流动性") / fixa_val(mode, "__总超额")
        for mv in ("legacy", "new"):
            for bn in BENCHES:
                d = allv[(allv["mv"] == mv) & (allv["bench"] == bn)]
                r[f"{mv}_{bn}"] = pick(d, mode, "Liquidity 流动性") / pick(d, mode, "__总超额")
        rows.append(r)
        # 主要风格 β 与纯因子年化
        for term in ("Liquidity 流动性", "Size 市值", "NonlinSize 非线性市值"):
            for col, lab in (("beta", "β"), ("factor_ann", "纯因子年化")):
                r = dict(mode=mode, term=f"{term} {lab}", orig_d1=pick(orig, mode, term, col), orig_same_fixa=np.nan)
                for mv in ("legacy", "new"):
                    for bn in BENCHES:
                        d = allv[(allv["mv"] == mv) & (allv["bench"] == bn)]
                        r[f"{mv}_{bn}"] = pick(d, mode, term, col)
                rows.append(r)
    cmp_ = pd.DataFrame(rows)
    cmp_["delta_mv_same"] = cmp_["new_same"] - cmp_["legacy_same"]
    cmp_["delta_mv_d1"] = cmp_["new_d1"] - cmp_["legacy_d1"]
    cmp_["note"] = ("单位:年化小数(t / R2 / 占比除外);orig_d1 = 2026-09-02 原账本(_pre_mvfix 备份);"
                    "orig_same_fixa = fixa_r2_barra_delta.csv(原纯因子收益 + 同频基准);"
                    "legacy/new = 当前代码,只差 newdata.TOTAL_SHARE_METHOD;d1 = 每日重置池等权,same = 同频 rebalance=5")
    cmp_.to_csv(LEDGER / "barra2_compare.csv", index=False)
    show = cmp_.drop(columns="note").copy()
    print(show.to_string(float_format=lambda v: f"{v:,.4f}"))
    log(f"已写 barra2_attribution_summary.csv / barra2_compare.csv / barra2_checks.csv")


# ---------------------------------------------------------------- 阶段 bpep(审查 P2 敏感性)
BPEP = ("BookToPrice 价值", "EarningsYield 盈利")


def _rowwise_rank_corr(a: pd.DataFrame, b: pd.DataFrame, u: pd.DataFrame) -> pd.Series:
    """逐日截面秩相关(只在 u 内、两边都有值的格子上)。"""
    m = u & a.notna() & b.notna()
    ra = a.where(m).rank(axis=1)
    rb = b.where(m).rank(axis=1)
    ra = ra.sub(ra.mean(axis=1), axis=0)
    rb = rb.sub(rb.mean(axis=1), axis=0)
    return (ra * rb).sum(axis=1) / np.sqrt((ra ** 2).sum(axis=1) * (rb ** 2).sum(axis=1))


def stage_bpep() -> None:
    """BP / EP 分母:近似市值(正式口径)vs 真实市值。其余一切(y、其余 8 个风格、WLS 权重、行业约束)不变。"""
    assert ND.TOTAL_SHARE_METHOD == "songzhuan", "敏感性在新市值口径上做"
    FA.build_all()
    ref = pd.read_csv(LEDGER / "barra2_attribution_new.csv")
    rows, checks = [], []
    for mode in MODES:
        p = TailPanel() if ST.EXEC_MODES[mode]["tail"] else Panel()
        assert p.dates.max() <= pd.Timestamp(RESEARCH_END), "面板越过研究期"
        R = fwd_returns(p, mode)
        u = p["universe"].reindex_like(R).fillna(False)
        F_ref = pd.read_parquet(TMP / f"barra2_new_{mode}_F.parquet")
        rets = pd.read_parquet(TMP / f"barra2_new_{mode}_rets.parquet")
        idx = F_ref.index

        # (a) 基线:近似市值分母(正式口径),必须复现 run --mv new
        log(f"bpep {mode}: 基线暴露(BP/EP 分母 = p['market_cap'] 近似市值)…")
        X0 = build_exposures(p, u)
        F0 = pure_factor_returns(R, X0, u, p)
        checks.append(dict(mode=mode, check="基线纯因子收益(本阶段重算)== cache barra2_new_F",
                           n=len(idx), max_abs_diff=_maxdiff(F0.reindex(idx)[F_ref.columns], F_ref), value=np.nan,
                           note="近似市值分母 = 正式口径"))
        apx, tru = p["market_cap"], FA._real(p, "mv")
        ratio = (apx / tru).where(u.reindex_like(apx).fillna(False)).to_numpy(np.float64)
        pct = np.nanpercentile(ratio, [10, 50, 90])
        checks.append(dict(mode=mode, check="池内 近似市值 / 真实市值 的分位数 p10/p50/p90", n=int(np.isfinite(ratio).sum()),
                           max_abs_diff=np.nan, value=float(pct[1]),
                           note=f"p10 {pct[0]:.4f} / p50 {pct[1]:.4f} / p90 {pct[2]:.4f}"))
        del ratio

        # (b) 只换 BP / EP 的分母:fu_bp / fu_ep 在注册表里读 p["market_cap"](factors.py _build_fundamental 的 cap)
        log(f"bpep {mode}: BP/EP 分母换成真实市值 FA._real(p,'mv') …")
        p._derived["market_cap"] = tru
        X1 = build_exposures(p, u)
        assert list(X1) == list(X0), "风格顺序变了"
        for k in X0:
            a, b = X0[k].reindex_like(R), X1[k].reindex_like(R)
            if k in BPEP:
                rc = _rowwise_rank_corr(a, b, u)
                n0, n1 = int((a.notna() & u).to_numpy().sum()), int((b.notna() & u).to_numpy().sum())
                checks.append(dict(mode=mode, check=f"{k} 暴露 近似 vs 真实分母:逐日截面秩相关中位数", n=int(rc.notna().sum()),
                                   max_abs_diff=np.nan, value=float(rc.median()),
                                   note=f"p10 {rc.quantile(0.1):.4f};池内非空格子 近似 {n0} / 真实 {n1}"))
            else:
                same_nan = bool(a.isna().equals(b.isna()))
                checks.append(dict(mode=mode, check=f"其余风格暴露不变:{k}", n=int(a.notna().to_numpy().sum()),
                                   max_abs_diff=float(np.nanmax(np.abs(a.to_numpy(np.float64) - b.to_numpy(np.float64)))),
                                   value=float(same_nan), note="value=NaN 位置一致"))
        del X0
        F1 = pure_factor_returns(R, X1, u, p)
        del X1, p
        F1i = F1.reindex(idx).astype(float)
        checks.append(dict(mode=mode, check="真实分母下纯因子收益在原回归样本上完整", n=len(idx),
                           max_abs_diff=np.nan, value=float(F1i.notna().all().all()),
                           note=f"真实分母完整日 {len(F1.dropna(how='any'))},原 {len(idx)}"))
        assert F1i.notna().all().all(), "真实分母下回归样本有缺失日"
        for c in ("Country", "Liquidity 流动性", "Size 市值", "NonlinSize 非线性市值") + BPEP:
            checks.append(dict(mode=mode, check=f"纯因子收益 真实分母 vs 近似分母:{c}", n=len(idx),
                               max_abs_diff=float((F1i[c] - F_ref[c]).abs().max()),
                               value=float(F1i[c].corr(F_ref[c])),
                               note=f"value=日序列相关;年化 {F_ref[c].mean()*TRADING_DAYS:+.4%} → {F1i[c].mean()*TRADING_DAYS:+.4%}"))

        for bn in BENCHES:
            y = (rets["strat"].reindex(idx) - rets[f"bench_{bn}"].reindex(idx)).astype(float)
            for cap_lab, Fi in (("approx", F_ref.astype(float)), ("true", F1i)):
                r, s = attribute(y, Fi, mode)
                for x in r:
                    x.update(mv="new", bench=bn, bpep_cap=cap_lab, n_days=len(idx))
                rows += r
                log(f"  [{mode} {bn:4s} BP/EP 分母={cap_lab:6s}] 总超额 {s['tot']*100:+.3f}pp | 风格 {s['sty']*100:+.3f} | "
                    f"行业 {s['ind']*100:+.3f} | alpha {s['alpha']*100:+.3f} (NW q5 t {s['t5']:+.2f}) | R2 {s['r2']:.4f}")
                if cap_lab == "approx":
                    a = pd.DataFrame(r).set_index("term")
                    b = ref[(ref["mode"] == mode) & (ref["bench"] == bn)].set_index("term")
                    checks.append(dict(mode=mode, check=f"基线归因 == barra2_attribution_new.csv [{bn}]", n=len(b),
                                       max_abs_diff=_maxdiff(a.loc[b.index, ["beta", "contrib_ann"]].fillna(0),
                                                             b[["beta", "contrib_ann"]].fillna(0)),
                                       value=np.nan, note=f"项名一致={list(a.index) == list(b.index)}"))
        del F0, F1
    cols = ["mv", "bench", "bpep_cap", "mode", "term", "kind", "beta", "factor_ann", "contrib_ann", "n_days"]
    out = pd.DataFrame(rows)[cols]
    out.to_csv(LEDGER / "barra2_bpep_truecap.csv", index=False)
    ck = pd.DataFrame(checks)
    ck.to_csv(LEDGER / "barra2_bpep_checks.csv", index=False)
    print(ck.to_string(max_colwidth=100))
    # 摘要:alpha / t / BP / EP / 风格合计 的变化
    key = ["__alpha", "__alpha_t_q0", "__alpha_t_q5", "__alpha_t_q10", "__alpha_t_q20", "__风格合计", "__行业合计",
           "Liquidity 流动性"] + list(BPEP)
    pv = out[out["term"].isin(key)].pivot_table(index=["mode", "bench", "term"], columns="bpep_cap",
                                                values="contrib_ann", aggfunc="first")
    pv["delta"] = pv["true"] - pv["approx"]
    print(pv.to_string(float_format=lambda v: f"{v:+.5f}"))
    log("已写 barra2_bpep_truecap.csv / barra2_bpep_checks.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["selfcheck", "run", "compare", "bpep"], required=True)
    ap.add_argument("--mv", choices=["new", "legacy"], default="new")
    a = ap.parse_args()
    if a.stage == "selfcheck":
        ck = stage_selfcheck()
        ck.to_csv(LEDGER / "barra2_selfcheck.csv", index=False)
    elif a.stage == "run":
        stage_run(a.mv)
    elif a.stage == "bpep":
        stage_bpep()
    else:
        stage_compare()
    log("done")
