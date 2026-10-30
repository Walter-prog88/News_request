"""第一步:长历史换手率版排雷(前缀 lh_step1;预注册 research/LONG_HISTORY_PREREG.md §10/§11/§14)。

阶段(按顺序,可分开跑;日志 logs/lh_step1.log,耗时 research/ledger/lh_step1_timing.csv):
  verify    冻结哈希校验(lh_prereg_manifest.json),不一致即退出
  prep      LongPanel -> cache/_tmp/lh_step1_inputs/*.npy(信号、池、计价价、成本面板;不算收益),约 15 秒
  bridge    研究面板 2021-2025:引擎断言 5(流水线复现冻结序列)、B1(换手率版 vs COMP_v5)、B2(口径近似误差)
  runs      长历史全部配置(多进程 --workers,默认 5);结果 cache/_tmp/lh_step1_runs/<配置>.pkl
  report    账本 research/ledger/lh_step1_*.csv 等、cache/_tmp/lh_step1_daily.parquet、reports/lh_step1_nav.png
  review    审查修复(2026-10-11):lh_step1_review.csv(对照口径 x 区间、残差分解、Q9 逐年复核、幸存者上界、
            桥接解释份额、S-a 冷启动、2005-2009 持仓贡献分解),约 1 分钟;不改任何预注册判定
  speed     实测第二步 callable vs ndarray(W x pos) 单相位耗时与一致性,写入 lh_step1_timing.csv

用法: python3 scripts/lh_step1.py verify prep bridge runs report review [--workers 5] [--only name1,name2]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from afactor import lh_engine as LE
from afactor.config import LEDGER, TMP

LOG = ROOT / "logs" / "lh_step1.log"
RUNS = TMP / "lh_step1_runs"
TIMING = LEDGER / "lh_step1_timing.csv"
CHECKS = LEDGER / "lh_engine_checks.csv"
BRIDGE = LEDGER / "lh_bridge.csv"
BRIDGE_DAILY = TMP / "lh_step1_bridge_daily.parquet"
TD = 242


def log(*a):
    s = time.strftime("%H:%M:%S") + " " + " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.parent.mkdir(exist_ok=True)
    with open(LOG, "a") as f:
        f.write(s + "\n")


def timing(stage: str, sec: float, note: str = ""):
    row = pd.DataFrame([dict(stage=stage, seconds=round(sec, 1), minutes=round(sec / 60, 2), note=note,
                             at=time.strftime("%Y-%m-%d %H:%M:%S"))])
    old = pd.read_csv(TIMING) if TIMING.exists() else pd.DataFrame()
    if len(old):
        old = old[old["stage"] != stage]
    pd.concat([old, row], ignore_index=True).to_csv(TIMING, index=False)


def merge_csv(path: Path, df: pd.DataFrame, key: str):
    old = pd.read_csv(path) if path.exists() else pd.DataFrame()
    if len(old) and key in old.columns:
        old = old[~old[key].isin(df[key])]
    pd.concat([old, df], ignore_index=True).to_csv(path, index=False)


def corr_peak(rs: pd.Series, rb: pd.Series, ks=range(-3, 4)):
    a, b = rs.align(rb, join="inner")
    cs = {int(k): float(a.corr(b.shift(k))) for k in ks}
    return max(cs, key=cs.get), cs


# ================================================================ verify
def stage_verify():
    man = json.loads((LEDGER / "lh_prereg_manifest.json").read_text())
    bad = [f for f, h in man["sha256"].items() if hashlib.sha256((ROOT / f).read_bytes()).hexdigest() != h]
    if bad:
        log("[verify] 冻结哈希不一致,停止:", bad)
        sys.exit(2)
    log(f"[verify] 预注册冻结哈希一致({len(man['sha256'])} 个文件)")
    merge_csv(CHECKS, pd.DataFrame([dict(check="prereg_hash", step=1, value=0, ok=True,
                                         note=f"{len(man['sha256'])} 个冻结文件 sha256 与 manifest 一致")]), "check")


# ================================================================ prep
def stage_prep():
    t0 = time.time()
    meta = LE.prepare_inputs(log=log)
    timing("prep", time.time() - t0, f"T={len(meta['dates'])} N={len(meta['codes'])}")


# ================================================================ bridge(研究面板 2021-2025)
def _round2(x):
    return np.floor(np.asarray(x, np.float64) * 100.0 + 0.5 + 1e-9) / 100.0


def _open_at_limit(p) -> pd.DataFrame:
    """研究面板上的 §6.2 规则:不复权开盘集合竞价价 == 涨(跌)停价(半个价位内)。
    不复权价 = 前复权价 x (当日不复权收盘 / 前复权收盘);除权参考价 = 前一日前复权收盘 x 当日比例。
    涨跌幅 L 用 Panel 的规则(创业板 2020-08-24 前 10%、ST 只把 10% 降到 5%、北交所 30%),
    另按 LongPanel 口径把主板 ST 自 2025-07-07 起恢复 10%。"""
    from afactor import newdata as ND
    from afactor.panel import _limit_frame
    close = p.raw("close_px").to_numpy(np.float64)
    rawc = ND.raw_close(p.dates, p.codes).reindex(index=p.dates, columns=p.codes).to_numpy(np.float64)
    with np.errstate(all="ignore"):
        f = rawc / close
        o = _round2(p.raw("open_auction_px").to_numpy(np.float64) * f)
        prev = np.vstack([np.full((1, close.shape[1]), np.nan), close[:-1]])
        pre = _round2(prev * f)
    L = _limit_frame(p).astype(np.float64)
    main = np.array([c.startswith(("60", "00")) for c in p.codes])
    st = p["is_st"].reindex_like(L).fillna(False).to_numpy(bool)
    late = np.asarray(p.dates >= pd.Timestamp("2025-07-07"))[:, None]
    Lv = L.to_numpy()
    Lv = np.where(late & main[None, :] & st & (Lv == 0.05), 0.10, Lv)
    with np.errstate(all="ignore"):
        up = _round2(pre * (1 + Lv)); dn = _round2(pre * (1 - Lv))
        dead = (np.abs(o - up) < 0.005) | (np.abs(o - dn) < 0.005)
    return pd.DataFrame(dead & np.isfinite(o), index=p.dates, columns=p.codes)


def _pool92(p) -> pd.DataFrame:
    valid = p["valid"].to_numpy(bool)
    dl = p["days_listed"].to_numpy(np.float64)
    adv = p["adv20"].to_numpy(np.float64)
    shsz = np.array([c.startswith(("60", "68", "00", "30")) for c in p.codes])
    elig = valid & (dl >= 6) & np.isfinite(adv) & shsz[None, :]
    return pd.DataFrame(LE._rowwise_quantile_mask(adv, elig, 0.92), index=p.dates, columns=p.codes)


def _bridge_stats(r_comp: pd.Series, r_turn: pd.Series, b: pd.Series, W_comp, W_turn, U, s_comp, s_turn,
                  mask) -> dict:
    import statsmodels.api as sm
    ec = (r_comp - b).dropna(); et = (r_turn - b).dropna()
    ec, et = ec.align(et, join="inner")
    rho = float(ec.corr(et))
    xc = LE.ann_geo(r_comp) - LE.ann_geo(b); xt = LE.ann_geo(r_turn) - LE.ann_geo(b)
    X = sm.add_constant(et.to_numpy())
    fit = sm.OLS(ec.to_numpy(), X).fit(cov_type="HAC", cov_kwds=dict(maxlags=5))
    # 剔除集合 Jaccard(mask 内、池内、两边打分都有限的股票;剔除 = 权重 0)
    Uv = U.to_numpy(bool)
    sc = s_comp.reindex(index=U.index, columns=U.columns).to_numpy(np.float64)
    stn = s_turn.reindex(index=U.index, columns=U.columns).to_numpy(np.float64)
    both = Uv & np.isfinite(sc) & np.isfinite(stn) & mask[:, None]
    exc = both & (W_comp <= 0); ext = both & (W_turn <= 0)
    inter = (exc & ext).sum(1); uni = (exc | ext).sum(1)
    ok = uni > 0
    jac = float(np.mean(inter[ok] / uni[ok]))
    # 打分截面秩相关
    rs = []
    for t in np.flatnonzero(mask):
        m = both[t]
        if m.sum() < 50:
            continue
        a = pd.Series(sc[t, m]).rank().to_numpy(); c = pd.Series(stn[t, m]).rank().to_numpy()
        rs.append(np.corrcoef(a, c)[0, 1])
    return dict(rho_excess=rho, excess_comp=xc, excess_turn=xt, excess_ratio=xt / xc if xc else np.nan,
                jaccard_excluded=jac, excl_frac_comp=float(exc.sum() / both.sum()),
                excl_frac_turn=float(ext.sum() / both.sum()),
                reg_alpha_ann=float(fit.params[0] * TD), reg_alpha_nw_t=float(fit.tvalues[0]),
                reg_beta=float(fit.params[1]), reg_r2=float(fit.rsquared),
                score_rank_corr_mean=float(np.mean(rs)) if rs else np.nan,
                nw_t5_comp=LE.nw_t((r_comp - b).dropna().to_numpy(), 5),
                nw_t5_turn=LE.nw_t((r_turn - b).dropna().to_numpy(), 5))


def _grade_b1(st: dict) -> str:
    rho, ratio, xt = st["rho_excess"], st["excess_ratio"], st["excess_turn"]
    if xt <= 0:
        return "代理不成立"
    if rho >= 0.6 and 0.5 <= ratio <= 1.5:
        return "代理成立"
    if (0.3 <= rho < 0.6) or (0.25 <= ratio < 0.5) or (rho >= 0.6 and 0.25 <= ratio < 0.5):
        return "部分代理"
    return "代理不成立"


def stage_bridge():
    import m5_bridge as MB
    from afactor import strategy as ST, minute5 as M5, factors as FA
    from afactor.panel import Panel
    from afactor.impact import CostModel
    t_all = time.time()
    D = pd.read_parquet(LEDGER / "freeze_ref_daily.parquet")
    checks, rows, daily = [], [], {}
    key = "__eng_cost__20_10_0.01_tick"
    kw = dict(scheme="exclude", buffer=(0.20, 0.35))
    FA.build_all()

    # ---------------- C 口径(冻结序列所在面板:5 分钟信号 + 5 分钟执行)----------------
    t0 = time.time()
    p5 = M5.m5_panel(signal=True, exec_=True)
    s5 = pd.read_parquet(MB.SCORE5).reindex(index=p5.dates, columns=p5.codes)
    m5 = MB.year_mask(p5.dates, 2021, 2025)
    cm5 = CostModel(p5, impact_y=1.0)
    W5c = ST.build_weights(s5, p5, "V1000", date_mask=m5, **kw)
    ref, ref_old, refb = (D["T1_excl|Y1|strategy"].dropna(), D["T1_excl|old|strategy"].dropna(),
                          D["C|bench_pool5"].dropna())
    r_def = ST.run_weights(W5c, p5, "open10", 1e7, rebalance=5, date_mask=m5, cost_model=cm5)
    r_hook = ST.run_weights(W5c, p5, "open10", 1e7, rebalance=5, date_mask=m5, cost_model=cm5,
                            stamp_fn=LE.StampFnCurrent(p5.dates))
    ad5 = LE.LHAdapter(p5.raw("twap_open10"), p5.raw("amt_open10"), p5["one_word_up"] | p5["one_word_down"],
                       cost_panels=p5._derived[key])
    r_ad = LE.run(W5c, ad5, rebalance=5, cost="Y1", date_mask=m5, stamp="current")
    r_ad_old = LE.run(W5c, ad5, rebalance=5, cost="old", date_mask=m5, stamp="current")
    b5 = ST.benchmark(p5, "V1000", "open10", date_mask=m5, rebalance=5)["rets"]
    U5 = ST.universe(p5, "V1000")
    b5_ad = LE.bench(ad5, U5, 5, m5)

    def mad(a, b):
        a, b = a.align(b, join="outer")
        return float((a - b).abs().max())
    for nm, a, b_, tol in (("E5_default_engine_vs_frozen_T1_Y1", r_def["rets"], ref, 1e-10),
                           ("E5_stamp_hook_current_vs_default_bitwise", r_hook["rets"], r_def["rets"], 0.0),
                           ("E5_adapter_pipeline_vs_frozen_T1_Y1", r_ad["rets"], ref, 1e-10),
                           ("E5_adapter_pipeline_vs_frozen_T1_old", r_ad_old["rets"], ref_old, 1e-10),
                           ("E5_bench_engine_vs_frozen_pool5", b5, refb, 1e-10),
                           ("E5_bench_adapter_vs_frozen_pool5", b5_ad, refb, 1e-10)):
        d = mad(a, b_)
        checks.append(dict(check=nm, step=1, value=d, ok=bool(d <= tol),
                           note=f"逐日最大绝对差,容差 {tol:g};冻结序列 research/ledger/freeze_ref_daily.parquet"))
        log(f"[bridge] {nm}: {d:.2e} {'OK' if d <= tol else 'FAIL'}")
    log(f"[bridge] C 口径流水线一致性 {time.time() - t0:.0f}s")

    # B1(C 口径,主):换手率版 rt_turn_20 vs 冻结 COMP_v5
    st5 = FA.REGISTRY["rt_turn_20"][1](p5).reindex(index=p5.dates, columns=p5.codes).replace([np.inf, -np.inf], np.nan)
    W5t = ST.build_weights(st5, p5, "V1000", date_mask=m5, **kw)
    rt_y1 = ST.run_weights(W5t, p5, "open10", 1e7, rebalance=5, date_mask=m5, cost_model=cm5)["rets"]
    rt_old = ST.run_weights(W5t, p5, "open10", 1e7, rebalance=5, date_mask=m5)["rets"]
    for cost, rc, rtn in (("Y1", ref, rt_y1), ("old", ref_old, rt_old)):
        s = _bridge_stats(rc, rtn, refb, W5c, W5t, U5, s5, st5, m5)
        s.update(test="B1", panel="C 5m信号+5m执行(冻结口径)", cost=cost, grade=_grade_b1(s),
                 ann_comp=LE.ann_geo(rc), ann_turn=LE.ann_geo(rtn), ann_bench=LE.ann_geo(refb))
        rows.append(s)
        log(f"[bridge] B1-C {cost}: rho={s['rho_excess']:.3f} ratio={s['excess_ratio']:.2f} "
            f"comp {s['excess_comp'] * 100:+.2f}pp turn {s['excess_turn'] * 100:+.2f}pp -> {s['grade']}")
    daily.update({"C|comp_Y1": ref, "C|turn_Y1": rt_y1, "C|comp_old": ref_old, "C|turn_old": rt_old,
                  "C|bench_pool5": refb})

    # ---------------- A 口径(项目 1 分钟 Panel;comp_v5_score_open10.parquet)----------------
    t0 = time.time()
    p = Panel()
    sA = pd.read_parquet(TMP / "comp_v5_score_open10.parquet").reindex(index=p.dates, columns=p.codes)
    mA = MB.year_mask(p.dates, 2021, 2025)
    cmA = CostModel(p, impact_y=1.0)
    stA = FA.REGISTRY["rt_turn_20"][1](p).reindex(index=p.dates, columns=p.codes).replace([np.inf, -np.inf], np.nan)
    UA = ST.universe(p, "V1000")
    WAc = ST.build_weights(sA, p, "V1000", date_mask=mA, **kw)
    WAt = ST.build_weights(stA, p, "V1000", date_mask=mA, **kw)
    rAc = ST.run_weights(WAc, p, "open10", 1e7, rebalance=5, date_mask=mA, cost_model=cmA)["rets"]
    rAt = ST.run_weights(WAt, p, "open10", 1e7, rebalance=5, date_mask=mA, cost_model=cmA)["rets"]
    bA = ST.benchmark(p, "V1000", "open10", date_mask=mA, rebalance=5)["rets"]
    s = _bridge_stats(rAc, rAt, bA, WAc, WAt, UA, sA, stA, mA)
    s.update(test="B1", panel="A 1m 原版(comp_v5_score_open10)", cost="Y1", grade=_grade_b1(s),
             ann_comp=LE.ann_geo(rAc), ann_turn=LE.ann_geo(rAt), ann_bench=LE.ann_geo(bA))
    rows.append(s)
    log(f"[bridge] B1-A Y1: rho={s['rho_excess']:.3f} ratio={s['excess_ratio']:.2f} "
        f"comp {s['excess_comp'] * 100:+.2f}pp turn {s['excess_turn'] * 100:+.2f}pp -> {s['grade']}")
    daily.update({"A|comp_Y1": rAc, "A|turn_Y1": rAt, "A|bench_pool5": bA})

    # ---------------- B2:长历史口径逐项替换(A 面板,换手率版,Y=1)----------------
    adA = LE.LHAdapter(p.raw("twap_open10"), p.raw("amt_open10"), p["one_word_up"] | p["one_word_down"],
                       cost_panels=p._derived[key])
    base = LE.run(WAt, adA, rebalance=5, cost="Y1", date_mask=mA, stamp="current")["rets"]
    checks.append(dict(check="B2_adapter_baseline_equals_engine", step=1, value=mad(base, rAt),
                       ok=bool(mad(base, rAt) <= 1e-10), note="A 面板 LHAdapter 原口径 vs ST.run_weights"))
    V_auc = p.raw("open_auction_px")
    amt114 = (p.raw("amount") * np.float32(LE.OPEN10_RATIO_MAIN)).astype(np.float32)
    dead_lim = _open_at_limit(p)
    dead_ow = p["one_word_up"] | p["one_word_down"]
    U92 = _pool92(p)
    variants = {
        "base(项目口径)": (p.raw("twap_open10"), p.raw("amt_open10"), dead_ow, UA),
        "i 开盘集合竞价价": (V_auc, p.raw("amt_open10"), dead_ow, UA),
        "ii 0.114x全日成交额": (p.raw("twap_open10"), amt114, dead_ow, UA),
        "iii 开盘触板不可成交": (p.raw("twap_open10"), p.raw("amt_open10"), dead_lim, UA),
        "iv 92%ADV分位池(沪深)": (p.raw("twap_open10"), p.raw("amt_open10"), dead_ow, U92),
        "v 四项全换": (V_auc, amt114, dead_lim, U92),
    }
    b2 = {}
    for nm, (V, amt, dd, U) in variants.items():
        ad = LE.LHAdapter(V, amt, dd, cost_panels=p._derived[key])
        W = WAt if U is UA else ST.build_weights(stA, p, "V1000", date_mask=mA, universe_override=U, **kw)
        r = LE.run(W, ad, rebalance=5, cost="Y1", date_mask=mA, stamp="current")["rets"]
        b = LE.bench(ad, U, 5, mA)
        b2[nm] = dict(ann_strategy=LE.ann_geo(r), ann_bench=LE.ann_geo(b), excess=LE.ann_geo(r) - LE.ann_geo(b),
                      nw_t5=LE.nw_t((r - b).dropna().to_numpy(), 5))
        daily[f"B2|{nm}|strategy"] = r; daily[f"B2|{nm}|bench"] = b
        log(f"[bridge] B2 {nm}: excess {b2[nm]['excess'] * 100:+.2f}pp")
    ex0 = b2["base(项目口径)"]["excess"]
    for nm, v in b2.items():
        rows.append(dict(test="B2", panel="A 1m", cost="Y1", variant=nm, **v, diff_vs_base=v["excess"] - ex0,
                         grade=("近似可接受" if abs(v["excess"] - ex0) <= 0.01 else "口径误差单列")
                         if nm.startswith("v ") else ""))
    log(f"[bridge] A 口径 B1/B2 {time.time() - t0:.0f}s")
    # 研究面板 B2(v) 的序列留给 B3(report 阶段与 LongPanel 幸存股比较)
    pd.DataFrame({k: v for k, v in daily.items()}).to_parquet(BRIDGE_DAILY)
    merge_csv(CHECKS, pd.DataFrame(checks), "check")
    out = pd.DataFrame(rows)
    out.insert(0, "row", [f"{r.get('test')}|{r.get('panel')}|{r.get('cost')}|{r.get('variant', '')}" for r in rows])
    merge_csv(BRIDGE, out, "row")
    timing("bridge", time.time() - t_all, "研究面板 2021-2025:E5 + B1(C/A) + B2")


# ================================================================ runs(长历史配置)
BASE = dict(signal="S1", buffer=(0.20, 0.35), excl=0.3, k=5, pool="q92", dead="open", ratio="r114",
            cost="Y1", comm=1.0, disc=0.0, val="close", record=False, period="full", weights=None, group="prereg")


def _cfg(**kw):
    d = dict(BASE); d.update(kw); return d


BUFFERS = {"b10_25": (0.10, 0.25), "b20_35": (0.20, 0.35), "b30_45": (0.30, 0.45), "nobuf30": None}
CONFIGS = {
    "main": _cfg(record=True),
    "main_old": _cfg(cost="old"), "main_pess": _cfg(cost="pess"), "main_comm5": _cfg(comm=5.0),
    "main_d50": _cfg(disc=0.5), "main_d100": _cfg(disc=1.0), "main_vidx": _cfg(val="index"),
    "main_k20": _cfg(k=20), "main_q80": _cfg(pool="q80"), "main_q100": _cfg(pool="q100"),
    "main_exST": _cfg(pool="q92_exST"), "main_list60": _cfg(pool="q92_list60"), "main_ow": _cfg(dead="ow"),
    "main_r061": _cfg(ratio="r061"),
    "surv2025": _cfg(pool="q92_surv2025"),
    "fullpool_ew": _cfg(weights="fullpool", record=True),
    "p2125_full": _cfg(period="2125"), "p2125_survpanel": _cfg(period="2125", pool="q92_survpanel"),
}
for _s in ("S1", "S2", "S3"):
    for _bn, _b in BUFFERS.items():
        if _s == "S1" and _bn == "b20_35":
            continue
        CONFIGS[f"grid_{_s}_{_bn}"] = _cfg(signal=_s, buffer=_b)
for _seed in range(5):
    CONFIGS[f"placebo_{_seed}"] = _cfg(weights=f"placebo{_seed}")
# 事后诊断(E3 逐年未过后加跑;不进任何判定,见 lh_deviations.csv):去掉可成交约束,检查引擎记账恒等式并分解残差来源
CONFIGS["diag_fp_free"] = _cfg(weights="fullpool", record=True, dead="none", ratio="free", group="diag")
CONFIGS["diag_fp_nolimit"] = _cfg(weights="fullpool", record=True, dead="none", group="diag")
CONFIGS["diag_fp_nosusp"] = _cfg(weights="fullpool", record=True, ratio="susp_free", group="diag")
CONFIGS["diag_main_free"] = _cfg(dead="none", ratio="free", group="diag")


def run_one(name: str) -> dict:
    c = CONFIGS[name]
    t0 = time.time()
    X = LE.load_inputs()
    rows = None
    mask = None
    if c["period"] == "2125":
        i0 = int(X.dates.searchsorted(pd.Timestamp("2020-01-01")))
        rows = slice(i0, None)
    ad = LE.adapter(X, discount=c["disc"], valuation=c["val"], dead=c["dead"], ratio=c["ratio"], rows=rows)
    dates = ad.dates
    if c["period"] == "2125":
        mask = np.asarray((dates.year >= 2021) & (dates.year <= 2025))
    U = LE.pool(X, c["pool"], rows)
    if c["weights"] is None:
        W = LE.weights(X, c["signal"], c["buffer"], c["pool"], c["excl"], rows=rows, date_mask=mask)
    elif c["weights"] == "fullpool":
        Ua = U.to_numpy(bool)
        if mask is not None:
            Ua = Ua & mask[:, None]
        n = Ua.sum(1, keepdims=True)
        W = np.divide(Ua.astype(np.float32), n, out=np.zeros(Ua.shape, np.float32), where=n > 0)
    elif str(c["weights"]).startswith("placebo"):
        seed = int(str(c["weights"])[7:])
        s = np.random.default_rng(seed).random(len(X.codes)).astype(np.float32)
        S = pd.DataFrame(np.tile(s, (len(dates), 1)), index=dates, columns=X.codes)
        W = LE.weights(X, "S1", c["buffer"], c["pool"], c["excl"], rows=rows, score=S, date_mask=mask)
    else:
        raise ValueError(c["weights"])
    res = LE.run(W, ad, rebalance=c["k"], cost=c["cost"], commission_bps=c["comm"], date_mask=mask,
                 record=c["record"])
    b = LE.bench(ad, U, c["k"], mask)
    out = dict(name=name, cfg=c, rets=res["rets"], phase_rets=res["phase_rets"], bench=b,
               turnover=res["turnover"], cost_bps=res["cost_bps"], exposure=res["exposure"],
               holdings=res["holdings"], elapsed=time.time() - t0)
    if c["record"]:
        rec = res["record"]
        cols = ["ret", "gross_ret", "turnover", "cost_total", "cost_commission", "cost_stamp", "cost_spread",
                "cost_impact", "exposure", "cash", "n_hold", "stuck_w", "unfilled", "part_cut"]
        out["rec_daily"] = rec["daily"][cols]
        out["max_exposure_phase"] = float(rec["daily_by_phase"]["exposure"].max())
    RUNS.mkdir(parents=True, exist_ok=True)
    with open(RUNS / f"{name}.pkl", "wb") as f:
        pickle.dump(out, f)
    return dict(name=name, elapsed=out["elapsed"])


def stage_runs(workers: int, only: list[str] | None):
    import concurrent.futures as cf
    import multiprocessing as mp
    names = [n for n in CONFIGS if (only is None or n in only)]
    # 慢的先跑(20 相位、record)
    names.sort(key=lambda n: -(CONFIGS[n]["k"] * (2 if CONFIGS[n]["record"] else 1)))
    t0 = time.time()
    log(f"[runs] {len(names)} 个配置,{workers} 进程")
    done = []
    with cf.ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as ex:
        futs = {ex.submit(run_one, n): n for n in names}
        for f in cf.as_completed(futs):
            r = f.result()
            done.append(r)
            log(f"[runs] {r['name']} {r['elapsed']:.0f}s ({len(done)}/{len(names)})")
    sec = time.time() - t0
    serial = sum(r["elapsed"] for r in done)
    timing("runs" if only is None else f"runs_partial_{len(names)}", sec,
           f"{len(names)} 配置 x 5/20 相位;{workers} 进程;单进程累计 {serial / 60:.1f} 分钟")
    log(f"[runs] 完成 {sec:.0f}s(单进程累计 {serial:.0f}s)")


# ================================================================ report
def load_run(name: str) -> dict:
    with open(RUNS / f"{name}.pkl", "rb") as f:
        return pickle.load(f)


def _ev(r: pd.Series, a=LE.EVAL_START, b="2025-12-31") -> pd.Series:
    return r.loc[pd.Timestamp(a): pd.Timestamp(b)]


PERIODS = {"full": (LE.EVAL_START, "2025-12-31"), "dev": LE.DEV, "test": LE.TEST}


def index_frame(dates: pd.DatetimeIndex) -> tuple[pd.DataFrame, pd.DataFrame]:
    """指数:收盘(窗口用)与按信号日对齐的开盘对开盘收益(te 开盘 -> te+1 开盘;中证1000 缺开盘时用 close[t]->close[t+1])。"""
    from afactor import newdata as ND
    x = pd.read_csv(ROOT / "cache/_tmp/lh_prereg_index_prices.csv", parse_dates=["date"])
    close = x.pivot(index="date", columns="code", values="close").reindex(dates)
    opn = x.pivot(index="date", columns="code", values="open").reindex(dates)
    nd = ND.index_daily()
    c852 = close["sh.000852"].copy()
    fill = nd["000852"].reindex(dates)
    # 2011 缺口:用项目冻结快照按比例衔接
    gap = c852.isna() & fill.notna()
    if gap.any():
        c852[gap] = fill[gap]
    close["sh.000852"] = c852
    oo = opn.shift(-2) / opn.shift(-1) - 1.0
    cc_alt = close.shift(-1) / close - 1.0
    oo = oo.where(oo.notna(), cc_alt)
    names = {"sh.000001": "SSE", "sh.000300": "CSI300", "sh.000905": "CSI500", "sh.000852": "CSI1000",
             "sz.399317": "GZ_A"}
    return close.rename(columns=names), oo.rename(columns=names)


def summary_rows(name: str, R: dict, rec_cost: pd.DataFrame | None = None) -> list[dict]:
    out = []
    rs, rb = R["rets"], R["bench"]
    for pn, (a, b) in PERIODS.items():
        s = LE.slice_eval(rs, a, b); bb = LE.slice_eval(rb, a, b)
        if len(s) < 50:
            continue
        ps, pb = LE.perf(s), LE.perf(bb)
        row = dict(config=name, period=pn, start=str(s.index[0].date()), end=str(s.index[-1].date()))
        row.update({f"s_{k}": v for k, v in ps.items()})
        row.update({f"b_{k}": v for k, v in pb.items()})
        row.update(LE.rel_block(s, bb))
        y = LE.yearly(s, bb)
        row["year_t"] = LE.year_t(y["excess"]) if len(y) >= 3 else np.nan
        row["n_years"] = len(y)
        row["turnover_ann_full"] = R["turnover"] * TD
        row["cost_ann_full"] = R["cost_bps"] * 1e-4 * TD
        row["exposure_avg"] = R["exposure"]
        row["holdings_avg"] = R["holdings"]
        if rec_cost is not None:
            rc = LE.slice_eval(rec_cost, a, b)
            for c in ("cost_total", "cost_commission", "cost_stamp", "cost_spread", "cost_impact"):
                row[f"{c}_ann"] = float(rc[c].mean() * TD)
            row["turnover_ann"] = float(rc["turnover"].mean() * TD)
            row["exposure_avg_period"] = float(rc["exposure"].mean())
        out.append(row)
    return out


def window_rows(name: str, R: dict, close_idx: pd.DataFrame, which: dict, tag: str,
                refs: dict | None = None) -> list[dict]:
    """refs(审查修复 2026-10-11):额外对照序列 {名: 日收益},加列 ret_<名>、rel_diff_vs_<名>。
    fp = 同引擎全池等权(付成本、受停牌/触板约束,可投资);fpg = 它的毛收益(不计成本)。"""
    rs, rb = R["rets"], R["bench"]
    dates = rs.index
    out = []
    for w, (a, b) in which.items():
        ix = LE.window_rows(dates, a, b)
        if len(ix) == 0:
            continue
        s = rs.iloc[ix]; bb = rb.reindex(s.index)
        ret_s = float((1 + s).prod() - 1); ret_b = float((1 + bb).prod() - 1)
        row = dict(config=name, set=tag, window=w, start=a, end=b, n_days=len(ix), ret_strategy=ret_s,
                   ret_bench=ret_b, rel_diff=ret_s - ret_b, rel_ratio=(1 + ret_s) / (1 + ret_b) - 1,
                   mdd_strategy=LE.max_dd(s)[0], mdd_bench=LE.max_dd(bb)[0])
        for rn, rr in (refs or {}).items():
            ret_r = float((1 + rr.reindex(s.index)).prod() - 1)
            row[f"ret_{rn}"] = ret_r
            row[f"rel_diff_vs_{rn}"] = ret_s - ret_r
        for c in close_idx.columns:
            ca, cb = close_idx[c].get(pd.Timestamp(a), np.nan), close_idx[c].get(pd.Timestamp(b), np.nan)
            row[f"idx_{c}"] = float(cb / ca - 1) if np.isfinite(ca) and np.isfinite(cb) else np.nan
        out.append(row)
    return out


def loo_rows(name: str, R: dict) -> list[dict]:
    rs, rb = _ev(R["rets"]), _ev(R["bench"])
    out = []
    for w in (None,) + LE.BEARS:
        keep = np.ones(len(rs), bool)
        if w is not None:
            a, b = LE.WINDOWS_A[w]
            ix = LE.window_rows(rs.index, a, b)
            keep[ix] = False
        s, bb = rs[keep], rb.reindex(rs.index)[keep]
        rel = LE.rel_block(s, bb)
        out.append(dict(config=name, dropped=w or "(none)", n_days=int(keep.sum()), ann_strategy=LE.ann_geo(s),
                        ann_bench=LE.ann_geo(bb), excess_geo=rel["excess_geo"], nw_t5=rel["nw_t5"],
                        mdd_strategy=LE.max_dd(s)[0]))
    return out


def grade_step1(summ: pd.DataFrame, wins: pd.DataFrame, loo: pd.DataFrame, yearly_ex: pd.Series, cfg: str) -> dict:
    f = summ[(summ.config == cfg) & (summ.period == "full")].iloc[0]
    dv = summ[(summ.config == cfg) & (summ.period == "dev")].iloc[0]
    ts = summ[(summ.config == cfg) & (summ.period == "test")].iloc[0]
    bears = wins[(wins.config == cfg) & (wins.set == "A") & wins.window.isin(LE.BEARS)]
    l = loo[(loo.config == cfg) & (loo.dropped != "(none)")]
    a = (f.excess_geo > 0) and (f.nw_t5 >= 2.0) and (LE.year_t(yearly_ex) >= 1.73)
    b = (dv.excess_geo > 0) and (ts.excess_geo > 0)
    c = bool((l.excess_geo > 0).all())
    d = (int((bears.rel_diff >= 0).sum()) >= 4) and bool((bears.rel_diff >= -0.05).all())
    if f.excess_geo <= 0 or f.nw_t5 < 1.0:
        g = "不支持"
    elif a and b and c and d:
        g = "长历史支持"
    else:
        g = "部分支持"
    return dict(config=cfg, grade=g, a_full_t=bool(a), b_dev_test=bool(b), c_loo=bool(c), d_bears=bool(d),
                full_excess=f.excess_geo, nw_t5=f.nw_t5, year_t=LE.year_t(yearly_ex),
                dev_excess=dv.excess_geo, test_excess=ts.excess_geo, loo_min_excess=float(l.excess_geo.min()),
                bears_nonneg=int((bears.rel_diff >= 0).sum()), bears_min_rel=float(bears.rel_diff.min()),
                mdd_full=f.s_mdd, mdd_gt_50=bool(f.s_mdd < -0.50))


def stage_report():
    t0 = time.time()
    names = [n for n in CONFIGS if (RUNS / f"{n}.pkl").exists()]
    Rs = {n: load_run(n) for n in names}
    log(f"[report] 读入 {len(Rs)} 个配置")
    main = Rs["main"]
    dates_full = main["rets"].index
    close_idx, oo_idx = index_frame(LE.load_inputs().dates)
    rec = main["rec_daily"]
    rec_fp = Rs["fullpool_ew"]["rec_daily"]

    # ---------------- 引擎断言 ----------------
    checks = []
    for nm in ("main", "fullpool_ew"):
        mx = Rs[nm]["max_exposure_phase"]
        checks.append(dict(check=f"E1_exposure_le_1[{nm}]", step=1, value=mx, ok=bool(mx <= 1 + 1e-9),
                           note="逐相位逐日 exposure 最大值(引擎内硬断言 newh.sum() <= 1 + 1e-9 同时生效)"))
    for nm in ("main", "fullpool_ew", "main_d100", "surv2025"):
        k, cs = corr_peak(_ev(Rs[nm]["rets"]), _ev(Rs[nm]["bench"]))
        checks.append(dict(check=f"E2_align_peak_k0[{nm}]", step=1, value=k, ok=bool(k == 0),
                           note="corr(策略, B1.shift(k)) k∈[-3,3]: " + ", ".join(f"{kk}:{v:.3f}" for kk, v in cs.items())))
    fp = Rs["fullpool_ew"]
    s, b = _ev(fp["rets"]), _ev(fp["bench"])
    ex = LE.ann_geo(s) - LE.ann_geo(b)
    cost_ann = float(_ev(rec_fp["cost_total"]).mean() * TD)
    checks.append(dict(check="E3_fullpool_ew_excess_plus_cost_full", step=1, value=ex + cost_ann,
                       ok=bool(abs(ex + cost_ann) <= 0.01),
                       note=f"2006-2025 几何超额 {ex * 100:+.2f}pp,年成本 {cost_ann * 100:.2f}pp,残差门槛 ±1pp"))
    y = LE.yearly(s, b)
    yc = _ev(rec_fp["cost_total"]).groupby(_ev(rec_fp["cost_total"]).index.year).sum()
    resid = (y["excess"] + yc.reindex(y.index)).rename("resid")
    worst = resid.abs().idxmax()
    checks.append(dict(check="E3_fullpool_ew_excess_plus_cost_yearly_max", step=1, value=float(resid.abs().max()),
                       ok=bool(resid.abs().max() <= 0.02),
                       note=f"逐年 |超额 + 成本| 最大 {resid.abs().max() * 100:.2f}pp({worst});2015 年 {resid.get(2015, np.nan) * 100:+.2f}pp;"
                            "逐年值见 lh_step1_yearly.csv config=fullpool_ew"))
    pl = [LE.ann_geo(_ev(Rs[f"placebo_{i}"]["rets"])) - LE.ann_geo(_ev(Rs[f"placebo_{i}"]["bench"]))
          for i in range(5) if f"placebo_{i}" in Rs]
    pl_cost = [Rs[f"placebo_{i}"]["cost_bps"] * 1e-4 * TD for i in range(5) if f"placebo_{i}" in Rs]
    checks.append(dict(check="E4_placebo_centered_on_minus_cost", step=1, value=float(np.mean(pl) + np.mean(pl_cost)),
                       ok=bool(abs(np.mean(pl) + np.mean(pl_cost)) <= 0.01),
                       note=f"5 个固定随机打分:超额 " + ", ".join(f"{x * 100:+.2f}" for x in pl)
                            + f" pp;平均成本 {np.mean(pl_cost) * 100:.2f}pp(全期含 2005);判据 |均值 + 成本| <= 1pp(本文设定)"))
    # ---- E3 事后诊断:去掉可成交约束后的记账恒等式与残差分解(lh_step1_e3_diag.csv)----
    drows = []
    for n, lab in (("fullpool_ew", "主口径(停牌 + 开盘触板不可成交)"), ("diag_fp_nolimit", "只保留停牌约束"),
                   ("diag_fp_nosusp", "只保留开盘触板约束(停牌按陈旧价可成交)"), ("diag_fp_free", "无可成交约束")):
        if n not in Rs:
            continue
        R = Rs[n]
        s_, b_ = _ev(R["rets"]), _ev(R["bench"])
        cc = _ev(R["rec_daily"]["cost_total"])
        yy = LE.yearly(s_, b_)
        yy["cost"] = cc.groupby(cc.index.year).sum()
        yy["excess_plus_cost"] = yy["excess"] + yy["cost"]
        # 对数口径:Σ log(1+r_s) - Σ log(1+r_b) + Σ 成本 ≈ 0 才是不受年内复利放大的恒等式(年收益 ±60~200% 时算术差会被放大)
        lr = (np.log1p(s_) - np.log1p(b_)).groupby(s_.index.year).sum()
        yy["logrel_plus_cost"] = lr + yy["cost"]
        for yr, r_ in yy.iterrows():
            drows.append(dict(config=n, variant=lab, period=str(yr), excess=r_.excess, cost=r_.cost,
                              excess_plus_cost=r_.excess_plus_cost, logrel_plus_cost=r_.logrel_plus_cost))
        for pn, (a, b) in PERIODS.items():
            sa, ba, ca = LE.slice_eval(s_, a, b), LE.slice_eval(b_, a, b), LE.slice_eval(cc, a, b)
            e = LE.ann_geo(sa) - LE.ann_geo(ba)
            drows.append(dict(config=n, variant=lab, period=pn, excess=e, cost=float(ca.mean() * TD),
                              excess_plus_cost=e + float(ca.mean() * TD)))
    for n, ref_ in [("main", "fullpool_ew"), ("diag_main_free", "diag_fp_free")] + \
            [(f"placebo_{i}", "fullpool_ew") for i in range(5)]:
        if n in Rs and ref_ in Rs:
            for pn, (a, b) in PERIODS.items():
                sa = LE.slice_eval(Rs[n]["rets"], a, b); fa = LE.slice_eval(Rs[ref_]["rets"], a, b)
                ba = LE.slice_eval(Rs[n]["bench"], a, b)
                rel = LE.rel_block(sa, fa)
                drows.append(dict(config=f"{n} vs {ref_}", variant="策略相对同引擎全池等权(事后对照)", period=pn,
                                  excess=rel["excess_geo"], nw_t5=rel["nw_t5"],
                                  excess_vs_b1=LE.ann_geo(sa) - LE.ann_geo(ba)))
    dd_ = pd.DataFrame(drows)
    dd_.to_csv(LEDGER / "lh_step1_e3_diag.csv", index=False)
    def _ymax(cfg, col, excl2015=False):
        fr = dd_[(dd_.config == cfg) & dd_.period.str.isdigit()]
        if excl2015:
            fr = fr[fr.period != "2015"]
        i = fr[col].abs().idxmax()
        return float(fr.loc[i, col]), fr.loc[i, "period"]
    if "fullpool_ew" in Rs:
        v, yr = _ymax("fullpool_ew", "excess_plus_cost", True)
        checks.append(dict(check="E3_fullpool_ew_yearly_max_excl2015", step=1, value=v, ok=bool(abs(v) <= 0.02),
                           note=f"预注册字面口径(算术年超额 + 年成本),2015 单列后最大 {v * 100:+.2f}pp({yr})"))
        v, yr = _ymax("fullpool_ew", "logrel_plus_cost", True)
        checks.append(dict(check="E3_fullpool_ew_yearly_max_excl2015_log", step=1, value=v, ok=bool(abs(v) <= 0.02),
                           note=f"事后:对数口径(去掉年内复利放大),2015 单列后最大 {v * 100:+.2f}pp({yr});"
                                f"2015 = {float(dd_[(dd_.config == 'fullpool_ew') & (dd_.period == '2015')].logrel_plus_cost.iloc[0]) * 100:+.2f}pp"))
    for cfg, lab, tol in (("diag_fp_free", "无可成交约束(引擎记账恒等式)", 0.002),
                          ("diag_fp_nosusp", "只保留开盘触板约束", 0.005),
                          ("diag_fp_nolimit", "只保留停牌约束", 0.02)):
        if cfg in Rs:
            v, yr = _ymax(cfg, "logrel_plus_cost")
            checks.append(dict(check=f"E3_diag_{cfg}_log_yearly_max", step=1, value=v, ok=bool(abs(v) <= tol),
                               note=f"事后诊断:全池等权 {lab},对数口径逐年 |超额 + 成本| 最大 {v * 100:+.2f}pp({yr}),"
                                    f"门槛 {tol * 100:.1f}pp(诊断用,不是预注册门槛)"))
    merge_csv(CHECKS, pd.DataFrame(checks), "check")
    for c in checks:
        log(f"[report] {c['check']}: {c['value']:.4g} {'OK' if c['ok'] else 'FAIL'}")

    # ---------------- 汇总 / 窗口 / 逐年 / LOO ----------------
    summ = []
    for n, R in Rs.items():
        if CONFIGS[n]["period"] == "2125" or CONFIGS[n]["group"] == "diag":
            continue
        summ += summary_rows(n, R, rec if n == "main" else (rec_fp if n == "fullpool_ew" else None))
    summ = pd.DataFrame(summ)
    summ.to_csv(LEDGER / "lh_step1_summary.csv", index=False)

    wcfgs = ["main", "main_old", "main_d50", "main_d100", "main_vidx", "surv2025", "main_q100", "fullpool_ew"]
    # 审查修复(P1):窗口相对收益另给"同引擎全池等权"对照(只对与 fullpool_ew 同适配器的 main / main_old 有意义)
    fp_refs = None
    if "fullpool_ew" in Rs:
        fp_refs = {"fp": Rs["fullpool_ew"]["rets"], "fpg": rec_fp["gross_ret"].reindex(dates_full)}
    wins = []
    for n in wcfgs:
        if n not in Rs:
            continue
        R = Rs[n]
        rf = fp_refs if n in ("main", "main_old") else None
        wins += window_rows(n, R, close_idx, LE.WINDOWS_A, "A", rf)
        wins += window_rows(n, R, close_idx, LE.WINDOWS_B, "B", rf)
        wins += window_rows(n, R, close_idx, LE.WINDOWS_ORIG, "orig", rf)
        vw = {}
        for v, tr in LE.V_TROUGHS.items():
            for h in (60, 120):
                vw[f"{v}_{h}"] = LE.v_window(R["rets"].index, tr, h)
        wins += window_rows(n, R, close_idx, vw, "V", rf)
    wins = pd.DataFrame(wins)
    wins.to_csv(LEDGER / "lh_step1_windows.csv", index=False)

    yl = []
    for n in ("main", "main_old", "main_d50", "main_d100", "surv2025", "fullpool_ew", "main_vidx"):
        if n not in Rs:
            continue
        y = LE.yearly(_ev(Rs[n]["rets"]), _ev(Rs[n]["bench"]))
        y["config"] = n
        if n == "fullpool_ew":
            cc = _ev(rec_fp["cost_total"])
            y["cost"] = cc.groupby(cc.index.year).sum()
            y["excess_plus_cost"] = y["excess"] + y["cost"]
        y["excess_logrel"] = np.log1p(y["strategy"]) - np.log1p(y["bench"])
        if n == "main":
            cc = _ev(rec["cost_total"])
            y["cost"] = cc.groupby(cc.index.year).sum()
            if "fullpool_ew" in Rs:
                fpy = LE.yearly(_ev(Rs["fullpool_ew"]["rets"]), _ev(Rs["fullpool_ew"]["bench"]))
                y["minus_fullpool_ew"] = y["strategy"] - fpy["strategy"]
            for c in close_idx.columns:
                oo = _ev(oo_idx[c].reindex(dates_full))
                y[f"idx_{c}_oo"] = (1 + oo.fillna(0)).groupby(oo.index.year).prod() - 1
        yl.append(y.reset_index().rename(columns={"index": "year"}))
    yl = pd.concat(yl, ignore_index=True)
    yl.to_csv(LEDGER / "lh_step1_yearly.csv", index=False)

    loo = pd.DataFrame(sum([loo_rows(n, Rs[n]) for n in ("main", "main_d50", "main_d100") if n in Rs], []))
    loo.to_csv(LEDGER / "lh_step1_loo.csv", index=False)

    # ---------------- 判定(§10.2)----------------
    grades = []
    for n in ("main", "main_d50", "main_d100"):
        ye = yl[yl.config == n].set_index("year")["excess"]
        grades.append(grade_step1(summ, wins, loo, ye, n))
    grades = pd.DataFrame(grades)

    # ---------------- 网格(§10.3)----------------
    grid = []
    for n, R in Rs.items():
        c = CONFIGS[n]
        if c["period"] == "2125" or c["group"] == "diag":
            continue
        row = dict(config=n, signal=c["signal"] if c["weights"] is None else c["weights"],
                   buffer=str(c["buffer"]), rebalance=c["k"], pool=c["pool"], dead=c["dead"], ratio=c["ratio"],
                   cost=c["cost"], commission_bps=c["comm"], discount=c["disc"], valuation=c["val"],
                   cost_ann_full=R["cost_bps"] * 1e-4 * TD, turnover_ann=R["turnover"] * TD,
                   exposure=R["exposure"], holdings=R["holdings"], elapsed_s=R["elapsed"])
        for pn, (a, b) in PERIODS.items():
            s = LE.slice_eval(R["rets"], a, b); bb = LE.slice_eval(R["bench"], a, b)
            rel = LE.rel_block(s, bb)
            row[f"{pn}_ann"] = LE.ann_geo(s); row[f"{pn}_bench_ann"] = LE.ann_geo(bb)
            row[f"{pn}_excess"] = rel["excess_geo"]; row[f"{pn}_nw_t5"] = rel["nw_t5"]
            row[f"{pn}_mdd"] = LE.max_dd(s)[0]; row[f"{pn}_rel_mdd"] = rel["rel_mdd"]
        grid.append(row)
    grid = pd.DataFrame(grid)
    # 窗口口径 B(只是评价窗口不同):在 main 行上附 B 窗口熊市相对收益
    cells = ["main"] + [f"grid_{s_}_{bn}" for s_ in ("S1", "S2", "S3") for bn in BUFFERS
                        if not (s_ == "S1" and bn == "b20_35")]
    grid["sig_x_buffer_cell"] = grid.config.isin(cells)
    rob = []
    for pn in ("dev", "test", "full"):
        g = grid[grid.sig_x_buffer_cell]
        frac = float((g[f"{pn}_excess"] > 0).mean())
        med = float(g[f"{pn}_excess"].median())
        mval = float(grid.loc[grid.config == "main", f"{pn}_excess"].iloc[0])
        ok = (frac >= 0.8) and (mval >= med - 0.01)
        bad = g.loc[g[f"{pn}_excess"] <= 0, "config"].tolist()
        rob.append(dict(period=pn, n_cells=len(g), frac_pos=frac, median_excess=med, main_excess=mval,
                        main_ge_median_minus_1pp=bool(mval >= med - 0.01), robust=bool(ok), nonpositive_cells=";".join(bad)))
    rob = pd.DataFrame(rob)
    grid.to_csv(LEDGER / "lh_step1_grid.csv", index=False)
    rob.to_csv(LEDGER / "lh_step1_robustness.csv", index=False)
    grades.to_csv(LEDGER / "lh_step1_grade.csv", index=False)

    # ---------------- 幸存者偏差(S-b 全期 / 逐年 / 窗口;S-a 2021-2025)----------------
    sv = []
    for pn, (a, b) in PERIODS.items():
        for n in ("main", "surv2025"):
            R = Rs[n]
            s = LE.slice_eval(R["rets"], a, b); bb = LE.slice_eval(R["bench"], a, b)
            sv.append(dict(test="S-b", period=pn, sample="含退市(主)" if n == "main" else "只含2025末在市",
                           ann_strategy=LE.ann_geo(s), ann_bench=LE.ann_geo(bb), excess=LE.ann_geo(s) - LE.ann_geo(bb),
                           mdd_strategy=LE.max_dd(s)[0]))
    ym = yl[yl.config == "main"].set_index("year"); ysv = yl[yl.config == "surv2025"].set_index("year")
    for yr in ym.index:
        sv.append(dict(test="S-b", period=str(yr), sample="差(幸存 - 含退市)",
                       ann_strategy=ysv.loc[yr, "strategy"] - ym.loc[yr, "strategy"],
                       ann_bench=ysv.loc[yr, "bench"] - ym.loc[yr, "bench"],
                       excess=ysv.loc[yr, "excess"] - ym.loc[yr, "excess"]))
    for n, lab in (("p2125_full", "含退市"), ("p2125_survpanel", "只含研究面板股票(cache/daily)")):
        if n in Rs:
            R = Rs[n]
            s, bb = R["rets"].loc["2021":"2025"], R["bench"].loc["2021":"2025"]
            sv.append(dict(test="S-a", period="2021-2025", sample=lab, ann_strategy=LE.ann_geo(s),
                           ann_bench=LE.ann_geo(bb), excess=LE.ann_geo(s) - LE.ann_geo(bb), mdd_strategy=LE.max_dd(s)[0]))
    # 窗口
    for w in list(LE.WINDOWS_A):
        for n in ("main", "surv2025"):
            r_ = wins[(wins.config == n) & (wins.set == "A") & (wins.window == w)]
            if len(r_):
                r_ = r_.iloc[0]
                sv.append(dict(test="S-b", period=w, sample="含退市(主)" if n == "main" else "只含2025末在市",
                               ann_strategy=r_.ret_strategy, ann_bench=r_.ret_bench, excess=r_.rel_diff,
                               mdd_strategy=r_.mdd_strategy, note="窗口累计收益(非年化)"))
    sv = pd.DataFrame(sv)
    sv.to_csv(LEDGER / "lh_survivorship.csv", index=False)

    # ---------------- B3 + S-a 写入桥接账本 ----------------
    bd = pd.read_parquet(BRIDGE_DAILY) if BRIDGE_DAILY.exists() else None
    brows = []
    if bd is not None and "p2125_survpanel" in Rs:
        R = Rs["p2125_survpanel"]
        s, bb = R["rets"].loc["2021":"2025"], R["bench"].loc["2021":"2025"]
        ex_lh = LE.ann_geo(s) - LE.ann_geo(bb)
        rv, bv = bd["B2|v 四项全换|strategy"].dropna(), bd["B2|v 四项全换|bench"].dropna()
        ex_rp = LE.ann_geo(rv) - LE.ann_geo(bv)
        a_, b_ = (s - bb).align(rv - bv, join="inner")
        brows.append(dict(row="B3|LongPanel幸存股 vs 研究面板四项全换|Y1|", test="B3", panel="LongPanel(研究面板股票) vs A 1m 四项全换",
                          cost="Y1", excess_turn=ex_lh, excess_comp=ex_rp, diff_vs_base=ex_lh - ex_rp,
                          rho_excess=float(a_.corr(b_)), ann_turn=LE.ann_geo(s), ann_comp=LE.ann_geo(rv),
                          ann_bench=LE.ann_geo(bb),
                          grade="一致(<=0.5pp)" if abs(ex_lh - ex_rp) <= 0.005 else "差异 > 0.5pp",
                          note="excess_turn = LongPanel 幸存股超额;excess_comp 列在此行 = 研究面板四项全换超额"))
    if "p2125_full" in Rs and "p2125_survpanel" in Rs:
        f_, s_ = Rs["p2125_full"], Rs["p2125_survpanel"]
        brows.append(dict(row="S-a|含退市 - 只含研究面板股票|Y1|", test="S-a", panel="LongPanel 2021-2025",
                          cost="Y1",
                          ann_turn=LE.ann_geo(f_["rets"].loc["2021":]) - LE.ann_geo(s_["rets"].loc["2021":]),
                          ann_bench=LE.ann_geo(f_["bench"].loc["2021":]) - LE.ann_geo(s_["bench"].loc["2021":]),
                          diff_vs_base=(LE.ann_geo(f_["rets"].loc["2021":]) - LE.ann_geo(f_["bench"].loc["2021":]))
                          - (LE.ann_geo(s_["rets"].loc["2021":]) - LE.ann_geo(s_["bench"].loc["2021":])),
                          note="ann_turn = 策略年化差,ann_bench = B1 年化差,diff_vs_base = 超额差(含退市 - 幸存)"))
    if brows:
        merge_csv(BRIDGE, pd.DataFrame(brows), "row")

    # ---------------- 每日序列(给第二步)----------------
    out = pd.DataFrame(index=dates_full)
    out.index.name = "signal_date"
    out["strat_main"] = main["rets"]
    for i, c in enumerate(main["phase_rets"].columns):
        out[f"strat_main_ph{i}"] = main["phase_rets"][c]
    out["bench_b1"] = main["bench"]
    for n, col in (("main_old", "strat_main_oldcost"), ("main_d50", "strat_main_d50"), ("main_d100", "strat_main_d100"),
                   ("main_vidx", "strat_main_vidx"), ("surv2025", "strat_surv2025"), ("fullpool_ew", "strat_fullpool_ew")):
        if n in Rs:
            out[col] = Rs[n]["rets"]
    if "fullpool_ew" in Rs:
        out["strat_fullpool_ew_gross"] = rec_fp["gross_ret"].reindex(dates_full)
    for n, col in (("main_d50", "bench_b1_d50"), ("main_d100", "bench_b1_d100"), ("main_vidx", "bench_b1_vidx"),
                   ("surv2025", "bench_b1_surv2025"), ("main_q100", "bench_b4_allA_ew")):
        if n in Rs:
            out[col] = Rs[n]["bench"]
    for c in oo_idx.columns:
        out[f"idx_{c}_oo"] = oo_idx[c].reindex(dates_full)
    X = LE.load_inputs()
    ew = X.ew_allA_ret
    out["ew_allA_close_ret"] = ew.reindex(dates_full)
    for c in ("exposure", "cash", "turnover", "cost_total", "cost_commission", "cost_stamp", "cost_spread", "cost_impact",
              "n_hold"):
        out[f"main_{c}"] = rec[c].reindex(dates_full)
    out.to_parquet(TMP / "lh_step1_daily.parquet")
    (TMP / "lh_step1_daily_columns.json").write_text(json.dumps(DAILY_COLUMNS, ensure_ascii=False, indent=1))

    # ---------------- 图 ----------------
    plot_nav(main, close_idx, Rs["fullpool_ew"]["rets"] if "fullpool_ew" in Rs else None)
    timing("report", time.time() - t0, "账本 + 每日序列 + 图")
    log("[report] 判定:\n" + grades.to_string())
    log("[report] 稳健性:\n" + rob.to_string())


DAILY_COLUMNS = {
    "index": "signal_date = 信号日 t(引擎约定:收益 = te=t+1 开盘 -> te+1 开盘,打在 t)",
    "!!_timing_rule_step2": ("⚠ 预注册 §12.1:strat_* / bench_* / idx_*_oo / main_* 第 s 行的收益要到 s+2 开盘才实现。"
                             "决策日 t 只能用 s <= t-2 的行,即 strat_main.shift(2)(或引擎 ctx['nav_dec']);"
                             "O2 的 σ̂、O4 的影子净值一律如此。按 t 行直接当 t 日信号 = 前视 2 天。"
                             "唯一例外 ew_allA_close_ret:第 t 行 = 收盘 t-1 -> 收盘 t,t 日收盘可知,可直接用于 t 日决策"
                             "(审查复核:与 strat_main 的相关峰值逐年在 k=-1)。"
                             "建议第二步加可证伪探针:不滞后版本若明显'更好',即存在前视。"),
    "strat_main": ("第一步主配置(S1 turn20 排雷,缓冲 20/35,5 日,5 相位平均,池 92%,开盘触板不可成交,Y=1 一跳,史实印花税,"
                   "退市折价 0%)净收益;⚠ 第二步作信号时只能用 shift(2)"),
    "strat_fullpool_ew_gross": "strat_fullpool_ew 的毛收益(record gross_ret,不扣成本;同样受停牌/触板约束 = 可投资且无成本的全池等权,审查修复对照)",
    "strat_main_ph0..ph4": "主配置逐相位净收益(相位 = 调仓日历偏移)",
    "bench_b1": "B1 同频(5 日)池等权,同池同计价价,不计成本",
    "strat_main_oldcost": "主配置,旧成本(只扣佣金 1bp + 史实印花税)",
    "strat_main_d50 / strat_main_d100": "退市折价 -50% / -100%",
    "bench_b1_d50 / bench_b1_d100": "对应折价下的 B1",
    "strat_main_vidx / bench_b1_vidx": "停牌指数法估值(敏感性)",
    "strat_surv2025 / bench_b1_surv2025": "只含 2025-12-31 仍在市股票的面板(幸存者偏差 S-b)",
    "strat_fullpool_ew": "全池等权(常数打分)5 日调仓,Y=1(引擎断言 E3)",
    "bench_b4_allA_ew": "B4 全 A 等权同频(池不设 ADV 门槛 = main_q100 的基准)",
    "idx_<SSE|CSI300|CSI500|CSI1000|GZ_A>_oo": "价格指数开盘对开盘收益,按信号日对齐(te 开盘 -> te+1 开盘);中证1000 2011 年缺开盘时用收盘对收盘",
    "ew_allA_close_ret": "自建全 A 等权收盘指数日收益(收盘 t-1 -> 收盘 t,条件 1-3 的股票,复牌含跳空;t 日收盘可知),§12.1 O3 用",
    "main_exposure/cash/turnover/cost_*/n_hold": "主配置 5 相位平均的逐日仓位、现金、单边换手、成本分项(占净值)、持股数(同样记在信号日 s,作信号时只能用 shift(2))",
}


def plot_nav(main: dict, close_idx: pd.DataFrame, fp: pd.Series | None = None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "Arial Unicode MS", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    s, b = _ev(main["rets"]), _ev(main["bench"])
    ns, nb = (1 + s).cumprod(), (1 + b).cumprod()
    rel = ns / nb
    dds, ddb = ns / ns.cummax() - 1, nb / nb.cummax() - 1
    fig, ax = plt.subplots(3, 1, figsize=(12, 10), sharex=True, gridspec_kw=dict(height_ratios=[2.2, 1, 1]))
    ax[0].plot(ns.index, ns.values, color="#c0392b", lw=1.3, label="换手率版排雷(主配置,Y=1 成本后)")
    ax[0].plot(nb.index, nb.values, color="#2c3e50", lw=1.1, label="同频池等权 B1(不计成本)")
    ax[0].set_yscale("log"); ax[0].set_ylabel("净值(对数)")
    ax[1].plot(rel.index, rel.values, color="#16a085", lw=1.2, label="超额净值 = 策略 / B1")
    if fp is not None:   # 审查修复(P1):B1 停牌估值抬高了它的波动,另画相对同引擎全池等权(可投资对照)
        nf = (1 + _ev(fp).reindex(s.index).fillna(0)).cumprod()
        ax[1].plot(nf.index, (ns / nf).values, color="#e67e22", lw=1.0, ls="--",
                   label="策略 / 同引擎全池等权(付成本,可投资对照)")
    ax[1].axhline(1.0, color="grey", lw=0.6)
    ax[2].fill_between(dds.index, dds.values, 0, color="#c0392b", alpha=0.35, lw=0, label="策略回撤")
    ax[2].plot(ddb.index, ddb.values, color="#2c3e50", lw=0.8, label="B1 回撤")
    ax[2].axhline(-0.5, color="#8e44ad", ls=":", lw=1.2, label="用户约束 -50%")
    ax[2].set_ylabel("回撤"); ax[2].set_ylim(-0.8, 0.02)
    top = float(ns.max()) * 1.15
    for w in LE.BEARS:
        a, bb = LE.WINDOWS_A[w]
        for x in ax:
            x.axvspan(pd.Timestamp(a), pd.Timestamp(bb), color="#95a5a6", alpha=0.25, lw=0)
        ax[0].text(pd.Timestamp(a), top, w, fontsize=8, color="#555", va="bottom")
    ax[0].set_ylim(top=top * 1.35)
    for x in ax:
        x.axvline(pd.Timestamp("2015-01-05"), color="#7f8c8d", ls="--", lw=0.8)
        x.grid(alpha=0.3)
    ax[0].legend(loc="lower right", fontsize=9); ax[1].legend(loc="upper left", fontsize=9)
    ax[2].legend(loc="lower right", fontsize=8, ncol=3)
    ax[0].set_title("长历史 2006-2025:换手率版排雷 vs 同频池等权(灰带 = 预注册熊市窗口 A;虚线 = 开发/检验分界)")
    fig.tight_layout()
    out = ROOT / "reports" / "lh_step1_nav.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    log(f"[report] 图 {out}")


# ================================================================ review(审查修复,2026-10-11)
REVIEW = LEDGER / "lh_step1_review.csv"
REVIEW_PERIODS = {
    "full": (LE.EVAL_START, "2025-12-31"), "dev": LE.DEV, "test": LE.TEST,
    "oos_2006_2019": (LE.EVAL_START, "2019-12-31"),        # 设计期 2020-2025 之前 = 真正样本外
    "dev_2008_2014": ("2008-01-01", "2014-12-31"),
    "test_2015_2019": ("2015-01-05", "2019-12-31"),
    "design_2020_2025": ("2020-01-01", "2025-12-31"),      # 排雷参数与"换手率是驱动"推论的来源期
    "slice_2021_2025": ("2021-01-01", "2025-12-31"),
}


def _rel_row(item: str, period: str, comp: str, a: pd.Series, b: pd.Series, note: str = "") -> dict:
    a, b = a.align(b, join="inner")
    m = a.notna() & b.notna()
    a, b = a[m], b[m]
    rel = LE.rel_block(a, b)
    rn = (1 + a).cumprod() / (1 + b).cumprod()
    dd = rn / rn.cummax() - 1
    tr = dd.idxmin()
    pk = rn.loc[:tr].idxmax()
    after = rn.loc[tr:]
    rec = after.index[after >= rn.loc[pk]]
    y = LE.yearly(a, b)
    return dict(item=item, period=period, comparator=comp, start=str(a.index[0].date()), end=str(a.index[-1].date()),
                n_days=len(a), ann_s=LE.ann_geo(a), ann_b=LE.ann_geo(b), excess_geo=rel["excess_geo"],
                excess_arith=rel["excess_arith"], nw_t5=rel["nw_t5"], year_t=LE.year_t(y["excess"]),
                year_t_log=LE.year_t(np.log1p(y["strategy"]) - np.log1p(y["bench"])), n_years=len(y),
                years_pos=int((y["excess"] > 0).sum()), vol_s=float(a.std() * np.sqrt(TD)),
                vol_b=float(b.std() * np.sqrt(TD)), rel_mdd=float(dd.min()), rel_mdd_peak=str(pk.date()),
                rel_mdd_trough=str(tr.date()), rel_mdd_recover=str(rec[0].date()) if len(rec) else "未收复",
                note=note)


def stage_review():
    """审查修复:把审查指出的口径/表述问题落成账本 lh_step1_review.csv(不改任何预注册判定)。
    1 对照口径(B1 / 同引擎全池等权 fp / fp 毛收益 fpg / 停牌指数法 vidx)× 区间(含 2006-2019 样本外、设计期、剔 BULL06、剔 2006-07)
    2 s-B1 = (s-fp) + (fp-B1) 分解;3 Q9 有条件接受的逐年对齐复核(写入 lh_engine_checks.csv)
    4 幸存者偏差上界(B1 持有到退市);5 桥接时间序列解释份额;6 S-a 冷启动;7 2005-2009 持仓贡献分解(复权/股改日)"""
    t0 = time.time()
    rows = []
    Rs = {n: load_run(n) for n in ("main", "fullpool_ew", "main_vidx", "p2125_full", "p2125_survpanel")}
    main, fp = Rs["main"], Rs["fullpool_ew"]
    s, b1 = main["rets"], main["bench"]
    dates = s.index
    f = fp["rets"].reindex(dates)
    fcost = fp["rec_daily"]["cost_total"].reindex(dates)
    fg = fp["rec_daily"]["gross_ret"].reindex(dates)
    sv, bv = Rs["main_vidx"]["rets"].reindex(dates), Rs["main_vidx"]["bench"].reindex(dates)
    comps = {"B1": (s, b1, "预注册口径:同频池等权,不计成本,停牌股按陈旧价可建/清仓(不可投资)"),
             "fp": (s, f, "事后对照:同引擎全池等权,付 Y=1 成本,受停牌/触板约束(可投资)"),
             "fpg": (s, fg, "事后对照:同引擎全池等权的毛收益(不计成本,受同样约束)= 去掉 B1 可成交性假象的无成本基准"),
             "vidx": (sv, bv, "敏感性:停牌指数法估值下的策略 vs B1")}
    # ---- 1 对照 × 区间 ----
    for pn, (a, b) in REVIEW_PERIODS.items():
        for cn, (x, y, note) in comps.items():
            rows.append(_rel_row("compare", pn, cn, LE.slice_eval(x, a, b), LE.slice_eval(y, a, b), note))
    ev = (dates >= pd.Timestamp(LE.EVAL_START)) & (dates <= pd.Timestamp("2025-12-31"))
    a6, b6 = LE.WINDOWS_A["BULL06"]
    drops = {"drop_BULL06": ev & ~((dates >= pd.Timestamp(a6)) & (dates < pd.Timestamp(b6))),
             "drop_2006_2007": ev & (dates.year >= 2008)}
    for pn, keep in drops.items():
        for cn, (x, y, note) in comps.items():
            rows.append(_rel_row("compare", pn, cn, x[keep], y[keep], note + ";剔除段直接拼接"))
    # 2006-07 / BULL06 占累计对数超额的份额
    le = (np.log1p(s) - np.log1p(b1))[ev]
    for lab, m in (("2006_2007", le.index.year <= 2007),
                   ("BULL06", (le.index >= pd.Timestamp(a6)) & (le.index < pd.Timestamp(b6)))):
        rows.append(dict(item="log_share_vs_B1", period=lab, comparator="B1", excess_geo=float(le[m].sum()),
                         value=float(le[m].sum() / le.sum()), note=f"该段累计对数超额 / 全期累计对数超额({le.sum():.3f})"))
    # ---- 2 分解:s-B1 = (s-fp) + (fp-B1);fp-B1 = -fp成本 + E3 残差 ----
    for pn, (a, b) in PERIODS.items():
        S, B, F, G = (LE.slice_eval(z, a, b) for z in (s, b1, f, fg))
        c_ann = float(LE.slice_eval(fcost, a, b).mean() * TD)
        rows.append(dict(item="decomp", period=pn, comparator="s-B1 = (s-fp) + (fp-B1)",
                         excess_geo=LE.ann_geo(S) - LE.ann_geo(B), s_minus_fp=LE.ann_geo(S) - LE.ann_geo(F),
                         fp_minus_b1=LE.ann_geo(F) - LE.ann_geo(B), fp_cost=c_ann,
                         e3_resid=LE.ann_geo(F) - LE.ann_geo(B) + c_ann, s_minus_fpg=LE.ann_geo(S) - LE.ann_geo(G),
                         fpg_minus_b1=LE.ann_geo(G) - LE.ann_geo(B),
                         excess_arith=float((S - B).mean() * TD), fp_minus_b1_arith=float((F - B).mean() * TD),
                         vol_s=float(S.std() * np.sqrt(TD)), vol_b=float(B.std() * np.sqrt(TD)),
                         vol_fp=float(F.std() * np.sqrt(TD)), vol_fpg=float(G.std() * np.sqrt(TD)),
                         note="e3_resid = (fp-B1) + fp 成本 = B1 可成交性残差(几何);s_minus_fpg ≈ s-B1 - e3_resid"))
    for yr in (2006, 2007, 2008, 2015):
        S, B, F = (z[z.index.year == yr] for z in (s, b1, f))
        rows.append(dict(item="vol_year", period=str(yr), comparator="B1 vs fp",
                         vol_s=float(S.std() * np.sqrt(TD)), vol_b=float(B.std() * np.sqrt(TD)),
                         vol_fp=float(F.std() * np.sqrt(TD)),
                         excess_geo=float((1 + S).prod() - (1 + B).prod()),
                         s_minus_fp=float((1 + S).prod() - (1 + F).prod()),
                         excess_arith=float((S - B).mean() * TD),
                         value=float(np.log((1 + S).prod() / (1 + B).prod())),
                         note="excess_geo / s_minus_fp = 算术年差;excess_arith = 日超额均值 x 242;value = 对数相对"))
    # ---- 3 Q9 有条件接受的逐年复核(lh_gate_decision.json Q9.condition)----
    X = LE.load_inputs()
    close_idx, _ = index_frame(X.dates)
    ew = X.ew_allA_ret
    q9 = []
    for idx_name in ("GZ_A", "CSI1000", "SSE"):
        r_i = close_idx[idx_name] / close_idx[idx_name].shift(1) - 1
        for yr in range(2005, 2026):
            m = X.dates.year == yr
            cs = {k: float(ew[m].corr(r_i.shift(k)[m])) for k in range(-3, 4)}
            q9.append(dict(index=idx_name, year=yr, peak=max(cs, key=cs.get), corr0=cs[0]))
    q9 = pd.DataFrame(q9)
    g = q9[q9["index"] == "GZ_A"]
    n0 = int((g.peak == 0).sum())
    mn = g.loc[g.corr0.idxmin()]
    others = "; ".join(f"{i} {int((q9[q9['index'] == i].peak == 0).sum())}/21" for i in ("CSI1000", "SSE"))
    merge_csv(CHECKS, pd.DataFrame([dict(
        check="Q9_cond_yearly_peak_k0", step=1, value=n0, ok=bool(n0 == len(g)),
        note=(f"lh_gate_decision.json Q9 有条件接受的条件:逐年 corr(全A等权 ew_allA_ret, 国证A股收盘收益.shift(k)) k∈[-3,3] "
              f"峰值在 k=0 的年数 {n0}/{len(g)}(2005-2025);k=0 相关最低 {mn.corr0:.3f}({int(mn.year)});"
              f"2021 年 {float(g[g.year == 2021].corr0.iloc[0]):.3f};旁证 {others}。审查修复补跑(第一步原未执行)"))]), "check")
    for _, r_ in q9.iterrows():
        rows.append(dict(item="q9_yearly_peak", period=str(r_.year), comparator=r_["index"], value=float(r_.peak),
                         excess_geo=r_.corr0, note="value = 峰值 k;excess_geo 列放 k=0 相关"))
    # 交接时点核对:每日序列各外生列与 strat_main 的逐年相关峰值(ew_allA_close_ret 应在 k=-1,其余开盘对开盘列在 k=0)
    D = pd.read_parquet(TMP / "lh_step1_daily.parquet")
    for col, want in (("ew_allA_close_ret", -1), ("idx_GZ_A_oo", 0), ("idx_CSI1000_oo", 0), ("idx_SSE_oo", 0),
                      ("bench_b4_allA_ew", 0)):
        pk = []
        for yr in range(2006, 2026):
            m = D.index.year == yr
            cs = {k: float(D["strat_main"][m].corr(D[col].shift(k)[m])) for k in range(-3, 4)}
            pk.append(max(cs, key=cs.get))
        rows.append(dict(item="align_daily_cols", period="2006_2025", comparator=col, value=float(sum(p_ == want for p_ in pk)),
                         nw_t5=float(want), note=f"value = 逐年 corr(strat_main, 列.shift(k)) 峰值在 k={want} 的年数(共 20);"
                                                 f"nw_t5 列放期望 k;逐年峰值 {pk}"))
    # ---- 4 幸存者偏差上界:B1 若必须持有到退市(折价 -100%)----
    U = X.arr("pool_q92")
    n = U.sum(1)
    od = np.asarray(X.meta["od_idx"]); lt = np.asarray(X.meta["lt_idx"])
    Wm = LE.weights(X, "S1")
    srows = []
    for j in np.flatnonzero(od >= 0):
        L = int(lt[j])
        if L < 0:
            continue
        wb, ws = [], []
        for ph in range(5):          # 相位 ph 的重置日 = {t: (t - ph) % 5 == 0}(无 date_mask 时 t0 = 0)
            cand = [t for t in range(max(L - 4, 0), L + 1) if (t - ph) % 5 == 0]
            t = cand[-1] if cand else L
            wb.append(1.0 / n[t] if U[t, j] and n[t] > 0 else 0.0)
            ws.append(float(Wm[t, j]))
        srows.append(dict(year=X.dates[L].year, w_b1=np.mean(wb), w_s=np.mean(ws)))
    sd = pd.DataFrame(srows)
    gy = sd.groupby("year")[["w_b1", "w_s"]].sum()
    for a, b in ((2006, 2025), (2021, 2025)):
        z = gy.loc[a:b]
        ny = b - a + 1
        rows.append(dict(item="surv_upper", period=f"{a}_{b}", comparator="持有到退市、折价 -100%",
                         value=float(z.w_b1.sum() / ny), s_minus_fp=float(z.w_s.sum() / ny),
                         note="value = B1 在最后交易日的(近似)权重之和 / 年数 = B1 年化再低的上界;"
                              "s_minus_fp 列放策略目标权重的同口径值;5 相位平均,重置日内漂移忽略"))
    # ---- 5 桥接:换手率版在时间序列上能解释 COMP_v5 超额的份额 ----
    import statsmodels.api as sm
    bd = pd.read_parquet(BRIDGE_DAILY)
    for cost in ("Y1", "old"):
        ec = (bd[f"C|comp_{cost}"] - bd["C|bench_pool5"]).dropna()
        et = (bd[f"C|turn_{cost}"] - bd["C|bench_pool5"]).dropna()
        ec, et = ec.align(et, join="inner")
        fit = sm.OLS(ec.to_numpy(), sm.add_constant(et.to_numpy())).fit()
        bb_ = float(fit.params[1])
        xc = LE.ann_geo(bd[f"C|comp_{cost}"].dropna()) - LE.ann_geo(bd["C|bench_pool5"].dropna())
        xt = LE.ann_geo(bd[f"C|turn_{cost}"].dropna()) - LE.ann_geo(bd["C|bench_pool5"].dropna())
        rows.append(dict(item="bridge_share", period="2021_2025", comparator=f"C|{cost}", value=bb_ * et.mean() / ec.mean(),
                         excess_geo=bb_ * xt / xc, excess_arith=float(fit.params[0] * TD), nw_t5=bb_,
                         s_minus_fp=float(fit.rsquared), fp_minus_b1=xt / xc,
                         note="value = b·均值(换手率版日超额)/均值(COMP 日超额)(算术);excess_geo = b·几何超额比;"
                              "excess_arith = 截距年化;nw_t5 列放 b;s_minus_fp 列放 R²;fp_minus_b1 列放均值比"))
    # ---- 6 S-a 冷启动:掩码重跑 vs 连续主回测切片 ----
    pf = Rs["p2125_full"]
    a_, b_ = "2021-01-01", "2025-12-31"
    sp, bp = LE.slice_eval(pf["rets"], a_, b_), LE.slice_eval(pf["bench"], a_, b_)
    sm_, bm_ = LE.slice_eval(s, a_, b_), LE.slice_eval(b1, a_, b_)
    jan = (sp.index >= pd.Timestamp("2021-01-01")) & (sp.index <= pd.Timestamp("2021-01-31"))
    sm_a = sm_.reindex(sp.index)
    ph = pf["phase_rets"].loc["2021-01-01":"2021-01-15"]
    zero_days = {c: int((ph[c].abs() < 1e-15).sum()) for c in ph.columns}
    rows.append(dict(item="sa_coldstart", period="2021_2025", comparator="p2125_full(掩码冷启动)vs main 连续切片",
                     excess_geo=LE.ann_geo(sp) - LE.ann_geo(bp), s_minus_fp=LE.ann_geo(sm_) - LE.ann_geo(bm_),
                     fp_minus_b1=float((bp - bm_.reindex(bp.index)).abs().max()),
                     value=float((1 + sp).prod() / (1 + sm_a).prod() - 1),
                     excess_arith=float((1 + sp[jan]).prod() / (1 + sm_a[jan]).prod() - 1),
                     note="excess_geo = 掩码重跑超额;s_minus_fp 列放连续切片超额;fp_minus_b1 列放两条 B1 逐日最大差;"
                          "value = 策略累计收益比 -1(掩码/连续);excess_arith 列放 2021-01 月内的同一比值;"
                          "2021-01-01~15 各相位零收益天数 " + ",".join(f"{k}:{v}" for k, v in zero_days.items())))
    # ---- 7 2005-2009(及全期)持仓贡献分解:主配置 - 全池等权,按段类型 ----
    rows += _early_decomp(X, Wm)
    pd.DataFrame(rows).to_csv(REVIEW, index=False)
    timing("review", time.time() - t0, "审查修复账本 lh_step1_review.csv(含 10 次带持仓记录的单相位回测)")
    log(f"[review] {REVIEW} {len(rows)} 行,{time.time() - t0:.0f}s")


def _early_decomp(X, Wm) -> list[dict]:
    """主配置 - 同引擎全池等权 的逐日毛收益差(5 相位平均,算术),按记账段类型拆分:
    gap = te 停牌(陈旧价)且 te+1 复牌的跳空段;gap_nolimit = 其中复牌日无涨跌幅限制(股改复牌、恢复上市等);
    nolimit_day = te+1 为无涨跌幅限制日;factor_jump = te 或 te+1 复权因子跳变 > 2%(送转/对价)。"""
    from afactor import strategy as ST
    from afactor.lh_panel import LongPanel
    dates = X.dates; T, N = len(dates), len(X.codes)
    ad = LE.adapter(X, discount=0.0); cm = LE.cost_model(ad, "Y1"); sf = LE.StampFn(dates)
    W = np.asarray(Wm)
    Ua = X.arr("pool_q92").astype(bool); n = Ua.sum(1, keepdims=True)
    Wfp = np.divide(Ua.astype(np.float32), n, out=np.zeros(Ua.shape, np.float32), where=n > 0)
    p = LongPanel()
    valid = p.raw("trade").to_numpy(bool) & (p.raw("amount").to_numpy(np.float32) > 0)
    nolim = p.raw("no_limit_day").to_numpy(bool)
    c = p.raw("close_px").to_numpy(np.float64); cr = p.raw("close_raw").to_numpy(np.float64)
    fac = pd.DataFrame(np.where(np.isfinite(c) & (cr > 0), c / cr, np.nan)).ffill().to_numpy()
    del c, cr
    fj = np.zeros((T, N), bool); fj[1:] = np.abs(fac[1:] / fac[:-1] - 1) > 0.02
    del fac
    V = X.arr("V_d0").astype(np.float64)
    r = np.full((T, N), np.nan); r[:-2] = V[2:] / V[1:-1] - 1.0
    r = np.where(np.isfinite(r), r, 0.0)
    masks = {"gap": np.zeros((T, N), bool), "gap_nolimit": np.zeros((T, N), bool),
             "nolimit_day": np.zeros((T, N), bool), "factor_jump": np.zeros((T, N), bool)}
    masks["gap"][:-2] = (~valid[1:-1]) & valid[2:]
    masks["gap_nolimit"][:-2] = masks["gap"][:-2] & nolim[2:]
    masks["nolimit_day"][:-2] = nolim[2:]
    masks["factor_jump"][:-2] = fj[2:] | fj[1:-1]
    del valid, nolim, fj
    acc = {k: [] for k in ("all", *masks)}
    for ph in range(5):
        Hm = np.asarray(ST.simulate(W, ad, "open10", LE.CAPITAL, rebalance=5, phase=ph, cost_model=cm, record=True,
                                    stamp_fn=sf)["record"]["holdings"], np.float64)
        Hf = np.asarray(ST.simulate(Wfp, ad, "open10", LE.CAPITAL, rebalance=5, phase=ph, cost_model=cm, record=True,
                                    stamp_fn=sf)["record"]["holdings"], np.float64)
        m = Hm.shape[0]
        d = (Hm - Hf) * r[:m]
        del Hm, Hf
        acc["all"].append(pd.Series(d.sum(1), index=dates[:m]))
        for k, mk in masks.items():
            acc[k].append(pd.Series((d * mk[:m]).sum(1), index=dates[:m]))
        del d
    df = pd.DataFrame({k: pd.concat(v, axis=1).mean(1) for k, v in acc.items()})
    yr = df.groupby(df.index.year).sum()
    out = []
    for y_, r_ in yr.iterrows():
        out.append(dict(item="early_decomp", period=str(y_), comparator="main - fullpool_ew 毛持仓贡献(算术,5 相位平均)",
                        value=r_["all"], excess_geo=r_["gap"], excess_arith=r_["gap_nolimit"],
                        s_minus_fp=r_["nolimit_day"], fp_minus_b1=r_["factor_jump"],
                        note="value = 合计;excess_geo 列 = 停牌->复牌跳空段;excess_arith 列 = 其中复牌日无涨跌幅限制;"
                             "s_minus_fp 列 = 无涨跌幅限制日;fp_minus_b1 列 = 复权因子跳变 >2% 段(均为年内求和)"))
    for lab, sel in (("2006_2007", (yr.index >= 2006) & (yr.index <= 2007)),
                     ("2008_2025", (yr.index >= 2008) & (yr.index <= 2025))):
        z = yr[sel].sum()
        out.append(dict(item="early_decomp", period=lab, comparator="main - fullpool_ew 毛持仓贡献(算术,5 相位平均)",
                        value=z["all"], excess_geo=z["gap"], excess_arith=z["gap_nolimit"], s_minus_fp=z["nolimit_day"],
                        fp_minus_b1=z["factor_jump"], note="同上,区间求和"))
    return out


# ================================================================ speed(回答"要跑多久、能不能加速")
def stage_speed():
    """实测第二步两种接法的单相位耗时与一致性:callable target_fn vs 预先算好仓位序列的 ndarray(W x pos, normalize=False)。"""
    from afactor import strategy as ST
    X = LE.load_inputs(); ad = LE.adapter(X); cm = LE.cost_model(ad, "Y1"); sf = LE.StampFn(ad.dates)
    t = time.time(); W = LE.weights(X, "S1"); tw = time.time() - t
    T = len(ad.dates)
    yrs = np.asarray(ad.dates.year)
    pos = np.where(yrs % 2 == 0, 0.5, 1.0)            # 示意仓位序列(只测速度与一致性,不是任何规则)
    t = time.time()
    ST.simulate(W, ad, "open10", LE.CAPITAL, rebalance=5, phase=0, cost_model=cm, stamp_fn=sf)
    t_nd = time.time() - t
    t = time.time()
    Wp = (W * pos[:, None].astype(np.float32))
    r_ndp = ST.simulate(Wp, ad, "open10", LE.CAPITAL, rebalance=5, phase=0, normalize=False, cost_model=cm,
                        stamp_fn=sf)["rets"]
    t_ndp = time.time() - t
    t = time.time()
    r_fn = ST.simulate(lambda t_, h, ctx: W[t_].astype(np.float64) * pos[t_], ad, "open10", LE.CAPITAL, rebalance=5,
                       phase=0, normalize=False, cost_model=cm, stamp_fn=sf)["rets"]
    t_fn = time.time() - t
    a, b = r_ndp.align(r_fn, join="inner")
    diff = float((a - b).abs().max())
    timing("speed_weights_S1", tw, "LE.weights(X,'S1') 一次")
    timing("speed_ndarray_phase", t_nd, "ST.simulate ndarray W,单相位(O0 同路径)")
    timing("speed_ndarray_pos_phase", t_ndp, f"ndarray W x pos(t),normalize=False,单相位(含构造 W x pos);与 callable 逐日最大差 {diff:.1e}")
    timing("speed_callable_phase", t_fn, f"callable target_fn 返回 W[t] x pos[t],normalize=False,单相位;慢 {t_fn / max(t_nd, 1e-9):.1f} 倍")
    log(f"[speed] weights {tw:.1f}s ndarray {t_nd:.1f}s ndarray*pos {t_ndp:.1f}s callable {t_fn:.1f}s maxdiff {diff:.1e}")
    # 参照项(不在本脚本内跑,引自日志/账本)
    import re
    reg = (ROOT / "logs" / "lh_step1_regression.log").read_text()
    m = re.findall(r"(\d+) passed.* in ([\d.]+)s", reg)
    if m:
        timing("ref_regression_tests", float(m[-1][1]), f"tests/test_engine_regression.py {m[-1][0]} 项(logs/lh_step1_regression.log)")
    ft = pd.read_csv(LEDGER / "lh_data_fetch_timing.csv")
    pb = ft[ft.job == "面板构建"]
    if len(pb):
        timing("ref_panel_build", float(pb.minutes.iloc[0]) * 60, "LongPanel 构建(lh_data_fetch_timing.csv;原始抓数已缓存,不需重抓)")


# ================================================================ main
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stages", nargs="+")
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    only = a.only.split(",") if a.only else None
    T0 = time.time()
    for st in a.stages:
        if st == "verify":
            stage_verify()
        elif st == "prep":
            stage_prep()
        elif st == "bridge":
            stage_bridge()
        elif st == "runs":
            stage_runs(a.workers, only)
        elif st == "report":
            stage_report()
        elif st == "review":
            stage_review()
        elif st == "speed":
            stage_speed()
        else:
            raise SystemExit(f"未知阶段 {st}")
    log(f"[done] {a.stages} 共 {time.time() - T0:.0f}s")
