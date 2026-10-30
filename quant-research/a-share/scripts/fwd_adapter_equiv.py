"""前向适配器的研究期等价性检验(预注册 FORWARD_TEST_PREREG §4.4,前缀 fwd)。

每个适配器在前向使用之前,先在研究期数据上用适配器重算一段输出,与冻结代码的原输出逐位比较
(浮点求和顺序不同的允许 <= 1e-6)。**全部只用 <= 2025-12-31 的研究期数据,不读任何 2026 行情,不算任何收益。**
结果追加到 research/ledger/fwd_adapter_equiv.csv(每项一行:对象、比较范围、最大绝对差、逐位相同比例、是否通过)。

    python3 scripts/fwd_adapter_equiv.py m5 statind cost risk idxw real pead
"""
from __future__ import annotations

import os
import sys
import time
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from afactor.config import LEDGER, TMP, DAILY

OUT = LEDGER / "fwd_adapter_equiv.csv"
TOL = 1e-6
_rows: list[dict] = []


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def _cmp(obj: str, scope: str, a, b, tol: float = TOL, note: str = "") -> bool:
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    assert a.shape == b.shape, (obj, a.shape, b.shape)
    nan_same = bool(np.array_equal(np.isnan(a), np.isnan(b)))
    fin = np.isfinite(a) & np.isfinite(b)
    d = np.abs(a[fin] - b[fin])
    mx = float(d.max()) if d.size else 0.0
    rel = float((d / np.maximum(np.abs(b[fin]), 1e-12)).max()) if d.size else 0.0
    same = float((a[fin] == b[fin]).mean()) if d.size else 1.0
    ok = nan_same and (mx <= tol or rel <= tol)
    _rows.append(dict(adapter=obj, scope=scope, n_cells=int(a.size), n_finite=int(fin.sum()), nan_pattern_same=nan_same,
                      max_abs_diff=mx, max_rel_diff=rel, bit_identical_share=same, tol=tol, passed=ok, note=note,
                      run_at=time.strftime("%Y-%m-%d %H:%M:%S")))
    log(f"[{obj}] {scope}: max|d|={mx:.3g} rel={rel:.3g} 逐位相同 {same:.4%} NaN 位置一致={nan_same} -> {'通过' if ok else '不通过'}")
    return ok


def _flush():
    new = pd.DataFrame(_rows)
    old = pd.read_csv(OUT) if OUT.exists() else pd.DataFrame()
    if len(old):
        old = old[~(old["adapter"].isin(new["adapter"]) & old["scope"].isin(new["scope"]))]
    pd.concat([old, new], ignore_index=True).to_csv(OUT, index=False)
    _rows.clear()


# ------------------------------------------------------------------ 适配器 A:1 分钟 -> 5 分钟原料
def eq_m5():
    from afactor import fwd_adapters as FA_
    from afactor import minute5 as M5
    r = FA_.minute_month_features("202512", drop_bj=False, clean=True)
    for f, arr in r["fields"].items():
        ref = pd.read_parquet(M5.M5_DIR / f"{f}.parquet").reindex(index=r["dates"], columns=r["codes"])
        _cmp("A minute_month_features", f"202512/{f}", arr, ref.to_numpy(np.float32),
             note="drop_bj=False(与 cache/m5 同一输入);clean_minute_month 在研究期文件上为恒等")
    _flush()


# ------------------------------------------------------------------ 适配器 D:统计行业标签
def eq_statind():
    from afactor import fwd_adapters as FA_
    from afactor import statind as SI
    close = pd.read_parquet(DAILY / "close_px.parquet").astype(np.float32)
    ref = SI.load_labels()
    for q in ("2025Q2", "2025Q3"):
        L = FA_.statind_labels_for_quarters(close, [q])
        nxt = pd.Period(q, "Q") + 1
        m = L.index.to_period("Q") == nxt
        _cmp("D statind_labels_for_quarters", f"标签季度 {q} -> 应用于 {nxt}",
             L[m].to_numpy(np.float64), ref.reindex(index=L.index[m], columns=L.columns).to_numpy(np.float64),
             tol=0.0, note="KMeans 同种子同数据,要求逐位相同")
    _flush()


# ------------------------------------------------------------------ 适配器 C:成本模型(放宽守卫 + 未复权价换源)
def eq_cost():
    from afactor import fwd_adapters as FA_
    from afactor import minute5 as M5, newdata as ND
    from afactor.impact import CostModel
    p_ref = M5.m5_panel(signal=True, exec_=True)
    cm_ref = CostModel(p_ref, impact_y=1.0)
    p_ad = M5.m5_panel(signal=True, exec_=True)
    rc = ND.raw_close(p_ad.dates, p_ad.codes)               # 研究期同一数据源,经适配器的换源钩子传入
    with FA_.lifted_guards(raw_close=rc):
        cm_ad = CostModel(p_ad, impact_y=1.0)
    rows = np.flatnonzero(p_ref.dates.year == 2025)
    for nm in ("_hs", "_sigma", "_adv"):
        _cmp("C CostModel(lifted_guards)", f"2025/{nm}", getattr(cm_ad, nm)[rows], getattr(cm_ref, nm)[rows], tol=0.0)
    _flush()


# ------------------------------------------------------------------ 适配器 C:风险模型(放宽守卫)
def eq_risk():
    """在同一份研究期输入(5 分钟口径面板截取 2024-01-02 ~ 2025-12-31)上分别用冻结代码与适配器各建一次风险模型,
    比较 10 个 2025 年决策日的 cov / X / spec / mask。"""
    from afactor import fwd_adapters as FA_, minute5 as M5, factors as FA
    from afactor.barra import RiskModel

    def sub_panel():
        base = M5.m5_panel(signal=True, exec_=True)
        keep = base.dates[base.dates >= pd.Timestamp("2024-01-02")]

        class _Sub(type(base)):
            def raw(self, name):
                return super().raw(name).loc[keep[0]:keep[-1]]
        p = _Sub()
        return p
    t0 = time.time()
    p1 = sub_panel()
    rm1 = RiskModel(cache_dir=TMP / "fwd_equiv_risk_frozen").build(p1, "open10", force=True, verbose=False)
    log(f"[risk] 冻结代码构建 {time.time() - t0:.0f}s")
    p2 = sub_panel()
    with FA_.lifted_guards():
        rm2 = RiskModel(cache_dir=TMP / "fwd_equiv_risk_adapter").build(p2, "open10", force=True, verbose=False)
    log(f"[risk] 适配器构建 {time.time() - t0:.0f}s")
    avail = np.flatnonzero(np.isfinite(rm1.cov[:, 0, 0]) & (rm1.dates.year == 2025))
    pick = avail[np.linspace(0, len(avail) - 1, 10).astype(int)]
    ok = True
    for i in pick:
        tag = str(rm1.dates[i].date())
        ok &= _cmp("C RiskModel.build(lifted_guards)", f"{tag}/cov", rm2.cov[i], rm1.cov[i], tol=0.0)
        ok &= _cmp("C RiskModel.build(lifted_guards)", f"{tag}/X", np.asarray(rm2.X[i]), np.asarray(rm1.X[i]), tol=0.0)
        ok &= _cmp("C RiskModel.build(lifted_guards)", f"{tag}/spec", np.asarray(rm2.spec[i]), np.asarray(rm1.spec[i]),
                   tol=0.0)
        ok &= _cmp("C RiskModel.build(lifted_guards)", f"{tag}/mask", np.asarray(rm2.mask[i]).astype(float),
                   np.asarray(rm1.mask[i]).astype(float), tol=0.0)
    _flush()


# ------------------------------------------------------------------ 指数权重计价:由快照反推 t 日收盘权重
def eq_idxw():
    """研究期:取 PIT 代理在某个定期调整区间内某日 d_S 的权重当"快照",按 w_i(t) ∝ w_i(d_S)·P_i(t)/P_i(d_S)
    反推同一区间内其它日的收盘权重,与 index_proxy.proxy_weights 的 W[t] 比较(两者都是"股本不变的市值权重")。"""
    from afactor import fwd_weights as FW
    from afactor import index_proxy as IP
    from afactor.panel import Panel
    p = Panel()
    W = IP.proxy_weights(p, "000852")
    anchor = p.raw("close_px").where(p["valid"]).ffill()
    for dS, rng in (("2025-11-28", ("2025-06-16", "2025-12-12")), ("2025-03-31", ("2024-12-16", "2025-06-13"))):
        dS = pd.Timestamp(dS)
        snap = W.loc[dS]
        snap = snap[snap > 0]
        days = W.index[(W.index >= rng[0]) & (W.index <= rng[1])]
        Wb = FW.backout_close_weights(snap, dS, anchor.loc[days[0]:days[-1]].reindex(columns=p.codes),
                                      anchor_at_dS=anchor.loc[dS])
        same_members = (W.loc[days] > 0).eq(W.loc[dS] > 0, axis=1).all(axis=1)
        dd = days[same_members.to_numpy()]
        _cmp("E backout_close_weights", f"代理 d_S={dS.date()} 区间内同一成分的 {len(dd)} 天",
             Wb.loc[dd].to_numpy(), W.loc[dd].to_numpy(), tol=1e-5,
             note="代理权重按收盘价漂移(股本不变),反推应只有浮点误差")
    _flush()


# ------------------------------------------------------------------ 真实 000852 对齐到信号日
def eq_real():
    from afactor import fwd_weights as FW
    from afactor import index_proxy as IP
    from afactor.panel import Panel
    p = Panel()
    ref = IP.index_returns_on_signal_days(p, "open10", "000852")
    mine = FW.real_index_on_signal_days(p.dates, "open10")
    common = ref.index[ref.index >= pd.Timestamp("2021-01-01")]
    _cmp("F real_index_on_signal_days", "2021-2025 信号日", mine.reindex(common).to_numpy(), ref.reindex(common).to_numpy(),
         tol=1e-9, note="新浪 sh000852 快照(前向下载)vs ETF 项目冻结快照;收盘对收盘对齐到信号日同式")
    _flush()


# ------------------------------------------------------------------ PEAD:事件表与 2025Q3 批次逐行比较
def eq_pead():
    """研究期模式(截止日 2025-12-31、研究期日历、研究期 ST 名单、不拼东财):
      1) fwd_pead.main_events(load_income_fwd(False)) vs 冻结事件表 pead_events.parquet 的主事件(逐行);
      2) fwd_pead.build_batches 在 2025Q3 的 E / B / B@tol05 行 vs 冻结批次表 pead_batches.parquet(逐行)。
    批次表生成于 02:15(fixb 截止日改造之前);匹配变量里的 log 市值经 newdata.total_mv,fixb 后 29 只股票的研究期市值有变,
    所以冻结函数今天重跑也可能与旧批次表在少数 B 槽位上不同 —— 另报"冻结函数今天重跑"的对照。"""
    import pead_build_batches as PBB
    from afactor import earnings_events as EE, fwd_pead as FP, newdata as ND
    from afactor.panel import Panel
    inc = FP.load_income_fwd(splice_em=False)
    mine = FP.main_events(inc, EE.trading_calendar())
    ref = EE.main_events(EE.load_events())
    key = ["stock_code", "end_date", "P", "V"]
    a = mine[key + ["value"]].sort_values(key).reset_index(drop=True)
    b = ref[key + ["value"]].sort_values(key).reset_index(drop=True)
    same_rows = len(a) == len(b) and bool((a[key].astype(str).values == b[key].astype(str).values).all())
    _cmp("G fwd_pead.main_events", f"研究期主事件 {len(b)} 行(键一致={same_rows})",
         a["value"].to_numpy() if same_rows else np.zeros(1), b["value"].to_numpy() if same_rows else np.ones(1), tol=0.0)
    # 批次
    db = PBB.sealed_daily_basic()
    p = Panel()
    ev, pool, Q, Qn, _ = PBB.load_qualified_event_pool(p)
    mv = PBB.sealed_total_mv(p.dates, p.codes)
    circ = ND.circ_mv(p.dates, p.codes)
    qv, _, ind_lab = FP.match_inputs(p, Q, mv, circ)
    rng = (pd.Timestamp("2025-07-01"), pd.Timestamp("2025-09-30"))
    Bm, _ = FP.build_batches(pool, Qn, qv, ind_lab, p.dates, p.codes, 0, T_range=rng)
    Bf = pd.read_parquet(TMP / "pead_batches.parquet")
    Bf = Bf[Bf["arm"].isin(["E", "B", "B@tol05"]) & (Bf["date_T"] >= rng[0]) & (Bf["date_T"] <= rng[1])]
    k = ["arm", "rep", "T_pos", "slot"]
    J = Bm.merge(Bf, on=k, how="outer", suffixes=("_m", "_f"), indicator=True)
    for arm in ("E", "B", "B@tol05"):
        x = J[J["arm"] == arm]
        both = x["_merge"] == "both"
        same = both & (x["j_m"] == x["j_f"]) & (x["level_m"] == x["level_f"])
        _rows.append(dict(adapter="G fwd_pead.build_batches", scope=f"2025Q3/{arm} vs 冻结批次表(02:15 生成)",
                          n_cells=int(len(x)), n_finite=int(both.sum()), nan_pattern_same=bool(both.all()),
                          max_abs_diff=float(1 - same.mean()), max_rel_diff=np.nan, bit_identical_share=float(same.mean()),
                          tol=0.0, passed=bool(same.all()),
                          note="槽位 (arm, rep, T_pos, slot) 的对照股票与匹配层级逐行比较;max_abs_diff 栏 = 不一致比例",
                          run_at=time.strftime("%Y-%m-%d %H:%M:%S")))
        log(f"[G build_batches] 2025Q3/{arm}: {len(x)} 行,逐行相同 {same.mean():.4%}")
    _flush()


def eq_pead_today():
    """把冻结的 pead_build_batches.main() 在今天的研究期输入上原样重跑(输出目录重定向到 cache/_tmp/fwd_equiv_pead/,
    不碰原有 PEAD 产物),与适配器 fwd_pead.build_batches 在同一输入上的 2025Q3 结果逐行比较 —— 隔离"适配器公式"与
    "fixb 之后研究期市值输入变化"两件事。"""
    import pead_build_batches as PBB
    from afactor import fwd_pead as FP, newdata as ND
    from afactor.panel import Panel
    out = TMP / "fwd_equiv_pead"
    out.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copy2(LEDGER / "pead_frozen_inputs.json", out / "pead_frozen_inputs.json")          # 只读输入的副本
    shutil.copy2(TMP / "comp_v5_score_open10.parquet", out / "comp_v5_score_open10.parquet")  # (md5 核对照常进行)
    saved = (PBB.TMP, PBB.LEDGER)
    PBB.TMP, PBB.LEDGER = out, out
    t0 = time.time()
    try:
        PBB.main()
    finally:
        PBB.TMP, PBB.LEDGER = saved
    log(f"[pead_today] 冻结 main() 重跑 {time.time() - t0:.0f}s")
    Bf = pd.read_parquet(out / "pead_batches.parquet")
    db = PBB.sealed_daily_basic()
    p = Panel()
    ev, pool, Q, Qn, _ = PBB.load_qualified_event_pool(p)
    mv = PBB.sealed_total_mv(p.dates, p.codes)
    circ = ND.circ_mv(p.dates, p.codes)
    qv, _, ind_lab = FP.match_inputs(p, Q, mv, circ)
    rng = (pd.Timestamp("2025-07-01"), pd.Timestamp("2025-09-30"))
    Bm, _ = FP.build_batches(pool, Qn, qv, ind_lab, p.dates, p.codes, 0, T_range=rng)
    Bf = Bf[Bf["arm"].isin(["E", "B", "B@tol05"]) & (Bf["date_T"] >= rng[0]) & (Bf["date_T"] <= rng[1])]
    k = ["arm", "rep", "T_pos", "slot"]
    J = Bm.merge(Bf, on=k, how="outer", suffixes=("_m", "_f"), indicator=True)
    for arm in ("E", "B", "B@tol05"):
        x = J[J["arm"] == arm]
        both = x["_merge"] == "both"
        same = both & (x["j_m"] == x["j_f"]) & (x["level_m"] == x["level_f"])
        _rows.append(dict(adapter="G fwd_pead.build_batches", scope=f"2025Q3/{arm} vs 冻结 main() 今天重跑(同一输入)",
                          n_cells=int(len(x)), n_finite=int(both.sum()), nan_pattern_same=bool(both.all()),
                          max_abs_diff=float(1 - same.mean()), max_rel_diff=np.nan, bit_identical_share=float(same.mean()),
                          tol=0.0, passed=bool(same.all()),
                          note="同一份研究期输入上冻结代码与适配器逐行比较;这一项是等价性判据(与 02:15 旧批次表的差来自输入变化)",
                          run_at=time.strftime("%Y-%m-%d %H:%M:%S")))
        log(f"[G build_batches vs 冻结今天重跑] 2025Q3/{arm}: {len(x)} 行,逐行相同 {same.mean():.4%}")
    _flush()


if __name__ == "__main__":
    for st in sys.argv[1:]:
        t0 = time.time()
        log(f"===== {st} =====")
        globals()[f"eq_{st}"]()
        log(f"===== {st} 完成 {time.time() - t0:.0f}s =====")
