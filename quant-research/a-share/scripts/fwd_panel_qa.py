"""前向面板质检(前缀 fwd;预注册 FORWARD_TEST_PREREG §4.8、§3.3,任务第 2 步)。

**不计算任何前向策略收益、IC、超额。** 各节:
  overlap2025  与研究面板(桥接 C 口径 m5_panel)在 2025 年的一致性:日级字段、ST、决策池、COMP_v5 成员与打分
  splice       拼接核对:面板收盘对收盘收益 vs baostock 日线官方涨跌幅 pctChg(07-02 起逐股票日,数据对数据);
               2025-01-03 ~ 08-17 全段 vs tushare 官方复权收益(未复权收盘 × adj_factor,逐月计数;审查 P0 后新增);
               2025-12-31 -> 2026-01-05、08-17 -> 08-18 两个接缝日;除权日逐只抽查(>= 20 只)
  coverage     各原料按来源分段的覆盖率(V1000 决策池内有限值比例)
  engine2025   引擎回归断言在前向面板上成立(用 2025 年数据,不用 2026 收益):exposure<=1、对齐峰值 k=0、
               全池等权 vs 同频池基准 超额+成本 ≈ 0;并与研究期 m5_panel 同一调用比较。
               **前向面板先截到 2025-12-31**(审查 P1:不截断时信号日 12-30 / 12-31 会在 2026-01-05 / 06 成交实现,
               等于计算了预热期收益);比较前先断言两条序列索引相同
  overlap0817  预注册 §4.8 第 3 条:08-03 ~ 08-17 baostock 5 分钟原料 vs 1 分钟聚合原料的逐日截面秩相关(沪 / 深分开),
               以及 COMP_v5 打分(最后 11 天分别用两个来源,其余相同)的秩相关与 Top20 重合率
  lastday      预注册 §3.3:10-09 数据缺失面(V1000 池内有执行价的股票比例)
结果写 research/ledger/fwd_panel_qa.csv;日志 logs/fwd_panel_qa.log。

    python3 scripts/fwd_panel_qa.py overlap2025 splice coverage engine2025 overlap0817 lastday
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

from afactor.config import LEDGER, TMP
from afactor.fwd_panel import ForwardPanel, forward_context, PANEL_DIR
import fwd_common as FC

OUT = LEDGER / "fwd_panel_qa.csv"
_rows: list[dict] = []
log = FC.log


def rec(section, item, value, note=""):
    _rows.append(dict(section=section, item=item, value=value, note=note))
    v = f"{value:.6g}" if isinstance(value, (float, np.floating)) else value
    log(f"[{section}] {item} = {v}" + (f"  ({note})" if note else ""))


def flush():
    """整节重写:本次写到的 section 的旧行先全部删掉(条目改名时不留陈旧行)。"""
    secs = {r["section"] for r in _rows}
    FC.ledger_append(OUT, _rows, drop_where=lambda o: o["section"].isin(secs))
    _rows.clear()


def _cs_spearman_row(a: np.ndarray, b: np.ndarray, m: np.ndarray) -> float:
    ok = m & np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 50:
        return np.nan
    return float(pd.Series(a[ok]).rank().corr(pd.Series(b[ok]).rank()))


# ============================================================================== overlap2025
def qa_overlap2025():
    from afactor import minute5 as M5, fundamental as FU, newdata as ND, strategy as ST
    p = ForwardPanel()
    q = M5.m5_panel(signal=True, exec_=True)
    d25 = p.dates[p.dates.year == 2025]
    codes = q.codes
    for f in ("amount", "amt_open10", "rng", "rvol", "semibeta_mp"):
        a = p.raw(f).reindex(index=d25, columns=codes).to_numpy(np.float64)
        b = q.raw(f).reindex(index=d25, columns=codes).to_numpy(np.float64)
        same = (a == b) | (np.isnan(a) & np.isnan(b))
        rec("overlap2025", f"{f}_bit_identical_share", float(same.mean()), "前向面板 2025 行 vs 研究期 m5_panel(C 口径)")
    # 价格水平字段:审查 P0 之后 <= 08-17 统一到 08-17 单一基准(未复权 × AF(t)/AF(08-17)),与研究期面板的基准不同,
    # 只比较日收益(逐股票日 |Δlog 收益|)与日内比值
    import fwd_build_panel as FB
    for f in FB.PRICE_LEVEL_FIELDS:
        a = p.raw(f).reindex(index=d25, columns=codes).astype(np.float64)
        b = q.raw(f).reindex(index=d25, columns=codes).astype(np.float64)
        rec("overlap2025", f"{f}_finite_mask_identical", bool((a.notna() == b.notna()).to_numpy().all()), "")
        dr = (np.log(a / a.shift(1)) - np.log(b / b.shift(1))).abs().to_numpy()
        fin = np.isfinite(dr)
        by = pd.Series((dr > 1e-3).sum(axis=1), index=d25).groupby(d25.to_period("M")).sum()
        rec("overlap2025", f"{f}_dlogret_vs_research_le_1e-4_share", float((dr[fin] <= 1e-4).mean()),
            f"n={int(fin.sum())};>1e-4 {int((dr[fin] > 1e-4).sum())} 个、>1e-3 {int((dr[fin] > 1e-3).sum())} 个"
            f"(其中 2025-12 {int(by.get(pd.Period('2025-12', 'M'), 0))} 个:研究期 2025-12 月文件的基准变化与月内未复权);"
            "差异全部是研究期数据与 tushare 官方复权的不一致处")
        if f != "close_px":
            ra = (a / p.raw("close_px").reindex(index=d25, columns=codes).astype(np.float64)).to_numpy()
            rb = (b / q.raw("close_px").reindex(index=d25, columns=codes).astype(np.float64)).to_numpy()
            dd = np.abs(ra / rb - 1.0)
            rec("overlap2025", f"{f}_over_close_rel_diff_max", float(np.nanmax(dd)), "日内比值(价格字段 / 收盘)与研究期相同(float32 舍入级)")
    a = p.raw("raw_close").reindex(index=d25, columns=codes)
    b = ND.raw_close(d25, codes).where(q.raw("close_px").reindex(index=d25, columns=codes).notna())
    rec("overlap2025", "raw_close_bit_identical_share", float(((a == b) | (a.isna() & b.isna())).to_numpy().mean()),
        "vs newdata.raw_close(daily_basic)")
    st_f = p["is_st"].reindex(index=d25, columns=codes)
    st_r = FU.build_st_panel(d25, codes)
    rec("overlap2025", "is_st_identical_share", float((st_f == st_r).to_numpy().mean()), "vs fundamental.build_st_panel(研究期截止日)")
    uf = ST.universe(p, "V1000").reindex(index=d25, columns=codes)
    ur = ST.universe(q, "V1000").reindex(index=d25, columns=codes)
    eq_day = (uf == ur).all(axis=1)
    rec("overlap2025", "V1000_universe_identical_days", int(eq_day.sum()),
        f"共 {len(d25)} 天;首个完全一致日 {eq_day[eq_day].index.min().date()}(面板起点 2025-01-02 的 adv20/上市天数预热)")
    rec("overlap2025", "V1000_universe_identical_from_2025-02-06", bool(eq_day[d25 >= "2025-02-06"].all()),
        "2025-02-06 起 adv20 的 20 日窗口完全落在面板内")
    # COMP_v5 打分与成员:前向面板 vs 研究期 m5_panel(同一冻结函数)
    import m5_bridge as MB
    from afactor import factors as FA, ops
    members, signs, _ = MB.load_members()
    with forward_context("前向检验 QA:2025 年重叠段打分对照"):
        s_f = FC.comp_score(p)
    s_r = pd.read_parquet(MB.SCORE5).reindex(index=d25, columns=codes)
    dq4 = d25[d25 >= "2025-10-01"]
    U = ST.universe(q, "V1000").reindex(index=d25, columns=codes).to_numpy(bool)
    rc = [_cs_spearman_row(s_f.loc[d].reindex(codes).to_numpy(), s_r.loc[d].to_numpy(), U[d25.get_loc(d)]) for d in dq4]
    rec("overlap2025", "score_rankcorr_Q4_mean", float(np.nanmean(rc)), f"vs m5_bridge 打分(m5_score_5m),2025Q4 {len(dq4)} 天")
    rec("overlap2025", "score_rankcorr_Q4_min", float(np.nanmin(rc)), "")
    dall = d25[d25 >= "2025-03-01"]
    rc2 = [_cs_spearman_row(s_f.loc[d].reindex(codes).to_numpy(), s_r.loc[d].to_numpy(), U[d25.get_loc(d)]) for d in dall]
    rec("overlap2025", "score_rankcorr_2025Mar-Dec_mean", float(np.nanmean(rc2)), "")
    rec("overlap2025", "score_rankcorr_2025Mar-Dec_min", float(np.nanmin(rc2)),
        f"最低日 {dall[int(np.nanargmin(rc2))].date()};差异来自面板起点 2025-01-02 的长窗口预热与 fixb 后的财报 PIT")
    # 成员级(2025Q4):定位差异
    with forward_context("前向检验 QA:成员级对照"):
        for n in members:
            fn = FA.REGISTRY[n][1]
            zf = fn(p).reindex(index=dq4, columns=codes)
            zr = fn(q).reindex(index=dq4, columns=codes)
            r_ = [_cs_spearman_row(zf.loc[d].to_numpy(np.float64), zr.loc[d].to_numpy(np.float64), U[d25.get_loc(d)])
                  for d in dq4[::5]]
            rec("overlap2025_member", n, float(np.nanmean(r_)), "2025Q4 每 5 天抽样的截面秩相关均值(前向面板 vs 研究 m5_panel)")
    flush()


# ============================================================================== splice
def qa_splice():
    from afactor import fwd_data as FD
    p = ForwardPanel()
    c = p.raw("close_px").astype(np.float64)
    r = c / c.shift(1) - 1.0                                   # 面板收盘对收盘(复权后)
    d = FD.load_daily()
    d = d[(d["tradestatus"].astype(float) == 1) & (d["date"] >= "2026-07-02")]
    pc = d.pivot(index="date", columns="code", values="pctChg").reindex(columns=p.codes) / 100.0
    pc = pc.reindex(r.index[r.index >= "2026-07-02"])
    rr = r.reindex(pc.index)
    prev_ok = c.shift(1).reindex(pc.index).notna()
    m = pc.notna() & rr.notna() & prev_ok
    diff = (rr - pc).abs().where(m)
    # pctChg 四舍五入到 1e-6 量级(百分数 4 位小数);价格 2 位小数 -> 1e-4 内视为一致
    ok = (diff <= 1e-4)
    rec("splice", "ret_vs_pctChg_within_1e-4_share_0702_1009", float(ok.to_numpy()[m.to_numpy()].mean()),
        f"面板复权收盘的日收益 vs baostock 官方涨跌幅(按 preclose,已含除权),{int(m.to_numpy().sum())} 个股票日")
    rcp = p.raw("raw_close")
    lst = diff[diff > 1e-4].stack()
    rec("splice", "ret_vs_pctChg_mismatch_list", len(lst),
        "; ".join(f"{c_}@{d_.date()}:{v_:.5f}(未复权收盘{'有' if np.isfinite(rcp.at[d_, c_]) else '缺'})" for (d_, c_), v_ in lst.head(20).items()))
    for lab, a, b in (("pre_splice_0801_0817", "2026-08-01", "2026-08-17"), ("splice_day_0818", "2026-08-18", "2026-08-18"),
                      ("post_splice_0819_1009", "2026-08-19", "2026-10-09")):
        mm = m.loc[a:b].to_numpy()
        rec("splice", f"ret_vs_pctChg_within_1e-4_share[{lab}]", float(ok.loc[a:b].to_numpy()[mm].mean()),
            f"n={int(mm.sum())}")
    bad = diff.loc["2026-08-18"].dropna().sort_values(ascending=False)
    rec("splice", "splice_day_0818_n_mismatch_gt_1e-4", int((bad > 1e-4).sum()),
        "接缝日(08-17 一分钟收盘 -> 08-18 baostock 日线收盘)收益与官方涨跌幅不一致的股票数;例 "
        + ",".join(f"{k}:{v:.4f}" for k, v in bad.head(5).items()))
    # 除权日抽查:08-18 起 ratio != 1 的事件,面板收益 vs 官方涨跌幅
    ratio = pd.read_parquet(TMP / "fwd_aux" / "bs_ratio.parquet")
    ev = ratio.loc["2026-08-18":].stack()
    ev = ev[ev != 1.0].reset_index()
    ev.columns = ["date", "code", "ratio"]
    rng = np.random.default_rng(20261010)
    pick = ev.iloc[rng.choice(len(ev), size=min(30, len(ev)), replace=False)].sort_values(["date", "code"])
    n_ok = 0
    lines = []
    for _, e in pick.iterrows():
        a_ = rr.at[e["date"], e["code"]] if e["code"] in rr.columns else np.nan
        b_ = pc.at[e["date"], e["code"]] if e["code"] in pc.columns else np.nan
        good = np.isfinite(a_) and np.isfinite(b_) and abs(a_ - b_) <= 1e-4
        n_ok += int(good)
        lines.append(f"{e['code']}@{e['date'].date()} 比例{e['ratio']:.4f} 面板{a_:+.4f} 官方{b_:+.4f}")
    rec("splice", "exrights_spotcheck_30_ok", n_ok, "08-18 起除权事件随机抽 30 只:面板复权收益与官方涨跌幅 1e-4 内;"
        + " | ".join(lines[:8]))
    ev_all = ev.copy()
    ev_all["panel"] = [rr.at[a, b] if b in rr.columns else np.nan for a, b in zip(ev_all["date"], ev_all["code"])]
    ev_all["official"] = [pc.at[a, b] if b in pc.columns else np.nan for a, b in zip(ev_all["date"], ev_all["code"])]
    v = ev_all.dropna(subset=["panel", "official"])
    rec("splice", "exrights_all_ok_share", float(((v["panel"] - v["official"]).abs() <= 1e-4).mean()),
        f"08-18 起全部除权事件 {len(v)} 个(有面板与官方收益的)")
    # 2025-01-03 ~ 08-17 全段(审查 P0):面板复权收盘日收益 vs 官方复权收益(未复权收盘 × 目标复权系数;
    # 除权比例 <= 07-01 取 tushare adj_factor,07-02 ~ 08-17 取 baostock 交易所口径,见 fwd_build_panel.rebase_target),逐月计数
    import fwd_build_panel as FB
    rc = p.raw("raw_close").astype(np.float64)
    seg = c.index[c.index <= FB.M1_END]
    tgt = FB.rebase_target(p.dates, p.codes).reindex(index=seg)
    AF = FB.load_ts_adj_factor(p.dates, p.codes).reindex(index=seg)
    cs = c.reindex(seg)
    pr = cs / cs.shift(1)
    rel = (pr / ((rc.reindex(seg) * tgt) / (rc.reindex(seg) * tgt).shift(1)) - 1.0).iloc[1:]
    rel_ts = (pr / ((rc.reindex(seg) * AF) / (rc.reindex(seg) * AF).shift(1)) - 1.0).iloc[1:]
    fin = rel.notna()
    for lab, a, b in (("2025", "2025-01-03", "2025-12-31"), ("seam_20260105", "2026-01-05", "2026-01-05"),
                      ("2026-01-05~07-01", "2026-01-05", "2026-07-01"), ("07-02~08-17", "2026-07-02", "2026-08-17")):
        x = rel.loc[a:b]; f_ = fin.loc[a:b].to_numpy()
        n_bad3 = int(((x.abs() > 1e-3).to_numpy() & f_).sum()); n_bad4 = int(((x.abs() > 1e-4).to_numpy() & f_).sum())
        rec("splice", f"ret_vs_official_gt_1e-3[{lab}]", n_bad3,
            f"n={int(f_.sum())};>1e-4 的 {n_bad4} 个;官方 = 未复权收盘 × 目标复权系数(只用 08-17 及以前的除权)")
    byM = ((rel.abs() > 1e-3) & fin).sum(axis=1).groupby(rel.index.to_period("M")).sum()
    rec("splice", "ret_vs_official_gt_1e-3_by_month", "; ".join(f"{k}:{int(v)}" for k, v in byM.items()),
        "逐月计数(修正前:2026-06-01 1,456、07-01 713、05-06 602 等月初跳变,合计 3,397,见审查记录)")
    x = rel_ts.loc["2026-07-02":"2026-08-17"]; f_ = x.notna().to_numpy()
    rec("splice", "convention_tushare_vs_exchange_0702_0817_gt_1e-4", int(((x.abs() > 1e-4).to_numpy() & f_).sum()),
        f"同一面板若改用 tushare adj_factor 口径,07-02~08-17 与之相差 > 1e-4 的股票日(> 1e-3 的 {int(((x.abs() > 1e-3).to_numpy() & f_).sum())} 个,"
        "主要是低价股除权日 tushare 理论除权价与交易所 preclose 的舍入差,另有 689009.SH 08-07 CDR 分红 tushare 漏记 −2.9%);"
        "这是 07-02 起改用交易所口径的原因")
    mult = pd.read_parquet(TMP / "fwd_aux" / "rebase_mult_0817.parquet")
    chg = (mult - 1.0).abs() > 1e-4
    rec("splice", "rebase_mult_cells_changed", int(chg.to_numpy().sum()),
        f"<= 08-17 价格水平字段乘 m(t) 且 |m-1|>1e-4 的股票日(2026 年 {int(chg.loc['2026-01-01':].to_numpy().sum())});"
        f"涉及 {int(chg.any().sum())} 只;2025 年行的 m 多为逐股常数(研究期基准 -> 08-17 基准)")
    flush()


# ============================================================================== coverage
def qa_coverage():
    from afactor import strategy as ST, fwd_adapters as FA_
    p = ForwardPanel()
    U = ST.universe(p, "V1000")
    src = pd.read_parquet(PANEL_DIR / "_src_code.parquet")
    segs = {"2025(cache/m5)": ("2025-01-02", "2025-12-31"), "2026-01~07(1分钟)": ("2026-01-01", "2026-08-02"),
            "08-03~08-17(1分钟,缺口 bs 补)": ("2026-08-03", "2026-08-17"), "08-18~10-09(baostock)": ("2026-08-18", "2026-10-09"),
            "前向信号窗 09-01~09-30": ("2026-09-01", "2026-09-30")}
    for f in list(FA_.FIELDS) + ["raw_close", "turnover_rate", "mv", "float_share"]:
        x = p.raw(f)
        for lab, (a, b) in segs.items():
            u = U.loc[a:b].to_numpy(bool)
            v = np.isfinite(x.loc[a:b].to_numpy())
            rec("coverage", f"{f}[{lab}]", float(v[u].mean()) if u.any() else np.nan, "V1000 决策池内有限值比例")
    for lab, (a, b) in segs.items():
        s = src.loc[a:b]
        u = U.loc[a:b].to_numpy(bool)
        vals, cnt = np.unique(s.to_numpy()[u], return_counts=True)
        rec("coverage", f"source_mix[{lab}]", "; ".join(f"{int(v)}:{c}" for v, c in zip(vals, cnt)),
            "决策池股票日按来源编码计数(见 _meta.json src_codes)")
    st = p["is_st"]
    rec("coverage", "is_st_count_0901", int(st.loc["2026-09-01"].sum()), "前向首个信号日 ST 股数")
    lab = p._derived["statind_L"]
    rec("coverage", "statind_labeled_share_Sep", float((lab.loc["2026-09-01":"2026-09-30"].to_numpy() >= 0)[U.loc["2026-09-01":"2026-09-30"].to_numpy(bool)].mean()),
        "9 月决策池内有统计行业标签的比例")
    mv = p.raw("mv")
    rec("coverage", "mv_finite_in_pool_Sep", float(np.isfinite(mv.loc["2026-09-01":"2026-09-30"].to_numpy())[U.loc["2026-09-01":"2026-09-30"].to_numpy(bool)].mean()), "")
    bj = np.array([c.endswith(".BJ") for c in p.codes])
    rec("coverage", "BJ_in_V1000_pool_2025-12-31", int(U.loc["2025-12-31"].to_numpy()[bj].sum()),
        "研究期面板 2025-12 才有北交所列;2026 起北交所无前向数据,不进入决策池")
    rec("coverage", "new_listings_in_pool_Sep_max", int(U.loc["2026-09-01":"2026-09-30", p.codes[-16:]].sum(axis=1).max()),
        "08-18 以后上市的 16 只新股在 9 月决策池里的最多只数")
    flush()


# ============================================================================== engine2025
def qa_engine2025():
    """引擎三条回归断言在前向面板上成立(2025-03-03 ~ 2025-12-31,研究期数据;不用 2026 收益),
    并与研究期 m5_panel 的同一调用逐位比较(两面板 2025 年数据相同,结果应逐位相同)。"""
    from afactor import strategy as ST, minute5 as M5
    from afactor.impact import CostModel
    from afactor import fwd_adapters as FA_
    END25 = pd.Timestamp("2025-12-31")

    class _Cut25(ForwardPanel):
        """前向特征面板截到 2025-12-31(审查 P1):不让任何信号日的成交 / 实现落进 2026。"""
        def raw(self, name):
            if name not in self._cache:
                self._cache[name] = super().raw(name).loc[:END25]
            return self._cache[name]
    p = _Cut25()
    for k in list(p._derived):
        if isinstance(p._derived[k], pd.DataFrame):
            p._derived[k] = p._derived[k].loc[:END25]
    assert p.dates.max() == END25
    q = M5.m5_panel(signal=True, exec_=True)
    assert q.dates.max() == END25
    out = {}
    for nm, pp in (("forward", p), ("research_m5", q)):
        m = np.asarray((pp.dates >= "2025-03-03") & (pp.dates <= "2025-12-31"))
        ones = pd.DataFrame(1.0, index=pp.dates, columns=pp.codes, dtype=np.float32)
        b = ST.benchmark(pp, "V1000", "open10", date_mask=m, rebalance=5)
        if nm == "forward":
            with FA_.lifted_guards(raw_close=pp.raw("raw_close")):
                cm = CostModel(pp, impact_y=1.0)
        else:
            cm = CostModel(pp, impact_y=1.0)
        s = ST.run_strategy(ones, pp, "V1000", "open10", scheme="top", n=100000, rebalance=5, date_mask=m,
                            cost_model=cm, record=True)
        s0 = ST.run_strategy(ones, pp, "V1000", "open10", scheme="exclude", exclude_frac=0.0, rebalance=5, date_mask=m)
        out[nm] = (s, b, s0)
        rs, rb = s["rets"], b["rets"]
        assert rs.index.equals(rb.index) and rs.index.max() <= END25
        last = pp.dates.get_loc(rs.index[-1])
        assert last + 2 <= len(pp.dates) - 1 and pp.dates[last + 2] <= END25, "实现日越过 2025-12-31"
        rec("engine2025", f"{nm}_n_signal_days_and_last", f"{len(rs)} / {rs.index[-1].date()} -> 实现 {pp.dates[last + 2].date()}",
            "成交 / 实现全部在 2025 年内")
        cs = {k: float(rs.corr(rb.shift(k))) for k in (-2, -1, 0, 1, 2)}
        kk = max(cs, key=cs.get)
        exp_max = float(s["record"]["daily"]["exposure"].max())
        cost = s["record"]["daily"]["cost_total"]
        ex_pc = float((rs - rb + cost.reindex(rs.index)).iloc[4:].sum())
        rec("engine2025", f"{nm}_fullpool_Y1_exposure_max", exp_max, "全池等权(常数打分 top n=100000)、5 日、Y=1、1000 万")
        rec("engine2025", f"{nm}_fullpool_Y1_corr_peak_k", kk, str({k: round(v, 4) for k, v in cs.items()}))
        rec("engine2025", f"{nm}_fullpool_Y1_sum_excess_plus_cost_skip4", ex_pc,
            "2025-03-03~12-31 逐日 (r - r_bench + cost) 之和(去开头 4 个建仓日),应 ≈ 0(研究期可成交性残差约 +0.3pp/年)")
        ex0 = (s0["ann_return"] - b["ann_return"] + s0["cost_bps"] * 1e-4 * ST.TRADING_DAYS) * 100
        rec("engine2025", f"{nm}_fullpool_oldcost_excess_plus_cost_ann_pp", ex0, "桥接 m5_engine_checks 同一判据(旧成本)")
    for i, lab in ((0, "Y1 全池等权"), (1, "同频池基准"), (2, "旧成本全池等权")):
        a, b_ = out["forward"][i]["rets"], out["research_m5"][i]["rets"]
        assert a.index.equals(b_.index), f"{lab}:两条序列索引不同({len(a)} vs {len(b_)})"
        rec("engine2025", f"forward_vs_research_maxabsdiff[{lab}]", float((a - b_).abs().max()),
            f"n={len(a)},索引逐日相同;差异来自 2025 年价格字段统一到 08-17 基准(研究期面板在除权日与 tushare 官方复权的"
            "微小差异、2025-12 月文件基准问题被修正),修正前为 1e-17 级")
    flush()


# ============================================================================== overlap0817
def qa_overlap0817():
    """§4.8 第 3 条:只看原料 / 打分,不看收益。"""
    from afactor import strategy as ST, fwd_adapters as FA_
    import fwd_build_panel as FB
    p = ForwardPanel()
    dts, cs, fl = FB._load_npz(FB.BS5_FILE)
    ov = dts[(dts >= "2026-08-03") & (dts <= "2026-08-17")]
    U = ST.universe(p, "V1000")
    src = pd.read_parquet(PANEL_DIR / "_src_code.parquet")
    exch = np.array([c[-2:] for c in cs])
    for f in ("rvol", "gap_share", "smart_money", "amt_gini", "amt_openauc_share", "semibeta_mp", "twap_open10",
              "amt_first30_share", "amt_time_wgt", "vwap", "open_auction_px", "amt_open10", "rng"):
        x1 = p.raw(f).reindex(columns=cs)
        for ex in ("SH", "SZ", "ALL"):
            sel = np.ones(len(cs), bool) if ex == "ALL" else (exch == ex)
            vals = []
            for d in ov:
                k = dts.get_loc(d)
                a = fl[f][k].astype(np.float64)
                b = x1.loc[d].to_numpy(np.float64)
                m = U.loc[d].reindex(cs).fillna(False).to_numpy(bool) & sel & (src.loc[d].reindex(cs).to_numpy() == 2)
                vals.append(_cs_spearman_row(a, b, m))
            rec("overlap0817", f"{f}[{ex}]_rankcorr_mean", float(np.nanmean(vals)),
                f"baostock 5 分钟 vs 1 分钟聚合,V1000 池内逐日截面 Spearman,11 天均值(最低 {np.nanmin(vals):.4f})")
        if f == "twap_open10":
            for ex in ("SH", "SZ"):
                sel = exch == ex
                errs = []
                for d in ov:
                    k = dts.get_loc(d)
                    a = fl[f][k].astype(np.float64)
                    b = x1.loc[d].to_numpy(np.float64)
                    m = U.loc[d].reindex(cs).fillna(False).to_numpy(bool) & sel & np.isfinite(a) & np.isfinite(b) & \
                        (src.loc[d].reindex(cs).to_numpy() == 2)
                    errs.append(np.abs(np.log(a[m] / b[m])) * 1e4)
                e = np.concatenate(errs)
                rec("overlap0817", f"twap_open10_mid4_abs_log_err_bp[{ex}]_median", float(np.median(e)),
                    f"p90 {np.quantile(e, 0.9):.1f}bp;同一 mid4 公式,两源 bar 差异造成的执行价差")
    # COMP_v5 打分:最后 11 天换成 baostock 来源,其余相同
    class _Alt(ForwardPanel):
        def raw(self, name):
            if name not in self._cache:
                df = super().raw(name).copy()
                if name in FA_.FIELDS:
                    blk = pd.DataFrame(np.stack([fl[name][dts.get_loc(d)] for d in ov]), index=ov, columns=cs)
                    sub = blk.reindex(columns=df.columns)
                    df.loc[ov] = sub.where(sub.notna(), df.loc[ov]).astype(np.float32).to_numpy()
                    df = df.loc[:ov[-1]]
                else:
                    df = df.loc[:ov[-1]]
                self._cache[name] = df
            return self._cache[name]

    class _Base(ForwardPanel):
        def raw(self, name):
            if name not in self._cache:
                self._cache[name] = super().raw(name).loc[:ov[-1]]
            return self._cache[name]

    def _seed_cut(pp):
        for k in list(pp._derived):
            v = pp._derived[k]
            if isinstance(v, pd.DataFrame):
                pp._derived[k] = v.loc[:ov[-1]]
    pa, pb = _Alt(), _Base()
    _seed_cut(pa); _seed_cut(pb)
    with forward_context("前向检验 QA:重叠段打分两源对照(只看打分,不看收益)"):
        sa = FC.comp_score(pa, cache_file=None)
        sb = FC.comp_score(pb, cache_file=None)
    Ub = ST.universe(pb, "V1000")
    rcs, ovl = [], []
    for d in ov:
        u = Ub.loc[d].to_numpy(bool)
        a = sa.loc[d].to_numpy(np.float64); b = sb.loc[d].to_numpy(np.float64)
        rcs.append(_cs_spearman_row(a, b, u))
        ok = u & np.isfinite(a) & np.isfinite(b)
        ia = np.flatnonzero(ok)
        ta = set(ia[np.argsort(-a[ok])[:20]]); tb = set(ia[np.argsort(-b[ok])[:20]])
        ovl.append(len(ta & tb) / 20)
    rec("overlap0817", "comp_v5_score_rankcorr_mean", float(np.nanmean(rcs)),
        f"08-03~08-17 打分(原料最后 11 天换成 baostock)vs 1 分钟;最低 {np.nanmin(rcs):.4f};08-17 当日 {rcs[-1]:.4f}")
    rec("overlap0817", "comp_v5_top20_overlap_mean", float(np.mean(ovl)), f"08-17 当日 {ovl[-1]:.2f};桥接 1 分钟 vs 5 分钟口径约 0.60")
    flush()


# ============================================================================== lastday
def qa_lastday():
    """预注册 §3.3:如果 10-09 的数据缺失面超过 5%(按 V1000 池内有执行价的股票数计),最后一个信号日前移一天。只看覆盖。"""
    from afactor import strategy as ST
    p = ForwardPanel()
    U = ST.universe(p, "V1000")
    tw = p.raw("twap_open10")
    u = U.loc["2026-09-30"].to_numpy(bool)
    for d in ("2026-10-08", "2026-10-09"):
        v = np.isfinite(tw.loc[d].to_numpy())
        rec("lastday", f"exec_px_coverage_{d}_of_0930_pool", float(v[u].mean()), "09-30 的 V1000 池内有执行价的比例;< 95% 则最后信号日前移")
    cov = float(np.isfinite(tw.loc["2026-10-09"].to_numpy())[u].mean())
    rec("lastday", "decision", "保持 09-30 为最后信号日" if cov >= 0.95 else "最后信号日前移到 09-29",
        f"10-09 缺失面 {1 - cov:.2%}")
    flush()


if __name__ == "__main__":
    for st in sys.argv[1:]:
        t0 = time.time()
        log(f"===== {st} =====")
        globals()[f"qa_{st}"]()
        log(f"===== {st} 完成 {time.time() - t0:.0f}s =====")
