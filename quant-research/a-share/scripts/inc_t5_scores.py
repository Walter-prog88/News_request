"""T5 打分构造与数据验收(前缀 inc_t5;预注册 §3.4、§4、§12 步骤 2)。

阶段:
  verify      冻结哈希校验
  devs        登记预注册没写清、须在看到任何收益之前定的口径(inc_t5_deviations.csv)
  scores_lh   LongPanel 全部篮子打分 -> cache/_tmp/inc_t5_scores/lh_*.npy(T x N float32,全 5101 日)
  scores_min  Panel(研究期)全部篮子打分 -> cache/_tmp/inc_t5_scores/min_*.parquet
  qa          数据验收门槛 D1-D4、D7(D3 需要 EM/baostock 分红抓取完成)-> research/ledger/inc_t5_data_qa.csv
用法: python3 scripts/inc_t5_scores.py verify devs scores_lh scores_min qa
"""
from __future__ import annotations

import sys
import time

from inc_t5_common import *      # noqa: F401,F403
from inc_t5_common import ROOT, SC, FETCH, LEDGER, QA, TD, LE, log, log_dev, qa_row, merge_csv, verify_prereg

import numpy as np
import pandas as pd

from afactor import inc_t5_signals as SG


# ================================================================ verify / devs
def stage_verify():
    if not verify_prereg():
        log("[verify] 预注册冻结哈希不一致,停止")
        sys.exit(2)
    log("[verify] 预注册冻结哈希一致")


def stage_devs():
    s = "devs"
    log_dev(s, "D6_frozen_panel", "D6:B-MIN 的 COMP_v5 排雷重跑 vs freeze_ref_daily.parquet 的 T1_excl|Y1|strategy,B1 vs C|bench_pool5,最大差 ≤ 1e-10",
            "冻结序列属于 C 口径面板(m5_panel:5 分钟信号 + 5 分钟执行),不是 Panel()(A 口径 1 分钟)。D6 拆成两条:"
            "(a) 在 C 面板上用同一流水线复现冻结序列(≤1e-10);(b) B-MIN 主检验按预注册 §3.2 用 Panel() + comp_v5_score_open10,"
            "A 口径序列与冻结序列的差只报告(lh_bridge.csv 已知 A 口径 COMP_v5 超额 +5.69pp vs 冻结 +5.61pp)。",
            "预注册 D6 与 §3.2 互相矛盾(Panel() 上的序列不可能逐位等于 C 口径冻结序列);按 §3.2/§12 主文取 Panel()。")
    log_dev(s, "CORE_A2_weights", "A 轨:CORE-A2(未写权重)",
            "CORE-A2 = LIQ 2/3 + LOWVOL 1/3(只有一个卫星时 1/2 与 EW 重合)", "避免与 CMB-A2 重合")
    log_dev(s, "EW6_CORE6_weights", "CMB-EW6/CORE6/ERC6 加 REV(未写权重)",
            "EW6 = 各 1/6;CORE6 = LIQ 1/2、其余 5 个各 1/10;ERC6 = 6 篮子逆 TE,截到 [0.5/6, 2/6]", "与 EW5/CORE5/ERC5 同构")
    log_dev(s, "placebo_construction", "CMB-PLACEBO(s):LIQ + 4 个随机篮子等权,s = 0..9;随机篮子种子 0-9",
            "第 s 个安慰剂的 4 个随机篮子用独立种子 numpy.default_rng(1000*s + j),j = 0..3(每股一个固定 U(0,1));"
            "A 轨安慰剂 = 1/2 LIQ + 1/2 随机篮子(种子 default_rng(1000*s + 9)),s = 0..9", "预注册只给了 10 个种子,未说明 4 个篮子如何取")
    log_dev(s, "new_listing_fe", "BP 的 e⁺ 取 ≤ e 的最后一个有效收盘日",
            "报告期末早于上市日(新股首份财报)时,e⁺ 取该股第一个有效交易日的复权比;DIV 的 a 同理",
            "否则新股首年 BP/EP 全 NaN,D1 覆盖率受损")
    log_dev(s, "qual_sigma_final_rows", "QUAL 稳定分量:σ(ROE_q, 8 期,最少 6 期)",
            "σ 只在最终快照(每报告期最新版)上算;可见日 = 窗口内参与计算各期 avail_q 的最大值",
            "利润表每期只存一版,资产负债表多版本只影响分母(377 个报告期);as-of 快照行不参与 σ")
    log_dev(s, "qual_missing_component", "QUAL = mean[三个 cs_rank];金融股杠杆分量取 0.5",
            "三个分量任一缺失(非金融股)则 QUAL = NaN,不进篮子;金融股 Lev 分量 = 0.5", "避免用部分分量混排")
    log_dev(s, "em_empty_period", "EM stock_fhps_em 按报告期抓取",
            "akshare 对无预案的报告期抛 TypeError('NoneType'...):记为空表(2005Q1 等早期季度/一季报);2025Q4 报告期不抓(预案公告在 2026 年,封存)",
            "已逐期核对失败均为该异常;重试 3 次后一致")
    log_dev(s, "p6_definition", "P6:6 个变体中 ≥ 4 个满足 P1 的'三窗同向'",
            "'三窗同向' = IR(变体) > IR(LIQ) 在全期、开发窗、检验窗同时成立(不要求 ΔIR ≥ 0.10)", "字面解释")
    log_dev(s, "verdict_p1_subwindow", "判定规则 §10.2",
            "全期 ΔIR > 0 且 a > 0 且 P5 通过,但 P1 在某窗口不成立(或 P2 t<2,或 P3/P4/P6/P7 不满足)-> 证据不足;"
            "全期 ΔIR ≤ 0 或 a ≤ 0 或 P5 不通过 -> 无增量;全部 P1-P7 + D1-D6 通过 -> 有增量", "§10.2 表格未覆盖'P1 只在子窗口不成立'的情形,按保守方向归入证据不足")
    log_dev(s, "s4s5_depth", "S4/S5 = 不含 LIQ 的风格等权",
            "S4/S5 各篮子用 45/60 做主(生成检验用);另在 20/35 深度各跑一次作变体", "§11 只说 S4、S5 共 2 个")
    log_dev(s, "div_same_day_visibility", "分红可见日 = 预案公告日",
            "公告日 a 当天收盘信号即可见(与项目财报 avail <= t 约定一致);t+1 开盘成交", "沿用项目 PIT 约定")
    log_dev(s, "d5_e3_period", "D5:全池等权(付成本)− B1 ≈ −成本(|差| ≤ 0.3pp/年)",
            "在各轨道评价期上按几何超额 + 年成本报告;LongPanel 轨道若不满足 0.3pp 如实写出,并给 step1 同检验的既有值作对照", "step1 已知 LongPanel 的 E3 逐年残差可达 ±3pp(停牌)")
    log_dev(s, "ir_dir_cmb_vs_liq_window", "P1 IR 比较的窗口",
            "B-LH:全期 2017-01-03~2025-12-31、开发 2017-2019、检验 2020-2025;A 轨:2006~2025 / 2006-2019 / 2020-2025;B-MIN:2021-2025 / 2021-2023 / 2024-2025", "与 §8 一致")
    log("[devs] 已登记")


# ================================================================ scores_lh
def _lp_arrays():
    from afactor.lh_panel import LongPanel
    p = LongPanel(suspend_fill=True, open10_ratio="pool")
    close = p.raw("close_px").to_numpy(np.float64)
    raw = p.raw("close_raw").to_numpy(np.float64)
    amt = p.raw("amount").to_numpy(np.float64)
    trade = p.raw("trade").to_numpy(bool)
    valid = trade & (amt > 0) & np.isfinite(close)
    return p, close, raw, valid


def stage_scores_lh():
    from afactor import fundamental as FU
    SC.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    X = LE.load_inputs()
    dates, codes = X.dates, X.codes
    pool = np.asarray(X.arr("pool_q92"), bool)
    p, close, raw, valid = _lp_arrays()
    assert (p.dates == dates).all() and (p.codes == codes).all()
    save = lambda nm, a: np.save(SC / f"lh_{nm}.npy", np.asarray(a, np.float32))

    # ---- 价格类 ----
    for w in (60, 120, 250):
        save(f"LOWVOL{w}", SG.lowvol(close, valid, w, 60))
    for k in (20, 60):
        save(f"REV{k}", SG.reversal(close, valid, k))
    log(f"[scores_lh] 价格类 {time.time() - t0:.0f}s")

    # ---- 财报类 ----
    f_raw = np.where(np.isfinite(close) & np.isfinite(raw) & (raw > 0), close / raw, np.nan)
    rep = FU.build_report_level()
    log(f"[scores_lh] PIT 报告表 {rep.shape} {time.time() - t0:.0f}s")
    rr = SG.report_row_cols(rep, codes, dates, f_raw)
    sig = SG.roe_sigma_rows(rep[rep["pit_vintage"] == FU.PIT_FINAL])
    log(f"[scores_lh] 行级衍生 + ROE 稳定性 {len(rr)} 行 / sigma {len(sig)} 行 {time.time() - t0:.0f}s")
    bps = FU._pit_daily(rr, "bps_h", dates, codes).to_numpy(np.float64)
    roe = FU._pit_daily(rr, "roe_ttm", dates, codes).to_numpy(np.float64)
    lev = FU._pit_daily(rr, "lev", dates, codes).to_numpy(np.float64)
    sgm = FU._pit_daily(sig, "roe_sig_ttm", dates, codes).to_numpy(np.float64)
    with np.errstate(all="ignore"):
        BP = bps / close
        EP = roe * BP
    BP = np.where(valid, BP, np.nan); EP = np.where(valid, EP, np.nan)
    save("BP", BP); save("EP", EP)
    rbp, rep_ = SG.cs_rank_pool(BP, pool), SG.cs_rank_pool(EP, pool)
    save("VALUE", 0.5 * rbp + 0.5 * rep_)
    # 质量
    ct = rep.dropna(subset=["comp_type"]).drop_duplicates("stock_code", keep="last").set_index("stock_code")["comp_type"]
    import pandas as pd_
    sb = pd_.read_csv(ROOT / "data_8:21" / "stock_basic.csv").set_index("ts_code")["industry"]
    isfin = SG.fin_mask(codes, ct, sb)
    log(f"[scores_lh] 金融股 {int(isfin.sum())} 只(comp_type 缺失 {int((~codes.isin(ct.index)).sum())} 只,其中按行业补 "
        f"{int((~codes.isin(ct.index) & isfin).sum())})")
    r_roe = SG.cs_rank_pool(np.where(valid, roe, np.nan), pool)
    r_stab = SG.cs_rank_pool(np.where(valid, -sgm, np.nan), pool)
    lev_nf = np.where(valid & ~isfin[None, :], -lev, np.nan)            # 杠杆只在非金融股之间排名
    r_lev = SG.cs_rank_pool(lev_nf, pool)
    r_lev = np.where(isfin[None, :] & pool & valid, np.float32(0.5), r_lev)
    save("Q_ROE", np.where(valid, roe, np.nan)); save("Q_STAB", np.where(valid, -sgm, np.nan))
    save("Q_LEVR", r_lev)
    with np.errstate(all="ignore"):
        qual = (r_roe + r_stab + r_lev) / 3.0
    save("QUAL", qual)
    np.save(SC / "lh_isfin.npy", isfin)
    log(f"[scores_lh] 财报类 {time.time() - t0:.0f}s")

    log(f"[scores_lh] 完成 {time.time() - t0:.0f}s")


def stage_scores_div():
    """分红打分(需要 EM + baostock 抓取完成):LongPanel 的 DY365/DY730/DY365I 与 Panel 的 DIV。"""
    t0 = time.time()
    X = LE.load_inputs()
    dates, codes = X.dates, X.codes
    p, close, raw, valid = _lp_arrays()
    f_raw = np.where(np.isfinite(close) & np.isfinite(raw) & (raw > 0), close / raw, np.nan)
    save = lambda nm, a: np.save(SC / f"lh_{nm}.npy", np.asarray(a, np.float32))
    c6 = pd.Series(codes, index=[c[:6] for c in codes])
    dup = c6.index.duplicated(keep=False)
    log(f"[scores_div] 6 位代码重复 {int(dup.sum())} 个(重复的不映射)")
    code6_to_lp = c6[~dup].to_dict()
    ev, st = SG.load_dividend_events(FETCH / "em_div", FETCH / "bs_div_rows.csv", code6_to_lp, set(codes))
    ev.to_parquet(SC / "div_events.parquet")
    log(f"[scores_div] 分红事件 {st}")
    for nm, wd, io in (("DY365", 365, False), ("DY730", 730, False), ("DY365I", 365, True)):
        num = SG.dy_numerator(ev, dates, codes, f_raw, wd, io)
        with np.errstate(all="ignore"):
            dy = num.astype(np.float64) / close
        save(nm, np.where(valid, dy, np.nan))
    log(f"[scores_div] LongPanel 完成 {time.time() - t0:.0f}s")
    # Panel(前复权价 / 未复权价)
    from afactor import newdata as ND
    from afactor.panel import Panel
    pm = Panel()
    adj = pm.raw("close_px").to_numpy(np.float64)
    rawm = ND.raw_close(pm.dates, pm.codes).reindex(index=pm.dates, columns=pm.codes).to_numpy(np.float64)
    c6m = pd.Series(pm.codes, index=[c[:6] for c in pm.codes])
    c6m = c6m[~c6m.index.duplicated(keep=False)]
    evm, _ = SG.load_dividend_events(FETCH / "em_div", None, c6m.to_dict(), set(pm.codes))
    fm = np.where(np.isfinite(adj) & np.isfinite(rawm) & (rawm > 0), adj / rawm, np.nan)
    num = SG.dy_numerator(evm, pm.dates, pm.codes, fm, 365, False)
    with np.errstate(all="ignore"):
        dy = num.astype(np.float64) / adj
    pd.DataFrame(dy.astype(np.float32), index=pm.dates, columns=pm.codes).to_parquet(SC / "min_DIV.parquet")
    log(f"[scores_div] Panel 完成 {time.time() - t0:.0f}s")


# ================================================================ scores_min(研究期 Panel)
def stage_scores_min():
    from afactor import fundamental as FU, newdata as ND, factors as FA, strategy as ST
    from afactor.panel import Panel
    t0 = time.time()
    p = Panel()
    dates, codes = p.dates, p.codes
    U = ST.universe(p, "V1000").to_numpy(bool)
    adj = p.raw("close_px").to_numpy(np.float64)
    raw = ND.raw_close(dates, codes).reindex(index=dates, columns=codes).to_numpy(np.float64)
    mv = ND.total_mv(dates, codes).to_numpy(np.float64)
    FA.build_all()
    out = {}
    out["LOWVOL"] = FA.REGISTRY["dvol_120"][1](p).reindex(index=dates, columns=codes).replace([np.inf, -np.inf], np.nan)
    out["REV"] = FA.REGISTRY["rev_20"][1](p).reindex(index=dates, columns=codes).replace([np.inf, -np.inf], np.nan)
    rep = FU.build_report_level()
    E = FU._pit_daily(rep, "total_hldr_eqy_exc_min_int", dates, codes).to_numpy(np.float64)
    NI = FU._pit_daily(rep, "n_income_attr_p_ttm", dates, codes).to_numpy(np.float64)
    lev = FU._pit_daily(rep, "total_liab", dates, codes).to_numpy(np.float64) / \
        np.where(FU._pit_daily(rep, "total_assets", dates, codes).to_numpy(np.float64) > 0,
                 FU._pit_daily(rep, "total_assets", dates, codes).to_numpy(np.float64), np.nan)
    with np.errstate(all="ignore"):
        BP = np.where(E > 0, E / mv, np.nan)
        EP = np.where(E > 0, NI / mv, np.nan)
        roe = np.where(E > 0, NI / E, np.nan)
    sig = SG.roe_sigma_rows(rep[rep["pit_vintage"] == FU.PIT_FINAL])
    sgm = FU._pit_daily(sig, "roe_sig_ttm", dates, codes).to_numpy(np.float64)
    ct = rep.dropna(subset=["comp_type"]).drop_duplicates("stock_code", keep="last").set_index("stock_code")["comp_type"]
    sb = pd.read_csv(ROOT / "data_8:21" / "stock_basic.csv").set_index("ts_code")["industry"]
    isfin = SG.fin_mask(codes, ct, sb)
    rbp, rep_ = SG.cs_rank_pool(BP, U), SG.cs_rank_pool(EP, U)
    out["VALUE"] = 0.5 * rbp + 0.5 * rep_
    r_roe, r_stab = SG.cs_rank_pool(roe, U), SG.cs_rank_pool(-sgm, U)
    r_lev = SG.cs_rank_pool(-lev, U)
    r_lev = np.where(isfin[None, :] & U, np.float32(0.5), r_lev)
    out["QUAL"] = (r_roe + r_stab + r_lev) / 3.0
    out["BP"], out["EP"] = BP, EP
    for k, v in out.items():
        df = v if isinstance(v, pd.DataFrame) else pd.DataFrame(np.asarray(v, np.float32), index=dates, columns=codes)
        df.astype(np.float32).to_parquet(SC / f"min_{k}.parquet")
    log(f"[scores_min] 完成 {time.time() - t0:.0f}s")


# ================================================================ qa
def _month_ends(dates: pd.DatetimeIndex, a: str, b: str) -> np.ndarray:
    s = pd.Series(np.arange(len(dates)), index=dates).loc[a:b]
    return s.groupby([s.index.year, s.index.month]).last().to_numpy()


def stage_qa():
    t0 = time.time()
    X = LE.load_inputs()
    dates, codes = X.dates, X.codes
    pool = np.asarray(X.arr("pool_q92"), bool)
    rows = []
    ld = lambda n: np.load(SC / f"lh_{n}.npy", mmap_mode="r")
    # ---- D1 覆盖 ----
    for y in range(2017, 2026):
        ix = np.flatnonzero((dates.year == y))
        pm = pool[ix]
        den = pm.sum()
        for nm, thr in (("BP", 0.85), ("EP", 0.85), ("Q_ROE", 0.85), ("QUAL", 0.80), ("VALUE", 0.85)):
            a = np.asarray(ld(nm)[ix])
            frac = float((np.isfinite(a) & pm).sum() / den)
            rows.append(qa_row("D1", f"{nm}@{y}", frac, f">= {thr}", frac >= thr, "q92 池股票-日非空比例"))
    # ---- D2 BP 两种算法 ----
    from afactor import newdata as ND
    from scipy.stats import spearmanr
    mv = ND.total_mv(dates[dates >= "2020-01-02"], codes)
    from afactor import fundamental as FU
    rep = FU.build_report_level()
    d20 = dates[dates >= "2020-01-02"]
    E = FU._pit_daily(rep, "total_hldr_eqy_exc_min_int", d20, codes).to_numpy(np.float64)
    bp_alt = np.where(E > 0, E / mv.to_numpy(np.float64), np.nan)
    i20 = int(dates.searchsorted(pd.Timestamp("2020-01-02")))
    sp = []
    for ix in _month_ends(dates, "2020-01-02", "2025-12-31"):
        a = np.asarray(ld("BP")[ix]); b = bp_alt[ix - i20]
        m = pool[ix] & np.isfinite(a) & np.isfinite(b)
        if m.sum() > 200:
            sp.append(spearmanr(a[m], b[m]).correlation)
    rows.append(qa_row("D2", "BP_two_algorithms_spearman_median", float(np.median(sp)), ">= 0.95", float(np.median(sp)) >= 0.95,
                       f"月末 {len(sp)} 个截面;min {np.min(sp):.3f} p10 {np.percentile(sp, 10):.3f}"))
    # ---- D3 分红 ----
    ev = pd.read_parquet(SC / "div_events.parquet")
    evc = ev[ev["cash_ps"] > 0]
    for y in range(2016, 2026):
        ix = np.flatnonzero(dates.year == y)[-1]
        pm = np.flatnonzero(pool[ix])
        have = set(evc[(evc["ann"].dt.year == y)]["lp_code"])
        frac = float(np.mean([codes[j] in have for j in pm]))
        rows.append(qa_row("D3a", f"cash_plan_fraction@{y}", frac, "[0.40, 0.85]", 0.40 <= frac <= 0.85,
                           "年末 q92 池内当年有现金预案(公告年 = y)的比例"))
    bsr = pd.read_csv(FETCH / "bs_div_rows.csv", dtype=str)
    sv = bsr[bsr["kind"] == "surv"].copy()
    sv["ann"] = pd.to_datetime(sv["dividPlanAnnounceDate"], errors="coerce")
    sv["cash"] = pd.to_numeric(sv["dividCashPsBeforeTax"], errors="coerce")
    sv = sv[sv["ann"].notna() & (sv["cash"] > 0) & sv["ann"].dt.year.between(2016, 2024)]
    em = ev[ev["src"] == "em"]
    n_miss = n_match = n_close = 0
    for _, r in sv.iterrows():
        e = em[(em["lp_code"] == r["lp_code"]) & ((em["ann"] - r["ann"]).abs() <= pd.Timedelta(days=5))]
        if not len(e) or not (e["cash_ps"] > 0).any():
            n_miss += 1
            continue
        n_match += 1
        v = e.loc[(e["ann"] - r["ann"]).abs().idxmin(), "cash_ps"]
        n_close += int(abs(v - r["cash"]) / r["cash"] <= 0.01)
    tot = len(sv)
    rows.append(qa_row("D3b", "em_missing_rate", n_miss / max(tot, 1), "<= 0.03", n_miss / max(tot, 1) <= 0.03,
                       f"抽样 40 只幸存股 2016-2024 baostock 现金预案 {tot} 条,EM 缺失 {n_miss}"))
    rows.append(qa_row("D3b", "within_1pct_share", n_close / max(n_match, 1), ">= 0.95", n_close / max(n_match, 1) >= 0.95,
                       f"两源都有 {n_match} 条,差 ≤ 1% {n_close} 条"))
    done = (FETCH / "bs_div_done.txt").read_text().split("\n")
    dn = [x for x in done if x.startswith("del|")]
    bsdel_ok = len(dn)
    import re
    log_txt = (ROOT / "logs" / "inc_t5_fetch.log").read_text()
    n_fail = len(re.findall(r"\[bs_div\] FAIL del\|", log_txt))
    rows.append(qa_row("D3c", "delisted_request_success", 1 - n_fail / max(bsdel_ok + n_fail, 1), ">= 0.95",
                       1 - n_fail / max(bsdel_ok + n_fail, 1) >= 0.95, f"退市股请求成功 {bsdel_ok} 失败 {n_fail}"))
    # ---- D7 PIT 可见日合理性(只报告)----
    r = rep[rep["pit_vintage"] == FU.PIT_FINAL]
    dl = pd.to_datetime(r["end_date"].dt.year.astype(str) + r["end_date"].dt.month.map(
        {3: "-04-30", 6: "-08-31", 9: "-10-31", 12: "-04-30"}))
    dl = dl + pd.to_timedelta((r["end_date"].dt.month == 12).astype(int) * 365, unit="D")
    late = ((r["avail"] - dl).dt.days > 30)
    rows.append(qa_row("D7", "avail_later_than_deadline_30d_share", float(late[r["avail"].notna()].mean()), "report", True,
                       "最终快照行里 avail 晚于法定截止日 30 天以上的比例(2015Q1 起)"))
    # ---- D4 B-LH 与 B-MIN 实现一致性 ----
    mapn = [("LOWVOL", "LOWVOL120"), ("REV", "REV20"), ("VALUE", "VALUE"), ("QUAL", "QUAL"), ("DIV", "DY365")]
    pm_all = None
    from afactor import strategy as ST
    from afactor.panel import Panel
    p = Panel()
    Um = ST.universe(p, "V1000").to_numpy(bool)
    cm = codes.get_indexer(p.codes)
    ok_c = cm >= 0
    dm = p.dates
    for nm_min, nm_lh in mapn:
        a_min = pd.read_parquet(SC / f"min_{nm_min}.parquet")
        sp = []
        for ix in _month_ends(dm, "2021-01-04", "2025-12-31"):
            t = dm[ix]
            ilh = int(dates.get_loc(t))
            a = np.asarray(ld(nm_lh)[ilh])[cm[ok_c]]
            b = a_min.iloc[ix].to_numpy()[ok_c]
            m = pool[ilh][cm[ok_c]] & Um[ix][ok_c] & np.isfinite(a) & np.isfinite(b)
            if nm_min == "DIV":
                m = pool[ilh][cm[ok_c]] & Um[ix][ok_c] & np.isfinite(a) & np.isfinite(b)
            if m.sum() > 200:
                sp.append(spearmanr(a[m], b[m]).correlation)
        thr = 0.95 if nm_min == "REV" else 0.90
        v = float(np.median(sp)) if sp else np.nan
        rows.append(qa_row("D4", f"{nm_min}_LH_vs_MIN_spearman_median", v, f">= {thr}", bool(v >= thr),
                           f"2021-2025 月末 {len(sp)} 截面,共同股票 ∩ 两个池;min {np.min(sp):.3f}"))
    df = pd.DataFrame(rows)
    df.to_csv(QA, index=False)
    log(f"[qa] {len(df)} 行 {time.time() - t0:.0f}s;未通过: " + "; ".join(df[~df['ok'].astype(bool)]['item'].tolist()))


if __name__ == "__main__":
    for c in sys.argv[1:]:
        globals()[f"stage_{c}"]()
