"""T5 多风格篮子的打分构造(前缀 inc_t5;预注册 research/INC_T5_PREREG.md §4,2026-10-11)。

篮子(信号方向:值高 = 好,全部只用 t 日收盘及以前的信息):
  LOWVOL  -std(收盘对收盘日收益,过去 W 个交易行,停牌/复牌日 NaN 不计,最少 60 个观测),W = 120(主)/ 60 / 250
  REV     -(close_px_t / close_px_{t-k} - 1),k = 20(主)/ 60,两端必须是有效成交日
  VALUE   0.5*cs_rank(BP) + 0.5*cs_rank(EP);BP = 每股净资产 x hfq_e+ / close_px_t;EP = ROE_ttm x BP
  DIV     过去 365 自然日内预案公告的每股税前现金分红 x hfq_a / close_px_t(无预案 = 0)
  QUAL    mean[cs_rank(ROE_ttm), cs_rank(-sigma(ROE_q, 8 期, 最少 6 期)), cs_rank(-Lev)]

通用的"每股量换算":复权价 adj 与不复权价 raw 之比 f = adj/raw 把某一日的每股量换算到"当日股本基准"之外的另一日:
    每股量_t(以 t 日股本为基准) = 每股量_x * f_x / adj_t        (x = 报告期末 e+ 或公告日 a)
LongPanel:adj = close_px(后复权),raw = close_raw;Panel:adj = close_px(前复权),raw = newdata.raw_close。
后复权因子含现金分红,换算有每年 1~2% 的近似误差(预注册 §14.2)。

可见日:财报用项目 PIT 表的生效可见日(fundamental._pit_daily 的 avail / avail_ttm),分红用预案公告日。
cs_rank 是决策池内(pool & 有限值)的百分位排名,逐日横截面。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TD = 242


# ================================================================ 通用小工具
def cs_rank_pool(x: np.ndarray, pool: np.ndarray) -> np.ndarray:
    """逐行百分位排名(0,1],只在 pool & 有限值 内排;其余 NaN。并列取平均名次。"""
    a = np.where(pool & np.isfinite(x), x, np.nan).astype(np.float64)
    r = pd.DataFrame(a).rank(axis=1, pct=True, method="average").to_numpy()
    return r.astype(np.float32)


def adj_ratio_ffill(adj: np.ndarray, raw: np.ndarray) -> np.ndarray:
    """f = adj/raw,仅在两者都有限且 raw>0 的行有值;再沿时间前向填充(报告期末/公告日可能是非交易日)。"""
    with np.errstate(all="ignore"):
        f = np.where(np.isfinite(adj) & np.isfinite(raw) & (raw > 0), adj / raw, np.nan)
    return pd.DataFrame(f).ffill().to_numpy()


def asof_row(dates: pd.DatetimeIndex, ts: pd.Series | np.ndarray, side: str = "right") -> np.ndarray:
    """每个时间戳对应的面板行号:side='right' -> 最后一个 <= ts 的交易日;'left' -> 第一个 >= ts 的交易日。
    不在面板范围内给 -1(right 时 ts 早于首日)或 len(dates)(left 时 ts 晚于末日)。"""
    v = pd.DatetimeIndex(ts).values
    i = np.searchsorted(dates.values, v, side=side)
    return (i - 1) if side == "right" else i


def f_at(fff: np.ndarray, rows: np.ndarray, cols: np.ndarray, fallback_first: np.ndarray | None = None) -> np.ndarray:
    """取 f 前向填充面板在 (rows, cols) 的值;rows<0(早于面板首日)或值缺失时,
    用 fallback_first[cols](该股第一个有效 f,新股报告期早于上市日)。"""
    T = fff.shape[0]
    out = np.full(len(rows), np.nan)
    ok = (rows >= 0) & (rows < T)
    out[ok] = fff[rows[ok], cols[ok]]
    if fallback_first is not None:
        bad = ~np.isfinite(out)
        out[bad] = fallback_first[cols[bad]]
    return out


def first_valid_f(f_raw: np.ndarray) -> np.ndarray:
    """每只股票第一个有效 f(新股的报告期末早于上市日时,用上市首日的复权比作股本基准,预注册偏离 inc_t5_deviations.csv)。"""
    fin = np.isfinite(f_raw)
    idx = np.argmax(fin, axis=0)
    out = f_raw[idx, np.arange(f_raw.shape[1])]
    out[~fin.any(axis=0)] = np.nan
    return out


# ================================================================ 价格类篮子
def lowvol(close: np.ndarray, valid: np.ndarray, window: int = 120, min_obs: int = 60) -> np.ndarray:
    """-std(日收益):收益 r_t = close_t/close_{t-1}-1,仅当 t 与 t-1 两行都是有效成交日;窗口 = 过去 window 行,
    NaN 不计,有效观测 < min_obs 给 NaN。"""
    with np.errstate(all="ignore"):
        r = np.full(close.shape, np.nan)
        ok = valid[1:] & valid[:-1] & np.isfinite(close[1:]) & np.isfinite(close[:-1]) & (close[:-1] > 0)
        r[1:] = np.where(ok, close[1:] / close[:-1] - 1.0, np.nan)
    sd = pd.DataFrame(r).rolling(window, min_periods=min_obs).std().to_numpy()
    return (-sd).astype(np.float32)


def reversal(close: np.ndarray, valid: np.ndarray, k: int = 20) -> np.ndarray:
    """-(close_t/close_{t-k}-1),t 与 t-k 两行都必须是有效成交日。"""
    out = np.full(close.shape, np.nan)
    with np.errstate(all="ignore"):
        ok = valid[k:] & valid[:-k] & np.isfinite(close[k:]) & np.isfinite(close[:-k]) & (close[:-k] > 0)
        out[k:] = np.where(ok, -(close[k:] / close[:-k] - 1.0), np.nan)
    return out.astype(np.float32)


# ================================================================ 财报类
def report_row_cols(rep: pd.DataFrame, codes: pd.Index, dates: pd.DatetimeIndex, f_raw: np.ndarray) -> pd.DataFrame:
    """在 PIT 报告期表(含 as-of 快照行)上加行级衍生列(每行用自己的股本/权益/净利,可见日沿用行的 avail*):

      f_e         报告期末 e+(<= e 的最后一个有效交易日)的复权比(早于上市时用该股首个有效 f)
      bps_h       每股净资产 x f_e  = 归母权益/总股本 x f_e                     (avail)
      roe_ttm     归母净利 TTM / 归母权益(权益>0)                                (avail_ttm)
      lev         总负债/总资产                                                  (avail)
    权益<=0 或股本<=0 -> NaN。返回副本(只含需要的列)。"""
    cols = ["stock_code", "end_date", "avail", "avail_ttm", "avail_q", "pit_vintage", "quarter", "qi",
            "total_share", "total_hldr_eqy_exc_min_int", "n_income_attr_p_ttm", "n_income_attr_p_q",
            "total_liab", "total_assets", "comp_type"]
    d = rep[[c for c in cols if c in rep.columns]].copy()
    cj = codes.get_indexer(d["stock_code"])
    d = d[cj >= 0].copy()
    cj = cj[cj >= 0]
    rows = asof_row(dates, d["end_date"], "right")
    fff = pd.DataFrame(f_raw).ffill().to_numpy()
    first = first_valid_f(f_raw)
    d["f_e"] = f_at(fff, rows, cj, first)
    E = d["total_hldr_eqy_exc_min_int"].to_numpy(np.float64)
    S = d["total_share"].to_numpy(np.float64)
    NI = d["n_income_attr_p_ttm"].to_numpy(np.float64)
    with np.errstate(all="ignore"):
        d["bps_h"] = np.where((E > 0) & (S > 0), E / S * d["f_e"].to_numpy(), np.nan)
        d["roe_ttm"] = np.where(E > 0, NI / E, np.nan)
        d["lev"] = d["total_liab"] / d["total_assets"].where(d["total_assets"] > 0)
    return d


def roe_sigma_rows(rep_final: pd.DataFrame, window: int = 8, min_periods: int = 6) -> pd.DataFrame:
    """单季 ROE_q = 单季归母净利 / 该期末归母权益(权益>0)的滚动标准差(连续季度网格上 window 期,最少 min_periods 期)。
    只用最终快照(每个报告期的最新版;利润表每期本就只有一版,资产负债表多版本只影响分母,极少)。
    可见日 = 窗口内参与计算的各期 avail_q 的最大值。返回列:stock_code, end_date, roe_sig_ttm, avail_ttm。"""
    d = rep_final[["stock_code", "qi", "end_date", "n_income_attr_p_q", "total_hldr_eqy_exc_min_int", "avail_q"]].copy()
    d = d[d["avail_q"].notna()]
    E = d["total_hldr_eqy_exc_min_int"]
    d["roe_q"] = d["n_income_attr_p_q"] / E.where(E > 0)
    d["av"] = d["avail_q"].astype("int64").astype(np.float64)
    d.loc[d["roe_q"].isna(), "av"] = np.nan
    out = []
    for code, g in d.sort_values(["stock_code", "qi"]).groupby("stock_code", sort=False):
        g = g.set_index("qi")
        full = np.arange(int(g.index.min()), int(g.index.max()) + 1)
        g = g.reindex(full)
        sd = g["roe_q"].rolling(window, min_periods=min_periods).std()
        av = g["av"].rolling(window, min_periods=1).max()
        keep = (sd.notna() & g["end_date"].notna()).to_numpy()
        if keep.any():
            out.append(pd.DataFrame({"stock_code": code, "end_date": g["end_date"].to_numpy()[keep],
                                     "roe_sig_ttm": sd.to_numpy()[keep], "av": av.to_numpy()[keep]}))
    if not out:
        return pd.DataFrame(columns=["stock_code", "end_date", "roe_sig_ttm", "avail_ttm"])
    o = pd.concat(out, ignore_index=True)
    o["avail_ttm"] = pd.to_datetime(o["av"].astype("int64"), unit="ns")
    return o[["stock_code", "end_date", "roe_sig_ttm", "avail_ttm"]]


def fin_mask(codes: pd.Index, comp_type: pd.Series, industry: pd.Series | None) -> np.ndarray:
    """金融股:comp_type ∈ {2,3,4};comp_type 缺失者按证监会行业 J66-J69;再缺按非金融。"""
    ct = comp_type.reindex(codes)
    isfin = ct.astype(str).isin(["2", "3", "4", "2.0", "3.0", "4.0"]).to_numpy()
    if industry is not None:
        ind = industry.reindex(codes)
        miss = ct.isna().to_numpy()
        isfin |= miss & ind.fillna("").astype(str).str.startswith(("J66", "J67", "J68", "J69")).to_numpy()
    return isfin


# ================================================================ 分红
def load_dividend_events(em_dir, bs_csv, code6_to_lp: dict, lp_code_ok: set, cutoff: str = "2025-12-31"
                         ) -> tuple[pd.DataFrame, dict]:
    """合并 EM stock_fhps_em(分报告期 parquet)与 baostock(退市股 + 抽检幸存股)的现金分红预案事件。
    事件 = (lp_code, ann[预案公告日], cash_ps[每股税前现金], impl[是否实施分配], src)。
    ann > cutoff 的行读入后立即丢弃。EM 现金分红比例是"每 10 股派息(税前)"。"""
    from pathlib import Path
    stats: dict = {}
    parts = []
    em_dir = Path(em_dir)
    nrow = 0
    for f in sorted(em_dir.glob("*.parquet")):
        d = pd.read_parquet(f)
        if not len(d):
            continue
        nrow += len(d)
        d = d.assign(period=f.stem)
        parts.append(d)
    if parts:
        em = pd.concat(parts, ignore_index=True)
        em["ann"] = pd.to_datetime(em["预案公告日"], errors="coerce")
        stats["em_rows"] = len(em)
        stats["em_ann_missing"] = int(em["ann"].isna().sum())
        em = em[em["ann"].notna() & (em["ann"] <= pd.Timestamp(cutoff))]
        per10 = pd.to_numeric(em["现金分红-现金分红比例"], errors="coerce")
        em = pd.DataFrame({"code6": em["代码"].astype(str).str.zfill(6), "ann": em["ann"],
                           "cash_ps": (per10 / 10.0).fillna(0.0), "cash_na": per10.isna(),
                           "impl": em["方案进度"].astype(str).eq("实施分配"), "period": em["period"], "src": "em"})
        em["lp_code"] = em["code6"].map(code6_to_lp)
        stats["em_unmapped"] = int(em["lp_code"].isna().sum())
        em = em[em["lp_code"].notna()]
        em = em.drop_duplicates(["lp_code", "ann", "cash_ps", "period"])
    else:
        em = pd.DataFrame(columns=["lp_code", "ann", "cash_ps", "impl", "src"])
    bs = pd.DataFrame(columns=["lp_code", "ann", "cash_ps", "impl", "src", "kind"])
    if bs_csv is not None and Path(bs_csv).exists():
        b = pd.read_csv(bs_csv, dtype=str)
        b["ann"] = pd.to_datetime(b["dividPlanAnnounceDate"], errors="coerce")
        stats["bs_rows"] = len(b)
        stats["bs_ann_missing"] = int(b["ann"].isna().sum())
        b = b[b["ann"].notna() & (b["ann"] <= pd.Timestamp(cutoff))]
        cash = pd.to_numeric(b["dividCashPsBeforeTax"], errors="coerce")
        bs = pd.DataFrame({"lp_code": b["lp_code"], "ann": b["ann"], "cash_ps": cash.fillna(0.0),
                           "impl": b["dividOperateDate"].fillna("").astype(str).str.len().gt(0),
                           "src": "bs", "kind": b["kind"]})
        bs = bs.drop_duplicates(["lp_code", "ann", "cash_ps"])
    ev = pd.concat([em.assign(kind="em_all")[["lp_code", "ann", "cash_ps", "impl", "src", "kind"]],
                    bs[["lp_code", "ann", "cash_ps", "impl", "src", "kind"]]], ignore_index=True)
    # 退市股用 bs(kind=del);幸存股抽检样本(kind=surv)只用于 D3(b) 核对,不进入打分
    ev = ev[ev["kind"] != "surv"]
    ev = ev[ev["lp_code"].isin(lp_code_ok)]
    stats["events_total"] = len(ev)
    stats["events_cash_pos"] = int((ev["cash_ps"] > 0).sum())
    return ev, stats


def dy_numerator(events: pd.DataFrame, dates: pd.DatetimeIndex, codes: pd.Index, f_raw: np.ndarray,
                 window_days: int = 365, impl_only: bool = False) -> np.ndarray:
    """过去 window_days 个自然日内预案公告的每股税前现金分红 x f_a 的滚动和 [T x N];事件从公告日所在(或之后第一个)
    交易日起生效,持续 window_days 个自然日(公告日 a 生效区间 [a, a + window_days))。"""
    T, N = f_raw.shape
    ev = events[events["cash_ps"] > 0]
    if impl_only:
        ev = ev[ev["impl"]]
    cj = codes.get_indexer(ev["lp_code"])
    m = cj >= 0
    ev, cj = ev[m], cj[m]
    a = pd.DatetimeIndex(ev["ann"])
    ia = asof_row(dates, a, "left")                       # 生效起点:第一个 >= a 的交易日
    ib = asof_row(dates, a + pd.Timedelta(days=window_days), "left")
    ir = asof_row(dates, a, "right")                      # 换算 f_a:最后一个 <= a 的交易日
    fff = pd.DataFrame(f_raw).ffill().to_numpy()
    first = first_valid_f(f_raw)
    fa = f_at(fff, ir, cj, first)
    val = ev["cash_ps"].to_numpy(np.float64) * fa
    ok = np.isfinite(val) & (ia < T)
    D = np.zeros((T + 1, N), np.float64)
    np.add.at(D, (ia[ok], cj[ok]), val[ok])
    ibb = np.minimum(ib[ok], T)
    np.add.at(D, (ibb, cj[ok]), -val[ok])
    return np.cumsum(D, axis=0)[:T].astype(np.float32)
