"""前向检验数据质检(前缀 fwd_data)。**不计算任何收益率、因子、IC 或回测。**

读 cache/forward/raw/ 下 scripts/fwd_data_fetch_bs.py 与 scripts/fwd_data_fetch_ak.py 的下载结果,
输出:
  research/ledger/fwd_data_qa.csv               全部质检数字(section, metric, value, note)
  research/ledger/fwd_data_manifest.json        下载清单(目录、行数、日期范围、失败清单、取数日期)
  research/ledger/fwd_data_m5_coverage_by_date.csv  逐日覆盖:5 分钟线股票数 / bar 数 / 日线交易状态
  research/ledger/fwd_data_overlap_by_stock.csv 重叠段 08-03~08-17 逐股:收盘价比值、成交额比值
  research/ledger/fwd_data_exrights.csv         除权事件清单(复权因子 × 日线 preclose 交叉核对)
  research/ledger/fwd_data_fin_mapping.csv      东方财富字段 ↔ akshare 中文列 ↔ 项目字段 映射与数值核对
  research/ledger/fwd_data_minute_warmup.csv    2026 年 1 分钟原始文件结构问题(逐月;预热输入用)
  (另含 §10 新浪北交所补充数据的覆盖与跨源对照,数字同在 fwd_data_qa.csv)
  cache/forward/m5_long.parquet                 5 分钟线合并长表(fwd_data.load_m5 的缓存)

纪律:
  - 重叠段只做"同一时刻两份原始数据"的对照(价格比值、成交额比值),不是收益;
  - 2026-01-01 ~ 2026-08-30 的数据只做结构检查(行数、时间戳、覆盖),不算任何收益/IC/超额;
  - 不读 cache/sealed、不用 Panel(sealed=True)。Panel() 只取 .codes(股票清单)。

用法:python3 scripts/fwd_data_qa.py   (约 5~10 分钟;读 1 分钟数据 8 个月 close_adj + 202608 三个字段)
"""
from __future__ import annotations
import os
import sys
import json
import time
import glob

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
import pandas as pd

from afactor import fwd_data as FD
from afactor.config import LEDGER, load_raw_minute, ROOT

DATA821 = ROOT / "data_8:21"
QA: list[dict] = []
T0 = time.time()


def put(section: str, metric: str, value, note: str = ""):
    if isinstance(value, (np.floating, float)):
        value = float(value)
    elif isinstance(value, (np.integer,)):
        value = int(value)
    QA.append(dict(section=section, metric=metric, value=value, note=note))


def log(msg: str):
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)


def qtl(s: pd.Series, qs=(0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 0.999)) -> dict:
    s = pd.Series(s).dropna()
    if s.empty:
        return {}
    return {f"q{q * 100:g}": float(s.quantile(q)) for q in qs}


# =============================================================================
# 0. 股票清单
# =============================================================================
def sec_universe():
    S = "0_universe"
    u = pd.read_csv(FD.DIRS["meta"] / "universe.csv", dtype=str)
    basic = pd.read_csv(FD.DIRS["meta"] / "query_stock_basic_20261010.csv", dtype=str)
    basic["code_p"] = basic["code"].map(lambda c: FD.from_bs(c) if c[:2] in ("sh", "sz") else c)
    b = basic.set_index("code_p")
    put(S, "n_panel_codes", int((u.in_panel == "1").sum()), "Panel().codes = cache/daily(2020-2025)")
    put(S, "n_panel_bj", int(((u.in_panel == "1") & u.code.str.endswith(".BJ")).sum()),
        "北交所:baostock 不支持(10004011),未请求")
    put(S, "n_bs_list_a_share", int((u.in_bs_list == "1").sum()), "query_all_stock(2026-10-09) 沪深 A 股")
    put(S, "n_union", len(u))
    put(S, "n_requested", int((u.covered == "1").sum()), "并集去掉 BJ")
    nb = u[(u.in_panel == "1") & (u.in_bs_list == "0") & (u.covered == "1")]
    st = nb.code.map(lambda c: "outDate=" + str(b.at[c, "outDate"]) if c in b.index and
                     isinstance(b.at[c, "outDate"], str) and b.at[c, "outDate"] else
                     ("status=" + str(b.at[c, "status"]) if c in b.index else "not_in_stock_basic"))
    put(S, "n_panel_not_in_bs_list", len(nb), "面板有、2026-10-09 清单没有(退市/更名);明细见 manifest")
    new = u[(u.in_panel == "0")]
    ipo = new.code.map(lambda c: b.at[c, "ipoDate"] if c in b.index else None)
    put(S, "n_bs_only", len(new), "2026-10-09 清单有、面板没有(主要是 2026 年新股)")
    put(S, "n_bs_only_ipo_2026", int((pd.to_datetime(ipo, errors="coerce") >= "2026-01-01").sum()))
    # 幸存者偏差检查:5 分钟窗口内(08-03~10-09)任何一天在市的沪深 A 股是否都在请求清单里
    a_sh_sz = basic[(basic["type"] == "1") & basic["code"].map(FD.is_a_share_bs)]
    alive = a_sh_sz[(a_sh_sz["ipoDate"] <= FD.M5_END) &
                    (a_sh_sz["outDate"].fillna("").eq("") | (a_sh_sz["outDate"].fillna("9999") >= FD.M5_START))]
    req = set(u.loc[u.covered == "1", "code"])
    miss_alive = sorted(set(alive["code_p"]) - req)
    put(S, "n_alive_in_m5_window", len(alive), "query_stock_basic:08-03~10-09 期间任一天在市的沪深 A 股")
    put(S, "n_alive_not_requested", len(miss_alive), "应为 0;非 0 即窗口内存在漏抓(幸存者偏差)")
    out_2026 = basic[(basic["type"] == "1") & basic["outDate"].fillna("").str.startswith("2026")]
    put(S, "n_stock_basic_delisted_2026", len(out_2026), "query_stock_basic outDate 在 2026 年(type=1 股票)")
    return dict(alive_not_requested=miss_alive,
                panel_not_in_bs_list=dict(zip(nb.code, st)),
                bs_only=dict(zip(new.code, ipo.astype(str))),
                delisted_2026=dict(zip(out_2026.code_p, out_2026.outDate)))


# =============================================================================
# 1. 下载状态
# =============================================================================
def sec_status():
    S = "1_download"
    out = {}
    for kind in ("m5", "daily", "adjf", "profit"):
        st = FD.read_status(kind)
        parts = list((FD.DIRS[kind] / "parts").glob("*.csv*"))
        parts = [p for p in parts if not p.name.endswith(".tmp") and ".redo" not in p.name]
        done = {p.name.split(".csv")[0] for p in parts}
        fails = st[st.status.str.startswith("fail") & ~st.code.isin(done)]
        n_ok = int(st[st.code.isin(done)].status.eq("ok").sum())
        n_empty = int(st[st.code.isin(done)].status.eq("empty").sum())
        put(S, f"{kind}_parts", len(parts))
        put(S, f"{kind}_ok", n_ok)
        put(S, f"{kind}_empty", n_empty, "空结果:整段停牌/未上市/窗口内无除权/未披露")
        put(S, f"{kind}_final_fail", len(fails))
        put(S, f"{kind}_retried", int((st.attempts.astype(int) > 1).sum()), "至少重试一次的股票数")
        out[kind] = dict(dir=str(FD.DIRS[kind].relative_to(ROOT)), parts=len(parts), ok=n_ok, empty=n_empty,
                         final_fail=fails[["code", "status"]].to_dict("records"),
                         empty_codes=sorted(st[st.code.isin(done) & st.status.eq("empty")].code.tolist()))
    return out


# =============================================================================
# 2. 1 分钟原始文件(2026 年,预热输入)的结构问题
# =============================================================================
def sec_minute_warmup(daily: pd.DataFrame | None = None):
    """逐月结构检查;07/08 两个月另用 baostock 日线(tradestatus=1 且有成交)核对 1 分钟整日缺失。"""
    S = "2_minute_2026_structure"
    rows = []
    tr = None
    if daily is not None:
        dd = daily[(daily.tradestatus == 1) & (daily.volume > 0)]
        tr = set(zip(dd.code, dd.date))
    for m in ["202601", "202602", "202603", "202604", "202605", "202606", "202607", "202608"]:
        w = load_raw_minute("close_adj", m)
        a = load_raw_minute("amt", m)
        vc = w.index.normalize().value_counts()
        wc, dg = FD.clean_minute_month(w)
        ac, dga = FD.clean_minute_month(a)
        has = wc.notna().groupby(wc.index.normalize()).any()
        per_day = has.sum(axis=1)
        gap = np.nan
        gap_stocks = np.nan
        if tr is not None and m in ("202607", "202608"):
            st = has.stack()
            have = set(st[st].index)
            days = set(has.index)
            colset = set(wc.columns)
            exp = {(c, d) for c, d in tr if d in days and c in colset}
            miss = exp - {(c, d) for d, c in have}
            gap = len(miss)
            gap_stocks = len({c for c, _ in miss})
            not_in_file = {(c, d) for c, d in tr if d in days and c not in colset}
            put(S, f"{m}_trading_stockdays_missing_in_1min", gap,
                f"baostock 日线有成交、1 分钟整日缺失({gap_stocks} 只);另有 {len(not_in_file)} 个股票日的代码不在 1 分钟文件里")
            mbd = pd.Series([d for _, d in miss]).value_counts().sort_index()
            put(S, f"{m}_missing_in_1min_by_date", json.dumps({str(k.date()): int(v) for k, v in mbd.items()}))
        rows.append(dict(month=m, n_cols=w.shape[1], n_rows=len(w), n_days=len(vc),
                         missing_trading_stockdays=gap, missing_trading_stocks=gap_stocks,
                         days_not_241=int((vc != 241).sum()),
                         utc_rows=dg["utc_rows"], utc_days=";".join(dg["utc_days"]),
                         utc_conflict_cells_close=dg["conflict_cells"], utc_conflict_cells_amt=dga["conflict_cells"],
                         missing_slots=json.dumps(dg["missing_slots"]),
                         stocks_any_data=int(wc.notna().any().sum()),
                         stocks_per_day_min=int(per_day.min()), stocks_per_day_median=float(per_day.median()),
                         stocks_per_day_max=int(per_day.max())))
        low = per_day[per_day < 0.97 * per_day.median()]
        if len(low):
            put(S, f"{m}_low_coverage_days", json.dumps({str(k.date()): int(v) for k, v in low.items()}),
                "当天有数据的股票数 < 当月中位数的 97%")
        log(f"  1min {m}: {rows[-1]['n_days']} 天 utc {dg['utc_rows']} 缺格 {len(dg['missing_slots'])}")
    df = pd.DataFrame(rows)
    df.to_csv(LEDGER / "fwd_data_minute_warmup.csv", index=False)
    put(S, "months_with_utc_rows", ";".join(df.loc[df.utc_rows > 0, "month"]), "当天部分股票用 UTC 时间戳(01:30..07:00)")
    put(S, "months_with_missing_slots", ";".join(df.loc[df.missing_slots != "{}", "month"]))
    for r in rows:
        put(S, f"{r['month']}_stocks_per_day_median", r["stocks_per_day_median"], "fwd_data_minute_warmup.csv")
    return df


def minute_month_holes():
    """整月缺失:上月与下月都有数据、本月一行都没有的股票数。"""
    S = "2_minute_2026_structure"
    ms = ["202601", "202602", "202603", "202604", "202605", "202606", "202607", "202608"]
    has = {m: load_raw_minute("close_adj", m).notna().any() for m in ms}
    for i in range(1, len(ms) - 1):
        prev = set(has[ms[i - 1]][has[ms[i - 1]]].index)
        nxt = set(has[ms[i + 1]][has[ms[i + 1]]].index)
        cur = set(has[ms[i]][has[ms[i]]].index)
        hole = sorted((prev & nxt) - cur)
        put(S, f"{ms[i]}_stocks_missing_whole_month", len(hole),
            "上月与下月都有数据、本月整月无数据;例:" + ",".join(hole[:5]))


def utc_version_check(m5w: dict) -> dict:
    """2026-08-10 / 08-11:同一股票既有 UTC 行又有上海时间行。两版聚合到 5 分钟后**只在两版不一致的格子上**
    与 baostock 对照,看哪版对(两版一致的格子不提供信息)。"""
    S = "4_overlap_utc"
    res = {}
    for field, how, key in (("close_adj", "last", "close"), ("amt", "sum", "amount")):
        w = load_raw_minute(field, "202608")
        t = w.index
        hh = t.hour * 100 + t.minute
        um = (hh >= 130) & (hh <= 700)
        utc = w[um].copy()
        utc.index = utc.index + pd.Timedelta(hours=8)
        days = sorted(set(utc.index.normalize()))
        cst = w[w.index.normalize().isin(days) & ~um]
        both_codes = utc.columns[utc.notna().any() & cst.notna().any()]
        put(S, f"n_stocks_both_versions_{key}", len(both_codes),
            "两版都有值的股票:" + (f"{both_codes[0]}..{both_codes[-1]}" if len(both_codes) else ""))
        grid = pd.DatetimeIndex([d + o for d in days for o in FD.minute_offsets()])
        au = FD.minute_to_bs5(utc.reindex(index=grid, columns=both_codes), how)
        ac = FD.minute_to_bs5(cst.reindex(index=grid, columns=both_codes), how)
        b = m5w[key].reindex(index=au.index, columns=both_codes)
        tol = (lambda x, y: (x - y).abs() <= 0.005) if key == "close" else \
              (lambda x, y: (x - y).abs() <= 1e-6 * y.abs().clip(lower=1))
        ok = au.notna() & ac.notna() & b.notna()
        diff = ok & ~tol(au, ac)
        n = int(diff.values.sum())
        put(S, f"{key}_5min_cells_versions_differ", n, f"两版聚合后不一致、且 baostock 有值的格子(共 {int(ok.values.sum())} 格)")
        for name, v in (("utc", au), ("cst", ac)):
            r = float(tol(v, b).astype(float).where(diff).stack().mean()) if n else np.nan
            res[f"{key}_{name}"] = r
            put(S, f"{key}_match_rate_on_diff_cells_{name}_version", r)
    return res


# =============================================================================
# 3. 5 分钟线结构
# =============================================================================
def sec_m5_structure(m5: pd.DataFrame, daily: pd.DataFrame, cal: list):
    S = "3_m5_structure"
    put(S, "rows", len(m5))
    put(S, "stocks", int(m5.code.nunique()))
    cnt = m5.groupby(["code", "date"], observed=True).size()
    put(S, "stock_days", len(cnt))
    put(S, "stock_days_48_bars_share", float((cnt == 48).mean()), "每个(股票,日)bar 数 == 48 的比例")
    vc = cnt.value_counts().sort_index()
    put(S, "bars_per_day_value_counts", json.dumps({int(k): int(v) for k, v in vc.items()}))
    bad_time = ~m5.time.astype(str).isin(FD.BS5_TIMES)
    put(S, "rows_time_not_in_48_grid", int(bad_time.sum()))
    put(S, "dup_code_dt", int(m5.duplicated(["code", "dt"]).sum()))
    dates = sorted(m5.date.unique())
    exp = [pd.Timestamp(d) for d in cal if FD.M5_START <= d <= FD.M5_END]
    put(S, "trade_dates_expected", len(exp), "baostock query_trade_dates 2026-08-03~10-09")
    put(S, "trade_dates_present", len(dates))
    put(S, "trade_dates_missing", ";".join(str(d.date()) for d in sorted(set(exp) - set(pd.DatetimeIndex(dates)))))
    put(S, "trade_dates_extra", ";".join(str(pd.Timestamp(d).date()) for d in sorted(set(pd.DatetimeIndex(dates)) - set(exp))))
    px = m5[["open", "high", "low", "close"]]
    nonpos = (px <= 0).any(axis=1)
    put(S, "rows_nonpositive_price", int(nonpos.sum()),
        "全部是 volume=0 的 bar:baostock 对开盘后尚无成交 / 停牌日给 0 价,下游须当 NaN")
    put(S, "rows_nonpositive_price_with_volume", int((nonpos & (m5.volume > 0)).sum()))
    put(S, "stock_days_all_bars_zero_price", int(nonpos.groupby([m5.code, m5.date], observed=True).sum().eq(48).sum()))
    put(S, "rows_nan_price", int(px.isna().any(axis=1).sum()))
    viol = (m5.high < m5[["open", "close"]].max(axis=1) - 1e-9) | (m5.low > m5[["open", "close"]].min(axis=1) + 1e-9)
    put(S, "rows_ohlc_violation", int(viol.sum()), "high < max(open,close) 或 low > min(open,close)")
    put(S, "bars_zero_volume_share", float((m5.volume == 0).mean()))
    put(S, "rows_amount_zero_volume_pos", int(((m5.amount == 0) & (m5.volume > 0)).sum()))
    put(S, "rows_amount_pos_volume_zero", int(((m5.amount > 0) & (m5.volume == 0)).sum()))
    dz = m5.groupby(["code", "date"], observed=True).volume.sum()
    put(S, "stock_days_all_zero_volume", int((dz == 0).sum()), "有 48 根 bar 但全天成交量 0(停牌日是否也给 bar)")

    # 与日线交易状态对照
    dd = daily[(daily.date >= FD.M5_START) & (daily.date <= FD.M5_END)].set_index(["code", "date"])
    m = pd.DataFrame({"n_bars": cnt, "vol5": dz})
    j = dd[["tradestatus", "volume", "amount"]].join(m, how="outer")
    trading = (j.tradestatus == 1) & (j.volume > 0)
    put(S, "daily_trading_stock_days", int(trading.sum()), "日线 tradestatus=1 且 volume>0")
    put(S, "daily_trading_but_no_m5", int((trading & j.n_bars.isna()).sum()))
    put(S, "daily_suspended_stock_days", int((j.tradestatus == 0).sum()))
    put(S, "daily_suspended_but_m5_bars", int(((j.tradestatus == 0) & j.n_bars.notna()).sum()))
    put(S, "m5_bars_but_no_daily_row", int((j.n_bars.notna() & j.tradestatus.isna()).sum()))
    miss = j[trading & j.n_bars.isna()].reset_index()[["code", "date"]]
    # 逐日覆盖
    byd = pd.DataFrame({
        "m5_stocks": cnt.groupby(level="date").size(),
        "m5_stocks_48": (cnt == 48).groupby(level="date").sum(),
        "m5_stocks_zero_volume_day": (dz == 0).groupby(level="date").sum(),
        "daily_rows": dd.groupby(level="date").size(),
        "daily_trading": trading[trading].groupby(level="date").size() if trading.any() else 0,
        "daily_suspended": (dd.tradestatus == 0).groupby(level="date").sum(),
    }).fillna(0).astype(int)
    byd.index.name = "date"
    byd.to_csv(LEDGER / "fwd_data_m5_coverage_by_date.csv")
    put(S, "m5_stocks_per_day_min", int(byd.m5_stocks.min()), "fwd_data_m5_coverage_by_date.csv")
    put(S, "m5_stocks_per_day_max", int(byd.m5_stocks.max()))
    return dict(daily_trading_but_no_m5=miss.assign(date=miss.date.astype(str)).to_dict("records")[:200],
                n_daily_trading_but_no_m5=len(miss))


# =============================================================================
# 4. 重叠段 2026-08-03 ~ 08-17:baostock 5 分钟 vs 1 分钟聚合
# =============================================================================
def sec_overlap(m5: pd.DataFrame, daily: pd.DataFrame, adjf: pd.DataFrame):
    S = "4_overlap"
    ov = m5[(m5.date >= FD.OVERLAP_START) & (m5.date <= FD.OVERLAP_END)]
    codes = sorted(set(ov.code.astype(str)))
    W = {}
    for f in ("close_adj", "open_adj", "amt"):
        w, dg = FD.clean_minute_month(load_raw_minute(f, "202608"))
        w = w[(w.index >= FD.OVERLAP_START) & (w.index < pd.Timestamp(FD.OVERLAP_END) + pd.Timedelta(days=1))]
        W[f] = w
        put(S, f"minute_{f}_utc_rows_fixed", dg["utc_rows"], ";".join(dg["utc_days"]))
        put(S, f"minute_{f}_utc_conflict_cells", dg["conflict_cells"], "同格两版都有值且不等(取上海时间版)")
    min_codes = set(W["close_adj"].columns)
    put(S, "stocks_m5", len(codes))
    put(S, "stocks_minute", int(W["close_adj"].notna().any().sum()))
    common = sorted(set(codes) & min_codes)
    put(S, "stocks_common", len(common))
    put(S, "stocks_m5_not_in_minute_file", len(set(codes) - min_codes), "1 分钟文件没有这一列(2026 新股等)")
    mc = FD.minute_to_bs5(W["close_adj"].reindex(columns=common), "last")
    mo = FD.minute_to_bs5(W["open_adj"].reindex(columns=common), "first")
    ma = FD.minute_to_bs5(W["amt"].reindex(columns=common), "sum")
    piv = {k: ov.pivot(index="dt", columns="code", values=k).reindex(index=mc.index, columns=common)
           for k in ("close", "open", "amount", "volume")}
    bc, bo, ba = piv["close"], piv["open"], piv["amount"]
    both = bc.notna() & mc.notna()
    nb = int(both.values.sum())
    put(S, "cells_both", nb, "(5 分钟 bar, 股票) 两边都有")
    put(S, "cells_m5_only", int((bc.notna() & mc.isna()).values.sum()))
    put(S, "cells_minute_only", int((bc.isna() & mc.notna()).values.sum()))
    sd_m5_only = (bc.notna() & mc.isna()).groupby(bc.index.normalize()).all() & bc.notna().groupby(bc.index.normalize()).any()
    put(S, "stock_days_m5_only_wholeday", int(sd_m5_only.values.sum()),
        "baostock 有整天 bar、1 分钟数据整天缺失(1 分钟原始文件的缺口)")
    put(S, "stocks_with_m5_only_days", int(sd_m5_only.any().sum()))

    # ---- 收盘价 ----
    ratio = (bc / mc).where(both)
    exact = ((bc - mc).abs() <= 0.005).astype(float).where(both)
    put(S, "close_exact_share_all", float(exact.stack().mean()), "|bs - 1min| <= 半个 tick(0.005 元)")
    rd = ratio.groupby(ratio.index.normalize()).median()            # 逐股逐日中位比值
    # 除权股按**事件**认定(复权因子 dividOperateDate 落在 (08-03, 08-17]),不按比值认定:
    # 深市逐根有 1 个 tick 量级的噪声,按"中位比值偏离 1e-4"认定会把 30 来只无事件的深市股误判为除权股。
    ev = adjf.groupby("code").dividOperateDate.apply(list).to_dict() if len(adjf) else {}
    ov_lo, ov_hi = pd.Timestamp(FD.OVERLAP_START), pd.Timestamp(FD.OVERLAP_END)
    exr_codes = [c for c in rd.columns if any(ov_lo < d <= ov_hi for d in ev.get(c, []))]
    exr_stock = pd.Series(rd.columns.isin(exr_codes), index=rd.columns)
    dev_day = (rd - 1).abs() > 1e-4
    put(S, "stocks_exr_event_in_overlap", int(exr_stock.sum()), "复权因子事件在 08-04~08-17 的股票")
    keep = exr_stock[~exr_stock].index
    big = ((rd[keep] - 1).abs() > 2e-3).any()
    put(S, "non_exr_stocks_dayratio_dev_gt_2e-3", int(big.sum()),
        "无除权事件却有某日中位价格比值偏离 1 超过 0.2%(应为 0;说明没有漏记的除权)")
    put(S, "non_exr_stocks_dayratio_dev_gt_1e-4", int(dev_day[keep].any().sum()),
        "偏离 1e-4~2e-3 的是深市逐根 1 tick 噪声,不是除权")
    put(S, "close_exact_share_non_exr", float(exact[keep].stack().mean()), "剔除重叠段内有除权事件的股票后")
    r_k = ratio[keep].stack()
    put(S, "close_ratio_non_exr_quantiles", json.dumps(qtl(r_k)))
    put(S, "close_cells_mismatch_non_exr", int((exact[keep].stack() == 0).sum()))
    cm = (exact[keep] == 0)
    by_bar = cm.groupby(cm.index.strftime("%H:%M")).sum().sum(axis=1)
    put(S, "close_mismatch_non_exr_by_bar_top", json.dumps(by_bar[by_bar > 0].sort_values(ascending=False).head(6).to_dict()))

    # ---- 除权股:事件前比值 = prev_close/preclose,事件当天起比值 = 1 ----
    dly = daily.sort_values(["code", "date"]).copy()
    dly["prev_close"] = dly.groupby("code").close.shift(1)
    dly["adj_ratio"] = dly.prev_close / dly.preclose               # 除权调整比例(不是收益)
    dl = dly.set_index(["code", "date"])
    exr_rows = []
    for c in exr_codes:
        s_ = rd[c].dropna()
        e = max(d for d in ev[c] if ov_lo < d <= ov_hi)             # 重叠段内最后一次除权
        pre, post = s_[s_.index < e], s_[s_.index >= e]
        dr = dl.adj_ratio.get((c, e), np.nan)
        # 重叠段内多次除权时,事件前比值是多次的乘积;这里只核对最后一次之后 = 1、最后一次之前 / 最后一次比例
        exr_rows.append(dict(code=c, event_date=str(e.date()), n_events_in_overlap=sum(ov_lo < d <= ov_hi for d in ev[c]),
                             pre_days=len(pre), post_days=len(post),
                             pre_ratio_median=float(pre.median()) if len(pre) else np.nan,
                             post_ratio_median=float(post.median()) if len(post) else np.nan,
                             daily_adj_ratio=dr,
                             pre_vs_daily=float(pre.median() / dr) if len(pre) and np.isfinite(dr) else np.nan))
    exr_df = pd.DataFrame(exr_rows)
    if len(exr_df):
        single = exr_df[exr_df.n_events_in_overlap == 1]
        pv = (single.pre_vs_daily - 1).abs().dropna()
        put(S, "exr_pre_ratio_vs_daily_adj_ratio_n", len(pv))
        put(S, "exr_pre_ratio_vs_daily_adj_ratio_within_1e-3", float((pv < 1e-3).mean()) if len(pv) else np.nan,
            "事件前 bs/1min 价格比值 / 日线 prev_close/preclose,应为 1(1 分钟前复权、baostock 不复权)")
        pp = (exr_df.post_ratio_median - 1).abs().dropna()
        put(S, "exr_post_ratio_within_1e-3", float((pp < 1e-3).mean()) if len(pp) else np.nan,
            "事件当天起比值回到 1")
        put(S, "exr_small_events_ratio_eq_1", int(((single.daily_adj_ratio - 1).abs() < 1e-9).sum()),
            "复权因子有事件但日线 preclose = 前收(分红小于 1 个 tick)")
    # 08-18 ~ 08-31 除权的股票:1 分钟价格是否已按它复权(决定前复权基准日)
    late = {c: [d for d in v if pd.Timestamp("2026-08-18") <= d <= pd.Timestamp("2026-08-31")] for c, v in ev.items()}
    late = {c: v for c, v in late.items() if v and c in rd.columns and c not in exr_codes}
    if late:
        lr = rd[list(late)].median()
        put(S, "stocks_exr_0818_0831", len(late), "复权因子事件在 08-18~08-31(重叠段内无事件)")
        put(S, "stocks_exr_0818_0831_minute_adjusted", int(((lr - 1).abs() > 2e-3).sum()),
            "这些股票重叠段比值偏离 1 超过 0.2% 的个数(=0 说明 1 分钟前复权基准不晚于 08-17)")
        put(S, "stocks_exr_0818_0831_ratio_median_absdev_max", float((lr - 1).abs().max()))

    # ---- 成交额 ----
    okA = both & (ma > 0)
    ar = (ba / ma).where(okA).stack()
    put(S, "amount_cells", len(ar))
    put(S, "amount_ratio_quantiles", json.dumps(qtl(ar)))
    put(S, "amount_exact_share_1e-6", float(((ar - 1).abs() < 1e-6).mean()))
    put(S, "amount_within_1e-4_share", float(((ar - 1).abs() < 1e-4).mean()))
    res = (ba - ma).where(both)
    mis = res.abs() > 1e-4 * ma.abs().clip(lower=1)
    pair = mis & (((res + res.shift(-1)).abs() <= 1.0) | ((res + res.shift(1)).abs() <= 1.0))
    put(S, "amount_mismatch_cells", int(mis.values.sum()))
    put(S, "amount_mismatch_explained_by_adjacent_bar_shift", int(pair.values.sum()),
        "相邻两根 bar 的差额正负抵消(成交被划到相邻 bar,两根之和一致)")
    byb = mis.groupby(mis.index.strftime("%H:%M")).sum().sum(axis=1)
    put(S, "amount_mismatch_by_bar_top", json.dumps(byb[byb > 0].sort_values(ascending=False).head(8).to_dict()))
    first = mis.index.strftime("%H:%M") == "09:35"
    put(S, "amount_0935_cells", int(okA[first].values.sum()))
    put(S, "amount_0935_exact_share", float(1 - mis[first].values.sum() / max(okA[first].values.sum(), 1)),
        "09:35 bar = 1 分钟 idx0..5(含开盘集合竞价)")
    last = mis.index.strftime("%H:%M") == "15:00"
    put(S, "amount_1500_exact_share", float(1 - mis[last].values.sum() / max(okA[last].values.sum(), 1)),
        "15:00 bar = idx236..240(含收盘集合竞价)")
    dsum_b = ba.where(both).groupby(ba.index.normalize()).sum(min_count=1)
    dsum_m = ma.where(both).groupby(ma.index.normalize()).sum(min_count=1)
    drat = (dsum_b / dsum_m).where(dsum_m > 0).stack()
    put(S, "amount_daysum_ratio_quantiles", json.dumps(qtl(drat)))
    put(S, "amount_daysum_within_1e-6_share", float(((drat - 1).abs() < 1e-6).mean()))
    z = both & (ma == 0)
    put(S, "amount_minute_zero_cells", int(z.values.sum()))
    put(S, "amount_minute_zero_bs_zero", int((z & (ba == 0)).values.sum()))

    # ---- 分交易所(2026-10-10 发现:深市逐根一致率远低于沪市,日合计仍一致) ----
    sd = pd.Series(bc.index.normalize(), index=bc.index)
    for ex in ("SH", "SZ"):
        cs = [c for c in keep if c.endswith("." + ex)]
        if not cs:
            continue
        b_ = both[cs]
        put(S, f"{ex}_stocks_non_exr", len(cs))
        dif = (bc[cs] - mc[cs]).abs().where(b_).stack()
        put(S, f"{ex}_close_exact_share", float((dif <= 0.005).mean()), "非除权股,|差| <= 0.005")
        put(S, f"{ex}_close_within_1tick_share", float((dif <= 0.015).mean()), "|差| <= 1 个 tick(0.01)")
        okx = b_ & (ma[cs] > 0)
        rx = (ba[cs] / ma[cs]).where(okx)
        rr = (rx - 1).abs().stack()
        put(S, f"{ex}_amount_exact_share_1e-6", float((rr < 1e-6).mean()))
        put(S, f"{ex}_amount_abs_relerr_quantiles", json.dumps(qtl(rr, (0.5, 0.75, 0.9, 0.95, 0.99))),
            "逐根 |bs/1min - 1|")
        mx = mis[cs] & b_
        px = pair[cs] & b_
        put(S, f"{ex}_amount_mismatch_adjacent_shift_share", float(px.values.sum() / max(mx.values.sum(), 1)))
        ds_b = ba[cs].where(b_).groupby(sd).sum(min_count=1)
        ds_m = ma[cs].where(b_).groupby(sd).sum(min_count=1)
        dr = ((ds_b / ds_m).where(ds_m > 0) - 1).abs().stack()
        put(S, f"{ex}_amount_daysum_within_1e-6_share", float((dr < 1e-6).mean()))
        for t in ("09:35", "09:40", "15:00"):
            sel = rx.index.strftime("%H:%M") == t
            rt = (rx[sel] - 1).abs().stack()
            put(S, f"{ex}_amount_{t.replace(':', '')}_exact_share", float((rt < 1e-6).mean()))
            put(S, f"{ex}_amount_{t.replace(':', '')}_abs_relerr_q50_q90",
                json.dumps([float(rt.quantile(0.5)), float(rt.quantile(0.9))]))
        byb_x = mx.groupby(mx.index.strftime("%H:%M")).sum().sum(axis=1)
        put(S, f"{ex}_amount_mismatch_by_bar_top", json.dumps(byb_x[byb_x > 0].sort_values(ascending=False).head(6).to_dict()))
        if ex == "SZ":
            # 平移检验:把 1 分钟数据在日内整体平移 ±1 分钟再聚合,看深市差异是不是固定的时间标签错位
            sdm = W["close_adj"].index.normalize()
            for sh_ in (1, -1):
                wc_ = W["close_adj"][cs].groupby(sdm).shift(sh_)
                wa_ = W["amt"][cs].groupby(sdm).shift(sh_)
                mc_ = FD.minute_to_bs5(wc_, "last")
                ma_ = FD.minute_to_bs5(wa_, "sum")
                ok_ = both[cs] & mc_.notna()
                ce_ = float(((bc[cs] - mc_).abs() <= 0.005).astype(float).where(ok_).stack().mean())
                okb_ = ok_ & (ma_ > 0)
                ae_ = float((((ba[cs] / ma_) - 1).abs() < 1e-6).astype(float).where(okb_).stack().mean())
                put(S, f"SZ_shift{sh_:+d}min_close_exact_share", ce_, "1 分钟数据日内平移后再聚合")
                put(S, f"SZ_shift{sh_:+d}min_amount_exact_share", ae_)
        cbd = (dif <= 0.005).groupby(level=0).mean()
        cbd = cbd.groupby(cbd.index.normalize()).mean()
        put(S, f"{ex}_close_exact_by_date", json.dumps({str(k.date()): round(float(v), 4) for k, v in cbd.items()}))

    # ---- 开盘价(baostock 的 open 口径) ----
    ex_k = ((bo - mo).abs() <= 0.005).where(both)[keep].stack()
    put(S, "open_exact_share_non_exr", float(ex_k.mean()), "bs open vs 组内第一分钟 open")
    prev_bc = bc.shift(1)
    same_day = pd.Series(bc.index.normalize(), index=bc.index)
    prev_ok = same_day.eq(same_day.shift(1)).to_numpy()
    eqp = ((bo - prev_bc).abs() <= 0.005).where(both)[keep]
    eqp = eqp[prev_ok].stack()
    put(S, "open_equals_prev_bs_close_share_non_exr", float(eqp.mean()), "日内第 2..48 根")
    mis_o = (((bo - mo).abs() > 0.005) & both)[keep].to_numpy() & prev_ok[:, None]
    eqp2 = ((bo - prev_bc).abs() <= 0.005)[keep].to_numpy()
    put(S, "open_mismatch_cells_non_exr", int(mis_o.sum()), "日内第 2..48 根")
    put(S, "open_mismatch_but_equals_prev_close_share", float(eqp2[mis_o].mean()) if mis_o.any() else np.nan,
        "open 不一致的格子里,bs open = 上一根 bs close 的比例")

    # ---- 逐股表 ----
    per = pd.DataFrame({
        "exchange": [c.split(".")[1] for c in common],
        "cells_both": both[common].sum(),
        "close_exact_share": exact.mean(),
        "close_ratio_median": ratio.median(),
        "close_ratio_min": ratio.min(),
        "close_ratio_max": ratio.max(),
        "close_ratio_not1_days": dev_day.sum(),
        "amount_ratio_median": (ba / ma).where(okA).median(),
        "amount_mismatch_cells": mis.sum(),
        "amount_mismatch_shift_cells": pair.sum(),
        "m5_only_wholedays": sd_m5_only.sum(),
    })
    per.index.name = "code"
    if len(exr_df):
        per = per.join(exr_df.set_index("code"), how="left")
    per.to_csv(LEDGER / "fwd_data_overlap_by_stock.csv")
    return dict(close=bc, amount=ba), exr_df


# =============================================================================
# 5. 除权事件 + 复权因子覆盖
# =============================================================================
def sec_exrights(daily: pd.DataFrame, adjf: pd.DataFrame, univ_codes: list):
    S = "5_exrights"
    put(S, "adjf_events_2026", len(adjf), "dividOperateDate in 2026-01-01~10-09")
    put(S, "adjf_stocks_with_events", int(adjf.code.nunique()) if len(adjf) else 0)
    if len(adjf):
        bym = adjf.dividOperateDate.dt.strftime("%Y-%m").value_counts().sort_index()
        put(S, "adjf_events_by_month", json.dumps(bym.to_dict()))
    d = daily.sort_values(["code", "date"]).copy()
    d["prev_close"] = d.groupby("code").close.shift(1)
    d["prev_date"] = d.groupby("code").date.shift(1)
    cand = d[d.prev_close.notna() & (d.preclose > 0)]
    dev = (cand.preclose - cand.prev_close).abs() > 0.005
    dev_ev = cand[dev].copy()
    dev_ev["adj_ratio_daily"] = dev_ev.prev_close / dev_ev.preclose
    put(S, "daily_preclose_mismatch_events", len(dev_ev), "日线 preclose ≠ 前一行 close(07-02~10-09)")
    A = adjf[(adjf.dividOperateDate > pd.Timestamp(FD.DAILY_START)) &
             (adjf.dividOperateDate <= pd.Timestamp(FD.DAILY_END))].copy() if len(adjf) else adjf
    ka = set(zip(A.code, A.dividOperateDate)) if len(A) else set()
    kd = set(zip(dev_ev.code, dev_ev.date))
    put(S, "adjf_events_in_daily_window", len(ka))
    put(S, "matched_adjf_and_daily", len(ka & kd))
    put(S, "adjf_only", len(ka - kd), "有复权因子事件,但日线 preclose = 前收(如分红极小 / 停牌中除权)")
    put(S, "daily_only", len(kd - ka), "日线 preclose ≠ 前收,但无复权因子事件")
    # 复权因子比值 vs 日线比值(同一股票 2026 年内有前一事件时可核对)
    rows = []
    if len(adjf):
        a2 = adjf.sort_values(["code", "dividOperateDate"]).copy()
        a2["prev_back"] = a2.groupby("code").backAdjustFactor.shift(1)
        a2["prev_fore"] = a2.groupby("code").foreAdjustFactor.shift(1)
        a2["ratio_back"] = a2.backAdjustFactor / a2.prev_back
        a2["ratio_fore"] = a2.foreAdjustFactor / a2.prev_fore
        m = a2.merge(dev_ev[["code", "date", "prev_date", "prev_close", "preclose", "adj_ratio_daily"]],
                     left_on=["code", "dividOperateDate"], right_on=["code", "date"], how="outer")
        m["event_date"] = m.dividOperateDate.fillna(m.date)
        chk = m.dropna(subset=["ratio_back", "adj_ratio_daily"])
        if len(chk):
            dv = (chk.ratio_back / chk.adj_ratio_daily - 1).abs()
            put(S, "factor_vs_daily_ratio_checked", len(chk))
            put(S, "factor_vs_daily_ratio_maxdev", float(dv.max()))
            put(S, "factor_vs_daily_ratio_within_1e-3", float((dv < 1e-3).mean()))
        m["source"] = np.where(m.dividOperateDate.notna() & m.date.notna(), "both",
                               np.where(m.dividOperateDate.notna(), "adjf_only", "daily_only"))
        m.loc[(m.source == "adjf_only") & (m.event_date <= pd.Timestamp(FD.DAILY_START)), "source"] = "adjf_before_daily_window"
        m["in_overlap"] = (m.event_date > pd.Timestamp(FD.OVERLAP_START)) & (m.event_date <= pd.Timestamp(FD.OVERLAP_END))
        m["in_m5_after_overlap"] = (m.event_date > pd.Timestamp(FD.OVERLAP_END)) & (m.event_date <= pd.Timestamp(FD.M5_END))
        m["in_forward_window"] = m.event_date >= pd.Timestamp(FD.FWD_START)
        cols = ["code", "event_date", "source", "foreAdjustFactor", "backAdjustFactor", "adjustFactor",
                "ratio_back", "ratio_fore", "prev_date", "prev_close", "preclose", "adj_ratio_daily",
                "in_overlap", "in_m5_after_overlap", "in_forward_window"]
        out = m[cols].sort_values(["event_date", "code"])
        out.to_csv(LEDGER / "fwd_data_exrights.csv", index=False)
        put(S, "events_in_m5_after_overlap", int(out.in_m5_after_overlap.sum()),
            "08-18~10-09 除权:拼接 1 分钟(前复权)与 5 分钟(不复权)时必须处理")
        put(S, "events_in_forward_window", int(out.in_forward_window.sum()), ">= 2026-09-01")
        put(S, "events_in_overlap", int(out.in_overlap.sum()))
    # 复权因子覆盖:请求成功(含空)的比例
    st = FD.read_status("adjf")
    done = st[st.status.isin(["ok", "empty"])]
    done_codes = {FD.from_bs(c) for c in done.code}
    put(S, "adjf_coverage_share", float(len(done_codes & set(univ_codes)) / max(len(univ_codes), 1)),
        "请求成功(含窗口内无除权)的股票 / 请求股票数")
    # 与本地 data_8:21/adj_factor.csv(到 2026-08-21)的除权日对照
    try:
        it = pd.read_csv(DATA821 / "adj_factor.csv", dtype={"ts_code": str, "trade_date": str}, chunksize=2_000_000)
        loc = pd.concat([c[c.trade_date >= "20251225"] for c in it], ignore_index=True)
        loc["date"] = pd.to_datetime(loc.trade_date)
        loc = loc.sort_values(["ts_code", "date"])
        loc["chg"] = loc.groupby("ts_code").adj_factor.diff().abs() > 1e-9
        le = loc[loc.chg & (loc.date >= "2026-01-01")]
        kl = set(zip(le.ts_code, le.date))
        last_loc = loc.date.max()
        ka2 = {(c, dte) for c, dte in zip(adjf.code, adjf.dividOperateDate) if dte <= last_loc} if len(adjf) else set()
        ka2 = {k for k in ka2 if k[0] in set(loc.ts_code)}
        put(S, "local_adjfactor_last_date", str(last_loc.date()))
        put(S, "local_events_2026", len(kl), "data_8:21/adj_factor.csv 因子变动日")
        put(S, "bs_events_same_range_same_stocks", len(ka2))
        put(S, "local_vs_bs_matched", len(kl & ka2))
        put(S, "local_only", len(kl - ka2))
        put(S, "bs_only", len(ka2 - kl))
        lo = sorted(kl - ka2)[:30]
        put(S, "local_only_examples", json.dumps([(c, str(x.date())) for c, x in lo]))
    except Exception as e:                                  # noqa: BLE001
        put(S, "local_adjfactor_compare", f"failed: {e}")


# =============================================================================
# 6. 日线 vs 5 分钟线
# =============================================================================
def sec_daily_vs_m5(m5: pd.DataFrame, daily: pd.DataFrame):
    S = "6_daily_vs_m5"
    g = m5.groupby(["code", "date"], observed=True)
    agg = pd.DataFrame({"amt5": g.amount.sum(), "vol5": g.volume.sum(), "hi5": g.high.max(), "lo5": g.low.min(),
                        "n": g.size()})
    first = m5[m5.time.astype(str) == "09:35"].set_index(["code", "date"])
    last = m5[m5.time.astype(str) == "15:00"].set_index(["code", "date"])
    agg["open5"] = first.open
    agg["close5"] = last.close
    dd = daily.set_index(["code", "date"])
    j = agg.join(dd[["open", "high", "low", "close", "volume", "amount", "tradestatus"]], how="inner")
    j = j[(j.tradestatus == 1) & (j.amount > 0)]
    put(S, "stock_days_compared", len(j))
    ra = j.amt5 / j.amount
    put(S, "amount_ratio_quantiles", json.dumps(qtl(ra)))
    put(S, "amount_within_1e-6_share", float(((ra - 1).abs() < 1e-6).mean()))
    put(S, "amount_within_1e-3_share", float(((ra - 1).abs() < 1e-3).mean()))
    put(S, "amount_within_1e-2_share", float(((ra - 1).abs() < 1e-2).mean()))
    rv = j.vol5 / j.volume
    put(S, "volume_within_1e-6_share", float(((rv - 1).abs() < 1e-6).mean()))
    put(S, "close_equal_share", float(((j.close5 - j.close).abs() <= 0.005).mean()), "日线 close vs 15:00 bar close")
    put(S, "open_equal_share", float(((j.open5 - j.open).abs() <= 0.005).mean()), "日线 open vs 09:35 bar open")
    put(S, "high_equal_share", float(((j.hi5 - j.high).abs() <= 0.005).mean()))
    put(S, "high_m5_below_daily_share", float((j.hi5 < j.high - 0.005).mean()), "5 分钟 max(high) < 日线 high")
    put(S, "high_m5_above_daily_share", float((j.hi5 > j.high + 0.005).mean()))
    put(S, "low_equal_share", float(((j.lo5 - j.low).abs() <= 0.005).mean()))
    put(S, "low_m5_above_daily_share", float((j.lo5 > j.low + 0.005).mean()), "5 分钟 min(low) > 日线 low")
    put(S, "low_m5_below_daily_share", float((j.lo5 < j.low - 0.005).mean()))
    bad = j[(ra - 1).abs() >= 1e-2]
    put(S, "amount_off_1pct_stock_days", len(bad))
    put(S, "amount_off_1pct_by_date", json.dumps(bad.reset_index().date.dt.strftime("%Y-%m-%d").value_counts().head(10).to_dict()))
    put(S, "amount_off_1pct_bars_lt48", int((bad.n < 48).sum()))


# =============================================================================
# 7. 股本
# =============================================================================
def sec_shares(daily: pd.DataFrame):
    S = "7_shares"
    pr = FD.load_profit()
    if pr.empty:
        put(S, "profit_rows", 0)
        return
    put(S, "profit_rows", len(pr))
    for q, g in pr.groupby("q_req"):
        put(S, f"profit_{q}_stocks", int(g.code.nunique()))
        put(S, f"profit_{q}_pubDate_range", f"{g.pubDate.min().date()}~{g.pubDate.max().date()}")
        put(S, f"profit_{q}_liqaShare_missing", int(g.liqaShare.isna().sum()))
    q2 = pr[pr.q_req == "2026Q2"]
    put(S, "2026Q2_pub_after_0831", int((q2.pubDate > "2026-08-31").sum()), "半年报法定截止 08-31;晚于此多为更正/补披")
    # 9 月 1 日 PIT 可见的最新一期
    vis = pr[pr.pubDate <= "2026-09-01"].sort_values(["code", "statDate", "pubDate"]).groupby("code").tail(1)
    put(S, "stocks_with_shares_visible_0901", int(vis.code.nunique()))
    put(S, "visible_0901_statDate_counts", json.dumps(vis.statDate.dt.strftime("%Y-%m-%d").value_counts().to_dict()))
    # volume / turn 反推流通股本(9 月,tradestatus=1, turn>0)
    d = daily[(daily.date >= "2026-09-01") & (daily.date <= "2026-09-30") & (daily.tradestatus == 1) & (daily.turn > 0)]
    implied = (d.volume / (d.turn / 100.0)).groupby(d.code).median()
    iq = (d.volume / (d.turn / 100.0)).groupby(d.code).agg(lambda s: s.max() / s.min() - 1)
    put(S, "implied_float_stocks_sep", len(implied))
    put(S, "implied_float_within_month_spread_q99", float(iq.quantile(0.99)), "同一股票 9 月内反推值 max/min-1")
    lq = pr[pr.pubDate <= "2026-09-30"].sort_values(["code", "statDate", "pubDate"]).groupby("code").tail(1).set_index("code")
    j = pd.DataFrame({"implied": implied}).join(lq[["liqaShare", "totalShare", "statDate"]], how="inner")
    r = j.implied / j.liqaShare
    put(S, "implied_vs_liqaShare_n", len(r))
    put(S, "implied_vs_liqaShare_quantiles", json.dumps(qtl(r)))
    put(S, "implied_vs_liqaShare_within_1pct", float(((r - 1).abs() < 0.01).mean()))
    put(S, "implied_vs_liqaShare_within_5pct", float(((r - 1).abs() < 0.05).mean()))
    # 与本地 daily_basic(到 2026-08-21)对照
    try:
        it = pd.read_csv(DATA821 / "daily_basic.csv", usecols=["ts_code", "trade_date", "total_share", "float_share"],
                         dtype={"ts_code": str, "trade_date": str}, chunksize=2_000_000)
        db = pd.concat([c[c.trade_date >= "20260801"] for c in it], ignore_index=True)
        lastd = db.trade_date.max()
        dbl = db[db.trade_date == lastd].set_index("ts_code")
        put(S, "daily_basic_last_date", lastd)
        da = daily[(daily.date >= "2026-08-03") & (daily.date <= pd.Timestamp(lastd)) & (daily.tradestatus == 1) &
                   (daily.turn > 0)]
        imp_aug = (da.volume / (da.turn / 100.0)).groupby(da.code).median()
        jj = pd.DataFrame({"imp_aug": imp_aug}).join(dbl[["float_share", "total_share"]] * 1e4, how="inner")
        r2 = jj.imp_aug / jj.float_share
        put(S, "implied_aug_vs_daily_basic_float_n", int(r2.notna().sum()))
        put(S, "implied_aug_vs_daily_basic_float_within_1pct", float(((r2 - 1).abs() < 0.01).mean()))
        put(S, "implied_aug_vs_daily_basic_float_quantiles", json.dumps(qtl(r2)))
        put(S, "daily_basic_total_share_nonnull_rows", int(db.total_share.notna().sum()),
            f"daily_basic 2026-08 共 {len(db)} 行;total_share 为空则无法对照总股本")
        vis821 = pr[pr.pubDate <= pd.Timestamp(lastd)].sort_values(["code", "statDate", "pubDate"]).groupby("code").tail(1)
        j3 = vis821.set_index("code")[["liqaShare"]].join(dbl[["float_share"]] * 1e4, how="inner").dropna()
        r3 = j3.liqaShare / j3.float_share
        put(S, "bs_liqaShare_vs_daily_basic_float_n", len(r3))
        put(S, "bs_liqaShare_vs_daily_basic_float_within_1pct", float(((r3 - 1).abs() < 0.01).mean()) if len(r3) else np.nan,
            f"baostock 截至 {lastd} 可见的最新一期 liqaShare vs daily_basic float_share({lastd})")
    except Exception as e:                                  # noqa: BLE001
        put(S, "daily_basic_compare", f"failed: {e}")


# =============================================================================
# 8. 财报 / 公告字段映射
# =============================================================================
FIN_MAP = [
    # (表, 东方财富原始字段, 项目字段(tushare 风格 pkl), 说明)
    ("lrb", "OPERATE_PROFIT", "operate_profit", "营业利润(PEAD 主信号 fu_op_yoy 的原料)"),
    ("lrb", "PARENT_NETPROFIT", "n_income_attr_p", "归母净利润"),
    ("lrb", "TOTAL_OPERATE_INCOME", "total_revenue", "营业总收入"),
    ("lrb", "OPERATE_INCOME", "revenue", "营业收入:东财只给金融类公司,一般企业为空"),
    ("lrb", "TOTAL_PROFIT", "total_profit", "利润总额"),
    ("lrb", "INCOME_TAX", "income_tax", "所得税"),
    ("lrb", "TOTAL_OPERATE_COST", "total_cogs", "营业总成本:**口径不同**——东财 = 营业成本+税金及附加+销售+管理+财务+研发,"
                                                "tushare 另含减值损失等,数值一般不等"),
    ("lrb", "OPERATE_COST", "oper_cost", "营业成本"),
    ("lrb", "SALE_EXPENSE", "sell_exp", "销售费用"),
    ("lrb", "MANAGE_EXPENSE", "admin_exp", "管理费用"),
    ("lrb", "FINANCE_EXPENSE", "fin_exp", "财务费用"),
    ("lrb", "OPERATE_TAX_ADD", "biz_tax_surchg", "税金及附加"),
    ("lrb", "DEDUCT_PARENT_NETPROFIT", None, "扣非归母净利润:项目利润表 pkl 无此列"),
    ("lrb", None, "basic_eps", "每股收益:东财利润表接口无"),
    ("lrb", None, "n_income", "净利润(含少数股东):东财利润表接口无(只有归母)"),
    ("lrb", None, "rd_exp", "研发费用:东财利润表接口无"),
    ("zcfz", "TOTAL_ASSETS", "total_assets", "总资产"),
    ("zcfz", "TOTAL_LIABILITIES", "total_liab", "总负债"),
    ("zcfz", "TOTAL_EQUITY", "total_hldr_eqy_inc_min_int", "股东权益合计(含少数股东)"),
    ("zcfz", None, "total_hldr_eqy_exc_min_int", "归母权益:东财资产负债表接口无"),
    ("zcfz", "MONETARYFUNDS", "money_cap", "货币资金"),
    ("zcfz", "ACCOUNTS_RECE", "accounts_receiv", "应收账款"),
    ("zcfz", "INVENTORY", "inventories", "存货"),
    ("zcfz", "FIXED_ASSET", "fix_assets_total", "固定资产(合计,含固定资产清理);对 fix_assets(净额)只有约 60% 相等"),
    ("zcfz", "ACCOUNTS_PAYABLE", "acct_payable", "应付账款"),
    ("zcfz", "ADVANCE_RECEIVABLES", "adv_receipts", "预收账款"),
    ("zcfz", "SHORT_LOAN", "st_borr", "短期借款:东财只给少数(金融类)公司"),
    ("zcfz", None, "total_share", "期末总股本:东财资产负债表接口无(用 baostock totalShare)"),
    ("xjll", "NETCASH_OPERATE", "n_cashflow_act", "经营活动现金流量净额"),
    ("xjll", "NETCASH_INVEST", "n_cashflow_inv_act", "投资活动现金流量净额"),
    ("xjll", "NETCASH_FINANCE", "n_cash_flows_fnc_act", "筹资活动现金流量净额"),
    ("xjll", "CCE_ADD", "n_incr_cash_cash_equ", "现金及现金等价物净增加额"),
    ("xjll", "SALES_SERVICES", "c_fr_sale_sg", "销售商品、提供劳务收到的现金"),
    ("xjll", "PAY_STAFF_CASH", "c_paid_to_for_empl", "支付给职工的现金"),
    ("xjll", "CONSTRUCT_LONG_ASSET", "c_pay_acq_const_fiolta", "购建长期资产支付的现金"),
    ("xjll", "BEGIN_CCE", "c_cash_equ_beg_period", "期初现金及现金等价物:东财只给金融类公司"),
    ("xjll", "END_CCE", "c_cash_equ_end_period", "期末现金及现金等价物:东财只给金融类公司"),
    ("xjll", None, "free_cashflow", "自由现金流:东财现金流量表接口无"),
]
PKL = {"lrb": "stock_income_2015_now.pkl", "zcfz": "stock_balancesheet_2015_now.pkl",
       "xjll": "stock_cashflow_2015_now.pkl"}
AK_TO_RAW_CANDIDATES = None


def _latest(pattern: str):
    fs = sorted(glob.glob(str(FD.RAW / "ak_em" / pattern)))
    return fs[-1] if fs else None


def _ak_raw_colmap(ak: pd.DataFrame, raw: pd.DataFrame) -> dict:
    """按数值逐列确认 akshare 中文列 = 原始哪个字段(防 akshare 按位置改名错位)。

    每一列先判定类型(数值 / 日期 / 文本),只在同类型列之间比较;返回 "字段 (一致比例)"。
    """
    import warnings
    warnings.filterwarnings("ignore")
    raw = raw.reset_index(drop=True)
    ak = ak.reset_index(drop=True)
    if len(ak) != len(raw):
        return {c: "rowcount_differs" for c in ak.columns}

    def kind(s: pd.Series):
        n = pd.to_numeric(s, errors="coerce")
        if n.notna().mean() > 0.5 and n.notna().sum() >= s.notna().sum() * 0.9:
            return "num", n
        d = pd.to_datetime(s, errors="coerce", format="mixed")
        if d.notna().mean() > 0.5:
            return "date", d.dt.normalize()
        return "str", s.astype(str).str.replace(" ", "").str.strip()

    rk = {c: kind(raw[c]) for c in raw.columns}
    out = {}
    for c in ak.columns:
        if c in ("报告日期", "序号"):
            continue
        k, a = kind(ak[c])
        best, best_rate = None, 0.0
        for rc, (kk, r) in rk.items():
            if kk != k:
                continue
            if k == "num":
                both = a.notna() & r.notna()
                if both.sum() == 0:
                    continue
                rate = float(((a[both] - r[both]).abs() <= 1e-6 * r[both].abs().clip(lower=1)).mean()) * float(both.mean())
            else:
                rate = float((a == r).mean())
            if rate > best_rate:
                best, best_rate = rc, rate
        out[c] = f"{best} ({best_rate:.4f})" if best else "none"
    return out


def sec_fin_mapping():
    S = "8_fin"
    rows = []
    # akshare 中文列 -> 原始字段
    for name in ("lrb", "zcfz", "xjll", "yjyg", "yjkb"):
        fa, fr = _latest(f"ak_{name}_20260630_*.csv"), _latest(f"raw_{name}_20260630_*.csv")
        if not (fa and fr):
            continue
        ak = pd.read_csv(fa, dtype=str)
        raw = pd.read_csv(fr, dtype=str)
        put(S, f"{name}_20260630_rows", len(raw), os.path.basename(fr))
        put(S, f"{name}_20260630_stocks", int(raw["SECURITY_CODE"].nunique()))
        fa3, fr3 = _latest(f"ak_{name}_20260930_*.csv"), _latest(f"raw_{name}_20260930_*.csv")
        put(S, f"{name}_20260930_rows", len(pd.read_csv(fr3)) if fr3 else 0, "截至取数日(2026-10-10)")
        cm = _ak_raw_colmap(ak, raw)
        for c, v in cm.items():
            rows.append(dict(table=name, kind="akshare_col->raw_field", ak_col=c, raw_field=v.split(" ")[0],
                             project_field="", n_compared=len(raw), agree_share=v.split("(")[-1].rstrip(")") if "(" in v else "",
                             note=""))
        if name == "yjkb" and {"NOTICE_DATE", "UPDATE_DATE", "EITIME"} <= set(raw.columns):
            nd0 = pd.to_datetime(raw["NOTICE_DATE"]).dt.normalize()
            put(S, "yjkb_notice_eq_update_share", float((nd0 == pd.to_datetime(raw["UPDATE_DATE"]).dt.normalize()).mean()),
                "akshare 快报'公告日期'取自 UPDATE_DATE")
            put(S, "yjkb_eitime_before_notice_share", float((pd.to_datetime(raw["EITIME"]).dt.normalize() < nd0).mean()),
                "EITIME(东财录入时刻)早于 NOTICE_DATE:公告多在前一晚发布;PIT 取 NOTICE_DATE 偏保守")
        if "NOTICE_DATE" in raw.columns:
            nd = pd.to_datetime(raw["NOTICE_DATE"], errors="coerce")
            put(S, f"{name}_20260630_notice_by_month", json.dumps(nd.dt.strftime("%Y-%m").value_counts().sort_index().to_dict()))
    # 原始字段 -> 项目字段(tushare pkl):在两边都有同一报告期的股票上核对数值与日期。
    # 项目 pkl 到 2026-08-21,2026-06-30 行很少(资产负债表 0 行),所以同时用 2026-03-31(一季报)核对。
    for tab in ("lrb", "zcfz", "xjll"):
        pk_all = pd.read_pickle(DATA821 / PKL[tab])
        pk_all["end_date"] = pd.to_datetime(pk_all.end_date)
        for rdate in ("20260331", "20260630"):
            fr = _latest(f"raw_{tab}_{rdate}_*.csv")
            if not fr:
                continue
            raw = pd.read_csv(fr, dtype={"SECURITY_CODE": str, "SECUCODE": str}).set_index("SECUCODE")
            pk = pk_all[pk_all.end_date == pd.Timestamp(rdate)].copy()
            pk["ann_date"] = pd.to_datetime(pk.ann_date)
            pk["f_ann_date"] = pd.to_datetime(pk.f_ann_date)
            if "report_type" in pk.columns:
                pk = pk[pk.report_type.astype(str) == "1"]
            pk = pk.sort_values("f_ann_date").drop_duplicates("stock_code", keep="last").set_index("stock_code")
            common = raw.index.intersection(pk.index)
            tag = f"{tab}_{rdate}"
            put(S, f"{tag}_pkl_rows", len(pk), f"data_8:21/{PKL[tab]} end_date={rdate}(pkl 到 2026-08-21)")
            put(S, f"{tag}_em_rows", len(raw), os.path.basename(fr))
            put(S, f"{tag}_common_stocks", len(common))
            if len(common) == 0:
                continue
            nd = pd.to_datetime(raw.loc[common, "NOTICE_DATE"]).dt.normalize()
            put(S, f"{tag}_notice_eq_ann_share", float((nd == pk.loc[common, "ann_date"]).mean()),
                "东财 NOTICE_DATE vs tushare ann_date")
            put(S, f"{tag}_notice_eq_f_ann_share", float((nd == pk.loc[common, "f_ann_date"]).mean()))
            put(S, f"{tag}_notice_later_than_ann", int((nd > pk.loc[common, "ann_date"]).sum()),
                "东财 NOTICE_DATE 晚于首次披露日(更正/补充后覆盖)")
            put(S, f"{tag}_notice_earlier_than_ann", int((nd < pk.loc[common, "ann_date"]).sum()))
            put(S, f"{tag}_f_ann_gt_ann", int((pk.loc[common, "f_ann_date"] > pk.loc[common, "ann_date"]).sum()),
                "tushare 自身标记的修正版本数")
            if "DATA_STATE" in raw.columns:
                put(S, f"{tag}_data_state_counts", json.dumps(raw["DATA_STATE"].astype(str).value_counts().to_dict()))
            for (t, rf, pf, note) in [x for x in FIN_MAP if x[0] == tab]:
                n = agree = None
                if rf and pf and rf in raw.columns and pf in pk.columns:
                    a = pd.to_numeric(raw.loc[common, rf], errors="coerce")
                    b = pd.to_numeric(pk.loc[common, pf], errors="coerce")
                    ok = a.notna() & b.notna()
                    n = int(ok.sum())
                    agree = float(((a[ok] - b[ok]).abs() <= np.maximum(1.0, 1e-6 * b[ok].abs())).mean()) if n else np.nan
                    note2 = f"{note};东财非空 {int(a.notna().sum())} / 项目非空 {int(b.notna().sum())} / 共同 {len(common)}"
                else:
                    note2 = note
                rows.append(dict(table=tab, kind=f"raw_field->project_field@{rdate}", ak_col="",
                                 raw_field=rf or "(缺)", project_field=pf or "(缺)", n_compared=n,
                                 agree_share=agree, note=note2))
    # 业绩预告 / 快报:新 akshare 输出与项目现有原始文件(source_raw/eastmoney_*_raw.csv)列与行的对照
    for name, oldf, key in (("yjyg", "eastmoney_forecast_raw.csv", ["股票代码", "预测指标"]),
                            ("yjkb", "eastmoney_express_raw.csv", ["股票代码"])):
        fa = _latest(f"ak_{name}_20260630_*.csv")
        if not fa:
            continue
        new = pd.read_csv(fa, dtype={"股票代码": str, "报告日期": str})
        old = pd.read_csv(DATA821 / "source_raw" / oldf, dtype={"股票代码": str, "报告日期": str})
        put(S, f"{name}_columns_identical_to_project_raw", bool(list(new.columns) == list(old.columns)),
            f"data_8:21/source_raw/{oldf};可直接追加给 earnings_events.load_forecast/load_express")
        o = old[old["报告日期"] == "20260630"]
        put(S, f"{name}_20260630_old_snapshot_dup_keys", int(o.duplicated(key).sum()), "旧快照里(" + "+".join(key) + ")重复的行数")
        put(S, f"{name}_20260630_rows_old_snapshot", len(o), f"旧快照最晚公告日 {o['公告日期'].max()}")
        put(S, f"{name}_20260630_rows_new", len(new))
        mg = o.merge(new, on=key, how="outer", suffixes=("_old", "_new"), indicator=True)
        put(S, f"{name}_20260630_rows_only_new", int((mg._merge == "right_only").sum()))
        put(S, f"{name}_20260630_rows_only_old", int((mg._merge == "left_only").sum()),
            "旧快照有、新快照消失(东财覆盖或删除)")
        bo = mg[mg._merge == "both"]
        put(S, f"{name}_20260630_notice_date_changed", int((bo["公告日期_old"].astype(str) != bo["公告日期_new"].astype(str)).sum()))
        vcol = "预测数值" if name == "yjyg" else "净利润-净利润"
        va = pd.to_numeric(bo[f"{vcol}_old"], errors="coerce")
        vb = pd.to_numeric(bo[f"{vcol}_new"], errors="coerce")
        put(S, f"{name}_20260630_value_changed", int(((va - vb).abs() > 1e-6 * vb.abs().clip(lower=1)).sum()),
            f"{vcol} 在两次快照间被改写的行数")
    rows.append(dict(table="forecast", kind="project_csv_vs_em", ak_col="", raw_field="PREDICT_AMT_LOWER/UPPER",
                     project_field="net_profit_min/max", n_compared=None, agree_share=None,
                     note="data_8:21/forecast.csv 是 tushare 口径(只含净利润,type/p_change);项目 PEAD 实际读的是 "
                          "source_raw/eastmoney_forecast_raw.csv(东财口径,多指标,预测数值=区间中值),新数据与后者同列"))
    rows.append(dict(table="express", kind="project_csv_vs_em", ak_col="", raw_field="TOTAL_OPERATE_INCOME/PARENT_NETPROFIT",
                     project_field="revenue/n_income", n_compared=None, agree_share=None,
                     note="data_8:21/express.csv 为 tushare 口径(含 operate_profit/total_profit,东财快报接口无营业利润);"
                          "项目 PEAD 读 source_raw/eastmoney_express_raw.csv,新数据与其同列;东财快报另有 EITIME(录入时刻)"))
    pd.DataFrame(rows).to_csv(LEDGER / "fwd_data_fin_mapping.csv", index=False)


# =============================================================================
# 10. 新浪 5 分钟(北交所补充)+ 跨源抽样
# =============================================================================
def _cmp_close_amount(px: pd.DataFrame, amt: pd.DataFrame, ref_px: pd.DataFrame, ref_amt: pd.DataFrame,
                      S: str, tag: str, ratio_exr: bool = True) -> None:
    """同一时刻两份 5 分钟数据的收盘价、成交额对照(不是收益)。ref 为基准(1 分钟聚合或 baostock)。"""
    both = px.notna() & ref_px.notna()
    put(S, f"{tag}_cells_both", int(both.values.sum()))
    if not both.values.any():
        return
    exact = ((px - ref_px).abs() <= 0.005).where(both)
    keep = px.columns
    if ratio_exr:
        rd = (px / ref_px).where(both).groupby(px.index.normalize()).median()
        exr = ((rd - 1).abs() > 2e-3).any()
        put(S, f"{tag}_stocks_ratio_not1", int(exr.sum()),
            "某日中位价格比值偏离 1 超过 0.2% 的股票(前复权 vs 不复权的除权股),剔除后再算一致率")
        keep = exr[~exr].index
    put(S, f"{tag}_close_exact_share", float(exact[keep].stack().mean()), "剔除比值≠1 的股票后,|差| <= 0.005")
    ok = both & (ref_amt > 0)
    r = (amt / ref_amt).where(ok).stack()
    put(S, f"{tag}_amount_ratio_quantiles", json.dumps(qtl(r)))
    put(S, f"{tag}_amount_within_1e-6_share", float(((r - 1).abs() < 1e-6).mean()))
    put(S, f"{tag}_amount_within_1e-3_share", float(((r - 1).abs() < 1e-3).mean()))
    res = (amt - ref_amt).where(both)
    mis = res.abs() > 1e-4 * ref_amt.abs().clip(lower=1)
    pair = mis & (((res + res.shift(-1)).abs() <= 1.0) | ((res + res.shift(1)).abs() <= 1.0))
    put(S, f"{tag}_amount_mismatch_cells", int(mis.values.sum()))
    put(S, f"{tag}_amount_mismatch_adjacent_shift", int(pair.values.sum()))
    ds = amt.where(both).groupby(amt.index.normalize()).sum(min_count=1)
    dr = ref_amt.where(both).groupby(ref_amt.index.normalize()).sum(min_count=1)
    dd = (ds / dr).where(dr > 0).stack()
    put(S, f"{tag}_amount_daysum_ratio_quantiles", json.dumps(qtl(dd)))
    put(S, f"{tag}_amount_daysum_within_1e-3_share", float(((dd - 1).abs() < 1e-3).mean()))


def sec_sina(m5: pd.DataFrame):
    S = "10_sina_bj"
    d0 = FD.DIRS["sina_bj"]
    if not (d0 / "_status.csv").exists():
        put(S, "sina_m5", "not downloaded")
        return {}
    out = {}
    for kind, key in (("m5", "sina_bj"), ("daily", "sina_daily_bj"), ("qfq", "sina_qfq_bj")):
        f = FD.DIRS[key] / "_status.csv"
        if not f.exists():
            continue
        st = pd.read_csv(f, dtype=str).drop_duplicates("code", keep="last")
        roles = pd.read_csv(FD.DIRS[key] / "codes.csv", dtype=str).set_index("code").role
        st["role"] = st.code.map(roles)
        for role, g in st.groupby("role"):
            put(S, f"{kind}_{role}_ok", int((g.status == "ok").sum()))
            put(S, f"{kind}_{role}_empty", int((g.status == "empty").sum()))
            put(S, f"{kind}_{role}_fail", int(g.status.str.startswith("fail").sum()))
        out[kind] = dict(dir=str(FD.DIRS[key].relative_to(ROOT)), n=len(st),
                         fail=st[st.status.str.startswith("fail")][["code", "status"]].to_dict("records"))
    sb = FD.load_sina_m5(None)
    roles = pd.read_csv(d0 / "codes.csv", dtype=str)
    bjc = set(roles.loc[roles.role == "bj", "code"])
    smc = set(roles.loc[roles.role == "shsz_sample", "code"])
    bj = sb[sb.code.isin(bjc)]
    put(S, "bj_rows", len(bj))
    put(S, "bj_stocks", int(bj.code.nunique()))
    put(S, "sina_dt_range", f"{sb.dt.min()}~{sb.dt.max()}", "新浪只给每只股票最近 1970 根;各股最早一天不完整")
    fd = bj.groupby("code").date.min()                       # 各股最早一天(不完整,剔除)
    put(S, "bj_first_date_quantiles", json.dumps({k: str(v.date()) for k, v in
                                                  fd.quantile([0, 0.5, 1.0], interpolation="nearest").items()}))
    put(S, "bj_stocks_first_date_le_0817", int((fd <= pd.Timestamp(FD.OVERLAP_END)).sum()),
        "最早日不晚于 08-17:覆盖整个 08-18~10-09")
    bj2 = bj[bj.date > bj.code.map(fd)]
    cnt = bj2[bj2.date >= FD.FWD_START].groupby(["code", "date"]).size()
    put(S, "bj_stock_days_48_share_fwd", float((cnt == 48).mean()) if len(cnt) else np.nan,
        "09-01 起各股 bar 数 == 48 的比例;新浪对无成交的 5 分钟不出 bar")
    put(S, "bj_bars_per_day_median_fwd", float(cnt.median()) if len(cnt) else np.nan)
    first_full = pd.Timestamp("2026-08-06")
    put(S, "bj_rows_time_not_in_48_grid", int((~bj.time.isin(FD.BS5_TIMES)).sum()))
    put(S, "bj_dup_code_dt", int(bj.duplicated(["code", "dt"]).sum()))
    # 北交所:新浪 vs 1 分钟聚合(重叠段中新浪完整覆盖的日子)
    lo = pd.Timestamp(first_full)
    W = {}
    for f in ("close_adj", "amt"):
        w, _ = FD.clean_minute_month(load_raw_minute(f, "202608"))
        w = w[(w.index >= lo) & (w.index < pd.Timestamp(FD.OVERLAP_END) + pd.Timedelta(days=1))]
        W[f] = w
    for tag, codes in (("bj_vs_1min", sorted(bjc & set(W["close_adj"].columns))),
                       ("sample_sina_vs_1min", sorted(smc & set(W["close_adj"].columns)))):
        if not codes:
            continue
        mc = FD.minute_to_bs5(W["close_adj"].reindex(columns=codes), "last")
        ma = FD.minute_to_bs5(W["amt"].reindex(columns=codes), "sum")
        sub = sb[sb.code.isin(codes)]
        pc = sub.pivot(index="dt", columns="code", values="close").reindex(index=mc.index, columns=codes)
        pa = sub.pivot(index="dt", columns="code", values="amount").reindex(index=mc.index, columns=codes)
        _cmp_close_amount(pc, pa, mc, ma, S, tag)
    # 沪深抽样:新浪 vs baostock(全部共同 bar,不复权对不复权)
    sm = sb[sb.code.isin(smc)]
    bm = m5[m5.code.astype(str).isin(smc)]
    if len(sm) and len(bm):
        idx = sorted(set(sm.dt) & set(bm.dt))
        codes = sorted(smc)
        piv = lambda d, v: d.pivot(index="dt", columns="code", values=v).reindex(index=pd.DatetimeIndex(idx), columns=codes)
        _cmp_close_amount(piv(sm, "close"), piv(sm, "amount"), piv(bm.assign(code=bm.code.astype(str)), "close"),
                          piv(bm.assign(code=bm.code.astype(str)), "amount"), S, "sample_sina_vs_bs", ratio_exr=False)
        vs = piv(sm, "volume") / piv(bm.assign(code=bm.code.astype(str)), "volume")
        vs = vs.where(np.isfinite(vs)).stack()
        put(S, "sample_sina_vs_bs_volume_within_1e-6_share", float(((vs - 1).abs() < 1e-6).mean()))
    # 北交所日线:新浪日线 vs 5 分钟合计
    try:
        bd = FD.load_sina_daily_bj()
        if len(bd):
            put(S, "bj_daily_rows", len(bd))
            put(S, "bj_daily_stocks", int(bd.code.nunique()))
            g = bj.groupby(["code", "date"])
            a5 = pd.DataFrame({"amt5": g.amount.sum(), "c5": g.close.last(), "n": g.size()})
            j = a5.join(bd.set_index(["code", "date"])[["amount", "close"]], how="inner")
            j = j[(j.n == 48) & (j.amount > 0)]
            r = j.amt5 / j.amount
            put(S, "bj_daily_vs_m5_amount_within_1e-3_share", float(((r - 1).abs() < 1e-3).mean()))
            put(S, "bj_daily_vs_m5_amount_ratio_quantiles", json.dumps(qtl(r)))
            put(S, "bj_daily_vs_m5_close_equal_share", float(((j.c5 - j.close).abs() <= 0.005).mean()))
        qf = FD.load_sina_qfq_bj()
        if len(qf):
            e26 = qf[(qf.date >= "2026-01-01")]
            put(S, "bj_qfq_events_2026", len(e26))
            put(S, "bj_qfq_events_m5_after_overlap", int(((e26.date > FD.OVERLAP_END) & (e26.date <= FD.M5_END)).sum()))
    except Exception as e:                                  # noqa: BLE001
        put(S, "bj_daily_qfq", f"failed: {e}")
    return out


# =============================================================================
# 9. 指数与中证1000 成分
# =============================================================================
def sec_index(cal: list, univ: pd.DataFrame, m5_codes: set):
    S = "9_index"
    from afactor import newdata as ND
    cal_win = [pd.Timestamp(d) for d in cal if "2026-07-01" <= d <= FD.M5_END]
    out = {}
    for sym in ("sh000852", "sh000300", "sh000905", "sh000001"):
        fs = sorted(glob.glob(str(FD.RAW / "ak_index" / f"{sym}_fetched*.csv")))
        if not fs:
            put(S, f"{sym}_file", "missing")
            continue
        d = pd.read_csv(fs[-1], parse_dates=["date"]).set_index("date").sort_index()
        put(S, f"{sym}_rows", len(d), os.path.basename(fs[-1]))
        put(S, f"{sym}_range", f"{d.index[0].date()}~{d.index[-1].date()}")
        put(S, f"{sym}_missing_trade_days_0701_1009", len(set(cal_win) - set(d.index)))
        put(S, f"{sym}_extra_days_0701_1009", len(set(d.loc["2026-07-01":FD.M5_END].index) - set(cal_win)))
        put(S, f"{sym}_nan_or_nonpos_close", int((d.close.isna() | (d.close <= 0)).sum()))
        code = sym[2:]
        try:
            ref = ND.index_daily()[code].dropna() if code in ND.INDEX_NAMES else None
        except Exception:                               # noqa: BLE001
            ref = None
        if ref is not None:
            com = d.index.intersection(ref.index)
            rel = (d.close.reindex(com) / ref.reindex(com) - 1).abs()
            put(S, f"{sym}_vs_frozen_snapshot_overlap_days", len(com), f"ETF 项目冻结快照(到 {ref.index.max().date()})")
            put(S, f"{sym}_vs_frozen_snapshot_maxreldev", float(rel.max()))
            put(S, f"{sym}_vs_frozen_snapshot_n_dev_gt_1e-6", int((rel > 1e-6).sum()))
        out[sym] = dict(file=os.path.relpath(fs[-1], ROOT), rows=len(d), first=str(d.index[0].date()),
                        last=str(d.index[-1].date()))
    # 中证1000 成分与权重
    fw = sorted(glob.glob(str(FD.RAW / "ak_csindex" / "000852_cons_weight_fetched*.csv")))
    fc = sorted(glob.glob(str(FD.RAW / "ak_csindex" / "000852_cons_fetched*.csv")))
    if fw and fc:
        w = pd.read_csv(fw[-1], dtype={"成分券代码": str, "指数代码": str})
        c = pd.read_csv(fc[-1], dtype={"成分券代码": str, "指数代码": str})
        put(S, "csi1000_weight_rows", len(w), os.path.basename(fw[-1]))
        put(S, "csi1000_weight_date", ";".join(sorted(w["日期"].astype(str).unique())), "权重日期(中证指数公司月末权重文件)")
        put(S, "csi1000_cons_date", ";".join(sorted(c["日期"].astype(str).unique())), "成分日期")
        put(S, "csi1000_fetch_date", fw[-1].split("fetched")[-1][:8])
        put(S, "csi1000_weight_sum", float(w["权重"].sum()))
        put(S, "csi1000_weight_max", float(w["权重"].max()))
        put(S, "csi1000_sets_equal", bool(set(w["成分券代码"]) == set(c["成分券代码"])),
            "09-30 权重名单 vs 10-09 成分名单")
        put(S, "csi1000_by_exchange", json.dumps(w["交易所"].value_counts().to_dict(), ensure_ascii=False))
        sfx = np.where(w["交易所"].str.contains("上海"), ".SH", np.where(w["交易所"].str.contains("深圳"), ".SZ", ".BJ"))
        codes = w["成分券代码"] + sfx
        put(S, "csi1000_in_panel_codes", int(codes.isin(set(univ.loc[univ.in_panel == "1", "code"])).sum()))
        put(S, "csi1000_with_m5_data", int(codes.isin(m5_codes).sum()))
        put(S, "csi1000_weight_with_m5_data", float(w.loc[codes.isin(m5_codes).to_numpy(), "权重"].sum()))
    snap = pd.read_csv(DATA821 / "index_cons" / "csindex_cons_snapshot_20260821.csv", dtype=str)
    put(S, "data821_index_cons_snapshot_index", ";".join(sorted(set(snap["index_code"] + " " + snap["index_name"]))),
        "data_8:21/index_cons 那份快照不是中证1000")
    return out


# =============================================================================
def main():
    LEDGER.mkdir(parents=True, exist_ok=True)
    cal_df = pd.read_csv(FD.DIRS["meta"] / "trade_dates_20251201_20261031.csv", dtype=str)
    cal = sorted(cal_df.loc[cal_df.is_trading_day == "1", "calendar_date"])
    univ = pd.read_csv(FD.DIRS["meta"] / "universe.csv", dtype=str)
    req_codes = sorted(univ.loc[univ.covered == "1", "code"])
    log("0 股票清单")
    uni_detail = sec_universe()
    log("1 下载状态")
    st = sec_status()
    log("读 5 分钟线 / 日线 / 复权因子")
    m5 = FD.load_m5()
    daily = FD.load_daily()
    adjf = FD.load_adjf()
    log(f"  m5 {m5.shape} daily {daily.shape} adjf {adjf.shape}")
    log("2 1 分钟原始文件结构")
    sec_minute_warmup(daily)
    minute_month_holes()
    log("3 5 分钟线结构")
    m5d = sec_m5_structure(m5, daily, cal)
    log("4 重叠段对照")
    m5w, exr = sec_overlap(m5, daily, adjf)
    utc = utc_version_check(m5w)
    del m5w
    log("5 除权事件")
    sec_exrights(daily, adjf, req_codes)
    log("6 日线 vs 5 分钟")
    sec_daily_vs_m5(m5, daily)
    log("7 股本")
    sec_shares(daily)
    log("8 财报字段映射")
    sec_fin_mapping()
    log("9 指数")
    idx = sec_index(cal, univ, set(m5.code.astype(str).unique()))
    log("10 新浪(北交所补充)")
    sina = sec_sina(m5)

    qa = pd.DataFrame(QA)
    qa.to_csv(LEDGER / "fwd_data_qa.csv", index=False)
    ak_man = []
    for f in sorted(glob.glob(str(FD.RAW / "_ak_manifest_*.json"))):
        ak_man += json.loads(open(f).read())
    ak_fail = []
    for f in sorted(glob.glob(str(FD.RAW / "_ak_failures_*.json"))):
        ak_fail += [dict(x, file=os.path.basename(f)) for x in json.loads(open(f).read())]
    manifest = dict(
        generated_at=pd.Timestamp.now().isoformat(timespec="seconds"),
        purpose="前向检验原始数据(用户 2026-10-10 明确要求下载);只做下载与质检,不含任何收益计算",
        scripts=["scripts/fwd_data_fetch_bs.py", "scripts/fwd_data_fetch_ak.py", "scripts/fwd_data_fetch_sina_bj.py",
                 "scripts/fwd_data_qa.py",
                 "src/afactor/fwd_data.py"],
        root=str(FD.RAW.relative_to(ROOT)),
        windows=dict(m5=[FD.M5_START, FD.M5_END], overlap=[FD.OVERLAP_START, FD.OVERLAP_END],
                     daily=[FD.DAILY_START, FD.DAILY_END], adjf=[FD.ADJF_START, FD.ADJF_END],
                     profit=["2025Q4", "2026Q1", "2026Q2"], forward_eval_start=FD.FWD_START),
        params=dict(baostock_adjustflag="3(不复权)", m5_frequency="5", timeout_s=60, retries=3,
                    m5_workers=4, other_workers=2),
        universe=dict(n_union=len(univ), n_requested=len(req_codes),
                      bj_uncovered=sorted(univ.loc[univ.covered == "0", "code"]),
                      **uni_detail),
        baostock=st,
        akshare=dict(files=ak_man, failures=ak_fail),
        index=idx,
        m5_coverage=m5d,
        utc_version_check=utc,
        sina_supplement=sina,
        meta_files=sorted(os.path.relpath(p, ROOT) for p in FD.DIRS["meta"].glob("*.csv")),
        dir_sizes={d.name: dict(files=sum(1 for _ in d.rglob("*") if _.is_file()),
                                mb=round(sum(_.stat().st_size for _ in d.rglob("*") if _.is_file()) / 1e6, 1))
                   for d in sorted(FD.RAW.iterdir()) if d.is_dir()},
    )
    (LEDGER / "fwd_data_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1, default=str))
    log(f"完成:{len(qa)} 条质检数字 -> research/ledger/fwd_data_qa.csv")


if __name__ == "__main__":
    main()
