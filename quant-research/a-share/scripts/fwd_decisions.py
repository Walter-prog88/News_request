"""前向检验:计算任何收益之前必须定下的三个数据决定 + 原料质检(预注册 §3.3、§4.3、§4.5、§4.8,前缀 fwd)。

**本脚本不计算任何收益、因子、IC**,只读日历、指数成分快照、两源财报数值、换手与股本,按预注册写死的规则做决定,
结果写 research/ledger/fwd_decisions.csv(决定 + 依据数字)与 logs/fwd_decisions.log。

决定一:日历(§3.3)。信号日从 2026-09-01 起,最后一个信号日 = 满足"其后第 2 个交易日 <= 2026-10-09"的最后一个;
        N = 信号日个数,n = N − FWD_SKIP_HEAD(4)。日历取 baostock query_trade_dates(交易所日历,不是行情)。
决定二:中证 1000 用实际成分还是回退 PIT 代理(§4.5)。有效条件四条全满足才用实际成分:
        成分数 ∈ [990, 1010];权重合计 ∈ 100% ± 0.5%;>= 99% 成分能映射到面板代码;权重日 d_S >= 2026-06-15。
决定三:财报拼接还是沿用最后已知(§4.3)。在两源重叠的报告期(2026Q1 全部;半年报中 08-20 以前已由 tushare
        收录的部分)上,operate_profit、revenue、n_income_attr_p 与股东权益四项的数值一致率(相对差 < 1e-4)
        全部 >= 99% 才拼接东方财富快照;否则在 accept_source_gap(FU.REPORT_TABLES) 内沿用最后已知的报告期。
        字段映射(在看数之前写定,取东财**原始英文字段**,akshare 中文列已由 fwd_data 逐列核对):
          operate_profit   <- OPERATE_PROFIT
          revenue          <- OPERATE_INCOME(东财只对金融类给出);缺失时以 TOTAL_OPERATE_INCOME(营业总收入)代替
          n_income_attr_p  <- PARENT_NETPROFIT
          股东权益          <- TOTAL_EQUITY(股东权益合计,含少数股东)对 tushare total_hldr_eqy_inc_min_int
        东财**没有**归母权益(total_hldr_eqy_exc_min_int)与总股本;这两项无论决定如何都只能沿用最后已知值。
另:换手(daily_basic turnover_rate vs baostock turn,07-01~08-17 重叠段)、股本(baostock totalShare vs tushare
total_share,2025Q4/2026Q1)的一致率,只报告(§4.8 第 4 条),不改变任何选定。
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
import pandas as pd

from afactor.config import LEDGER, ROOT, CACHE
from afactor import fwd_data as FD
from afactor import minute5 as M5

RAW = CACHE / "forward" / "raw"
DATA = ROOT / "data_8:21"
OUT = LEDGER / "fwd_decisions.csv"
DATA_END = pd.Timestamp("2026-10-09")
SIG_START = pd.Timestamp("2026-09-01")
REL_TOL = 1e-4
rows: list[dict] = []


def rec(section, item, value, note="", decision=None):
    rows.append(dict(section=section, item=item, value=value, decision=decision, note=note))
    print(f"[{section}] {item} = {value}" + (f"  -> 决定: {decision}" if decision else "") + (f"  ({note})" if note else ""),
          flush=True)


def calendar_decision():
    cal = pd.read_csv(RAW / "bs_meta" / "trade_dates_20251201_20261031.csv")
    td = pd.DatetimeIndex(pd.to_datetime(cal.loc[cal["is_trading_day"] == 1, "calendar_date"])).sort_values()
    td_upto = td[td <= DATA_END]
    sig = []
    for i, d in enumerate(td_upto):
        if d < SIG_START:
            continue
        if i + 2 < len(td_upto):                 # t+2 个交易日 <= 数据末日
            sig.append(d)
    N = len(sig)
    n = N - M5.FWD_SKIP_HEAD
    case = {21: "预计(09-25 休市、10-08 开市)", 20: "10-08 也休市", 22: "09-25 开市"}.get(N, "三种预设之外")
    rec("calendar", "signal_days", f"{sig[0].date()}~{sig[-1].date()}", f"N={N}; {case}")
    rec("calendar", "N", N, "信号日个数")
    rec("calendar", "n", n, f"评估信号日 {sig[M5.FWD_SKIP_HEAD].date()}~{sig[-1].date()}(前 {M5.FWD_SKIP_HEAD} 天只建仓)")
    rec("calendar", "build_days", ";".join(str(d.date()) for d in sig[:M5.FWD_SKIP_HEAD]), "")
    rec("calendar", "holidays_in_window", ";".join(str(d.date()) for d in pd.bdate_range("2026-09-01", "2026-10-09")
                                                 if d not in set(td)), "工作日中的休市日")
    rec("calendar", "decision", f"N={N}, n={n}", "由交易所日历决定,不看行情", decision=case)
    return sig


def csi1000_decision(codes: pd.Index):
    w = pd.read_csv(RAW / "ak_csindex" / "000852_cons_weight_fetched20261010.csv", dtype={"成分券代码": str})
    c = pd.read_csv(RAW / "ak_csindex" / "000852_cons_fetched20261010.csv", dtype={"成分券代码": str})
    dS = pd.Timestamp(w["日期"].iloc[0])
    w["code"] = w["成分券代码"].str.zfill(6) + np.where(w["交易所"].str.contains("上海"), ".SH",
                                                        np.where(w["交易所"].str.contains("深圳"), ".SZ", ".BJ"))
    n = len(w)
    s = float(w["权重"].sum())
    mapped = float(w["code"].isin(codes).mean())
    same_list = set(w["成分券代码"]) == set(c["成分券代码"])
    ok = (990 <= n <= 1010) and abs(s - 100.0) <= 0.5 and mapped >= 0.99 and dS >= pd.Timestamp("2026-06-15")
    rec("csi1000", "n_constituents", n, "有效条件 [990, 1010]")
    rec("csi1000", "weight_sum_pct", round(s, 4), "有效条件 100 ± 0.5")
    rec("csi1000", "mapped_to_panel_share", round(mapped, 4), "有效条件 >= 0.99")
    rec("csi1000", "weight_date", str(dS.date()), "有效条件 >= 2026-06-15")
    rec("csi1000", "cons_list_date", str(pd.Timestamp(c["日期"].iloc[0]).date()), f"成分名单与权重名单相同={same_list}")
    rec("csi1000", "max_weight_pct", float(w["权重"].max()), "")
    rec("csi1000", "decision", "actual" if ok else "proxy_fallback", "四条有效条件" + ("全部满足" if ok else "不全满足"),
        decision="中证1000 实际成分(成交时点权重)" if ok else "回退 PIT 代理")
    return ok


def _rel_agree(a, b):
    """相对差 < REL_TOL 的比例;分母 = tushare 值非空的股票(东财缺值记不一致)。b 为 0 时要求 a 也为 0。"""
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    base = np.isfinite(b)
    with np.errstate(invalid="ignore", divide="ignore"):
        rel = np.abs(a - b) / np.abs(b)
    good = np.isfinite(a) & base & (np.where(b == 0, a == 0, rel < REL_TOL))
    return float(good[base].mean()) if base.any() else np.nan, int(base.sum())


def _em(table: str, period: str) -> pd.DataFrame:
    d = pd.read_csv(RAW / "ak_em" / f"raw_{table}_{period}_fetched20261010.csv", dtype={"SECURITY_CODE": str})
    d["stock_code"] = d["SECUCODE"].astype(str)
    d["NOTICE_DATE"] = pd.to_datetime(d["NOTICE_DATE"])
    return d.drop_duplicates("stock_code", keep="last").set_index("stock_code")


def statements_decision():
    inc = pd.read_pickle(DATA / "stock_income_2015_now.pkl")
    bal = pd.read_pickle(DATA / "stock_balancesheet_2015_now.pkl")
    for d in (inc, bal):
        for c in ("ann_date", "f_ann_date", "end_date"):
            d[c] = pd.to_datetime(d[c], errors="coerce")
        d["avail"] = d[["ann_date", "f_ann_date"]].max(axis=1)
    rec("statements", "tushare_income_source_end", str(inc["avail"].max().date()), "利润表原始文件最晚可见日")
    rec("statements", "tushare_balance_source_end", str(bal["avail"].max().date()),
        f"资产负债表最晚报告期 {bal['end_date'].max().date()}(无 2026 半年报)")
    res = {}
    for per, lab in (("20260331", "2026Q1"), ("20260630", "2026H1(tushare 已收录部分)")):
        end = pd.Timestamp(per)
        ti = inc[inc["end_date"] == end].sort_values("avail").drop_duplicates("stock_code", keep="last").set_index("stock_code")
        tb = bal[bal["end_date"] == end].sort_values("avail").drop_duplicates("stock_code", keep="last").set_index("stock_code")
        el, ez = _em("lrb", per), _em("zcfz", per)
        mi = ti.join(el, how="inner")
        rev_em = mi["OPERATE_INCOME"].where(mi["OPERATE_INCOME"].notna(), mi["TOTAL_OPERATE_INCOME"])
        for fld, a, b in (("operate_profit", mi["OPERATE_PROFIT"], mi["operate_profit"]),
                          ("revenue", rev_em, mi["revenue"]),
                          ("n_income_attr_p", mi["PARENT_NETPROFIT"], mi["n_income_attr_p"])):
            r, n = _rel_agree(a, b)
            res[(lab, fld)] = r
            rec("statements", f"agree[{lab}][{fld}]", round(r, 5) if np.isfinite(r) else r, f"n={n}(tushare 非空且东财有行)")
        if len(tb):
            mb = tb.join(ez, how="inner")
            r, n = _rel_agree(mb["TOTAL_EQUITY"], mb["total_hldr_eqy_inc_min_int"])
            res[(lab, "股东权益")] = r
            rec("statements", f"agree[{lab}][股东权益]", round(r, 5), f"TOTAL_EQUITY vs total_hldr_eqy_inc_min_int, n={n}")
        else:
            rec("statements", f"agree[{lab}][股东权益]", np.nan, "tushare 无该期资产负债表,无重叠")
        # 诊断(不进判定):营业总收入两源一致率 —— 说明 revenue 不一致来自东财缺"营业收入"字段而非数值错误
        r, n = _rel_agree(mi["TOTAL_OPERATE_INCOME"], mi["total_revenue"])
        rec("statements", f"diag[{lab}][total_revenue<-TOTAL_OPERATE_INCOME]", round(r, 5), f"n={n};诊断,不进判定")
        r2 = float((mi["OPERATE_INCOME"].notna()).mean())
        rec("statements", f"diag[{lab}][EM OPERATE_INCOME 非空比例]", round(r2, 4), "东财只对金融类给营业收入")
    vals = [v for v in res.values() if np.isfinite(v)]
    ok = len(vals) > 0 and min(vals) >= 0.99
    fails = [f"{k[0]}/{k[1]}={v:.4f}" for k, v in res.items() if np.isfinite(v) and v < 0.99]
    rec("statements", "decision", "splice_em" if ok else "fallback_last_known",
        "四项一致率全部 >= 99%" if ok else "未达 99%: " + "; ".join(fails),
        decision=("拼接东财 2026H1/Q3 快照" if ok else
                  "原规则:accept_source_gap(FU.REPORT_TABLES) 内沿用最后已知报告期(tushare:利润/现金流到 08-20、资产负债表到 06-13)"))
    return ok


def turnover_shares_qa():
    """§4.8 第 4 条:换手与股本两源一致率(只报告)。"""
    db = pd.read_csv(DATA / "daily_basic.csv", usecols=["ts_code", "trade_date", "close", "turnover_rate"])
    db = db[(db["trade_date"] >= 20260701) & (db["trade_date"] <= 20260817)]
    db["date"] = pd.to_datetime(db["trade_date"].astype(str))
    bd = FD.load_daily()
    bd = bd[(bd["date"] >= "2026-07-01") & (bd["date"] <= "2026-08-17")]
    m = db.merge(bd[["code", "date", "turn", "close", "tradestatus"]], left_on=["ts_code", "date"],
                 right_on=["code", "date"], how="inner", suffixes=("_db", "_bs"))
    m = m[(m["tradestatus"] == 1) & (m["turn"] > 0)]
    with np.errstate(invalid="ignore", divide="ignore"):
        rel = np.abs(m["turn"] / m["turnover_rate"] - 1.0)
        relc = np.abs(m["close_bs"] / m["close_db"] - 1.0)
    rec("qa_turn", "turn_within_1pct_share", round(float((rel < 0.01).mean()), 5),
        f"daily_basic turnover_rate vs baostock turn,07-01~08-17,n={len(m)}")
    rec("qa_turn", "turn_rel_median", float(np.nanmedian(rel)), "")
    rec("qa_turn", "raw_close_within_1e-4_share", round(float((relc < 1e-4).mean()), 5),
        "daily_basic close vs baostock 日线 close(未复权),07-01~08-17")
    pr = FD.load_profit()
    bal = pd.read_pickle(DATA / "stock_balancesheet_2015_now.pkl")[["stock_code", "end_date", "ann_date", "f_ann_date",
                                                                       "total_share"]]
    bal["end_date"] = pd.to_datetime(bal["end_date"])
    bal["avail"] = pd.to_datetime(bal[["ann_date", "f_ann_date"]].max(axis=1).astype(str), errors="coerce")
    for per in ("2025-12-31", "2026-03-31"):
        tb = bal[bal["end_date"] == per].sort_values("avail").drop_duplicates("stock_code", keep="last").set_index("stock_code")
        pb = pr[pr["statDate"] == per].drop_duplicates("code", keep="last").set_index("code")
        j = tb.join(pb[["totalShare"]], how="inner")
        j = j[(j["total_share"] > 0) & (j["totalShare"] > 0)]
        r = np.abs(j["totalShare"] / j["total_share"] - 1.0)
        rec("qa_shares", f"totalShare_vs_tushare_within_1e-4[{per}]", round(float((r < 1e-4).mean()), 5), f"n={len(j)}")


def main():
    t0 = time.time()
    codes = pd.read_parquet(ROOT / "cache" / "daily" / "close_px.parquet", columns=["000001.SZ"]).columns  # 占位
    codes = pd.Index(pd.read_csv(RAW / "bs_meta" / "universe.csv")["code"])
    rec("meta", "run_at", time.strftime("%Y-%m-%d %H:%M:%S %z"), "计算任何前向收益之前")
    calendar_decision()
    csi1000_decision(codes)
    statements_decision()
    turnover_shares_qa()
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print(f"写 {OUT.name},{len(df)} 行,{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
