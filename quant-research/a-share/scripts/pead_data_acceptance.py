"""PEAD 事件数据验收(方案 v0.2 第 3 节 + 第 8 节"本地数值定义验收")。

**只做数据与定义:不读取、不计算任何前向收益、IC 或回测。**
用到的面板字段只有 close_px 的日期索引、amount(ADV20 / 当日有成交)与 ST 名单 ——
都是 t_signal 收盘及以前的非收益信息。

产出(research/ledger/):
  pead_event_coverage.csv         按 P 年 × 来源 × 指标 × 版本:事件数、股票数、信号状态分布
  pead_event_main_by_year.csv     主事件按 V 年 × 财季:事件数、股票数、新鲜度、P 为非交易日占比
  pead_event_v_distribution.csv   主事件 V 的月份分布、P→V 日历间隔
  pead_event_conflicts.csv        重复、版本、日期冲突、frontier、法定截止等计数
  pead_event_unit_checks.csv      预告/快报与正式财报的数值口径对照(单位、是否同一指标)
  pead_event_panel_consistency.csv 与 fu_op_yoy 现有面板(fundamental.build_report_level/_pit_daily)对照
  pead_event_spotcheck.csv        20 个随机主事件从原始利润表逐行重算
  pead_st_coverage.csv            ST 逐日名单的覆盖与区间
  pead_event_pool_size.csv        逐成交日 T 的事件池规模(过池前/过池后)、E 的入选人数与目标现金
  pead_event_pool_summary.csv     上表按年汇总
  pead_frozen_inputs.json         输入文件指纹(md5)、COMP_v5 版本、随机种子
缓存:cache/_tmp/pead_events.parquet
"""
from __future__ import annotations
import os, sys, json, hashlib, warnings
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

from afactor.config import LEDGER, ROOT, TMP
from afactor import earnings_events as EE

SEED = 20261010
OUT = {k: LEDGER / f"pead_{k}.csv" for k in [
    "event_coverage", "event_main_by_year", "event_v_distribution", "event_conflicts",
    "event_unit_checks", "event_panel_consistency", "event_spotcheck", "st_coverage",
    "event_pool_size", "event_pool_summary"]}


def md5(path, chunk=1 << 22):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


# ----------------------------------------------------------------------------- 1 覆盖
def coverage(ev: pd.DataFrame) -> pd.DataFrame:
    e = ev[ev["P"].dt.year.between(2019, 2025)].copy()
    e["P_year"] = e["P"].dt.year
    g = e.groupby(["P_year", "source", "indicator", "version"])
    out = g.agg(n_events=("stock_code", "size"), n_stocks=("stock_code", "nunique"),
                n_value_ok=("value_status", lambda s: int((s == "ok").sum())),
                n_V_in_panel=("V", lambda s: int(s.notna().sum())),
                n_main=("is_main", lambda s: int(s.fillna(False).sum()))).reset_index()
    out["signal_missing_rate"] = 1 - out["n_value_ok"] / out["n_events"]
    st = (e.groupby(["P_year", "source", "indicator", "version", "value_status"]).size()
          .unstack(fill_value=0).add_prefix("status_").reset_index())
    out = out.merge(st, on=["P_year", "source", "indicator", "version"], how="left")
    return out


def main_by_year(ev: pd.DataFrame) -> pd.DataFrame:
    m = ev[ev["is_main"]].copy()
    m["V_year"] = m["V"].dt.year
    rows = []
    for (y, q), g in m.groupby(["V_year", "fiscal_q"]):
        r = dict(V_year=y, fiscal_q=int(q), n_events=len(g), n_stocks=g["stock_code"].nunique(),
                 P_nontrading_share=float((~g["P_is_trading_day"].fillna(True).astype(bool)).mean()),
                 late_vs_deadline_share=float(g["late_vs_deadline"].fillna(False).astype(bool).mean()),
                 pre_listing_share=float(g["pre_listing"].mean()))
        for k, v in g["freshness"].value_counts().items():
            r[f"fresh_{k}"] = int(v)
        rows.append(r)
    out = pd.DataFrame(rows).fillna(0)
    tot = (m.groupby("V_year").agg(n_events=("stock_code", "size"), n_stocks=("stock_code", "nunique"))
           .reset_index().assign(fiscal_q=0))
    for y, g in m.groupby("V_year"):
        for k, v in g["freshness"].value_counts().items():
            tot.loc[tot["V_year"] == y, f"fresh_{k}"] = int(v)
        tot.loc[tot["V_year"] == y, "P_nontrading_share"] = float(
            (~g["P_is_trading_day"].fillna(True).astype(bool)).mean())
    return pd.concat([out, tot], ignore_index=True).sort_values(["V_year", "fiscal_q"])


def v_distribution(ev: pd.DataFrame) -> pd.DataFrame:
    m = ev[ev["is_main"]].copy()
    rows = []
    m["V_month"] = m["V"].dt.month
    for (y, mo), g in m.groupby([m["V"].dt.year, "V_month"]):
        rows.append(dict(section="V_month", V_year=y, key=int(mo), n_events=len(g),
                         n_stocks=g["stock_code"].nunique()))
    gap = (m["V"] - m["P"]).dt.days
    for k, v in gap.value_counts().sort_index().items():
        rows.append(dict(section="V_minus_P_calendar_days", V_year=0, key=int(k), n_events=int(v)))
    for k, v in m["P"].dt.dayofweek.value_counts().sort_index().items():
        rows.append(dict(section="P_weekday(0=Mon)", V_year=0, key=int(k), n_events=int(v)))
    # 对照:业绩预告(东财)公告日的星期分布 —— 用来判断"公告日"是夜间发布当日还是次日(报纸日)
    f = ev[(ev["source"] == "业绩预告") & ev["P"].dt.year.between(2020, 2025)]
    f = f.drop_duplicates(["stock_code", "end_date", "P"])
    for k, v in f["P"].dt.dayofweek.value_counts().sort_index().items():
        rows.append(dict(section="forecast_P_weekday(0=Mon)", V_year=0, key=int(k), n_events=int(v)))
    for q, g in m.groupby("fiscal_q"):
        lag = (g["P"] - g["end_date"]).dt.days
        for qq in (0.05, 0.5, 0.95):
            rows.append(dict(section=f"P_minus_end_days_q{int(qq*100)}", V_year=0, key=int(q),
                             n_events=float(lag.quantile(qq))))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- 2 冲突 / 重复
def conflicts(ev: pd.DataFrame, inc: pd.DataFrame, fc: pd.DataFrame, ex: pd.DataFrame) -> pd.DataFrame:
    R = []
    add = lambda sec, item, n, denom=None, note="": R.append(dict(
        section=sec, item=item, n=n, denom=denom, share=(n / denom if denom else np.nan), note=note))
    inc19 = inc[inc["ann_date"] >= EE.FIRST_EVENT_P]
    add("利润表", "行数(ann_date 2019-2025)", len(inc19))
    add("利润表", "重复(股票,报告期)", int(inc.duplicated(["stock_code", "end_date"]).sum()), len(inc))
    add("利润表", "f_ann_date>ann_date(修正版本)", int((inc19["f_ann_date"] > inc19["ann_date"]).sum()), len(inc19))
    add("利润表", "f_ann_date<ann_date(日期冲突)", int((inc19["f_ann_date"] < inc19["ann_date"]).sum()), len(inc19))
    add("利润表", "修正版本在 2026 公开(数值封存)", int(inc19["sealed_value"].sum()), len(inc19))
    gap = (inc19["f_ann_date"] - inc19["ann_date"]).dt.days
    add("利润表", "修正间隔中位天数", float(gap[gap > 0].median()))
    op = ev[(ev["source"] == "定期报告") & (ev["indicator"] == EE.MAIN_INDICATOR)]
    v1 = op[op["version"] == 1]
    add("主指标 v1", "首次披露行", len(v1))
    for k, v in v1["value_status"].value_counts().items():
        add("主指标 v1", f"value_status={k}", int(v), len(v1), EE.STATUS_TEXT.get(k, ""))
    add("主指标 v1", "非 frontier(同日或更早已有更晚报告期)", int((~v1["is_frontier"].fillna(False)).sum()), len(v1))
    ok = v1[v1["value_status"] == "ok"]
    add("主指标 v1", "ok 中非 frontier", int((~ok["is_frontier"].fillna(False)).sum()), len(ok))
    add("主指标 v1", "ok 中晚于法定截止", int(ok["late_vs_deadline"].fillna(False).sum()), len(ok))
    add("主指标 v1", "ok 中 P 早于上市日", int(ok["pre_listing"].sum()), len(ok))
    add("主指标 v1", "ok 中 V 缺失(P<2019-12-31 或 V 在封存期)", int(ok["V"].isna().sum()), len(ok))
    # 同一股票同一 P 多个报告期首次披露(典型:年报与一季报同日)
    same = v1.groupby(["stock_code", "P"]).size()
    add("主指标 v1", "同一(股票,P)多个报告期首次披露的组数", int((same > 1).sum()), int(len(same)))
    v2 = op[op["version"] == 2]
    add("主指标 v2", "修正/依赖项更新行", len(v2))
    for k, v in v2["version_type"].value_counts().items():
        add("主指标 v2", f"version_type={k}", int(v), len(v2))
    # ---- 预告 ----
    add("业绩预告", "行数(<=2025,去重后)", len(fc))
    add("业绩预告", "逐字段完全重复行(已删)", int(fc.attrs.get("n_exact_dup_dropped", 0)))
    add("业绩预告", "同(股票,报告期,指标)多个公告日(版本数>1)",
        int((fc.groupby(["stock_code", "end_date", "indicator"])["P"].nunique() > 1).sum()))
    add("业绩预告", "同(股票,报告期,指标,公告日)冲突行", int(fc.duplicated(["stock_code", "end_date", "indicator", "P"]).sum()))
    ann = fc.groupby(["stock_code", "end_date", "P"]).size()
    add("业绩预告", "公告数(股票,报告期,公告日)", int(len(ann)))
    add("业绩预告", "每公告平均指标行数", float(ann.mean()))
    fce = ev[ev["source"] == "业绩预告"]
    for k, v in fce["date_check"].value_counts().items():
        add("业绩预告 日期核对", k, int(v), len(fce))
    add("业绩预告", "P 早于报告期末 1 年以上(异常早)", int((fce["P"] < fce["end_date"] - pd.Timedelta(days=365)).sum()), len(fce))
    add("业绩预告", "股票代码为 B 股/北交所", int((~fce["stock_code"].str[:2].isin(["60", "00", "30", "68"])).sum()), len(fce))
    # ---- 快报 ----
    exe = ev[(ev["source"] == "业绩快报") & (ev["indicator"] == "ex_np_parent")]
    add("业绩快报", "行数(<=2025)", len(exe))
    add("业绩快报", "同(股票,报告期)多行", int(ex.duplicated(["stock_code", "end_date"]).sum()), len(ex))
    for k, v in exe["date_check"].value_counts().items():
        add("业绩快报 日期核对", k, int(v), len(exe))
    add("业绩快报", "报告期早于上市日(招股书回填)", int(exe["period_pre_listing"].sum()), len(exe))
    lag = (exe["P"] - exe["end_date"]).dt.days
    add("业绩快报", "P-报告期末 中位天数", float(lag.median()))
    ok_ex = exe[exe["date_check"] == "早于正式财报"]
    add("业绩快报", "早于正式财报者 P-报告期末 中位天数", float((ok_ex["P"] - ok_ex["end_date"]).dt.days.median()))
    for y, g in exe.groupby(exe["P"].dt.year):
        add("业绩快报 晚于正式财报占比(按 P 年)", str(y), int((g["date_check"] == "晚于正式财报").sum()), len(g))
    return pd.DataFrame(R)


# ----------------------------------------------------------------------------- 3 口径对照
def unit_checks(inc: pd.DataFrame, fc: pd.DataFrame, ex: pd.DataFrame) -> pd.DataFrame:
    """预告"上年同期值"、快报"去年同期/本期"与正式财报累计值对照:验证单位(元)与指标口径。"""
    R = []
    cum = inc[["stock_code", "end_date", "n_income_attr_p", "revenue", "ann_date"]].copy()
    ly = cum.copy()
    ly["end_date"] = ly["end_date"] + pd.offsets.DateOffset(years=1)
    ly["end_date"] = ly["end_date"] + pd.offsets.MonthEnd(0)
    for ind, col in [("fc_np_parent", "n_income_attr_p"), ("fc_revenue", "revenue"),
                     ("fc_np_deducted", "n_income_attr_p")]:
        x = fc[fc["indicator"] == ind][["stock_code", "end_date", "上年同期值"]].merge(
            ly[["stock_code", "end_date", col]], on=["stock_code", "end_date"], how="inner")
        x = x.dropna()
        rel = (x["上年同期值"] - x[col]).abs() / x[col].abs().clip(lower=1.0)
        R.append(dict(check=f"预告[{ind}].上年同期值 vs 正式财报去年同期累计 {col}", n=len(x),
                      within_1pct=float((rel <= 0.01).mean()), within_10pct=float((rel <= 0.10).mean()),
                      ratio_median=float((x["上年同期值"] / x[col]).median())))
    for nm, c_cur, col in [("净利润", "净利润-净利润", "n_income_attr_p"),
                           ("营业收入", "营业收入-营业收入", "revenue")]:
        x = ex[["stock_code", "end_date", c_cur, "P"]].merge(
            cum[["stock_code", "end_date", col, "ann_date"]], on=["stock_code", "end_date"], how="inner").dropna()
        rel = (x[c_cur] - x[col]).abs() / x[col].abs().clip(lower=1.0)
        for lab, msk in [("全部", np.ones(len(x), bool)), ("快报早于正式财报", x["P"] < x["ann_date"]),
                         ("快报晚于正式财报", x["P"] > x["ann_date"])]:
            r = rel[msk]
            R.append(dict(check=f"快报 {nm} 本期 vs 正式财报累计 {col} [{lab}]", n=int(msk.sum()),
                          within_1pct=float((r <= 0.01).mean()), within_10pct=float((r <= 0.10).mean()),
                          ratio_median=float((x.loc[msk, c_cur] / x.loc[msk, col]).median())))
    return pd.DataFrame(R)


# ----------------------------------------------------------------------------- 4 与现有面板一致性
def panel_consistency(ev: pd.DataFrame, inc: pd.DataFrame, cal: pd.DatetimeIndex, codes: pd.Index):
    """把 fundamental.build_report_level 的读取截断到 avail <= 2025-12-31 后重建现有面板。

    截断只影响 2026 年后才可见的值;对 <= 2025-12-31 的面板行与未截断版逐位相同
    (avail_yoy 是全部依赖项 avail 的最大值,任何依赖 2026 行的值在 2025 年都不可见)。
    """
    from afactor import fundamental as FU
    orig = FU._load

    def _load_sealed(name, items):
        d = orig(name, items)
        return d[d["avail"] <= EE.CUTOFF]
    FU._load = _load_sealed
    FU._REP_CACHE.clear()
    try:
        # fixb 第二轮审查 P2:报告表现在含 as-of 快照行(pit_vintage < 99),按 (股票, 报告期) merge 前只留最终快照,
        # 否则多版本报告期会被重复计数
        rep = FU._final_rows(FU.build_report_level())
    finally:
        FU._load = orig
    R = []
    # ---- (a) 报告期维度:数值与可见日 ----
    mine = EE.formal_signal(inc, "operate_profit")
    a = mine[["stock_code", "end_date", "value", "dep_max_avail", "P_first", "status_v1"]].merge(
        rep[["stock_code", "end_date", "operate_profit_yoy", "avail_yoy"]],
        on=["stock_code", "end_date"], how="inner")
    a = a[a["P_first"] >= EE.FIRST_EVENT_P]
    both = a["value"].notna() & a["operate_profit_yoy"].notna()
    diff = (a.loc[both, "value"] - a.loc[both, "operate_profit_yoy"]).abs()
    R.append(dict(level="报告期", item="两边都有值的(股票,报告期)", n=int(both.sum())))
    R.append(dict(level="报告期", item="数值逐位相同(|差|<1e-12)", n=int((diff < 1e-12).sum()), denom=int(both.sum())))
    R.append(dict(level="报告期", item="只有本模块有值", n=int((a["value"].notna() & a["operate_profit_yoy"].isna()).sum())))
    R.append(dict(level="报告期", item="只有现有面板有值", n=int((a["value"].isna() & a["operate_profit_yoy"].notna()).sum())))
    av_b = a.loc[both]
    R.append(dict(level="报告期", item="可见日相同(本模块 dep_max_avail == 现有 avail_yoy)",
                  n=int((av_b["dep_max_avail"] == av_b["avail_yoy"]).sum()), denom=len(av_b)))
    R.append(dict(level="报告期", item="现有 avail_yoy 更晚(三表合并/Q1 依赖上季)",
                  n=int((av_b["avail_yoy"] > av_b["dep_max_avail"]).sum()), denom=len(av_b)))
    R.append(dict(level="报告期", item="现有 avail_yoy 更早(不应出现)",
                  n=int((av_b["avail_yoy"] < av_b["dep_max_avail"]).sum()), denom=len(av_b)))
    later = av_b[av_b["avail_yoy"] > av_b["dep_max_avail"]]
    R.append(dict(level="报告期", item="现有更晚者的中位推迟天数",
                  n=float((later["avail_yoy"] - later["dep_max_avail"]).dt.days.median()) if len(later) else np.nan))
    # 原因拆分:把"Q1 也依赖上季(去年 Q4)"加回本模块的利润表依赖后能否复现现有 avail_yoy
    lat = later.merge(mine[["stock_code", "end_date", "quarter", "avail_prev", "avail_lyprev"]],
                      on=["stock_code", "end_date"], how="left")
    q1 = lat["quarter"] == 1
    alt = pd.concat([lat["dep_max_avail"], lat["avail_prev"].where(q1),
                     lat["avail_lyprev"].where(q1)], axis=1).max(axis=1)
    n_q1 = int((alt == lat["avail_yoy"]).sum())
    R.append(dict(level="报告期", item="现有更晚者:原因=Q1 依赖去年 Q4(利润表口径即可复现)", n=n_q1, denom=len(lat)))
    R.append(dict(level="报告期", item="现有更晚者:原因=三表合并(资产负债/现金流量表更正日更晚)",
                  n=len(lat) - n_q1, denom=len(lat)))
    # 本模块 status_v1 != ok 但现有面板照常给值 —— 现有面板把修正后的数值放在修正日可见
    R.append(dict(level="报告期", item="首次披露不可恢复、现有面板在修正日给出修正值",
                  n=int((a["status_v1"].isin(["own_revised_later", "dep_available_later"])
                         & a["operate_profit_yoy"].notna()).sum())))
    # ---- (b) 日频:主事件在 V-1 / V 行的面板值 ----
    pan = FU._pit_daily(rep, "operate_profit_yoy", cal, codes)
    m = ev[ev["is_main"] & ev["stock_code"].isin(codes)].copy()
    di = cal.to_numpy()
    iv = np.searchsorted(di, m["V"].to_numpy(dtype="datetime64[ns]"))
    jc = codes.get_indexer(m["stock_code"])
    P = pan.to_numpy(np.float64)
    val = m["value"].to_numpy(np.float64)
    at_v = P[iv, jc]
    at_vm1 = np.where(iv > 0, P[np.clip(iv - 1, 0, None), jc], np.nan)
    tol = lambda x: np.isfinite(x) & (np.abs(x - val) <= 1e-6 * np.maximum(1.0, np.abs(val)))
    eq_v, eq_vm1 = tol(at_v), tol(at_vm1)
    m["eq_V"], m["eq_Vm1"] = eq_v, eq_vm1
    n = len(m)
    R.append(dict(level="日频", item="主事件(股票在面板内)", n=n))
    R.append(dict(level="日频", item="面板 V-1 行已等于事件值(引擎在 V 成交,与协议一致)", n=int(eq_vm1.sum()), denom=n))
    R.append(dict(level="日频", item="面板 V 行才等于事件值(引擎晚 1 日)", n=int((eq_v & ~eq_vm1).sum()), denom=n))
    R.append(dict(level="日频", item="面板 V 行仍不等于事件值", n=int((~eq_v).sum()), denom=n))
    ptd = m["P_is_trading_day"].fillna(True).astype(bool).to_numpy()
    R.append(dict(level="日频", item="其中 P 为交易日:V-1 行已相等", n=int((eq_vm1 & ptd).sum()), denom=int(ptd.sum())))
    R.append(dict(level="日频", item="其中 P 为非交易日:V-1 行已相等", n=int((eq_vm1 & ~ptd).sum()), denom=int((~ptd).sum())))
    R.append(dict(level="日频", item="其中 P 为非交易日:V 行才相等", n=int((eq_v & ~eq_vm1 & ~ptd).sum()), denom=int((~ptd).sum())))
    # 不等者:之后多少个交易日面板才出现该值(<= 60)
    bad = np.flatnonzero(~eq_v)
    lags = []
    for i in bad:
        col = P[iv[i]:min(iv[i] + 61, len(di)), jc[i]]
        hit = np.flatnonzero(np.isfinite(col) & (np.abs(col - val[i]) <= 1e-6 * max(1.0, abs(val[i]))))
        lags.append(int(hit[0]) if hit.size else -1)
    lags = np.asarray(lags)
    mb = m.iloc[bad][["stock_code", "end_date", "P"]].merge(
        a[["stock_code", "end_date", "avail_yoy"]], on=["stock_code", "end_date"], how="left")
    R.append(dict(level="日频", item="V 行不等者:现有面板该期 avail_yoy 晚于事件 P",
                  n=int((mb["avail_yoy"] > mb["P"]).sum()), denom=int(len(mb))))
    # 其余:P 之后至 V 日(如年报周五、一季报周六或 V 日盘后)同一股票又首次公告了更晚的报告期 ->
    # 现有面板 V 行(= V 收盘后的信息集)显示更晚报告期,不是错误。event_pool() 对同一 (T, 股票)
    # 保留最新报告期;V 日盘后才公告的更晚报告期在 T=V 时尚不可见,两者口径一致。
    oth = mb[~(mb["avail_yoy"] > mb["P"])]
    ann = inc[["stock_code", "end_date", "ann_date"]]
    vv = m.iloc[bad][["stock_code", "end_date", "V"]]
    oth = oth.merge(vv, on=["stock_code", "end_date"], how="left")
    hit = 0
    for _, e in oth.iterrows():
        x = ann[(ann["stock_code"] == e["stock_code"]) & (ann["end_date"] > e["end_date"])
                & (ann["ann_date"] > e["P"]) & (ann["ann_date"] <= e["V"])]
        hit += int(len(x) > 0)
    R.append(dict(level="日频", item="V 行不等者:P 之后至 V 日(含,面板 V 行=V 收盘后信息)同股首次公告了更晚报告期",
                  n=hit, denom=int(len(mb))))
    R.append(dict(level="日频", item="V 行不等者:原因未归类", n=int(len(oth) - hit), denom=int(len(mb))))
    R.append(dict(level="日频", item="V 行不等者:60 日内出现该值", n=int((lags > 0).sum()), denom=int(len(lags))))
    R.append(dict(level="日频", item="V 行不等者:60 日内从未出现(被下一期覆盖或面板值不同)", n=int((lags < 0).sum()), denom=int(len(lags))))
    if (lags > 0).any():
        R.append(dict(level="日频", item="V 行不等者:出现时距 V 的中位交易日", n=float(np.median(lags[lags > 0]))))
    out = pd.DataFrame(R)
    out["share"] = out["n"] / out["denom"]
    return out, pan, rep


# ----------------------------------------------------------------------------- 5 抽查 20 个
def spotcheck(ev: pd.DataFrame, pan: pd.DataFrame, cal: pd.DatetimeIndex, k: int = 20) -> pd.DataFrame:
    """从原始利润表 pickle **独立重读**每个抽中事件的 4 个依赖行,手算单季与对称同比。"""
    m = ev[ev["is_main"] & ev["V"].dt.year.between(2020, 2025)]
    rng = np.random.default_rng(SEED)
    pick = m.iloc[rng.choice(len(m), size=k, replace=False)].sort_values(["V", "stock_code"])
    raw = pd.read_pickle(EE.INCOME_PKL)[["stock_code", "end_date", "ann_date", "f_ann_date", "operate_profit"]]
    raw = raw[raw["stock_code"].isin(pick["stock_code"])].copy()
    for c in ("end_date", "ann_date", "f_ann_date"):
        raw[c] = pd.to_datetime(raw[c])
    raw = raw[raw["ann_date"] <= EE.CUTOFF]
    di = cal.to_numpy()
    rows = []
    for _, e in pick.iterrows():
        r = raw[raw["stock_code"] == e["stock_code"]].set_index("end_date")
        end = e["end_date"]
        q = end.quarter
        prev_end = (end - pd.offsets.QuarterEnd(1))
        ly_end = end - pd.offsets.DateOffset(years=1) + pd.offsets.MonthEnd(0)
        lyp_end = (ly_end - pd.offsets.QuarterEnd(1))
        g = lambda d, c: (r.loc[d, c] if d in r.index else np.nan)
        cur, prev, ly, lyp = (g(end, "operate_profit"), g(prev_end, "operate_profit"),
                              g(ly_end, "operate_profit"), g(lyp_end, "operate_profit"))
        xq = cur if q == 1 else cur - prev
        xly = ly if q == 1 else ly - lyp
        den = (abs(xq) + abs(xly)) / 2
        hand = float(np.clip((xq - xly) / den, -2, 2)) if den > 1e-6 else np.nan
        deps = [end, ly_end] if q == 1 else [end, prev_end, ly_end, lyp_end]
        dep_dates = [max(g(d, "ann_date"), g(d, "f_ann_date")) for d in deps]
        iv = int(np.searchsorted(di, np.datetime64(e["V"])))
        pv = float(pan.iloc[iv, pan.columns.get_loc(e["stock_code"])]) if e["stock_code"] in pan.columns else np.nan
        pvm1 = float(pan.iloc[iv - 1, pan.columns.get_loc(e["stock_code"])]) if e["stock_code"] in pan.columns and iv > 0 else np.nan
        rows.append(dict(
            stock_code=e["stock_code"], end_date=end.date(), fiscal_q=q, P=e["P"].date(), V=e["V"].date(),
            P_is_trading_day=bool(e["P_is_trading_day"]) if pd.notna(e["P_is_trading_day"]) else None,
            op_cum=cur, op_cum_prevq=prev if q != 1 else np.nan, op_cum_ly=ly,
            op_cum_ly_prevq=lyp if q != 1 else np.nan, op_q=xq, op_q_ly=xly,
            dep_ends="|".join(str(d.date()) for d in deps),
            dep_public_dates="|".join(str(pd.Timestamp(d).date()) for d in dep_dates),
            hand_value=hand, event_value=float(e["value"]), panel_at_Vm1=pvm1, panel_at_V=pv,
            hand_eq_event=bool(np.isclose(hand, e["value"], rtol=0, atol=1e-12)),
            panel_eq_event_at_V=bool(np.isclose(pv, e["value"], rtol=1e-6, atol=1e-6)),
            freshness=e["freshness"], fc_first_P=(e["fc_first_P"].date() if pd.notna(e["fc_first_P"]) else None),
            fc_indicators=e["fc_indicators"], ex_P=(e["ex_P"].date() if pd.notna(e["ex_P"]) else None)))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- 6 ST
def st_coverage(cal: pd.DatetimeIndex) -> pd.DataFrame:
    st = pd.read_csv(EE.ST_CSV, usecols=["stock_code", "trade_date"])
    st["trade_date"] = pd.to_datetime(st["trade_date"])
    st = st[st["trade_date"] <= EE.CUTOFF]
    days = pd.DatetimeIndex(np.sort(st["trade_date"].unique()))
    R = []
    for y in range(2020, 2026):
        cy = cal[cal.year == y]
        dy = days[days.year == y]
        R.append(dict(item="名单覆盖交易日", year=y, n=int(len(cy.intersection(dy))), denom=len(cy)))
        R.append(dict(item="名单缺失交易日", year=y, n=int(len(cy.difference(dy))), denom=len(cy)))
        R.append(dict(item="名单上的股票数", year=y, n=int(st[st["trade_date"].dt.year == y]["stock_code"].nunique())))
    sp = EE.st_spells()
    sp = sp[sp["end"] >= pd.Timestamp("2020-01-01")]
    R.append(dict(item="2020-2025 内 ST 区间数", year=0, n=len(sp)))
    R.append(dict(item="2020-2025 内有 ST 区间的股票数", year=0, n=int(sp["stock_code"].nunique())))
    out = pd.DataFrame(R)
    out["share"] = out["n"] / out["denom"]
    return out


# ----------------------------------------------------------------------------- 7 事件池规模(不含收益)
def pool_sizes(ev: pd.DataFrame, cal: pd.DatetimeIndex):
    from afactor.panel import Panel
    p = Panel()
    Q = EE.qualified_pool(p)                       # 引擎信号行 t = 协议 T-1
    pool = EE.event_pool(EE.main_events(ev), cal)
    codes = p.codes
    jc = codes.get_indexer(pool["stock_code"])
    it = cal.get_indexer(pool["t_signal"])
    okidx = (jc >= 0) & (it >= 0)
    q = np.zeros(len(pool), bool)
    q[okidx] = Q.to_numpy()[it[okidx], jc[okidx]]
    pool["in_panel"] = jc >= 0
    pool["qualified"] = q
    sw = pd.read_csv(ROOT / "stock_sw_industry.csv", encoding="utf-8-sig").drop_duplicates("ts_code")
    pool["has_sw"] = pool["stock_code"].isin(set(sw["ts_code"]))
    g_all = pool.groupby("T").size().rename("n_events_raw")
    g_pan = pool[pool["in_panel"]].groupby("T").size().rename("n_events_in_panel")
    qp = pool[pool["qualified"]]
    g_q = qp.groupby("T").size().rename("n_pool")
    df = pd.concat([g_all, g_pan, g_q], axis=1).reindex(cal[1:]).fillna(0).astype(int)
    df.index.name = "T"
    n = df["n_pool"].to_numpy()
    nsel = np.where(n >= 5, np.ceil(0.2 * n), 0).astype(int)
    w = np.where(nsel > 0, np.minimum(1.0 / np.maximum(nsel, 1), 0.05), 0.0)
    df["n_sel_E"] = nsel
    df["E_target_cash"] = 1.0 - nsel * w
    df["batch_active"] = n >= 5
    df["qualified_pool_size"] = Q.sum(axis=1).shift(1).reindex(df.index).fillna(0).astype(int)
    fr = qp.groupby(["T", "freshness"]).size().unstack(fill_value=0).add_prefix("pool_fresh_")
    df = df.join(fr).fillna(0)
    df["pool_no_sw_share"] = (qp.groupby("T")["has_sw"].apply(lambda s: 1 - s.mean())
                              .reindex(df.index))
    df = df.reset_index()
    df["phase"] = np.arange(1, len(cal))[: len(df)] % 5        # 相位 = 交易日序号 mod 5(序号从 2020-01-02=0 起)
    df["year"] = pd.to_datetime(df["T"]).dt.year
    rows = []
    for y, g in df.groupby("year"):
        r = dict(year=y, n_T=len(g), pool_mean=g["n_pool"].mean(), pool_median=g["n_pool"].median(),
                 pool_p90=g["n_pool"].quantile(0.9), active_share=g["batch_active"].mean(),
                 nsel_mean_active=g.loc[g["batch_active"], "n_sel_E"].mean(),
                 target_cash_mean_active=g.loc[g["batch_active"], "E_target_cash"].mean(),
                 target_cash_mean_all=g["E_target_cash"].mean(),
                 events_raw=g["n_events_raw"].mean(), events_in_panel=g["n_events_in_panel"].mean(),
                 qualified_pool_mean=g["qualified_pool_size"].mean(),
                 pool_no_sw_share=g["pool_no_sw_share"].mean())
        for ph in range(5):
            gp = g[g["phase"] == ph]
            r[f"active_batches_phase{ph}"] = int(gp["batch_active"].sum())
            r[f"batches_phase{ph}"] = len(gp)
        rows.append(r)
    # 合计行(year=0):股票-T 对数口径,看面板缺失与合格池过滤各丢多少
    rows.append(dict(year=0, n_T=len(df), pairs_raw=int(len(pool)), pairs_in_panel=int(pool["in_panel"].sum()),
                     pairs_qualified=int(pool["qualified"].sum())))
    return df, pd.DataFrame(rows), Q


def lot_feasibility(ev: pd.DataFrame, cal: pd.DatetimeIndex, Q: pd.DataFrame) -> pd.DataFrame:
    """E 臂按方案规则选股后,整手约束下目标仓位能否落地(**只用信号与 t_signal 未复权收盘价,不含收益**)。

    E 选股:合格事件池内按主信号降序取 ceil(20%),同分按代码;池 < 5 只不建仓;
    权重 w = min(1/n_sel, 5%)。整手:沪深主板/创业板 100 股一手;科创板(688/689)最低 200 股、
    之后 1 股递增。价格用 t_signal(= T-1)未复权收盘(daily_basic.close,读入后截断到 <= 2025-12-31)。
    不含参与率上限(那需要 T 日成交窗口成交额,属于回测引擎的执行层)。
    """
    pool = EE.event_pool(EE.main_events(ev), cal)
    codes = Q.columns
    jc = codes.get_indexer(pool["stock_code"])
    it = cal.get_indexer(pool["t_signal"])
    ok = (jc >= 0) & (it >= 0)
    q = np.zeros(len(pool), bool)
    q[ok] = Q.to_numpy()[it[ok], jc[ok]]
    pool = pool[q].copy()
    db = pd.read_csv(ROOT / "data_8:21" / "daily_basic.csv", usecols=["ts_code", "trade_date", "close"])
    db["trade_date"] = pd.to_datetime(db["trade_date"].astype(str), format="%Y%m%d")
    db = db[db["trade_date"] <= EE.CUTOFF]                       # 封存截断
    db = db.rename(columns={"ts_code": "stock_code", "trade_date": "t_signal", "close": "raw_close"})
    pool = pool.merge(db, on=["stock_code", "t_signal"], how="left")
    del db
    pool["neg"] = -pool["value"]
    pool = pool.sort_values(["T", "neg", "stock_code"])
    pool["n_pool"] = pool.groupby("T")["stock_code"].transform("size")
    pool["rank"] = pool.groupby("T").cumcount() + 1
    pool["n_sel"] = np.ceil(0.2 * pool["n_pool"]).astype(int)
    sel = pool[(pool["n_pool"] >= 5) & (pool["rank"] <= pool["n_sel"])].copy()
    sel["w"] = np.minimum(1.0 / sel["n_sel"], 0.05)
    star = sel["stock_code"].str[:3].isin(["688", "689"])
    rows = []
    for cap in (1e6, 1e7):
        tv = cap * sel["w"]
        px = sel["raw_close"]
        sh_main = np.floor(tv / (px * 100)) * 100
        sh_star = np.floor(tv / px)
        sh = np.where(star, np.where(sh_star >= 200, sh_star, 0), sh_main)
        filled = sh * px / cap
        sel[f"fill_{int(cap)}"] = filled
        sel[f"zero_{int(cap)}"] = (sh == 0) & px.notna()
    # 截断点同分:第 n_sel 名与第 n_sel+1 名信号值相同 -> 入选由"代码排序"决定(方案的同分规则)
    nxt = pool[pool["rank"] == pool["n_sel"] + 1][["T", "value"]].rename(columns={"value": "v_next"})
    cut = sel[sel["rank"] == sel["n_sel"]][["T", "value"]].merge(nxt, on="T", how="left")
    cut["tie_at_cut"] = np.isclose(cut["value"], cut["v_next"], rtol=0, atol=1e-12)
    tie_T = cut.set_index("T")["tie_at_cut"]
    by_T = sel.groupby("T").agg(year=("T", lambda s: s.iloc[0].year), n_sel=("stock_code", "size"),
                                target=("w", "sum"), px_missing=("raw_close", lambda s: s.isna().mean()),
                                fill_1m=("fill_1000000", "sum"), fill_10m=("fill_10000000", "sum"),
                                zero_1m=("zero_1000000", "mean"), zero_10m=("zero_10000000", "mean"))
    by_T["tie_at_cut"] = tie_T.reindex(by_T.index).fillna(False).astype(bool)
    by_T["cut_value"] = cut.set_index("T")["value"].reindex(by_T.index)
    for y, g in by_T.groupby("year"):
        rows.append(dict(year=y, active_T=len(g), n_sel_mean=g["n_sel"].mean(), n_sel_max=int(g["n_sel"].max()),
                         target_invest_mean=g["target"].mean(),
                         lot_invest_mean_1m=g["fill_1m"].mean(), lot_invest_mean_10m=g["fill_10m"].mean(),
                         zero_lot_share_1m=g["zero_1m"].mean(), zero_lot_share_10m=g["zero_10m"].mean(),
                         per_stock_value_median_1m=float(np.median(1e6 / np.maximum(g["n_sel"], 20))),
                         raw_price_missing_share=g["px_missing"].mean(),
                         tie_at_cut_share=g["tie_at_cut"].mean(),
                         cut_at_plus2_share=float((g["cut_value"] >= 2 - 1e-12).mean())))
    for lo, hi in [(1, 20), (21, 100), (101, 300), (301, 10000)]:
        g = by_T[(by_T["n_sel"] >= lo) & (by_T["n_sel"] <= hi)]
        rows.append(dict(year=f"n_sel {lo}-{hi}", active_T=len(g), n_sel_mean=g["n_sel"].mean(),
                         target_invest_mean=g["target"].mean(), lot_invest_mean_1m=g["fill_1m"].mean(),
                         lot_invest_mean_10m=g["fill_10m"].mean(), zero_lot_share_1m=g["zero_1m"].mean(),
                         zero_lot_share_10m=g["zero_10m"].mean()))
    return pd.DataFrame(rows)


def signal_distribution(ev: pd.DataFrame) -> pd.DataFrame:
    """主信号值分布。对称同比在"由负转正"时恰为 +2(分子 = 分母×2),由正转负恰为 -2,
    所以 ±2 的堆积主要是扭亏/转亏,而不是 clip 截掉的极端增长。"""
    m = ev[ev["is_main"]].copy()
    m["year"] = m["V"].dt.year
    rows = []
    for y, g in list(m.groupby("year")) + [("全部", m)]:
        v = g["value"]
        top = g[v >= 2 - 1e-12]
        bot = g[v <= -2 + 1e-12]
        r = dict(year=y, n=len(g), mean=v.mean(), p10=v.quantile(.1), p50=v.median(), p90=v.quantile(.9),
                 share_plus2=float((v >= 2 - 1e-12).mean()), share_minus2=float((v <= -2 + 1e-12).mean()),
                 plus2_turnaround_share=float(((top["raw_prior"] < 0) & (top["raw_cur"] > 0)).mean()),
                 minus2_turn_to_loss_share=float(((bot["raw_prior"] > 0) & (bot["raw_cur"] < 0)).mean()),
                 share_cur_loss=float((g["raw_cur"] < 0).mean()))
        rows.append(r)
    return pd.DataFrame(rows)


def external_crosscheck(fc: pd.DataFrame, inc: pd.DataFrame) -> pd.DataFrame:
    """DESIGN_REVIEW 第 4 节两份巨潮原公告在本地数据里的对应记录(只比对日期与是否存在)。
    比亚迪 2023 年度预告:原文 PDF 落款 2024-01-29、巨潮归档路径 2024-01-30;
    保力新 2023 年度预告:原预告 2024-01-30(2024-008),修正 2024-04-25(2024-022)。"""
    sb = pd.read_csv(EE.STOCK_BASIC).drop_duplicates("ts_code").set_index("ts_code")
    rows = []
    for code, nm in [("002594.SZ", "比亚迪"), ("300116.SZ", "保力新")]:
        end = pd.Timestamp("2023-12-31")
        f = fc[(fc["stock_code"] == code) & (fc["end_date"] == end)]
        i = inc[(inc["stock_code"] == code) & (inc["end_date"] == end)]
        rows.append(dict(stock_code=code, name=nm, end_date="2023-12-31",
                         forecast_rows=len(f),
                         forecast_dates="|".join(sorted({str(d.date()) for d in f["P"]})),
                         forecast_indicators="|".join(sorted(set(f["预测指标"].astype(str)))),
                         income_ann_date=(str(i["ann_date"].iloc[0].date()) if len(i) else ""),
                         delist_date=str(sb["delist_date"].get(code, "")),
                         in_price_panel=None))
    return pd.DataFrame(rows)


def survivorship(ev: pd.DataFrame, codes: pd.Index) -> pd.DataFrame:
    """主事件中股票不在行情面板里的(多为已退市股):利润表含退市股,行情面板不含。"""
    m = ev[ev["is_main"]]
    out = m[~m["stock_code"].isin(codes)]
    sb = pd.read_csv(EE.STOCK_BASIC).drop_duplicates("ts_code").set_index("ts_code")
    dl = sb["delist_date"].reindex(out["stock_code"].unique())
    R = [dict(item="主事件总数", n=len(m)),
         dict(item="股票不在行情面板的主事件", n=len(out), denom=len(m)),
         dict(item="  涉及股票数", n=int(out["stock_code"].nunique())),
         dict(item="  其中 stock_basic 有退市日的股票数", n=int(dl.notna().sum()), denom=int(out["stock_code"].nunique())),
         dict(item="  其中北交所(.BJ)事件", n=int(out["stock_code"].str.endswith(".BJ").sum()), denom=len(out)),
         dict(item="主事件中北交所(.BJ)事件(合格池排除)", n=int(m["stock_code"].str.endswith(".BJ").sum()), denom=len(m))]
    for y, g in out.groupby(out["V"].dt.year):
        R.append(dict(item=f"  不在面板主事件 V 年={y}", n=len(g), denom=int((m["V"].dt.year == y).sum())))
    o = pd.DataFrame(R)
    o["share"] = o["n"] / o["denom"]
    return o


def main():
    cal = EE.trading_calendar()
    print("构建事件表 …")
    ev = EE.build_events(save=True)
    inc = EE.load_income()
    fc = EE.load_forecast()
    ex = EE.load_express()
    print(f"  {len(ev)} 行,主事件 {int(ev['is_main'].sum())}")

    coverage(ev).to_csv(OUT["event_coverage"], index=False)
    main_by_year(ev).to_csv(OUT["event_main_by_year"], index=False)
    v_distribution(ev).to_csv(OUT["event_v_distribution"], index=False)
    conflicts(ev, inc, fc, ex).to_csv(OUT["event_conflicts"], index=False)
    unit_checks(inc, fc, ex).to_csv(OUT["event_unit_checks"], index=False)
    st_coverage(cal).to_csv(OUT["st_coverage"], index=False)
    print("覆盖/冲突/口径/ST 已写出")

    codes = pd.Index(pd.read_parquet(ROOT / "cache" / "daily" / "close_px.parquet").columns)
    print("与现有面板对照 …")
    survivorship(ev, codes).to_csv(LEDGER / "pead_event_survivorship.csv", index=False)
    xc = external_crosscheck(fc, inc)
    xc["in_price_panel"] = xc["stock_code"].isin(codes)
    xc.to_csv(LEDGER / "pead_event_external_crosscheck.csv", index=False)
    signal_distribution(ev).to_csv(LEDGER / "pead_event_signal_distribution.csv", index=False)
    pc, pan, rep = panel_consistency(ev, inc, cal, codes)
    pc.to_csv(OUT["event_panel_consistency"], index=False)
    spotcheck(ev, pan, cal).to_csv(OUT["event_spotcheck"], index=False)
    del pan, rep
    print("事件池规模 …")
    df, summ, Q = pool_sizes(ev, cal)
    df.to_csv(OUT["event_pool_size"], index=False)
    summ.to_csv(OUT["event_pool_summary"], index=False)
    print("整手可行性 …")
    lot_feasibility(ev, cal, Q).to_csv(LEDGER / "pead_event_lot_feasibility.csv", index=False)
    del Q

    frozen = {
        "built_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "cutoff": str(EE.CUTOFF.date()),
        "seed": SEED,
        "events_parquet": str(EE.EVENTS_PATH.relative_to(ROOT)),
        "events_rows": int(len(ev)), "main_events": int(ev["is_main"].sum()),
        "md5": {str(pth.relative_to(ROOT)): md5(pth) for pth in [
            EE.INCOME_PKL, EE.FORECAST_RAW, EE.EXPRESS_RAW, EE.ST_CSV, EE.STOCK_BASIC,
            ROOT / "stock_sw_industry.csv", TMP / "comp_v5_score_open10.parquet",
            LEDGER / "comp_v5_members.json", EE.EVENTS_PATH]},
    }
    comp = pd.read_parquet(TMP / "comp_v5_score_open10.parquet")
    frozen["comp_v5_score"] = dict(shape=list(comp.shape), first=str(comp.index.min().date()),
                                   last=str(comp.index.max().date()),
                                   nonnull_share=float(np.isfinite(comp.to_numpy(np.float32)).mean()))
    man = json.load(open(LEDGER / "comp_v5_members.json"))
    from afactor import factors as FA
    FA.build_all()
    frozen["comp_v5_members_manifest"] = len(man["members"])
    frozen["comp_v5_members_effective"] = [n for n in man["members"] if n in FA.REGISTRY]
    frozen["comp_v5_members_missing_from_registry"] = [n for n in man["members"] if n not in FA.REGISTRY]
    frozen["main_signal_aliases"] = {n: (FA.REGISTRY[n][2] if n in FA.REGISTRY else None) for n in
                                     ["fu_op_yoy", "pead_sur_op", "lit2_pead_op_surprise_avail_lag_days0",
                                      "lit2_pead_op_surprise_avail_lag_days1"]}
    json.dump(frozen, open(LEDGER / "pead_frozen_inputs.json", "w"), ensure_ascii=False, indent=1)
    print("完成")


if __name__ == "__main__":
    main()
