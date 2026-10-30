"""PEAD 事件表:point-in-time 的(股票, 报告期, 指标, 来源, 版本)级盈利事件。

口径来源:research/EARNINGS_EVENT_RESEARCH_PROTOCOL.md(v0.2)第 3、4 节,
补充登记见 research/PEAD_PREREG_ADDENDUM.md,数据验收见 research/PEAD_DATA_ACCEPTANCE.md。
**本模块只做数据与定义,不读取、不计算任何前向收益。**

------------------------------------------------------------------------------
一、时钟(方案 v0.2,"P → V → T → S")
------------------------------------------------------------------------------
  P  有来源记录支持的公共披露日期,**以及信号全部依赖项**公共披露日期中的最晚者。
     数据只有日期、没有时分 -> 一律按 **P 日收盘后** 才公开处理(保守)。
  V  严格晚于 P 的第一个交易日(searchsorted side="right")。
       - P 为交易日:V = 下一交易日;
       - P 为周末/节假日:V = 其后第一个交易日。这是方案原文"严格晚于 P 的第一个交易日"
         的直接含义;周末公告在周一开盘前已公开,不再额外等一天(方案 v0.2 明确取消了
         v0.1 的重复 lag)。P 是否交易日单列 `P_is_trading_day`,只作诊断,**不得**看收益后改口径。
  T  不早于 V 的首个预定调仓日(由相位决定,本模块不定 T);S = T 日 09:25 信号截止;
     T 日 09:31-09:40 TWAP 执行(项目 open10 窗口)。
  事件池(T) = V ∈ [T-4, T](按交易日位置)的主事件。

  与项目回测引擎的索引换算(**最容易错的地方**):
    strategy.run_strategy 的"信号日 t"在 t+1 开盘成交,所以协议的 T = 引擎的 t+1。
    本模块 event_pool() 输出的 `T` 一律指**成交日**,另给 `t_signal` = T 的前一交易日,
    量价/排雷/股票池字段只许取 t_signal 收盘及以前。
    已有面板 FU._pit_daily 把可见日 eff=P(日历日)映射到"第一个 >= P 的面板行",
    对交易日 P 等价于本模块;对周末 P 会落在周一行 -> 周二成交,比本口径晚 1 个交易日。
    **禁止**把本表的 V 再当作公告日喂给 _pit_daily 或 newdata._event_panel(会重复 lag)。

------------------------------------------------------------------------------
二、主信号 fu_op_yoy 的实际定义(2026-10-10 核对)
------------------------------------------------------------------------------
  factors.py:  fu_op_yoy        = _fu(p, "operate_profit_yoy")       (fundamental 族)
               pead_sur_op      = _fu(p, "operate_profit_yoy")       (pead 族)
               lit2_pead_op_surprise_avail_lag_days{0,1}
                                = literature.fu("operate_profit_yoy") -> _FA._fu(...)
     三者同一函数、同一缓存键,数值逐位相同。lit2 的 avail_lag_days 只被注入公式命名空间,
     公式里没有引用它 -> 两个"滞后变体"完全相同(LONG_SIDE_TEST.md "滞后参数未生效"的根因)。
  fundamental.build_report_level:
     表   = data_8:21/stock_income_2015_now.pkl(tushare 利润表,report_type 全为 1 合并报表,
            每个(股票, 报告期)只有**一行**——数据源未保留修正史);
     字段 = operate_profit(营业利润,年内累计);
     单季 = Q1 取累计值,Q2-Q4 取 累计 - 上一季累计;
     同比 = 对称增长率 (x - x_ly) / ((|x| + |x_ly|)/2),分母 <= 1e-6 置 NaN,clip(-2, 2);
     可见日 avail_yoy = max(本季, 上季, 去年同季, 去年同季的上季) 的 avail,
            其中 avail = max(ann_date, f_ann_date),且是**三张表**(利润/现金流/资产负债)按
            (股票, 报告期) 合并后的最晚者;Q1 也把上季(去年 Q4)计入依赖。
  本模块的差异(按方案"依赖项"原则):
     - 只用利润表行作依赖(营业利润不依赖资产负债表/现金流量表的更正日期);
     - Q1 单季不依赖上季;
     - f_ann_date > ann_date 的行视为**修正版本**(tushare f_ann_date=实际公告日期,
       变更过的以变更后日期为准):我们只有修正后的数值,首次披露时的原值不可恢复 ->
       首次披露事件的信号记"不可恢复",修正后的数值单列为 version 2。
     数值公式与 build_report_level 完全相同,差异只在 P 与版本标注;一致性见验收报告。

------------------------------------------------------------------------------
三、封存(硬纪律)
------------------------------------------------------------------------------
  原始文件含 2026 行。读入后**立即**截断:
    - 利润表:ann_date > 2025-12-31 的行整行丢弃;ann_date <= 2025-12-31 < f_ann_date 的行
      (2026 年才出的修正版本)保留元数据、数值置 NaN,不进入任何计算;
    - 业绩预告/快报:公告日期 > 2025-12-31 的行丢弃;
    - ST 名单:trade_date > 2025-12-31 的行丢弃;
    - 交易日历只取 cache/daily(<= 2025-12-31)。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from afactor.config import DAILY, ROOT, TMP, RESEARCH_END

DATA = ROOT / "data_8:21"
INCOME_PKL = DATA / "stock_income_2015_now.pkl"
FORECAST_RAW = DATA / "source_raw" / "eastmoney_forecast_raw.csv"
EXPRESS_RAW = DATA / "source_raw" / "eastmoney_express_raw.csv"
ST_CSV = DATA / "st_info_latest.csv"
STOCK_BASIC = DATA / "stock_basic.csv"

CUTOFF = pd.Timestamp(RESEARCH_END)              # 2025-12-31,封存边界
PRE_PANEL_LAST_TD = pd.Timestamp("2019-12-31")   # 面板首日 2020-01-02 之前的最后一个交易日
EVENTS_PATH = TMP / "pead_events.parquet"
EVENT_WINDOW = 5                                  # 事件池:V ∈ [T-4, T]
FIRST_EVENT_P = pd.Timestamp("2019-01-01")       # 事件表只保留 P >= 此日的行(依赖项仍用全部历史)

# 主信号与辅助指标(定期报告)。key -> (利润表字段, 中文名)
FORMAL_ITEMS = {
    "op_yoy_q": ("operate_profit", "营业利润单季对称同比(主信号 fu_op_yoy)"),
    "np_yoy_q": ("n_income_attr_p", "归母净利润单季对称同比(辅助)"),
    "rev_yoy_q": ("revenue", "营业收入单季对称同比(辅助)"),
}
MAIN_INDICATOR = "op_yoy_q"

# 东财业绩预告 "预测指标" -> 指标键
FORECAST_INDICATORS = {
    "归属于上市公司股东的净利润": "fc_np_parent",
    "扣除非经常性损益后的净利润": "fc_np_deducted",
    "每股收益": "fc_eps",
    "营业收入": "fc_revenue",
    "非经常性损益": "fc_nonrecurring",
    "扣除后营业收入": "fc_revenue_deducted",
    "净利润": "fc_np",
    "主营业务收入": "fc_main_revenue",
    "扣非后每股收益": "fc_eps_deducted",
}

# 法定披露截止(月, 日, 年偏移):Q1 4/30、H1 8/31、Q3 10/31、年报次年 4/30
_DEADLINE = {1: (4, 30, 0), 2: (8, 31, 0), 3: (10, 31, 0), 4: (4, 30, 1)}

# 首次披露版本的信号状态(value_status)
STATUS_TEXT = {
    "ok": "可用",
    "own_revised_sealed": "首次披露值不可恢复:本行仅有 2026 年修正版本(封存,不读数值)",
    "own_revised_later": "首次披露值不可恢复:本行数值是后来修正版本(f_ann_date > ann_date)",
    "dep_available_later": "首次披露时依赖项(上季/去年同季)的现存版本尚未公开",
    "no_prev_quarter": "缺上一季累计值,无法还原单季",
    "no_prior_year": "缺去年同季(或其上季)记录,无法算同比",
    "missing_value": "依赖项字段为空",
    "zero_denominator": "对称同比分母为 0(本季与去年同季均为 0)",
    "sealed": "修正版本在封存期公开,不读数值",
}


# =============================================================================
# 交易日历与 V 映射
# =============================================================================
def trading_calendar() -> pd.DatetimeIndex:
    """研究期交易日历 = cache/daily/close_px.parquet 的行索引(2020-01-02 ~ 2025-12-31)。

    只读一列取索引,不读行情数值。"""
    probe = pd.read_parquet(DAILY / "close_px.parquet", columns=["000001.SZ"])
    cal = pd.DatetimeIndex(probe.index).sort_values()
    assert cal.max() <= CUTOFF, "交易日历越过封存边界"
    return cal


def map_v(P: pd.Series, cal: pd.DatetimeIndex) -> tuple[pd.Series, pd.Series]:
    """P(公开日期,按收盘后处理)-> V(严格晚于 P 的第一个交易日)。

    返回 (V, V_status)。V_status ∈ {ok, P早于面板, V在封存期, P缺失}。
    P 早于 2019-12-31 时,面板日历无法确定其后的第一个交易日(2019 年日历不在 cache 中),
    置 NaT —— 这些事件本来也落不进 2020 年以后的事件池。
    """
    P = pd.to_datetime(P)
    di = cal.to_numpy()
    pos = np.searchsorted(di, P.to_numpy(dtype="datetime64[ns]"), side="right")
    inside = pos < len(cal)
    V = pd.Series(np.where(inside, di[np.clip(pos, 0, len(cal) - 1)], np.datetime64("NaT")),
                  index=P.index, dtype="datetime64[ns]")
    status = pd.Series("ok", index=P.index, dtype=object)
    status[P.isna()] = "P缺失"
    pre = P < PRE_PANEL_LAST_TD
    V[pre] = pd.NaT
    status[pre & P.notna()] = "P早于面板"
    status[~inside & P.notna()] = "V在封存期"
    V[~inside] = pd.NaT
    return V, status


def _em_to_ts(code: pd.Series) -> pd.Series:
    """东财 6 位代码 -> tushare 代码。92/8/4 开头为北交所;6/9 开头为上交所(9 为 B 股 900);其余深交所。"""
    c = code.astype(str).str.zfill(6)
    suf = np.where(c.str.startswith("92") | c.str[0].isin(["8", "4"]), ".BJ",
                   np.where(c.str[0].isin(["6", "9"]), ".SH", ".SZ"))
    return c + suf


def _sym_yoy(x, prev):
    """对称同比,与 fundamental.build_report_level 逐位同式。"""
    x = np.asarray(x, np.float64)
    prev = np.asarray(prev, np.float64)
    den = (np.abs(x) + np.abs(prev)) / 2.0
    with np.errstate(invalid="ignore", divide="ignore"):
        v = np.where(den > 1e-6, (x - prev) / np.where(den > 1e-6, den, 1.0), np.nan)
    return np.clip(v, -2, 2)


def list_dates() -> pd.Series:
    """上市日(BaoStock stock_basic,含已退市);上市日是历史事实,用它不构成前视。"""
    sb = pd.read_csv(STOCK_BASIC).drop_duplicates("ts_code").set_index("ts_code")
    return pd.to_datetime(sb["list_date"].astype("Int64").astype(str), format="%Y%m%d",
                          errors="coerce")


# =============================================================================
# 定期报告(利润表)
# =============================================================================
def load_income(items=("operate_profit", "n_income_attr_p", "revenue")) -> pd.DataFrame:
    """读利润表并立即执行封存截断。返回每行一个(股票, 报告期)。

    列:stock_code, end_date, qi(季度序号), quarter, ann_date, f_ann_date, avail, sealed_value,
        comp_type, <items>。sealed_value=True 表示本行是 2026 年才公开的修正版本,数值已置 NaN。
    """
    cols = ["stock_code", "ann_date", "f_ann_date", "end_date", "comp_type", *items]
    d = pd.read_pickle(INCOME_PKL)[cols].copy()
    for c in ("ann_date", "f_ann_date", "end_date"):
        d[c] = pd.to_datetime(d[c], errors="coerce")
    d = d[d["ann_date"].notna() & d["end_date"].notna()]
    # ---- 封存截断:首次公告在 2026 的行整行丢弃 ----
    d = d[d["ann_date"] <= CUTOFF].copy()
    d["f_ann_date"] = d["f_ann_date"].fillna(d["ann_date"])
    d["avail"] = d[["ann_date", "f_ann_date"]].max(axis=1)
    d["sealed_value"] = d["avail"] > CUTOFF
    d.loc[d["sealed_value"], list(items)] = np.nan       # 2026 修正版本的数值不得进入计算
    assert not d.duplicated(["stock_code", "end_date"]).any(), "利润表出现重复(股票, 报告期)"
    d["qi"] = (d["end_date"].dt.year * 4 + (d["end_date"].dt.quarter - 1)).astype(int)
    d["quarter"] = d["end_date"].dt.quarter.astype(int)
    return d.reset_index(drop=True)


def _lagged(d: pd.DataFrame, k: int, cols: list[str], tag: str) -> pd.DataFrame:
    """取同一股票 qi-k 季度的行(按季度序号对齐,缺季即缺行,不会按"行"错位)。"""
    x = d[["stock_code", "qi", *cols]].copy()
    x["qi"] = x["qi"] + k
    x = x.rename(columns={c: f"{c}_{tag}" for c in cols})
    x[f"has_{tag}"] = True
    return x


def formal_signal(d: pd.DataFrame, item: str) -> pd.DataFrame:
    """在报告期维度计算某字段的单季对称同比,并给出每个依赖项的可见日。

    返回每个(股票, 报告期)一行:x_q(本季单季值)、x_q_ly(去年同季单季值)、value、
    P_first(=ann_date,首次披露日)、own_avail、dep_max_avail(全部依赖项 avail 的最大值)、
    以及缺失原因判定所需的标志位。
    """
    base_cols = ["avail", "ann_date", item, "sealed_value"]
    m = d[["stock_code", "end_date", "qi", "quarter", "ann_date", "f_ann_date", "avail",
           "sealed_value", item]].copy()
    for k, tag in [(1, "prev"), (4, "ly"), (5, "lyprev")]:
        m = m.merge(_lagged(d, k, base_cols, tag), on=["stock_code", "qi"], how="left")
        m[f"has_{tag}"] = m[f"has_{tag}"].eq(True)
        m[f"sealed_value_{tag}"] = m[f"sealed_value_{tag}"].eq(True)
    q1 = m["quarter"] == 1
    cur = m[item].to_numpy(np.float64)
    m["x_q"] = np.where(q1, cur, cur - m[f"{item}_prev"].to_numpy(np.float64))
    m["x_q_ly"] = np.where(q1, m[f"{item}_ly"].to_numpy(np.float64),
                           m[f"{item}_ly"].to_numpy(np.float64) - m[f"{item}_lyprev"].to_numpy(np.float64))
    m["value"] = _sym_yoy(m["x_q"], m["x_q_ly"])

    # ---- 依赖项可见日:Q1 = {本季, 去年同季};Q2-Q4 = {本季, 上季, 去年同季, 去年同季的上季} ----
    dep = pd.concat([m["avail"],
                     m["avail_prev"].where(~q1),
                     m["avail_ly"],
                     m["avail_lyprev"].where(~q1)], axis=1)
    m["dep_max_avail"] = dep.max(axis=1)
    m["P_first"] = m["ann_date"]
    m["own_avail"] = m["avail"]

    # ---- 首次披露版本的状态(优先级自上而下)----
    need_prev = ~q1
    row_missing_prev = need_prev & ~m["has_prev"]
    row_missing_ly = ~m["has_ly"] | (need_prev & ~m["has_lyprev"])
    sealed_any = (m["sealed_value"] | (need_prev & m["sealed_value_prev"]) | m["sealed_value_ly"]
                  | (need_prev & m["sealed_value_lyprev"]))
    val_nan = (pd.isna(m[item]) | (need_prev & pd.isna(m[f"{item}_prev"])) | pd.isna(m[f"{item}_ly"])
               | (need_prev & pd.isna(m[f"{item}_lyprev"])))
    later = m["dep_max_avail"] > m["P_first"]
    zero_den = (np.abs(m["x_q"]) + np.abs(m["x_q_ly"])) / 2.0 <= 1e-6

    st = pd.Series("ok", index=m.index, dtype=object)
    st[zero_den] = "zero_denominator"
    st[later & ~(m["own_avail"] > m["P_first"])] = "dep_available_later"
    st[later & (m["own_avail"] > m["P_first"])] = "own_revised_later"
    st[val_nan & ~sealed_any] = "missing_value"
    st[row_missing_ly] = "no_prior_year"
    st[row_missing_prev] = "no_prev_quarter"
    st[m["sealed_value"]] = "own_revised_sealed"
    m["status_v1"] = st
    # 修正/依赖项更新版本(version 2)只在"首次披露不可恢复、但现存版本可算"时产生
    m["has_v2"] = st.isin(["own_revised_later", "dep_available_later", "own_revised_sealed"])
    v2 = pd.Series(np.where(m["dep_max_avail"] > CUTOFF, "sealed",
                            np.where(zero_den, "zero_denominator", "ok")), index=m.index)
    v2[st == "own_revised_sealed"] = "sealed"
    m["status_v2"] = v2.where(m["has_v2"])
    return m


def _frontier_end(rows: pd.DataFrame, P: pd.Series, codes: pd.Series) -> pd.Series:
    """每个(股票, P):P 日及以前已首次公告的最新报告期(与 _pit_daily 的 frontier 同义)。"""
    r = rows[["stock_code", "ann_date", "end_date"]].sort_values(["stock_code", "ann_date", "end_date"])
    r["front"] = r.groupby("stock_code")["end_date"].cummax()
    r = r.drop_duplicates(["stock_code", "ann_date"], keep="last").sort_values("ann_date")
    q = pd.DataFrame({"stock_code": codes.to_numpy(), "P": pd.to_datetime(P).to_numpy(),
                      "_i": np.arange(len(P))}).sort_values("P")
    q = q[q["P"].notna()]
    out = pd.merge_asof(q, r[["stock_code", "ann_date", "front"]], left_on="P", right_on="ann_date",
                        by="stock_code", direction="backward")
    res = pd.Series(pd.NaT, index=range(len(P)), dtype="datetime64[ns]")
    res.loc[out["_i"].to_numpy()] = out["front"].to_numpy()
    res.index = P.index
    return res


def build_formal_events(d: pd.DataFrame, cal: pd.DatetimeIndex) -> pd.DataFrame:
    """定期报告事件行:每个(股票, 报告期, 指标)一个首次披露行(version 1),
    必要时再加一个修正/依赖项更新行(version 2)。"""
    out = []
    for key, (item, cn) in FORMAL_ITEMS.items():
        m = formal_signal(d, item)
        m = m[m["P_first"] >= FIRST_EVENT_P]
        common = ["stock_code", "end_date", "quarter", "ann_date", "f_ann_date", "own_avail",
                  "dep_max_avail", "x_q", "x_q_ly", "comp_type"]
        m = m.merge(d[["stock_code", "end_date", "comp_type"]], on=["stock_code", "end_date"], how="left")
        v1 = m[common].copy()
        v1["version"] = 1
        v1["version_type"] = "首次披露"
        v1["P"] = m["P_first"]
        v1["value_status"] = m["status_v1"].to_numpy()
        v1["value"] = np.where(m["status_v1"] == "ok", m["value"], np.nan)
        # 首次披露值不可恢复的行:原始数值列里是**后来**的版本,不能留在 version 1 行上被误用
        unrec = m["status_v1"].isin(["own_revised_later", "dep_available_later", "own_revised_sealed"]).to_numpy()
        v1.loc[unrec, ["x_q", "x_q_ly"]] = np.nan
        # 依赖项在 2026 才公开的修正版本:不生成 version 2 行(P 在封存期,连日期也不写入事件表)
        mv = m[m["has_v2"] & (m["dep_max_avail"] <= CUTOFF) & (m["status_v1"] != "own_revised_sealed")]
        v2 = mv[common].copy()
        v2["version"] = 2
        v2["version_type"] = np.where(mv["own_avail"] > mv["P_first"], "修正版本", "依赖项更新")
        v2["P"] = mv["dep_max_avail"]
        v2["value_status"] = mv["status_v2"].to_numpy()
        v2["value"] = np.where(mv["status_v2"] == "ok", mv["value"], np.nan)
        e = pd.concat([v1, v2], ignore_index=True)
        e["indicator"] = key
        e["indicator_cn"] = cn
        out.append(e)
    e = pd.concat(out, ignore_index=True)
    e["source"] = "定期报告"
    e = e.rename(columns={"quarter": "fiscal_q", "x_q": "raw_cur", "x_q_ly": "raw_prior",
                          "ann_date": "own_ann_date", "f_ann_date": "own_f_ann_date"})
    e["raw_basis"] = "单季(元)"
    # frontier:P 日及以前已首次公告的最新报告期 == 本行报告期
    fe = _frontier_end(d, e["P"], e["stock_code"])
    e["is_frontier"] = (e["end_date"] >= fe).fillna(False)
    # 法定截止
    y = e["end_date"].dt.year
    dl = [pd.Timestamp(yy + _DEADLINE[q][2], _DEADLINE[q][0], _DEADLINE[q][1])
          for yy, q in zip(y, e["fiscal_q"])]
    e["late_vs_deadline"] = e["own_ann_date"] > pd.DatetimeIndex(dl)
    return e


# =============================================================================
# 业绩预告 / 业绩快报(东方财富)
# =============================================================================
def load_forecast() -> pd.DataFrame:
    """业绩预告原表(含"预测指标"列),截断到 <= 2025-12-31,去掉逐字段完全相同的重复行。

    数据事实(2026-10-10 核对):每个(股票, 报告期, 指标)只有**一个**公告日期 ——
    东财只保留一个版本,修正预告无法与原预告区分;本表一律标"单一版本"。
    "预测数值"是区间中值(元;EPS 为元/股);"上年同期值"为累计期口径。
    """
    r = pd.read_csv(FORECAST_RAW, dtype={"股票代码": str})
    r["P"] = pd.to_datetime(r["公告日期"], errors="coerce")
    r = r[r["P"].notna() & (r["P"] <= CUTOFF)].copy()            # 封存截断
    n0 = len(r)
    r = r.drop_duplicates([c for c in r.columns if c != "序号"])
    r.attrs["n_exact_dup_dropped"] = n0 - len(r)
    r["stock_code"] = _em_to_ts(r["股票代码"])
    r["end_date"] = pd.to_datetime(r["报告日期"].astype(str), format="%Y%m%d")
    r["indicator"] = r["预测指标"].map(FORECAST_INDICATORS).fillna("fc_other")
    return r


def load_express() -> pd.DataFrame:
    """业绩快报原表,截断到 <= 2025-12-31。数值为累计期口径(元)。

    数据事实:东财快报"公告日期"与报告期末的中位间隔 392 天,大量记录晚于正式财报 ——
    该日期多半是数据更新日而不是快报首次公开日,不能当作可交易时点(方案 [S6])。
    """
    r = pd.read_csv(EXPRESS_RAW, dtype={"股票代码": str})
    r["P"] = pd.to_datetime(r["公告日期"], errors="coerce")
    r = r[r["P"].notna() & (r["P"] <= CUTOFF)].copy()            # 封存截断
    r["stock_code"] = _em_to_ts(r["股票代码"])
    r["end_date"] = pd.to_datetime(r["报告日期"].astype(str), format="%Y%m%d")
    return r


def build_forecast_events(fc: pd.DataFrame) -> pd.DataFrame:
    e = pd.DataFrame({
        "stock_code": fc["stock_code"].to_numpy(),
        "end_date": fc["end_date"].to_numpy(),
        "fiscal_q": fc["end_date"].dt.quarter.to_numpy(),
        "indicator": fc["indicator"].to_numpy(),
        "indicator_cn": ("业绩预告:" + fc["预测指标"].astype(str)).to_numpy(),
        "source": "业绩预告",
        "version": 1,
        "version_type": "单一版本(源不保留修正史)",
        "P": fc["P"].to_numpy(),
        "raw_cur": fc["预测数值"].to_numpy(np.float64),
        "raw_prior": fc["上年同期值"].to_numpy(np.float64),
        "raw_basis": "累计期(元;EPS 元/股),区间中值",
        "fc_change_pct": fc["业绩变动幅度"].to_numpy(np.float64),
        "fc_type": fc["预告类型"].astype(str).to_numpy(),
    })
    e["value"] = _sym_yoy(e["raw_cur"], e["raw_prior"])
    e["value_status"] = np.where(np.isfinite(e["value"]), "ok", "missing_value")
    e["value"] = np.where(e["value_status"] == "ok", e["value"], np.nan)
    return e


def build_express_events(ex: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for key, cur, prior, cn in [("ex_np_parent", "净利润-净利润", "净利润-去年同期", "业绩快报:净利润"),
                                ("ex_revenue", "营业收入-营业收入", "营业收入-去年同期", "业绩快报:营业收入")]:
        e = pd.DataFrame({
            "stock_code": ex["stock_code"].to_numpy(),
            "end_date": ex["end_date"].to_numpy(),
            "fiscal_q": ex["end_date"].dt.quarter.to_numpy(),
            "indicator": key, "indicator_cn": cn, "source": "业绩快报",
            "version": 1, "version_type": "单一版本(源不保留修正史)",
            "P": ex["P"].to_numpy(),
            "raw_cur": ex[cur].to_numpy(np.float64),
            "raw_prior": ex[prior].to_numpy(np.float64),
            "raw_basis": "累计期(元)",
        })
        e["value"] = _sym_yoy(e["raw_cur"], e["raw_prior"])
        e["value_status"] = np.where(np.isfinite(e["value"]), "ok", "missing_value")
        e["value"] = np.where(e["value_status"] == "ok", e["value"], np.nan)
        parts.append(e)
    return pd.concat(parts, ignore_index=True)


# =============================================================================
# 新鲜度与日期核对
# =============================================================================
def attach_freshness(ev: pd.DataFrame, fc: pd.DataFrame, ex: pd.DataFrame,
                     formal_first: pd.DataFrame) -> pd.DataFrame:
    """给定期报告行标"新鲜度",给快报/预告行标"日期核对"。

    定期报告(按本行 P 比较,同一(股票, 报告期)):
      预告在先     存在任一指标的业绩预告,其公告日 < P
      快报在先     无在先预告,但存在快报且快报日期 < P
      在先未知     无在先预告;存在快报但其记录日期 >= P(东财快报日期不可信,不能据此判"首次")
      无在先记录   预告/快报表里都没有该(股票, 报告期)的更早记录(**不等于**"首次意外":
                   数据源缺已退市股、东财只留单一版本,可能漏记)
    快报 ex_date_check:早于正式财报 / 与正式财报同日 / 晚于正式财报(日期不可信) / 无正式财报记录;
                       另标 pre_listing(快报日期或报告期早于上市日 -> 招股书回填数据)。
    预告 fc_date_check:早于正式财报 / 与正式财报同日 / 晚于正式财报(异常) / 无正式财报记录。
    """
    key = ["stock_code", "end_date"]
    fcmin = (fc.groupby(key)
             .agg(fc_first_P=("P", "min"),
                  fc_indicators=("indicator", lambda s: "|".join(sorted(set(s)))))
             .reset_index())
    exP = ex[key + ["P"]].rename(columns={"P": "ex_P"})
    ev = ev.merge(fcmin, on=key, how="left").merge(exP, on=key, how="left")
    ev = ev.merge(formal_first, on=key, how="left")      # formal_P_first

    is_formal = ev["source"] == "定期报告"
    fresh = pd.Series(pd.NA, index=ev.index, dtype=object)
    fc_prior = ev["fc_first_P"] < ev["P"]
    ex_prior = ev["ex_P"] < ev["P"]
    ex_after = ev["ex_P"].notna() & ~ex_prior
    fresh[is_formal] = "无在先记录"
    fresh[is_formal & ex_after] = "在先未知(快报日期不早于正式财报)"
    fresh[is_formal & ex_prior] = "快报在先"
    fresh[is_formal & fc_prior] = "预告在先"
    ev["freshness"] = fresh
    ev["fc_same_day"] = is_formal & (ev["fc_first_P"] == ev["P"])

    def _chk(P, F):
        c = pd.Series("无正式财报记录", index=P.index, dtype=object)
        c[F.notna() & (P < F)] = "早于正式财报"
        c[F.notna() & (P == F)] = "与正式财报同日"
        c[F.notna() & (P > F)] = "晚于正式财报"
        return c
    ev["date_check"] = pd.Series(pd.NA, index=ev.index, dtype=object)
    for src in ("业绩预告", "业绩快报"):
        msk = ev["source"] == src
        ev.loc[msk, "date_check"] = _chk(ev.loc[msk, "P"], ev.loc[msk, "formal_P_first"]).to_numpy()
    ld = list_dates()
    ev["list_date"] = ev["stock_code"].map(ld)
    # pre_listing:公开日期早于上市日(招股书/上市公告书回填);period_pre_listing:报告期早于上市日
    ev["pre_listing"] = (ev["P"] < ev["list_date"]).fillna(False).astype(bool)
    ev["period_pre_listing"] = (ev["end_date"] < ev["list_date"]).fillna(False).astype(bool)
    return ev


# =============================================================================
# 总装
# =============================================================================
EVENT_COLUMNS = [
    "stock_code", "end_date", "fiscal_q", "indicator", "indicator_cn", "source", "version",
    "version_type", "P", "P_is_trading_day", "V", "V_status", "value", "value_status",
    "raw_cur", "raw_prior", "raw_basis", "own_ann_date", "own_f_ann_date", "own_avail",
    "dep_max_avail", "is_frontier", "late_vs_deadline", "pre_listing", "period_pre_listing",
    "list_date",
    "freshness", "fc_first_P", "fc_indicators", "fc_same_day", "ex_P", "formal_P_first",
    "date_check", "fc_change_pct", "fc_type", "comp_type", "is_main",
]


def build_events(save: bool = True) -> pd.DataFrame:
    """构建完整事件表并落盘 cache/_tmp/pead_events.parquet。"""
    cal = trading_calendar()
    inc = load_income()
    formal = build_formal_events(inc, cal)
    fc = load_forecast()
    ex = load_express()
    ev = pd.concat([formal, build_forecast_events(fc), build_express_events(ex)], ignore_index=True)
    formal_first = (inc[["stock_code", "end_date", "ann_date"]]
                    .rename(columns={"ann_date": "formal_P_first"}))
    ev = attach_freshness(ev, fc, ex, formal_first)
    ev["V"], ev["V_status"] = map_v(ev["P"], cal)
    ev["P_is_trading_day"] = pd.array(np.where(ev["P"] < PRE_PANEL_LAST_TD, None, ev["P"].isin(cal)),
                                      dtype="boolean")
    # 主事件:定期报告 / 主指标 / 首次披露 / 信号可用 / P 日为最新报告期 / V 在研究期内
    ev["is_main"] = ((ev["source"] == "定期报告") & (ev["indicator"] == MAIN_INDICATOR)
                     & (ev["version"] == 1) & (ev["value_status"] == "ok")
                     & ev["is_frontier"].eq(True) & ev["V"].notna())
    for c in EVENT_COLUMNS:
        if c not in ev.columns:
            ev[c] = pd.NA
    ev = ev[EVENT_COLUMNS].sort_values(["source", "indicator", "stock_code", "end_date", "version"])
    ev = ev.reset_index(drop=True)
    for c in ("P", "V", "end_date", "own_ann_date", "own_f_ann_date", "own_avail", "dep_max_avail",
              "list_date", "fc_first_P", "ex_P", "formal_P_first"):
        ev[c] = pd.to_datetime(ev[c])
    for c in ("is_frontier", "late_vs_deadline", "fc_same_day"):
        ev[c] = ev[c].astype("boolean")
    ev["comp_type"] = ev["comp_type"].astype("string")
    ev["fc_type"] = ev["fc_type"].astype("string")
    assert (ev["P"].dropna() <= CUTOFF).all(), "事件表含 2026 年公开的行"
    assert not ev.loc[ev["value"].notna(), "dep_max_avail"].gt(CUTOFF).any(), "信号用了封存期数值"
    if save:
        ev.to_parquet(EVENTS_PATH, index=False)
    return ev


def load_events(path=EVENTS_PATH) -> pd.DataFrame:
    """读取已落盘的事件表(列见 EVENT_COLUMNS 与 research/PEAD_DATA_ACCEPTANCE.md)。"""
    return pd.read_parquet(path)


def main_events(ev: pd.DataFrame | None = None) -> pd.DataFrame:
    """主事件臂可用的事件(is_main=True)。"""
    ev = load_events() if ev is None else ev
    return ev[ev["is_main"].fillna(False).astype(bool)].copy()


def event_pool(ev_main: pd.DataFrame, cal: pd.DatetimeIndex | None = None,
               window: int = EVENT_WINDOW) -> pd.DataFrame:
    """把主事件展开为"成交日 T 的事件池":V ∈ [T-(window-1), T](交易日位置)。

    返回长表:T(成交日)、t_signal(T 的前一交易日 = 引擎信号行)、stock_code、end_date、
    value、P、V、freshness、days_since_V(0..window-1)。
    同一 T 同一股票若有多个主事件(如两个报告期在 5 日内先后首次披露),保留报告期最新者,
    再按 P 最晚者 —— 与 _pit_daily 的 frontier 取法一致。
    **不含股票池过滤**:ADV/上市天数/ST 由 qualified_pool() 在 t_signal 行给出。
    """
    cal = trading_calendar() if cal is None else cal
    di = cal.to_numpy()
    e = ev_main[ev_main["V"].notna()].copy()
    iv = np.searchsorted(di, e["V"].to_numpy(dtype="datetime64[ns]"))
    rows = []
    for k in range(window):
        it = iv + k
        ok = it < len(cal)
        x = e.loc[ok, ["stock_code", "end_date", "value", "P", "V", "freshness", "fiscal_q"]].copy()
        x["T"] = di[it[ok]]
        x["days_since_V"] = k
        rows.append(x)
    pool = pd.concat(rows, ignore_index=True)
    pool = (pool.sort_values(["T", "stock_code", "end_date", "P"])
            .drop_duplicates(["T", "stock_code"], keep="last"))
    pos = np.searchsorted(di, pool["T"].to_numpy(dtype="datetime64[ns]"))
    pool["t_signal"] = np.where(pos > 0, di[np.clip(pos - 1, 0, len(di) - 1)], np.datetime64("NaT"))
    pool["t_signal"] = pd.to_datetime(pool["t_signal"])
    return pool.reset_index(drop=True)


# =============================================================================
# ST 与合格股票池(只用 t_signal 收盘及以前的信息)
# =============================================================================
def st_daily_pit(dates: pd.DatetimeIndex, codes: pd.Index) -> pd.DataFrame:
    """逐日 ST 标记(风险警示板名单)。名单文件是**逐日快照**(2016-08-09 起),
    可直接给出历史 ST 区间;缺失的名单日按前一名单日填充(与 fundamental.build_st_panel 同法,
    但先截断到 <= 2025-12-31)。名单首日之前一律 False。"""
    st = pd.read_csv(ST_CSV, usecols=["stock_code", "trade_date"])
    st["trade_date"] = pd.to_datetime(st["trade_date"])
    st = st[st["trade_date"] <= CUTOFF]                      # 封存截断
    st = st[st["stock_code"].isin(codes)]
    m = pd.crosstab(st["trade_date"], st["stock_code"]) > 0
    m = m.reindex(columns=codes, fill_value=False)
    first = m.index.min()
    m = m.reindex(m.index.union(dates)).astype(float).ffill().reindex(dates)
    m.loc[dates < first] = 0.0
    return m.fillna(0.0) > 0.5


def st_spells() -> pd.DataFrame:
    """由逐日名单还原 ST 区间(连续出现在名单上的名单日为一段;名单缺日不打断区间)。"""
    st = pd.read_csv(ST_CSV, usecols=["stock_code", "trade_date"])
    st["trade_date"] = pd.to_datetime(st["trade_date"])
    st = st[st["trade_date"] <= CUTOFF]
    days = np.sort(st["trade_date"].unique())
    pos = {d: i for i, d in enumerate(days)}
    st["k"] = st["trade_date"].map(pos)
    st = st.sort_values(["stock_code", "k"])
    st["brk"] = st.groupby("stock_code")["k"].diff().ne(1)
    st["spell"] = st.groupby("stock_code")["brk"].cumsum()
    return (st.groupby(["stock_code", "spell"])["trade_date"].agg(["min", "max", "count"])
            .reset_index().rename(columns={"min": "start", "max": "end", "count": "n_list_days"}))


def listed_trading_days(dates: pd.DatetimeIndex, codes: pd.Index,
                        first_valid: pd.Series | None = None) -> pd.DataFrame:
    """截至每个交易日(含当日)自上市日起的交易日数。

    上市日取 stock_basic;2020-01-02 之前的部分用 numpy 工作日数近似(不扣节假日,
    只会让 2020 年 1-3 月的老股提前几天达标 —— 2020 年是预热期,不进入任何窗口统计)。
    stock_basic 缺上市日者用面板内首个有成交日代替(保守:2020 年前上市者会被晚放行)。
    """
    ld = list_dates().reindex(codes)
    if first_valid is not None:
        ld = ld.fillna(first_valid.reindex(codes))
    di = dates.to_numpy()
    start_pos = np.searchsorted(di, ld.to_numpy(dtype="datetime64[ns]"), side="left").astype(float)
    pre = ld < dates[0]
    extra = np.zeros(len(codes))
    if pre.any():
        extra[pre.to_numpy()] = np.busday_count(
            ld[pre].to_numpy(dtype="datetime64[D]"), np.datetime64(dates[0].date(), "D"))
    start_pos[pre.to_numpy()] = 0.0
    start_pos[ld.isna().to_numpy()] = np.nan
    t = np.arange(len(dates), dtype=float)[:, None]
    n = t - start_pos[None, :] + 1.0 + extra[None, :]
    n[n < 0] = 0.0
    return pd.DataFrame(n, index=dates, columns=codes)


def qualified_pool(p, min_adv: float = 2e7, min_list_td: int = 60) -> pd.DataFrame:
    """方案 4.1 的主比较股票池,按**引擎信号行 t(= 协议 T-1)**给出,只用 t 收盘及以前的信息:
      当日有成交(valid) & ADV20 >= 2000 万 & 上市满 60 个交易日 & 当日不在 ST 名单
      & 沪深普通股(排除 .BJ)。
    100 万与 1000 万两档资金用同一个 ADV 门槛(方案第 6 节),避免把换池与容量混在一起。
    """
    dates, codes = p.dates, p.codes
    valid = p["valid"]
    first_valid = valid.idxmax().where(valid.any())
    ltd = listed_trading_days(dates, codes, first_valid)
    st = st_daily_pit(dates, codes)
    shsz = pd.Series([c.endswith((".SH", ".SZ")) for c in codes], index=codes)
    ok = (valid & (p["adv20"] >= min_adv) & (ltd >= min_list_td) & ~st).fillna(False)
    return ok & shsz.to_numpy()[None, :]
