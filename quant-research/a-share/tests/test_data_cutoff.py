"""数据截止日纪律(前缀 fixb,2026-10-10):默认截止日下各加载器读不到 2026 行;
set_data_cutoff 上下文内能读到,退出后恢复;放宽只允许前向检验 / 测试代码;面板日期越过截止日直接报错。

读真实 data_8:21(daily_basic 600MB,两次读入约 1 分钟):
    python3 -m pytest tests/test_data_cutoff.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from afactor import config as C, newdata as ND, fundamental as FU   # noqa: E402

pytestmark = pytest.mark.skipif(not (ROOT / "data_8:21" / "daily_basic.csv").exists(),
                                reason="需要 data_8:21 原始数据")

DEFAULT = pd.Timestamp(C.RESEARCH_END)
FWD = pd.Timestamp("2026-08-31")          # 源文件最晚到 2026-08-22,放宽到月末即可读到全部 2026 行
PKLS = ["stock_income_2015_now.pkl", "stock_cashflow_2015_now.pkl", "stock_balancesheet_2015_now.pkl"]


def _max_dates() -> dict[str, pd.Timestamp]:
    """各加载器输出里最晚的日期(一律走模块自己的加载器,不直接读文件)。"""
    out = {"daily_basic": ND._load_daily_basic()["trade_date"].max()}
    for name, col in [("forecast.csv", "ann_date"), ("express.csv", "ann_date"), ("top_list.csv", "trade_date")]:
        out[name] = pd.to_datetime(str(int(ND._read_dated_csv(name, col)[col].max())), format="%Y%m%d")
    sb = ND._stock_basic()
    out["stock_basic.list_date"] = pd.to_datetime(str(int(sb["list_date"].max())), format="%Y%m%d")
    out["stock_basic.delist_date"] = pd.to_datetime(str(int(sb["delist_date"].max())), format="%Y%m%d")
    li = ND.listing_info(pd.Index(sb["ts_code"]))
    out["listing_info"] = max(li["list_date"].max(), li["delist_date"].max())
    out["index_daily"] = ND.index_daily().index.max() if len(ND.index_daily()) else DEFAULT
    out["st_info"] = FU._read_st()["trade_date"].max()
    for f in PKLS:
        out[f] = FU._load(f, ["total_share"] if "balance" in f else [])["avail"].max()
    rep = FU.build_report_level()
    out["report_level.avail*"] = max(rep[c].max() for c in rep.columns if c.startswith("avail"))
    return out


@pytest.fixture(scope="module")
def default_max():
    assert C.data_cutoff() == DEFAULT
    return _max_dates()


@pytest.fixture(scope="module")
def fwd_max(default_max):
    with C.set_data_cutoff(FWD, reason="单元测试:放宽后应能读到 2026 行") as cut:
        assert cut == FWD and C.data_cutoff() == FWD
        out = _max_dates()
    return out


def test_default_cutoff_is_research_end() -> None:
    assert C.DATA_CUTOFF == C.RESEARCH_END
    assert C.data_cutoff() == DEFAULT


def test_default_loaders_have_no_2026_rows(default_max) -> None:
    bad = {k: v for k, v in default_max.items() if pd.notna(v) and v > DEFAULT}
    assert not bad, f"默认截止日下仍读到 2026 行: {bad}"


def test_context_reads_2026_then_restores(default_max, fwd_max) -> None:
    # 有 2026 数据的源在上下文内必须读到 2026 行(指数快照到 2026-07、stock_basic 退市日到 2026-07)
    for k in ["daily_basic", "forecast.csv", "express.csv", "top_list.csv", "stock_basic.list_date",
              "stock_basic.delist_date", "index_daily", "st_info", *PKLS, "report_level.avail*"]:
        assert fwd_max[k] > DEFAULT, f"{k}: 放宽截止日后仍读不到 2026 行({fwd_max[k]})"
        assert fwd_max[k] <= FWD
    # 退出后恢复:截止日、加载器输出都回到默认
    assert C.data_cutoff() == DEFAULT
    again = _max_dates()
    assert again == default_max


def test_caches_are_keyed_by_cutoff() -> None:
    a = ND._load_daily_basic()
    with C.set_data_cutoff("2025-06-30", reason="单元测试:收紧截止日"):
        b = ND._load_daily_basic()
        assert b["trade_date"].max() <= pd.Timestamp("2025-06-30")
        rep = FU.build_report_level()
        assert rep["avail"].max() <= pd.Timestamp("2025-06-30")
    c = ND._load_daily_basic()
    assert c["trade_date"].max() == a["trade_date"].max()


def test_nested_contexts_restore_in_order() -> None:
    with C.set_data_cutoff("2026-03-31", reason="单元测试:外层"):
        with C.set_data_cutoff("2025-06-30", reason="单元测试:内层收紧"):
            assert C.data_cutoff() == pd.Timestamp("2025-06-30")
        assert C.data_cutoff() == pd.Timestamp("2026-03-31")
    assert C.data_cutoff() == DEFAULT


def test_loosening_requires_fwd_or_test_caller() -> None:
    """研究代码(文件名不以 fwd / test_ 开头)不能放宽截止日;收紧任何代码都可以;原因必填。"""
    ns = {"C": C}
    research = compile("C.set_data_cutoff('2026-09-30', reason='研究脚本偷读')", "scripts/run_research.py", "exec")
    with pytest.raises(PermissionError):
        exec(research, ns)
    fwd = compile("x = C.set_data_cutoff('2026-09-30', reason='前向检验')", "scripts/fwd_eval.py", "exec")
    exec(fwd, ns)                                     # 构造即检查调用者;fwd 前缀放行
    tighten = compile("y = C.set_data_cutoff('2024-12-31', reason='收紧检查')", "scripts/run_research.py", "exec")
    exec(tighten, ns)
    with pytest.raises(ValueError):
        C.set_data_cutoff("2026-09-30", reason="")
    assert C.data_cutoff() == DEFAULT


def test_panel_dates_beyond_cutoff_raise() -> None:
    """面板日期越过截止日 -> 报错,而不是把 2026 的行静默 ffill 成 2025 的旧值。"""
    dates = pd.bdate_range("2025-12-01", "2026-01-30")
    codes = pd.Index(["000001.SZ", "600519.SH"])
    rep = FU.build_report_level()
    with pytest.raises(PermissionError):
        FU._pit_daily(rep, "total_share", dates, codes)
    with pytest.raises(PermissionError):
        ND.total_mv(dates, codes)
    with pytest.raises(PermissionError):
        FU.build_st_panel(dates, codes)
    with pytest.raises(PermissionError):
        ND.toplist_panels(dates, codes)
    ok = FU._pit_daily(rep, "total_share", dates[dates <= DEFAULT], codes)
    assert ok.notna().all().all()
    with C.set_data_cutoff(FWD, reason="单元测试:前向面板在上下文内可算"):
        fwd = ND.total_mv(dates, codes)
        assert fwd.loc["2026-01-05":].notna().all().all()


def test_report_level_pit_versions() -> None:
    """报告表:最终快照每个 (股票, 报告期) 一行;早期版本行的可见日不晚于同期最终行。"""
    rep = FU.build_report_level()
    fin = rep[rep["pit_vintage"] == FU.PIT_FINAL]
    assert not fin.duplicated(["stock_code", "qi"]).any()
    early = rep[rep["pit_vintage"] != FU.PIT_FINAL]
    assert len(early) > 0
    m = early.merge(fin[["stock_code", "qi", "avail"]], on=["stock_code", "qi"], suffixes=("", "_fin"))
    ok = m["avail"].notna() & m["avail_fin"].notna()
    assert (m.loc[ok, "avail"] <= m.loc[ok, "avail_fin"]).all()
    # 003031.SZ 2022 年报资产负债表两版:2023-03-31 首发(总资产 17.79 亿)、2024-04-26 追溯重述(43.42 亿)。
    # 旧口径只留最新版,2022Q4 要到 2024-04-26 才可见;PIT 口径在 [2023-03-31, 2023Q1 报告) 用首发版,
    # 且首发日之前仍是 2022Q3 的数 —— 不会提前用到任何一版。
    dates = pd.bdate_range("2023-03-20", "2023-05-10")
    ta = FU._pit_daily(rep, "total_assets", dates, pd.Index(["003031.SZ"]))["003031.SZ"]
    assert ta.loc["2023-03-31"] == pytest.approx(1.779431e9, rel=1e-6)
    assert ta.loc["2023-03-30"] == pytest.approx(1.884252e9, rel=1e-6)
    assert (ta < 4e9).all()                       # 2024 年的重述值不会出现在 2023 年
    # 000027.SZ 2023H1 首发版股本为 0(占位):股本不因此变成缺失
    sh = FU._pit_daily(rep, "total_share", pd.bdate_range("2023-07-20", "2023-09-05"), pd.Index(["000027.SZ"]))
    assert sh.notna().all().all()


def test_wrapped_loader_keeps_legacy_cache_key() -> None:
    """index_proxy / pead 脚本会包装 FU._load 再截一次;包装期间的结果只进旧键 "rep",不污染正常路径。"""
    orig = FU._load
    FU._REP_CACHE.pop("rep", None)

    def _wrapped(name, items):
        x = orig(name, items)
        return x[x["avail"] <= pd.Timestamp("2024-12-31")]

    FU._load = _wrapped
    try:
        r = FU.build_report_level()
        assert r["avail"].max() <= pd.Timestamp("2024-12-31")
        assert "rep" in FU._REP_CACHE
    finally:
        FU._load = orig
        FU._REP_CACHE.pop("rep", None)
    assert FU.build_report_level()["avail"].max() > pd.Timestamp("2025-06-30")


# ============================================================ 第二轮审查(fixb 修复)
def test_source_coverage_guard_simulated_in_research_period() -> None:
    """P1 可证伪:模拟"ST 名单只到 2025-09-30、请求到 2025-12-31"(不读 2026 数据)。
    修复前 build_st_panel 0 条警告、静默 ffill;修复后默认报 SourceCoverageError,显式接受后只发警告。"""
    import warnings
    dates = pd.bdate_range("2025-09-01", "2025-12-31")
    codes = pd.Index(["000001.SZ", "600519.SH", "000004.SZ"])
    true = FU.build_st_panel(dates, codes)                       # 研究期:源覆盖到 2026,守卫不触发
    orig, saved = FU._read_st, dict(C._SOURCE_END)

    def _short():
        x = orig()
        C.record_source_end("st_info_latest.csv", pd.Timestamp("2025-09-30"))
        return x[x["trade_date"] <= "2025-09-30"]
    FU._read_st = _short
    try:
        with pytest.raises(C.SourceCoverageError):
            FU.build_st_panel(dates, codes)
        with pytest.raises(C.SourceCoverageError):               # 覆盖内的请求不受影响,覆盖外的才报错
            FU.build_st_panel(dates[dates >= "2025-09-15"], codes)
        ok = FU.build_st_panel(dates[dates <= "2025-09-30"], codes)
        assert ok.equals(true.loc[:"2025-09-30"])
        with C.accept_source_gap("st_info_latest.csv", reason="单元测试:显式接受 ST 缺口"):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                stale = FU.build_st_panel(dates, codes)
        assert any("st_info_latest.csv" in str(x.message) for x in w)
        assert stale.shape == true.shape
    finally:
        FU._read_st = orig
        C._SOURCE_END.clear(); C._SOURCE_END.update(saved)


def test_source_coverage_guard_forward_context() -> None:
    """P1 真实场景:截止日放宽到 2026-10-09 后,ST(源到 07-03)/ 龙虎榜(08-21)/ 财报(资产负债表 06-13)
    的面板请求到 10-09 必须报错;接受后:ST 沿用旧状态、龙虎榜覆盖之外置 NaN、财报沿用最后一期。不计算任何收益。"""
    import warnings
    dates = pd.bdate_range("2026-06-01", "2026-10-09")
    codes = pd.Index(["000001.SZ", "600519.SH"])
    with C.set_data_cutoff("2026-10-09", reason="单元测试:源覆盖守卫(只看面板形状,不算收益)"):
        with pytest.raises(C.SourceCoverageError):
            FU.build_st_panel(dates, codes)
        with pytest.raises(C.SourceCoverageError):
            ND.toplist_panels(dates, codes)
        rep = FU.build_report_level()
        with pytest.raises(C.SourceCoverageError):
            FU._pit_daily(rep, "total_share", dates, codes)
        with pytest.raises(C.SourceCoverageError):
            ND.surprise_panels(dates, codes)
        # 覆盖内的请求照常
        FU.build_st_panel(dates[dates <= C.source_end("st_info_latest.csv")], codes)
        with C.accept_source_gap(["st_info_latest.csv", "top_list.csv", *FU.REPORT_TABLES],
                                 reason="单元测试:显式接受缺口"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                st = FU.build_st_panel(dates, codes)
                tl = ND.toplist_panels(dates, codes)
                sh = FU._pit_daily(rep, "total_share", dates, codes)
        assert st.shape == (len(dates), len(codes))
        end = C.source_end("top_list.csv")
        late = dates[dates > end + pd.Timedelta(days=7)]
        assert tl["tl_on_list"].loc[late].isna().all().all()          # 不再 fillna(0) 冒充"未上榜"
        assert tl["tl_count20"].loc[late].isna().all().all()
        assert sh.loc["2026-10-09"].notna().all()                       # 财报:沿用最后一期
    assert C.data_cutoff() == DEFAULT


def test_legacy_daily_basic_injection_is_effective() -> None:
    """P2:默认截止日下加载器读旧键 "db",旧脚本的注入与断言真正生效;注入表含截止日之后的行时加载器再截一次。"""
    saved = {k: ND._CACHE.pop(k) for k in [k for k in ND._CACHE
                                           if k == "db" or (isinstance(k, tuple) and k[0] == "db")]}
    try:
        fake = pd.DataFrame({"ts_code": ["000001.SZ"] * 3, "close": [1.0, 2.0, 3.0],
                             "trade_date": pd.to_datetime(["2025-12-30", "2025-12-31", "2026-01-05"]),
                             "turnover_rate": 1.0, "turnover_rate_f": 1.0, "float_share": 1.0, "circ_mv": 1.0})
        ND._CACHE["db"] = fake
        got = ND._load_daily_basic()
        assert got["trade_date"].max() == DEFAULT and len(got) == 2
        assert ND._CACHE["db"] is got                                   # 写回旧键:旧脚本的断言检查的就是加载器用的这份
        ok = fake.iloc[:2].copy()
        ND._CACHE["db"] = ok
        assert ND._load_daily_basic() is ok                            # 已截断的注入原样使用,不再另读一份
        assert ND.raw_close(pd.DatetimeIndex(["2025-12-31"]), pd.Index(["000001.SZ"])).iloc[0, 0] == 2.0
    finally:
        ND._CACHE.pop("db", None)
        ND._CACHE.update(saved)


def _synthetic_tables():
    """一只股票 2020Q1~2023Q2,三表首版同日公告(期末 +30 天);资产负债表交错修订,含 0 股本占位。"""
    ends = pd.date_range("2020-03-31", "2023-06-30", freq="QE")
    inc, cfl, bal = [], [], []
    for k, e in enumerate(ends):
        a = e + pd.Timedelta(days=30)
        q = e.quarter
        base = dict(stock_code="X", ann_date=a, f_ann_date=a, end_date=e, comp_type=1)
        inc.append({**base, **{c: float(q * (100 + k)) for c in FU.INCOME_ITEMS}})
        cfl.append({**base, **{c: float(q * (50 + k)) for c in FU.CASHFLOW_ITEMS}})
        bal.append({**base, **{c: float(1000 + 10 * k) for c in FU.BALANCE_ITEMS}, "total_share": 1e8})
    bal = pd.DataFrame(bal)
    bal.loc[bal.end_date == "2021-06-30", "total_share"] = 0.0          # 最终版就是 0、从未修订
    rev = []
    for e, a, mult, share in [("2021-12-31", "2022-06-15", 1.5, 1e8),   # 2021Q4:2022Q1 公告后才修订
                              ("2022-03-31", "2022-09-15", 1.3, 1.2e8),  # 2022Q1 首版股本 0,后版为正
                              ("2022-06-30", "2022-12-01", 1.2, 1.2e8),
                              ("2022-06-30", "2023-02-10", 1.1, 1.2e8)]:
        r = bal[bal.end_date == e].iloc[0].copy()
        for c in FU.BALANCE_ITEMS:
            r[c] = r[c] * mult
        r["total_share"] = share
        r["ann_date"] = r["f_ann_date"] = pd.Timestamp(a)
        rev.append(r)
    bal.loc[bal.end_date == "2022-03-31", "total_share"] = 0.0
    bal = pd.concat([bal, pd.DataFrame(rev)], ignore_index=True)
    return {FU.REPORT_TABLES[0]: pd.DataFrame(inc), FU.REPORT_TABLES[1]: pd.DataFrame(cfl),
            FU.REPORT_TABLES[2]: bal}


def test_asof_snapshots_match_rolling_origin_truth() -> None:
    """P2 可证伪:每个 t,默认报告表在 t 日的取值 == 只用 avail <= t 的版本重建的报告表在 t 日的取值。
    覆盖交错修订(旧期新版 + 新期首版)与 0 股本占位(早期版与最终版一视同仁)。"""
    tabs = _synthetic_tables()
    state = {"cut": pd.Timestamp("2099-01-01")}

    def _fake_load(name, items):
        d = tabs[name][["stock_code", "ann_date", "f_ann_date", "end_date", "comp_type"] + items].copy()
        d["avail"] = d[["ann_date", "f_ann_date"]].max(axis=1)
        return d[d["avail"] <= state["cut"]].sort_values(["stock_code", "end_date", "avail"])

    cols = ["total_assets", "total_assets_avg", "total_assets_yoy", "total_share",
            "total_hldr_eqy_exc_min_int_avg", "n_income_attr_p_ttm"]
    codes = pd.Index(["X"])
    orig, saved = FU._load, dict(C._SOURCE_END)
    for n in FU.REPORT_TABLES:
        C.record_source_end(n, pd.Timestamp("2099-01-01"))
    FU._load = _fake_load
    try:
        FU._REP_CACHE.pop("rep", None)
        full = FU.build_report_level()
        FU._REP_CACHE.pop("rep", None)
        assert (full["pit_vintage"] != FU.PIT_FINAL).any()
        days = pd.bdate_range("2020-05-01", "2023-09-30")
        got = {c: FU._pit_daily(full, c, days, codes)["X"] for c in cols}
        bad = []
        for t in pd.bdate_range("2020-05-01", "2023-09-30", freq="W-FRI").union(
                pd.DatetimeIndex(["2022-06-15", "2022-09-15", "2022-12-01", "2023-02-10"])):
            state["cut"] = t
            rep_t = FU.build_report_level()
            FU._REP_CACHE.pop("rep", None)
            for c in cols:
                x, y = got[c].loc[t], FU._pit_daily(rep_t, c, pd.DatetimeIndex([t]), codes).iloc[0, 0]
                if not ((np.isnan(x) and np.isnan(y)) or np.isclose(x, y, rtol=1e-6)):
                    bad.append((t.date(), c, x, y))
        assert not bad, bad[:10]
        # 0 股本占位一律视为未披露:沿用上一期股本,而不是 0
        assert (got["total_share"].dropna() > 0).all()
    finally:
        FU._load = orig
        FU._REP_CACHE.pop("rep", None)
        C._SOURCE_END.clear(); C._SOURCE_END.update(saved)


def test_report_level_avail_sue_covers_14_quarters() -> None:
    """P2:avail_sue 覆盖 sue_bt 的全部 14 个季度依赖,avail_streak 覆盖 earn_streak 的 17 个季度。"""
    rep = FU.build_report_level()
    r = rep[rep["pit_vintage"] == FU.PIT_FINAL].sort_values(["stock_code", "qi"])
    r = r[r["stock_code"] == r["stock_code"].iloc[0]].set_index("qi")
    av = r["avail"]
    for q in r.index[r["sue_bt"].notna()][:5]:
        win = av.loc[q - 13:q]
        assert r.at[q, "avail_sue"] == win.max()
        assert r.at[q, "avail_streak"] == av.loc[q - 16:q].max()
