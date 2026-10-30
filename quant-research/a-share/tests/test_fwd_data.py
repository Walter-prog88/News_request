"""fwd_data 纯逻辑单测(不读任何数据文件、不联网)。

覆盖:baostock 5 分钟分组规则、1 分钟 -> 5 分钟聚合、UTC 时间戳修复与缺格补齐、代码格式转换。
"""
import numpy as np
import pandas as pd
import pytest

from afactor import fwd_data as FD


def _day_grid(day="2026-08-03"):
    d = pd.Timestamp(day)
    return pd.DatetimeIndex([d + o for o in FD.minute_offsets()], name="trade_time")


def test_bs5_groups_rule():
    g = FD.bs5_groups()
    assert len(g) == 241
    assert (np.bincount(g) == np.array([6] + [5] * 47)).all()
    assert list(np.where(g == 0)[0]) == [0, 1, 2, 3, 4, 5]               # 09:35 含开盘集合竞价
    assert list(np.where(g == 23)[0]) == [116, 117, 118, 119, 120]       # 11:30
    assert list(np.where(g == 24)[0]) == [121, 122, 123, 124, 125]       # 13:05
    assert list(np.where(g == 47)[0]) == [236, 237, 238, 239, 240]       # 15:00 含收盘集合竞价
    assert FD.BS5_TIMES[0] == "09:35" and FD.BS5_TIMES[23] == "11:30"
    assert FD.BS5_TIMES[24] == "13:05" and FD.BS5_TIMES[-1] == "15:00"


def test_minute_offsets_match_project_bar_index():
    t = _day_grid()
    assert len(t) == 241
    assert t[0].strftime("%H:%M") == "09:30" and t[1].strftime("%H:%M") == "09:31"
    assert t[120].strftime("%H:%M") == "11:30" and t[121].strftime("%H:%M") == "13:01"
    assert t[240].strftime("%H:%M") == "15:00"


def test_minute_to_bs5_last_first_sum():
    idx = _day_grid().append(_day_grid("2026-08-04"))
    x = np.arange(len(idx), dtype=float)
    w = pd.DataFrame({"A.SZ": x, "B.SZ": np.nan}, index=idx)
    w.loc[w.index.normalize() == "2026-08-04", "B.SZ"] = 1.0
    c = FD.minute_to_bs5(w, "last")
    o = FD.minute_to_bs5(w, "first")
    s = FD.minute_to_bs5(w, "sum")
    assert len(c) == 96
    assert c.index[0] == pd.Timestamp("2026-08-03 09:35") and c.index[47] == pd.Timestamp("2026-08-03 15:00")
    assert c["A.SZ"].iloc[0] == 5 and o["A.SZ"].iloc[0] == 0 and s["A.SZ"].iloc[0] == sum(range(6))
    assert c["A.SZ"].iloc[1] == 10 and o["A.SZ"].iloc[1] == 6
    assert c["A.SZ"].iloc[23] == 120 and c["A.SZ"].iloc[24] == 125 and c["A.SZ"].iloc[47] == 240
    assert s["A.SZ"].iloc[:48].sum() == x[:241].sum()
    # 整日 NaN 保持 NaN(停牌),不被 nansum 变成 0
    assert s["B.SZ"].iloc[:48].isna().all() and (s["B.SZ"].iloc[48:] == [6] + [5] * 47).all()


def test_minute_to_bs5_rejects_wrong_rowcount():
    w = pd.DataFrame({"A.SZ": 1.0}, index=_day_grid()[1:])
    with pytest.raises(ValueError):
        FD.minute_to_bs5(w, "last")


def test_clean_minute_month_utc_and_missing_slot():
    day1 = _day_grid("2026-08-10")
    day2 = _day_grid("2026-04-08")[1:]                    # 缺 09:30 那一行
    cst = pd.DataFrame({"A.SZ": 1.0, "B.SZ": np.nan}, index=day1)
    utc = pd.DataFrame({"A.SZ": np.nan, "B.SZ": 2.0}, index=day1 - pd.Timedelta(hours=8))
    utc.iloc[5, 0] = 9.0                                  # 与上海时间版冲突的一格
    other = pd.DataFrame({"A.SZ": 3.0, "B.SZ": 3.0}, index=day2)
    w = pd.concat([utc, cst, other]).sort_index()
    out, dg = FD.clean_minute_month(w)
    assert len(out) == 2 * 241
    assert dg["utc_rows"] == 241 and dg["utc_days"] == ["2026-08-10"]
    assert dg["conflict_cells"] == 1
    assert dg["missing_slots"] == {"2026-04-08": ["09:30"]}
    x = out.loc["2026-08-10"]
    assert (x["A.SZ"] == 1.0).all()                       # 冲突取上海时间版
    assert (x["B.SZ"] == 2.0).all()                       # UTC 版补上海时间版的空
    assert np.isnan(out.loc[pd.Timestamp("2026-04-08 09:30"), "A.SZ"])


def test_code_conversion():
    assert FD.to_bs("000001.SZ") == "sz.000001" and FD.to_bs("600000.SH") == "sh.600000"
    assert FD.to_bs("920978.BJ") is None
    assert FD.from_bs("sz.300750") == "300750.SZ"
    assert FD.is_a_share_bs("sh.688981") and FD.is_a_share_bs("sz.302132")
    assert not FD.is_a_share_bs("sh.000001") and not FD.is_a_share_bs("sz.399001")
    assert not FD.is_a_share_bs("sh.510300") and not FD.is_a_share_bs("sz.200002")
