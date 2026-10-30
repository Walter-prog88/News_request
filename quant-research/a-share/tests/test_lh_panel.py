"""长历史日线面板(前缀 lh_data)测试。

运行:  PYTHONPATH=src python3 -m pytest tests/test_lh_panel.py -q

1-7  构建逻辑(scripts/lh_build_panel.process)在手工构造的 baostock 行上的口径:停牌估值价、持有收益、
     不复权精确涨跌停、ST 5%、创业板 2020-08-24 切换、新股首日规则、越带兜底(无涨跌幅日 / ST 退市整理 10%)、
     一字板(任务书口径:最高 == 最低 且 |日收益| > 4.5%,不依赖 ST)
7b   新浪源:sina_to_bs 补停牌行 / 前收 / 发行价 / 换手率单位;StLookup 的 ST 优先级;两源选择
8    LongPanel 接口(在合成面板目录上):raw / __getitem__ / dates / codes / 派生量 / amt_open10 / 封存守卫
9    引擎能直接跑(合成数据,不是任何真实策略收益):run_weights / simulate / benchmark(rebalance=k) /
     cost_model;三条引擎断言 exposure<=1、相关峰值 k=0、全池等权 vs 同频基准 超额 ≈ -成本
10   真实面板(cache/long_daily/panel 存在时):日期范围 <= 2025-12-31、标记自洽、与 Panel 重叠段收益一致
"""
from __future__ import annotations

import gzip
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from afactor.lh_panel import LongPanel, LH_DIR          # noqa: E402

_spec = importlib.util.spec_from_file_location("lh_build_panel", ROOT / "scripts" / "lh_build_panel.py")
B = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(B)

COLS = ["date", "code", "open", "high", "low", "close", "preclose", "volume", "amount", "turn",
        "tradestatus", "pctChg", "isST", "close_raw"]


def rows_df(code, recs, factor=1.0):
    """recs: list of dict(date, o, h, l, c, pc, vol, amt, ts=1, st=0)(不复权价);后复权 = 不复权 * factor。"""
    out = []
    for r in recs:
        ts = r.get("ts", 1)
        out.append(dict(date=r["date"], code=code,
                        open=f"{r['o'] * factor:.10f}", high=f"{r['h'] * factor:.10f}",
                        low=f"{r['l'] * factor:.10f}", close=f"{r['c'] * factor:.10f}",
                        preclose=f"{r['pc'] * factor:.10f}",
                        volume=str(r.get("vol", 1000000 if ts else 0)),
                        amount=f"{r.get('amt', 1e8 if ts else 0):.4f}",
                        turn="1.0" if ts else "", tradestatus=str(ts),
                        pctChg=f"{(r['c'] / r['pc'] - 1) * 100:.6f}", isST=str(r.get("st", 0)),
                        close_raw=f"{r['c']:.4f}"))
    return pd.DataFrame(out, columns=COLS)


def bars(dates, closes, pc0, flat_days=(), st=0, ts=None):
    """按收盘序列生成行;flat_days 内最高 = 最低 = 收盘(一字板),其余日内 ±1%。"""
    recs, pc = [], pc0
    for i, (d, c) in enumerate(zip(dates, closes)):
        t = 1 if ts is None else ts[i]
        if not t:
            recs.append(dict(date=d, o=pc, h=pc, l=pc, c=pc, pc=pc, ts=0, st=st))
            continue
        if i in flat_days:
            o = h = l = c
        else:
            o, h, l = pc, max(c, pc) * 1.01, min(c, pc) * 0.99
        recs.append(dict(date=d, o=round(o, 2), h=round(h, 2), l=round(l, 2), c=c, pc=pc, st=st))
        pc = c
    return recs


def cal_of(start, n):
    return pd.bdate_range(start, periods=n)


def run(code, recs, cal, ipo=None, factor=1.0):
    df = rows_df(code, recs, factor)
    df["date"] = [str(pd.Timestamp(d).date()) for d in df["date"]]
    return B.process(code, df, cal, ipo)


# ------------------------------------------------------------------ 1 停牌估值与持有收益
def test_suspension_valuation_and_hold_return():
    cal = cal_of("2018-03-01", 10)
    closes = [10.0, 10.2, 10.3, None, None, None, 11.5, 11.6, 11.4, 11.4]
    ts = [1, 1, 1, 0, 0, 0, 1, 1, 1, 1]
    recs = bars(cal, [c if c else 0 for c in closes], 10.0, ts=ts)
    r = run("sh.600123", recs, cal, ipo=pd.Timestamp("2001-01-01"), factor=2.0)
    f = r["fields"]
    assert list(f["trade"]) == [bool(t) for t in ts]
    assert np.isnan(f["close_px"][3:6]).all() and np.isnan(f["amount"][3:6]).all()
    # 停牌日估值价 = 最近交易日后复权收盘;交易日 = 后复权开盘
    assert np.allclose(f["twap_open10"][3:6], 10.3 * 2.0)
    assert np.isclose(f["twap_open10"][6], recs[6]["o"] * 2.0)
    # 持有收益:停牌日 0,复牌日 = 跨停牌跳空
    assert np.allclose(f["ret"][3:6], 0.0)
    assert np.isclose(f["ret"][6], 11.5 / 10.3 - 1)
    assert f["suspended"][3:6].all() and not f["suspended"][6]
    assert r["meta"]["sus_close_mismatch"] == 0


# ------------------------------------------------------------------ 2 主板 10% / ST 5% / 低价取整
def test_limits_main_st_and_rounding():
    cal = cal_of("2018-03-01", 4)
    # 主板:10.00 -> 11.00 一字涨停;-> 9.90 跌停价 9.90(11*0.9) 非一字
    recs = [dict(date=cal[0], o=10, h=10.1, l=9.9, c=10.0, pc=10.0),
            dict(date=cal[1], o=11, h=11, l=11, c=11.0, pc=10.0),
            dict(date=cal[2], o=10.5, h=10.6, l=9.9, c=9.9, pc=11.0),
            dict(date=cal[3], o=9.9, h=10.8, l=9.9, c=10.88, pc=9.9)]
    f = run("sz.000555", recs, cal, ipo=pd.Timestamp("2000-01-01"))["fields"]
    assert list(f["limit_up"]) == [False, True, False, False]
    assert list(f["one_word_up"]) == [False, True, False, False]
    assert list(f["limit_down"]) == [False, False, True, False]
    assert not f["one_word_down"].any()
    assert not f["limit_up"][3]                       # 10.88 < 涨停价 10.89
    # 低价股取整:前收 2.03 -> 涨停 2.23(2.233),跌停 1.83(1.827);ST 5%:前收 2.03 -> 2.13(2.1315)
    recs = [dict(date=cal[0], o=2.03, h=2.05, l=2.0, c=2.03, pc=2.03),
            dict(date=cal[1], o=2.23, h=2.23, l=2.23, c=2.23, pc=2.03),
            dict(date=cal[2], o=2.0, h=2.05, l=2.007, c=2.01, pc=2.23),
            dict(date=cal[3], o=1.9, h=1.95, l=1.81, c=1.81, pc=2.01)]
    f = run("sh.600999", recs, cal, ipo=pd.Timestamp("2000-01-01"))["fields"]
    assert f["one_word_up"][1] and f["limit_down"][3]      # 2.01*0.9 = 1.809 -> 1.81
    st_recs = [dict(r, st=1) for r in recs]
    st_recs[1] = dict(date=cal[1], o=2.13, h=2.13, l=2.13, c=2.13, pc=2.03, st=1)
    f = run("sh.600999", st_recs, cal, ipo=pd.Timestamp("2000-01-01"))["fields"]
    assert f["one_word_up"][1] and np.isclose(f["lim_up_rate"][1], 0.05)


# ------------------------------------------------------------------ 3 创业板 2020-08-24 切换
def test_gem_switch_20200824():
    cal = pd.DatetimeIndex(["2020-08-20", "2020-08-21", "2020-08-24", "2020-08-25"])
    recs = [dict(date=cal[0], o=10, h=10.1, l=9.9, c=10.0, pc=10.0),
            dict(date=cal[1], o=11, h=11, l=11, c=11.0, pc=10.0),          # 10% = 改革前涨停
            dict(date=cal[2], o=12, h=12.2, l=11.8, c=12.1, pc=11.0),       # +10% 改革后不是涨停
            dict(date=cal[3], o=14.52, h=14.52, l=14.52, c=14.52, pc=12.1)]  # +20% 一字
    f = run("sz.300999", recs, cal, ipo=pd.Timestamp("2015-01-01"))["fields"]
    assert list(f["limit_up"]) == [False, True, False, True]
    assert list(f["one_word_up"]) == [False, True, False, True]
    assert np.isclose(f["lim_up_rate"][1], 0.10) and np.isclose(f["lim_up_rate"][2], 0.20)


# ------------------------------------------------------------------ 4 新股首日
def test_ipo_first_day_rules():
    cal = cal_of("2015-06-01", 7)
    # 2015 主板新股:首日 +44% 一字(发行价 10 -> 14.40),之后 10%
    closes = [14.40, 15.84, 17.42, 19.16, 21.08, 23.19, 25.51]
    recs = bars(cal, closes, 10.0, flat_days=range(7))
    f = run("sh.603999", recs, cal, ipo=cal[0])["fields"]
    assert f["one_word_up"].all()
    assert np.isclose(f["lim_up_rate"][0], 0.44) and np.isclose(f["lim_up_rate"][1], 0.10)
    assert not f["no_limit_day"].any()
    # 科创板(注册制):前 5 日无涨跌幅,第 6 日起 20%
    cal2 = cal_of("2021-06-01", 7)
    closes = [30.0, 33.0, 31.0, 30.0, 29.5, 35.4, 36.0]
    recs = bars(cal2, closes, 10.0)
    f = run("sh.688999", recs, cal2, ipo=cal2[0])["fields"]
    assert f["no_limit_day"][:5].all() and not f["no_limit_day"][5:].any()
    assert not f["limit_up"][:5].any() and np.isnan(f["lim_up_rate"][:5]).all()
    assert np.isclose(f["lim_up_rate"][5], 0.20)
    # 2010 主板新股首日:无涨跌幅限制
    cal3 = cal_of("2010-06-01", 3)
    recs = bars(cal3, [18.0, 18.5, 18.3], 10.0)
    f = run("sz.002999", recs, cal3, ipo=cal3[0])["fields"]
    assert f["no_limit_day"][0] and not f["no_limit_day"][1:].any()
    assert f["days_listed"][0] == 1 and f["days_listed"][2] == 3


# ------------------------------------------------------------------ 5/6 越带兜底
def test_out_of_band_fallbacks():
    cal = cal_of("2006-03-01", 4)
    # 股改复牌首日 +30%:越出 10% 带 -> 无涨跌幅日,不标涨停
    recs = [dict(date=cal[0], o=5, h=5.1, l=4.9, c=5.0, pc=5.0),
            dict(date=cal[1], o=6.5, h=6.5, l=6.5, c=6.5, pc=5.0),
            dict(date=cal[2], o=6.5, h=7.15, l=6.4, c=7.15, pc=6.5),
            dict(date=cal[3], o=7.0, h=7.2, l=6.9, c=7.0, pc=7.15)]
    f = run("sh.600998", recs, cal, ipo=pd.Timestamp("1999-01-01"))["fields"]
    assert list(f["no_limit_day"]) == [False, True, False, False]
    assert list(f["out_band"]) == [False, True, False, False]
    assert not f["limit_up"][1] and not f["ow_lim_up"][1]
    # 任务书口径的一字板不看涨跌停价:最高 == 最低 且 +30% -> 锁板,不可成交
    assert f["one_word_up"][1]
    assert f["limit_up"][2]
    # ST 股跌 10%(退市整理期按 10%):5% 带越带 -> 10% 带,一字跌停可识别
    recs = [dict(date=cal[0], o=2, h=2.02, l=1.98, c=2.0, pc=2.0, st=1),
            dict(date=cal[1], o=1.8, h=1.8, l=1.8, c=1.8, pc=2.0, st=1),
            dict(date=cal[2], o=1.62, h=1.62, l=1.62, c=1.62, pc=1.8, st=1),
            dict(date=cal[3], o=1.6, h=1.66, l=1.55, c=1.6, pc=1.62, st=1)]
    r = run("sh.600997", recs, cal, ipo=pd.Timestamp("1999-01-01"))
    f = r["fields"]
    assert list(f["one_word_down"]) == [False, True, True, False]
    assert np.isclose(f["lim_dn_rate"][1], 0.10) and np.isclose(f["lim_dn_rate"][3], 0.05)
    assert r["meta"]["n_esc10"] == 2 and not f["no_limit_day"].any()


def test_main_board_st_10pct_from_20250707():
    cal = pd.DatetimeIndex(["2025-07-03", "2025-07-04", "2025-07-07", "2025-07-08"])
    recs = [dict(date=cal[0], o=3, h=3.02, l=2.98, c=3.0, pc=3.0, st=1),
            dict(date=cal[1], o=3.15, h=3.15, l=3.15, c=3.15, pc=3.0, st=1),
            dict(date=cal[2], o=3.2, h=3.4, l=3.15, c=3.3, pc=3.15, st=1),
            dict(date=cal[3], o=3.63, h=3.63, l=3.63, c=3.63, pc=3.3, st=1)]
    f = run("sz.000998", recs, cal, ipo=pd.Timestamp("1999-01-01"))["fields"]
    assert np.isclose(f["lim_up_rate"][1], 0.05) and np.isclose(f["lim_up_rate"][2], 0.10)
    assert list(f["one_word_up"]) == [False, True, False, True]
    assert not f["no_limit_day"].any()


# ------------------------------------------------------------------ 7 后复权 -> 不复权还原
def test_adjusted_prices_reconstructed_to_raw():
    cal = cal_of("2019-03-01", 3)
    recs = [dict(date=cal[0], o=7.0, h=7.1, l=6.9, c=7.0, pc=7.0),
            dict(date=cal[1], o=7.7, h=7.7, l=7.7, c=7.7, pc=7.0),
            dict(date=cal[2], o=7.5, h=7.8, l=7.4, c=7.6, pc=7.7)]
    f = run("sh.600996", recs, cal, ipo=pd.Timestamp("1999-01-01"), factor=37.123456)["fields"]
    assert list(f["one_word_up"]) == [False, True, False]
    assert np.isclose(f["close_raw"][1], 7.7)
    assert np.isclose(f["close_px"][1], 7.7 * 37.123456, rtol=1e-6)


def test_one_word_simple_rule_independent_of_st():
    """ST 股 +5% 一字(不知道它是 ST 也能识别);非一字的涨停不算;最高 == 最低但只涨 3% 不算。"""
    cal = cal_of("2012-03-01", 4)
    recs = [dict(date=cal[0], o=4, h=4.04, l=3.96, c=4.0, pc=4.0),
            dict(date=cal[1], o=4.2, h=4.2, l=4.2, c=4.2, pc=4.0),             # +5% 一字(当成非 ST 也识别)
            dict(date=cal[2], o=4.4, h=4.62, l=4.3, c=4.62, pc=4.2),           # +10% 涨停但非一字
            dict(date=cal[3], o=4.75, h=4.75, l=4.75, c=4.75, pc=4.62)]        # 最高 == 最低,+2.8%
    f = run("sh.600995", recs, cal, ipo=pd.Timestamp("1999-01-01"))["fields"]
    assert list(f["one_word_up"]) == [False, True, False, False]
    assert list(f["limit_up"]) == [False, False, True, False]                  # 当成非 ST:+5% 不是涨停价
    assert not f["ow_lim_up"].any()


# ------------------------------------------------------------------ 7b 新浪源
def sina_rows(dates, raw_closes, factor, ipo_price=None, vol=1_000_000):
    out = []
    for i, (d, c) in enumerate(zip(dates, raw_closes)):
        v = vol + 100 * i
        out.append(dict(date=str(pd.Timestamp(d).date()), open=c, high=round(c * 1.01, 2), low=round(c * 0.99, 2),
                        close=c, volume=v, amount=v * c, post_vol=np.nan, post_amt=np.nan,
                        prevclose_ipo=ipo_price if (i == 0 and ipo_price) else np.nan,
                        outstanding_share=1e8, turnover=v / 1e8, hfq_factor=factor[i] if np.ndim(factor) else factor))
    return pd.DataFrame(out)


def test_sina_to_bs_fills_suspension_and_preclose():
    cal = cal_of("2018-03-01", 10)
    # 第 3-5 天停牌(新浪无行);第 6 天复牌时除权(因子 2 -> 2.2)
    keep = [0, 1, 2, 6, 7, 8, 9]
    raw = [10.0, 10.2, 10.3, 9.9, 10.0, 10.1, 10.2]
    fac = [2.0, 2.0, 2.0, 2.2, 2.2, 2.2, 2.2]
    s = sina_rows(cal[keep], raw, fac)
    st = np.array([False] * 7 + [True] * 3)
    d, info = B.sina_to_bs("sh.600123", s, cal, pd.Timestamp("2001-01-01"), st)
    assert len(d) == 10 and info["sina_fill_rows"] == 3
    assert list(d["tradestatus"]) == [1, 1, 1, 0, 0, 0, 1, 1, 1, 1]
    assert np.allclose(d["close"].iloc[3:6], 10.3 * 2.0) and (d["volume"].iloc[3:6] == 0).all()
    assert np.isnan(d["preclose"].iloc[0])                                      # 2005 年前上市:窗口首行前收未知
    assert np.isclose(d["preclose"].iloc[6], 10.3 * 2.0)                        # 复牌前收 = 停牌前后复权收盘
    assert np.allclose(d["turn"].iloc[[0, 6]], [1.0, 1.0003])                  # 新浪小数 x 100 = %
    assert list(d["isST"]) == [0] * 7 + [1] * 3
    r = B.process("sh.600123", d, cal, pd.Timestamp("2001-01-01"))
    f = r["fields"]
    assert np.allclose(f["ret"][3:6], 0.0) and np.isclose(f["ret"][6], 9.9 * 2.2 / (10.3 * 2.0) - 1)
    assert np.allclose(f["twap_open10"][3:6], 10.3 * 2.0) and f["suspended"][3:6].all()
    assert np.isclose(f["close_raw"][6], 9.9) and np.isclose(f["close_px"][6], 9.9 * 2.2)
    assert np.isnan(f["pct_chg"]).all()
    # 新股:窗口内上市,首行前收 = 发行价 x 因子;上市前不补行
    s2 = sina_rows(cal[3:], [14.4, 15.84, 17.42, 19.16, 21.08, 23.19, 25.51], 1.0, ipo_price=10.0)
    s2["high"] = s2["low"] = s2["close"]
    d2, _ = B.sina_to_bs("sh.603998", s2, cal, cal[3], None)
    assert len(d2) == 7 and np.isclose(d2["preclose"].iloc[0], 10.0)
    f2 = B.process("sh.603998", d2, cal, cal[3])["fields"]
    assert f2["one_word_up"].all() and f2["days_listed"][0] == 1


def test_st_lookup_priority(tmp_path):
    cal = pd.DatetimeIndex(pd.bdate_range("2016-08-01", periods=12))
    parts, pst = tmp_path / "parts", tmp_path / "parts_st"
    parts.mkdir(); pst.mkdir()
    # baostock ST 专抓只覆盖前 6 天:第 2-3 天 ST
    with gzip.open(pst / "sh.600111.csv.gz", "wt") as fh:
        pd.DataFrame(dict(date=[str(d.date()) for d in cal[:6]], tradestatus="1",
                          isST=["0", "0", "1", "1", "0", "0"])).to_csv(fh, index=False)
    # st_info 从第 6 天起:600111 第 8 天起 ST;第 10 天名单缺失 -> 沿用前一名单日
    days = [cal[6], cal[7], cal[8], cal[10], cal[11]]
    info = pd.DataFrame(False, index=days, columns=["sh.600111", "sh.600222"])
    info.loc[cal[8]:, "sh.600111"] = True
    info.loc[cal[7], "sh.600222"] = True
    lk = B.StLookup(cal, parts, pst, st_info=info)
    st, known, src = lk.get("sh.600111")
    assert src == "baostock+st_info"
    assert list(st) == [False, False, True, True, False, False, False, False, True, True, True, True]
    assert known.all()
    st2, known2, src2 = lk.get("sh.600222")
    assert src2 == "st_info" and list(st2[6:]) == [False, True, False, False, False, False]
    assert not known2[:6].any() and known2[6:].all()


def test_build_panel_two_sources(tmp_path):
    cal = cal_of("2019-01-02", 30)
    parts, sina, panel = tmp_path / "parts", tmp_path / "sina", tmp_path / "panel"
    parts.mkdir(); sina.mkdir()
    rng = np.random.default_rng(3)
    closes = list(np.round(10 * np.cumprod(1 + rng.normal(0, 0.01, 30)), 2))
    # 600001 退市股只有 baostock;600002 现存两源都有(取新浪);600003 现存只有 baostock(新浪失败 -> 回退);
    # 600004 退市股两源都有:价格取新浪,换手率 / 上市区间取 baostock(baostock 最后 3 行停牌,新浪无行)
    for code in ("sh.600001", "sh.600002", "sh.600003", "sh.600004"):
        ts = [1] * 27 + [0] * 3 if code == "sh.600004" else None
        df = rows_df(code, bars(cal, closes, 10.0, ts=ts), factor=1.5)
        df["date"] = [str(pd.Timestamp(d).date()) for d in df["date"]]
        df["turn"] = np.where(df["tradestatus"] == "1", "2.5", "")
        with gzip.open(parts / f"{code}.csv.gz", "wt") as fh:
            df.to_csv(fh, index=False)
    s = sina_rows(cal, closes, 3.0)
    with gzip.open(sina / "sh.600002.csv.gz", "wt") as fh:
        s.to_csv(fh, index=False)
    s4 = sina_rows(cal[:27], closes[:27], 2.0)
    s4["turnover"] = np.nan; s4["outstanding_share"] = np.nan
    s4 = pd.concat([s4, s4.iloc[[-1]].assign(date="2019-02-10")], ignore_index=True)   # 周末残行(重复行)
    with gzip.open(sina / "sh.600004.csv.gz", "wt") as fh:
        s4.to_csv(fh, index=False)
    (sina / "sh.600003.nosina").write_text("x")
    uni = pd.DataFrame(dict(code=["sh.600001", "sh.600002", "sh.600003", "sh.600004"], ipoDate="2001-01-01",
                            outDate=["2019-03-01", "", "", "2019-03-01"], status=["0", "1", "1", "0"]))
    lk = B.StLookup(cal, parts, None, st_info=pd.DataFrame(dtype=bool))
    B.LOG = tmp_path / "build.log"
    meta = B.build_panel(parts, panel, uni, cal, dict(open10_amt_ratio=0.1), sina_dir=sina, st=lk)
    assert meta["sources"] == {"baostock": 1, "sina": 1, "baostock_fallback": 1, "sina+bs": 1}
    p = LongPanel(panel_dir=panel)
    assert list(p.code_info["src"]) == ["baostock", "sina", "baostock_fallback", "sina+bs"]
    # 两源收益一致(后复权因子不同不影响收益)
    r = p["ret_c2c"]
    for k in ("600002.SH", "600004.SH"):
        assert np.allclose(r["600001.SH"].iloc[1:27], r[k].iloc[1:27], atol=1e-6)
    assert np.isnan(p.raw("pct_chg")["600002.SH"]).all() and p.raw("pct_chg")["600001.SH"].notna().all()
    # 退市股 sina+bs:换手率取 baostock(新浪为 NaN),上市区间到 baostock 末行,末 3 行停牌
    assert np.allclose(p.raw("turn")["600004.SH"].iloc[:27], 2.5)
    assert np.allclose(p.raw("turn")["600002.SH"].iloc[:27], 1.0 + 1e-4 * np.arange(27), rtol=1e-5)   # 新浪小数 x 100
    assert p.raw("listed")["600004.SH"].all() and p.raw("suspended")["600004.SH"].iloc[27:].all()
    assert p.code_info.loc["600004.SH", "sina_dup_rows"] == 1
    assert p.code_info.loc["600004.SH", "st_src"] == "baostock"


# ------------------------------------------------------------------ 合成面板(8/9 共用)
@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    """40 只主板股 x 160 个工作日的随机游走;一只中途停牌、一只中途退市、一只 ST;成交额都 >= 2000 万。"""
    rng = np.random.default_rng(7)
    root = tmp_path_factory.mktemp("lh")
    parts, panel = root / "parts", root / "panel"
    parts.mkdir()
    cal = cal_of("2016-01-04", 160)
    uni = []
    for j in range(40):
        code = f"sh.60{j:04d}"
        c0 = float(rng.uniform(3, 30))
        closes, c = [], c0
        for i in range(len(cal)):
            c = round(max(c * (1 + rng.normal(0, 0.02)), 0.5), 2)
            closes.append(c)
        ts = [1] * len(cal)
        if j == 1:
            ts[40:55] = [0] * 15
        n = len(cal) if j != 2 else 120                    # j=2 第 120 天后退市(无行)
        recs = bars(cal[:n], closes[:n], c0, st=1 if j == 3 else 0, ts=ts[:n])
        for r in recs:
            if r.get("ts", 1):
                r["amt"] = float(rng.uniform(3e7, 3e8)); r["vol"] = int(r["amt"] / r["c"])
        df = rows_df(code, recs, factor=1.0 + j / 10)
        df["date"] = [str(pd.Timestamp(d).date()) for d in df["date"]]
        with gzip.open(parts / f"{code}.csv.gz", "wt") as fh:
            df.to_csv(fh, index=False)
        uni.append(dict(code=code, ipoDate="2001-01-01", outDate=str(cal[121].date()) if j == 2 else ""))
    uni = pd.DataFrame(uni)
    B.LOG = root / "build.log"
    B.build_panel(parts, panel, uni, cal, dict(open10_amt_ratio=0.113), allow_partial=False)
    return panel


def test_longpanel_interface(synth):
    p = LongPanel(panel_dir=synth)
    assert len(p.dates) == 160 and len(p.codes) == 40 and p.codes[0] == "600000.SH"
    for nm in ("open_px", "close_px", "amount", "twap_open10", "amt_open10", "vwap", "rng", "turn"):
        assert p.raw(nm).shape == (160, 40)
        assert p[nm] is p.raw(nm)
    assert np.allclose(p.raw("amt_open10").to_numpy(), (p.raw("amount") * np.float32(0.113)).to_numpy(), equal_nan=True)
    v = p["valid"]
    assert not v["600001.SH"].iloc[40:55].any() and v["600001.SH"].iloc[39]
    assert p["twap_open10"]["600001.SH"].iloc[40:55].notna().all()          # 停牌日估值价
    assert LongPanel(panel_dir=synth, suspend_fill=False)["twap_open10"]["600001.SH"].iloc[40:55].isna().all()
    assert p["twap_open10"]["600002.SH"].iloc[120:].isna().all()            # 退市后 NaN
    assert p["delisted"]["600002.SH"].iloc[120:].all() and not p["delisted"]["600002.SH"].iloc[:120].any()
    assert p["last_trade_day"]["600002.SH"].sum() == 1 and p["last_trade_day"]["600002.SH"].iloc[119]
    assert p["is_st"]["600003.SH"].all() and not p["is_st"]["600004.SH"].any()
    dl = p["days_listed"]
    assert (dl.dropna() >= 6).all().all()                                   # 2001 年上市
    u = p["universe"]
    assert u.dtypes.eq(bool).all() and not u.iloc[:9].any().any() and u.iloc[30].sum() >= 38
    assert not LongPanel(panel_dir=synth, min_adv=1e12)["universe"].any().any()
    adv = p["adv20"]
    assert np.isclose(adv.iloc[30, 0], p.raw("amount").iloc[11:31, 0].mean(), rtol=1e-5)
    assert set(("twap_open10_f", "one_word_up", "universe")) <= set(p.features) | set(("twap_open10_f", "universe"))
    assert p.code_info.loc["600002.SH", "outDate"] != ""
    with pytest.raises(PermissionError):
        LongPanel(panel_dir=synth, end="2026-01-05")
    with pytest.raises(ValueError):
        p.fwd_ret(1, "tail")
    q = LongPanel(panel_dir=synth, start="2016-03-01")
    assert q.dates[0] >= pd.Timestamp("2016-03-01")


def test_engine_runs_on_longpanel(synth):
    """strategy.run_weights / benchmark 直接吃 LongPanel(合成数据;这不是任何真实策略的收益)。"""
    from afactor import strategy as ST
    p = LongPanel(panel_dir=synth)
    U = ST.universe(p, "V1000").to_numpy(bool)
    W = U / np.maximum(U.sum(1, keepdims=True), 1)
    res = ST.run_weights(W, p, "open10", 1e7, rebalance=5, record=True)
    assert res["record"]["daily_by_phase"]["exposure"].max() <= 1.0 + 1e-9          # 断言 1
    b = ST.benchmark(p, "V1000", "open10", rebalance=5)
    assert res["rets"].index.equals(b["rets"].index)
    cs = {k: res["rets"].corr(b["rets"].shift(k)) for k in (-2, -1, 0, 1, 2)}
    assert max(cs, key=cs.get) == 0                                                    # 断言 2
    d = res["rets"] - b["rets"] + res["record"]["daily"]["cost_total"]
    assert abs(d.mean() * 242 * 100) < 1.0, d.mean() * 242 * 100                      # 断言 3(pp/年)
    cm = p.cost_model(impact_y=1.0)
    r2 = ST.run_weights(W, p, "open10", 1e8, rebalance=5, cost_model=cm)
    assert r2["exposure"] <= 1.0 + 1e-9 and r2["cost_bps"] > res["cost_bps"]
    r3 = ST.run_weights(W, p, "vwap1", 1e7, rebalance=5)
    assert np.isfinite(r3["ann_return"])
    # 停牌股持仓:估值价让复牌跳空进入持仓收益(引擎 ok_px 两端都有价)
    assert p["twap_open10"]["600001.SH"].iloc[39:56].notna().all()


# ------------------------------------------------------------------ 10 真实面板
_REAL = (LH_DIR / "_meta.json").exists()


@pytest.fixture(scope="module")
def real():
    if not _REAL:
        pytest.skip("长历史面板未构建")
    return LongPanel()


@pytest.mark.skipif(not _REAL, reason="长历史面板未构建")
def test_real_panel_ranges_and_flags(real):
    p = real
    assert p.dates[0] >= pd.Timestamp("2005-01-04") and p.dates[-1] <= pd.Timestamp("2025-12-31")
    assert p.codes.str.match(r"^(60|68|00|30)\d{4}\.(SH|SZ)$").all()
    assert not p.codes.str.startswith(("000", "399")).any() or p.codes.str.endswith(".SZ")[p.codes.str.startswith("000")].all()
    ow = p.raw("ow_lim_up").to_numpy(bool); lu = p.raw("limit_up").to_numpy(bool)
    assert not (ow & ~lu).any()
    ows = p.raw("one_word_up").to_numpy(bool) | p.raw("one_word_down").to_numpy(bool)
    hi, lo = p.raw("high_px").to_numpy(), p.raw("low_px").to_numpy()
    assert not (ows & ~p.raw("trade").to_numpy(bool)).any()
    assert np.nanmax(np.abs(hi[ows] / lo[ows] - 1)) < 5e-3                    # 一字板:最高 == 最低(不复权取整级)
    assert set(p.code_info["src"]) <= {"sina", "sina+bs", "baostock", "baostock_fallback"}
    # ST 身份:新浪源股票的交易日几乎全部有来源;退市股都在,且都是 baostock 定上市区间
    sk, tr2 = p.raw("st_known").to_numpy(bool), p.raw("trade").to_numpy(bool)
    assert sk[tr2].mean() > 0.999
    info = p.code_info
    dl = info[(info["outDate"].fillna("") != "") & (info["outDate"].fillna("") <= "2025-12-31")]
    assert len(dl) >= 290 and set(dl["src"]) <= {"sina+bs", "baostock"}
    sus = p.raw("suspended").to_numpy(bool)
    assert np.isnan(p.raw("close_px").to_numpy()[sus]).all()
    tw = p.raw("twap_open10").to_numpy()
    tr = p.raw("trade").to_numpy(bool)
    assert np.isfinite(tw[tr]).all()
    # 交易日的后复权收益与 pctChg 一致(四舍五入级)
    r = p.raw("ret").to_numpy(np.float64); pc = p.raw("pct_chg").to_numpy(np.float64)
    ok = tr & np.isfinite(r) & np.isfinite(pc)
    assert np.quantile(np.abs(r[ok] * 100 - pc[ok]), 0.999) < 0.01
    meta = json.loads((LH_DIR / "_meta.json").read_text())
    assert 0.05 < meta["open10_amt_ratio"] < 0.25


@pytest.mark.skipif(not _REAL, reason="长历史面板未构建")
def test_real_overlap_with_project_panel(real):
    from afactor.panel import Panel
    P = Panel()
    cs = real.codes.intersection(P.codes)[:400]
    a = real.raw("close_px").loc["2020-01-01":, cs].reindex(P.dates)
    b = P.raw("close_px")[cs]
    ra, rb = a / a.shift(1) - 1, b / b.shift(1) - 1
    m = ra.notna() & rb.notna()
    d = (ra - rb).to_numpy()[m.to_numpy()]
    assert np.median(np.abs(d)) < 1e-4 and np.mean(np.abs(d) > 1e-3) < 1e-3
    am = real.raw("amount").loc["2020-01-01":, cs].reindex(P.dates)
    rat = (am / P.raw("amount")[cs]).to_numpy()
    rat = rat[np.isfinite(rat) & (rat > 0)]
    assert abs(np.median(rat) - 1) < 1e-3
