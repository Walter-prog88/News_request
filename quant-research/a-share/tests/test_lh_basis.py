"""lh_basis 的口径单测(合成数据,不依赖下载文件)。PYTHONPATH=src pytest tests/test_lh_basis.py"""
import numpy as np
import pandas as pd

from afactor import lh_basis as LB


def _cal(a="2022-01-03", b="2022-04-29"):
    return pd.bdate_range(a, b)


def test_third_friday_rule():
    cal = _cal("2015-01-02", "2022-12-30")
    assert LB.third_friday_rule(201506, cal) == pd.Timestamp("2015-06-19")
    assert LB.third_friday_rule(202203, cal) == pd.Timestamp("2022-03-18")
    # 第三个周五不是交易日 -> 顺延
    cal2 = cal.drop(pd.Timestamp("2022-03-18"))
    assert LB.third_friday_rule(202203, cal2) == pd.Timestamp("2022-03-21")


def _synthetic():
    """一个品种 IC,三张合约:2202(2/18 到期)、2203(3/18)、2206(6/17)。现货恒为 100;
    期货价 = 100 * (1 - 0.10 * 剩余天数/365),即恒定 -10% 年化贴水,线性收敛。"""
    cal = _cal("2022-01-04", "2022-03-31")
    exp = {"IC2202": pd.Timestamp("2022-02-18"), "IC2203": pd.Timestamp("2022-03-18"),
           "IC2206": pd.Timestamp("2022-06-17")}
    rows = []
    for d in cal:
        for s, e in exp.items():
            if d <= e:
                dte = (e - d).days
                rows.append({"date": d, "symbol": s, "close": 100 * (1 - 0.10 * dte / 365),
                             "settle": np.nan, "oi": 100.0 if s == "IC2203" else 50.0})
    p = pd.DataFrame(rows)
    p["variety"] = "IC"
    p["cmonth"] = p["symbol"].str[2:].astype(int) + 200000
    expiry = pd.DataFrame({"expiry": pd.Series(exp)})
    spot = pd.DataFrame({"000905": 100.0}, index=cal)
    c = LB.build_contracts(p, expiry, spot)
    return c, spot


def test_basis_ann_and_roll_rules():
    c, spot = _synthetic()
    live = c[c["dte_cal"] > 0]
    assert np.allclose(live["basis_ann"], -0.10)
    h = LB.build_hedge(c, spot, "IC", "m1", roll_days=2)
    # m1 不持有 td_left <= 2 的合约
    td = c.set_index(["date", "symbol"])["td_left"]
    assert all(td[(d, s)] > 2 for d, s in h["held"].items())
    # 2202 在 2/16 收盘(td_left=2)换到 2203
    assert h.loc["2022-02-15", "held"] == "IC2202" and h.loc["2022-02-16", "held"] == "IC2203"
    assert bool(h.loc["2022-02-16", "roll"])
    # 恒定 -10% 年化贴水 + 现货不动 -> 空头每天亏损 = 收敛,年化已实现成本接近 +10%(按自然日/交易日换算)
    hc = h["hedge_cost"].dropna()
    assert (hc > 0).all()
    realized_cal = hc.sum() / ((h.index[-1] - h.index[0]).days / 365)
    assert abs(realized_cal - 0.10) < 0.005
    # m2: 当月到期后第一个交易日收盘换月(合成数据里没有下下月合约,2203 到期前 2nd 就是 2206)
    h2 = LB.build_hedge(c, spot, "IC", "m2", roll_days=2)
    assert h2.loc["2022-02-18", "held"] == "IC2203"          # 2202 到期日:2nd 仍是 2203
    assert h2.loc["2022-02-21", "held"] == "IC2206"          # 2202 摘牌后:2nd = 2206
    # q1: 季月合约
    hq = LB.build_hedge(c, spot, "IC", "q1", roll_days=2)
    assert hq.loc["2022-01-10", "held"] == "IC2203"
    assert hq.loc["2022-03-16", "held"] == "IC2206"
    # short_ret = -fut_ret - roll_tc
    assert np.allclose((hq["short_ret"] + hq["fut_ret"] + hq["roll_tc"]).dropna(), 0)


def test_main_uses_previous_day_oi():
    c, spot = _synthetic()
    h = LB.build_hedge(c, spot, "IC", "main", roll_days=2)
    # 首日没有前一日持仓量 -> 按到期最近;之后 2203 持仓最大
    assert h.iloc[0]["held"] == "IC2202"
    assert h.loc["2022-01-10", "held"] == "IC2203"
