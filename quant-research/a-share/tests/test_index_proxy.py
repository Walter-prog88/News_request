"""index_proxy 的纯逻辑单测(不读数据):调整日历、缓冲区/10% 上限、权重漂移与收益时点。"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from afactor import index_proxy as IP  # noqa: E402


def _cal(start="2020-01-02", end="2025-12-31") -> pd.DatetimeIndex:
    return pd.bdate_range(start, end)


def test_schedule_second_friday_next_trading_day():
    s = IP.review_schedule(_cal())
    r = s.set_index("review")
    # 2021-06 第二个星期五 = 06-11;工作日历下一交易日 06-14
    assert r.loc["2021H1", "second_friday"] == pd.Timestamp("2021-06-11")
    assert r.loc["2021H1", "effective"] == pd.Timestamp("2021-06-14")
    assert r.loc["2025H2", "second_friday"] == pd.Timestamp("2025-12-12")
    # 截止日 <= 4/30、10/31,且早于生效日;基准日 = 生效日前 2 个交易日
    assert (s["cutoff"] < s["announce"]).all() and (s["announce"] < s["base"]).all()
    assert (s["base"] < s["effective"]).all()
    assert (s["eff_idx"] - s["base_idx"] == IP.BASE_LAG).all()
    assert all(c.month in (4, 10) for c in s["cutoff"])


def test_buffer_first_period_takes_top_n():
    ranked = [f"s{i}" for i in range(20)]
    sel, _ = IP.buffer_select(ranked, None, n=10, new_in=8, old_keep=12, max_change=None)
    assert sel == ranked[:10]


def test_buffer_keeps_old_within_old_keep_and_blocks_new_outside_new_in():
    ranked = [f"s{i}" for i in range(20)]
    old = {"s11", "s5"}                       # s11 名次 12 <= old_keep=12,应保留
    sel, dg = IP.buffer_select(ranked, old, n=10, new_in=8, old_keep=12, max_change=None)
    assert "s11" in sel and "s5" in sel
    assert len(sel) == 10
    # 优先集合 = 前 8 名中的 7 个新样本 + s5 + s11 = 9 个;差 1 个按名次补 s8,名次 10 的 s9 落选
    assert "s8" in sel and "s9" not in sel
    assert dg["old_kept_beyond_n"] == 1


def test_cap_limits_new_entries_and_refills_with_old():
    ranked = [f"s{i}" for i in range(30)]
    old = {f"s{i}" for i in range(20, 30)}    # 老成分全部掉到 21-30 名(超出 old_keep=12)
    sel, dg = IP.buffer_select(ranked, old, n=10, new_in=8, old_keep=12, max_change=2)
    new = [c for c in sel if c not in old]
    assert len(sel) == 10 and len(new) == 2
    assert new == ["s0", "s1"]                # 只放行名次最好的 2 个新样本
    assert dg["new_blocked_by_cap"] == 8


def test_drift_and_index_timing():
    T, N = 6, 2
    R = np.zeros((T, N))
    R[2] = [0.10, 0.0]                        # 第 2 天 A 涨 10%
    R[4] = [0.0, 0.20]
    w0 = np.array([0.5, 0.5])
    D, Dp = IP.drift_weights([w0], [1], R)
    assert np.allclose(D[0], 0) and np.allclose(D[1], 0)   # 基准日 b=1 之前及当日不属于本期
    assert np.allclose(D[2], [0.55 / 1.05, 0.5 / 1.05])
    assert np.allclose(Dp[2], w0)                           # 尾盘口径期首 = 基准权重
    I = IP.index_from_weights(D, R)
    # 第 2 天的收益用第 1 天收盘权重(此时尚无成分)-> NaN;第 4 天用第 3 天收盘权重
    assert np.isnan(I[2])
    assert np.isclose(I[4], D[3][1] * 0.20)


def test_band_table():
    f = np.array([0.10, 0.18, 0.25, 0.79, 0.81, 1.0])
    assert np.allclose(IP._band(f), [0.10, 0.20, 0.30, 0.80, 1.0, 1.0])
