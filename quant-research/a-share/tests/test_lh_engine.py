"""afactor.lh_engine 的合成数据测试(前缀 lh;不读真实面板,不算任何真实收益)。

PYTHONPATH=src python3 -m pytest -q tests/test_lh_engine.py
"""
import numpy as np
import pandas as pd

from afactor import lh_engine as LE
from afactor import strategy as ST


def _synth(T=40, N=60, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2008-09-10", periods=T)
    codes = [f"{i:06d}.SZ" for i in range(N)]
    V = pd.DataFrame(np.cumprod(1 + rng.normal(0, 0.02, (T, N)), axis=0).astype(np.float32) * 10,
                     index=dates, columns=codes)
    amt = pd.DataFrame(np.full((T, N), 1e9, np.float32), index=dates, columns=codes)
    dead = pd.DataFrame(np.zeros((T, N), bool), index=dates, columns=codes)
    return V, amt, dead


def test_stamp_schedule_boundaries():
    d = pd.DatetimeIndex(["2005-01-21", "2005-01-24", "2007-05-29", "2007-05-30", "2008-04-24",
                          "2008-09-18", "2008-09-19", "2023-08-25", "2023-08-28"])
    b, s = LE.stamp_arrays(d)
    assert list(b) == [0.002, 0.001, 0.001, 0.003, 0.001, 0.001, 0.0, 0.0, 0.0]
    assert list(s) == [0.002, 0.001, 0.001, 0.003, 0.001, 0.001, 0.001, 0.001, 0.0005]
    b5, s5 = LE.stamp_arrays(d, extra_bps=4.0)
    assert np.allclose(b5 - b, 4e-4) and np.allclose(s5 - s, 4e-4)


def test_stamp_hook_current_bitwise_equals_default():
    V, amt, dead = _synth()
    ad = LE.LHAdapter(V, amt, dead)
    W = np.full(V.shape, 1.0 / V.shape[1], np.float32)
    W[:, :10] = 0
    r0 = ST.simulate(W, ad, "open10", 1e7, rebalance=5, phase=1)
    r1 = ST.simulate(W, ad, "open10", 1e7, rebalance=5, phase=1, stamp_fn=LE.StampFnCurrent(ad.dates))
    assert (r0["rets"] - r1["rets"]).abs().max() == 0.0


def test_hist_stamp_charges_buy_side_before_2008_09_19():
    V, amt, dead = _synth()
    ad = LE.LHAdapter(V, amt, dead)
    W = np.full(V.shape, 1.0 / V.shape[1], np.float32)
    rec = ST.simulate(W, ad, "open10", 1e7, rebalance=5, phase=0, record=True,
                      stamp_fn=LE.StampFn(ad.dates))["record"]["daily"]
    first = rec.iloc[0]                     # 建仓日:成交日 2008-09-11,只买不卖,史实税率有 0.1% 买方税
    assert abs(first["cost_stamp"] - first["buy_value"] / first["aum"] * 0.001) < 1e-12


def test_frozen_delisted_holding_is_cleared_on_next_rebalance():
    V, amt, dead = _synth()
    j, od = 5, 12
    last = float(V.iloc[od - 1, j])
    V.iloc[od:, j] = np.float32(last * 0.5)            # 退市日起冻结价 = 最后收盘 x (1 - 50%)
    amt.iloc[od - 3:od, j] = np.nan                       # 退市前 3 日停牌:卖不掉
    amt.iloc[od:, j] = np.float32(LE.FROZEN_AMT)          # 冻结后可按冻结价卖出
    ad = LE.LHAdapter(V, amt, dead)
    W = np.full(V.shape, 1.0 / V.shape[1], np.float32)
    W[od - 4:, j] = 0                                     # 不再在池里
    res = ST.simulate(W, ad, "open10", 1e7, rebalance=5, phase=0, record=True)
    H = res["record"]["holdings"]
    held_at = np.flatnonzero(H[:, j] > 1e-9)
    # 成交日 te = t+1;最后一次持有不晚于退市日之后的第一个调仓日
    assert held_at.max() <= od + 5
    assert H[-1, j] == 0.0
    # 退市日当天的组合收益包含 -50% 的跳空(按持仓权重)
    t_gap = od - 2                                        # 信号日 t 的收益 = V[t+2]/V[t+1] - 1
    assert res["rets"].iloc[t_gap] < res["rets"].drop(res["rets"].index[t_gap]).mean()


def test_rowwise_quantile_pool():
    adv = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]], np.float32)
    elig = np.ones_like(adv, bool)
    m = LE._rowwise_quantile_mask(adv, elig, 0.8)
    assert m.sum() == 8 and not m[0, 0] and not m[0, 1]
    assert LE._rowwise_quantile_mask(adv, elig, 1.0).all()


def test_window_rows_alignment():
    d = pd.bdate_range("2020-01-01", periods=10)
    ix = LE.window_rows(d, str(d[2].date()), str(d[6].date()))
    assert list(ix) == [2, 3, 4, 5]                      # a <= t < b:成交日 te = t+1 ∈ (a, b]
