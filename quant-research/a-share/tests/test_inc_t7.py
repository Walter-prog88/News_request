"""INC-T7 脚本的合成数据测试(不读真实面板,不算任何真实收益)。

PYTHONPATH=src python3 -m pytest -q tests/test_inc_t7.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from afactor import lh_engine as LE
from afactor import strategy as ST
from afactor.backtest import weights_exclude_buffer
import inc_t7_grid as G
import inc_t7_report as R


def _synth(T=80, N=70, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-04", periods=T)
    codes = [f"{i:06d}.SZ" for i in range(N)]
    V = pd.DataFrame(np.cumprod(1 + rng.normal(0, 0.02, (T, N)), axis=0).astype(np.float32) * 10,
                     index=dates, columns=codes)
    V.iloc[10:13, 3] = np.nan                                  # 停牌 / 缺价
    amt = pd.DataFrame(np.full((T, N), 1e9, np.float32), index=dates, columns=codes)
    dead = pd.DataFrame(np.zeros((T, N), bool), index=dates, columns=codes)
    U = pd.DataFrame(rng.random((T, N)) > 0.1, index=dates, columns=codes)
    S = pd.DataFrame(rng.normal(size=(T, N)).astype(np.float32), index=dates, columns=codes)
    return LE.LHAdapter(V, amt, dead), U, S


class _C:
    """run_cell 需要的最小上下文。"""

    def __init__(self, ad, mask=None):
        self.p, self.mask, self.stamp_fn = ad, mask, None
        self.T, self.N = len(ad.dates), len(ad.codes)


def _W(ad, U, S, band=(0.2, 0.35)):
    shape = LE._Shape(ad.dates, ad.codes)
    return ST.build_weights(S, shape, "V1000", scheme="exclude", buffer=band, universe_override=U)


def test_pool_ew_custom_matches_samefreq_benchmark_phases():
    ad, U, _ = _synth()
    P = ad.raw("twap_open10").to_numpy(np.float64)
    R_ref = ST._pool_ew_drift_returns(ad, "V1000", "open10", 5, 0, universe_override=U)
    for ph in range(5):
        reset = ((np.arange(len(ad.dates) - 2) - ph) % 5) == 0
        r = G.pool_ew_custom(P, U.to_numpy(bool), reset)
        assert np.max(np.abs(r - R_ref[ph])) < 1e-14


def test_run_cell_equals_run_weights_bitwise():
    ad, U, S = _synth()
    W = _W(ad, U, S)
    ref = ST.run_weights(W, ad, "open10", 1e7, rebalance=5)
    res, daily = G.run_cell(_C(ad), W, 5, None, capital=1e7)
    assert (res["rets"] - ref["rets"]).abs().max() == 0.0
    assert res["_dret"] < 1e-15
    assert np.allclose(daily["n_reb"].groupby(daily.index).first().to_numpy() >= 0, True)


def test_e1_theta_zero_equals_daily_ndarray():
    ad, U, S = _synth()
    W = _W(ad, U, S)
    r_nd = ST.simulate(W, ad, "open10", 1e7, rebalance=1)
    r_fn = ST.simulate(G.E1Target(W, 0.0), ad, "open10", 1e7, rebalance=1)
    assert (r_nd["rets"] - r_fn["rets"]).abs().max() == 0.0


def test_e1_skips_when_tau_below_theta_and_uses_h_dec():
    W = np.array([[0.5, 0.5, 0.0]])
    fn = G.E1Target(W, 0.06)
    ctx = dict(h_dec=np.array([0.48, 0.52, 0.0]))
    assert fn(0, np.array([0.0, 0.0, 1.0]), ctx) is None          # τ = 0.02 < 6%;位置参数 h 不参与判断
    ctx = dict(h_dec=np.array([0.40, 0.45, 0.15]))
    out = fn(0, np.zeros(3), ctx)
    assert out is not None and np.allclose(out, [0.5, 0.5, 0.0])


def test_e2b_incremental_and_full_days():
    W = np.array([[0.25, 0.25, 0.25, 0.25, 0.0]] * 3)
    fn = G.E2bTarget(W, keq=10, t_start=0)
    can = np.array([True, True, True, False, True])
    full = fn(0, np.zeros(5), dict(can=can, budget_dec=0.9, h_dec=np.zeros(5)))
    assert np.allclose(full, [0.3, 0.3, 0.3, 0.0, 0.0])            # 限于可成交、缩放到 budget_dec
    hd = np.array([0.3, 0.3, 0.0, 0.2, 0.2])                       # 股票 2 应持有未持有,股票 4 应剔除仍持有
    inc = fn(1, np.zeros(5), dict(can=can, budget_dec=1.0, h_dec=hd))
    assert inc[4] == 0.0 and inc[2] == 0.25
    assert np.isnan(inc[[0, 1, 3]]).all()
    assert fn(2, np.zeros(5), dict(can=can, budget_dec=1.0, h_dec=np.array([0.25] * 4 + [0.0]))) is None


def test_e2b_runs_in_engine_without_leverage():
    ad, U, S = _synth()
    W = _W(ad, U, S)
    r = ST.simulate(G.E2bTarget(W, 10, 0), ad, "open10", 1e7, rebalance=1, normalize=False, record=True)
    d = r["record"]["daily"]
    assert d["exposure"].max() <= 1 + 1e-9 and d["exposure"].iloc[5:].min() > 0.5


def test_sampled_semantics_state_only_updates_on_rebalance_days():
    rng = np.random.default_rng(1)
    T, N = 30, 80
    F = rng.normal(size=(T, N)).astype(np.float32)
    M = np.ones((T, N), bool)
    k, ph = 5, 2
    Fs = F.copy()
    Fs[(np.arange(T) - ph) % k != 0] = np.nan
    W = weights_exclude_buffer(Fs, M, 1, 0.2, 0.35)
    for t in range(T):
        if (t - ph) % k != 0 and t > 0:
            assert np.array_equal(W[t], W[t - 1])
    W1 = weights_exclude_buffer(F[ph:ph + 1], M[:1], 1, 0.2, 0.35)
    assert np.array_equal(W[ph], W1[0])                         # 首个调仓日 = 从空状态只看当天打分
    assert (W[:ph] == 0).all()                                  # 首个调仓日之前没有目标


def test_neighbors_table():
    assert sorted(R.neighbors("k5|20/35")) == sorted(["k1|20/35", "k10|20/35", "k5|20/30", "k5|20/45", "k5|30/45"])
    assert sorted(R.neighbors("k20|30/55")) == sorted(["k10|30/55", "k20|30/45", "k20|20/45"])
    assert sorted(R.neighbors("E1_6")) == ["E1_10", "E1_3"]
    assert R.neighbors("E2b_10") == ["E2b_20"]


def test_nw_matches_lh_engine():
    x = np.random.default_rng(3).normal(0.001, 0.01, 600)
    t, se = R.nw(x, 20)
    assert abs(t - LE.nw_t(x, 20)) < 1e-12
    assert abs(t * se - x.mean()) < 1e-15


def test_decomposition_identity_on_synthetic():
    idx = pd.bdate_range("2021-01-04", periods=300)
    rng = np.random.default_rng(5)
    def frame(scale):
        g = rng.normal(0.0005, 0.01, len(idx)); c = np.abs(rng.normal(0.0002, 0.0001, len(idx)))
        return pd.DataFrame(dict(ret=g - c, gross_ret=g, cost_total=c, bench=rng.normal(0, 0.01, len(idx)),
                                 b5=rng.normal(0, 0.01, len(idx))), index=idx)
    ps, _ = R.paired_stats(frame(1), frame(1))
    assert abs(ps["dec_resid"]) < 1e-12


def _frames(idx, seed):
    rng = np.random.default_rng(seed)
    g = rng.normal(0.0005, 0.01, len(idx))
    c = np.abs(rng.normal(0.0002, 0.0001, len(idx)))
    return pd.DataFrame(dict(ret=g - c, gross_ret=g, cost_total=c, cost_commission=c / 4, cost_stamp=c / 4,
                             cost_spread=c / 4, cost_impact=c / 4, turnover=np.abs(g), exposure=1.0, n_hold=100.0,
                             bench=rng.normal(0, 0.01, len(idx)), b5=rng.normal(0, 0.01, len(idx))), index=idx)


class _PD:
    def __init__(self, panel, idx):
        self.panel = panel
        self.S = {(R.CUR, "Y1"): _frames(idx, 1), ("k1|20/35", "Y1"): _frames(idx, 2),
                  ("k1|fp", "Y1"): _frames(idx, 3), ("k5|fp", "Y1"): _frames(idx, 4)}


def test_decomp_yearly_components_sum_to_delta():
    idx = pd.bdate_range("2021-01-04", "2023-12-29")
    pdat = _PD("minute", idx)
    pdat.S[(R.CUR, "Y1")]["b5"] = pdat.S[(R.CUR, "Y1")]["bench"]      # 现行格子的同频基准 = b5
    out = R.build_decomp_yearly({"minute": pdat}, cells=("k1|20/35",))
    assert len(out) == 4                                                # 3 个日历年 + 全期(year=0)
    resid = out["delta_ann"] - (out["rebal_ann"] + out["signal_ann"] + out["cost_ann"])
    assert resid.abs().max() < 1e-12


def test_wslice_exclusion_windows_per_panel():
    idx = pd.bdate_range("2017-01-02", "2025-12-31")
    df = pd.DataFrame({"x": 1.0}, index=idx)
    m = R.wslice(df, "minute", "full_exc")
    assert not ((m.index >= "2024-01-02") & (m.index <= "2024-02-07")).any() and "2024-02-08" in m.index
    lh = R.wslice(df, "lh", "full_exc")
    assert not (lh.index.year == 2018).any() and (lh.index.year == 2019).any()


def test_fp_control_identity_and_shape():
    idx = pd.bdate_range("2021-01-04", "2023-12-29")
    pdat = _PD("minute", idx)
    for k in (10, 20):
        pdat.S[(f"k{k}|fp", "Y1")] = _frames(idx, 10 + k)
    F, C = R.build_fp_control({"minute": pdat})
    assert set(F.k) <= {1, 10, 20} and (F.panel == "minute").all()
    r = C[(C.cell == "k1|20/35") & (C.window == "full")]
    assert len(r) == 1                                                     # 只有 k1|20/35 与 k1|fp 存在
    x = r.iloc[0]
    assert abs(x.d_minus_fp - (x.d_arith - x.fp_d_arith)) < 1e-15
    fr = F[(F.k == 1) & (F.window == "full")].iloc[0]
    assert abs(fr.fp_d_arith - float((pdat.S[("k1|fp", "Y1")]["ret"] - pdat.S[("k5|fp", "Y1")]["ret"]).mean() * 242)) < 1e-12
