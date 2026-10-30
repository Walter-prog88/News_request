"""INC-T7 审查修复脚本(scripts/inc_t7_fix.py、inc_t7_report.revise_verdict)的合成数据测试(不读真实面板)。

PYTHONPATH=src python3 -m pytest -q tests/test_inc_t7_fix.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from afactor import lh_engine as LE
from afactor import strategy as ST
import inc_t7_grid as G
import inc_t7_report as R
import inc_t7_fix as F


def _synth(T=90, N=60, seed=0, locks=True):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-04", periods=T)
    codes = [f"{i:06d}.SZ" for i in range(N)]
    V = pd.DataFrame(np.cumprod(1 + rng.normal(0, 0.02, (T, N)), axis=0) * 10, index=dates, columns=codes)
    V.iloc[10:13, 3] = np.nan
    amt = pd.DataFrame(np.full((T, N), 3e8), index=dates, columns=codes)
    dead = pd.DataFrame(np.zeros((T, N), bool), index=dates, columns=codes)
    if locks:
        dead.iloc[20:24, 5] = True
        dead.iloc[30:33, 8] = True
    U = pd.DataFrame(rng.random((T, N)) > 0.1, index=dates, columns=codes)
    S = pd.DataFrame(rng.normal(size=(T, N)).astype(np.float32), index=dates, columns=codes)
    return LE.LHAdapter(V.astype(np.float32), amt.astype(np.float32), dead), U, S


def _shell(ad, U, owd=None):
    """不读盘的 MinSim:把合成面板塞进必要属性。"""
    m = object.__new__(F.MinSim)
    d, codes = ad.dates, ad.codes
    m.p, m.d, m.codes = ad, d, codes
    m.T, m.N = len(d), len(codes)
    m.mask0 = np.ones(m.T, bool)
    m.t_start = 0
    px = ad.raw("twap_open10").to_numpy(np.float64)
    m.PX = {"twap": px, "auction": px, "close": px}
    amt = ad.raw("amt_open10").to_numpy(np.float64)
    m.WIN = {"twap": amt, "auction_real": amt, "close": amt}
    m.owu = ad["one_word_up"].to_numpy(bool)
    m.owd = np.zeros((m.T, m.N), bool) if owd is None else owd
    m.owu_ol, m.owd_ol = m.owu.copy(), m.owd.copy()
    m.RC = px.copy()
    m.U = U.to_numpy(bool)
    m.CM = {"old": None, "Y1": None, "pess": None}
    m.stamp_switch = int(np.searchsorted(d.values, np.datetime64("2023-08-28")))
    m.i2021 = 0
    return m


def _W(ad, U, S, band=(0.2, 0.35), mask=None):
    shape = LE._Shape(ad.dates, ad.codes)
    return ST.build_weights(S, shape, "V1000", scheme="exclude", buffer=band, universe_override=U, date_mask=mask)


def test_minsim_fixed_matches_engine_bitwise_with_locks():
    ad, U, S = _synth()
    W = _W(ad, U, S)
    m = _shell(ad, U)
    for k, ph in ((1, 0), (5, 2), (5, 4)):
        ref = ST.simulate(W, ad, "open10", 1e7, rebalance=k, phase=ph)["rets"]
        mine = m.run_path(W, "fixed", k=k, phase=ph, cost="old", cap=1e7)["ret"]
        assert (mine.reindex(ref.index) - ref).abs().max() < 1e-15


def test_minsim_e2b_matches_engine_bitwise():
    ad, U, S = _synth(seed=3)
    m = _shell(ad, U)
    for keq, off in ((10, 0), (10, 3), (20, 1)):
        mask = np.arange(m.T) >= off
        W = _W(ad, U, S, mask=mask)
        ref = ST.simulate(G.E2bTarget(W, keq, off), ad, "open10", 1e7, rebalance=1, normalize=False,
                          date_mask=mask)["rets"]
        mine = m.run_path(W, "e2b", keq=keq, off=off, cost="old", cap=1e7)["ret"]
        assert (mine.reindex(ref.index) - ref).abs().max() < 1e-15


def _tiny(lock_up_day=1):
    """3 只股票、价格不变:t=0 -> te=1。股票 0 在 te=1 封涨停。"""
    T, N = 6, 3
    dates = pd.bdate_range("2021-01-04", periods=T)
    codes = ["000001.SZ", "000002.SZ", "000003.SZ"]
    V = pd.DataFrame(10.0, index=dates, columns=codes)
    amt = pd.DataFrame(1e10, index=dates, columns=codes)
    dead = pd.DataFrame(False, index=dates, columns=codes)
    dead.iloc[lock_up_day, 0] = True
    ad = LE.LHAdapter(V.astype(np.float32), amt.astype(np.float32), dead)
    U = pd.DataFrame(True, index=dates, columns=codes)
    W = np.full((T, N), 1.0 / 3)
    return ad, U, W


def test_directional_lock_blocks_buy_but_not_sell_and_dir2_reallocates():
    ad, U, W = _tiny()
    m = _shell(ad, U)
    sym = m.run_path(W, "fixed", k=1, cost="old", lock="sym", cap=1e7)["exposure"]
    dr = m.run_path(W, "fixed", k=1, cost="old", lock="dir", cap=1e7)["exposure"]
    d2 = m.run_path(W, "fixed", k=1, cost="old", lock="dir2", cap=1e7)["exposure"]
    i = 0                                            # t=0 -> te=1:股票 0 封涨停
    assert abs(sym.iloc[i] - 1.0) < 1e-12            # 双向不交易:预算归一给另外两只
    assert abs(dr.iloc[i] - 2.0 / 3) < 1e-12         # 方向性:买单被挡,留现金
    assert abs(d2.iloc[i] - 1.0) < 1e-12             # 被挡名字冻结,其余重新归一
    # 持有后封涨停:允许卖出(方向性),双向不交易规则下不能动
    ad2, U2, _ = _tiny(lock_up_day=3)
    m2 = _shell(ad2, U2)
    W2 = W.copy()
    W2[2:, 0] = 0.0                                  # 信号日 t=2 起股票 0 目标为 0,te=3 封涨停
    W2[2:, 1:] = 0.5
    # te=3 当天 dir 把股票 0 卖掉(换手大),sym 保持其持仓(换手小)
    t_dir = m2.run_path(W2, "fixed", k=1, cost="old", lock="dir", cap=1e7)["turnover"].iloc[2]
    t_sym = m2.run_path(W2, "fixed", k=1, cost="old", lock="sym", cap=1e7)["turnover"].iloc[2]
    assert t_dir > t_sym + 1e-6


def test_round_lots_rounds_to_hundred_shares_and_keeps_full_exit():
    ad, U, W = _tiny()
    m = _shell(ad, U)
    cap = 1e6
    m.RC = np.full((m.T, m.N), 25.0)                 # 一手 = 2500 元 = 0.0025 权重
    lotw = 100 * 25.0 / cap
    delta = np.array([0.0031, -0.0012, -0.0049])
    h = np.array([0.0, 0.0012, 0.0049])
    tgt = np.array([0.0031, 0.0, 0.0])               # 后两只是清仓
    out = m._round_lots(delta, tgt, h, 2, cap)
    assert abs(out[0] - lotw) < 1e-15                # 1.24 手 -> 1 手
    assert out[1] == delta[1] and out[2] == delta[2]  # 清仓允许零股
    out2 = m._round_lots(np.array([0.0012, 0.0, -0.002]), np.array([0.01, 0.01, 0.01]), np.array([0.009, 0.01, 0.012]), 2, cap)
    assert out2[0] == 0.0                            # 0.48 手 -> 不交易
    assert abs(out2[2] + lotw) < 1e-15               # -0.8 手 -> -1 手


def test_min_commission_adds_cost_only_for_small_orders():
    ad, U, W = _tiny()
    m = _shell(ad, U)
    cap = 1e6
    a = m.run_path(W, "fixed", k=1, cost="old", cap=cap)["cost"]
    b = m.run_path(W, "fixed", k=1, cost="old", cap=cap, min_comm=5.0)["cost"]
    # 第一天买入 3 笔、每笔约 33 万元,1bp 佣金 = 33 元 > 5 元:最低佣金不起作用
    assert abs(a.iloc[0] - b.iloc[0]) < 1e-15
    cap2 = 1e8
    c = m.run_path(W, "fixed", k=1, cost="old", cap=cap2)["cost"]
    d = m.run_path(W, "fixed", k=1, cost="old", cap=cap2, min_comm=5.0)["cost"]
    assert abs(a.iloc[0] - c.iloc[0]) < 1e-15        # 比例成本与规模无关
    assert d.iloc[0] - c.iloc[0] == 0.0 or d.iloc[0] >= c.iloc[0]


def test_paired_row_matches_manual():
    idx = pd.bdate_range("2021-01-04", "2025-12-31")
    rng = np.random.default_rng(0)
    a = pd.Series(rng.normal(0.0006, 0.01, len(idx)), index=idx)
    b = pd.Series(rng.normal(0.0004, 0.01, len(idx)), index=idx)
    r = F.paired_row(a, b, "full")
    assert abs(r["d_arith"] - float((a - b).mean() * 242)) < 1e-12
    assert abs(r["d_geo"] - (R.ann_geo(a) - R.ann_geo(b))) < 1e-12
    t, se = R.nw((a - b).to_numpy(), 20)
    assert abs(r["t_nw20"] - t) < 1e-12


def _crit(extra=None):
    rows = [
        dict(cell="k1|20/35", minute_pass=True, lh_pass=True, pass_all=True, C5=True, rule73_trigger=False,
             lh0619_clear_better=True, minute_min_dev_test=0.0035, lh_min_dev_test=0.0065),
        dict(cell="E2b_10", minute_pass=True, lh_pass=False, pass_all=False, C5=True, rule73_trigger=False,
             lh0619_clear_better=False, minute_min_dev_test=0.010, lh_min_dev_test=-0.0001),
        dict(cell="k5|30/30", minute_pass=False, lh_pass=True, pass_all=False, C5=False, rule73_trigger=True,
             lh0619_clear_better=True, minute_min_dev_test=-0.001, lh_min_dev_test=0.0045),
    ]
    return pd.DataFrame(rows)


def _price_csv(path, twap, auction):
    rows = []
    for price, vals in (("twap", twap), ("auction", auction)):
        for w, v in zip(("full", "dev", "test"), vals):
            rows.append(dict(panel="minute", signal="S1_turn20", price=price, cost="Y1", window=w, d_geo=v, t_nw20=2.0))
    pd.DataFrame(rows).to_csv(path, index=False)


def test_revise_verdict_downgrades_when_twap_edge_vanishes(tmp_path, monkeypatch):
    f = tmp_path / "p.csv"
    _price_csv(f, twap=(0.0002, -0.0008, 0.0019), auction=(0.0095, 0.0085, 0.0112))
    monkeypatch.setattr(R, "FIX_PRICE", f)
    C = _crit()
    mech = R._decide(C)
    assert mech["verdict"] == "有增量"
    C2, out = R.revise_verdict(C, mech)
    assert out["verdict"] == "证据不足" and out["verdict_mechanical_prereg"] == "有增量"
    assert out["revision"]["invalidated_lh_cells"] == ["k1|20/35"]
    assert not C2.loc[C2.cell == "k1|20/35", "pass_all"].iloc[0]
    assert C.loc[C.cell == "k1|20/35", "pass_all"].iloc[0]             # 机械表不被原地修改


def test_revise_verdict_keeps_verdict_when_twap_edge_survives(tmp_path, monkeypatch):
    f = tmp_path / "p.csv"
    _price_csv(f, twap=(0.0060, 0.0050, 0.0070), auction=(0.0095, 0.0085, 0.0112))
    monkeypatch.setattr(R, "FIX_PRICE", f)
    C = _crit()
    C2, out = R.revise_verdict(C, R._decide(C))
    assert out["verdict"] == "有增量" and out["revision"]["invalidated_lh_cells"] == []


def test_revise_verdict_is_noop_without_fix_ledger(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "FIX_PRICE", tmp_path / "missing.csv")
    C = _crit()
    mech = R._decide(C)
    C2, out = R.revise_verdict(C, mech)
    assert out is mech and C2 is C
