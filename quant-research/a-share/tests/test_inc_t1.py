"""INC-T1(执行优化)的门槛与实现测试(预注册 research/INC_T1_PREREG.md §3.3)。

1. 窗口聚合:合成一日 241 根 bar,核对 twap / vwap / amt / 单根 bar 窗口取 close 的定义;
2. 门槛账本:inc_t1_window_build_checks / inc_t1_engine_checks / inc_t1_tradelist_checks 全部通过;
3. 第一层腿的分项:成交比例、未成交顺延、池子级 / 个股相对拆分在合成数据上的手算值;
4. 方案组装:买卖分离对角线 = 单窗口;分批 = 两个半量腿;
5. Newey-West t 与 statsmodels HAC 一致。
需要真实数据的测试在缺少文件时跳过。
"""
from __future__ import annotations
import os, sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import inc_t1_build_windows as BW          # noqa: E402
import inc_t1_common as C                  # noqa: E402
import inc_t1_tradelist as TL              # noqa: E402

LEDGER = ROOT / "research" / "ledger"


# ------------------------------------------------------------------ 1 窗口聚合定义
def _synthetic_day(n=3, seed=0):
    rng = np.random.default_rng(seed)
    P = 10 * np.exp(np.cumsum(rng.normal(0, 1e-3, (241, n)), axis=0))
    O = P * (1 + rng.normal(0, 2e-4, (241, n)))
    V = rng.uniform(1e4, 1e5, (241, n))
    V[238:240] = 0.0                       # 14:58-59 无量
    V[5:8, 1] = 0.0                        # 股票 1 在 W1 内有 3 分钟无成交
    V[1:11, 2] = 0.0                       # 股票 2 在 W1 全无成交
    return P, O, V


def test_day_windows_definitions():
    P, O, V = _synthetic_day()
    out = BW.day_windows(P, O, V)
    mid = (P + O) / 2
    # twap_W1:有成交分钟的 mid 均值;全无成交退化为 mid 均值
    for j in range(3):
        m = mid[1:11, j]; v = V[1:11, j]
        exp = m[v > 0].mean() if (v > 0).any() else m.mean()
        assert np.isclose(out["twap_W1"][j], exp, rtol=1e-12)
    # vwap_W5 = Σamt / Σ(amt/mid)
    j = 0
    exp = V[121:181, j].sum() / (V[121:181, j] / mid[121:181, j]).sum()
    assert np.isclose(out["vwap_W5"][j], exp, rtol=1e-12)
    # 单根 bar 窗口取 close,不取 mid
    assert np.array_equal(out["twap_W0"], P[0]) and np.array_equal(out["vwap_W8"], P[240])
    # 成交额分窗口加总 = 全日
    tot = sum(out[f"amt_{w}"] for w in ["W0", "W1", "W2", "W3", "W4", "W5", "W6", "W7", "W8"]) + out["amt_GAP"]
    assert np.allclose(tot, V.sum(axis=0), rtol=1e-12)
    assert np.allclose(out["amt_W9"], V.sum(axis=0), rtol=1e-12)
    assert np.allclose(out["amt_W10"] + out["amt_W11"], out["amt_W9"], rtol=1e-12)
    # 股票 2 的 W1 无成交:vwap 为 NaN(使用时退回 twap)
    assert np.isnan(out["vwap_W1"][2])
    # 九个时点 = bar close
    for i in BW.POINTS:
        assert np.array_equal(out[f"pt{i}"], P[i])


def test_windows_partition_session():
    """W1..W7 + GAP + W0 + W8 恰好不重不漏覆盖 241 根 bar;W10 ∪ W11 = W9。"""
    cover = np.zeros(241, int)
    for w in ["W0", "W1", "W2", "W3", "W4", "W5", "W6", "W7", "W8"]:
        lo, hi = BW.WINDOWS[w]
        cover[lo:hi] += 1
    cover[BW.GAP[0]:BW.GAP[1]] += 1
    assert (cover == 1).all()
    assert BW.WINDOWS["W10"][1] == BW.WINDOWS["W11"][0] and BW.WINDOWS["W9"] == (0, 241)


# ------------------------------------------------------------------ 2 门槛账本
def _ledger(name):
    f = LEDGER / name
    if not f.exists():
        pytest.skip(f"缺 {name}")
    return pd.read_csv(f)


def test_window_build_gates_pass():
    d = _ledger("inc_t1_window_build_checks.csv")
    g = d[d.check.isin(["reproduce", "reproduce_bitwise", "sanity"])]
    assert len(g) >= 37 and g.passed.all(), g[~g.passed]


def test_engine_gates_pass():
    """无后缀族(修复前口径,14 个模式)与 _seal 族(审查 P0 修复后,14 个模式)的引擎断言都必须全过;
    金标准只对无后缀的 W1_twap(封板规则开启后 W1_twap 不再逐位等于金标准,只记信息性行)。"""
    d = _ledger("inc_t1_engine_checks.csv")
    assert d.passed.all(), d[~d.passed]
    sealed = d["mode"].str.endswith("_seal")
    assert d.loc[~sealed, "mode"].nunique() == 14
    if sealed.any():
        assert d.loc[sealed, "mode"].nunique() == 14
        assert (d.loc[sealed, "check"] == "exposure_max_comp_exclude").sum() == 14
    gold = d[d.check == "golden_a_comp_exclude_std_full_max_abs_diff"]
    assert len(gold) == 1 and float(gold.value.iloc[0]) <= 1e-6


def test_tradelist_gates_pass():
    d = _ledger("inc_t1_tradelist_checks.csv")
    main = d[d.check.isin(["captured_turnover_vs_record_maxabs", "capture_path_bitwise_same"])]
    assert main.passed.all(), main[~main.passed]
    assert set(d.strategy) == set(C.STRAT_KEYS)


# ------------------------------------------------------------------ 3 第一层腿的分项(合成)
class _FakeCtx:
    """最小面板:T=6 日、N=3 只;窗口 W1 的价格与成交额手工给定。"""

    def __init__(self):
        self.T, self.N = 6, 3
        self.close = np.full((6, 3), 10.0)
        self.U = np.ones((6, 3), bool)
        self.hs = np.full((6, 3), 0.001)
        self.sig = np.full((6, 3), 0.02)
        self.adv = np.full((6, 3), 1e8)
        self.P = np.array([[10.0, 10, 10], [10.2, 9.9, 10], [10.4, 9.8, 10], [10.6, 9.7, 10],
                           [10.8, 9.6, 10], [11, 9.5, 10]], np.float32)
        self.A = np.full((6, 3), 1e6, np.float32)
        self.A[1, 0] = 2e5                 # t=0 买入股票 0:当日上限 0.1*2e5/1e7 = 0.002
        self.years = np.array([2021] * 6)

    def px(self, w, kind):
        return self.P

    def amt(self, w):
        return self.A

    def amt_exante(self, w):
        return self.A

    def M(self, w, kind, k):
        with np.errstate(invalid="ignore"):
            R = self.P[k:].astype(np.float64) / self.close[:self.T - k] - 1
        out = np.full(self.T, np.nan)
        out[:self.T - k] = R.mean(axis=1)
        return out

    def s_pool(self, w):
        return np.full(self.T, 1.25)

    def s_stock(self, w):
        return np.full((self.T, self.N), 1.25, np.float32)


def test_leg_components_hand_calc():
    ctx = _FakeCtx()
    X = 1e7
    tr = dict(t=np.array([0, 0]), i=np.array([0, 1]), dw=np.array([0.004, -0.001]))
    out = TL.leg_components(ctx, tr, X, ("W1", 1, 1.0), "vwap", "pov")
    # 买入股票 0:w=0.004,f = 0.002/0.004 = 0.5;r1 = 10.2/10-1 = 0.02,r2 = 0.04;M1 = mean(0.02,-0.01,0)=1/300
    M1 = (0.02 - 0.01 + 0.0) / 3; M2 = (0.04 - 0.02 + 0.0) / 3
    exp_rel = 0.004 * (0.02 - M1) / 5
    assert np.isclose(out["px_rel"][0, 0], exp_rel, rtol=1e-5)
    assert np.isclose(out["px_mkt"][0, 0], 0.004 * M1 / 5, rtol=1e-5)
    assert np.isclose(out["unf_tot"][0, 0], 0.004 * 0.5 * (0.04 - 0.02) / 5, rtol=1e-5)
    assert np.isclose(out["unf_mkt"][0, 0], 0.004 * 0.5 * (M2 - M1) / 5, rtol=1e-5)
    assert np.isclose(out["w_unfilled"][0, 0], 0.002 / 5)
    c = 0.02 * np.sqrt(X / 1e8)
    exp_imp = (c * 0.002 ** 1.5 * 1.25 + c * 0.002 ** 1.5 * 1.25) / 5
    assert np.isclose(out["imp_pool"][0, 0], exp_imp, rtol=1e-5)
    assert np.isclose(out["spread"][0, 0], 0.004 * 0.001 / 5)
    # 卖出股票 1:全成交;s=-1, r1 = -0.01 -> 成本 = -1*0.001*(-0.01 - M1)
    assert np.isclose(out["px_rel"][0, 1], -0.001 * (-0.01 - M1) / 5, rtol=1e-5)
    assert out["unf_tot"][0, 1] == 0.0 and out["w_capped"][0, 1] == 0.0


def test_scheme_legs_and_diagonal():
    assert TL.scheme_legs("S:W5|B:W5") == TL.scheme_legs("W5")
    sl = TL.scheme_legs("S:W1|B:W7")
    assert sl == {"sell": ["W1|k1|1"], "buy": ["W7|k1|1"]}
    assert TL.scheme_legs("W3+W7")["buy"] == ["W3|k1|0.5", "W7|k1|0.5"]
    assert TL.scheme_legs("D2")["sell"] == ["W1|k1|0.5", "W1|k2|0.5"]
    assert len(TL.SCHEMES) == 44
    assert set(C.PRIMARY) <= set(TL.SCHEMES)
    legs_needed = {lg for s in TL.SCHEMES for v in TL.scheme_legs(s).values() for lg in v}
    have = {f"{w}|k{k}|{phi:g}" for (w, k, phi) in TL.LEGS}
    assert legs_needed <= have


# ------------------------------------------------------------------ 4 统计
def test_nw_t_matches_statsmodels():
    sm = pytest.importorskip("statsmodels.api")
    rng = np.random.default_rng(1)
    e = rng.normal(0, 1, 800)
    x = 0.1 + e + 0.5 * np.r_[0, e[:-1]]
    res = sm.OLS(x, np.ones_like(x)).fit(cov_type="HAC", cov_kwds=dict(maxlags=5, use_correction=False))
    assert np.isclose(C.nw_t(x, 5), float(res.tvalues[0]), rtol=1e-5)


# ------------------------------------------------------------------ 5 审查 P0:窗口内封板
def test_mode_name_roundtrip():
    for seal in (False, True):
        for lag in (1, 2):
            m = C.mode_name("W1", "twap", lag, seal)
            d = C.parse_mode(m)
            assert d == dict(window="W1", kind="twap", lag=lag, seal=seal), (m, d)
    assert C.mode_name("W5", "vwap", 1, True) == "inc_t1_W5_vwap_seal"
    assert C.mode_name("W5", "vwap") == "inc_t1_W5_vwap"          # 无后缀 = 修复前口径(账本里保留)


def test_leg_components_sealed_roll(monkeypatch):
    """买入股票 0 在 t=0 的成交日 te=1 被整窗口涨停封死:按 dir 口径整条腿顺延到 te=2 同窗口(只顺延一次),
    分项用 te=2 的价格 / 价差 / 冲击,未成交为 0;卖出股票 1 同样封死涨停:dir 不挡(卖得出),any 挡。"""
    ctx = _FakeCtx()
    ctx.p = object()
    up = np.zeros((6, 3), bool); dn = np.zeros((6, 3), bool)
    up[1, 0] = True; up[1, 1] = True
    monkeypatch.setattr(C, "window_seal", lambda p, w, seal="any": (up, dn))
    X = 1e7
    ctx.A[:] = 1e6                                          # 取消截量,只看封板顺延
    tr = dict(t=np.array([0, 0]), i=np.array([0, 1]), dw=np.array([0.004, -0.001]))
    M1 = (0.02 - 0.01 + 0.0) / 3; M2 = (0.04 - 0.02 + 0.0) / 3
    o_none = TL.leg_components(ctx, tr, X, ("W1", 1, 1.0), "vwap", "pov", "none")
    o_dir = TL.leg_components(ctx, tr, X, ("W1", 1, 1.0), "vwap", "pov", "dir")
    o_any = TL.leg_components(ctx, tr, X, ("W1", 1, 1.0), "vwap", "pov", "any")
    # none:买入股票 0 在 te=1 价成交
    assert np.isclose(o_none["px_rel"][0, 0], 0.004 * (0.02 - M1) / 5, rtol=1e-5)
    assert o_none["n_sealed"].sum() == 0
    # dir:买入被挡 -> 用 te=2 的价:r2 = 0.04, M2;卖出不挡
    assert np.isclose(o_dir["px_rel"][0, 0], 0.004 * (0.04 - M2) / 5, rtol=1e-5)
    assert np.isclose(o_dir["n_sealed"][0, 0], 1 / 5) and o_dir["n_sealed"][0, 1] == 0.0
    assert np.isclose(o_dir["w_sealed"][0, 0], 0.004 / 5)
    assert o_dir["unf_tot"][0, 0] == 0.0 and o_dir["w_unfilled"][0, 0] == 0.0
    assert np.isclose(o_dir["px_rel"][0, 1], o_none["px_rel"][0, 1])           # 卖出未被挡:与 none 相同
    # any:卖出股票 1(涨停封死)也被挡 -> 用 te=2 价 r2 = -0.02
    assert np.isclose(o_any["n_sealed"][0, 1], 1 / 5)
    assert np.isclose(o_any["px_rel"][0, 1], -0.001 * (-0.02 - M2) / 5, rtol=1e-5)


def test_window_seal_real_example():
    """审查给的例子:002222.SZ 2021-12-23,W1 在 −0.27% 附近成交,W5 整窗口贴在 +9.99% 涨停价上。"""
    pytest.importorskip("pyarrow")
    try:
        from afactor.panel import Panel
        p = Panel()
        up1, dn1 = C.window_seal(p, "W1", "any")
        up5, dn5 = C.window_seal(p, "W5", "any")
    except Exception as e:                                   # 缺数据 / 缓存
        pytest.skip(f"缺面板或窗口缓存:{e}")
    d = p.dates.get_loc(pd.Timestamp("2021-12-23")); j = list(p.codes).index("002222.SZ")
    assert up5[d, j] and not dn5[d, j]
    assert not up1[d, j]
    # 一字板(全日封死)应在 W9 也封死;W9 的封板掩码与引擎 one_word 的重合约 93%(实测 0.932):
    # 引擎的一字板允许日内振幅 < 0.4%,窗口规则要求最低价 ≥ 涨停幅度的 98.5%,所以引擎多判了一些
    # 价格贴在涨停价 0.4% 内、最低价略低于 98.5% 线的日子(引擎本来就把这些日子判不可成交)
    up9, dn9 = C.window_seal(p, "W9", "any")
    dead = (p["one_word_up"] | p["one_word_down"]).to_numpy(bool)
    assert (up9 | dn9)[dead].mean() > 0.9
    assert ((up9 | dn9) & ~dead).sum() < 0.01 * (up9 | dn9).sum()      # 反向:W9 封死而引擎不判一字板的极少


def test_layer2_same_convention_differences():
    """Every reported difference must subtract the declared reference, including sealed families."""
    import inc_t1_engine_modes as EM
    d = _ledger('inc_t1_layer2_grid.csv')
    keys = ['strategy', 'X', 'Y', 'period']
    for _, g in d.groupby(keys):
        bymode = g.set_index('mode')
        for r in g.itertuples():
            ref = EM.ref_modes_for(r.mode)['ref']
            b = bymode.loc[ref]
            assert r.ref_mode == ref
            assert np.isclose(r.d_excess_vs_ref_pp, r.excess_geo_pp - b.excess_geo_pp, atol=1e-10)
            assert np.isclose(r.total_diff_vs_ref_pp, (r.total_close_ann - b.total_close_ann) * 100, atol=1e-10)


def test_review_dispositions_and_persistent_seals():
    s = _ledger('inc_t1_summary.csv')
    e = s[s.strategy == 'COMP_v5_exclude']
    assert (e[e.scheme.isin(['W5', 'W9'])].verdict == '有增量').all()
    assert (e[e.scheme.isin(['W7', 'W8'])].verdict == '证据不足').all()
    r = e[(e.scheme == 'W7') & (e.X == 1e9)].iloc[0]
    assert r.verdict_mechanical == '有增量' and r.rollN_t < 2
    # This diagnostic must never silently overwrite the preregistered mechanical outcome.
    changed = s.verdict != s.verdict_mechanical
    assert changed.sum() == 1


def test_seal_diagnostic_reproduces_single_roll():
    r = _ledger('inc_t1_seal_multiroll.csv')
    g = _ledger('inc_t1_layer1_grid.csv')
    g = g[(g.strategy == 'COMP_v5_exclude') & (g.variant == 'main')]
    for v in r[r['mode'] == 'roll1'].itertuples():
        h = g[(g.X == v.X) & (g.scheme == v.window) & (g.period == v.period)]
        assert len(h) == 1
        assert np.isclose(h.d_px_rel_pp.iloc[0], v.px_rel_saving_pp, atol=1e-9)
