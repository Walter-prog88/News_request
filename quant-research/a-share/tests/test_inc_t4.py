"""INC-T4 的单元测试:口径工具与成员集(不跑回测)。"""
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import inc_t4_common as C  # noqa: E402


def test_prereg_frozen():
    assert C.check_prereg() == C.PREREG_SHA


def test_ewm_identity_and_causal():
    rng = np.random.default_rng(0)
    D = pd.DataFrame(rng.normal(size=(60, 4)).astype(np.float32))
    D.iloc[10:15, 1] = np.nan
    assert C.ewm_smooth(D, 0) is D
    a = C.ewm_smooth(D, 5)
    D2 = D.copy()
    D2.iloc[40:] = 99.0                      # 改未来值不得影响过去
    b = C.ewm_smooth(D2, 5)
    np.testing.assert_allclose(a.iloc[:40].to_numpy(), b.iloc[:40].to_numpy())
    # min_periods=1、ignore_na:缺失日沿用上一个平滑值
    assert np.isfinite(a.iloc[12, 1]) and a.iloc[12, 1] == pytest.approx(a.iloc[9, 1])


def test_nw_t_matches_capacity_impl():
    import cap_capacity_curve as CAP
    x = np.random.default_rng(1).normal(0.001, 0.01, 800)
    assert C.nw_t(x, 10) == pytest.approx(CAP._nw_t(x, 10))


def test_paired_stats_mde():
    idx = pd.bdate_range("2021-01-01", "2025-12-31")
    rng = np.random.default_rng(2)
    a = pd.Series(rng.normal(0.0004, 0.01, len(idx)), idx)
    b = pd.Series(rng.normal(0.0, 0.01, len(idx)), idx)
    st = C.paired_stats(a, b, "test")
    assert st["mde"] == pytest.approx(2 * st["se"])
    assert st["lo95"] == pytest.approx(st["delta"] - st["mde"])
    d = (a - b)[(idx.year >= 2024)]
    assert st["delta"] == pytest.approx(d.mean() * 242)


def test_two_layer_partition():
    rng = np.random.default_rng(3)
    W1 = (rng.random((20, 50)) > 0.3).astype(np.float32)
    W2 = (rng.random((20, 50)) > 0.2).astype(np.float32)
    W1[5] = 0
    H, E = C.two_layer(W1, W2)
    assert not ((H > 0) & (E > 0)).any()                  # 不相交
    np.testing.assert_array_equal((H > 0) | (E > 0), W1 > 0)   # 并集 = 第一层
    s = H.sum(1)
    assert np.allclose(s[s > 0], 1.0) and s[5] == 0


def test_members_reproduced():
    f = C.TMP / f"{C.PFX}_members.json"
    if not f.exists():
        pytest.skip("成员清单尚未生成")
    m = json.load(open(f))["sets"]
    for s, n in C.EXPECT_N.items():
        assert len(m[s]) == n
        assert len(m[f"{s}_nopead"]) == C.EXPECT_NOPEAD[s]
        assert set(m[f"{s}_nopead"]) <= set(m[s])
    union = set().union(*[set(m[s]) for s in C.SPLITS])
    assert len(union) == 636


def test_nw_t_matches_naive_loop():
    """nw_t 与逐项循环实现一致(macOS Accelerate 的 matmul 会报伪 RuntimeWarning,数值不受影响)。"""
    x = np.random.default_rng(5).normal(0.0005, 0.01, 400)
    u = x - x.mean()
    n, bw = len(x), 10
    lrv = sum(u * u) / n
    for L in range(1, bw + 1):
        lrv += 2 * (1 - L / (bw + 1)) * sum(u[L:] * u[:-L]) / n
    assert C.nw_t(x, bw) == pytest.approx(x.mean() / np.sqrt(lrv / n), rel=1e-9)


def test_overlap_and_equal_weight_identity():
    import inc_t4_extra as X
    W = np.array([[0.5, 0.5, 0.0], [0.0, 0.5, 0.5]], np.float32)
    assert X.overlap(W, W, np.array([True, True])) == pytest.approx(1.0)
    W2 = np.array([[0.0, 0.5, 0.5], [0.0, 0.5, 0.5]], np.float32)
    assert X.overlap(W, W2, np.array([True, True])) == pytest.approx((0.5 + 1.0) / 2)


def test_verdict_rules_synthetic():
    import inc_t4_summary as S
    def mk(dev, test, full, t_test, t_full):
        rows = []
        for w, d, t in (("dev", dev, 2.5), ("test", test, t_test), ("full", full, t_full)):
            rows.append(dict(method="a", variant="v", baseline="b", window=w, delta=d, t_nw=t, mde=0.01))
        return pd.DataFrame(rows)
    assert S.verdict_for(mk(0.02, 0.015, 0.02, 2.4, 2.5), "a", "v", "b")["core"] == "有增量候选"
    assert S.verdict_for(mk(0.02, 0.005, 0.02, 2.4, 2.5), "a", "v", "b")["core"] == "证据不足"     # Δtest < 1pp
    assert S.verdict_for(mk(-0.01, -0.005, -0.01, -0.5, -1.0), "a", "v", "b")["core"] == "无增量"
    assert S.verdict_for(mk(0.01, -0.02, 0.0, -2.5, -1.0), "a", "v", "b")["core"] == "无增量"      # Δtest<0 且 |t|>=2


def test_hl_star_rule_consistent_with_ledger():
    f = C.LEDGER / f"{C.PFX}_checks.csv"
    if not f.exists():
        pytest.skip("checks 尚未生成")
    ck = pd.read_csv(f)
    r = ck[ck.check == "hl_rule_turnover"]
    star = int(ck[ck.check == "hl_star"]["value"].iloc[0])
    ok = [int(i.split("_hl")[1].split(" ")[0]) for i, p_ in zip(r["item"], r["passed"].astype(bool)) if p_]
    assert star == (min(ok) if ok else 40)


def test_engine_checks_all_passed():
    f = C.LEDGER / f"{C.PFX}_checks.csv"
    if not f.exists():
        pytest.skip("checks 尚未生成")
    ck = pd.read_csv(f)
    eng = ck[ck.check.isin(["baseline_reproduce", "exposure_le_1", "align_peak_k0", "fullpool_resid_pp", "eqv_weights", "v1rep"])]
    assert len(eng) >= 8 and eng["passed"].astype(bool).all()


def test_cells_aligned_and_bounded():
    """所有落盘的 (a)(b) 单元:exposure<=1、对齐峰值 k=0、日期连续路径从 2020-07 起。"""
    import glob
    fs = glob.glob(str(C.TMP / f"{C.PFX}_cell_*.parquet"))
    if not fs:
        pytest.skip("单元尚未生成")
    for f in fs[:60]:
        d = pd.read_parquet(f)
        assert int(d["align_peak_k"].iloc[0]) == 0
        assert float(d["exp_max_phase"].max()) <= 1 + 1e-9
        assert d.index.min() >= pd.Timestamp(C.SIM_START)


# ---------------------------------------------------------------- 审查修复(2026-10-11)新增
def test_audit_resid_ic_reproduces_old_ledger():
    """残差 IC 与 2026-08-25 账本 midband_composite.csv(稳健A 持出窗 2024-2025 h1)逐项一致;HL5 平滑后残差 IC 塌缩。"""
    f = C.LEDGER / f"{C.PFX}_diag_resid.csv"
    if not f.exists():
        pytest.skip("diag_resid 尚未生成")
    R = pd.read_csv(f)
    old = pd.read_csv(C.LEDGER / "midband_composite.csv")
    oa = old[old["切法"] == "稳健A"].set_index("组")
    g = lambda s, h, w: R[(R.series == s) & (R.h == h) & (R.window == w)].ic_mean.iloc[0]
    assert g("D_A_HL0", 1, "test") == pytest.approx(float(oa.loc["D中段组", "ic_mean"]), abs=2e-4)
    assert g("resid(D_A_HL0|COMP_v5)", 1, "test") == pytest.approx(float(oa.loc["D组对COMP_v5正交后", "ic_mean"]), abs=2e-4)
    # P1 结论:HL0 残差显著为正;HL5 平滑后 h1/h5 残差 IC 低于 HL0 的 1/3
    assert g("resid(D_A_HL0|COMP_v5)", 5, "full") > 0.012
    assert g("resid(D_A_HL5|COMP_v5)", 1, "full") < g("resid(D_A_HL0|COMP_v5)", 1, "full") / 3
    assert g("resid(D_A_HL5|COMP_v5)", 5, "full") < g("resid(D_A_HL0|COMP_v5)", 5, "full") / 2
    ms = pd.read_csv(C.LEDGER / f"{C.PFX}_diag_resid_misc.csv")
    c0 = ms[(ms.item == "xs_corr_pearson") & (ms.hl == 0)].value.iloc[0]
    c5 = ms[(ms.item == "xs_corr_pearson") & (ms.hl == 5)].value.iloc[0]
    assert c5 > c0 + 0.1                      # 平滑让 D 更像 COMP_v5
    ac = lambda s_, l: ms[(ms.item == f'rank_autocorr_lag{l}') & (ms.series == s_)].value.iloc[0]
    assert ac('smooth(resid_HL0,HL5)', 5) > ac('resid_HL0', 5) + 0.3        # 先正交再平滑:换手大幅下降
    assert g('smooth(resid(D_A_HL0|COMP_v5),HL5)', 5, 'full') > 0.8 * g('resid(D_A_HL0|COMP_v5)', 5, 'full')   # 且保留大部分 IC


def test_audit_h2b_identity_and_ceiling():
    """H2b:w_extra × 保留-额外剔除差 ≈ 实测 (b)−基线 毛增量(量级吻合);所有 D 叠加格成本前毛增量都 < 1pp。"""
    f = C.LEDGER / f"{C.PFX}_h2b.csv"
    g = C.LEDGER / f"{C.PFX}_gross_ceiling.csv"
    if not (f.exists() and g.exists()):
        pytest.skip("h2b / gross_ceiling 尚未生成")
    H = pd.read_csv(f)
    a = H[(H.set == "A") & (H.window == "full")].iloc[0]
    assert abs(a.implied_gain_w_x_gap - a.measured_d_gross_b_minus_base) < 0.0005      # 5bp 内
    assert a.weight_extra < 0.03 and a.required_gap_for_1pp > 0.3                      # 1pp 需要 >30pp 的差
    CE = pd.read_csv(g)
    assert (CE.max_d_gross_pp < 1.0).all() and (CE.n_cells_ge_1pp == 0).all()


def test_audit_verdict_columns():
    """§7.1 功效上限:(c) MDE 4.7pp>2pp 且 |Δ|<MDE -> 正式判定证据不足,纯 §8 规则列保持无增量;(a)(b) 不受影响。"""
    f = C.LEDGER / f"{C.PFX}_verdict.csv"
    if not f.exists():
        pytest.skip("verdict 尚未生成")
    v = pd.read_csv(f).set_index("method")
    assert "verdict_rule8" in v.columns and "cap_7_1_low_power" in v.columns
    assert v.loc["c", "cap_7_1_low_power"] and v.loc["c", "verdict"] == "证据不足" and v.loc["c", "verdict_rule8"] == "无增量"
    assert not v.loc["a", "cap_7_1_low_power"] and v.loc["a", "verdict"] == "无增量"
    assert not v.loc["b", "cap_7_1_low_power"] and v.loc["b", "verdict"] == "无增量"
    assert v.loc["b", "verdict_if_diag_control"] == "证据不足"       # 持仓匹配对照下规则标签会不同(披露项)
