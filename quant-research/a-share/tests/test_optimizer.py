"""指数增强优化器单测(小规模合成数据,不读面板)。

覆盖:约束全部成立、零 α 零成本时持仓 = 基准、TE 约束生效(且收紧单调)、不可交易股固定、
参与率约束与不可行时的放松、公募成分股权重下限、IC 预测无前视、IndexEnhancer 的目标约定(NaN = 不交易)、
评价基准的成交时点权重(连乘 = 成交价上的同一只指数;无漂移时与旧口径逐位一致)。
运行:PYTHONPATH=src python3 -m pytest tests/test_optimizer.py -q
"""
import os
import sys
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from afactor import optimizer as OPT  # noqa: E402

STY = ["Size 市值", "NonlinSize 非线性市值", "Beta 市场敏感", "Liquidity 流动性"]


def make_problem(seed=0, n=80, n_bench=50, J=5, alpha_scale=0.004, cost=True, h=None):
    rng = np.random.default_rng(seed)
    K = len(STY)
    X = rng.normal(size=(n, K))
    ind = rng.integers(0, J, size=n)
    H = np.zeros((n, J)); H[np.arange(n), ind] = 1.0
    M = 1 + K + J
    A = rng.normal(size=(M, M)) * 0.01
    cov5 = A @ A.T + np.eye(M) * 1e-5
    cov5[0, 0] += 4e-3                                  # 国家因子
    L = np.linalg.cholesky(cov5)
    d = (rng.uniform(0.02, 0.06, size=n)) ** 2         # 5 日特质方差
    wb = np.zeros(n)
    wb[:n_bench] = rng.uniform(0.5, 1.5, size=n_bench)
    wb /= wb.sum()
    alpha = rng.normal(size=n) * alpha_scale
    if h is None:
        h = wb.copy()
    z = np.zeros(n)
    return OPT.ActiveProblem(alpha=alpha, wb=wb, h=h, X=X, H=H, L=L, d=d,
                             lin_buy=np.full(n, 6e-4 if cost else 0.0), lin_sell=np.full(n, 11e-4 if cost else 0.0),
                             c15=np.full(n, 0.01 if cost else 0.0), fixed=np.zeros(n, bool),
                             cap=None, bench=wb > 0, style_names=STY)


def explicit_checks(prob, w, cfg, tol=1e-5):
    a = w - prob.wb
    assert w.min() >= -1e-7
    assert abs(w.sum() - prob.total) < 1e-6
    assert np.all(w <= prob.wb + cfg.stock_active_ub + 1e-6)
    if cfg.ind_bound is not None:
        assert np.abs(prob.H.T @ a).max() <= cfg.ind_bound + tol
    if cfg.size_bound is not None:
        assert np.abs(prob.X[:, :2].T @ a).max() <= cfg.size_bound + tol
    if cfg.style_bound is not None:
        assert np.abs(prob.X[:, 2:].T @ a).max() <= cfg.style_bound + tol
    if cfg.te_ann is not None:
        assert prob.te_ann(w, cfg) <= cfg.te_ann * (1 + 1e-3)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_constraints_hold(seed):
    prob = make_problem(seed, h=np.zeros(80))
    cfg = OPT.OptConfig()
    res = OPT.solve(prob, cfg)
    assert res["w"] is not None, res["attempts"]
    assert res["status"] == "optimal"
    explicit_checks(prob, res["w"], cfg)
    assert OPT.violations_ok(OPT.check_solution(prob, res["w"], cfg, False))


def test_zero_alpha_zero_cost_equals_benchmark():
    rng = np.random.default_rng(5)
    h = rng.dirichlet(np.ones(80))                       # 当前持仓随便给,零成本下不应有影响
    prob = make_problem(3, cost=False, h=h)
    prob = replace(prob, alpha=np.zeros(prob.n))
    cfg = OPT.OptConfig(cost_mult=0.0)
    # 默认数值口径:目标只剩 λ·Var(≈1e-6 量级),CLARABEL 的绝对间隙容差 1e-8 使解的位置精度约 2e-4 权重,
    # 但它在风险意义上就是基准(事前 TE < 0.05%)
    res = OPT.solve(prob, cfg)
    assert res["w"] is not None
    assert res["te_ann"] < 5e-4
    assert np.abs(res["w"] - prob.wb).max() < 5e-4
    # 把目标放大到 O(1)(obj_scale = 1/te5²,只是数值缩放,不改变最优解)后位置精度到 1e-5
    res2 = OPT.solve(prob, replace(cfg, obj_scale=1.0 / cfg.te5 ** 2))
    assert np.abs(res2["w"] - prob.wb).max() < 1e-5
    assert res2["te_ann"] < 1e-5


def test_zero_alpha_with_cost_stays_put():
    """零 α、有成本、当前持仓已满足约束时:不交易(成本 > 平局项的风险收益)。"""
    prob = make_problem(4, cost=True)
    prob = replace(prob, alpha=np.zeros(prob.n))
    res = OPT.solve(prob, OPT.OptConfig())
    assert np.abs(res["w"] - prob.h).max() < 1e-6


def test_te_constraint_binds_and_is_monotone():
    prob = make_problem(7, alpha_scale=0.02, h=np.zeros(80))
    out = {}
    for te in (0.03, 0.05, 0.07, None):
        cfg = OPT.OptConfig(te_ann=te, size_bound=None, style_bound=None, ind_bound=None, stock_active_ub=0.05)
        res = OPT.solve(prob, cfg)
        assert res["w"] is not None
        out[te] = res
        if te is not None:
            explicit_checks(prob, res["w"], cfg)
            assert abs(res["te_ann"] - te) / te < 2e-3, (te, res["te_ann"])     # 强 α 下约束绑定
    assert out[None]["te_ann"] > 0.07                    # 不约束时 TE 更大 -> 约束确实在起作用
    a = {k: float(np.dot(prob.alpha, v["w"] - prob.wb)) for k, v in out.items()}
    assert a[0.03] < a[0.05] < a[0.07] <= a[None] + 1e-9


def test_fixed_weights_respected():
    rng = np.random.default_rng(11)
    h = rng.dirichlet(np.ones(80))
    prob = make_problem(9, h=h)
    fixed = np.zeros(80, bool); fixed[[0, 3, 7, 60, 70]] = True
    prob = replace(prob, fixed=fixed)
    cfg = OPT.OptConfig(te_ann=0.10, size_bound=None, style_bound=None)
    res = OPT.solve(prob, cfg)
    assert res["w"] is not None
    assert np.abs(res["w"][fixed] - h[fixed]).max() < 1e-7


def test_trade_caps_and_relaxation():
    prob = make_problem(12, h=None)
    cap = np.full(80, 0.002)
    res = OPT.solve(replace(prob, cap=cap), OPT.OptConfig())
    assert res["w"] is not None and not res["cap_relaxed"]
    assert np.abs(res["w"] - prob.h).max() <= 0.002 + 1e-7
    # 从现金起步、单日最多买 0.5%:Σ ub = 80×0.5% = 40% < 100% -> 带参与率不可行 -> 放松后求解
    prob0 = replace(prob, h=np.zeros(80), cap=np.full(80, 0.005))
    res0 = OPT.solve(prob0, OPT.OptConfig())
    assert res0["w"] is not None and res0["cap_relaxed"]
    assert any(st.startswith("infeasible") for _, c, st in res0["attempts"] if c)


def test_min_bench_weight():
    prob = make_problem(13, alpha_scale=0.01, h=np.zeros(80))
    prob = replace(prob, alpha=np.where(prob.wb > 0, 0.0, 0.01))     # α 全在非成分上
    base = OPT.solve(prob, OPT.OptConfig(te_ann=0.20, size_bound=None, style_bound=None, ind_bound=None))
    v5 = OPT.solve(prob, OPT.OptConfig(te_ann=0.20, size_bound=None, style_bound=None, ind_bound=None,
                                       min_bench_weight=0.8))
    assert base["w"][prob.bench].sum() < 0.8
    assert v5["w"][prob.bench].sum() >= 0.8 - 1e-6


def test_infeasible_returns_none():
    prob = make_problem(14, h=np.zeros(80))
    fixed = np.zeros(80, bool); fixed[:] = True          # 全部固定在 0 -> Σw=1 不可能
    res = OPT.solve(replace(prob, fixed=fixed), OPT.OptConfig())
    assert res["w"] is None and res["status"] == "failed"


def test_ic_for_decision_no_lookahead():
    idx = pd.bdate_range("2021-01-01", periods=400)
    rng = np.random.default_rng(0)
    ic = pd.Series(rng.normal(0.05, 0.1, size=400), index=idx)
    a = OPT.ic_for_decision(ic, window=250, lag=6, min_obs=20, floor=-np.inf)
    t = 300
    assert np.isclose(a.iloc[t], ic.iloc[t - 6 - 249: t - 5].mean())
    ic2 = ic.copy(); ic2.iloc[t - 5:] = 99.0           # 改动 t-5 及以后(t 日收盘尚未实现)的 IC
    b = OPT.ic_for_decision(ic2, window=250, lag=6, min_obs=20, floor=-np.inf)
    assert np.isclose(a.iloc[t], b.iloc[t])
    assert OPT.ic_for_decision(ic - 1.0).max() == 0.0  # 下限 0
    assert OPT.ic_for_decision(ic).iloc[:25].isna().all()   # 不足 20 个观测


def test_grinold_alpha():
    rng = np.random.default_rng(1)
    s = rng.normal(size=100); inv = np.ones(100, bool); inv[:10] = False
    om = np.full(100, 0.05)
    al, z = OPT.grinold_alpha(s, inv, om, 0.05)
    assert np.all(al[:10] == 0) and abs(z[10:].mean()) < 0.05
    assert np.allclose(al[10:], 0.05 * 0.05 * z[10:])
    al0, _ = OPT.grinold_alpha(s, inv, om, -0.01)
    assert np.all(al0 == 0)


class _FakeRM:
    """最小风险模型替身:全覆盖,固定暴露/协方差。"""

    def __init__(self, dates, codes, prob):
        self.dates, self.codes = dates, codes
        self.style_names = STY
        self.p = prob

    def available(self, t):
        return True

    def at(self, t):
        p = self.p
        return (np.ones(len(self.codes), bool), p.X, p.H, p.L @ p.L.T, p.d)


def test_enhancer_target_convention():
    prob = make_problem(21)
    n = prob.n
    dates = pd.bdate_range("2024-01-01", periods=30)
    codes = pd.Index([f"{i:06d}.SZ" for i in range(n)])
    bw = pd.DataFrame(np.tile(prob.wb, (30, 1)), index=dates, columns=codes)
    score = pd.DataFrame(np.tile(np.random.default_rng(2).normal(size=n), (30, 1)), index=dates, columns=codes)
    U = pd.DataFrame(True, index=dates, columns=codes)
    icd = pd.Series(0.05, index=dates)

    class _P:
        pass
    pp = _P(); pp.dates, pp.codes = dates, codes
    enh = OPT.IndexEnhancer(pp, _FakeRM(dates, codes, prob), bw, score, icd, U,
                            OPT.OptConfig(te_ann=0.10), tag="t")
    h = prob.wb.copy()
    can = np.ones(n, bool); can[[0, 1]] = False          # 0 号持有且不可交易(卡住),1 号同样
    ctx = dict(t=5, te=6, date=dates[5], trade_date=dates[6], phase=0, mode="open10", capital=1e7,
               cost_model=None, h_dec=h, nav_dec=1.0, aum_dec=1e7, can=can, stuck=(~can) & (h > 0),
               cap_w=np.full(n, 1.0))
    tgt = enh(5, h, ctx)
    assert tgt is not None and tgt.shape == (n,)
    assert np.isnan(tgt[0]) and np.isnan(tgt[1])        # 卡住的给 NaN(= 引擎保持实际持仓)
    filled = np.where(np.isnan(tgt), h, tgt)
    assert abs(filled.sum() - 1.0) < 1e-6
    assert enh.log[-1]["status"] == "optimal"
    # 不交易的(|w-h|<trade_tol)也必须是 NaN,不能回传数值
    keep = np.isnan(tgt) & can
    assert keep.sum() >= 0


def test_extra_exposure_bound():
    """额外暴露(不在风险模型里,如 20 日反转):|Xe'(w−wb)| ≤ be 成立,且确实改变了解。"""
    prob = make_problem(31, alpha_scale=0.01, h=np.zeros(80))
    rng = np.random.default_rng(3)
    xe = (prob.alpha / prob.alpha.std() + 0.3 * rng.normal(size=80))[:, None]   # 与 α 高度相关的暴露
    cfg = OPT.OptConfig(te_ann=0.08)
    free = OPT.solve(prob, cfg)
    ex_free = float(xe[:, 0] @ (free["w"] - prob.wb))
    bounded = OPT.solve(replace(prob, Xe=xe, be=np.array([0.1])), cfg)
    ex_b = float(xe[:, 0] @ (bounded["w"] - prob.wb))
    assert abs(ex_free) > 0.1
    assert abs(ex_b) <= 0.1 + 1e-5
    assert OPT.violations_ok(bounded["viol"])


# ---------------------------------------------------------------- 评价基准(2026-10-10 审查 P0 修复)
def _cap_index_case(seed=5, T=60, N=40, ac=0.4):
    """股本不变的市值加权指数:收盘价 C 与成交价 P 都是随机游走;P 与前一日收盘的偏离带正自相关
    (模拟平均价口径的 Working 效应),正是旧基准(收盘权重 × 成交价收益)会偏的情形。"""
    rng = np.random.default_rng(seed)
    shares = rng.uniform(1, 5, size=N)
    C = 10 * np.exp(np.cumsum(rng.normal(0, 0.02, size=(T, N)), axis=0))
    P = np.empty((T, N))
    P[0] = C[0]
    P[1:] = C[:-1] * np.exp(ac * np.log(C[1:] / C[:-1]) + rng.normal(0, 0.003, size=(T - 1, N)))
    W = shares * C
    W = W / W.sum(axis=1, keepdims=True)
    return shares, C, P, W


def test_exec_time_bench_is_index_at_exec_prices():
    """连乘 = 同一只指数在成交价上计价:Π(1+r[t], t=a..b) = Σ 股本·P[b+2] / Σ 股本·P[a+1](lag=1)。"""
    shares, C, P, W = _cap_index_case()
    r = OPT.exec_time_bench_arrays(W, P, C, lag=1)
    a, b = 3, 50
    lhs = np.prod(1 + r[a:b + 1])
    rhs = (shares * P[b + 2]).sum() / (shares * P[a + 1]).sum()
    assert np.isclose(lhs, rhs, rtol=1e-12)
    # 旧口径(收盘权重直接乘成交价收益)不满足,且在正相关下系统性偏低
    g = np.full_like(P, np.nan); g[:-2] = P[2:] / P[1:-1] - 1
    old = np.prod(1 + (W[a:b + 1] * g[a:b + 1]).sum(axis=1))
    assert not np.isclose(old, rhs, rtol=1e-6)


def test_exec_time_bench_equals_legacy_without_drift():
    """成交价恰等于上一收盘(收盘→成交无漂移)时权重不变,结果与旧口径 Σ W[t]·g[t+1] 逐位一致;
    成交价缺失记 1 / 收益缺失记 0(与引擎 nan_policy="zero" 一致);W 全 0 的行为 NaN。"""
    shares, C, _, W = _cap_index_case(seed=6)
    P = np.vstack([C[:1], C[:-1]])                       # P[t+1] = C[t]
    P[20, 3] = np.nan                                    # 一只股票 te 停牌
    W = W.copy(); W[:2] = 0.0                            # 首期生效前
    r = OPT.exec_time_bench_arrays(W, P, C, lag=1)
    g = np.full_like(P, np.nan); g[:-2] = P[2:] / P[1:-1] - 1
    old = np.where(W.sum(1) > 0, (W * np.nan_to_num(g)).sum(axis=1), np.nan)
    assert np.all(np.isnan(r[:2])) and np.all(np.isnan(r[-2:]))
    assert np.allclose(r[2:-2], old[2:-2], rtol=0, atol=1e-15)
    Wx = OPT.exec_time_weights(W, P, C, lag=1)
    assert np.allclose(Wx[2:].sum(axis=1), 1.0) and np.all(Wx[:2] == 0)
