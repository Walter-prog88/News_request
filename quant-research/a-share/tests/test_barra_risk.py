"""风险模型(src/afactor/barra.py)的合成数据单测:时序因果性、因子形式 = 稠密形式、回归识别。

不读面板,秒级跑完:python3 -m pytest tests/test_barra_risk.py -q(不再需要 PYTHONPATH=src)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))   # 与其它测试一致:自己插入 src

from afactor import barra as B   # noqa: E402


def test_cov_series_is_causal() -> None:
    """第 s 行协方差只能用 F[0..s]:改动 s 之后的行,s 及之前的输出必须逐位不变。"""
    rng = np.random.default_rng(0)
    F = rng.normal(size=(120, 6)) * 0.01
    C = B.factor_cov_series(F, 90, 90, 2, 5)
    F2 = F.copy()
    F2[80:] *= 7.0
    C2 = B.factor_cov_series(F2, 90, 90, 2, 5)
    assert np.array_equal(C[:80], C2[:80])
    assert not np.allclose(C[80], C2[80])


def test_ewma_no_lag_matches_manual() -> None:
    rng = np.random.default_rng(1)
    F = rng.normal(size=(50, 3))
    hl = 10.0
    C = B.factor_cov_series(F, hl, hl, nw_lags=0, horizon=1)
    lam = 0.5 ** (1 / hl)
    w = lam ** np.arange(49, -1, -1)
    manual = (F * w[:, None]).T @ F / w.sum()
    assert np.allclose(C[-1], manual)


def test_nw_and_horizon_scaling() -> None:
    """NW 2 阶 Bartlett:Σ = Γ0 + 2/3(Γ1+Γ1') + 1/3(Γ2+Γ2'),再 ×horizon。"""
    rng = np.random.default_rng(2)
    F = rng.normal(size=(200, 2))
    hl = 1e9                                       # 近似等权,便于手算
    C = B.factor_cov_series(F, hl, hl, nw_lags=2, horizon=5)[-1]
    T = len(F)
    g0 = F.T @ F / T
    g1 = F[1:].T @ F[:-1] / T
    g2 = F[2:].T @ F[:-2] / T
    manual = 5 * (g0 + 2 / 3 * (g1 + g1.T) + 1 / 3 * (g2 + g2.T))
    assert np.allclose(C, manual, rtol=1e-6)


def test_separate_half_lives_combine_vol_and_corr() -> None:
    rng = np.random.default_rng(3)
    F = rng.normal(size=(300, 3)) * np.array([0.01, 0.02, 0.03])
    Cv = B.factor_cov_series(F, 90, 90, 2, 1)
    Cc = B.factor_cov_series(F, 30, 30, 2, 1)
    C = B.factor_cov_series(F, 30, 90, 2, 1)
    assert np.allclose(np.diag(C[-1]), np.diag(Cv[-1]))
    d = np.sqrt(np.diag(C[-1])); dc = np.sqrt(np.diag(Cc[-1]))
    assert np.allclose(C[-1] / np.outer(d, d), Cc[-1] / np.outer(dc, dc))


def test_spec_var_is_causal_and_skips_missing() -> None:
    rng = np.random.default_rng(4)
    r = rng.normal(size=(60, 4)).astype(np.float32)
    r[10:20, 1] = np.nan
    v, n = B.ewma_spec_var(r, 90)
    r2 = r.copy(); r2[40:] = 9.0
    v2, _ = B.ewma_spec_var(r2, 90)
    assert np.array_equal(v[:40], v2[:40])
    assert n[-1, 1] == 50 and n[-1, 0] == 60
    assert np.isfinite(v[15, 1])                   # 缺失日沿用之前的估计,不变成 NaN


def test_decision_source_rows_respects_info_lag() -> None:
    pos_F = np.array([0, 1, 2, 4, 5])              # 面板第 3 天回归失败
    src = B.decision_source_rows(pos_F, 8, info_lag=2)
    assert src.tolist() == [-1, -1, 0, 1, 2, 2, 3, 4]
    # 决策日 i 用到的 F 行在面板中的位置必须 ≤ i−2
    for i, s in enumerate(src):
        if s >= 0:
            assert pos_F[s] <= i - 2


def test_shrink_extremes() -> None:
    rng = np.random.default_rng(5)
    N = 200
    sig = rng.uniform(0.01, 0.05, N)
    nobs = np.full(N, 100); nobs[:5] = 3
    mv = rng.lognormal(10, 1, N)
    mask = np.ones(N, bool)
    s0 = B.shrink_spec_vol(sig, nobs, mv, mask, shrink=0.0)
    assert np.allclose(s0[5:], sig[5:])
    s1 = B.shrink_spec_vol(sig, nobs, mv, mask, shrink=1.0)
    rk = pd.Series(mv).rank(method="first").to_numpy()
    b = np.minimum((rk - 1) * 10 // N, 9).astype(int)
    for k in range(10):
        assert np.allclose(s1[b == k], s1[b == k][0])          # 同桶同值
    # 观测不足的股票直接取桶均值(与 shrink 无关)
    assert np.allclose(s0[:5], s1[:5])


def test_psd_sqrt() -> None:
    rng = np.random.default_rng(6)
    A = rng.normal(size=(8, 5))
    C = A @ A.T                                    # 秩 5,Cholesky 必失败 -> 走特征分解
    Cp, L = B.psd_sqrt(C)
    assert np.allclose(L @ L.T, Cp)
    assert np.allclose(Cp, C, atol=1e-10)


def _fake_model(N=30, K=3, J=4, seed=7):
    rng = np.random.default_rng(seed)
    rm = B.RiskModel()
    rm.mode = "test"
    rm.dates = pd.DatetimeIndex(["2024-01-02"])
    rm.codes = pd.Index([f"{i:06d}.SZ" for i in range(N)])
    rm.style_names = [f"s{k}" for k in range(K)]
    rm.ind_names = [f"行业:{j}" for j in range(J)]
    rm.factor_names = ["Country"] + rm.style_names + rm.ind_names
    rm.K, rm.J = K, J
    X = rng.normal(size=(1, N, K)).astype(np.float32)
    rm.X = X
    rm.ind = rng.integers(-1, J, N).astype(np.int16)
    M = 1 + K + J
    A = rng.normal(size=(M, M)) * 0.01
    rm.cov = (A @ A.T)[None]
    rm.spec = (rng.uniform(0.01, 0.03, (1, N)) ** 2).astype(np.float32)
    rm.mask = np.ones((1, N), bool)
    rm.mask[0, -2:] = False
    rm.F = pd.DataFrame(rng.normal(size=(1, M)) * 0.01, index=rm.dates, columns=rm.factor_names)
    return rm


def test_factor_form_equals_dense_form() -> None:
    rm = _fake_model()
    N = len(rm.codes)
    rng = np.random.default_rng(8)
    w = np.where(rm.mask[0], rng.uniform(0, 1, N), 0.0); w /= w.sum()
    wb = np.where(rm.mask[0], 1.0, 0.0); wb /= wb.sum()
    m, X, H, cov, d = rm.at("2024-01-02")
    Bm = np.column_stack([np.ones(m.sum()), X, H])
    a = (w - wb)[m]
    dense = a @ (Bm @ cov @ Bm.T + np.diag(d)) @ a
    assert rm.active_risk(w, wb, "2024-01-02") == pytest.approx(np.sqrt(dense), rel=1e-10)
    det = rm.active_risk(w, wb, "2024-01-02", detail=True)
    assert det["var_factor"] + det["var_specific"] == pytest.approx(dense, rel=1e-10)
    e = rm.active_exposure(w, wb, "2024-01-02")
    assert np.allclose(e.to_numpy(), Bm.T @ a)
    assert abs(e["Country"]) < 1e-12


def test_active_risk_rejects_uncovered_weight() -> None:
    rm = _fake_model()
    w = np.zeros(len(rm.codes)); w[-1] = 1.0       # 全仓在覆盖池外
    with pytest.raises(ValueError, match="覆盖池外"):
        rm.active_risk(w, None, "2024-01-02")


def test_attribute_identity() -> None:
    """total = country + style + industry + specific,且因子部分 = (B^T a)·f。"""
    rm = _fake_model()
    N = len(rm.codes)
    rng = np.random.default_rng(9)
    a = np.where(rm.mask[0], rng.normal(size=N) * 0.01, 0.0)
    a -= np.where(rm.mask[0], a.sum() / rm.mask[0].sum(), 0.0)
    r = rng.normal(size=N) * 0.02
    A = pd.DataFrame([a], index=rm.dates, columns=rm.codes)
    Rr = pd.DataFrame([r], index=rm.dates, columns=rm.codes)
    out, fac = rm.attribute(A, Rr, by_factor=True)
    row = out.iloc[0]
    assert row[["country", "style", "industry", "specific"]].sum() == pytest.approx(row["total"])
    m = rm.mask[0]
    Bm = np.column_stack([np.ones(m.sum()), np.asarray(rm.X[0])[m], rm._onehot(rm.ind[m])])
    assert np.allclose(fac.iloc[0].to_numpy(), (Bm.T @ a[m]) * rm.F.iloc[0].to_numpy())


def test_regression_recovers_noise_free_factor_returns() -> None:
    """无噪声合成数据:风险模型版回归应精确还原因子收益,并满足行业市值加权和为零。"""
    rng = np.random.default_rng(10)
    T, N, K, J = 3, 400, 3, 5
    dates = pd.bdate_range("2024-01-01", periods=T)
    codes = [f"{i:06d}.SZ" for i in range(N)]
    Xs = rng.normal(size=(T, N, K)).astype(np.float32)
    ind = rng.integers(0, J, N).astype(np.int16)
    cap = rng.lognormal(10, 1, (T, N))
    f_c = rng.normal(size=T) * 0.01
    f_s = rng.normal(size=(T, K)) * 0.005
    f_i = rng.normal(size=(T, J)) * 0.01
    for t in range(T):                              # 让行业收益满足约束
        cs = np.array([cap[t][ind == j].sum() for j in range(J)])
        f_i[t] -= (cs @ f_i[t]) / cs.sum()
    r = f_c[:, None] + np.einsum("tnk,tk->tn", Xs.astype(np.float64), f_s) + f_i[:, ind]
    R = pd.DataFrame(r, index=dates, columns=codes)
    ones = pd.DataFrame(True, index=dates, columns=codes)
    mv = pd.DataFrame(cap, index=dates, columns=codes)
    F, resid, diag, ok = B.risk_factor_regression(
        R, Xs, ind, J, ones, ones, mv, [f"s{k}" for k in range(K)], [f"行业:{j}" for j in range(J)])
    assert ok.all()
    assert np.allclose(F.iloc[:, 0], f_c, atol=1e-10)
    assert np.allclose(F.iloc[:, 1:1 + K], f_s, atol=1e-10)
    assert np.allclose(F.iloc[:, 1 + K:], f_i, atol=1e-10)
    assert np.nanmax(np.abs(resid)) < 1e-6


def test_regression_skips_day_when_a_style_is_mostly_missing() -> None:
    """MIN_STYLE_COVER 门槛:某风格在估计池内非缺失率 < 50% 的日子不出因子收益(2020 年动量全缺的情形)。
    回归必须拿未填充的暴露;先填 0 会让门槛失效(2026-10 构建时踩过)。"""
    rng = np.random.default_rng(11)
    T, N, K, J = 2, 300, 2, 3
    dates = pd.bdate_range("2024-01-01", periods=T)
    codes = [f"{i:06d}.SZ" for i in range(N)]
    Xs = rng.normal(size=(T, N, K)).astype(np.float32)
    Xs[0, :200, 1] = np.nan                         # 第 0 天第 2 个风格只有 1/3 有值
    Xs[1, :50, 1] = np.nan                          # 第 1 天 5/6 有值 -> 照常回归,缺失按 0 处理
    R = pd.DataFrame(rng.normal(size=(T, N)) * 0.02, index=dates, columns=codes)
    ones = pd.DataFrame(True, index=dates, columns=codes)
    mv = pd.DataFrame(rng.lognormal(10, 1, (T, N)), index=dates, columns=codes)
    ind = rng.integers(0, J, N).astype(np.int16)
    F, resid, diag, ok = B.risk_factor_regression(
        R, Xs, ind, J, ones, ones, mv, ["s0", "s1"], [f"行业:{j}" for j in range(J)])
    assert ok.tolist() == [False, True]
    assert np.isnan(resid[0]).all() and np.isfinite(resid[1]).all()


def test_robust_clip_uses_estimation_universe_and_skips_zero_mad() -> None:
    dates = pd.bdate_range("2024-01-01", periods=2)
    codes = [f"c{i}" for i in range(7)]
    x = pd.DataFrame([[1.0, 2.0, 3.0, 2.0, 1.0, 2.0, -191.0],      # 第 0 天:-191 是数据异常
                      [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 100.0]],       # 第 1 天:u 内 MAD=0 -> 不截
                     index=dates, columns=codes)
    u = pd.DataFrame(True, index=dates, columns=codes)
    u.iloc[:, 6] = [True, False]
    y = B.robust_clip(x, u, 5.0)
    med = np.median(x.iloc[0]); mad = 1.4826 * np.median(np.abs(x.iloc[0] - med))
    assert y.iloc[0, 6] == pytest.approx(med - 5 * mad)
    assert (y.iloc[0, :6] == x.iloc[0, :6]).all()
    assert (y.iloc[1] == x.iloc[1]).all()


def test_build_exposure_defaults_are_unchanged_signature() -> None:
    """搬迁核对之外的保险:原版调用方式 build_exposures(p, u) / zscore(x, u, w) 的默认参数不得变。"""
    import inspect
    sig = inspect.signature(B.build_exposures)
    assert sig.parameters["cover"].default is None and sig.parameters["raw_clip"].default is None
    assert sig.parameters["size_clip"].default is True
    zs = inspect.signature(B.zscore).parameters
    assert zs["cover"].default is None and zs["clip"].default is True
    assert B.RiskModel().variant == "v3" and B.RiskModel(**B.RiskModel.VARIANTS["v1"]).tag == "_v1"
