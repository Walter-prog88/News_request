from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from index_timing.statistics import (
    benjamini_hochberg,
    bootstrap_alpha,
    bootstrap_regression_alpha,
    hac_alpha,
    max_drawdown,
    paired_neural_increment_test,
    performance_metrics,
    stationary_block_bootstrap,
)
from index_timing.types import ValidationStatus
from index_timing.validation import (
    ValidationEvidence,
    assert_prefix_invariant,
    classify_validation,
    neural_increment_is_material,
    purged_expanding_walk_forward,
)


def test_performance_metrics_and_hac_alpha_are_economic() -> None:
    dates = pd.date_range("2020-01-02", periods=600, freq="B")
    benchmark = pd.Series(np.tile([0.01, -0.005, 0.002], 200), index=dates)
    strategy = benchmark + 0.0001
    metrics = performance_metrics(strategy, benchmark_returns=benchmark)
    assert metrics["n_obs"] == 600
    assert metrics["alpha"] == pytest.approx(0.0252, rel=1e-7)
    assert metrics["beta"] == pytest.approx(1.0, rel=1e-7)
    assert metrics["max_drawdown"] < 0
    direct = hac_alpha(strategy, benchmark)
    assert direct["alpha_ci_low"] > 0


def test_hac_alpha_uses_excess_returns_with_time_varying_cash() -> None:
    rng = np.random.default_rng(91)
    dates = pd.bdate_range("2018-01-02", periods=900)
    risk_free = pd.Series(
        0.00005 + 0.00002 * np.sin(np.arange(len(dates)) / 25), index=dates
    )
    market_excess = pd.Series(rng.normal(0.0003, 0.01, len(dates)), index=dates)
    benchmark = risk_free + market_excess
    strategy = risk_free + 0.0001 + 0.7 * market_excess

    result = hac_alpha(
        strategy,
        benchmark,
        risk_free_returns=risk_free,
    )
    boot = bootstrap_regression_alpha(
        strategy,
        benchmark,
        risk_free_returns=risk_free,
        n_boot=100,
        seed=3,
    )

    assert result["alpha"] == pytest.approx(0.0252, rel=1e-7)
    assert result["beta"] == pytest.approx(0.7, rel=1e-7)
    assert boot["alpha"] == pytest.approx(0.0252, rel=1e-7)


def test_drawdown_includes_initial_capital() -> None:
    assert max_drawdown([-0.10, 0.05]) == pytest.approx(-0.10)


def test_stationary_bootstrap_is_seed_deterministic() -> None:
    values = pd.Series(np.linspace(-0.01, 0.02, 100))
    first = stationary_block_bootstrap(values, n_boot=50, mean_block=10, seed=7)
    second = stationary_block_bootstrap(values, n_boot=50, mean_block=10, seed=7)
    np.testing.assert_array_equal(first, second)
    interval = bootstrap_alpha(values, n_boot=100, mean_block=10, seed=3)
    assert interval["alpha_ci_low"] < interval["alpha"] < interval["alpha_ci_high"]


def test_regression_alpha_bootstrap_targets_intercept_not_active_mean() -> None:
    rng = np.random.default_rng(4)
    benchmark = pd.Series(rng.normal(0.0003, 0.01, 500))
    strategy = 0.0001 + 0.5 * benchmark
    result = bootstrap_regression_alpha(strategy, benchmark, n_boot=100, seed=9)
    assert result["alpha"] == pytest.approx(0.0252, rel=1e-6)
    assert result["alpha_ci_low"] > 0


def test_paired_neural_increment_test_detects_a_consistent_increment() -> None:
    rng = np.random.default_rng(41)
    index = pd.bdate_range("2020-01-02", periods=600)
    benchmark = pd.Series(rng.normal(0.0002, 0.01, len(index)), index=index)
    baseline = 0.7 * benchmark + pd.Series(
        rng.normal(0.0, 0.003, len(index)), index=index
    )
    neural = baseline + 0.0005
    result = paired_neural_increment_test(
        neural,
        baseline,
        benchmark,
        n_boot=200,
        seed=11,
    )

    assert result["annual_alpha_increment"] == pytest.approx(0.126, abs=1e-10)
    assert result["sharpe_increment"] > 0.10
    assert result["alpha_increment_pvalue_one_sided"] <= 0.01


def test_benjamini_hochberg_controls_fdr() -> None:
    result = benjamini_hochberg({"a": 0.001, "b": 0.02, "c": 0.20}, q=0.10)
    assert bool(result.loc["a", "reject"])
    assert bool(result.loc["b", "reject"])
    assert not bool(result.loc["c", "reject"])
    assert result.loc["a", "adjusted_p_value"] == pytest.approx(0.003)


def _validated_evidence(**overrides: object) -> ValidationEvidence:
    values = {
        "years_history": 12.0,
        "net_alpha": 0.02,
        "alpha_ci_low": 0.003,
        "fdr_reject": True,
        "sharpe_improvement": 0.20,
        "max_drawdown_improvement": 0.20,
        "positive_fold_fraction": 0.70,
        "double_cost_alpha": 0.01,
        "delayed_alpha": 0.005,
        "robust_seed_parameter_fraction": 0.80,
        "best_year_alpha_share": 0.40,
    }
    values.update(overrides)
    return ValidationEvidence(**values)


def test_validation_classification_is_strict() -> None:
    assert classify_validation(_validated_evidence()).status is ValidationStatus.VALIDATED
    promising = classify_validation(_validated_evidence(alpha_ci_low=-0.001, fdr_reject=False))
    assert promising.status is ValidationStatus.PROMISING
    overlay = classify_validation(
        _validated_evidence(
            net_alpha=-0.01,
            alpha_ci_low=-0.02,
            fdr_reject=False,
            positive_fold_fraction=0.30,
        )
    )
    assert overlay.status is ValidationStatus.RISK_OVERLAY
    short = classify_validation(_validated_evidence(years_history=7.9))
    assert short.status is ValidationStatus.INSUFFICIENT_HISTORY
    assert neural_increment_is_material(
        annual_alpha_increment=0.005, sharpe_increment=0.10, increment_pvalue=0.10
    )
    ineligible = classify_validation(_validated_evidence(), formal_eligible=False)
    assert ineligible.status is ValidationStatus.PROMISING
    assert not ineligible.checks["formal_evidence_eligible"]


def test_walk_forward_has_annual_outer_folds_and_90_observation_purge() -> None:
    dates = pd.date_range("2000-01-03", "2012-12-31", freq="B")
    folds = purged_expanding_walk_forward(dates)
    assert folds[0].fold_id == "2011"
    first = folds[0]
    assert first.test.min().year == 2011
    assert first.validation.max() < pd.Timestamp("2011-01-01")
    assert first.validation.min().year == 2009
    assert (first.train.max() - first.train.min()).days / 365.25 >= 8
    assert sum((dates > first.validation.max()) & (dates < first.test.min())) == 90
    assert sum((dates > first.train.max()) & (dates < first.validation.min())) == 90
    assert all(fold.training_history_years >= 8.0 for fold in folds)


def test_prefix_invariance_detects_future_leakage() -> None:
    frame = pd.DataFrame({"x": np.arange(20.0)})

    def causal(data: pd.DataFrame) -> pd.Series:
        return data["x"].expanding().mean()

    assert_prefix_invariant(causal, frame, checkpoints=[5, 10, 19])

    def leaking(data: pd.DataFrame) -> pd.Series:
        return pd.Series(data["x"].mean(), index=data.index)

    with pytest.raises(AssertionError):
        assert_prefix_invariant(leaking, frame, checkpoints=[5])
