"""Walk-forward design and preregistered evidence classification."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .statistics import hac_alpha, max_drawdown
from .types import ValidationStatus


@dataclass(frozen=True, slots=True)
class WalkForwardSplit:
    """One expanding-window fit/validation/outer-test partition."""

    fold_id: str
    train: pd.DatetimeIndex
    validation: pd.DatetimeIndex
    test: pd.DatetimeIndex
    purge_observations: int
    training_history_years: float


def purged_expanding_walk_forward(
    dates: pd.DatetimeIndex | Sequence[pd.Timestamp | str],
    *,
    minimum_history_years: int = 8,
    validation_years: int = 2,
    purge_observations: int = 90,
) -> list[WalkForwardSplit]:
    """Create annual outer folds with an expanding history and boundary purge.

    ``minimum_history_years`` applies to the fit sample alone.  The internal
    validation window is additional: with the default settings an outer test
    may start only after at least eight fit years plus two validation years.
    Ninety *observations*, not calendar days, are removed immediately before
    each validation and test boundary.
    """

    if minimum_history_years <= validation_years:
        raise ValueError("minimum history must exceed the validation window")
    if validation_years < 1 or purge_observations < 0:
        raise ValueError("validation years must be positive and purge non-negative")
    index = pd.DatetimeIndex(pd.to_datetime(list(dates))).tz_localize(None).normalize()
    index = pd.DatetimeIndex(index.unique()).sort_values()
    if index.empty:
        return []
    eligible_date = index.min() + pd.DateOffset(
        years=minimum_history_years + validation_years
    )
    folds: list[WalkForwardSplit] = []
    for test_year in sorted(set(index.year)):
        test_start = pd.Timestamp(year=test_year, month=1, day=1)
        if test_start < eligible_date:
            continue
        test = index[(index >= test_start) & (index < test_start + pd.DateOffset(years=1))]
        if test.empty:
            continue
        validation_start = test_start - pd.DateOffset(years=validation_years)
        train = index[index < validation_start]
        validation = index[(index >= validation_start) & (index < test_start)]
        if purge_observations:
            train = train[:-purge_observations] if len(train) > purge_observations else train[:0]
            validation = (
                validation[:-purge_observations]
                if len(validation) > purge_observations
                else validation[:0]
            )
        if train.empty or validation.empty:
            continue
        training_history_years = float(
            (train.max() - train.min()).days / 365.2425
        )
        # The purge is part of the fit boundary.  Recheck the minimum after
        # removing it so an exact 8+2 calendar-year sample cannot silently
        # become a sub-eight-year training fold.
        if training_history_years < minimum_history_years:
            continue
        folds.append(
            WalkForwardSplit(
                fold_id=str(test_year),
                train=train,
                validation=validation,
                test=test,
                purge_observations=purge_observations,
                training_history_years=training_history_years,
            )
        )
    return folds


@dataclass(frozen=True, slots=True)
class ValidationThresholds:
    """Locked thresholds from the project's validation protocol."""

    minimum_history_years: float = 8.0
    minimum_net_alpha: float = 0.01
    minimum_sharpe_improvement: float = 0.15
    minimum_drawdown_improvement: float = 0.15
    minimum_positive_fold_fraction: float = 0.60
    minimum_robust_fraction: float = 0.70
    maximum_best_year_share: float = 0.50
    neural_alpha_increment: float = 0.005
    neural_sharpe_increment: float = 0.10
    neural_pvalue: float = 0.10


@dataclass(frozen=True, slots=True)
class ValidationEvidence:
    """All evidence needed to classify one asset without hidden discretion."""

    years_history: float
    net_alpha: float
    alpha_ci_low: float
    fdr_reject: bool
    sharpe_improvement: float
    max_drawdown_improvement: float
    positive_fold_fraction: float
    double_cost_alpha: float
    delayed_alpha: float
    robust_seed_parameter_fraction: float
    best_year_alpha_share: float


@dataclass(frozen=True, slots=True)
class ValidationDecision:
    status: ValidationStatus
    checks: dict[str, bool]
    failed_checks: tuple[str, ...] = field(default_factory=tuple)
    summary: str = ""


def classify_validation(
    evidence: ValidationEvidence,
    *,
    thresholds: ValidationThresholds | None = None,
    formal_eligible: bool = True,
) -> ValidationDecision:
    """Apply the preregistered Validated/Promising/Risk-overlay decision tree."""

    limit = thresholds or ValidationThresholds()
    finite_fields = {
        "history": evidence.years_history,
        "net_alpha": evidence.net_alpha,
        "alpha_ci": evidence.alpha_ci_low,
        "sharpe": evidence.sharpe_improvement,
        "drawdown": evidence.max_drawdown_improvement,
        "folds": evidence.positive_fold_fraction,
        "double_cost": evidence.double_cost_alpha,
        "delay": evidence.delayed_alpha,
        "robustness": evidence.robust_seed_parameter_fraction,
        "concentration": evidence.best_year_alpha_share,
    }
    if any(not np.isfinite(value) for value in finite_fields.values()):
        checks = {name: bool(np.isfinite(value)) for name, value in finite_fields.items()}
        return ValidationDecision(
            status=ValidationStatus.FAILED,
            checks=checks,
            failed_checks=tuple(name for name, passed in checks.items() if not passed),
            summary="证据字段不完整，不能作有效性判断。",
        )

    checks = {
        "formal_evidence_eligible": bool(formal_eligible),
        "history_at_least_8y": evidence.years_history >= limit.minimum_history_years,
        "net_alpha_at_least_1pct": evidence.net_alpha >= limit.minimum_net_alpha,
        "bootstrap_alpha_ci_positive": evidence.alpha_ci_low > 0,
        "passes_bh_fdr": bool(evidence.fdr_reject),
        "sharpe_improvement_at_least_0_15": (
            evidence.sharpe_improvement >= limit.minimum_sharpe_improvement
        ),
        "drawdown_improvement_at_least_15pct": (
            evidence.max_drawdown_improvement >= limit.minimum_drawdown_improvement
        ),
        "positive_outer_folds_at_least_60pct": (
            evidence.positive_fold_fraction >= limit.minimum_positive_fold_fraction
        ),
        "double_cost_alpha_positive": evidence.double_cost_alpha > 0,
        "extra_delay_alpha_positive": evidence.delayed_alpha > 0,
        "seed_parameter_robustness_at_least_70pct": (
            evidence.robust_seed_parameter_fraction >= limit.minimum_robust_fraction
        ),
        "best_year_contribution_at_most_50pct": (
            evidence.best_year_alpha_share <= limit.maximum_best_year_share
        ),
    }
    failed = tuple(name for name, passed in checks.items() if not passed)
    if all(checks.values()):
        return ValidationDecision(
            status=ValidationStatus.VALIDATED,
            checks=checks,
            summary="全部预注册经济性、统计显著性和稳健性门槛均通过。",
        )
    if not checks["history_at_least_8y"]:
        return ValidationDecision(
            status=ValidationStatus.INSUFFICIENT_HISTORY,
            checks=checks,
            failed_checks=failed,
            summary="历史不足8年，仅可列为方法重构，不能授予Validated。",
        )

    promising = (
        evidence.net_alpha > 0
        and evidence.sharpe_improvement > 0
        and evidence.positive_fold_fraction >= 0.50
        and evidence.robust_seed_parameter_fraction >= 0.50
    )
    risk_overlay = (
        evidence.max_drawdown_improvement >= limit.minimum_drawdown_improvement
        and evidence.sharpe_improvement >= 0
    )
    if not formal_eligible and all(
        passed for name, passed in checks.items() if name != "formal_evidence_eligible"
    ):
        status = ValidationStatus.PROMISING
        summary = "数值门槛通过，但运行不具备正式证据资格，最多列为Promising。"
    elif promising:
        status = ValidationStatus.PROMISING
        summary = "方向性证据为正，但至少一项正式验证门槛未通过。"
    elif risk_overlay:
        status = ValidationStatus.RISK_OVERLAY
        summary = "收益增强证据不足，但回撤控制达到风险覆盖层门槛。"
    else:
        status = ValidationStatus.FAILED
        summary = "未达到收益增强或风险覆盖层的最低证据标准。"
    return ValidationDecision(status=status, checks=checks, failed_checks=failed, summary=summary)


def neural_increment_is_material(
    *,
    annual_alpha_increment: float,
    sharpe_increment: float,
    increment_pvalue: float,
    thresholds: ValidationThresholds | None = None,
) -> bool:
    """Test whether a neural model adds value beyond its linear/PCA baseline."""

    limit = thresholds or ValidationThresholds()
    return bool(
        annual_alpha_increment >= limit.neural_alpha_increment
        and sharpe_increment >= limit.neural_sharpe_increment
        and increment_pvalue <= limit.neural_pvalue
    )


def yearly_active_diagnostics(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    risk_free_returns: pd.Series | float | None = None,
) -> dict[str, float]:
    """Compute positive active years and concentration of annual HAC alpha."""

    if risk_free_returns is None:
        risk_free = pd.Series(0.0, index=strategy_returns.index)
    elif isinstance(risk_free_returns, pd.Series):
        risk_free = risk_free_returns
    else:
        risk_free = pd.Series(float(risk_free_returns), index=strategy_returns.index)

    aligned = pd.concat(
        [
            pd.to_numeric(strategy_returns, errors="coerce").rename("strategy"),
            pd.to_numeric(benchmark_returns, errors="coerce").rename("benchmark"),
            pd.to_numeric(risk_free, errors="coerce").rename("risk_free"),
        ],
        axis=1,
    ).dropna()
    if aligned.empty:
        return {"positive_fold_fraction": float("nan"), "best_year_alpha_share": float("nan")}
    yearly_active = aligned.groupby(aligned.index.year).apply(
        lambda group: (1.0 + group["strategy"]).prod() - (1.0 + group["benchmark"]).prod(),
        include_groups=False,
    )
    yearly_alpha = aligned.groupby(aligned.index.year).apply(
        lambda group: hac_alpha(
            group["strategy"],
            group["benchmark"],
            risk_free_returns=group["risk_free"],
        )["alpha"],
        include_groups=False,
    )
    positive_fraction = float((yearly_active > 0).mean())
    cumulative_alpha = float(yearly_alpha.sum())
    best_share = (
        float(yearly_alpha.max() / cumulative_alpha) if cumulative_alpha > 0 else 1.0
    )
    return {
        "positive_fold_fraction": positive_fraction,
        "best_year_alpha_share": best_share,
        "positive_active_year_fraction": positive_fraction,
        "best_year_hac_alpha_share": best_share,
    }


def drawdown_improvement(
    strategy_returns: pd.Series | Sequence[float],
    benchmark_returns: pd.Series | Sequence[float],
) -> float:
    """Fractional reduction in absolute maximum drawdown versus benchmark."""

    strategy_drawdown = max_drawdown(strategy_returns)
    benchmark_drawdown = max_drawdown(benchmark_returns)
    if not np.isfinite(benchmark_drawdown) or benchmark_drawdown == 0:
        return float("nan")
    return float((abs(benchmark_drawdown) - abs(strategy_drawdown)) / abs(benchmark_drawdown))


def assert_prefix_invariant(
    builder: Callable[[pd.DataFrame], pd.Series],
    data: pd.DataFrame,
    *,
    checkpoints: Sequence[int] | None = None,
    rtol: float = 1e-10,
    atol: float = 1e-12,
) -> None:
    """Verify that adding future rows never changes already-issued signals."""

    if checkpoints is None:
        checkpoints = [max(1, len(data) // 2), len(data) - 1]
    full = builder(data)
    for stop in checkpoints:
        if stop <= 0 or stop > len(data):
            raise ValueError(f"invalid prefix checkpoint: {stop}")
        prefix = builder(data.iloc[:stop])
        expected = full.reindex(prefix.index)
        np.testing.assert_allclose(
            prefix.to_numpy(),
            expected.to_numpy(),
            rtol=rtol,
            atol=atol,
            equal_nan=True,
        )


__all__ = [
    "ValidationDecision",
    "ValidationEvidence",
    "ValidationThresholds",
    "WalkForwardSplit",
    "assert_prefix_invariant",
    "classify_validation",
    "drawdown_improvement",
    "neural_increment_is_material",
    "purged_expanding_walk_forward",
    "yearly_active_diagnostics",
]
