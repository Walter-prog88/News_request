"""Performance measurement and multiple-testing-aware inference."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm


def _series(values: pd.Series | Sequence[float], name: str) -> pd.Series:
    result = values.copy() if isinstance(values, pd.Series) else pd.Series(values)
    result = pd.to_numeric(result, errors="coerce").replace([np.inf, -np.inf], np.nan)
    result.name = name
    return result.astype(float)


def max_drawdown(returns: pd.Series | Sequence[float]) -> float:
    clean = _series(returns, "return").dropna()
    if clean.empty:
        return float("nan")
    if (clean < -1.0).any():
        raise ValueError("period returns cannot be below -100%")
    # Include initial capital so a loss on the first observation is measured
    # as a drawdown rather than incorrectly becoming the high-water mark.
    equity = np.concatenate([[1.0], (1.0 + clean).cumprod().to_numpy()])
    high_water = np.maximum.accumulate(equity)
    return float(np.min(equity / high_water - 1.0))


def hac_alpha(
    strategy_returns: pd.Series | Sequence[float],
    benchmark_returns: pd.Series | Sequence[float],
    *,
    risk_free_returns: pd.Series | Sequence[float] | float | None = None,
    periods_per_year: int = 252,
    maxlags: int | None = None,
) -> dict[str, float]:
    """Estimate annualized excess-return alpha with Newey-West errors.

    The fitted equation is ``strategy - rf = alpha + beta * (benchmark - rf)``.
    Passing no risk-free series is equivalent to an explicit zero rate.
    """

    strategy = _series(strategy_returns, "strategy")
    benchmark = _series(benchmark_returns, "benchmark")
    aligned = pd.concat([strategy, benchmark], axis=1)
    if risk_free_returns is None:
        aligned["risk_free"] = 0.0
    elif np.isscalar(risk_free_returns):
        aligned["risk_free"] = float(risk_free_returns)
    else:
        aligned["risk_free"] = _series(risk_free_returns, "risk_free")
    aligned = aligned.dropna()
    if len(aligned) < 3:
        return {
            "alpha": float("nan"),
            "alpha_se": float("nan"),
            "alpha_pvalue": float("nan"),
            "alpha_ci_low": float("nan"),
            "alpha_ci_high": float("nan"),
            "beta": float("nan"),
        }
    if maxlags is None:
        maxlags = max(1, int(np.floor(4 * (len(aligned) / 100) ** (2 / 9))))
    try:
        import statsmodels.api as sm

        benchmark_excess = aligned["benchmark"] - aligned["risk_free"]
        strategy_excess = aligned["strategy"] - aligned["risk_free"]
        design = sm.add_constant(benchmark_excess.to_numpy())
        fitted = sm.OLS(strategy_excess.to_numpy(), design).fit(
            cov_type="HAC", cov_kwds={"maxlags": maxlags}
        )
        intercept = float(fitted.params[0])
        standard_error = float(fitted.bse[0])
        pvalue = float(fitted.pvalues[0])
        beta = float(fitted.params[1])
    except (ImportError, ValueError, np.linalg.LinAlgError):
        # Minimal deterministic fallback.  It preserves point estimates but
        # does not pretend that ordinary standard errors are HAC robust.
        x = (aligned["benchmark"] - aligned["risk_free"]).to_numpy()
        y = (aligned["strategy"] - aligned["risk_free"]).to_numpy()
        design = np.column_stack([np.ones(len(x)), x])
        params, *_ = np.linalg.lstsq(design, y, rcond=None)
        intercept, beta = map(float, params)
        standard_error = float("nan")
        pvalue = float("nan")
    annual_alpha = intercept * periods_per_year
    annual_se = standard_error * periods_per_year
    return {
        "alpha": annual_alpha,
        "alpha_se": annual_se,
        "alpha_pvalue": pvalue,
        "alpha_ci_low": annual_alpha - 1.959963984540054 * annual_se,
        "alpha_ci_high": annual_alpha + 1.959963984540054 * annual_se,
        "beta": beta,
    }


def performance_metrics(
    returns: pd.Series | Sequence[float],
    *,
    benchmark_returns: pd.Series | Sequence[float] | None = None,
    risk_free_returns: pd.Series | Sequence[float] | float | None = None,
    positions: pd.Series | Sequence[float] | None = None,
    costs: pd.Series | Sequence[float] | None = None,
    periods_per_year: int = 252,
) -> dict[str, float | int | None]:
    """Compute the standard economic and risk statistics for a daily run."""

    strategy = _series(returns, "strategy")
    frame = pd.DataFrame({"strategy": strategy})
    if benchmark_returns is not None:
        frame["benchmark"] = _series(benchmark_returns, "benchmark")
    if risk_free_returns is None:
        frame["risk_free"] = 0.0
    elif np.isscalar(risk_free_returns):
        frame["risk_free"] = float(risk_free_returns)
    else:
        frame["risk_free"] = _series(risk_free_returns, "risk_free")
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=["strategy"])
    n_obs = len(frame)
    if n_obs == 0:
        return {"n_obs": 0}
    clean = frame["strategy"]
    if (clean < -1.0).any():
        raise ValueError("period returns cannot be below -100%")
    excess = clean - frame["risk_free"].fillna(0.0)
    wealth = float((1.0 + clean).prod())
    cagr = wealth ** (periods_per_year / n_obs) - 1.0 if wealth > 0 else float("nan")
    volatility = float(clean.std(ddof=1) * np.sqrt(periods_per_year))
    excess_std = float(excess.std(ddof=1))
    sharpe = (
        float(excess.mean() / excess_std * np.sqrt(periods_per_year))
        if excess_std > 0
        else float("nan")
    )
    downside = np.minimum(excess.to_numpy(), 0.0)
    downside_deviation = float(np.sqrt(np.mean(np.square(downside))))
    sortino = (
        float(excess.mean() / downside_deviation * np.sqrt(periods_per_year))
        if downside_deviation > 0
        else float("nan")
    )
    drawdown = max_drawdown(clean)
    calmar = cagr / abs(drawdown) if drawdown < 0 and np.isfinite(cagr) else float("nan")
    result: dict[str, float | int | None] = {
        "n_obs": n_obs,
        "years": n_obs / periods_per_year,
        "total_return": wealth - 1.0,
        "cagr": cagr,
        "annualized_volatility": volatility,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": drawdown,
        "calmar": calmar,
        "positive_day_fraction": float((clean > 0).mean()),
        "skew": float(clean.skew()),
        "kurtosis": float(clean.kurtosis() + 3.0),
    }
    if benchmark_returns is not None:
        comparison = frame[["strategy", "benchmark"]].dropna()
        active = comparison["strategy"] - comparison["benchmark"]
        tracking_error = float(active.std(ddof=1) * np.sqrt(periods_per_year))
        result["information_ratio"] = (
            float(active.mean() * periods_per_year / tracking_error)
            if tracking_error > 0
            else float("nan")
        )
        result["tracking_error"] = tracking_error
        result["annualized_active_return"] = float(active.mean() * periods_per_year)
        alpha_result = hac_alpha(
            comparison["strategy"],
            comparison["benchmark"],
            risk_free_returns=frame["risk_free"].reindex(comparison.index),
            periods_per_year=periods_per_year,
        )
        result.update(alpha_result)
        result["regression_alpha_vs_benchmark"] = alpha_result["alpha"]
        result["active_return_vs_benchmark"] = result["annualized_active_return"]
        up = comparison["benchmark"] > 0
        down = comparison["benchmark"] < 0
        result["upside_capture"] = (
            float(comparison.loc[up, "strategy"].sum() / comparison.loc[up, "benchmark"].sum())
            if up.any() and comparison.loc[up, "benchmark"].sum() != 0
            else float("nan")
        )
        result["downside_capture"] = (
            float(comparison.loc[down, "strategy"].sum() / comparison.loc[down, "benchmark"].sum())
            if down.any() and comparison.loc[down, "benchmark"].sum() != 0
            else float("nan")
        )
    if positions is not None:
        exposure = _series(positions, "position").reindex(frame.index)
        result["average_exposure"] = float(exposure.mean())
        turnover = exposure.diff().abs()
        if not turnover.empty:
            turnover.iloc[0] = abs(exposure.iloc[0])
        result["annualized_turnover"] = float(
            turnover.sum() / n_obs * periods_per_year
        )
    if costs is not None:
        cost_series = _series(costs, "cost").reindex(frame.index).fillna(0.0)
        result["total_cost"] = float(cost_series.sum())
        result["annualized_cost"] = float(cost_series.mean() * periods_per_year)
    return result


def stationary_block_bootstrap(
    values: pd.Series | Sequence[float],
    *,
    statistic: Callable[[np.ndarray], float] = np.mean,
    n_boot: int = 10_000,
    mean_block: int = 20,
    seed: int = 0,
) -> np.ndarray:
    """Politis-Romano stationary bootstrap with deterministic random state."""

    sample = _series(values, "value").dropna().to_numpy()
    if len(sample) == 0:
        raise ValueError("bootstrap input cannot be empty")
    if n_boot <= 0 or mean_block <= 0:
        raise ValueError("n_boot and mean_block must be positive")
    rng = np.random.default_rng(seed)
    probability_new_block = min(1.0, 1.0 / mean_block)
    output = np.empty(n_boot, dtype=float)
    n = len(sample)
    for replication in range(n_boot):
        indices = np.empty(n, dtype=np.int64)
        indices[0] = rng.integers(n)
        restart = rng.random(n - 1) < probability_new_block
        starts = rng.integers(n, size=n - 1)
        for offset in range(1, n):
            indices[offset] = (
                starts[offset - 1]
                if restart[offset - 1]
                else (indices[offset - 1] + 1) % n
            )
        output[replication] = statistic(sample[indices])
    return output


def bootstrap_alpha(
    active_returns: pd.Series | Sequence[float],
    *,
    n_boot: int = 10_000,
    mean_block: int = 20,
    seed: int = 0,
    periods_per_year: int = 252,
    confidence: float = 0.95,
) -> dict[str, float]:
    """Block-bootstrap annual active return and its two-sided interval."""

    active = _series(active_returns, "active").dropna()
    distribution = stationary_block_bootstrap(
        active, n_boot=n_boot, mean_block=mean_block, seed=seed
    ) * periods_per_year
    tail = (1.0 - confidence) / 2.0
    return {
        "alpha": float(active.mean() * periods_per_year),
        "alpha_ci_low": float(np.quantile(distribution, tail)),
        "alpha_ci_high": float(np.quantile(distribution, 1.0 - tail)),
        "probability_positive": float(np.mean(distribution > 0)),
        "bootstrap_pvalue": float(2 * min(np.mean(distribution <= 0), np.mean(distribution >= 0))),
    }


def bootstrap_regression_alpha(
    strategy_returns: pd.Series | Sequence[float],
    benchmark_returns: pd.Series | Sequence[float],
    *,
    risk_free_returns: pd.Series | Sequence[float] | float | None = None,
    n_boot: int = 10_000,
    mean_block: int = 20,
    seed: int = 0,
    periods_per_year: int = 252,
    confidence: float = 0.95,
) -> dict[str, float]:
    """Stationary paired bootstrap of the strategy-on-benchmark intercept.

    Resampling the two return series with identical block indices preserves
    their contemporaneous relationship.  The reported interval therefore
    targets regression alpha, rather than the simpler mean active return.
    """

    aligned = pd.concat(
        [
            _series(strategy_returns, "strategy"),
            _series(benchmark_returns, "benchmark"),
        ],
        axis=1,
    )
    if risk_free_returns is None:
        aligned["risk_free"] = 0.0
    elif np.isscalar(risk_free_returns):
        aligned["risk_free"] = float(risk_free_returns)
    else:
        aligned["risk_free"] = _series(risk_free_returns, "risk_free")
    aligned = aligned.dropna()
    if len(aligned) < 3:
        raise ValueError("at least three paired returns are required")
    if n_boot <= 0 or mean_block <= 0:
        raise ValueError("n_boot and mean_block must be positive")
    x = (aligned["benchmark"] - aligned["risk_free"]).to_numpy()
    y = (aligned["strategy"] - aligned["risk_free"]).to_numpy()
    n = len(aligned)

    def intercept(sample_x: np.ndarray, sample_y: np.ndarray) -> float:
        variance = float(np.var(sample_x))
        beta = float(np.cov(sample_x, sample_y, ddof=0)[0, 1] / variance) if variance > 0 else 0.0
        return float((sample_y.mean() - beta * sample_x.mean()) * periods_per_year)

    point = intercept(x, y)
    rng = np.random.default_rng(seed)
    restart_probability = min(1.0, 1.0 / mean_block)
    distribution = np.empty(n_boot, dtype=float)
    for replication in range(n_boot):
        indices = np.empty(n, dtype=np.int64)
        indices[0] = rng.integers(n)
        restart = rng.random(n - 1) < restart_probability
        starts = rng.integers(n, size=n - 1)
        for offset in range(1, n):
            indices[offset] = (
                starts[offset - 1]
                if restart[offset - 1]
                else (indices[offset - 1] + 1) % n
            )
        distribution[replication] = intercept(x[indices], y[indices])
    tail = (1.0 - confidence) / 2.0
    return {
        "alpha": point,
        "alpha_ci_low": float(np.quantile(distribution, tail)),
        "alpha_ci_high": float(np.quantile(distribution, 1.0 - tail)),
        "probability_positive": float(np.mean(distribution > 0)),
        "bootstrap_pvalue": float(
            min(1.0, 2 * min(np.mean(distribution <= 0), np.mean(distribution >= 0)))
        ),
    }


def paired_neural_increment_test(
    neural_returns: pd.Series | Sequence[float],
    baseline_returns: pd.Series | Sequence[float],
    benchmark_returns: pd.Series | Sequence[float],
    *,
    baseline_benchmark_returns: pd.Series | Sequence[float] | None = None,
    risk_free_returns: pd.Series | Sequence[float] | float | None = None,
    n_boot: int = 10_000,
    mean_block: int = 20,
    seed: int = 0,
    periods_per_year: int = 252,
) -> dict[str, float]:
    """Paired block-bootstrap neural incremental alpha and Sharpe.

    All three return paths are resampled with identical stationary blocks.  A
    one-sided p-value tests whether the neural model's increment is positive;
    economic materiality remains a separate pre-registered threshold.
    """

    neural_benchmark = _series(benchmark_returns, "neural_benchmark")
    baseline_benchmark = (
        neural_benchmark.rename("baseline_benchmark")
        if baseline_benchmark_returns is None
        else _series(baseline_benchmark_returns, "baseline_benchmark")
    )
    frame = pd.concat(
        [
            _series(neural_returns, "neural"),
            _series(baseline_returns, "baseline"),
            neural_benchmark,
            baseline_benchmark,
        ],
        axis=1,
    )
    if risk_free_returns is None:
        frame["risk_free"] = 0.0
    elif np.isscalar(risk_free_returns):
        frame["risk_free"] = float(risk_free_returns)
    else:
        frame["risk_free"] = _series(risk_free_returns, "risk_free")
    frame = frame.dropna()
    if len(frame) < 3:
        raise ValueError("at least three aligned return rows are required")
    if n_boot <= 0 or mean_block <= 0:
        raise ValueError("n_boot and mean_block must be positive")
    values = frame.to_numpy(dtype=float)
    n = len(values)

    def alpha(y: np.ndarray, x: np.ndarray) -> float:
        variance = float(np.var(x))
        beta = float(np.cov(x, y, ddof=0)[0, 1] / variance) if variance > 0 else 0.0
        return float((y.mean() - beta * x.mean()) * periods_per_year)

    def sharpe(y: np.ndarray, rf: np.ndarray) -> float:
        excess = y - rf
        standard_deviation = float(np.std(excess, ddof=1))
        return (
            float(np.mean(excess) / standard_deviation * np.sqrt(periods_per_year))
            if standard_deviation > 0
            else float("nan")
        )

    def increments(sample: np.ndarray) -> tuple[float, float]:
        neural = sample[:, 0]
        baseline = sample[:, 1]
        neural_benchmark_values = sample[:, 2]
        baseline_benchmark_values = sample[:, 3]
        risk_free = sample[:, 4]
        return (
            alpha(neural - risk_free, neural_benchmark_values - risk_free)
            - alpha(baseline - risk_free, baseline_benchmark_values - risk_free),
            sharpe(neural, risk_free) - sharpe(baseline, risk_free),
        )

    point_alpha, point_sharpe = increments(values)
    rng = np.random.default_rng(seed)
    restart_probability = min(1.0, 1.0 / mean_block)
    alpha_distribution = np.empty(n_boot, dtype=float)
    sharpe_distribution = np.empty(n_boot, dtype=float)
    for replication in range(n_boot):
        indices = np.empty(n, dtype=np.int64)
        indices[0] = rng.integers(n)
        restart = rng.random(n - 1) < restart_probability
        starts = rng.integers(n, size=n - 1)
        for offset in range(1, n):
            indices[offset] = (
                starts[offset - 1]
                if restart[offset - 1]
                else (indices[offset - 1] + 1) % n
            )
        alpha_distribution[replication], sharpe_distribution[replication] = increments(
            values[indices]
        )
    return {
        "annual_alpha_increment": point_alpha,
        "sharpe_increment": point_sharpe,
        "alpha_increment_pvalue_one_sided": float(
            np.mean(alpha_distribution <= 0.0)
        ),
        "sharpe_increment_pvalue_one_sided": float(
            np.mean(sharpe_distribution <= 0.0)
        ),
    }


def benjamini_hochberg(
    pvalues: Mapping[str, float] | pd.Series | Sequence[float],
    *,
    q: float = 0.10,
) -> pd.DataFrame:
    """Benjamini-Hochberg adjusted p-values and FDR rejection flags."""

    if not 0 < q < 1:
        raise ValueError("q must lie strictly between zero and one")
    if isinstance(pvalues, Mapping):
        values = pd.Series(pvalues, dtype=float)
    elif isinstance(pvalues, pd.Series):
        values = pvalues.astype(float)
    else:
        values = pd.Series(list(pvalues), dtype=float)
    valid = values.dropna()
    if ((valid < 0) | (valid > 1)).any():
        raise ValueError("p-values must lie in [0, 1]")
    result = pd.DataFrame(index=values.index, data={"p_value": values})
    result["adjusted_p_value"] = np.nan
    result["reject"] = False
    if valid.empty:
        return result
    ordered = valid.sort_values()
    ranks = np.arange(1, len(ordered) + 1)
    adjusted = (ordered.to_numpy() * len(ordered) / ranks)[::-1]
    adjusted = np.minimum.accumulate(adjusted)[::-1]
    adjusted = np.minimum(adjusted, 1.0)
    result.loc[ordered.index, "adjusted_p_value"] = adjusted
    result.loc[ordered.index, "reject"] = adjusted <= q
    return result


def probabilistic_sharpe_ratio(
    observed_sharpe: float,
    benchmark_sharpe: float,
    *,
    n_obs: int,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """Probability that a per-period Sharpe exceeds a benchmark Sharpe."""

    if n_obs < 2:
        return float("nan")
    denominator = 1.0 - skew * observed_sharpe + (kurtosis - 1.0) * observed_sharpe**2 / 4.0
    if denominator <= 0:
        return float("nan")
    z_score = (observed_sharpe - benchmark_sharpe) * np.sqrt(n_obs - 1) / np.sqrt(denominator)
    return float(norm.cdf(z_score))


def deflated_sharpe_ratio(
    returns: pd.Series | Sequence[float],
    *,
    n_trials: int,
    trial_sharpes: Sequence[float] | None = None,
    periods_per_year: int = 252,
) -> dict[str, float]:
    """Deflate the observed Sharpe for non-normality and strategy selection.

    Trial Sharpes, when supplied, must use the same annualized convention as
    the returned observed Sharpe.  Supplying them is strongly preferred;
    otherwise the cross-trial standard deviation is conservatively estimated
    from the observed return history.
    """

    clean = _series(returns, "return").dropna()
    if len(clean) < 2 or n_trials < 1:
        return {
            "observed_sharpe": float("nan"),
            "expected_max_sharpe": float("nan"),
            "dsr": float("nan"),
        }
    daily_std = float(clean.std(ddof=1))
    observed_daily = float(clean.mean() / daily_std) if daily_std > 0 else float("nan")
    observed_annual = observed_daily * np.sqrt(periods_per_year)
    if n_trials == 1:
        benchmark_annual = 0.0
    else:
        if trial_sharpes is not None and len(trial_sharpes) >= 2:
            candidates = np.asarray(trial_sharpes, dtype=float)
            mean_trial = float(np.nanmean(candidates))
            std_trial = float(np.nanstd(candidates, ddof=1))
        else:
            mean_trial = 0.0
            std_trial = np.sqrt(periods_per_year / max(len(clean) - 1, 1))
        euler_gamma = 0.5772156649015329
        expected_order = (
            (1.0 - euler_gamma) * norm.ppf(1.0 - 1.0 / n_trials)
            + euler_gamma * norm.ppf(1.0 - 1.0 / (n_trials * np.e))
        )
        benchmark_annual = mean_trial + std_trial * expected_order
    dsr = probabilistic_sharpe_ratio(
        observed_daily,
        benchmark_annual / np.sqrt(periods_per_year),
        n_obs=len(clean),
        skew=float(clean.skew()),
        kurtosis=float(clean.kurtosis() + 3.0),
    )
    return {
        "observed_sharpe": observed_annual,
        "expected_max_sharpe": benchmark_annual,
        "dsr": dsr,
    }


def spa_test(
    benchmark_returns: pd.Series | Sequence[float],
    model_returns: pd.DataFrame | np.ndarray,
    *,
    reps: int = 1_000,
    block_size: int = 20,
    seed: int = 0,
) -> dict[str, Any]:
    """Hansen SPA / Reality Check wrapper, available when ``arch`` is installed."""

    benchmark = _series(benchmark_returns, "benchmark").to_numpy()
    models = np.asarray(model_returns, dtype=float)
    if models.ndim == 1:
        models = models[:, None]
    if len(benchmark) != models.shape[0]:
        raise ValueError("benchmark and model return histories must have equal length")
    try:
        from arch.bootstrap import SPA
    except ImportError:
        return {"available": False, "reason": "arch is not installed", "pvalues": None}
    try:
        # SPA is defined on losses, so negate returns.
        comparison = SPA(
            -benchmark,
            -models,
            block_size=block_size,
            reps=reps,
            bootstrap="stationary",
            seed=seed,
        )
        comparison.compute()
        pvalues = comparison.pvalues
        return {
            "available": True,
            "pvalues": pvalues.to_dict() if hasattr(pvalues, "to_dict") else dict(pvalues),
        }
    except Exception as exc:  # pragma: no cover - version-specific arch failures
        return {"available": False, "reason": f"SPA failed: {exc}", "pvalues": None}


reality_check = spa_test


__all__ = [
    "benjamini_hochberg",
    "bootstrap_alpha",
    "bootstrap_regression_alpha",
    "deflated_sharpe_ratio",
    "hac_alpha",
    "max_drawdown",
    "paired_neural_increment_test",
    "performance_metrics",
    "probabilistic_sharpe_ratio",
    "reality_check",
    "spa_test",
    "stationary_block_bootstrap",
]
