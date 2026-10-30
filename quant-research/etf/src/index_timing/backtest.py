"""Causal long-cash execution and transaction-cost backtesting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from .data import validate_ohlcv
from .types import BacktestResult


@dataclass(frozen=True, slots=True)
class BacktestConfig:
    """Execution assumptions shared by paper reconstruction and validation."""

    one_way_cost_bps: float
    execution_delay: int = 1
    cost_multiplier: float = 1.0
    periods_per_year: int = 252
    cash_rate_lag: int = 1
    cash_rate_unit: Literal["decimal", "percent", "daily"] = "decimal"
    cash_day_count_basis: float = 365.0

    def __post_init__(self) -> None:
        if self.one_way_cost_bps < 0 or self.cost_multiplier < 0:
            raise ValueError("transaction costs cannot be negative")
        if self.execution_delay < 1:
            raise ValueError("execution_delay must be at least one trading day")
        if self.cash_rate_lag < 0:
            raise ValueError("cash_rate_lag cannot be negative")
        if self.periods_per_year <= 0:
            raise ValueError("periods_per_year must be positive")
        if self.cash_day_count_basis <= 0:
            raise ValueError("cash_day_count_basis must be positive")


def _target_positions(
    target: pd.Series,
    index: pd.DatetimeIndex,
    *,
    execution_delay: int,
) -> tuple[pd.Series, pd.Series]:
    signal = target.copy()
    signal.index = pd.DatetimeIndex(pd.to_datetime(signal.index)).tz_localize(None).normalize()
    signal = signal[~signal.index.duplicated(keep="last")].sort_index()
    signal = pd.to_numeric(signal, errors="coerce").reindex(index).ffill().fillna(0.0)
    if ((signal < 0) | (signal > 1)).any():
        bad = signal[(signal < 0) | (signal > 1)].iloc[0]
        raise ValueError(f"target positions must remain in [0, 1]; found {bad}")
    signal.name = "signal_position"
    executed = signal.shift(execution_delay, fill_value=0.0)
    executed.name = "position"
    return signal.astype(float), executed.astype(float)


def _cash_return(
    cash_rate: pd.Series | float | int | None,
    index: pd.DatetimeIndex,
    *,
    periods_per_year: int,
    lag: int,
    unit: Literal["decimal", "percent", "daily"],
    day_count_basis: float,
) -> pd.Series:
    if cash_rate is None:
        annual_or_daily = pd.Series(0.0, index=index)
    elif np.isscalar(cash_rate):
        annual_or_daily = pd.Series(float(cash_rate), index=index)
    else:
        annual_or_daily = pd.to_numeric(cash_rate, errors="coerce").copy()
        annual_or_daily.index = (
            pd.DatetimeIndex(pd.to_datetime(annual_or_daily.index)).tz_localize(None).normalize()
        )
        annual_or_daily = annual_or_daily[
            ~annual_or_daily.index.duplicated(keep="last")
        ].sort_index()
        # Preserve releases that occur on a non-trading date (for example a
        # Friday observation marked available on Saturday) before sampling the
        # latest available value on each market session.
        asof_index = annual_or_daily.index.union(index).sort_values()
        annual_or_daily = annual_or_daily.reindex(asof_index).ffill().reindex(index)
        # A value observed on date t is usable only after the configured lag.
        annual_or_daily = annual_or_daily.shift(lag)
    annual_or_daily = annual_or_daily.fillna(0.0).astype(float)
    if unit == "percent":
        annual_or_daily = annual_or_daily / 100.0
    if unit == "daily":
        result = annual_or_daily
    else:
        if (annual_or_daily <= -1).any():
            raise ValueError("annual cash yields must be greater than -100%")
        # Each row represents the holding period from this session's open to
        # the next session's open.  That interval spans three calendar days on
        # a normal Friday-to-Monday transition, so a fixed 1/252 accrual would
        # systematically understate cash returns for low-exposure strategies.
        normalized_index = pd.DatetimeIndex(index).tz_localize(None).normalize()
        next_session = pd.Series(normalized_index, index=normalized_index).shift(-1)
        holding_days = (
            next_session
            - pd.Series(normalized_index, index=normalized_index)
        ).dt.total_seconds().div(86_400.0)
        # The final row has no known next open and is dropped by the backtest;
        # one day keeps the standalone conversion helper finite and explicit.
        holding_days = holding_days.fillna(1.0).clip(lower=1.0)
        result = np.expm1(
            np.log1p(annual_or_daily).to_numpy(dtype=float)
            * holding_days.to_numpy(dtype=float)
            / day_count_basis
        )
        result = pd.Series(result, index=index)
    result.name = "cash_return"
    return result


def cash_rate_to_daily_returns(
    cash_rate: pd.Series | float | int | None,
    index: pd.DatetimeIndex,
    *,
    periods_per_year: int = 252,
    lag: int = 1,
    unit: Literal["decimal", "percent", "daily"] = "decimal",
    day_count_basis: float = 365.0,
) -> pd.Series:
    """Convert an annual yield into causal open-to-open cash returns.

    Annual yields accrue over the actual calendar days to the next market
    session using an ACT/365 convention by default.
    """

    return _cash_return(
        cash_rate,
        index,
        periods_per_year=periods_per_year,
        lag=lag,
        unit=unit,
        day_count_basis=day_count_basis,
    )


def run_long_cash_backtest(
    prices: pd.DataFrame,
    target_position: pd.Series,
    *,
    one_way_cost_bps: float,
    cash_rate: pd.Series | float | int | None = None,
    execution_delay: int = 1,
    cost_multiplier: float = 1.0,
    periods_per_year: int = 252,
    cash_rate_lag: int = 1,
    cash_rate_unit: Literal["decimal", "percent", "daily"] = "decimal",
    cash_day_count_basis: float = 365.0,
) -> pd.DataFrame:
    """Backtest close-generated signals executed at a future session's open.

    A signal generated after the close of session ``t`` first becomes an open
    position at ``t+1``.  Return at row ``t+1`` is the adjusted open-to-open
    return from ``t+1`` to ``t+2``.  This convention gives the model no access
    to the overnight move immediately following the signal and makes trading
    costs land on the actual execution row.
    """

    config = BacktestConfig(
        one_way_cost_bps=one_way_cost_bps,
        execution_delay=execution_delay,
        cost_multiplier=cost_multiplier,
        periods_per_year=periods_per_year,
        cash_rate_lag=cash_rate_lag,
        cash_rate_unit=cash_rate_unit,
        cash_day_count_basis=cash_day_count_basis,
    )
    validate_ohlcv(prices)
    market = prices.sort_index().copy()
    if (market["open"] <= 0).any() or market["open"].isna().any():
        raise ValueError("next-open backtests require strictly positive, non-missing open prices")
    signal, position = _target_positions(
        target_position, market.index, execution_delay=config.execution_delay
    )
    asset_return = market["open"].shift(-1).div(market["open"]).sub(1.0)
    asset_return.name = "asset_return"
    cash_return = _cash_return(
        cash_rate,
        market.index,
        periods_per_year=config.periods_per_year,
        lag=config.cash_rate_lag,
        unit=config.cash_rate_unit,
        day_count_basis=config.cash_day_count_basis,
    )
    turnover = position.diff().abs()
    if not turnover.empty:
        turnover.iloc[0] = abs(position.iloc[0])
    turnover.name = "turnover"
    cost = turnover * config.one_way_cost_bps * config.cost_multiplier / 10_000.0
    cost.name = "cost"
    gross_return = position * asset_return + (1.0 - position) * cash_return
    gross_return.name = "gross_return"
    net_return = gross_return - cost
    net_return.name = "net_return"

    daily = pd.concat(
        [signal, position, turnover, asset_return, cash_return, gross_return, cost, net_return],
        axis=1,
    )
    # The final open-to-open return is unknowable until the next open.
    daily = daily.loc[daily["asset_return"].notna()].copy()
    daily["benchmark_return"] = daily["asset_return"]
    average_exposure = float(daily["position"].mean()) if len(daily) else 0.0
    daily["equal_exposure_return"] = (
        average_exposure * daily["asset_return"]
        + (1.0 - average_exposure) * daily["cash_return"]
    )
    daily["active_return"] = daily["net_return"] - daily["equal_exposure_return"]
    daily["equity"] = (1.0 + daily["net_return"]).cumprod()
    daily["benchmark_equity"] = (1.0 + daily["benchmark_return"]).cumprod()
    daily["equal_exposure_equity"] = (1.0 + daily["equal_exposure_return"]).cumprod()
    daily.index.name = "date"
    return daily


def run_cost_scenarios(
    prices: pd.DataFrame,
    target_position: pd.Series,
    *,
    one_way_cost_bps: float,
    cash_rate: pd.Series | float | int | None = None,
    execution_delay: int = 1,
    periods_per_year: int = 252,
    cash_rate_lag: int = 1,
    cash_rate_unit: Literal["decimal", "percent", "daily"] = "decimal",
    cash_day_count_basis: float = 365.0,
) -> dict[str, pd.DataFrame]:
    """Run preregistered zero/base/double-cost and extra-delay scenarios."""

    common = {
        "prices": prices,
        "target_position": target_position,
        "one_way_cost_bps": one_way_cost_bps,
        "cash_rate": cash_rate,
        "periods_per_year": periods_per_year,
        "cash_rate_lag": cash_rate_lag,
        "cash_rate_unit": cash_rate_unit,
        "cash_day_count_basis": cash_day_count_basis,
    }
    return {
        "zero_cost": run_long_cash_backtest(
            **common, execution_delay=execution_delay, cost_multiplier=0.0
        ),
        "base": run_long_cash_backtest(
            **common, execution_delay=execution_delay, cost_multiplier=1.0
        ),
        "double_cost": run_long_cash_backtest(
            **common, execution_delay=execution_delay, cost_multiplier=2.0
        ),
        "extra_delay": run_long_cash_backtest(
            **common, execution_delay=execution_delay + 1, cost_multiplier=1.0
        ),
    }


def slice_backtest_result(
    daily: pd.DataFrame,
    *,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Slice an evaluation window and rebuild all window-dependent benchmarks.

    The equal-exposure comparator must use the strategy's *evaluation-window*
    mean exposure.  Computing it on a longer price history would treat dates
    before the first saved signal as zero exposure and distort alpha/IR.
    Equity curves are rebased to one for the same reason.
    """

    selected = daily.loc[daily.index >= pd.Timestamp(start)].copy()
    if end is not None:
        selected = selected.loc[selected.index <= pd.Timestamp(end)].copy()
    if selected.empty:
        return selected
    required = {"position", "asset_return", "cash_return", "net_return"}
    missing = sorted(required - set(selected.columns))
    if missing:
        raise ValueError(f"backtest slice is missing columns: {missing}")
    average_exposure = float(selected["position"].mean())
    selected["equal_exposure_return"] = (
        average_exposure * selected["asset_return"]
        + (1.0 - average_exposure) * selected["cash_return"]
    )
    selected["active_return"] = selected["net_return"] - selected["equal_exposure_return"]
    selected["equity"] = (1.0 + selected["net_return"]).cumprod()
    if "benchmark_return" in selected:
        selected["benchmark_equity"] = (1.0 + selected["benchmark_return"]).cumprod()
    selected["equal_exposure_equity"] = (
        1.0 + selected["equal_exposure_return"]
    ).cumprod()
    return selected


def make_backtest_result(
    prices: pd.DataFrame,
    target_position: pd.Series,
    *,
    asset_id: str,
    experiment_id: str,
    one_way_cost_bps: float,
    cash_rate: pd.Series | float | int | None = None,
    **kwargs: object,
) -> BacktestResult:
    """Create the project's public backtest result with standard metrics."""

    from .statistics import performance_metrics

    daily = run_long_cash_backtest(
        prices,
        target_position,
        one_way_cost_bps=one_way_cost_bps,
        cash_rate=cash_rate,
        **kwargs,
    )
    metrics = performance_metrics(
        daily["net_return"],
        benchmark_returns=daily["equal_exposure_return"],
        risk_free_returns=daily["cash_return"],
        positions=daily["position"],
        costs=daily["cost"],
    )
    return BacktestResult(
        asset_id=asset_id,
        experiment_id=experiment_id,
        daily=daily,
        metrics=metrics,
        diagnostics={"execution": "close signal, next open trade, open-to-open return"},
    )


# Concise alias retained for notebook users.
backtest_long_cash = run_long_cash_backtest


__all__ = [
    "BacktestConfig",
    "backtest_long_cash",
    "cash_rate_to_daily_returns",
    "make_backtest_result",
    "run_cost_scenarios",
    "run_long_cash_backtest",
    "slice_backtest_result",
]
