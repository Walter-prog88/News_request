from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from index_timing.backtest import (
    run_cost_scenarios,
    run_long_cash_backtest,
    slice_backtest_result,
)
from index_timing.data import normalize_ohlcv


def _prices() -> pd.DataFrame:
    dates = pd.date_range("2024-01-02", periods=6, freq="B")
    opens = np.array([100.0, 110.0, 121.0, 133.1, 146.41, 161.051])
    return normalize_ohlcv(
        pd.DataFrame(
            {
                "date": dates,
                "open": opens,
                "high": opens * 1.01,
                "low": opens * 0.99,
                "close": opens,
                "adj_close": opens,
                "volume": 1_000.0,
            }
        )
    )


def test_close_signal_trades_at_next_open_without_capturing_prior_move() -> None:
    prices = _prices()
    signal = pd.Series(0.0, index=prices.index)
    signal.iloc[0] = 1.0
    result = run_long_cash_backtest(prices, signal, one_way_cost_bps=0)
    assert result.iloc[0]["position"] == 0.0
    assert result.iloc[1]["position"] == 1.0
    assert result.iloc[1]["asset_return"] == pytest.approx(0.10)
    assert result.iloc[1]["net_return"] == pytest.approx(0.10)


def test_turnover_cost_is_charged_on_execution_row() -> None:
    prices = _prices()
    signal = pd.Series(1.0, index=prices.index)
    result = run_long_cash_backtest(prices, signal, one_way_cost_bps=10)
    assert result.iloc[0]["cost"] == 0
    assert result.iloc[1]["turnover"] == 1.0
    assert result.iloc[1]["cost"] == pytest.approx(0.001)
    assert result.iloc[1]["net_return"] == pytest.approx(0.099)


def test_cash_yield_compounds_and_series_is_lagged() -> None:
    prices = _prices()
    signal = pd.Series(0.0, index=prices.index)
    annual = pd.Series([10.0] * len(prices), index=prices.index)
    result = run_long_cash_backtest(
        prices,
        signal,
        one_way_cost_bps=0,
        cash_rate=annual,
        cash_rate_unit="percent",
        cash_rate_lag=1,
    )
    assert result.iloc[0]["cash_return"] == 0.0
    assert result.iloc[1]["cash_return"] == pytest.approx((1.10 ** (1 / 365)) - 1)


def test_cash_accrues_actual_calendar_days_across_weekend() -> None:
    dates = pd.DatetimeIndex(["2024-01-05", "2024-01-08", "2024-01-09"])
    opens = np.array([100.0, 100.0, 100.0])
    prices = normalize_ohlcv(
        pd.DataFrame(
            {
                "date": dates,
                "open": opens,
                "high": opens,
                "low": opens,
                "close": opens,
                "volume": 1_000.0,
            }
        )
    )
    annual = pd.Series(10.0, index=dates)
    result = run_long_cash_backtest(
        prices,
        pd.Series(0.0, index=dates),
        one_way_cost_bps=0,
        cash_rate=annual,
        cash_rate_unit="percent",
        cash_rate_lag=0,
    )

    assert result.loc[pd.Timestamp("2024-01-05"), "cash_return"] == pytest.approx(
        1.10 ** (3 / 365) - 1
    )
    assert result.loc[pd.Timestamp("2024-01-08"), "cash_return"] == pytest.approx(
        1.10 ** (1 / 365) - 1
    )


def test_evaluation_slice_rebuilds_equal_exposure_and_rebases_equity() -> None:
    prices = _prices()
    signal = pd.Series([0.0] * 3 + [1.0] * 3, index=prices.index)
    full = run_long_cash_backtest(prices, signal, one_way_cost_bps=0)
    selected = slice_backtest_result(full, start=prices.index[3])

    expected_exposure = float(selected["position"].mean())
    expected = expected_exposure * selected["asset_return"]
    np.testing.assert_allclose(selected["equal_exposure_return"], expected)
    assert selected["equity"].iloc[0] == pytest.approx(1.0 + selected["net_return"].iloc[0])
    assert not np.allclose(
        selected["equal_exposure_return"],
        full.loc[selected.index, "equal_exposure_return"],
    )


def test_scenarios_include_double_cost_and_extra_delay() -> None:
    prices = _prices()
    signal = pd.Series(1.0, index=prices.index)
    scenarios = run_cost_scenarios(prices, signal, one_way_cost_bps=10)
    assert set(scenarios) == {"zero_cost", "base", "double_cost", "extra_delay"}
    base_buy = scenarios["base"].loc[scenarios["base"]["turnover"] > 0, "cost"].iloc[0]
    double_buy = scenarios["double_cost"].loc[
        scenarios["double_cost"]["turnover"] > 0, "cost"
    ].iloc[0]
    assert double_buy == pytest.approx(2 * base_buy)
    assert scenarios["extra_delay"].iloc[1]["position"] == 0.0
    assert scenarios["extra_delay"].iloc[2]["position"] == 1.0


def test_leverage_and_short_positions_are_rejected() -> None:
    prices = _prices()
    with pytest.raises(ValueError, match="\[0, 1\]"):
        run_long_cash_backtest(
            prices,
            pd.Series([1.1] * len(prices), index=prices.index),
            one_way_cost_bps=0,
        )
