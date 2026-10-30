from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from index_timing.backtest import run_long_cash_backtest
from index_timing.data import align_to_trading_calendar
from index_timing.features.caitong import assemble_caitong_features


@pytest.mark.parametrize("calendar", ["XSHG", "XHKG", "XNYS"])
def test_causal_feature_to_execution_path_across_markets(calendar: str) -> None:
    raw_dates = pd.date_range("2023-01-02", "2024-03-29", freq="B")
    trend = np.linspace(100.0, 135.0, len(raw_dates))
    raw = pd.DataFrame(
        {
            "open": trend,
            "high": trend * 1.01,
            "low": trend * 0.99,
            "close": trend * (1.0 + 0.001 * np.sin(np.arange(len(raw_dates)))),
            "volume": 1_000_000.0 + np.arange(len(raw_dates)),
        },
        index=raw_dates,
    )
    raw["adj_close"] = raw["close"]
    raw.index.name = "date"
    market, _ = align_to_trading_calendar(raw, calendar)
    bundle = assemble_caitong_features(market)
    momentum = bundle.frame["return_20d"]
    close_signal = momentum.gt(0).astype(float).where(momentum.notna())
    daily = run_long_cash_backtest(
        market,
        close_signal,
        one_way_cost_bps=2.0 if calendar == "XNYS" else 10.0,
        cash_rate=0.03,
        cash_rate_lag=0,
    )

    assert len(daily) > 200
    assert daily.index.is_monotonic_increasing
    assert daily[["position", "net_return", "cost"]].notna().all().all()
    assert daily["position"].between(0.0, 1.0).all()
    first_signal = close_signal.first_valid_index()
    assert first_signal is not None
    signal_row = market.index.get_loc(first_signal)
    if signal_row + 1 < len(market):
        next_session = market.index[signal_row + 1]
        assert daily.loc[next_session, "position"] == close_signal.loc[first_signal]
