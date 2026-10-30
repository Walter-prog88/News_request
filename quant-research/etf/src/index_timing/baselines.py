"""Pre-registered transparent timing baselines."""

from __future__ import annotations

import numpy as np
import pandas as pd


def buy_and_hold(index: pd.Index) -> pd.Series:
    return pd.Series(1.0, index=index, name="buy_and_hold")


def moving_average_signal(close: pd.Series, *, window: int = 200) -> pd.Series:
    prices = pd.to_numeric(close, errors="coerce")
    signal = (prices > prices.rolling(window, min_periods=window).mean()).astype(float)
    return signal.where(prices.rolling(window, min_periods=window).count() == window).rename(
        f"ma{window}"
    )


def time_series_momentum_signal(close: pd.Series, *, lookback: int = 252) -> pd.Series:
    prices = pd.to_numeric(close, errors="coerce")
    momentum = prices.pct_change(lookback, fill_method=None)
    return (momentum > 0).astype(float).where(momentum.notna()).rename(f"tsmom_{lookback}")


def volatility_target_position(
    close: pd.Series,
    *,
    window: int = 20,
    annual_target: float = 0.10,
) -> pd.Series:
    returns = pd.to_numeric(close, errors="coerce").pct_change(fill_method=None)
    realised = returns.rolling(window, min_periods=window).std(ddof=1) * np.sqrt(252)
    return (annual_target / realised.replace(0.0, np.nan)).clip(0.0, 1.0).rename("vol_target")


def all_baselines(close: pd.Series) -> dict[str, pd.Series]:
    return {
        "buy_and_hold": buy_and_hold(close.index),
        "ma200": moving_average_signal(close),
        "tsmom_12m": time_series_momentum_signal(close),
        "vol_target_10pct": volatility_target_position(close),
    }

