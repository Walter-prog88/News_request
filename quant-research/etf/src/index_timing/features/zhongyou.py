"""Disclosed Zhongyou LSTM-GHMM features and six-state soft proxy.

The report says "25+" raw features, but the five enumerated groups contain
only 17 reproducible fields (3 momentum + 4 volatility + 4 price position +
4 market structure + 2 tails).  This module deliberately implements those 17
and does not fill the dimensional inconsistency with undisclosed volume or
fundamental variables.  Six trend-volatility probabilities are constructed
separately and bypass the LSTM, exactly as described in the report.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd
from scipy.special import expit

DISCLOSED_FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    "momentum_10d",
    "momentum_20d",
    "momentum_60d",
    "realized_vol_10d",
    "realized_vol_20d",
    "realized_vol_60d",
    "volatility_percentile_90d",
    "close_to_ma20",
    "break_above_20d_high",
    "break_below_20d_low",
    "range_position_20d",
    "return_autocorr_lag1_20d",
    "return_skew_20d",
    "return_kurtosis_20d",
    "directional_consistency_20d",
    "return_quantile_95_20d",
    "return_quantile_05_20d",
)

SOFT_STATE_COLUMNS: Final[tuple[str, ...]] = (
    "soft_trend_up",
    "soft_trend_down",
    "soft_mild_up",
    "soft_mild_down",
    "soft_wide_range",
    "soft_narrow_range",
)

ZHONGYOU_FEATURE_ASSUMPTIONS: Final[tuple[str, ...]] = (
    "正文声称动量4维但只列出10/20/60日三项；忠实代理仅采用这三项。",
    "突破高点和跌破低点按相对前20日高低点的有符号幅度计算。",
    "方向一致性代理为20日涨跌方向均值的绝对值。",
    "波动率状态代理为20日已实现波动率在过去90日中的百分位。",
    "六软状态沿用10日收益、10日ATR变化和90日动态阈值；原文未披露双Sigmoid公式。",
)


@dataclass(frozen=True, slots=True)
class ZhongyouFeatureBundle:
    """Raw disclosed features, causal z-scores and bypass probabilities."""

    raw: pd.DataFrame
    standardized: pd.DataFrame
    soft_states: pd.DataFrame
    assumptions: tuple[str, ...] = ZHONGYOU_FEATURE_ASSUMPTIONS


def _validate_market(market: pd.DataFrame) -> pd.DataFrame:
    required = {"high", "low", "close"}
    missing = sorted(required - set(market.columns))
    if missing:
        raise ValueError(f"market is missing required fields: {missing}")
    if not isinstance(market.index, pd.DatetimeIndex):
        raise TypeError("market must use a DatetimeIndex")
    if not market.index.is_monotonic_increasing or market.index.has_duplicates:
        raise ValueError("market index must be sorted and unique")
    output = market.loc[:, ["high", "low", "close"]].apply(
        pd.to_numeric, errors="coerce"
    )
    if (output <= 0).any().any():
        raise ValueError("high, low and close prices must be strictly positive")
    return output.astype(float)


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return (numerator / denominator.replace(0.0, np.nan)).replace(
        [np.inf, -np.inf], np.nan
    )


def _rolling_last_percentile(values: pd.Series, window: int) -> pd.Series:
    """Percentile rank of the current value within its trailing window."""

    def rank_last(sample: np.ndarray) -> float:
        current = sample[-1]
        lower = np.count_nonzero(sample < current)
        equal = np.count_nonzero(sample == current)
        return float((lower + 0.5 * equal) / len(sample))

    return values.rolling(window, min_periods=window).apply(rank_last, raw=True)


def _rolling_autocorrelation(returns: pd.Series, window: int) -> pd.Series:
    def lag_one(sample: np.ndarray) -> float:
        left = sample[:-1]
        right = sample[1:]
        if np.std(left) == 0.0 or np.std(right) == 0.0:
            return 0.0
        return float(np.corrcoef(left, right)[0, 1])

    # window+1 returns are required to form window lagged pairs.
    return returns.rolling(window + 1, min_periods=window + 1).apply(lag_one, raw=True)


def build_disclosed_features(market: pd.DataFrame) -> pd.DataFrame:
    """Build exactly the 17 feature fields explicitly enumerated in section 2.2."""

    prices = _validate_market(market)
    close = prices["close"]
    returns = close.pct_change(fill_method=None)
    output = pd.DataFrame(index=prices.index)

    for window in (10, 20, 60):
        output[f"momentum_{window}d"] = close.pct_change(window, fill_method=None)
        output[f"realized_vol_{window}d"] = (
            returns.rolling(window, min_periods=window).std(ddof=0) * np.sqrt(252.0)
        )

    vol_20 = output["realized_vol_20d"]
    output["volatility_percentile_90d"] = _rolling_last_percentile(vol_20, 90)

    moving_average = close.rolling(20, min_periods=20).mean()
    previous_high = prices["high"].shift(1).rolling(20, min_periods=20).max()
    previous_low = prices["low"].shift(1).rolling(20, min_periods=20).min()
    current_high = prices["high"].rolling(20, min_periods=20).max()
    current_low = prices["low"].rolling(20, min_periods=20).min()
    output["close_to_ma20"] = _safe_divide(close, moving_average) - 1.0
    output["break_above_20d_high"] = (_safe_divide(close, previous_high) - 1.0).clip(
        lower=0.0
    )
    output["break_below_20d_low"] = (_safe_divide(close, previous_low) - 1.0).clip(
        upper=0.0
    )
    output["range_position_20d"] = _safe_divide(
        close - current_low, current_high - current_low
    )

    output["return_autocorr_lag1_20d"] = _rolling_autocorrelation(returns, 20)
    output["return_skew_20d"] = returns.rolling(20, min_periods=20).skew()
    output["return_kurtosis_20d"] = returns.rolling(20, min_periods=20).kurt()
    output["directional_consistency_20d"] = (
        np.sign(returns).rolling(20, min_periods=20).mean().abs()
    )
    output["return_quantile_95_20d"] = returns.rolling(20, min_periods=20).quantile(0.95)
    output["return_quantile_05_20d"] = returns.rolling(20, min_periods=20).quantile(0.05)

    output = output.loc[:, DISCLOSED_FEATURE_COLUMNS]
    return output.replace([np.inf, -np.inf], np.nan).astype(float)


def causal_rolling_zscore(
    features: pd.DataFrame,
    *,
    window: int = 90,
    min_periods: int | None = None,
) -> pd.DataFrame:
    """Standardize each row using only its trailing window, including itself."""

    if window < 2:
        raise ValueError("z-score window must be at least two")
    minimum = window if min_periods is None else min_periods
    if not 2 <= minimum <= window:
        raise ValueError("min_periods must lie between two and window")
    numeric = features.apply(pd.to_numeric, errors="coerce").astype(float)
    rolling = numeric.rolling(window, min_periods=minimum)
    mean = rolling.mean()
    standard_deviation = rolling.std(ddof=0)
    standardized = (numeric - mean) / standard_deviation.replace(0.0, np.nan)
    # A feature that is constant throughout a complete trailing window carries
    # zero standardized information, not a missing observation.
    constant = standard_deviation.eq(0.0) & numeric.notna() & mean.notna()
    standardized = standardized.mask(constant, 0.0)
    return standardized.replace([np.inf, -np.inf], np.nan)


def true_range(market: pd.DataFrame) -> pd.Series:
    """Causal Wilder true range used by the inherited six-state rules."""

    prices = _validate_market(market)
    previous_close = prices["close"].shift(1)
    components = pd.concat(
        [
            prices["high"] - prices["low"],
            (prices["high"] - previous_close).abs(),
            (prices["low"] - previous_close).abs(),
        ],
        axis=1,
    )
    return components.max(axis=1).rename("true_range")


def trend_volatility_scores(
    market: pd.DataFrame,
    *,
    lookback: int = 10,
    threshold_window: int = 90,
) -> pd.DataFrame:
    """Return causal z-scores for 10-day return and 10-day ATR change."""

    prices = _validate_market(market)
    trend = prices["close"].pct_change(lookback, fill_method=None)
    atr = true_range(prices).rolling(lookback, min_periods=lookback).mean()
    atr_change = atr.pct_change(lookback, fill_method=None)
    inputs = pd.DataFrame({"trend": trend, "atr_change": atr_change}, index=prices.index)
    standardized = causal_rolling_zscore(inputs, window=threshold_window)
    return standardized.rename(columns={"trend": "trend_z", "atr_change": "atr_change_z"})


def build_soft_state_probabilities(
    market: pd.DataFrame,
    *,
    threshold: float = 0.524,
    slope: float = 5.0,
    lookback: int = 10,
    threshold_window: int = 90,
) -> pd.DataFrame:
    """Construct the six inherited state memberships with smooth boundaries.

    The paper discloses the threshold (mean ± 0.524 sigma) and says an
    asymmetric double sigmoid is used, but provides no equation.  The proxy
    below softens the ordered 1.0 hard rules: strong trend first, then wide or
    narrow range, and finally mild direction.  Scores are normalized row-wise
    so they are valid GHMM covariates rather than a hard one-hot label.
    """

    if threshold <= 0 or slope <= 0:
        raise ValueError("threshold and sigmoid slope must be positive")
    scores = trend_volatility_scores(
        market, lookback=lookback, threshold_window=threshold_window
    )
    trend = scores["trend_z"].to_numpy(dtype=float)
    volatility = scores["atr_change_z"].to_numpy(dtype=float)

    trend_up = expit(slope * (trend - threshold))
    trend_down = expit(slope * (-trend - threshold))
    non_trend = (1.0 - trend_up) * (1.0 - trend_down)
    wide_gate = expit(slope * (volatility - threshold))
    narrow_gate = expit(slope * (-volatility - threshold))
    middle_volatility = (1.0 - wide_gate) * (1.0 - narrow_gate)
    mild_up_gate = expit(slope * trend)

    raw = np.column_stack(
        [
            trend_up,
            trend_down,
            non_trend * middle_volatility * mild_up_gate,
            non_trend * middle_volatility * (1.0 - mild_up_gate),
            non_trend * wide_gate,
            non_trend * narrow_gate,
        ]
    )
    denominator = raw.sum(axis=1, keepdims=True)
    probabilities = np.divide(
        raw,
        denominator,
        out=np.full_like(raw, np.nan),
        where=np.isfinite(denominator) & (denominator > 0),
    )
    return pd.DataFrame(probabilities, index=scores.index, columns=SOFT_STATE_COLUMNS)


def hard_six_state(
    market: pd.DataFrame,
    *,
    threshold: float = 0.524,
    lookback: int = 10,
    threshold_window: int = 90,
) -> pd.Series:
    """Auditable 1.0-style hard classification used only for diagnostics."""

    scores = trend_volatility_scores(
        market, lookback=lookback, threshold_window=threshold_window
    )
    trend = scores["trend_z"]
    volatility = scores["atr_change_z"]
    state = pd.Series(pd.NA, index=scores.index, dtype="string", name="hard_state")
    valid = trend.notna() & volatility.notna()
    state.loc[valid & (trend > threshold)] = "trend_up"
    state.loc[valid & (trend < -threshold)] = "trend_down"
    unresolved = valid & state.isna()
    state.loc[unresolved & (volatility > threshold)] = "wide_range"
    state.loc[unresolved & (volatility < -threshold)] = "narrow_range"
    unresolved = valid & state.isna()
    state.loc[unresolved & (trend >= 0)] = "mild_up"
    state.loc[unresolved & (trend < 0)] = "mild_down"
    return state


def assemble_zhongyou_features(
    market: pd.DataFrame,
    *,
    zscore_window: int = 90,
    soft_threshold: float = 0.524,
    soft_slope: float = 5.0,
) -> ZhongyouFeatureBundle:
    raw = build_disclosed_features(market)
    standardized = causal_rolling_zscore(raw, window=zscore_window)
    soft_states = build_soft_state_probabilities(
        market,
        threshold=soft_threshold,
        slope=soft_slope,
        threshold_window=zscore_window,
    )
    return ZhongyouFeatureBundle(raw=raw, standardized=standardized, soft_states=soft_states)


__all__ = [
    "DISCLOSED_FEATURE_COLUMNS",
    "SOFT_STATE_COLUMNS",
    "ZHONGYOU_FEATURE_ASSUMPTIONS",
    "ZhongyouFeatureBundle",
    "assemble_zhongyou_features",
    "build_disclosed_features",
    "build_soft_state_probabilities",
    "causal_rolling_zscore",
    "hard_six_state",
    "trend_volatility_scores",
    "true_range",
]
