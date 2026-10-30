from __future__ import annotations

import numpy as np
import pandas as pd
import pandas.testing as pdt

from index_timing.features.zhongyou import (
    DISCLOSED_FEATURE_COLUMNS,
    SOFT_STATE_COLUMNS,
    assemble_zhongyou_features,
    build_disclosed_features,
    build_soft_state_probabilities,
    causal_rolling_zscore,
    trend_volatility_scores,
)


def _market(rows: int = 360) -> pd.DataFrame:
    rng = np.random.default_rng(41)
    index = pd.bdate_range("2018-01-02", periods=rows)
    regimes = np.concatenate(
        [
            np.full(rows // 3, 0.002),
            np.full(rows // 3, -0.0015),
            np.zeros(rows - 2 * (rows // 3)),
        ]
    )
    returns = regimes + rng.normal(0.0, 0.009, rows)
    close = 100.0 * np.exp(np.cumsum(returns))
    spread = 0.004 + np.abs(rng.normal(0.0, 0.003, rows))
    return pd.DataFrame(
        {
            "open": close * (1.0 + rng.normal(0.0, 0.002, rows)),
            "high": close * (1.0 + spread),
            "low": close * (1.0 - spread),
            "close": close,
            "volume": rng.lognormal(12.0, 0.2, rows),
        },
        index=index,
    )


def test_disclosed_feature_dictionary_is_exactly_17_not_claimed_25_plus() -> None:
    features = build_disclosed_features(_market())
    assert tuple(features.columns) == DISCLOSED_FEATURE_COLUMNS
    assert features.shape[1] == 17
    assert {"momentum_10d", "momentum_20d", "momentum_60d"}.issubset(features)
    assert "volume" not in " ".join(features.columns)


def test_feature_formulas_use_prior_breakout_level_and_20_day_tails() -> None:
    market = _market()
    features = build_disclosed_features(market)
    row = market.index[80]
    previous_high = market["high"].shift(1).rolling(20).max().loc[row]
    expected_breakout = max(market.loc[row, "close"] / previous_high - 1.0, 0.0)
    expected_tail = market["close"].pct_change().rolling(20).quantile(0.95).loc[row]
    assert np.isclose(features.loc[row, "break_above_20d_high"], expected_breakout)
    assert np.isclose(features.loc[row, "return_quantile_95_20d"], expected_tail)


def test_rolling_zscore_is_causal_and_constant_windows_map_to_zero() -> None:
    index = pd.bdate_range("2023-01-02", periods=12)
    features = pd.DataFrame(
        {"varying": np.arange(12.0), "constant": 3.0}, index=index
    )
    standardized = causal_rolling_zscore(features, window=5)
    expected = (4.0 - 2.0) / np.std(np.arange(5.0), ddof=0)
    assert np.isclose(standardized.iloc[4]["varying"], expected)
    assert standardized.iloc[4]["constant"] == 0.0

    changed = features.copy()
    changed.iloc[-1] = 1_000_000.0
    changed_standardized = causal_rolling_zscore(changed, window=5)
    pdt.assert_frame_equal(standardized.iloc[:-1], changed_standardized.iloc[:-1])


def test_soft_states_use_a_0524_k5_are_probabilities_and_causal() -> None:
    market = _market()
    probabilities = build_soft_state_probabilities(market, threshold=0.524, slope=5.0)
    assert tuple(probabilities.columns) == SOFT_STATE_COLUMNS
    valid = probabilities.dropna()
    assert len(valid) > 100
    np.testing.assert_allclose(valid.sum(axis=1), 1.0, atol=1e-12)
    assert ((valid >= 0.0) & (valid <= 1.0)).all().all()

    scores = trend_volatility_scores(market)
    strongest_up = scores["trend_z"].idxmax()
    assert probabilities.loc[strongest_up].idxmax() == "soft_trend_up"

    changed_future = market.copy()
    changed_future.loc[market.index[-1], ["high", "low", "close"]] *= [10.0, 0.1, 5.0]
    changed = build_soft_state_probabilities(changed_future)
    pdt.assert_frame_equal(probabilities.iloc[:-1], changed.iloc[:-1])


def test_bundle_keeps_17_lstm_inputs_separate_from_six_ghmm_bypass_inputs() -> None:
    bundle = assemble_zhongyou_features(_market())
    assert bundle.raw.shape[1] == 17
    assert bundle.standardized.shape[1] == 17
    assert bundle.soft_states.shape[1] == 6
    assert bundle.raw.index.equals(bundle.soft_states.index)

