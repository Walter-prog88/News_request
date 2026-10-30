from __future__ import annotations

import numpy as np
import pandas as pd

from index_timing.baselines import (
    moving_average_signal,
    time_series_momentum_signal,
    volatility_target_position,
)


def test_baselines_are_causal_and_bounded() -> None:
    index = pd.bdate_range("2020-01-01", periods=400)
    close = pd.Series(100 * np.exp(np.linspace(0, 0.4, len(index))), index=index)
    altered = close.copy()
    altered.iloc[-1] *= 0.2

    for builder in (
        moving_average_signal,
        time_series_momentum_signal,
        volatility_target_position,
    ):
        original = builder(close)
        changed = builder(altered)
        pd.testing.assert_series_equal(original.iloc[:-1], changed.iloc[:-1])
        assert original.dropna().between(0, 1).all()

