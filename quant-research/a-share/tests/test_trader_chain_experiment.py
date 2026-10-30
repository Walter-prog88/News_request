from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from afactor.trader_chain_experiment import (
    CHAIN_DIRECTION_FACTORS,
    build_chain_direction_inputs,
    trailing_return_correlations,
)


def test_build_chain_direction_inputs_uses_market_percentiles_and_breadth() -> None:
    rows = []
    for index in range(6):
        rows.append(
            {
                "decision_date": "2025-01-06",
                "symbol": f"S{index}",
                "chain_id": "robot",
                "chain_name": "机器人",
                "eligible_for_ranking": index != 5,
                "mom20_pct": 0.1 * (index + 1),
                "mom60_pct": 0.2 + 0.1 * index,
                "ma20_dist": -0.1 if index < 2 else 0.1,
                "amt_ratio20_pct": 0.15 * (index + 1),
            }
        )
    result = build_chain_direction_inputs(pd.DataFrame(rows), minimum_members=5)
    assert len(result) == 1
    row = result.iloc[0]
    assert row["eligible_member_count"] == 5
    assert row["chain_mom20_member_market_pct"] == pytest.approx(0.3)
    assert row["chain_breadth_above_ma20"] == pytest.approx(0.6)
    assert bool(row["chain_eligible"])
    assert result[list(CHAIN_DIRECTION_FACTORS)].notna().all(axis=None)


def test_build_chain_direction_inputs_rejects_duplicate_membership() -> None:
    row = {
        "decision_date": "2025-01-06",
        "symbol": "000001.SZ",
        "chain_id": "robot",
        "eligible_for_ranking": True,
        "mom20_pct": 0.5,
        "mom60_pct": 0.5,
        "ma20_dist": 0.1,
        "amt_ratio20_pct": 0.5,
    }
    with pytest.raises(ValueError, match="unique"):
        build_chain_direction_inputs(pd.DataFrame([row, row]))


def test_trailing_return_correlations_uses_only_sessions_before_decision() -> None:
    dates = pd.bdate_range("2024-01-01", periods=70)
    base = np.arange(1.0, 71.0)
    closes = pd.DataFrame(
        {
            "A": base,
            "B": base * 2.0,
            "C": base[::-1] + 100.0,
        },
        index=dates,
    )
    decision = dates[-1]
    before = trailing_return_correlations(
        decision,
        ["A", "B", "C"],
        closes,
        lookback_sessions=60,
        minimum_observations=40,
    )
    changed = closes.copy()
    changed.loc[decision, ["A", "B", "C"]] = [10_000, 1, 50_000]
    after = trailing_return_correlations(
        decision,
        ["A", "B", "C"],
        changed,
        lookback_sessions=60,
        minimum_observations=40,
    )
    pd.testing.assert_frame_equal(before, after)
    assert before["lookback_end"].max() < decision
    ab = before.loc[
        before[["symbol_a", "symbol_b"]].apply(set, axis=1).eq({"A", "B"}),
        "correlation",
    ]
    assert ab.iloc[0] == pytest.approx(1.0)
