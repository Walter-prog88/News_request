from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from afactor.trader_chain_rules import (  # noqa: E402
    ChainRuleConfig,
    score_chain_directions,
    score_within_chains,
    select_chain_portfolio,
    select_top_chains,
)


FAMILIES = {
    "strength": ["relative_strength"],
    "leadership": ["leadership"],
    "shape": ["shape_acceleration", "shape_rebound", "shape_breakout"],
    "evidence": ["evidence"],
}


def test_chain_direction_requires_four_complete_equal_weight_factors() -> None:
    panel = pd.DataFrame(
        {
            "decision_date": ["2026-08-20"] * 3 + ["2026-08-21"] * 2,
            "chain_id": ["B", "A", "C", "D", "E"],
            "chain_eligible": True,
            "breadth": [1.0, 3.0, 2.0, 2.0, 1.0],
            "momentum": [3.0, 1.0, 2.0, 2.0, 1.0],
            "liquidity": [3.0, 3.0, 1.0, 2.0, 1.0],
            "confirmation": [3.0, 3.0, 1.0, np.nan, 1.0],
            # These columns must never affect generated scores.
            "is_disclosed_buy": [True, False, True, False, True],
            "forward_return": [100.0, -100.0, 50.0, -50.0, 20.0],
        }
    )
    factors = ["breadth", "momentum", "liquidity", "confirmation"]
    result = score_chain_directions(panel, factors).set_index(
        ["decision_date", "chain_id"]
    )

    day = pd.Timestamp("2026-08-20")
    assert result.loc[(day, "A"), "chain_direction_score"] == result.loc[
        (day, "B"), "chain_direction_score"
    ]
    assert result.loc[(day, "A"), "chain_direction_rank"] == 1
    assert result.loc[(day, "B"), "chain_direction_rank"] == 2
    incomplete = (pd.Timestamp("2026-08-21"), "D")
    assert result.loc[incomplete, "chain_factor_count"] == 3
    assert np.isnan(result.loc[incomplete, "chain_direction_score"])

    contaminated = panel.assign(
        is_disclosed_buy=~panel["is_disclosed_buy"],
        forward_return=-panel["forward_return"],
        pnl=np.arange(len(panel)) * 1e9,
    )
    generated = [
        *(f"chain_pct__{factor}" for factor in factors),
        "chain_factor_count",
        "chain_direction_score",
        "chain_direction_rank",
    ]
    assert_frame_equal(
        score_chain_directions(panel, factors)[generated],
        score_chain_directions(contaminated, factors)[generated],
    )


def test_within_chain_uses_four_families_shape_max_and_neutral_fill() -> None:
    panel = pd.DataFrame(
        {
            "decision_date": ["2026-08-20"] * 7,
            "chain_id": ["A", "A", "A", "A", "B", "B", "B"],
            "symbol": ["A1", "A2", "A3", "A4", "B1", "B2", "B3"],
            "eligible_for_ranking": [True, True, True, False, True, True, True],
            "relative_strength": [3, 2, 1, 999, 30, 20, 10],
            "leadership": [3, 2, 1, 999, 30, 20, 10],
            "shape_acceleration": [3, 1, 2, 999, 30, 20, 10],
            "shape_rebound": [1, 3, 2, 999, 30, 20, 10],
            "shape_breakout": [1, 2, 3, 999, 30, 20, 10],
            "evidence": [3, 2, np.nan, 999, 30, 20, 10],
        }
    )

    result = score_within_chains(panel, FAMILIES).set_index("symbol")
    assert result.loc["A1", "within_chain_score"] == 1.0
    assert result.loc["A2", "family_score__shape"] == 1.0
    assert result.loc["A3", "family_score__shape"] == 1.0
    assert np.isnan(result.loc["A3", "family_score__evidence"])
    assert result.loc["A3", "family_gate__evidence"] == 0.5
    assert result.loc["A3", "neutral_fill_count"] == 1
    expected_a3 = (1 / 3 + 1 / 3 + 1.0 + 0.5) / 4
    assert np.isclose(result.loc["A3", "within_chain_score"], expected_a3)
    assert np.isnan(result.loc["A4", "within_chain_score"])
    # B percentiles are independent of the numerical scale in chain A.
    assert result.loc["B1", "within_chain_score"] == 1.0
    assert result.loc["B3", "within_chain_rank"] == 3


def test_select_top_chains_is_per_day_and_input_order_independent() -> None:
    config = ChainRuleConfig(n_chains=2)
    scored = pd.DataFrame(
        {
            "decision_date": ["2026-08-20"] * 4 + ["2026-08-21"] * 2,
            "chain_id": ["B", "C", "A", "D", "Y", "X"],
            "chain_eligible": [True, True, True, False, True, True],
            "chain_direction_score": [0.9, 0.8, 0.9, 1.0, 0.2, 0.3],
        }
    )
    first = select_top_chains(scored, config)
    second = select_top_chains(
        scored.sample(frac=1.0, random_state=9).reset_index(drop=True), config
    )

    columns = ["decision_date", "chain_id", "chain_selection_rank"]
    assert_frame_equal(first[columns], second[columns])
    assert first.loc[
        first["decision_date"].eq(pd.Timestamp("2026-08-20")), "chain_id"
    ].tolist() == ["A", "B"]
    assert first.loc[
        first["decision_date"].eq(pd.Timestamp("2026-08-21")), "chain_id"
    ].tolist() == ["X", "Y"]


def _selected_chains(day: str = "2026-08-20") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "decision_date": [day, day],
            "chain_id": ["A", "B"],
            "chain_eligible": [True, True],
            "chain_direction_score": [0.9, 0.8],
            "chain_selection_rank": [1, 2],
            "chain_selected": [True, True],
        }
    )


def _portfolio_members(day: str = "2026-08-20") -> pd.DataFrame:
    rows = [
        ("A", "A1", 1.00, "s1"),
        ("A", "A2", 0.95, "s1"),
        ("A", "A3", 0.90, "s2"),
        ("A", "A4", 0.80, "s2"),
        ("A", "A5", 0.70, "s3"),
        ("B", "B1", 1.00, "t1"),
        ("B", "B2", 0.95, "t1"),
        ("B", "B3", 0.90, "t2"),
        ("B", "B4", 0.80, "t1"),
        ("B", "B5", 0.70, "t2"),
    ]
    return pd.DataFrame(
        [
            {
                "decision_date": day,
                "chain_id": chain,
                "symbol": symbol,
                "eligible_for_ranking": True,
                "within_chain_score": score,
                "subsegment": subsegment,
            }
            for chain, symbol, score, subsegment in rows
        ]
    )


def _correlations(day: str = "2026-08-20") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "decision_date": [day, day],
            "symbol_a": ["A1", "A1"],
            "symbol_b": ["A2", "B1"],
            "correlation": [0.90, 0.91],
        }
    )


def test_portfolio_uses_full_universe_greedy_diversification_and_fixed_slots() -> None:
    config = ChainRuleConfig(n_chains=2, names_per_chain=4)
    first = select_chain_portfolio(
        _portfolio_members(), _selected_chains(), _correlations(), config
    )
    second = select_chain_portfolio(
        _portfolio_members().sample(frac=1.0, random_state=3),
        _selected_chains().sample(frac=1.0, random_state=4),
        _correlations().sample(frac=1.0, random_state=5),
        config,
    )

    position_columns = [
        "decision_date",
        "chain_id",
        "symbol",
        "position_rank_in_chain",
        "position_weight",
    ]
    assert_frame_equal(first.positions[position_columns], second.positions[position_columns])
    by_chain = first.positions.groupby("chain_id")["symbol"].apply(list)
    assert by_chain["A"] == ["A1", "A3", "A4", "A5"]
    assert by_chain["B"] == ["B2", "B3", "B4", "B5"]
    assert first.positions["symbol"].is_unique
    assert first.positions["position_weight"].eq(1 / 8).all()
    assert np.isclose(first.positions["position_weight"].sum(), 1.0)
    assert first.chain_audit["selection_status"].eq("selected_full").all()
    assert first.chain_audit["selected_subsegment_count"].ge(2).all()
    assert first.chain_audit["n_rejected_high_correlation"].ge(1).all()
    assert first.date_summary.iloc[0]["cash_weight"] == 0.0


def test_partial_chain_is_kept_and_unfilled_slots_remain_cash() -> None:
    day = "2026-08-20"
    config = ChainRuleConfig(n_chains=1, names_per_chain=4)
    one_chain = _selected_chains(day).iloc[[0]].assign(chain_selection_rank=1)
    single_subsegment = pd.DataFrame(
        {
            "decision_date": [day] * 4,
            "chain_id": ["A"] * 4,
            "symbol": ["X1", "X2", "X3", "X4"],
            "eligible_for_ranking": True,
            "within_chain_score": [1.0, 0.9, 0.8, 0.7],
            "subsegment": ["only"] * 4,
        }
    )
    result = select_chain_portfolio(single_subsegment, one_chain, config=config)

    assert result.positions["symbol"].tolist() == ["X1", "X2"]
    assert result.positions["position_weight"].eq(0.25).all()
    assert result.chain_audit.iloc[0]["selection_status"] == "selected_partial"
    assert result.chain_audit.iloc[0]["selected_subsegment_count"] == 1
    assert not result.chain_audit.iloc[0]["subsegment_target_met"]
    assert result.chain_audit.iloc[0]["unfilled_slots"] == 2
    assert np.isclose(result.date_summary.iloc[0]["invested_weight"], 0.5)
    assert np.isclose(result.date_summary.iloc[0]["cash_weight"], 0.5)


def test_correlation_dedupe_can_leave_partial_chain_instead_of_zeroing_it() -> None:
    day = "2026-08-20"
    config = ChainRuleConfig(n_chains=1, names_per_chain=4)
    one_chain = _selected_chains(day).iloc[[0]].assign(chain_selection_rank=1)
    members = pd.DataFrame(
        {
            "decision_date": [day] * 3,
            "chain_id": ["A"] * 3,
            "symbol": ["X1", "X2", "X3"],
            "eligible_for_ranking": True,
            "within_chain_score": [1.0, 0.9, 0.8],
            "subsegment": ["s1", "s2", "s3"],
        }
    )
    correlations = pd.DataFrame(
        {
            "decision_date": [day, day],
            "symbol_a": ["X1", "X1"],
            "symbol_b": ["X2", "X3"],
            "correlation": [0.90, 0.91],
        }
    )
    result = select_chain_portfolio(members, one_chain, correlations, config)

    assert result.positions["symbol"].tolist() == ["X1"]
    assert result.chain_audit.iloc[0]["selection_status"] == "selected_partial"
    assert result.chain_audit.iloc[0]["n_rejected_high_correlation"] == 2
    assert np.isclose(result.date_summary.iloc[0]["cash_weight"], 0.75)


def test_legacy_chain_stage_alias_is_canonicalized_to_subsegment() -> None:
    day = "2026-08-20"
    config = ChainRuleConfig(n_chains=1, names_per_chain=2)
    one_chain = _selected_chains(day).iloc[[0]].assign(chain_selection_rank=1)
    members = pd.DataFrame(
        {
            "decision_date": [day, day],
            "chain_id": ["A", "A"],
            "symbol": ["X1", "X2"],
            "eligible_for_ranking": True,
            "within_chain_score": [1.0, 0.9],
            "chain_stage": ["s1", "s2"],
        }
    )
    result = select_chain_portfolio(members, one_chain, config=config)
    assert result.positions["subsegment"].tolist() == ["s1", "s2"]
    assert result.chain_audit.iloc[0]["subsegment_source_col"] == "chain_stage"
