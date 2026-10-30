from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from afactor.trader_choice import (  # noqa: E402
    ChoiceConfig,
    add_same_day_percentiles,
    assign_candidate_pools,
    build_matched_controls,
    deterministic_random_basket,
    evaluate_ablations,
    evaluate_ranking,
    load_choice_labels,
)


def _labels(*rows: tuple[str, str, str]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "decision_date": decision_date,
                "symbol": symbol,
                "label_confidence": "L4",
                "event_id": event_id,
            }
            for decision_date, symbol, event_id in rows
        ]
    )


def _candidate_row(
    decision_date: str,
    symbol: str,
    *,
    eligible: bool = True,
    gap: float = 0.0,
    mom5: float = 0.60,
    mom10: float = 0.75,
    mom20: float = 0.50,
    mom60: float = 0.65,
    ma20: float = 0.60,
    prev_ret: float = 0.60,
    vol20: float = 0.75,
    amount: float = 0.65,
    liq20: float = 0.95,
) -> dict[str, object]:
    return {
        "decision_date": decision_date,
        "symbol": symbol,
        "eligible_for_ranking": eligible,
        "gap": gap,
        "mom5_pct": mom5,
        "mom10_pct": mom10,
        "mom20_pct": mom20,
        "mom60_pct": mom60,
        "ma20_dist_pct": ma20,
        "prev_ret_pct": prev_ret,
        "vol20_pct": vol20,
        "amt_ratio20_pct": amount,
        "liq20_pct": liq20,
    }


def test_load_choice_labels_keeps_only_l4_exact_day_ordinary_stock_buys(
    tmp_path: Path,
) -> None:
    events = pd.DataFrame(
        [
            ("E1", "buy", "600001", "2026-08-01", "2026-08-01", "L4", "L4", "ordinary"),
            ("E2", "buy", "000001", "2026-08-02", "2026-08-02", "l4", "l4", "ordinary"),
            ("E3", "buy", "430001", "2026-08-03", "2026-08-03", "L4", "L4", "ordinary"),
            ("E4", "buy", "300001", "2026-08-04", "2026-08-05", "L4", "L4", "interval"),
            ("E5", "buy", "300002", "2026-08-05", "2026-08-05", "L3", "L4", "date low"),
            ("E6", "buy", "300003", "2026-08-06", "2026-08-06", "L4", "L3", "direction low"),
            ("E7", "sell_all", "300004", "2026-08-07", "2026-08-07", "L4", "L4", "sell"),
            ("E8", "buy_to_target", "300005", "2026-08-08", "2026-08-08", "L4", "L4", "not new buy"),
            ("E9", "buy", "NAME:MASKED", "2026-08-09", "2026-08-09", "L4", "L4", "masked"),
            ("E10", "buy", "510300", "2026-08-10", "2026-08-10", "L4", "L4", "ETF fund"),
        ],
        columns=[
            "event_id",
            "event_type",
            "security_code",
            "event_date_start",
            "event_date_end",
            "date_confidence",
            "direction_confidence",
            "notes",
        ],
    )
    path = tmp_path / "events.csv"
    events.to_csv(path, index=False)

    result = load_choice_labels(path)

    assert result.to_dict("records") == [
        {
            "decision_date": pd.Timestamp("2026-08-01"),
            "symbol": "600001.SH",
            "label_confidence": "L4",
            "event_id": "E1",
        },
        {
            "decision_date": pd.Timestamp("2026-08-02"),
            "symbol": "000001.SZ",
            "label_confidence": "L4",
            "event_id": "E2",
        },
        {
            "decision_date": pd.Timestamp("2026-08-03"),
            "symbol": "430001.BJ",
            "label_confidence": "L4",
            "event_id": "E3",
        },
    ]


def test_legacy_p150_boundaries_are_inclusive_and_eligibility_is_required() -> None:
    day = "2026-08-01"
    panel = pd.DataFrame(
        [
            _candidate_row(day, "A", gap=-0.06, mom10=0.70, vol20=0.70, liq20=0.90),
            _candidate_row(day, "B", gap=0.03, mom10=0.70, vol20=0.70, liq20=0.90),
            _candidate_row(day, "C", gap=-0.060001, mom10=0.70, vol20=0.70, liq20=0.90),
            _candidate_row(day, "D", gap=0.0, mom10=0.699999, vol20=0.70, liq20=0.90),
            _candidate_row(day, "E", eligible=False, gap=0.0, mom10=1.0, vol20=1.0, liq20=1.0),
        ]
    )

    result = assign_candidate_pools(panel).set_index("symbol")

    assert result.loc["A", "legacy_P150"]
    assert result.loc["B", "legacy_P150"]
    assert not result.loc["C", "legacy_P150"]
    assert not result.loc["D", "legacy_P150"]
    assert not result.loc["E", "legacy_P150"]
    assert not result.loc["E", "full_eligible"]


def test_buy_outside_pool_is_a_recall_failure_and_is_not_injected() -> None:
    day = "2026-08-01"
    panel = pd.DataFrame(
        {
            "decision_date": [day, day, day],
            "symbol": ["000001.SZ", "000002.SZ", "000003.SZ"],
            "high_recall_parent": [True, True, False],
            "choice_score": [0.9, 0.8, 100.0],
        }
    )
    labels = _labels((day, "000003.SZ", "E1"))

    result = evaluate_ranking(panel, labels, ks=(1, 2)).iloc[0]

    assert result["n_labels"] == 1
    assert result["n_labels_in_pool"] == 0
    assert result["pool_recall"] == 0.0
    assert result["recall_at_1"] == 0.0
    assert result["recall_at_2"] == 0.0
    assert not result["hit_day"]
    assert result["ranked_selected_symbols"] == ""


def test_ranking_has_one_statistical_row_per_event_day() -> None:
    panel = pd.DataFrame(
        {
            "decision_date": [
                "2026-08-01",
                "2026-08-01",
                "2026-08-01",
                "2026-08-02",
                "2026-08-02",
            ],
            "symbol": ["A", "B", "C", "D", "E"],
            "high_recall_parent": True,
            "choice_score": [3.0, 2.0, 1.0, 2.0, 1.0],
        }
    )
    labels = _labels(
        ("2026-08-01", "A", "E1"),
        ("2026-08-01", "C", "E2"),
        ("2026-08-02", "D", "E3"),
    )

    result = evaluate_ranking(panel, labels, ks=(1, 3)).set_index("decision_date")

    assert len(result) == 2
    first = result.loc[pd.Timestamp("2026-08-01")]
    second = result.loc[pd.Timestamp("2026-08-02")]
    assert first["n_labels"] == 2
    assert first["recall_at_1"] == 0.5
    assert first["recall_at_3"] == 1.0
    assert first["full_basket_at_3"]
    assert second["n_labels"] == 1


def test_score_ties_are_broken_by_symbol_ascending() -> None:
    day = "2026-08-01"
    panel = pd.DataFrame(
        {
            "decision_date": [day, day, day],
            "symbol": ["B", "C", "A"],
            "high_recall_parent": True,
            "choice_score": [1.0, 0.0, 1.0],
        }
    )
    labels = _labels((day, "B", "E1"))

    result = evaluate_ranking(panel, labels, ks=(1, 2)).iloc[0]

    assert result["recall_at_1"] == 0.0
    assert result["recall_at_2"] == 1.0
    assert result["mrr"] == 0.5
    assert result["ranked_selected_symbols"] == "B"


def test_percentiles_are_computed_within_day_and_honor_factor_direction() -> None:
    panel = pd.DataFrame(
        {
            "decision_date": [
                "2026-08-01",
                "2026-08-01",
                "2026-08-02",
                "2026-08-02",
                "2026-08-02",
            ],
            "symbol": ["A", "B", "C", "D", "E"],
            "high_recall_parent": [True, True, True, True, False],
            "larger_better": [1.0, 2.0, 10.0, 20.0, 1000.0],
            "smaller_better": [10.0, 5.0, 100.0, 50.0, -1000.0],
        }
    )

    result = add_same_day_percentiles(
        panel,
        ["larger_better", "smaller_better"],
        factor_directions={"smaller_better": -1},
    ).set_index("symbol")

    assert result.loc["A", "pct__larger_better"] == 0.5
    assert result.loc["B", "pct__larger_better"] == 1.0
    assert result.loc["C", "pct__larger_better"] == 0.5
    assert result.loc["D", "pct__larger_better"] == 1.0
    assert result.loc["A", "pct__smaller_better"] == 0.5
    assert result.loc["B", "pct__smaller_better"] == 1.0
    assert result.loc["C", "pct__smaller_better"] == 0.5
    assert result.loc["D", "pct__smaller_better"] == 1.0
    assert np.isnan(result.loc["E", "pct__larger_better"])


def test_matched_controls_are_same_day_non_buys_and_shortage_stays_na() -> None:
    panel = pd.DataFrame(
        {
            "decision_date": [
                "2026-08-01",
                "2026-08-01",
                "2026-08-01",
                "2026-08-01",
                "2026-08-01",
                "2026-08-02",
            ],
            "symbol": ["A", "B", "C", "D", "E", "F"],
            "high_recall_parent": True,
            "full_eligible": True,
            "industry": ["X", "X", "X", "Y", "Z", "X"],
        }
    )
    labels = _labels(
        ("2026-08-01", "A", "E1"),
        ("2026-08-01", "E", "E2"),
    )

    result = build_matched_controls(
        panel,
        labels,
        ["industry"],
        controls_per_label=2,
        random_seed=7,
    )

    matched = result[result["event_id"].eq("E1")]
    assert matched["match_status"].eq("matched").all()
    assert set(matched["control_symbol"]) == {"B", "C"}
    assert matched["decision_date"].eq(pd.Timestamp("2026-08-01")).all()
    assert not set(matched["control_symbol"]) & {"A", "E", "F"}

    shortage = result[result["event_id"].eq("E2")]
    assert shortage["match_status"].eq("insufficient_matches").all()
    assert shortage["control_symbol"].isna().all()
    assert shortage["n_available_controls"].eq(0).all()


def test_random_baskets_are_seeded_and_independent_of_input_order() -> None:
    panel = pd.DataFrame(
        {
            "decision_date": ["2026-08-01"] * 5 + ["2026-08-02"] * 4,
            "symbol": list("ABCDE") + list("FGHI"),
            "high_recall_parent": True,
        }
    )
    kwargs = {
        "basket_size": 3,
        "n_baskets": 3,
        "seed": 123,
    }

    first = deterministic_random_basket(
        panel,
        ["2026-08-02", "2026-08-01"],
        **kwargs,
    )
    second = deterministic_random_basket(
        panel.sample(frac=1.0, random_state=99),
        ["2026-08-01", "2026-08-02"],
        **kwargs,
    )

    assert_frame_equal(first, second)
    assert (
        first.groupby(["decision_date", "basket_id"])["symbol"].nunique() == 3
    ).all()
    assert first["basket_complete"].all()


def test_ablation_matrix_is_deterministic_and_has_frozen_experiment_set() -> None:
    day = "2026-08-01"
    panel = pd.DataFrame(
        {
            "decision_date": [day] * 4,
            "symbol": ["D", "B", "A", "C"],
            "high_recall_parent": True,
            "price_factor": [1.0, 3.0, 4.0, 2.0],
            "industry_factor": [4.0, 2.0, 1.0, 3.0],
        }
    )
    labels = _labels((day, "A", "E1"))
    families = {
        "price": ["price_factor"],
        "industry": ["industry_factor"],
    }
    config = ChoiceConfig(rank_ks=(1, 2))

    first = evaluate_ablations(panel, labels, families, config)
    second = evaluate_ablations(
        panel.sample(frac=1.0, random_state=42), labels, families, config
    )

    assert_frame_equal(first, second)
    assert first[["experiment_kind", "experiment_name", "families"]].to_dict(
        "records"
    ) == [
        {
            "experiment_kind": "baseline",
            "experiment_name": "pool_only",
            "families": "",
        },
        {
            "experiment_kind": "single_family",
            "experiment_name": "only_price",
            "families": "price",
        },
        {
            "experiment_kind": "single_family",
            "experiment_name": "only_industry",
            "families": "industry",
        },
        {
            "experiment_kind": "cumulative",
            "experiment_name": "through_price",
            "families": "price",
        },
        {
            "experiment_kind": "cumulative",
            "experiment_name": "through_industry",
            "families": "price|industry",
        },
        {
            "experiment_kind": "leave_one_family_out",
            "experiment_name": "full_minus_price",
            "families": "industry",
        },
        {
            "experiment_kind": "leave_one_family_out",
            "experiment_name": "full_minus_industry",
            "families": "price",
        },
    ]
