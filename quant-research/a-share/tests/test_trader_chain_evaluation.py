from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from afactor.trader_chain_evaluation import (
    ChainEvaluationConfig,
    R1GateConfig,
    aggregate_equal_weight_slots,
    equal_weight_return_with_cash,
    evaluate_chain_returns,
    prepare_daily_strategy_returns,
    summarize_paired_net_excess_inference,
)


def test_equal_weight_cash_slots_and_slot_aggregation() -> None:
    assert equal_weight_return_with_cash([0.12, 0.06], 4) == pytest.approx(0.045)
    with pytest.raises(ValueError, match="missing outcomes are not cash"):
        equal_weight_return_with_cash([0.12, np.nan], 4)

    slots = pd.DataFrame(
        {
            "decision_date": ["2024-01-02", "2024-01-02"],
            "strategy": ["R1", "R1"],
            "asset_return": [0.12, 0.06],
            "slots_target": [4, 4],
            "event_id": ["ignored-a", "ignored-b"],
        }
    )
    daily = aggregate_equal_weight_slots(slots)
    assert daily.loc[0, "gross_return"] == pytest.approx(0.045)
    assert daily.loc[0, "slots_target"] == 4
    assert daily.loc[0, "slots_filled"] == 2


def test_historical_cost_switch_scales_cost_by_invested_fraction() -> None:
    daily = pd.DataFrame(
        {
            "decision_date": ["2023-08-25", "2023-08-28", "2023-08-29"],
            "strategy": ["R1", "R1-a", "cash"],
            "gross_return": [0.10, 0.10, 0.0],
            "slots_target": [12, 12, 12],
            "slots_filled": [6, 6, 0],
            "turnover": [99.0, 99.0, 99.0],
        }
    )
    result = prepare_daily_strategy_returns(daily).set_index("strategy")

    assert result.at["R1", "round_trip_cost_rate"] == pytest.approx(0.0012)
    assert result.at["R1", "applied_round_trip_cost"] == pytest.approx(0.0006)
    assert result.at["R1", "net_return"] == pytest.approx(0.0994)
    assert result.at["R1-a", "round_trip_cost_rate"] == pytest.approx(0.0007)
    assert result.at["R1-a", "applied_round_trip_cost"] == pytest.approx(0.00035)
    assert result.at["R1-a", "net_return_double_cost"] == pytest.approx(0.0993)
    assert result.at["cash", "applied_round_trip_cost"] == 0.0
    assert not result["turnover_used_for_cost"].any()


def _passing_monthly_panel() -> pd.DataFrame:
    dates = pd.date_range("2021-01-01", "2025-12-01", freq="MS")
    rows: list[dict[str, object]] = []
    # Non-constant excess keeps IR finite while remaining well above the gate.
    r1_pattern = np.resize(np.array([0.024, 0.026, 0.025, 0.027]), len(dates))
    all_members_pattern = np.resize(np.array([0.010, 0.012, 0.011, 0.013]), len(dates))
    random_pattern = np.resize(np.array([0.007, 0.010, 0.009, 0.011]), len(dates))
    for date, r1_return, member_return, random_return in zip(
        dates, r1_pattern, all_members_pattern, random_pattern
    ):
        for strategy, gross in (
            ("R1", r1_return),
            ("top3_chain_all_members", member_return),
            ("top3_chain_random_4", random_return),
        ):
            rows.append(
                {
                    "decision_date": date,
                    "strategy": strategy,
                    "gross_return": gross,
                    "slots_target": 12,
                    "slots_filled": 12,
                }
            )
    return pd.DataFrame(rows)


def test_complete_evaluation_compounds_and_passes_the_fixed_gate() -> None:
    result = evaluate_chain_returns(
        _passing_monthly_panel(), chain_contribution_share=0.40
    )

    assert len(result.daily) == 180
    assert set(result.strategy_summary["return_basis"]) == {
        "gross",
        "net_historical_cost",
        "net_double_cost",
    }
    assert set(result.paired_summary["control_strategy"]) == {
        "top3_chain_all_members",
        "top3_chain_random_4",
    }
    assert len(result.half_year_summary) == 20
    assert result.gate.status == "pass"
    assert result.gate.all_required_checks_evaluated
    assert result.gate.daily_computable_checks_pass
    assert result.gate.all_protocol_checks_pass
    checks = result.gate.checks.set_index("check_id")
    assert checks.at["minimum_decisions", "observed"] == 60
    assert checks.at["positive_half_years_both_controls", "observed"] == 10
    assert checks.at["half_year_contribution_concentration", "observed"] < 0.50
    assert checks.at["chain_contribution_concentration", "observed"] == 0.40


def test_missing_chain_concentration_is_not_silently_passed() -> None:
    result = evaluate_chain_returns(_passing_monthly_panel())
    checks = result.gate.checks.set_index("check_id")

    assert result.gate.status == "not_evaluated"
    assert result.gate.daily_computable_checks_pass
    assert not result.gate.all_required_checks_evaluated
    assert not result.gate.all_protocol_checks_pass
    assert checks.at["chain_contribution_concentration", "status"] == "not_evaluated"
    assert pd.isna(checks.at["chain_contribution_concentration", "passed"])


def test_extra_transaction_and_label_columns_cannot_change_results() -> None:
    baseline = _passing_monthly_panel()
    contaminated = baseline.assign(
        event_id="E1",
        is_disclosed_buy=True,
        label_confidence="L4",
        realized_pnl=999.0,
    )
    first = evaluate_chain_returns(baseline, chain_contribution_share=0.40)
    second = evaluate_chain_returns(contaminated, chain_contribution_share=0.40)

    core = [
        "decision_date",
        "strategy",
        "gross_return",
        "net_return",
        "net_return_double_cost",
        "invested_fraction",
    ]
    assert_frame_equal(first.daily[core], second.daily[core])
    assert_frame_equal(first.paired_summary, second.paired_summary)
    assert_frame_equal(first.gate.checks, second.gate.checks)


def test_invalid_grain_slots_and_gate_configuration_are_rejected() -> None:
    daily = pd.DataFrame(
        {
            "decision_date": ["2024-01-02", "2024-01-02"],
            "strategy": ["R1", "R1"],
            "gross_return": [0.01, 0.02],
            "slots_target": [12, 12],
            "slots_filled": [12, 12],
        }
    )
    with pytest.raises(ValueError, match="unique by date and strategy"):
        prepare_daily_strategy_returns(daily)
    with pytest.raises(ValueError, match="cannot exceed"):
        prepare_daily_strategy_returns(
            daily.iloc[[0]].assign(slots_filled=13)
        )
    with pytest.raises(ValueError, match="one of controls"):
        R1GateConfig(main_control="not-a-control")
    with pytest.raises(ValueError, match="non-negative"):
        evaluate_chain_returns(
            _passing_monthly_panel(), chain_contribution_share=-0.1
        )
    with pytest.raises(ValueError, match="finite"):
        evaluate_chain_returns(
            _passing_monthly_panel(), chain_contribution_share=np.inf
        )


def test_weekly_annualisation_is_configurable_without_changing_cost_clock() -> None:
    monthly = evaluate_chain_returns(
        _passing_monthly_panel(),
        chain_contribution_share=0.40,
    )
    weekly_config = ChainEvaluationConfig(periods_per_year=52)
    weekly = evaluate_chain_returns(
        _passing_monthly_panel(),
        config=weekly_config,
        chain_contribution_share=0.40,
    )
    monthly_row = monthly.strategy_summary.set_index(
        ["strategy", "return_basis"]
    ).loc[("R1", "net_historical_cost")]
    weekly_row = weekly.strategy_summary.set_index(
        ["strategy", "return_basis"]
    ).loc[
        ("R1", "net_historical_cost")
    ]
    assert weekly_row["annualized_return"] > monthly_row["annualized_return"]
    assert weekly.daily["round_trip_cost_rate"].isin([0.0012, 0.0007]).all()


def test_empty_or_incomplete_strategy_set_returns_not_evaluated_gate() -> None:
    empty = pd.DataFrame(
        columns=[
            "decision_date",
            "strategy",
            "gross_return",
            "slots_target",
            "slots_filled",
        ]
    )
    result = evaluate_chain_returns(empty)
    assert result.daily.empty
    assert result.paired_daily.empty
    assert result.paired_summary.empty
    assert result.gate.status == "not_evaluated"
    assert not result.gate.all_protocol_checks_pass

    only_r1 = _passing_monthly_panel().loc[lambda x: x["strategy"].eq("R1")]
    result = evaluate_chain_returns(only_r1, chain_contribution_share=0.4)
    assert result.gate.status == "not_evaluated"
    assert not result.gate.daily_computable_checks_pass


def test_custom_input_column_names_keep_canonical_paired_outputs() -> None:
    custom = _passing_monthly_panel().rename(
        columns={
            "decision_date": "date",
            "strategy": "model",
            "gross_return": "return_value",
            "slots_target": "capacity",
            "slots_filled": "positions",
        }
    )
    config = ChainEvaluationConfig(
        decision_date_col="date",
        strategy_col="model",
        gross_return_col="return_value",
        slots_target_col="capacity",
        slots_filled_col="positions",
    )
    result = evaluate_chain_returns(
        custom, config=config, chain_contribution_share=0.4
    )
    assert {"decision_date", "r1_gross_return", "control_gross_return"}.issubset(
        result.paired_daily.columns
    )
    assert result.gate.all_protocol_checks_pass


def test_paired_inference_is_grouped_by_control_and_reproducible() -> None:
    paired = pd.DataFrame(
        {
            "decision_date": pd.to_datetime(
                [
                    "2024-01-02",
                    "2024-02-01",
                    "2024-03-01",
                    "2024-01-02",
                    "2024-02-01",
                    "2024-03-01",
                ]
            ),
            "control_strategy": ["B", "B", "B", "A", "A", "A"],
            "paired_net_excess": [0.03, -0.01, 0.02, 0.01, 0.02, 0.03],
        }
    )
    first = summarize_paired_net_excess_inference(
        paired, bootstrap_iterations=1_000, seed=17
    )
    second = summarize_paired_net_excess_inference(
        paired.sample(frac=1.0, random_state=99),
        bootstrap_iterations=1_000,
        seed=17,
    )

    assert_frame_equal(first, second)
    assert first["control_strategy"].tolist() == ["A", "B"]
    indexed = first.set_index("control_strategy")
    assert indexed.at["A", "mean_paired_net_excess"] == pytest.approx(0.02)
    assert indexed.at["B", "mean_paired_net_excess"] == pytest.approx(0.04 / 3)
    assert indexed.at["A", "bootstrap_iterations"] == 1_000
    assert indexed.at["A", "bootstrap_method"] == "decision_date_percentile"
    assert indexed.at["A", "bootstrap_ci_lower"] <= 0.02
    assert indexed.at["A", "bootstrap_ci_upper"] >= 0.02


def test_paired_inference_default_is_10000_date_bootstraps_and_exact_sign_test() -> None:
    paired = pd.DataFrame(
        {
            "decision_date": pd.date_range("2024-01-01", periods=5),
            "control_strategy": ["control"] * 5,
            "paired_net_excess": [0.04, 0.02, -0.01, 0.0, np.nan],
        }
    )
    row = summarize_paired_net_excess_inference(paired).iloc[0]

    assert row["n_decision_dates"] == 5
    assert row["n_evaluable_decision_dates"] == 4
    assert row["n_missing_decision_dates"] == 1
    assert row["bootstrap_iterations"] == 10_000
    assert row["bootstrap_confidence_level"] == pytest.approx(0.95)
    assert row["sign_nonzero_decision_dates"] == 3
    assert row["sign_positive_decision_dates"] == 2
    assert row["sign_negative_decision_dates"] == 1
    # Exact one-sided Binomial(3, 0.5) P[X >= 2] = 4 / 8.
    assert row["sign_test_pvalue_greater"] == pytest.approx(0.5)
    assert row["sign_test_alternative"] == "greater"


def test_paired_inference_handles_constant_singleton_zero_and_all_missing_groups() -> None:
    paired = pd.DataFrame(
        {
            "decision_date": [
                "2024-01-02",
                "2024-01-02",
                "2024-01-02",
                "2024-02-01",
            ],
            "control_strategy": ["positive", "zero", "missing", "missing"],
            "paired_net_excess": [0.01, 0.0, np.nan, np.nan],
        }
    )
    result = summarize_paired_net_excess_inference(
        paired, bootstrap_iterations=10, seed=1
    ).set_index("control_strategy")

    assert result.at["positive", "bootstrap_ci_lower"] == pytest.approx(0.01)
    assert result.at["positive", "bootstrap_ci_upper"] == pytest.approx(0.01)
    assert result.at["positive", "sign_test_pvalue_greater"] == pytest.approx(0.5)
    assert pd.isna(result.at["zero", "sign_test_pvalue_greater"])
    assert result.at["zero", "sign_nonzero_decision_dates"] == 0
    assert result.at["missing", "n_evaluable_decision_dates"] == 0
    assert pd.isna(result.at["missing", "mean_paired_net_excess"])
    assert pd.isna(result.at["missing", "bootstrap_ci_lower"])
    assert pd.isna(result.at["missing", "sign_test_pvalue_greater"])


def test_paired_inference_rejects_non_date_grain_and_invalid_parameters() -> None:
    paired = pd.DataFrame(
        {
            "decision_date": ["2024-01-02", "2024-01-02"],
            "control_strategy": ["control", "control"],
            "paired_net_excess": [0.01, 0.02],
        }
    )
    with pytest.raises(ValueError, match="unique by decision_date"):
        summarize_paired_net_excess_inference(paired, bootstrap_iterations=10)
    unique = paired.iloc[[0]]
    with pytest.raises(ValueError, match="positive integer"):
        summarize_paired_net_excess_inference(unique, bootstrap_iterations=0)
    with pytest.raises(ValueError, match="strictly between"):
        summarize_paired_net_excess_inference(unique, confidence_level=1.0)
    with pytest.raises(ValueError, match="non-negative integer"):
        summarize_paired_net_excess_inference(unique, seed=-1)

    empty = summarize_paired_net_excess_inference(paired.iloc[0:0])
    assert empty.empty
    assert "sign_test_pvalue_greater" in empty.columns
