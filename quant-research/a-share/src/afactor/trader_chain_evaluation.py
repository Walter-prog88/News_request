"""Pure return evaluation for the frozen trader-chain R1 experiment.

The public functions in this module accept in-memory data frames only.  They
never load a market-data file, a disclosed-trade file, a label, a holding, or
an outcome other than the strategy returns explicitly supplied by the runner.

The canonical runner input has one row per ``(decision_date, strategy)`` with
``gross_return``, ``slots_target`` and ``slots_filled``.  ``gross_return`` must
already be the equal-weight return across *all* target slots: unfilled slots
are zero-return cash and filled slots retain their fixed ``1 / slots_target``
weight.  The helper :func:`aggregate_equal_weight_slots` builds that canonical
table from filled-slot returns when useful.

Each decision period is treated as an independent open-to-expiry round trip.
Historical round-trip costs are 12 bp before 2023-08-28 and 7 bp from that date
onward.  Cost is charged once per period, scaled by
``invested_fraction = slots_filled / slots_target``; cash is not traded and
therefore pays no cost.  A double-cost series is retained as a preregistered
stress test.  The optional ``turnover`` input is audit-only and never changes
costs in this frozen evaluator.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np
import pandas as pd


HISTORICAL_COST_SWITCH_DATE = pd.Timestamp("2023-08-28")
DEFAULT_CONTROLS: tuple[str, str] = (
    "top3_chain_all_members",
    "top3_chain_random_4",
)
RETURN_BASES: tuple[tuple[str, str], ...] = (
    ("gross", "gross_return"),
    ("net_historical_cost", "net_return"),
    ("net_double_cost", "net_return_double_cost"),
)


@dataclass(frozen=True)
class ChainEvaluationConfig:
    """Column names, cost schedule and annualisation for strategy evaluation."""

    decision_date_col: str = "decision_date"
    strategy_col: str = "strategy"
    gross_return_col: str = "gross_return"
    slots_target_col: str = "slots_target"
    slots_filled_col: str = "slots_filled"
    turnover_col: str = "turnover"
    periods_per_year: int = 12
    cost_switch_date: object = HISTORICAL_COST_SWITCH_DATE
    early_round_trip_cost: float = 0.0012
    late_round_trip_cost: float = 0.0007
    double_cost_multiplier: float = 2.0

    def __post_init__(self) -> None:
        names = (
            self.decision_date_col,
            self.strategy_col,
            self.gross_return_col,
            self.slots_target_col,
            self.slots_filled_col,
            self.turnover_col,
        )
        if any(not isinstance(name, str) or not name.strip() for name in names):
            raise ValueError("configured column names must be non-empty strings")
        if (
            isinstance(self.periods_per_year, bool)
            or int(self.periods_per_year) != self.periods_per_year
            or self.periods_per_year <= 0
        ):
            raise ValueError("periods_per_year must be a positive integer")
        costs = (
            self.early_round_trip_cost,
            self.late_round_trip_cost,
            self.double_cost_multiplier,
        )
        if any(not np.isfinite(value) or value < 0 for value in costs):
            raise ValueError("costs and cost multiplier must be finite and non-negative")
        switch = pd.Timestamp(self.cost_switch_date)
        if pd.isna(switch):
            raise ValueError("cost_switch_date must be valid")
        if switch.tzinfo is not None:
            switch = switch.tz_localize(None)
        object.__setattr__(self, "cost_switch_date", switch.normalize())


@dataclass(frozen=True)
class R1GateConfig:
    """Frozen historical-PIT return-gate thresholds from the R1 protocol."""

    r1_strategy: str = "R1"
    controls: tuple[str, str] = DEFAULT_CONTROLS
    main_control: str = "top3_chain_all_members"
    first_year: int = 2021
    last_year: int = 2025
    minimum_decisions: int = 48
    minimum_positive_half_years: int = 7
    minimum_annualized_net_excess: float = 0.02
    minimum_information_ratio: float = 0.5
    maximum_mdd_degradation: float = 0.03
    maximum_contribution_share: float = 0.50

    def __post_init__(self) -> None:
        if not isinstance(self.r1_strategy, str) or not self.r1_strategy.strip():
            raise ValueError("r1_strategy must be a non-empty string")
        controls = tuple(str(value).strip() for value in self.controls)
        if len(controls) != 2 or any(not value for value in controls):
            raise ValueError("controls must contain exactly two non-empty strategies")
        if len(set(controls)) != 2:
            raise ValueError("controls must be distinct")
        if self.main_control not in controls:
            raise ValueError("main_control must be one of controls")
        if self.first_year > self.last_year:
            raise ValueError("first_year must not exceed last_year")
        if self.minimum_decisions <= 0 or self.minimum_positive_half_years < 0:
            raise ValueError("gate counts must be non-negative and decisions positive")
        expected_half_years = 2 * (self.last_year - self.first_year + 1)
        if self.minimum_positive_half_years > expected_half_years:
            raise ValueError("minimum_positive_half_years exceeds the evaluation window")
        finite_thresholds = (
            self.minimum_annualized_net_excess,
            self.minimum_information_ratio,
            self.maximum_mdd_degradation,
            self.maximum_contribution_share,
        )
        if any(not np.isfinite(value) for value in finite_thresholds):
            raise ValueError("gate thresholds must be finite")
        if self.maximum_mdd_degradation < 0:
            raise ValueError("maximum_mdd_degradation must be non-negative")
        if not 0 <= self.maximum_contribution_share <= 1:
            raise ValueError("maximum_contribution_share must be in [0, 1]")
        object.__setattr__(self, "controls", controls)


@dataclass(frozen=True)
class GateEvaluation:
    """Auditable fixed-gate output.

    ``all_protocol_checks_pass`` is deliberately strict: it is false whenever
    any required check is missing, including chain-contribution concentration.
    ``status`` is ``not_evaluated`` only when no evaluated check failed but at
    least one required check could not be evaluated.
    """

    checks: pd.DataFrame
    status: str
    all_evaluable_checks_pass: bool
    all_required_checks_evaluated: bool
    daily_computable_checks_pass: bool
    all_protocol_checks_pass: bool


@dataclass(frozen=True)
class ChainEvaluationResult:
    """All recomputable in-memory outputs of :func:`evaluate_chain_returns`."""

    daily: pd.DataFrame
    strategy_summary: pd.DataFrame
    paired_daily: pd.DataFrame
    paired_summary: pd.DataFrame
    half_year_summary: pd.DataFrame
    gate: GateEvaluation


def _normalised_dates(values: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce", utc=True)
    return parsed.dt.tz_convert(None).dt.normalize()


def _numeric(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)


def equal_weight_return_with_cash(
    filled_slot_returns: Sequence[float], target_slots: int
) -> float:
    """Return the equal-weight portfolio return with absent slots held in cash.

    ``filled_slot_returns`` must contain only actually filled, evaluable slots.
    Missing future outcomes must not be passed as ``NaN`` and silently turned
    into cash.  Cash is represented only by ``target_slots`` exceeding the
    number of supplied filled-slot returns.
    """

    if isinstance(target_slots, bool) or int(target_slots) != target_slots:
        raise ValueError("target_slots must be a positive integer")
    target = int(target_slots)
    if target <= 0:
        raise ValueError("target_slots must be a positive integer")
    values = np.asarray(list(filled_slot_returns), dtype=float)
    if values.ndim != 1:
        raise ValueError("filled_slot_returns must be one-dimensional")
    if len(values) > target:
        raise ValueError("filled slots cannot exceed target_slots")
    if not np.isfinite(values).all():
        raise ValueError("filled_slot_returns must be finite; missing outcomes are not cash")
    if (values < -1.0).any():
        raise ValueError("a simple return cannot be below -100%")
    return float(values.sum() / target)


def aggregate_equal_weight_slots(
    slots: pd.DataFrame,
    *,
    decision_date_col: str = "decision_date",
    strategy_col: str = "strategy",
    slot_return_col: str = "asset_return",
    slots_target_col: str = "slots_target",
) -> pd.DataFrame:
    """Aggregate filled-slot rows to the canonical daily strategy grain.

    Each input row is one filled slot.  All rows in a date/strategy group must
    agree on ``slots_target``.  A day with no filled slots cannot be represented
    by an empty group; the runner should supply its canonical zero-return row
    directly with ``slots_filled=0``.
    """

    required = {
        decision_date_col,
        strategy_col,
        slot_return_col,
        slots_target_col,
    }
    missing = sorted(required.difference(slots.columns))
    if missing:
        raise ValueError(f"slot table is missing required columns: {missing}")
    if slots.empty:
        return pd.DataFrame(
            columns=[
                "decision_date",
                "strategy",
                "gross_return",
                "slots_target",
                "slots_filled",
            ]
        )

    frame = slots.loc[:, list(required)].copy()
    frame[decision_date_col] = _normalised_dates(frame[decision_date_col])
    frame[strategy_col] = frame[strategy_col].astype("string").str.strip()
    frame[slot_return_col] = _numeric(frame[slot_return_col])
    frame[slots_target_col] = _numeric(frame[slots_target_col])
    if frame[[decision_date_col, strategy_col]].isna().any(axis=None):
        raise ValueError("slot table contains an invalid date or strategy")
    if frame[strategy_col].eq("").any():
        raise ValueError("strategy values must be non-empty")
    if frame[slot_return_col].isna().any():
        raise ValueError("filled slot returns must be finite; missing outcomes are not cash")
    if frame[slot_return_col].lt(-1.0).any():
        raise ValueError("a simple return cannot be below -100%")
    target = frame[slots_target_col]
    if target.isna().any() or target.le(0).any() or target.ne(target.round()).any():
        raise ValueError("slots_target must contain positive integers")

    rows: list[dict[str, object]] = []
    keys = [decision_date_col, strategy_col]
    for (date, strategy), group in frame.groupby(keys, sort=True, dropna=False):
        targets = group[slots_target_col].astype(int).unique()
        if len(targets) != 1:
            raise ValueError("slots_target must be constant within date/strategy")
        filled = len(group)
        if filled > int(targets[0]):
            raise ValueError("filled slots cannot exceed target slots")
        rows.append(
            {
                "decision_date": date,
                "strategy": str(strategy),
                "gross_return": equal_weight_return_with_cash(
                    group[slot_return_col].to_numpy(float), int(targets[0])
                ),
                "slots_target": int(targets[0]),
                "slots_filled": filled,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["decision_date", "strategy"], kind="mergesort"
    ).reset_index(drop=True)


def prepare_daily_strategy_returns(
    daily: pd.DataFrame,
    config: ChainEvaluationConfig = ChainEvaluationConfig(),
) -> pd.DataFrame:
    """Validate and enrich one-row-per-date/strategy returns without I/O.

    Extra columns are preserved but never inspected.  In particular, no label,
    transaction, disclosed holding or realised-PnL field can influence any
    calculation.  ``turnover`` is retained if supplied, but frozen costs are
    applied once per independent round trip and do not use it.
    """

    required = {
        config.decision_date_col,
        config.strategy_col,
        config.gross_return_col,
        config.slots_target_col,
        config.slots_filled_col,
    }
    missing = sorted(required.difference(daily.columns))
    if missing:
        raise ValueError(f"daily strategy table is missing required columns: {missing}")
    result = daily.copy()
    result[config.decision_date_col] = _normalised_dates(
        result[config.decision_date_col]
    )
    result[config.strategy_col] = (
        result[config.strategy_col].astype("string").str.strip()
    )
    result[config.gross_return_col] = _numeric(result[config.gross_return_col])
    result[config.slots_target_col] = _numeric(result[config.slots_target_col])
    result[config.slots_filled_col] = _numeric(result[config.slots_filled_col])

    if result[[config.decision_date_col, config.strategy_col]].isna().any(axis=None):
        raise ValueError("daily strategy table contains an invalid date or strategy")
    if result[config.strategy_col].eq("").any():
        raise ValueError("strategy values must be non-empty")
    key = [config.decision_date_col, config.strategy_col]
    if result.duplicated(key, keep=False).any():
        raise ValueError("daily strategy table must be unique by date and strategy")

    target = result[config.slots_target_col]
    filled = result[config.slots_filled_col]
    if target.isna().any() or target.le(0).any() or target.ne(target.round()).any():
        raise ValueError("slots_target must contain positive integers")
    if filled.isna().any() or filled.lt(0).any() or filled.ne(filled.round()).any():
        raise ValueError("slots_filled must contain non-negative integers")
    if filled.gt(target).any():
        raise ValueError("slots_filled cannot exceed slots_target")
    gross = result[config.gross_return_col]
    if gross.dropna().lt(-1.0).any():
        raise ValueError("a simple return cannot be below -100%")
    impossible_cash = filled.eq(0) & gross.notna() & gross.abs().gt(1e-12)
    if impossible_cash.any():
        raise ValueError("a zero-filled portfolio must have zero gross return")

    result[config.slots_target_col] = target.astype("int64")
    result[config.slots_filled_col] = filled.astype("int64")
    result["invested_fraction"] = filled / target
    result["cash_fraction"] = 1.0 - result["invested_fraction"]
    before_switch = result[config.decision_date_col].lt(config.cost_switch_date)
    result["round_trip_cost_rate"] = np.where(
        before_switch,
        config.early_round_trip_cost,
        config.late_round_trip_cost,
    )
    result["applied_round_trip_cost"] = (
        result["round_trip_cost_rate"] * result["invested_fraction"]
    )
    result["applied_double_cost"] = (
        config.double_cost_multiplier * result["applied_round_trip_cost"]
    )
    result["net_return"] = gross - result["applied_round_trip_cost"]
    result["net_return_double_cost"] = gross - result["applied_double_cost"]
    result["return_evaluable"] = gross.notna()
    result["year"] = result[config.decision_date_col].dt.year.astype("int64")
    result["half_year"] = (
        result["year"].astype(str)
        + "H"
        + np.where(result[config.decision_date_col].dt.month.le(6), "1", "2")
    )
    result["gross_return_cash_slot_inclusive"] = True
    result["cost_scaled_by_invested_fraction"] = True
    result["turnover_used_for_cost"] = False
    result["cost_application_rule"] = (
        "one independent round trip per decision period; cost rate x "
        "slots_filled/slots_target; cash pays zero"
    )
    return result.sort_values(key, kind="mergesort").reset_index(drop=True)


def _maximum_drawdown(values: pd.Series) -> float:
    returns = _numeric(values).dropna().to_numpy(float)
    if not len(returns):
        return np.nan
    wealth = np.concatenate(([1.0], np.cumprod(1.0 + returns)))
    peaks = np.maximum.accumulate(wealth)
    with np.errstate(divide="ignore", invalid="ignore"):
        drawdown = wealth / peaks - 1.0
    finite = drawdown[np.isfinite(drawdown)]
    return float(finite.min()) if len(finite) else np.nan


def _information_ratio(values: pd.Series, periods_per_year: int) -> float:
    excess = _numeric(values).dropna().to_numpy(float)
    if len(excess) < 2:
        return np.nan
    standard_deviation = float(np.std(excess, ddof=1))
    mean = float(np.mean(excess))
    if standard_deviation <= 1e-15:
        if mean > 0:
            return np.inf
        if mean < 0:
            return -np.inf
        return np.nan
    return float(mean / standard_deviation * np.sqrt(periods_per_year))


def _return_metrics(values: pd.Series, periods_per_year: int) -> dict[str, float | int]:
    returns = _numeric(values).dropna().to_numpy(float)
    n = len(returns)
    if not n:
        return {
            "n_periods": 0,
            "compound_return": np.nan,
            "annualized_return": np.nan,
            "mean_period_return": np.nan,
            "annualized_volatility": np.nan,
            "information_ratio": np.nan,
            "max_drawdown": np.nan,
            "win_rate": np.nan,
        }
    wealth = float(np.prod(1.0 + returns))
    compound = wealth - 1.0
    annualized = -1.0 if wealth == 0 else wealth ** (periods_per_year / n) - 1.0
    volatility = (
        float(np.std(returns, ddof=1) * np.sqrt(periods_per_year))
        if n >= 2
        else np.nan
    )
    return {
        "n_periods": n,
        "compound_return": compound,
        "annualized_return": float(annualized),
        "mean_period_return": float(np.mean(returns)),
        "annualized_volatility": volatility,
        "information_ratio": _information_ratio(pd.Series(returns), periods_per_year),
        "max_drawdown": _maximum_drawdown(pd.Series(returns)),
        "win_rate": float(np.mean(returns > 0.0)),
    }


def summarize_strategy_returns(
    daily: pd.DataFrame,
    config: ChainEvaluationConfig = ChainEvaluationConfig(),
) -> pd.DataFrame:
    """Report compound/risk metrics for gross, historical-cost and double-cost returns."""

    prepared = prepare_daily_strategy_returns(daily, config)
    rows: list[dict[str, object]] = []
    for strategy, group in prepared.groupby(config.strategy_col, sort=True):
        evaluable_dates = group.loc[group["return_evaluable"], config.decision_date_col]
        common = {
            "strategy": str(strategy),
            "start_date": evaluable_dates.min() if len(evaluable_dates) else pd.NaT,
            "end_date": evaluable_dates.max() if len(evaluable_dates) else pd.NaT,
            "calendar_rows": int(len(group)),
            "mean_slots_target": float(group[config.slots_target_col].mean()),
            "mean_slots_filled": float(group[config.slots_filled_col].mean()),
            "mean_invested_fraction": float(group["invested_fraction"].mean()),
            "mean_cash_fraction": float(group["cash_fraction"].mean()),
        }
        for basis, canonical_column in RETURN_BASES:
            column = (
                config.gross_return_col
                if canonical_column == "gross_return"
                else canonical_column
            )
            rows.append(
                {
                    **common,
                    "return_basis": basis,
                    **_return_metrics(group[column], config.periods_per_year),
                }
            )
    columns = [
        "strategy",
        "return_basis",
        "start_date",
        "end_date",
        "calendar_rows",
        "n_periods",
        "compound_return",
        "annualized_return",
        "mean_period_return",
        "annualized_volatility",
        "information_ratio",
        "max_drawdown",
        "win_rate",
        "mean_slots_target",
        "mean_slots_filled",
        "mean_invested_fraction",
        "mean_cash_fraction",
    ]
    return pd.DataFrame(rows, columns=columns)


def build_r1_paired_excess(
    daily: pd.DataFrame,
    config: ChainEvaluationConfig = ChainEvaluationConfig(),
    gate_config: R1GateConfig = R1GateConfig(),
) -> pd.DataFrame:
    """Inner-pair R1 with each frozen control at the decision-date grain."""

    prepared = prepare_daily_strategy_returns(daily, config)
    keep = [
        config.decision_date_col,
        config.strategy_col,
        config.gross_return_col,
        "net_return",
        "net_return_double_cost",
        "invested_fraction",
        "applied_round_trip_cost",
        "applied_double_cost",
        "half_year",
    ]
    r1 = prepared.loc[
        prepared[config.strategy_col].eq(gate_config.r1_strategy), keep
    ].copy()
    rows: list[pd.DataFrame] = []
    for control in gate_config.controls:
        comparison = prepared.loc[prepared[config.strategy_col].eq(control), keep].copy()
        if r1.empty or comparison.empty:
            continue
        left_names = {
            column: f"r1_{column}"
            for column in keep
            if column not in {config.decision_date_col, config.strategy_col, "half_year"}
        }
        right_names = {
            column: f"control_{column}"
            for column in keep
            if column not in {config.decision_date_col, config.strategy_col, "half_year"}
        }
        # Paired outputs keep a stable public gross-return name even when a
        # runner customises its input column name through the config.
        left_names[config.gross_return_col] = "r1_gross_return"
        right_names[config.gross_return_col] = "control_gross_return"
        left = r1.drop(columns=config.strategy_col).rename(columns=left_names)
        right = comparison.drop(columns=[config.strategy_col, "half_year"]).rename(
            columns=right_names
        )
        paired = left.merge(
            right,
            on=config.decision_date_col,
            how="inner",
            validate="one_to_one",
            sort=True,
        )
        paired.insert(1, "control_strategy", control)
        paired["paired_gross_excess"] = (
            paired["r1_gross_return"] - paired["control_gross_return"]
        )
        paired["paired_net_excess"] = (
            paired["r1_net_return"] - paired["control_net_return"]
        )
        paired["paired_double_cost_excess"] = (
            paired["r1_net_return_double_cost"]
            - paired["control_net_return_double_cost"]
        )
        if config.decision_date_col != "decision_date":
            paired = paired.rename(columns={config.decision_date_col: "decision_date"})
        rows.append(paired)
    if not rows:
        return pd.DataFrame(
            columns=[
                "decision_date",
                "control_strategy",
                "half_year",
                "r1_gross_return",
                "control_gross_return",
                "r1_net_return",
                "control_net_return",
                "r1_net_return_double_cost",
                "control_net_return_double_cost",
                "paired_gross_excess",
                "paired_net_excess",
                "paired_double_cost_excess",
            ]
        )
    return pd.concat(rows, ignore_index=True).sort_values(
        ["decision_date", "control_strategy"], kind="mergesort"
    ).reset_index(drop=True)


def _relative_compound(r1: pd.Series, control: pd.Series) -> float:
    left = _numeric(r1)
    right = _numeric(control)
    valid = left.notna() & right.notna()
    if not valid.any():
        return np.nan
    left_wealth = float(np.prod(1.0 + left.loc[valid].to_numpy(float)))
    right_wealth = float(np.prod(1.0 + right.loc[valid].to_numpy(float)))
    if right_wealth == 0:
        return np.nan
    return left_wealth / right_wealth - 1.0


def summarize_r1_pairs(
    paired_daily: pd.DataFrame,
    *,
    periods_per_year: int = 12,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize paired excess overall and in calendar half-year intervals."""

    required = {
        "decision_date",
        "control_strategy",
        "half_year",
        "r1_net_return",
        "control_net_return",
        "r1_net_return_double_cost",
        "control_net_return_double_cost",
        "paired_net_excess",
    }
    missing = sorted(required.difference(paired_daily.columns))
    if missing:
        raise ValueError(f"paired daily table is missing required columns: {missing}")
    summary_rows: list[dict[str, object]] = []
    half_rows: list[dict[str, object]] = []
    for control, group in paired_daily.groupby("control_strategy", sort=True):
        group = group.sort_values("decision_date", kind="mergesort")
        valid = group["r1_net_return"].notna() & group["control_net_return"].notna()
        sample = group.loc[valid]
        r1_metrics = _return_metrics(sample["r1_net_return"], periods_per_year)
        control_metrics = _return_metrics(sample["control_net_return"], periods_per_year)
        double_relative = _relative_compound(
            sample["r1_net_return_double_cost"],
            sample["control_net_return_double_cost"],
        )
        gross_relative = (
            _relative_compound(
                sample["r1_gross_return"], sample["control_gross_return"]
            )
            if {"r1_gross_return", "control_gross_return"}.issubset(sample.columns)
            else np.nan
        )
        net_relative = _relative_compound(
            sample["r1_net_return"], sample["control_net_return"]
        )
        annualized_relative = (
            (1.0 + net_relative) ** (periods_per_year / len(sample)) - 1.0
            if len(sample) and pd.notna(net_relative) and net_relative > -1.0
            else (-1.0 if net_relative == -1.0 and len(sample) else np.nan)
        )
        summary_rows.append(
            {
                "control_strategy": str(control),
                "n_periods": int(len(sample)),
                "start_date": sample["decision_date"].min() if len(sample) else pd.NaT,
                "end_date": sample["decision_date"].max() if len(sample) else pd.NaT,
                "years_covered": "|".join(
                    map(str, sorted(pd.DatetimeIndex(sample["decision_date"]).year.unique()))
                ),
                "gross_relative_compound_excess": gross_relative,
                "net_relative_compound_excess": net_relative,
                "double_cost_relative_compound_excess": double_relative,
                "r1_net_annualized_return": r1_metrics["annualized_return"],
                "control_net_annualized_return": control_metrics["annualized_return"],
                "annualized_net_excess": (
                    r1_metrics["annualized_return"]
                    - control_metrics["annualized_return"]
                ),
                "relative_annualized_net_excess": annualized_relative,
                "paired_net_information_ratio": _information_ratio(
                    sample["paired_net_excess"], periods_per_year
                ),
                "mean_paired_net_excess": float(sample["paired_net_excess"].mean())
                if len(sample)
                else np.nan,
                "paired_net_win_rate": float(sample["paired_net_excess"].gt(0).mean())
                if len(sample)
                else np.nan,
                "r1_net_max_drawdown": r1_metrics["max_drawdown"],
                "control_net_max_drawdown": control_metrics["max_drawdown"],
                "max_drawdown_delta": (
                    r1_metrics["max_drawdown"] - control_metrics["max_drawdown"]
                ),
            }
        )
        for half_year, half in sample.groupby("half_year", sort=True):
            half_rows.append(
                {
                    "control_strategy": str(control),
                    "half_year": str(half_year),
                    "n_periods": int(len(half)),
                    "net_relative_compound_excess": _relative_compound(
                        half["r1_net_return"], half["control_net_return"]
                    ),
                    "double_cost_relative_compound_excess": _relative_compound(
                        half["r1_net_return_double_cost"],
                        half["control_net_return_double_cost"],
                    ),
                    "sum_paired_net_excess": float(half["paired_net_excess"].sum()),
                    "mean_paired_net_excess": float(half["paired_net_excess"].mean()),
                    "paired_net_win_rate": float(half["paired_net_excess"].gt(0).mean()),
                }
            )
    summary_columns = [
        "control_strategy",
        "n_periods",
        "start_date",
        "end_date",
        "years_covered",
        "gross_relative_compound_excess",
        "net_relative_compound_excess",
        "double_cost_relative_compound_excess",
        "r1_net_annualized_return",
        "control_net_annualized_return",
        "annualized_net_excess",
        "relative_annualized_net_excess",
        "paired_net_information_ratio",
        "mean_paired_net_excess",
        "paired_net_win_rate",
        "r1_net_max_drawdown",
        "control_net_max_drawdown",
        "max_drawdown_delta",
    ]
    half_columns = [
        "control_strategy",
        "half_year",
        "n_periods",
        "net_relative_compound_excess",
        "double_cost_relative_compound_excess",
        "sum_paired_net_excess",
        "mean_paired_net_excess",
        "paired_net_win_rate",
    ]
    return (
        pd.DataFrame(summary_rows, columns=summary_columns),
        pd.DataFrame(half_rows, columns=half_columns),
    )


def _exact_sign_test_pvalue_greater(positive: int, nonzero: int) -> float:
    """Exact ``P[Binomial(nonzero, 0.5) >= positive]`` without SciPy."""

    if nonzero <= 0:
        return np.nan
    if positive < 0 or positive > nonzero:
        raise ValueError("positive sign count must be in [0, nonzero]")

    denominator = 1 << nonzero
    # Sum the shorter tail.  The integer numerator keeps this an exact
    # binomial calculation before the final conversion to a float.
    if positive <= nonzero // 2:
        lower_count = sum(math.comb(nonzero, wins) for wins in range(positive))
        probability = 1.0 - lower_count / denominator
    else:
        upper_count = sum(
            math.comb(nonzero, wins) for wins in range(positive, nonzero + 1)
        )
        probability = upper_count / denominator
    return float(min(1.0, max(0.0, probability)))


def summarize_paired_net_excess_inference(
    paired_daily: pd.DataFrame,
    *,
    bootstrap_iterations: int = 10_000,
    confidence_level: float = 0.95,
    seed: int = 20260831,
) -> pd.DataFrame:
    """Infer mean paired net excess separately for each control strategy.

    The sampling unit is one unique ``decision_date`` within a control.  The
    confidence interval is the equal-tailed percentile interval from a
    non-parametric bootstrap of the mean ``paired_net_excess``.  The sign test
    excludes exact zeroes and reports the exact one-sided ``greater`` p-value
    under a 50% positive-sign null.  Missing excess outcomes are disclosed in
    the counts and excluded from both inferential calculations.

    This function is pure: it copies its required columns, uses only the
    supplied data and fixed seed, and performs no file or global-state I/O.
    """

    columns = [
        "control_strategy",
        "n_decision_dates",
        "n_evaluable_decision_dates",
        "n_missing_decision_dates",
        "mean_paired_net_excess",
        "bootstrap_ci_lower",
        "bootstrap_ci_upper",
        "bootstrap_confidence_level",
        "bootstrap_iterations",
        "bootstrap_seed",
        "bootstrap_method",
        "sign_nonzero_decision_dates",
        "sign_positive_decision_dates",
        "sign_negative_decision_dates",
        "sign_test_pvalue_greater",
        "sign_test_alternative",
    ]
    required = {"decision_date", "control_strategy", "paired_net_excess"}
    missing = sorted(required.difference(paired_daily.columns))
    if missing:
        raise ValueError(f"paired daily table is missing required columns: {missing}")
    if (
        isinstance(bootstrap_iterations, bool)
        or int(bootstrap_iterations) != bootstrap_iterations
        or bootstrap_iterations <= 0
    ):
        raise ValueError("bootstrap_iterations must be a positive integer")
    if (
        not np.isfinite(confidence_level)
        or confidence_level <= 0.0
        or confidence_level >= 1.0
    ):
        raise ValueError("confidence_level must be finite and strictly between 0 and 1")
    if isinstance(seed, bool) or int(seed) != seed or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    iterations = int(bootstrap_iterations)
    fixed_seed = int(seed)

    frame = paired_daily.loc[:, sorted(required)].copy()
    if frame.empty:
        return pd.DataFrame(columns=columns)
    frame["decision_date"] = _normalised_dates(frame["decision_date"])
    frame["control_strategy"] = (
        frame["control_strategy"].astype("string").str.strip()
    )
    frame["paired_net_excess"] = _numeric(frame["paired_net_excess"])
    if frame[["decision_date", "control_strategy"]].isna().any(axis=None):
        raise ValueError("paired daily table contains an invalid date or control strategy")
    if frame["control_strategy"].eq("").any():
        raise ValueError("control_strategy values must be non-empty")
    if frame.duplicated(["decision_date", "control_strategy"], keep=False).any():
        raise ValueError(
            "paired daily table must be unique by decision_date and control_strategy"
        )
    frame = frame.sort_values(
        ["control_strategy", "decision_date"], kind="mergesort"
    ).reset_index(drop=True)

    alpha = 1.0 - float(confidence_level)
    quantiles = [alpha / 2.0, 1.0 - alpha / 2.0]
    rows: list[dict[str, object]] = []
    for control_index, (control, group) in enumerate(
        frame.groupby("control_strategy", sort=True)
    ):
        values = group["paired_net_excess"].dropna().to_numpy(float)
        n_dates = int(len(group))
        n_evaluable = int(len(values))
        mean_excess = float(np.mean(values)) if n_evaluable else np.nan
        ci_lower = np.nan
        ci_upper = np.nan
        if n_evaluable:
            rng = np.random.default_rng(
                np.random.SeedSequence([fixed_seed, control_index])
            )
            bootstrap_means = np.empty(iterations, dtype=float)
            # Bound temporary memory while preserving decision-date resampling.
            batch_size = min(2_048, iterations)
            for start in range(0, iterations, batch_size):
                stop = min(start + batch_size, iterations)
                sample_indices = rng.integers(
                    0,
                    n_evaluable,
                    size=(stop - start, n_evaluable),
                )
                bootstrap_means[start:stop] = values[sample_indices].mean(axis=1)
            ci_lower, ci_upper = np.quantile(
                bootstrap_means, quantiles, method="linear"
            )

        nonzero = values[values != 0.0]
        n_nonzero = int(len(nonzero))
        n_positive = int(np.count_nonzero(nonzero > 0.0))
        n_negative = int(np.count_nonzero(nonzero < 0.0))
        rows.append(
            {
                "control_strategy": str(control),
                "n_decision_dates": n_dates,
                "n_evaluable_decision_dates": n_evaluable,
                "n_missing_decision_dates": n_dates - n_evaluable,
                "mean_paired_net_excess": mean_excess,
                "bootstrap_ci_lower": float(ci_lower),
                "bootstrap_ci_upper": float(ci_upper),
                "bootstrap_confidence_level": float(confidence_level),
                "bootstrap_iterations": iterations,
                "bootstrap_seed": fixed_seed,
                "bootstrap_method": "decision_date_percentile",
                "sign_nonzero_decision_dates": n_nonzero,
                "sign_positive_decision_dates": n_positive,
                "sign_negative_decision_dates": n_negative,
                "sign_test_pvalue_greater": _exact_sign_test_pvalue_greater(
                    n_positive, n_nonzero
                ),
                "sign_test_alternative": "greater",
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _expected_half_years(config: R1GateConfig) -> list[str]:
    return [
        f"{year}H{half}"
        for year in range(config.first_year, config.last_year + 1)
        for half in (1, 2)
    ]


def _finite_scalar(value: object) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if np.isfinite(numeric) else None


def evaluate_r1_gate(
    paired_daily: pd.DataFrame,
    paired_summary: pd.DataFrame,
    half_year_summary: pd.DataFrame,
    *,
    chain_contribution_share: float | None = None,
    gate_config: R1GateConfig = R1GateConfig(),
) -> GateEvaluation:
    """Apply every historical-PIT return gate that can be evaluated here.

    ``chain_contribution_share`` is the runner's maximum single-chain share of
    total net excess versus the main control.  It is intentionally a separate
    scalar derived from the runner's position/chain attribution detail.  When
    omitted, that preregistered check is ``not_evaluated`` and the strict
    ``all_protocol_checks_pass`` flag remains false.
    """

    required_pair_columns = {"decision_date", "control_strategy", "paired_net_excess"}
    if not required_pair_columns.issubset(paired_daily.columns):
        raise ValueError(
            "paired_daily is missing required columns: "
            f"{sorted(required_pair_columns - set(paired_daily.columns))}"
        )
    summary_lookup = (
        paired_summary.set_index("control_strategy", drop=False)
        if "control_strategy" in paired_summary.columns
        else pd.DataFrame()
    )
    half = half_year_summary.copy()
    expected_years = set(range(gate_config.first_year, gate_config.last_year + 1))
    expected_halves = _expected_half_years(gate_config)

    controls_present = set(summary_lookup.index.astype(str)) if len(summary_lookup) else set()
    both_controls = all(control in controls_present for control in gate_config.controls)
    paired_by_control = {
        control: paired_daily.loc[
            paired_daily["control_strategy"].astype(str).eq(control)
            & paired_daily["paired_net_excess"].notna()
        ]
        for control in gate_config.controls
    }
    common_dates: set[pd.Timestamp] | None = None
    for control in gate_config.controls:
        dates = set(pd.to_datetime(paired_by_control[control]["decision_date"]).dt.normalize())
        common_dates = dates if common_dates is None else common_dates.intersection(dates)
    common_dates = common_dates or set()
    common_years = {date.year for date in common_dates}

    if both_controls and not half.empty:
        pivot = half.pivot(
            index="half_year",
            columns="control_strategy",
            values="net_relative_compound_excess",
        ).reindex(index=expected_halves, columns=list(gate_config.controls))
        half_complete = pivot[list(gate_config.controls)].notna().all(axis=1)
        half_positive = pivot[list(gate_config.controls)].gt(0).all(axis=1) & half_complete
        observed_half_years = int(half_complete.sum())
        positive_half_years = int(half_positive.sum())
    else:
        observed_half_years = 0
        positive_half_years = 0

    main_rows = paired_by_control[gate_config.main_control]
    total_main_excess = float(main_rows["paired_net_excess"].sum()) if len(main_rows) else np.nan
    if len(main_rows) and total_main_excess > 0:
        half_contribution = (
            main_rows.assign(
                _half=pd.to_datetime(main_rows["decision_date"]).dt.year.astype(str)
                + "H"
                + np.where(pd.to_datetime(main_rows["decision_date"]).dt.month.le(6), "1", "2")
            )
            .groupby("_half", sort=True)["paired_net_excess"]
            .sum()
        )
        max_half_contribution_share = float(half_contribution.max() / total_main_excess)
    else:
        max_half_contribution_share = np.nan

    checks: list[dict[str, object]] = []

    def add_check(
        check_id: str,
        description: str,
        observed: object,
        threshold: str,
        evaluated: bool,
        passed: bool = False,
        *,
        daily_computable: bool = True,
    ) -> None:
        checks.append(
            {
                "check_id": check_id,
                "description": description,
                "observed": observed,
                "threshold": threshold,
                "evaluated": bool(evaluated),
                "passed": bool(passed) if evaluated else pd.NA,
                "status": "pass" if evaluated and passed else "fail" if evaluated else "not_evaluated",
                "daily_computable": bool(daily_computable),
            }
        )

    min_common_periods = min(
        (len(paired_by_control[control]) for control in gate_config.controls),
        default=0,
    )
    add_check(
        "minimum_decisions",
        "paired monthly decisions are sufficient for both controls",
        min_common_periods,
        f">= {gate_config.minimum_decisions}",
        both_controls,
        min_common_periods >= gate_config.minimum_decisions,
    )
    add_check(
        "covers_each_year",
        "common paired dates cover every frozen evaluation year",
        "|".join(map(str, sorted(common_years))),
        f"contains {gate_config.first_year}..{gate_config.last_year}",
        both_controls and bool(common_dates),
        expected_years.issubset(common_years),
    )
    add_check(
        "covers_all_half_years",
        "both controls have an evaluable result in all ten half-years",
        observed_half_years,
        f"= {len(expected_halves)}",
        both_controls,
        observed_half_years == len(expected_halves),
    )

    for control in gate_config.controls:
        row = summary_lookup.loc[control] if control in controls_present else None
        net_relative = (
            _finite_scalar(row["net_relative_compound_excess"]) if row is not None else None
        )
        add_check(
            f"net_excess_positive__{control}",
            f"R1 compound net excess versus {control} is positive",
            net_relative,
            "> 0",
            net_relative is not None,
            bool(net_relative is not None and net_relative > 0),
        )

    add_check(
        "positive_half_years_both_controls",
        "half-years with positive compound net excess versus both controls",
        positive_half_years,
        f">= {gate_config.minimum_positive_half_years} of {len(expected_halves)}",
        both_controls and observed_half_years == len(expected_halves),
        positive_half_years >= gate_config.minimum_positive_half_years,
    )

    main_row = (
        summary_lookup.loc[gate_config.main_control]
        if gate_config.main_control in controls_present
        else None
    )
    annualized_excess = (
        _finite_scalar(main_row["annualized_net_excess"]) if main_row is not None else None
    )
    information_ratio = (
        float(main_row["paired_net_information_ratio"])
        if main_row is not None
        and pd.notna(main_row["paired_net_information_ratio"])
        else None
    )
    mdd_delta = (
        _finite_scalar(main_row["max_drawdown_delta"]) if main_row is not None else None
    )
    add_check(
        "main_control_annualized_net_excess",
        "annualized R1 net return minus main-control annualized net return",
        annualized_excess,
        f">= {gate_config.minimum_annualized_net_excess}",
        annualized_excess is not None,
        bool(
            annualized_excess is not None
            and annualized_excess >= gate_config.minimum_annualized_net_excess
        ),
    )
    add_check(
        "main_control_information_ratio",
        "annualized information ratio of paired net excess",
        information_ratio,
        f">= {gate_config.minimum_information_ratio}",
        information_ratio is not None and not np.isnan(information_ratio),
        bool(
            information_ratio is not None
            and information_ratio >= gate_config.minimum_information_ratio
        ),
    )
    add_check(
        "main_control_mdd_degradation",
        "R1 max drawdown minus main-control max drawdown",
        mdd_delta,
        f">= -{gate_config.maximum_mdd_degradation}",
        mdd_delta is not None,
        bool(
            mdd_delta is not None
            and mdd_delta >= -gate_config.maximum_mdd_degradation
        ),
    )

    for control in gate_config.controls:
        row = summary_lookup.loc[control] if control in controls_present else None
        double_relative = (
            _finite_scalar(row["double_cost_relative_compound_excess"])
            if row is not None
            else None
        )
        add_check(
            f"double_cost_excess_positive__{control}",
            f"R1 compound excess versus {control} remains positive at double cost",
            double_relative,
            "> 0",
            double_relative is not None,
            bool(double_relative is not None and double_relative > 0),
        )

    half_share = _finite_scalar(max_half_contribution_share)
    add_check(
        "half_year_contribution_concentration",
        "largest half-year share of additive net excess versus main control",
        half_share,
        f"<= {gate_config.maximum_contribution_share}",
        half_share is not None,
        bool(half_share is not None and half_share <= gate_config.maximum_contribution_share),
    )
    if (
        chain_contribution_share is not None
        and _finite_scalar(chain_contribution_share) is None
    ):
        raise ValueError("chain_contribution_share must be finite when supplied")
    chain_share = _finite_scalar(chain_contribution_share)
    if chain_share is not None and chain_share < 0:
        raise ValueError("chain_contribution_share must be non-negative")
    add_check(
        "chain_contribution_concentration",
        "largest single-chain share of additive net excess versus main control",
        chain_share,
        f"<= {gate_config.maximum_contribution_share}",
        chain_share is not None,
        bool(chain_share is not None and chain_share <= gate_config.maximum_contribution_share),
        daily_computable=False,
    )

    check_frame = pd.DataFrame(checks)
    evaluated = check_frame["evaluated"].astype(bool)
    passed_evaluated = check_frame.loc[evaluated, "passed"].astype(bool)
    all_evaluable_pass = bool(len(passed_evaluated) and passed_evaluated.all())
    all_required_evaluated = bool(evaluated.all())
    daily_checks = check_frame.loc[check_frame["daily_computable"].astype(bool)]
    daily_computable_pass = bool(
        len(daily_checks)
        and daily_checks["evaluated"].astype(bool).all()
        and daily_checks["passed"].astype(bool).all()
    )
    all_protocol_pass = bool(
        all_required_evaluated
        and check_frame["passed"].astype(bool).all()
    )
    if evaluated.any() and (~check_frame.loc[evaluated, "passed"].astype(bool)).any():
        status = "fail"
    elif all_required_evaluated:
        status = "pass"
    else:
        status = "not_evaluated"
    return GateEvaluation(
        checks=check_frame,
        status=status,
        all_evaluable_checks_pass=all_evaluable_pass,
        all_required_checks_evaluated=all_required_evaluated,
        daily_computable_checks_pass=daily_computable_pass,
        all_protocol_checks_pass=all_protocol_pass,
    )


def evaluate_chain_returns(
    daily: pd.DataFrame,
    *,
    config: ChainEvaluationConfig = ChainEvaluationConfig(),
    gate_config: R1GateConfig = R1GateConfig(),
    chain_contribution_share: float | None = None,
) -> ChainEvaluationResult:
    """Run the complete pure R1 return-evaluation pipeline on a daily long table."""

    prepared = prepare_daily_strategy_returns(daily, config)
    strategy_summary = summarize_strategy_returns(prepared, config)
    paired_daily = build_r1_paired_excess(prepared, config, gate_config)
    paired_summary, half_year_summary = summarize_r1_pairs(
        paired_daily, periods_per_year=config.periods_per_year
    )
    gate = evaluate_r1_gate(
        paired_daily,
        paired_summary,
        half_year_summary,
        chain_contribution_share=chain_contribution_share,
        gate_config=gate_config,
    )
    return ChainEvaluationResult(
        daily=prepared,
        strategy_summary=strategy_summary,
        paired_daily=paired_daily,
        paired_summary=paired_summary,
        half_year_summary=half_year_summary,
        gate=gate,
    )


__all__ = [
    "ChainEvaluationConfig",
    "ChainEvaluationResult",
    "DEFAULT_CONTROLS",
    "GateEvaluation",
    "HISTORICAL_COST_SWITCH_DATE",
    "R1GateConfig",
    "aggregate_equal_weight_slots",
    "build_r1_paired_excess",
    "equal_weight_return_with_cash",
    "evaluate_chain_returns",
    "evaluate_r1_gate",
    "prepare_daily_strategy_returns",
    "summarize_paired_net_excess_inference",
    "summarize_r1_pairs",
    "summarize_strategy_returns",
]
