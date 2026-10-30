"""Helpers that make release timing and as-of joins explicit."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


def add_release_dates(
    frame: pd.DataFrame,
    *,
    observation_col: str = "observation_date",
    available_col: str = "available_at",
    lag_days: int = 1,
    business_days: bool = False,
) -> pd.DataFrame:
    """Attach the earliest date on which each observation may be consumed."""

    if lag_days < 0:
        raise ValueError("release lag cannot be negative")
    if observation_col not in frame:
        raise ValueError(f"Missing observation column: {observation_col}")
    result = frame.copy()
    observations = pd.to_datetime(result[observation_col], errors="raise").dt.normalize()
    offset: pd.DateOffset
    offset = pd.offsets.BDay(lag_days) if business_days else pd.Timedelta(days=lag_days)
    result[observation_col] = observations
    result[available_col] = observations + offset
    return result


def point_in_time_join(
    decision_dates: pd.DatetimeIndex | Iterable[pd.Timestamp | str],
    observations: pd.DataFrame,
    *,
    available_col: str = "available_at",
    value_columns: list[str] | None = None,
    max_age: pd.Timedelta | str | None = None,
) -> pd.DataFrame:
    """As-of join observations using only values available by each decision date.

    The source availability timestamp is retained as ``source_available_at`` so
    callers and tests can audit that no future value entered a signal.
    """

    if available_col not in observations:
        raise ValueError(f"Missing point-in-time availability column: {available_col}")
    dates = pd.DatetimeIndex(pd.to_datetime(list(decision_dates))).tz_localize(None).normalize()
    left = pd.DataFrame({"decision_date": dates}).sort_values("decision_date")
    right = observations.copy()
    right[available_col] = (
        pd.to_datetime(right[available_col], errors="raise")
        .dt.tz_localize(None)
        .dt.normalize()
    )
    if value_columns is None:
        value_columns = [column for column in right if column != available_col]
    selected = right[[available_col, *value_columns]].sort_values(available_col)
    selected = selected.drop_duplicates(available_col, keep="last")
    joined = pd.merge_asof(
        left,
        selected,
        left_on="decision_date",
        right_on=available_col,
        direction="backward",
        tolerance=None if max_age is None else pd.Timedelta(max_age),
    )
    joined = joined.rename(columns={available_col: "source_available_at"})
    joined = joined.set_index("decision_date")
    joined.index.name = "date"
    return joined.reindex(dates)


def assert_point_in_time(
    frame: pd.DataFrame,
    *,
    decision_col: str | None = None,
    available_col: str = "source_available_at",
) -> None:
    """Raise if any row consumes an observation from the future."""

    if available_col not in frame:
        raise ValueError(f"Missing availability column: {available_col}")
    decision = pd.to_datetime(frame.index if decision_col is None else frame[decision_col])
    available = pd.to_datetime(frame[available_col])
    violation = available.notna() & (available.to_numpy() > decision.to_numpy())
    if bool(violation.any()):
        raise AssertionError("Future observations detected in point-in-time frame")
