"""Label-free integration helpers for the frozen trader-chain experiment.

The functions here bridge the point-in-time membership and deterministic rule
modules.  They accept in-memory market features only and perform no file I/O.
In particular, disclosed trades, labels, holdings, forward returns and PnL are
not accepted by any feature-side helper.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd


CHAIN_DIRECTION_FACTORS: tuple[str, ...] = (
    "chain_mom20_member_market_pct",
    "chain_mom60_member_market_pct",
    "chain_breadth_above_ma20",
    "chain_amount_ratio_member_market_pct",
)


def _require_columns(
    frame: pd.DataFrame, columns: Sequence[str], *, where: str
) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{where} is missing required columns: {missing}")


def _date_series(values: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce", utc=True)
    return parsed.dt.tz_convert(None).dt.normalize()


def _bool_series(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values.dtype):
        return values.fillna(False).astype(bool)
    if pd.api.types.is_numeric_dtype(values.dtype):
        return pd.to_numeric(values, errors="coerce").fillna(0).ne(0)
    return (
        values.astype("string")
        .str.strip()
        .str.lower()
        .isin({"1", "true", "t", "yes", "y"})
        .fillna(False)
    )


def build_chain_direction_inputs(
    member_panel: pd.DataFrame,
    *,
    minimum_members: int = 5,
) -> pd.DataFrame:
    """Aggregate eligible active members to the four frozen chain factors.

    The momentum and amount inputs must already be each stock's same-day
    percentile in the full eligible market, as required by the protocol.
    Breadth is the fraction of eligible members with ``ma20_dist > 0``.
    """

    if isinstance(minimum_members, bool) or int(minimum_members) != minimum_members:
        raise ValueError("minimum_members must be a positive integer")
    minimum = int(minimum_members)
    if minimum <= 0:
        raise ValueError("minimum_members must be a positive integer")
    required = (
        "decision_date",
        "symbol",
        "chain_id",
        "eligible_for_ranking",
        "mom20_pct",
        "mom60_pct",
        "ma20_dist",
        "amt_ratio20_pct",
    )
    _require_columns(member_panel, required, where="member_panel")
    frame = member_panel.copy()
    frame["decision_date"] = _date_series(frame["decision_date"])
    frame["symbol"] = frame["symbol"].astype("string").str.strip()
    frame["chain_id"] = frame["chain_id"].astype("string").str.strip()
    keys = ["decision_date", "symbol", "chain_id"]
    if frame[keys].isna().any(axis=None):
        raise ValueError("member_panel contains invalid date/symbol/chain keys")
    if frame["symbol"].eq("").any() or frame["chain_id"].eq("").any():
        raise ValueError("member_panel contains blank symbol or chain_id")
    if frame.duplicated(keys, keep=False).any():
        raise ValueError("member_panel must be unique by date, symbol and chain")

    eligible = _bool_series(frame["eligible_for_ranking"])
    frame = frame.loc[eligible].copy()
    numeric_columns = ("mom20_pct", "mom60_pct", "ma20_dist", "amt_ratio20_pct")
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
    frame["_above_ma20"] = frame["ma20_dist"].gt(0).where(
        frame["ma20_dist"].notna()
    )

    group_columns = ["decision_date", "chain_id"]
    grouped = frame.groupby(group_columns, sort=True, dropna=False)
    result = grouped.agg(
        eligible_member_count=("symbol", "nunique"),
        chain_mom20_member_market_pct=("mom20_pct", "median"),
        chain_mom60_member_market_pct=("mom60_pct", "median"),
        chain_breadth_above_ma20=("_above_ma20", "mean"),
        chain_amount_ratio_member_market_pct=("amt_ratio20_pct", "median"),
    ).reset_index()
    if "chain_name" in frame.columns:
        names = grouped["chain_name"].agg(
            lambda values: next(
                (
                    str(value).strip()
                    for value in values
                    if pd.notna(value) and str(value).strip()
                ),
                "",
            )
        ).rename("chain_name")
        result = result.merge(
            names.reset_index(),
            on=group_columns,
            how="left",
            validate="one_to_one",
        )
    factor_complete = result[list(CHAIN_DIRECTION_FACTORS)].notna().all(axis=1)
    result["chain_eligible"] = (
        result["eligible_member_count"].ge(minimum) & factor_complete
    )
    result["minimum_members"] = minimum
    result["chain_direction_input_clock"] = (
        "member full-market percentiles and prices/amounts through T-1"
    )
    first_columns = ["decision_date", "chain_id"]
    if "chain_name" in result.columns:
        first_columns.append("chain_name")
    trailing = [column for column in result.columns if column not in first_columns]
    return result.loc[:, [*first_columns, *trailing]].sort_values(
        ["decision_date", "chain_id"], kind="mergesort"
    ).reset_index(drop=True)


def trailing_return_correlations(
    decision_date: object,
    symbols: Iterable[object],
    closes: pd.DataFrame,
    *,
    lookback_sessions: int = 60,
    minimum_observations: int = 40,
) -> pd.DataFrame:
    """Build finite pairwise correlations using returns strictly before ``T``.

    The output is intentionally one decision day at a time so the runner can
    evaluate every candidate without materialising a multi-million-row lookup.
    Missing/insufficient pairs remain absent and are audited by the selector.
    """

    if lookback_sessions <= 1:
        raise ValueError("lookback_sessions must exceed one")
    if minimum_observations < 2 or minimum_observations > lookback_sessions:
        raise ValueError(
            "minimum_observations must be between 2 and lookback_sessions"
        )
    if not isinstance(closes, pd.DataFrame):
        raise TypeError("closes must be a pandas DataFrame")
    prices = closes.copy()
    index = pd.to_datetime(prices.index, errors="coerce", utc=True)
    if index.isna().any():
        raise ValueError("closes index contains invalid dates")
    prices.index = index.tz_convert(None).normalize()
    if prices.index.duplicated().any():
        raise ValueError("closes index must be unique")
    prices = prices.sort_index()
    date = pd.Timestamp(decision_date)
    if date.tzinfo is not None:
        date = date.tz_localize(None)
    date = date.normalize()
    if date not in prices.index:
        raise ValueError(f"decision date {date.date()} is absent from closes")
    position = int(prices.index.get_loc(date))
    start = max(0, position - lookback_sessions - 1)
    history = prices.iloc[start:position]
    requested = tuple(
        dict.fromkeys(
            str(symbol).strip()
            for symbol in symbols
            if pd.notna(symbol) and str(symbol).strip()
        )
    )
    available = [symbol for symbol in requested if symbol in history.columns]
    output_columns = (
        "decision_date",
        "symbol_a",
        "symbol_b",
        "correlation",
        "lookback_start",
        "lookback_end",
        "lookback_sessions",
        "minimum_observations",
    )
    if len(available) < 2 or len(history) < 3:
        return pd.DataFrame(columns=output_columns)
    returns = history.loc[:, available].pct_change(fill_method=None).iloc[1:]
    correlations = returns.corr(min_periods=minimum_observations)
    values = correlations.to_numpy(float)
    upper_i, upper_j = np.triu_indices(len(available), k=1)
    pair_values = values[upper_i, upper_j]
    finite = np.isfinite(pair_values)
    if not finite.any():
        return pd.DataFrame(columns=output_columns)
    end = history.index[-1]
    result = pd.DataFrame(
        {
            "decision_date": date,
            "symbol_a": np.asarray(available, dtype=object)[upper_i[finite]],
            "symbol_b": np.asarray(available, dtype=object)[upper_j[finite]],
            "correlation": pair_values[finite],
            "lookback_start": history.index[0],
            "lookback_end": end,
            "lookback_sessions": lookback_sessions,
            "minimum_observations": minimum_observations,
        }
    )
    return result.loc[:, output_columns]


__all__ = [
    "CHAIN_DIRECTION_FACTORS",
    "build_chain_direction_inputs",
    "trailing_return_correlations",
]
