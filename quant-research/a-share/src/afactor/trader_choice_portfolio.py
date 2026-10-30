"""Strict point-in-time portfolio-replacement scoring.

This module adds the sixth, portfolio-aware factor family to an already scored
candidate panel.  A decision day is scored only when the supplied trading
calendar maps it to a position snapshot on the immediately preceding session.
Snapshots are never carried forward.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd


_FAMILY_KEYS = (
    "price_volume",
    "dynamic_direction",
    "within_direction_leader",
    "event_attention",
    "familiarity",
)
_DEFAULT_FAMILY_COLUMNS = {
    key: f"family_score__{key}" for key in _FAMILY_KEYS
}
_CONTEXT_KEYS = (
    "dynamic_direction",
    "within_direction_leader",
)
_ATTENTION_KEYS = ("event_attention", "familiarity")


def _require_columns(frame: pd.DataFrame, columns: list[str], where: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{where} is missing required columns: {missing}")


def _normalize_dates(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, errors="coerce").dt.normalize()


def _normalize_symbol(value: object) -> str | None:
    if pd.isna(value):
        return None
    symbol = str(value).strip().upper()
    if not symbol:
        return None
    if symbol.endswith((".SH", ".SZ", ".BJ")):
        return symbol
    if len(symbol) == 6 and symbol.isdigit():
        if symbol.startswith("6"):
            return f"{symbol}.SH"
        if symbol.startswith(("0", "3")):
            return f"{symbol}.SZ"
        if symbol.startswith(("4", "8", "9")):
            return f"{symbol}.BJ"
    # Preserve explicit synthetic/test symbols and name-coded disclosed assets.
    return symbol


def _resolve_family_columns(
    family_columns: Mapping[str, str] | None,
) -> dict[str, str]:
    resolved = dict(_DEFAULT_FAMILY_COLUMNS)
    if family_columns is not None:
        unknown = sorted(set(family_columns) - set(_FAMILY_KEYS))
        if unknown:
            raise ValueError(f"unknown family mapping keys: {unknown}")
        resolved.update({str(key): str(value) for key, value in family_columns.items()})
    if any(not resolved[key].strip() for key in _FAMILY_KEYS):
        raise ValueError("family column names must not be empty")
    return resolved


def _prepare_previous_session(
    previous_session: Mapping[object, object],
) -> dict[pd.Timestamp, pd.Timestamp]:
    if not hasattr(previous_session, "items"):
        raise TypeError("previous_session must be a date-to-date mapping")
    result: dict[pd.Timestamp, pd.Timestamp] = {}
    for raw_date, raw_previous in previous_session.items():
        decision_date = pd.to_datetime(raw_date, errors="coerce")
        snapshot_date = pd.to_datetime(raw_previous, errors="coerce")
        if pd.isna(decision_date) or pd.isna(snapshot_date):
            continue
        decision_date = pd.Timestamp(decision_date).normalize()
        snapshot_date = pd.Timestamp(snapshot_date).normalize()
        if snapshot_date >= decision_date:
            continue
        if decision_date in result and result[decision_date] != snapshot_date:
            raise ValueError(f"conflicting previous sessions for {decision_date.date()}")
        result[decision_date] = snapshot_date
    return result


def _load_complete_snapshots(
    position_state_path: str | Path,
) -> dict[pd.Timestamp, tuple[str, ...]]:
    states = pd.read_csv(position_state_path, dtype={"security_code": "string"})
    required = [
        "trading_date",
        "security_code",
        "total_quantity_best",
        "confidence",
    ]
    _require_columns(states, required, "position_state")
    states = states.copy()
    states["trading_date"] = _normalize_dates(states["trading_date"])
    states["_quantity"] = pd.to_numeric(
        states["total_quantity_best"], errors="coerce"
    )
    states["_symbol"] = states["security_code"].map(_normalize_symbol)
    states["_confidence"] = (
        states["confidence"].astype("string").str.strip().str.upper()
    )
    states = states.loc[states["trading_date"].notna()].copy()

    snapshots: dict[pd.Timestamp, tuple[str, ...]] = {}
    for snapshot_date, snapshot in states.groupby("trading_date", sort=True):
        # A missing quantity makes the disclosed cross-section incomplete.
        if snapshot["_quantity"].isna().any():
            continue
        positive = snapshot.loc[snapshot["_quantity"].gt(0)].copy()
        # The choice model ranks ordinary A shares.  ETF/name-coded rows may be
        # present in the account screenshot, but they are neither candidate
        # stocks nor valid replacement targets for this layer.
        symbol_text = positive["_symbol"].astype("string")
        ordinary_stock = (
            symbol_text.str.fullmatch(r"\d{6}\.(?:SH|SZ)", na=False)
            & symbol_text.str.slice(0, 1).isin(["0", "3", "6"])
        )
        positive = positive.loc[ordinary_stock].copy()
        if positive.empty:
            continue
        # Zero-quantity exit rows and non-stock assets do not belong to this
        # comparison set.  Every live ordinary-stock holding must be L4.
        if not positive["_confidence"].eq("L4").all():
            continue
        if positive["_symbol"].isna().any():
            continue
        holdings = tuple(sorted(set(positive["_symbol"].astype(str))))
        if holdings:
            snapshots[pd.Timestamp(snapshot_date)] = holdings
    return snapshots


def _prepare_closes(closes: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(closes, pd.DataFrame):
        raise TypeError("closes must be a wide pandas DataFrame")
    result = closes.copy()
    dates = pd.to_datetime(result.index, errors="coerce")
    if pd.isna(dates).any():
        raise ValueError("closes index contains invalid dates")
    result.index = pd.DatetimeIndex(dates).normalize()
    if result.index.has_duplicates:
        raise ValueError("closes index must contain at most one row per session")
    normalized_columns = [_normalize_symbol(column) for column in result.columns]
    if any(column is None for column in normalized_columns):
        raise ValueError("closes contains an invalid symbol column")
    if len(set(normalized_columns)) != len(normalized_columns):
        raise ValueError("closes symbol columns collide after normalization")
    result.columns = normalized_columns
    result = result.apply(pd.to_numeric, errors="coerce").sort_index()
    return result


def _mapped_difference(candidate: object, holding: object) -> float | None:
    candidate_value = pd.to_numeric(pd.Series([candidate]), errors="coerce").iloc[0]
    holding_value = pd.to_numeric(pd.Series([holding]), errors="coerce").iloc[0]
    if not np.isfinite(candidate_value) or not np.isfinite(holding_value):
        return None
    raw_difference = float(candidate_value) - float(holding_value)
    return (float(np.clip(raw_difference, -1.0, 1.0)) + 1.0) / 2.0


def _return_window(
    closes: pd.DataFrame,
    snapshot_date: pd.Timestamp,
    correlation_lookback: int,
) -> pd.DataFrame:
    prices = closes.loc[closes.index <= snapshot_date].tail(correlation_lookback + 1)
    if len(prices) < 2:
        return pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)
    returns = prices.pct_change(fill_method=None).iloc[1:]
    return returns.replace([np.inf, -np.inf], np.nan)


def _pair_correlation(
    returns: pd.DataFrame,
    left: str,
    right: str,
    minimum_observations: int,
    cache: dict[tuple[str, str], float],
) -> float:
    cache_key = tuple(sorted((left, right)))
    if cache_key in cache:
        return cache[cache_key]
    if left not in returns.columns or right not in returns.columns:
        cache[cache_key] = np.nan
        return np.nan
    if left == right:
        values = returns[left].dropna()
        correlation = (
            1.0
            if len(values) >= minimum_observations and values.nunique() >= 2
            else np.nan
        )
        cache[cache_key] = correlation
        return correlation
    pairs = returns[[left, right]].dropna()
    if len(pairs) < minimum_observations:
        correlation = np.nan
    elif pairs[left].nunique() < 2 or pairs[right].nunique() < 2:
        correlation = np.nan
    else:
        correlation = float(pairs[left].corr(pairs[right]))
        if not np.isfinite(correlation):
            correlation = np.nan
    cache[cache_key] = correlation
    return correlation


def _diversification_component(
    candidate: str,
    holding: str,
    holdings: tuple[str, ...],
    returns: pd.DataFrame,
    minimum_observations: int,
    cache: dict[tuple[str, str], float],
) -> float:
    differences = []
    for other in holdings:
        if other == holding:
            continue
        holding_correlation = _pair_correlation(
            returns, holding, other, minimum_observations, cache
        )
        candidate_correlation = _pair_correlation(
            returns, candidate, other, minimum_observations, cache
        )
        if np.isfinite(holding_correlation) and np.isfinite(candidate_correlation):
            differences.append(holding_correlation - candidate_correlation)
    if not differences:
        # Correlation is an explicit neutral component when history or portfolio
        # breadth is insufficient; it is never backfilled from another date.
        return 0.5
    raw_difference = float(np.mean(differences))
    return (float(np.clip(raw_difference, -1.0, 1.0)) + 1.0) / 2.0


def _pair_score(
    candidate_symbol: str,
    holding_symbol: str,
    holdings: tuple[str, ...],
    candidate_row: pd.Series,
    holding_row: pd.Series | None,
    family_columns: Mapping[str, str],
    returns: pd.DataFrame,
    minimum_observations: int,
    correlation_cache: dict[tuple[str, str], float],
) -> float:
    components = []
    if holding_row is not None:
        price_volume = _mapped_difference(
            candidate_row[family_columns["price_volume"]],
            holding_row[family_columns["price_volume"]],
        )
        if price_volume is not None:
            components.append(price_volume)

        context_differences = []
        for family in _CONTEXT_KEYS:
            candidate_value = pd.to_numeric(
                pd.Series([candidate_row[family_columns[family]]]), errors="coerce"
            ).iloc[0]
            holding_value = pd.to_numeric(
                pd.Series([holding_row[family_columns[family]]]), errors="coerce"
            ).iloc[0]
            if np.isfinite(candidate_value) and np.isfinite(holding_value):
                context_differences.append(float(candidate_value) - float(holding_value))
        if context_differences:
            raw_context = float(np.mean(context_differences))
            components.append((float(np.clip(raw_context, -1.0, 1.0)) + 1.0) / 2.0)

        attention_differences = []
        for family in _ATTENTION_KEYS:
            candidate_value = pd.to_numeric(
                pd.Series([candidate_row[family_columns[family]]]), errors="coerce"
            ).iloc[0]
            holding_value = pd.to_numeric(
                pd.Series([holding_row[family_columns[family]]]), errors="coerce"
            ).iloc[0]
            if np.isfinite(candidate_value) and np.isfinite(holding_value):
                attention_differences.append(
                    float(candidate_value) - float(holding_value)
                )
        if attention_differences:
            raw_attention = float(np.mean(attention_differences))
            components.append(
                (float(np.clip(raw_attention, -1.0, 1.0)) + 1.0) / 2.0
            )

    components.append(
        _diversification_component(
            candidate_symbol,
            holding_symbol,
            holdings,
            returns,
            minimum_observations,
            correlation_cache,
        )
    )
    return float(np.mean(components))


def add_portfolio_replacement(
    panel: pd.DataFrame,
    closes: pd.DataFrame,
    position_state_path: str | Path,
    previous_session: Mapping[object, object],
    date_col: str = "decision_date",
    symbol_col: str = "symbol",
    family_columns: Mapping[str, str] | None = None,
    correlation_lookback: int = 60,
    correlation_minimum_observations: int = 40,
) -> pd.DataFrame:
    """Add strict-T-1 portfolio replacement scores to a candidate panel.

    For candidate ``i`` and live holding ``h``, the pair score is the mean of
    available price/volume, context, attention, and diversification components.
    Context equally combines dynamic-direction and within-direction leadership;
    attention equally combines event attention and familiarity.  Each composite
    uses only jointly observed candidate/holding pairs.  Raw score differences
    are clipped to ``[-1, 1]`` and mapped to ``[0, 1]``.  Diversification uses
    60 sessions and requires 40 paired returns by default; when correlation is
    not estimable, it contributes neutral ``0.5``.  The returned F6 score is the
    maximum pair score across holdings.

    A snapshot is eligible only when every positive holding on exactly
    ``previous_session[decision_date]`` is L4.  Missing, stale, or lower-
    confidence snapshots produce no score for the entire decision day.
    """

    if correlation_lookback < 2:
        raise ValueError("correlation_lookback must be at least 2")
    if correlation_minimum_observations < 2:
        raise ValueError("correlation_minimum_observations must be at least 2")
    families = _resolve_family_columns(family_columns)
    _require_columns(panel, [date_col, symbol_col, *families.values()], "panel")

    original_index = panel.index.copy()
    original_index_name = panel.index.name
    # Use a private unique row index while computing.  Research panels are often
    # concatenated with repeated external index labels; output restores them.
    result = panel.copy().reset_index(drop=True)
    result[date_col] = _normalize_dates(result[date_col])
    result[symbol_col] = result[symbol_col].map(_normalize_symbol).astype("string")
    if result[date_col].isna().any():
        raise ValueError(f"panel contains invalid {date_col} values")
    if result[symbol_col].isna().any():
        raise ValueError(f"panel contains invalid {symbol_col} values")
    if result.duplicated([date_col, symbol_col]).any():
        raise ValueError(f"panel must be unique by ({date_col}, {symbol_col})")
    for column in families.values():
        result[column] = pd.to_numeric(result[column], errors="coerce")

    prepared_closes = _prepare_closes(closes)
    sessions = _prepare_previous_session(previous_session)
    snapshots = _load_complete_snapshots(position_state_path)

    score_column = "family_score__portfolio_replacement"
    result[score_column] = np.nan
    result["predicted_sell_symbol"] = pd.Series(
        pd.NA, index=result.index, dtype="string"
    )
    result["portfolio_snapshot_date"] = pd.NaT
    result["portfolio_context_available"] = False
    result["portfolio_holding_count"] = pd.Series(
        pd.NA, index=result.index, dtype="Int64"
    )

    for decision_date, day_index in result.groupby(date_col, sort=True).groups.items():
        decision_date = pd.Timestamp(decision_date)
        snapshot_date = sessions.get(decision_date)
        if snapshot_date is None:
            continue
        # Exact lookup is intentional.  Never use the most recent older state.
        holdings = snapshots.get(snapshot_date)
        if not holdings:
            continue

        day = result.loc[day_index]
        day_by_symbol = day.set_index(symbol_col, drop=False)
        returns = _return_window(
            prepared_closes, snapshot_date, correlation_lookback
        )
        correlation_cache: dict[tuple[str, str], float] = {}
        day_scores: dict[object, float] = {}
        day_sells: dict[object, str] = {}

        for row_index, candidate_row in day.iterrows():
            candidate_symbol = str(candidate_row[symbol_col])
            best_score = -np.inf
            best_holding: str | None = None
            for holding_symbol in holdings:
                holding_row = (
                    day_by_symbol.loc[holding_symbol]
                    if holding_symbol in day_by_symbol.index
                    else None
                )
                pair_score = _pair_score(
                    candidate_symbol,
                    holding_symbol,
                    holdings,
                    candidate_row,
                    holding_row,
                    families,
                    returns,
                    correlation_minimum_observations,
                    correlation_cache,
                )
                # Holdings are sorted, so an exact tie deterministically keeps
                # the lexicographically smallest predicted sale.
                if pair_score > best_score:
                    best_score = pair_score
                    best_holding = holding_symbol
            if best_holding is not None:
                day_scores[row_index] = best_score
                day_sells[row_index] = best_holding

        score_series = pd.Series(day_scores, dtype=float)
        sell_series = pd.Series(day_sells, dtype="string")
        result.loc[score_series.index, score_column] = score_series
        result.loc[sell_series.index, "predicted_sell_symbol"] = sell_series
        result.loc[day_index, "portfolio_snapshot_date"] = snapshot_date
        result.loc[day_index, "portfolio_context_available"] = True
        result.loc[day_index, "portfolio_holding_count"] = len(holdings)

    result.index = original_index
    result.index.name = original_index_name
    return result


def _self_test() -> None:
    """Run a compact no-carry and deterministic-max smoke test."""

    import tempfile

    dates = pd.date_range("2026-01-01", periods=70, freq="B")
    base = np.linspace(100.0, 130.0, len(dates))
    closes = pd.DataFrame(
        {
            "H1": base,
            "H2": base * 1.01,
            "C": base[::-1],
        },
        index=dates,
    )
    snapshot_date = dates[-3]
    scored_date = dates[-2]
    unavailable_date = dates[-1]
    family_names = list(_DEFAULT_FAMILY_COLUMNS.values())
    rows = []
    for decision_date in (scored_date, unavailable_date):
        for symbol, score in (("C", 1.0), ("H1", 0.0), ("H2", 0.8)):
            row = {"decision_date": decision_date, "symbol": symbol}
            row.update({column: score for column in family_names})
            rows.append(row)
    panel = pd.DataFrame(rows)
    states = pd.DataFrame(
        {
            "trading_date": [snapshot_date, snapshot_date],
            "security_code": ["H1", "H2"],
            "total_quantity_best": [100, 200],
            "confidence": ["L4", "L4"],
        }
    )
    with tempfile.TemporaryDirectory() as temporary_directory:
        state_path = Path(temporary_directory) / "position_state.csv"
        states.to_csv(state_path, index=False)
        result = add_portfolio_replacement(
            panel,
            closes,
            state_path,
            {
                scored_date: snapshot_date,
                unavailable_date: scored_date,
            },
        )
    scored = result.loc[result["decision_date"].eq(scored_date)]
    unavailable = result.loc[result["decision_date"].eq(unavailable_date)]
    assert scored["portfolio_context_available"].all()
    assert scored["family_score__portfolio_replacement"].notna().all()
    assert scored.loc[
        scored["symbol"].eq("C"), "family_score__portfolio_replacement"
    ].item() > 0.5
    assert unavailable["portfolio_context_available"].eq(False).all()
    assert unavailable["family_score__portfolio_replacement"].isna().all()
    assert unavailable["predicted_sell_symbol"].isna().all()


if __name__ == "__main__":
    _self_test()


__all__ = ["add_portfolio_replacement"]
