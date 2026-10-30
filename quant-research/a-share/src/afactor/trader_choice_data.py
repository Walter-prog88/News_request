"""Point-in-time data builders for the trader-choice research line.

This module implements the five frozen, deterministic factor families used to
study *which* stocks the disclosed trader selected.  It is deliberately a data
layer rather than a fitted model:

``F1`` multi-shape price/amount, ``F2`` dynamic statistical direction,
``F3`` within-cluster leadership, ``F4`` event/attention, and ``F5`` the
trader's disclosed familiarity set.

The feature clock is strict.  For a decision made at the open of session ``T``
only the open of ``T`` and closes/amounts through ``T-1`` are available.  A
report, forecast, express report, or top-list observation dated ``d`` first
becomes usable on the first trading session strictly after ``d``.  Forward
returns are produced by a separate function so that they cannot accidentally
enter the feature builders.

The two mixed-time-zone minute sessions found in the source audit (2026-08-10
and 2026-08-11) are removed everywhere, including rolling windows and forward
return horizons.
"""

from __future__ import annotations

import csv
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


REJECTED_DATES = pd.DatetimeIndex(["2026-08-10", "2026-08-11"])
RAW_FEATURES: tuple[str, ...] = (
    "gap",
    "prev_ret",
    "mom5",
    "mom10",
    "mom20",
    "mom60",
    "ma20_dist",
    "drawdown20",
    "vol20",
    "amt_ratio20",
    "liq20",
)
SHAPE_VECTOR_PERCENTILES: tuple[str, ...] = (
    "mom5_pct",
    "mom10_pct",
    "mom20_pct",
    "mom60_pct",
    "vol20_pct",
    "amt_ratio20_pct",
    "liq20_pct",
    "gap_pct",
    "ma20_dist_pct",
    "drawdown20_pct",
)
DEFAULT_HORIZONS: tuple[int, ...] = (1, 3, 5)


@dataclass(frozen=True)
class ChoiceDataPaths:
    """Locations used by the reusable choice-data builders.

    Passing only ``astock_root`` and ``trader_root`` is sufficient for a copied
    or test repository; all individual files are resolved relative to them.
    No directory is created and no source file is modified by this module.
    """

    astock_root: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[2]
    )
    trader_root: Path = field(
        default_factory=lambda: Path(
            os.environ.get(
                "TRADER_RESEARCH_ROOT",
                "/path/to/user/Documents/ChatGPT/Stock model",
            )
        )
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "astock_root", Path(self.astock_root))
        object.__setattr__(self, "trader_root", Path(self.trader_root))

    @property
    def daily_research_dir(self) -> Path:
        return self.astock_root / "cache" / "daily"

    @property
    def daily_sealed_dir(self) -> Path:
        return self.astock_root / "cache" / "sealed"

    @property
    def recent_parts_dir(self) -> Path:
        return self.astock_root / "cache" / "recent_daily" / "parts"

    @property
    def statind_labels_file(self) -> Path:
        return self.daily_research_dir / "statind_labels.parquet"

    @property
    def reports_file(self) -> Path:
        return (
            self.astock_root
            / "data_8:21"
            / "source_raw"
            / "eastmoney_analyst_reports_raw.csv"
        )

    @property
    def forecast_file(self) -> Path:
        return self.astock_root / "data_8:21" / "forecast.csv"

    @property
    def express_file(self) -> Path:
        return self.astock_root / "data_8:21" / "express.csv"

    @property
    def toplist_file(self) -> Path:
        return self.astock_root / "data_8:21" / "top_list.csv"

    @property
    def events_file(self) -> Path:
        return self.trader_root / "data" / "normalized" / "events.csv"

    @property
    def positions_file(self) -> Path:
        return self.trader_root / "data" / "normalized" / "position_state.csv"


def _timestamp(value: object, *, name: str) -> pd.Timestamp:
    result = pd.Timestamp(value)
    if pd.isna(result):
        raise ValueError(f"{name} must be a valid date")
    if result.tzinfo is not None:
        result = result.tz_localize(None)
    return result.normalize()


def _date_series(values: pd.Series) -> pd.Series:
    """Parse mixed YYYYMMDD / ISO / timestamp columns without timezone drift."""

    text = values.astype("string").str.strip()
    compact = text.str.fullmatch(r"\d{8}", na=False)
    out = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns]")
    if compact.any():
        out.loc[compact] = pd.to_datetime(
            text.loc[compact], format="%Y%m%d", errors="coerce"
        )
    if (~compact).any():
        parsed = pd.to_datetime(text.loc[~compact], errors="coerce", utc=True)
        out.loc[~compact] = parsed.dt.tz_convert(None).dt.normalize()
    return out


def _standard_symbol(value: object) -> str | pd.NA:
    if value is None or pd.isna(value):
        return pd.NA
    text = str(value).strip().upper()
    if not text:
        return pd.NA
    if "." in text:
        left, right = text.split(".", 1)
        if left in {"SH", "SZ", "BJ"} and right.isdigit():
            return f"{right.zfill(6)}.{left}"
        if right in {"SH", "SZ", "BJ"} and left.isdigit():
            return f"{left.zfill(6)}.{right}"
    digits = "".join(ch for ch in text if ch.isdigit())
    if len(digits) != 6:
        return pd.NA
    # Shanghai/SZ B-share codes need their real exchange suffix for joins, even
    # though they are explicitly excluded from the ordinary-A eligibility mask.
    if digits.startswith(("6", "900")):
        exchange = "SH"
    elif digits.startswith(("0", "3", "200")):
        exchange = "SZ"
    elif digits.startswith(("4", "8", "9")):
        exchange = "BJ"
    else:
        return pd.NA
    return f"{digits}.{exchange}"


def _normalize_wide(frame: pd.DataFrame, *, name: str) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame")
    result = frame.copy()
    index = pd.to_datetime(result.index, errors="coerce", utc=True)
    if index.isna().any():
        raise ValueError(f"{name} contains invalid dates")
    result.index = index.tz_convert(None).normalize()
    result = result.loc[~result.index.isin(REJECTED_DATES)]
    result = result[~result.index.duplicated(keep="last")].sort_index()
    normalized_columns = pd.Index([_standard_symbol(c) for c in result.columns])
    valid = normalized_columns.notna()
    result = result.loc[:, valid]
    result.columns = normalized_columns[valid].astype(str)
    if result.columns.duplicated().any():
        result = result.T.groupby(level=0, sort=False).last().T
    return result.apply(pd.to_numeric, errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )


def _align_wide(
    opens: pd.DataFrame,
    closes: pd.DataFrame,
    amounts: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    normalized = (
        _normalize_wide(opens, name="opens"),
        _normalize_wide(closes, name="closes"),
        _normalize_wide(amounts, name="amounts"),
    )
    dates = normalized[0].index.union(normalized[1].index).union(normalized[2].index)
    dates = dates.sort_values().difference(REJECTED_DATES)
    symbols = normalized[0].columns.union(normalized[1].columns).union(
        normalized[2].columns
    )
    return tuple(frame.reindex(index=dates, columns=symbols) for frame in normalized)  # type: ignore[return-value]


def _combine_daily_frames(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    available = [frame for frame in frames if frame is not None and not frame.empty]
    if not available:
        return pd.DataFrame()
    out = pd.concat(available, axis=0, sort=False).sort_index()
    return out.loc[~out.index.duplicated(keep="last")]


def _stack_all(frame: pd.DataFrame) -> pd.Series:
    """Compatibility wrapper for pandas' pre/post-2.1 stack implementations."""

    try:
        return frame.stack(future_stack=True)
    except TypeError:  # pandas < 2.1
        return frame.stack(dropna=False)


def _read_cached_field(paths: ChoiceDataPaths, field_name: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for directory in (paths.daily_research_dir, paths.daily_sealed_dir):
        path = directory / f"{field_name}.parquet"
        if path.exists():
            frames.append(pd.read_parquet(path))
    return _combine_daily_frames(frames)


def _read_recent_parts(
    directory: Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read the small per-symbol recent CSVs without requiring a glob concat."""

    rows: list[tuple[pd.Timestamp, str, float, float, float]] = []
    if not directory.exists():
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    start_text, end_text = start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")
    for path in sorted(directory.glob("*.csv")):
        if path.stat().st_size == 0:
            continue
        try:
            with path.open(newline="", encoding="utf-8") as handle:
                for record in csv.DictReader(handle):
                    raw_date = str(record.get("date", ""))[:10]
                    if not (start_text <= raw_date <= end_text):
                        continue
                    symbol = _standard_symbol(record.get("code"))
                    if pd.isna(symbol):
                        continue
                    values: list[float] = []
                    for column in ("open", "close", "amount"):
                        try:
                            values.append(float(record.get(column, "nan")))
                        except (TypeError, ValueError):
                            values.append(np.nan)
                    rows.append(
                        (pd.Timestamp(raw_date), str(symbol), values[0], values[1], values[2])
                    )
        except (OSError, UnicodeError, csv.Error):
            continue
    if not rows:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    recent = pd.DataFrame(rows, columns=["date", "symbol", "open", "close", "amount"])
    recent = recent.drop_duplicates(["date", "symbol"], keep="last")
    pivots = tuple(
        recent.pivot(index="date", columns="symbol", values=column).sort_index()
        for column in ("open", "close", "amount")
    )
    return pivots  # type: ignore[return-value]


def load_daily_inputs(
    start: object,
    end: object,
    *,
    paths: ChoiceDataPaths | None = None,
    lookback_sessions: int = 520,
    forward_sessions: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load and align daily open, close, and amount panels.

    Cached research/sealed daily panels are joined to the recent per-symbol
    CSVs, with recent observations winning exact-date overlaps.  The returned
    range includes up to ``lookback_sessions`` sessions before ``start`` and
    ``forward_sessions`` sessions after ``end``.  The longer default lookback
    supports 120-day statistical clustering and historical-shape familiarity.
    """

    paths = paths or ChoiceDataPaths()
    start_ts = _timestamp(start, name="start")
    end_ts = _timestamp(end, name="end")
    if start_ts > end_ts:
        raise ValueError("start must not be after end")
    if lookback_sessions < 61:
        raise ValueError("lookback_sessions must be at least 61")
    if forward_sessions < 0:
        raise ValueError("forward_sessions must be non-negative")

    cached = tuple(
        _read_cached_field(paths, field)
        for field in ("open_auction_px", "close_px", "amount")
    )
    scan_start = start_ts - pd.Timedelta(days=max(40, 2 * lookback_sessions))
    scan_end = end_ts + pd.Timedelta(days=max(20, 3 * forward_sessions + 7))
    recent = _read_recent_parts(paths.recent_parts_dir, scan_start, scan_end)
    combined: list[pd.DataFrame] = []
    for old, new in zip(cached, recent):
        if not new.empty and not old.empty:
            old = old.drop(index=old.index.intersection(new.index), errors="ignore")
        combined.append(_combine_daily_frames([old, new]))
    if not any(not frame.empty for frame in combined):
        raise FileNotFoundError(
            "No daily open/close/amount inputs were found under "
            f"{paths.astock_root}"
        )

    opens, closes, amounts = _align_wide(*combined)
    dates = opens.index
    start_pos = max(int(dates.searchsorted(start_ts, side="left")) - lookback_sessions, 0)
    end_pos = min(int(dates.searchsorted(end_ts, side="right")) + forward_sessions, len(dates))
    if start_pos >= end_pos:
        raise ValueError(f"No daily sessions overlap {start_ts.date()}..{end_ts.date()}")
    keep = dates[start_pos:end_pos]
    return opens.loc[keep], closes.loc[keep], amounts.loc[keep]


def _rank_pct(values: pd.Series, mask: pd.Series | None = None) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if mask is not None:
        numeric = numeric.where(mask.fillna(False))
    return numeric.rank(method="average", pct=True, ascending=True)


def _row_mean_at_least(frame: pd.DataFrame, columns: Sequence[str], minimum: int) -> pd.Series:
    count = frame[list(columns)].notna().sum(axis=1)
    return frame[list(columns)].mean(axis=1, skipna=True).where(count >= minimum)


def _finite_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    with np.errstate(divide="ignore", invalid="ignore"):
        out = numerator / denominator.where(denominator.abs() > 1e-12)
    return out.replace([np.inf, -np.inf], np.nan)


def build_base_panel(
    decision_dates: Iterable[object],
    opens: pd.DataFrame,
    closes: pd.DataFrame,
    amounts: pd.DataFrame,
    *,
    liquidity_floor: float = 20_000_000.0,
    gap_low: float = -0.06,
    gap_high: float = 0.03,
    candidate_fraction: float = 0.03,
) -> pd.DataFrame:
    """Build the T-open base panel and frozen F1 multi-shape score.

    ``pv_score_raw`` is the maximum of four equal-weight shape scores;
    ``pv_score`` is its same-day percentile within the liquid, history-complete
    and gap-gated universe.  ``pv_candidate`` is exactly the best
    ``ceil(candidate_fraction * N)`` rows (ties break by symbol), which avoids
    silently changing pool size when many scores tie.
    """

    if not 0 < candidate_fraction <= 1:
        raise ValueError("candidate_fraction must be in (0, 1]")
    if liquidity_floor < 0:
        raise ValueError("liquidity_floor must be non-negative")
    opens, closes, amounts = _align_wide(opens, closes, amounts)
    dates = opens.index
    symbols = closes.columns
    decision_index = pd.DatetimeIndex(
        sorted({_timestamp(d, name="decision_date") for d in decision_dates})
    ).difference(REJECTED_DATES)
    missing_dates = decision_index.difference(dates)
    if len(missing_dates):
        raise ValueError(
            "Decision dates are absent from the aligned trading calendar: "
            + ", ".join(str(d.date()) for d in missing_dates)
        )

    returns = closes.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    rows: list[pd.DataFrame] = []
    for decision_date in decision_index:
        pos = int(dates.get_loc(decision_date))
        if pos < 61:
            continue
        previous_close = closes.iloc[pos - 1]
        frame = pd.DataFrame(index=symbols)
        frame["decision_date"] = decision_date
        frame["symbol"] = symbols.astype(str)
        frame["feature_cutoff_date"] = dates[pos - 1]
        frame["previous_close"] = previous_close
        frame["decision_open"] = opens.iloc[pos]
        frame["gap"] = frame["decision_open"] / previous_close - 1.0
        frame["mom5"] = previous_close / closes.iloc[pos - 6] - 1.0
        frame["mom10"] = previous_close / closes.iloc[pos - 11] - 1.0
        frame["mom20"] = previous_close / closes.iloc[pos - 21] - 1.0
        frame["mom60"] = previous_close / closes.iloc[pos - 61] - 1.0
        frame["prev_ret"] = previous_close / closes.iloc[pos - 2] - 1.0
        frame["ma20_dist"] = (
            previous_close / closes.iloc[pos - 20 : pos].mean(axis=0) - 1.0
        )
        frame["drawdown20"] = (
            previous_close / closes.iloc[pos - 20 : pos].max(axis=0) - 1.0
        )
        frame["vol20"] = returns.iloc[pos - 20 : pos].std(axis=0)
        frame["amt_ratio20"] = (
            amounts.iloc[pos - 1]
            / amounts.iloc[pos - 21 : pos - 1].mean(axis=0)
        )
        frame["liq20"] = amounts.iloc[pos - 20 : pos].mean(axis=0)
        frame["vol_pre5"] = returns.iloc[pos - 6 : pos - 1].std(axis=0)
        frame["amt_pre5"] = amounts.iloc[pos - 6 : pos - 1].mean(axis=0)
        frame["amt20_prev"] = amounts.iloc[pos - 21 : pos - 1].mean(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            frame["acc5_20"] = (
                np.log1p(frame["mom5"]) / 5.0
                - np.log1p(frame["mom20"]) / 20.0
            )
        frame["vol_contract"] = frame["vol_pre5"] / frame["vol20"]
        frame["amt_contract"] = frame["amt_pre5"] / frame["amt20_prev"]
        frame["amount_expand"] = amounts.iloc[pos - 1] / frame["amt_pre5"]
        frame = frame.replace([np.inf, -np.inf], np.nan)

        code = frame["symbol"].str.slice(0, 6)
        exchange_ok = (
            frame["symbol"].str.endswith((".SH", ".SZ"))
            & code.str.fullmatch(r"[036]\d{5}", na=False)
        )
        complete_history = frame[["mom60", "gap", "liq20"]].notna().all(axis=1)
        frame["eligible_for_ranking"] = (
            exchange_ok & complete_history & frame["liq20"].ge(liquidity_floor)
        )
        eligible = frame["eligible_for_ranking"]
        percentile_features = (
            *RAW_FEATURES,
            "acc5_20",
            "vol_contract",
            "amt_contract",
            "amount_expand",
        )
        for column in percentile_features:
            frame[f"{column}_pct"] = _rank_pct(frame[column], eligible)

        frame["pv_acceleration_raw"] = _row_mean_at_least(
            frame,
            ["mom10_pct", "vol20_pct", "liq20_pct", "amt_ratio20_pct"],
            2,
        )
        pull_components = pd.DataFrame(
            {
                "mom60": frame["mom60_pct"],
                "deep_pullback": 1.0 - frame["drawdown20_pct"],
                "acceleration": frame["acc5_20_pct"],
                "liquidity": frame["liq20_pct"],
            },
            index=frame.index,
        )
        frame["pv_pullback_raw"] = pull_components.mean(axis=1).where(
            pull_components.notna().sum(axis=1) >= 2
        )
        reset_components = pd.DataFrame(
            {
                "shock": frame["prev_ret_pct"],
                "weak_open": 1.0 - frame["gap_pct"],
                "volatility": frame["vol20_pct"],
                "liquidity": frame["liq20_pct"],
            },
            index=frame.index,
        )
        frame["pv_reset_raw"] = reset_components.mean(axis=1).where(
            reset_components.notna().sum(axis=1) >= 2
        )
        vcp_components = pd.DataFrame(
            {
                "vol_contraction": 1.0 - frame["vol_contract_pct"],
                "amount_contraction": 1.0 - frame["amt_contract_pct"],
                "amount_expansion": frame["amount_expand_pct"],
                "ma20_position": frame["ma20_dist_pct"],
            },
            index=frame.index,
        )
        frame["pv_vcp_raw"] = vcp_components.mean(axis=1).where(
            vcp_components.notna().sum(axis=1) >= 2
        )
        branch_columns = [
            "pv_acceleration_raw",
            "pv_pullback_raw",
            "pv_reset_raw",
            "pv_vcp_raw",
        ]
        frame["pv_score_raw"] = frame[branch_columns].max(axis=1, skipna=True)
        frame.loc[frame[branch_columns].notna().sum(axis=1).eq(0), "pv_score_raw"] = np.nan
        frame["pv_gap_eligible"] = eligible & frame["gap"].between(
            gap_low, gap_high, inclusive="both"
        )
        frame["pv_score"] = _rank_pct(frame["pv_score_raw"], frame["pv_gap_eligible"])

        # Preserve the earlier descriptive score as an audit bridge, not as F1.
        legacy_components = pd.DataFrame(
            {
                "mom20": 0.30 * frame["mom20_pct"],
                "mom60": 0.20 * frame["mom60_pct"],
                "amount": 0.15 * frame["amt_ratio20_pct"],
                "liquidity": 0.15 * frame["liq20_pct"],
                "volatility": 0.10 * frame["vol20_pct"],
                "weak_open": 0.10 * (1.0 - frame["gap_pct"]),
            },
            index=frame.index,
        )
        frame["legacy_fixed_score"] = legacy_components.sum(axis=1, min_count=6)
        frame["legacy_fixed_score_pct"] = _rank_pct(
            frame["legacy_fixed_score"], eligible
        )

        frame["pv_rank"] = pd.Series(pd.NA, index=frame.index, dtype="Int64")
        frame["pv_candidate"] = False
        ranked = frame.loc[
            frame["pv_gap_eligible"] & frame["pv_score_raw"].notna(),
            ["pv_score_raw", "symbol"],
        ].sort_values(["pv_score_raw", "symbol"], ascending=[False, True], kind="mergesort")
        if not ranked.empty:
            rank_values = pd.Series(
                np.arange(1, len(ranked) + 1, dtype=int), index=ranked.index
            )
            frame.loc[ranked.index, "pv_rank"] = rank_values.astype("Int64")
            n_select = max(1, int(math.ceil(candidate_fraction * len(ranked))))
            frame.loc[ranked.index[:n_select], "pv_candidate"] = True
        frame["eligible_count"] = int(eligible.sum())
        frame["pv_pool_count"] = int(frame["pv_gap_eligible"].sum())
        frame["pv_candidate_count"] = int(frame["pv_candidate"].sum())
        frame["base_asof_rule"] = "T-1 close/amount plus T open"
        frame["same_day_close_or_amount_used"] = False
        rows.append(frame.reset_index(drop=True))

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True).sort_values(
        ["decision_date", "symbol"], kind="mergesort"
    ).reset_index(drop=True)


def extend_statind_2026(
    closes: pd.DataFrame,
    existing_labels: pd.DataFrame | None = None,
    *,
    paths: ChoiceDataPaths | None = None,
    extension_start: object = "2026-01-01",
    n_clusters: int = 30,
    lookback: int = 120,
    min_valid: int = 80,
) -> pd.DataFrame:
    """Extend quarterly statistical-industry labels without writing a cache.

    Each completed quarter is clustered using only the last ``lookback`` daily
    returns through that quarter's final session; the labels are applied solely
    to sessions in the following quarter.  The random seed is the quarter
    offset from 2020Q1, matching :mod:`afactor.statind` even when the supplied
    close panel starts later than 2020.
    """

    if n_clusters < 2 or lookback < 20 or min_valid < 2 or min_valid > lookback:
        raise ValueError("invalid statind clustering parameters")
    from sklearn.cluster import KMeans

    closes = _normalize_wide(closes, name="closes")
    dates, symbols = closes.index, closes.columns
    extension_ts = _timestamp(extension_start, name="extension_start")
    if existing_labels is None:
        candidate = (paths or ChoiceDataPaths()).statind_labels_file
        if candidate.exists():
            existing_labels = pd.read_parquet(candidate)
    labels = pd.DataFrame(-1, index=dates, columns=symbols, dtype=np.int16)
    if existing_labels is not None and not existing_labels.empty:
        old = existing_labels.copy()
        old.index = pd.to_datetime(old.index, errors="coerce").normalize()
        old.columns = [_standard_symbol(c) for c in old.columns]
        old = old.loc[:, pd.Index(old.columns).notna()]
        old.columns = pd.Index(old.columns).astype(str)
        old = old.reindex(index=dates, columns=symbols)
        old_values = old.apply(pd.to_numeric, errors="coerce").to_numpy(float)
        label_values = labels.to_numpy(copy=True)
        take = np.isfinite(old_values) & (old_values >= 0)
        label_values[take] = old_values[take].astype(np.int16)
        labels = pd.DataFrame(
            label_values.astype(np.int16), index=dates, columns=symbols
        )

    returns = (closes / closes.shift(1) - 1.0).clip(-0.25, 0.25)
    periods = dates.to_period("Q")
    quarter0 = pd.Period("2020Q1", freq="Q")
    unique_periods = pd.Index(periods.unique()).sort_values()
    for period in unique_periods:
        next_period = period + 1
        target_mask = (periods == next_period) & (dates >= extension_ts)
        if not target_mask.any():
            continue
        source_positions = np.flatnonzero(periods == period)
        if source_positions.size == 0:
            continue
        source_pos = int(source_positions[-1])
        if source_pos + 1 < lookback:
            continue
        window = returns.iloc[source_pos - lookback + 1 : source_pos + 1]
        valid = window.notna().sum(axis=0).ge(min_valid)
        candidate_symbols = valid[valid].index
        if len(candidate_symbols) < n_clusters:
            continue
        matrix = window[candidate_symbols].to_numpy(dtype=np.float64)
        mean = np.nanmean(matrix, axis=0)
        std = np.nanstd(matrix, axis=0)
        usable = std > 1e-9
        if int(usable.sum()) < n_clusters:
            continue
        with np.errstate(invalid="ignore", divide="ignore"):
            standardized = (matrix - mean) / np.where(std > 1e-9, std, np.nan)
        observations = np.where(np.isfinite(standardized), standardized, 0.0).T
        random_state = int(period.ordinal - quarter0.ordinal)
        model = KMeans(
            n_clusters=n_clusters, n_init=4, random_state=random_state
        ).fit(observations[usable])
        values = np.full(len(candidate_symbols), -1, dtype=np.int16)
        values[usable] = model.labels_.astype(np.int16)
        target_dates = dates[target_mask]
        labels.loc[target_dates, candidate_symbols] = values[None, :]
    return labels.astype(np.int16)


def _candidate_rank(
    frame: pd.DataFrame,
    column: str,
    *,
    pool_col: str = "pv_candidate",
) -> pd.Series:
    if pool_col in frame.columns:
        pool = frame[pool_col].fillna(False).astype(bool)
    else:
        pool = frame.get("eligible_for_ranking", pd.Series(True, index=frame.index))
        pool = pool.fillna(False).astype(bool)
    values = pd.to_numeric(frame[column], errors="coerce").where(pool)
    return values.groupby(frame["decision_date"], sort=False).rank(
        method="average", pct=True
    )


def _rank_loo_against_group_means(
    data: pd.DataFrame,
    raw_column: str,
    metric_column: str,
    active: pd.Series,
    min_group_size: int,
) -> pd.Series:
    result = pd.Series(np.nan, index=data.index, dtype=float)
    for _, day_index in data.groupby("decision_date", sort=False).groups.items():
        day = data.loc[day_index]
        valid = (
            active.loc[day_index]
            & day["statind_label"].ge(0)
            & pd.to_numeric(day[metric_column], errors="coerce").notna()
        )
        reference = (
            day.loc[valid]
            .groupby("statind_label", sort=False)[metric_column]
            .agg(["mean", "count"])
        )
        reference = reference.loc[reference["count"].ge(min_group_size), "mean"]
        if reference.empty:
            continue
        for idx in day.index:
            value = data.at[idx, raw_column]
            label = data.at[idx, "statind_label"]
            if pd.isna(value) or pd.isna(label) or int(label) not in reference.index:
                continue
            peers = reference.copy()
            peers.loc[int(label)] = float(value)
            result.at[idx] = float(peers.rank(method="average", pct=True).loc[int(label)])
    return result


def _lead_days_for_decision(
    decision_date: pd.Timestamp,
    labels_at_decision: pd.Series,
    closes: pd.DataFrame,
    amounts: pd.DataFrame,
    *,
    min_group_size: int,
) -> pd.Series:
    returns = closes.pct_change(fill_method=None)
    history = returns.index[returns.index < decision_date][-20:]
    out = pd.Series(np.nan, index=closes.columns, dtype=float)
    if len(history) < 15:
        return out
    hits = pd.Series(0.0, index=closes.columns)
    valid_days = pd.Series(0.0, index=closes.columns)
    labels_at_decision = pd.to_numeric(labels_at_decision, errors="coerce").reindex(
        closes.columns
    )
    for date in history:
        values = returns.loc[date]
        active = values.notna() & amounts.loc[date].gt(0) & labels_at_decision.ge(0)
        group_size = active.groupby(labels_at_decision).transform("sum")
        valid = active & group_size.ge(min_group_size)
        ranks = values.where(valid).groupby(labels_at_decision).rank(
            method="average", pct=True
        )
        valid_days = valid_days.add(ranks.notna().astype(float), fill_value=0.0)
        hits = hits.add(ranks.ge(0.80).astype(float), fill_value=0.0)
    out = (hits / valid_days.replace(0, np.nan)).where(valid_days.ge(15))
    return out


def add_direction_and_leader(
    panel: pd.DataFrame,
    closes: pd.DataFrame,
    amounts: pd.DataFrame,
    *,
    statind_labels: pd.DataFrame | None = None,
    paths: ChoiceDataPaths | None = None,
    min_group_size: int = 10,
    candidate_pool_col: str = "pv_candidate",
) -> pd.DataFrame:
    """Attach PIT statistical direction (F2) and within-group leadership (F3)."""

    required = {
        "decision_date",
        "symbol",
        "eligible_for_ranking",
        "mom5",
        "mom10",
        "mom20",
        "mom60",
        "ma20_dist",
        "amt_ratio20",
        "liq20",
    }
    missing = sorted(required.difference(panel.columns))
    if missing:
        raise ValueError(f"panel is missing required columns: {missing}")
    if min_group_size < 2:
        raise ValueError("min_group_size must be at least 2")
    closes = _normalize_wide(closes, name="closes")
    amounts = _normalize_wide(amounts, name="amounts").reindex_like(closes)
    labels = (
        extend_statind_2026(closes, paths=paths)
        if statind_labels is None
        else statind_labels.copy()
    )
    labels.index = pd.to_datetime(labels.index, errors="coerce").normalize()
    labels.columns = [_standard_symbol(c) for c in labels.columns]
    labels = labels.loc[:, pd.Index(labels.columns).notna()]
    labels.columns = pd.Index(labels.columns).astype(str)
    labels = labels.reindex(index=closes.index, columns=closes.columns, fill_value=-1)

    result = panel.copy()
    result["decision_date"] = pd.to_datetime(result["decision_date"]).dt.normalize()
    result["symbol"] = result["symbol"].map(_standard_symbol).astype("string")
    label_lookup = _stack_all(labels).rename("statind_label")
    label_lookup.index.names = ["decision_date", "symbol"]
    key = pd.MultiIndex.from_frame(result[["decision_date", "symbol"]])
    result["statind_label"] = pd.to_numeric(
        label_lookup.reindex(key).to_numpy(), errors="coerce"
    )
    result.loc[result["statind_label"].lt(0), "statind_label"] = np.nan

    active = result["eligible_for_ranking"].fillna(False).astype(bool)
    group_key = [result["decision_date"], result["statind_label"]]
    active_count = active.astype(int).groupby(group_key, dropna=False).transform("sum")
    result["statind_group_size"] = active_count.where(result["statind_label"].notna())
    for date, idx in result.groupby("decision_date", sort=False).groups.items():
        values = pd.to_numeric(result.loc[idx, "amt_ratio20"], errors="coerce")
        valid_values = values.loc[active.loc[idx]].dropna()
        if valid_values.empty:
            result.loc[idx, "_amt_accel_winsor"] = np.nan
        else:
            lo, hi = valid_values.quantile([0.01, 0.99])
            result.loc[idx, "_amt_accel_winsor"] = values.clip(lo, hi)
    result["_breadth20"] = result["ma20_dist"].gt(0).astype(float)

    direction_specs = {
        "f2_group_mom5_raw": "mom5",
        "f2_group_mom20_raw": "mom20",
        "f2_group_breadth20_raw": "_breadth20",
        "f2_group_amt_accel_raw": "_amt_accel_winsor",
    }
    for raw_column, metric_column in direction_specs.items():
        metric = pd.to_numeric(result[metric_column], errors="coerce")
        valid = active & result["statind_label"].notna() & metric.notna()
        sums = metric.where(valid).groupby(group_key, dropna=False).transform("sum")
        counts = valid.astype(int).groupby(group_key, dropna=False).transform("sum")
        own = metric.where(valid, 0.0)
        own_count = valid.astype(int)
        loo = (sums - own) / (counts - own_count).replace(0, np.nan)
        result[raw_column] = loo.where(active_count.ge(min_group_size))
        pct_column = raw_column.replace("_raw", "_pct")
        result[pct_column] = _rank_loo_against_group_means(
            result,
            raw_column,
            metric_column,
            active,
            min_group_size,
        )

    f2_pct_columns = [column.replace("_raw", "_pct") for column in direction_specs]
    f2_count = result[f2_pct_columns].notna().sum(axis=1)
    result["f2_component_count"] = f2_count
    result["f2_direction_score"] = result[f2_pct_columns].mean(axis=1).where(
        f2_count >= 2
    )
    result["f2_missing"] = f2_count.lt(2)
    result.loc[result["f2_missing"], "f2_direction_score"] = 0.5

    lead_frames: list[pd.Series] = []
    for decision_date in sorted(result["decision_date"].dropna().unique()):
        date = pd.Timestamp(decision_date)
        if date not in labels.index:
            continue
        lead = _lead_days_for_decision(
            date,
            labels.loc[date],
            closes,
            amounts,
            min_group_size=min_group_size,
        )
        lead.index = pd.MultiIndex.from_product([[date], lead.index])
        lead_frames.append(lead)
    if lead_frames:
        lead_lookup = pd.concat(lead_frames)
        result["f3_lead_days20_raw"] = lead_lookup.reindex(key).to_numpy()
    else:
        result["f3_lead_days20_raw"] = np.nan

    leader_specs = {
        "f3_liq20_pct": "liq20",
        "f3_mom10_pct": "mom10",
        "f3_mom60_pct": "mom60",
        "f3_ma20_dist_pct": "ma20_dist",
        "f3_lead_days20_pct": "f3_lead_days20_raw",
    }
    valid_group = active & result["statind_group_size"].ge(min_group_size)
    for pct_column, raw_column in leader_specs.items():
        values = pd.to_numeric(result[raw_column], errors="coerce").where(valid_group)
        result[pct_column] = values.groupby(group_key, dropna=False).rank(
            method="average", pct=True
        )
    f3_pct_columns = list(leader_specs)
    f3_count = result[f3_pct_columns].notna().sum(axis=1)
    result["f3_component_count"] = f3_count
    result["f3_leader_score"] = result[f3_pct_columns].mean(axis=1).where(f3_count >= 3)
    result["f3_missing"] = f3_count.lt(3)
    result.loc[result["f3_missing"], "f3_leader_score"] = 0.5

    result["f2_direction_candidate_pct"] = _candidate_rank(
        result, "f2_direction_score", pool_col=candidate_pool_col
    )
    result["f3_leader_candidate_pct"] = _candidate_rank(
        result, "f3_leader_score", pool_col=candidate_pool_col
    )
    result["statind_asof_rule"] = "previous-quarter 120-session clustering"
    return result.drop(columns=["_amt_accel_winsor", "_breadth20"])


def _load_optional_frame(
    value: pd.DataFrame | str | Path | None,
    default_path: Path,
) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value.copy()
    path = Path(value) if value is not None else default_path
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def _next_trading_session(
    raw_dates: pd.Series,
    trading_dates: pd.DatetimeIndex,
) -> pd.Series:
    parsed = _date_series(raw_dates)
    calendar = trading_dates.to_numpy(dtype="datetime64[ns]")
    positions = np.searchsorted(calendar, parsed.to_numpy(dtype="datetime64[ns]"), side="right")
    valid = parsed.notna().to_numpy() & (positions < len(calendar))
    out = np.full(len(parsed), np.datetime64("NaT"), dtype="datetime64[ns]")
    out[valid] = calendar[positions[valid]]
    return pd.Series(out, index=raw_dates.index)


def _source_bounds(frame: pd.DataFrame, date_column: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    if frame.empty or date_column not in frame.columns:
        return pd.NaT, pd.NaT
    dates = _date_series(frame[date_column]).dropna()
    if dates.empty:
        return pd.NaT, pd.NaT
    return dates.min(), dates.max()


def _coverage_by_decision(
    decisions: pd.DatetimeIndex,
    trading_dates: pd.DatetimeIndex,
    source_start: pd.Timestamp,
    source_cutoff: pd.Timestamp,
    *,
    calendar_day_source: bool,
    lookback_sessions: int,
) -> pd.DataFrame:
    rows = []
    for decision in decisions:
        position = int(trading_dates.searchsorted(decision, side="left"))
        if calendar_day_source:
            required = decision - pd.Timedelta(days=1)
        elif position > 0:
            required = trading_dates[position - 1]
        else:
            required = pd.NaT
        history_pos = max(position - lookback_sessions, 0)
        history_start = trading_dates[history_pos] if len(trading_dates) else pd.NaT
        covered = (
            pd.notna(required)
            and pd.notna(source_cutoff)
            and pd.Timestamp(source_cutoff) >= pd.Timestamp(required)
            and pd.notna(source_start)
            and pd.Timestamp(source_start) <= pd.Timestamp(history_start)
        )
        rows.append((decision, required, history_start, bool(covered)))
    return pd.DataFrame(
        rows,
        columns=["decision_date", "required_date", "history_start", "covered"],
    ).set_index("decision_date")


def _latest_decay(
    events: pd.DataFrame,
    decision: pd.Timestamp,
    symbols: pd.Index,
    value_column: str,
    trading_dates: pd.DatetimeIndex,
    *,
    max_age: int = 60,
) -> pd.Series:
    out = pd.Series(0.0, index=symbols, dtype=float)
    if events.empty:
        return out
    usable = events.loc[
        events["eff_date"].notna() & events["eff_date"].le(decision)
    ].sort_values(["symbol", "eff_date", "raw_date"], kind="mergesort")
    if usable.empty:
        return out
    latest = usable.dropna(subset=[value_column]).drop_duplicates("symbol", keep="last")
    latest = latest.loc[latest["symbol"].astype(str).isin(out.index)]
    if latest.empty:
        return out
    decision_pos = int(trading_dates.get_indexer([decision])[0])
    event_pos = trading_dates.get_indexer(pd.DatetimeIndex(latest["eff_date"]))
    age = decision_pos - event_pos
    value = pd.to_numeric(latest[value_column], errors="coerce").clip(-2.0, 2.0)
    decayed = value.to_numpy() * np.exp(-age / 40.0)
    decayed[(age < 0) | (age > max_age)] = 0.0
    # Event sources include B shares, delisted names, and newly listed symbols
    # that are not necessarily members of the supplied panel.  Filter before
    # assignment so those source-only names cannot enlarge the result or raise
    # a strict-label KeyError.
    out.loc[latest["symbol"].astype(str)] = decayed
    return out


def _prepare_reports(frame: pd.DataFrame, trading_dates: pd.DatetimeIndex) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["symbol", "raw_date", "eff_date", "org", "industry"])
    required = {"stockCode", "publishDate"}
    if not required.issubset(frame.columns):
        raise ValueError(f"reports is missing columns: {sorted(required - set(frame.columns))}")
    out = pd.DataFrame(index=frame.index)
    out["symbol"] = frame["stockCode"].map(_standard_symbol)
    out["raw_date"] = _date_series(frame["publishDate"])
    out["eff_date"] = _next_trading_session(frame["publishDate"], trading_dates)
    out["org"] = frame.get("orgCode", pd.Series(pd.NA, index=frame.index)).astype("string")
    out["industry"] = frame.get(
        "indvInduName", pd.Series(pd.NA, index=frame.index)
    ).astype("string").str.strip().replace("", pd.NA)
    return out.dropna(subset=["symbol", "raw_date", "eff_date"])


def _prepare_surprise_events(
    forecast: pd.DataFrame,
    express: pd.DataFrame,
    trading_dates: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fc_columns = ["symbol", "end_date", "raw_date", "eff_date", "fc_value", "np_mid"]
    if forecast.empty:
        fc = pd.DataFrame(columns=fc_columns)
    else:
        required = {"ts_code", "ann_date", "end_date", "p_change_min", "p_change_max"}
        if not required.issubset(forecast.columns):
            raise ValueError(
                f"forecast is missing columns: {sorted(required - set(forecast.columns))}"
            )
        fc = pd.DataFrame(index=forecast.index)
        fc["symbol"] = forecast["ts_code"].map(_standard_symbol)
        fc["end_date"] = _date_series(forecast["end_date"])
        fc["raw_date"] = _date_series(forecast["ann_date"])
        fc["eff_date"] = _next_trading_session(forecast["ann_date"], trading_dates)
        pmin = pd.to_numeric(forecast["p_change_min"], errors="coerce")
        pmax = pd.to_numeric(forecast["p_change_max"], errors="coerce")
        fc["fc_value"] = pd.concat([pmin, pmax], axis=1).mean(axis=1) / 100.0
        npmin = pd.to_numeric(
            forecast.get("net_profit_min", pd.Series(np.nan, index=forecast.index)),
            errors="coerce",
        )
        npmax = pd.to_numeric(
            forecast.get("net_profit_max", pd.Series(np.nan, index=forecast.index)),
            errors="coerce",
        )
        fc["np_mid"] = pd.concat([npmin, npmax], axis=1).mean(axis=1)
        fc = fc.dropna(subset=["symbol", "raw_date", "eff_date"])

    ex_columns = ["symbol", "end_date", "raw_date", "eff_date", "ex_value", "n_income"]
    if express.empty:
        ex = pd.DataFrame(columns=ex_columns)
    else:
        required = {"ts_code", "ann_date", "end_date", "yoy_net_profit"}
        if not required.issubset(express.columns):
            raise ValueError(
                f"express is missing columns: {sorted(required - set(express.columns))}"
            )
        ex = pd.DataFrame(index=express.index)
        ex["symbol"] = express["ts_code"].map(_standard_symbol)
        ex["end_date"] = _date_series(express["end_date"])
        ex["raw_date"] = _date_series(express["ann_date"])
        ex["eff_date"] = _next_trading_session(express["ann_date"], trading_dates)
        ex["ex_value"] = pd.to_numeric(express["yoy_net_profit"], errors="coerce") / 100.0
        ex["n_income"] = pd.to_numeric(
            express.get("n_income", pd.Series(np.nan, index=express.index)), errors="coerce"
        )
        ex = ex.dropna(subset=["symbol", "raw_date", "eff_date"])

    exfc_rows: list[dict[str, object]] = []
    if not fc.empty and not ex.empty:
        grouped_fc = {
            key: group.sort_values("raw_date")
            for key, group in fc.dropna(subset=["end_date", "np_mid"]).groupby(
                ["symbol", "end_date"], sort=False
            )
        }
        for row in ex.dropna(subset=["end_date", "n_income"]).itertuples(index=False):
            history = grouped_fc.get((row.symbol, row.end_date))
            if history is None:
                continue
            prior = history.loc[history["raw_date"].lt(row.raw_date)]
            if prior.empty:
                continue
            forecast_row = prior.iloc[-1]
            denominator = abs(float(forecast_row["np_mid"]))
            if not np.isfinite(denominator) or denominator <= 1e-6:
                continue
            exfc_rows.append(
                {
                    "symbol": row.symbol,
                    "raw_date": row.raw_date,
                    "eff_date": row.eff_date,
                    "exfc_value": np.clip(
                        (float(row.n_income) - float(forecast_row["np_mid"])) / denominator,
                        -2.0,
                        2.0,
                    ),
                }
            )
    exfc = pd.DataFrame(
        exfc_rows, columns=["symbol", "raw_date", "eff_date", "exfc_value"]
    )
    return fc, ex, exfc


def _prepare_toplist(frame: pd.DataFrame, trading_dates: pd.DatetimeIndex) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["symbol", "raw_date", "eff_date", "net_ratio", "on_list"])
    required = {"ts_code", "trade_date", "net_amount_ratio"}
    if not required.issubset(frame.columns):
        raise ValueError(f"toplist is missing columns: {sorted(required - set(frame.columns))}")
    out = pd.DataFrame(index=frame.index)
    out["symbol"] = frame["ts_code"].map(_standard_symbol)
    out["raw_date"] = _date_series(frame["trade_date"])
    out["net_ratio"] = pd.to_numeric(frame["net_amount_ratio"], errors="coerce")
    out = out.dropna(subset=["symbol", "raw_date"])
    out = out.groupby(["symbol", "raw_date"], as_index=False).agg(
        net_ratio=("net_ratio", "mean")
    )
    out["eff_date"] = _next_trading_session(out["raw_date"], trading_dates)
    out["on_list"] = 1.0
    return out.dropna(subset=["eff_date"])


def add_event_attention(
    panel: pd.DataFrame,
    trading_dates: Sequence[object],
    *,
    reports: pd.DataFrame | str | Path | None = None,
    forecast: pd.DataFrame | str | Path | None = None,
    express: pd.DataFrame | str | Path | None = None,
    toplist: pd.DataFrame | str | Path | None = None,
    paths: ChoiceDataPaths | None = None,
    source_cutoffs: Mapping[str, object] | None = None,
    candidate_pool_col: str = "pv_candidate",
) -> pd.DataFrame:
    """Attach event/attention subfactors with explicit per-source coverage.

    A covered source with no event contributes raw zero.  If the source file is
    stale relative to the decision, the corresponding raw component is NaN and
    remains distinguishable from a true zero.  Calendar-dated reports and
    earnings announcements must cover through ``T-1`` calendar day (weekend and
    holiday releases can affect the next open); top-list data must cover through
    the previous trading session.
    """

    required = {"decision_date", "symbol"}
    if not required.issubset(panel.columns):
        raise ValueError(f"panel is missing required columns: {sorted(required - set(panel.columns))}")
    paths = paths or ChoiceDataPaths()
    reports_raw = _load_optional_frame(reports, paths.reports_file)
    forecast_raw = _load_optional_frame(forecast, paths.forecast_file)
    express_raw = _load_optional_frame(express, paths.express_file)
    toplist_raw = _load_optional_frame(toplist, paths.toplist_file)
    source_cutoffs = dict(source_cutoffs or {})

    calendar = pd.DatetimeIndex(pd.to_datetime(list(trading_dates), errors="coerce"))
    calendar = calendar.dropna().tz_localize(None).normalize().difference(REJECTED_DATES)
    calendar = calendar.drop_duplicates().sort_values()
    result = panel.copy()
    result["decision_date"] = pd.to_datetime(result["decision_date"]).dt.normalize()
    result["symbol"] = result["symbol"].map(_standard_symbol).astype("string")
    decisions = pd.DatetimeIndex(sorted(result["decision_date"].dropna().unique()))
    absent = decisions.difference(calendar)
    if len(absent):
        raise ValueError(
            "trading_dates is missing decision sessions: "
            + ", ".join(str(d.date()) for d in absent)
        )

    source_specs = {
        "reports": (reports_raw, "publishDate", True, 20),
        "forecast": (forecast_raw, "ann_date", True, 60),
        "express": (express_raw, "ann_date", True, 60),
        "toplist": (toplist_raw, "trade_date", False, 20),
    }
    coverage: dict[str, pd.DataFrame] = {}
    bounds: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {}
    for source, (frame, date_column, calendar_day, lookback) in source_specs.items():
        start, inferred_cutoff = _source_bounds(frame, date_column)
        cutoff = (
            _timestamp(source_cutoffs[source], name=f"{source}_cutoff")
            if source in source_cutoffs and pd.notna(source_cutoffs[source])
            else inferred_cutoff
        )
        bounds[source] = (start, cutoff)
        coverage[source] = _coverage_by_decision(
            decisions,
            calendar,
            start,
            cutoff,
            calendar_day_source=calendar_day,
            lookback_sessions=lookback,
        )
        result[f"event_{source}_source_start"] = result["decision_date"].map(
            {date: start for date in decisions}
        )
        result[f"event_{source}_cutoff"] = result["decision_date"].map(
            {date: cutoff for date in decisions}
        )
        result[f"event_{source}_required_date"] = result["decision_date"].map(
            coverage[source]["required_date"]
        )
        result[f"event_{source}_covered"] = (
            result["decision_date"].map(coverage[source]["covered"]).fillna(False).astype(bool)
        )

    prepared_reports = _prepare_reports(reports_raw, calendar)
    prepared_fc, prepared_ex, prepared_exfc = _prepare_surprise_events(
        forecast_raw, express_raw, calendar
    )
    prepared_toplist = _prepare_toplist(toplist_raw, calendar)

    raw_columns = [
        "f4_report_org20_raw",
        "f4_stock_attention_accel_raw",
        "f4_sector_attention_accel_raw",
        "f4_surprise_decay_raw",
        "f4_toplist_net20_raw",
        "f4_toplist_count20_raw",
    ]
    for column in raw_columns:
        result[column] = np.nan
    result["f4_report_count5"] = np.nan
    result["f4_report_count_prev15"] = np.nan
    result["f4_forecast_decay"] = np.nan
    result["f4_express_decay"] = np.nan
    result["f4_express_vs_forecast_decay"] = np.nan

    for decision, idx in result.groupby("decision_date", sort=False).groups.items():
        decision = pd.Timestamp(decision)
        symbols = pd.Index(result.loc[idx, "symbol"].astype(str))
        position = int(calendar.get_loc(decision))
        window20 = calendar[max(0, position - 19) : position + 1]
        window5 = calendar[max(0, position - 4) : position + 1]
        previous15 = calendar[max(0, position - 19) : max(0, position - 4)]

        if bool(coverage["reports"].at[decision, "covered"]):
            reports20 = prepared_reports.loc[prepared_reports["eff_date"].isin(window20)]
            reports5 = prepared_reports.loc[prepared_reports["eff_date"].isin(window5)]
            reports15 = prepared_reports.loc[prepared_reports["eff_date"].isin(previous15)]
            org20 = reports20.groupby("symbol")["org"].nunique(dropna=True)
            count5 = reports5.groupby("symbol").size()
            count15 = reports15.groupby("symbol").size()
            org_values = symbols.to_series(index=idx).map(org20).fillna(0.0)
            c5_values = symbols.to_series(index=idx).map(count5).fillna(0.0)
            c15_values = symbols.to_series(index=idx).map(count15).fillna(0.0)
            result.loc[idx, "f4_report_org20_raw"] = np.log1p(org_values).to_numpy()
            result.loc[idx, "f4_report_count5"] = c5_values.to_numpy()
            result.loc[idx, "f4_report_count_prev15"] = c15_values.to_numpy()
            result.loc[idx, "f4_stock_attention_accel_raw"] = np.log(
                (1.0 + c5_values) / (1.0 + c15_values / 3.0)
            ).to_numpy()

            known = prepared_reports.loc[
                prepared_reports["eff_date"].le(decision)
                & prepared_reports["industry"].notna()
            ].sort_values(["symbol", "eff_date", "raw_date"], kind="mergesort")
            latest_industry = known.drop_duplicates("symbol", keep="last").set_index(
                "symbol"
            )["industry"]
            labelled_count = latest_industry.value_counts()
            c5_by_ind = reports5.dropna(subset=["industry"]).groupby("industry").size()
            c15_by_ind = reports15.dropna(subset=["industry"]).groupby("industry").size()
            industry = symbols.to_series(index=idx).map(latest_industry)
            denom = industry.map(labelled_count)
            rate5 = industry.map(c5_by_ind).fillna(0.0) / denom
            rate15 = industry.map(c15_by_ind).fillna(0.0) / denom
            sector_accel = np.log((1.0 + rate5) / (1.0 + rate15 / 3.0))
            result.loc[idx, "f4_sector_attention_accel_raw"] = sector_accel.to_numpy()

        forecast_covered = bool(coverage["forecast"].at[decision, "covered"])
        express_covered = bool(coverage["express"].at[decision, "covered"])
        decay_components: list[pd.Series] = []
        if forecast_covered:
            fc_decay = _latest_decay(
                prepared_fc,
                decision,
                symbols,
                "fc_value",
                calendar,
            )
            result.loc[idx, "f4_forecast_decay"] = fc_decay.to_numpy()
            decay_components.append(fc_decay)
        if express_covered:
            ex_decay = _latest_decay(
                prepared_ex,
                decision,
                symbols,
                "ex_value",
                calendar,
            )
            result.loc[idx, "f4_express_decay"] = ex_decay.to_numpy()
            decay_components.append(ex_decay)
        if forecast_covered and express_covered:
            exfc_decay = _latest_decay(
                prepared_exfc,
                decision,
                symbols,
                "exfc_value",
                calendar,
            )
            result.loc[idx, "f4_express_vs_forecast_decay"] = exfc_decay.to_numpy()
            decay_components.append(exfc_decay)
        if decay_components:
            surprise = pd.concat(decay_components, axis=1).mean(axis=1)
            result.loc[idx, "f4_surprise_decay_raw"] = surprise.to_numpy()

        if bool(coverage["toplist"].at[decision, "covered"]):
            recent = prepared_toplist.loc[prepared_toplist["eff_date"].isin(window20)]
            net20 = recent.groupby("symbol")["net_ratio"].sum()
            count20 = recent.groupby("symbol")["eff_date"].nunique()
            result.loc[idx, "f4_toplist_net20_raw"] = (
                symbols.to_series(index=idx).map(net20).fillna(0.0).to_numpy()
            )
            result.loc[idx, "f4_toplist_count20_raw"] = (
                symbols.to_series(index=idx).map(count20).fillna(0.0).to_numpy()
            )

    pct_columns: list[str] = []
    for raw_column in raw_columns:
        pct_column = raw_column.replace("_raw", "_pct")
        result[pct_column] = _candidate_rank(
            result, raw_column, pool_col=candidate_pool_col
        )
        pct_columns.append(pct_column)
    component_count = result[pct_columns].notna().sum(axis=1)
    result["f4_component_count"] = component_count
    result["f4_event_attention_score"] = result[pct_columns].mean(axis=1).where(
        component_count >= 3
    )
    result["f4_missing"] = component_count.lt(3)
    result.loc[result["f4_missing"], "f4_event_attention_score"] = 0.5
    result["f4_event_attention_candidate_pct"] = _candidate_rank(
        result, "f4_event_attention_score", pool_col=candidate_pool_col
    )
    result["event_asof_rule"] = "raw date maps to first strictly later trading session"
    return result


def _prepare_disclosed_history(
    events: pd.DataFrame,
    positions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    exposure_rows: list[pd.DataFrame] = []
    buys = pd.DataFrame(columns=["history_date", "symbol"])
    if not events.empty:
        required = {"security_code", "event_date_end", "status", "date_confidence"}
        if not required.issubset(events.columns):
            raise ValueError(f"events is missing columns: {sorted(required - set(events.columns))}")
        e = pd.DataFrame(index=events.index)
        e["symbol"] = events["security_code"].map(_standard_symbol)
        e["history_date"] = _date_series(events["event_date_end"])
        e["status"] = events["status"].astype("string").str.lower()
        e["date_confidence"] = events["date_confidence"].astype("string").str.upper()
        e["direction_confidence"] = events.get(
            "direction_confidence", pd.Series("L3", index=events.index)
        ).astype("string").str.upper()
        e["event_type"] = events.get(
            "event_type", pd.Series("", index=events.index)
        ).astype("string").str.lower()
        allowed = (
            e["status"].isin(["reported", "inferred"])
            & e["date_confidence"].isin(["L3", "L4"])
            & e["symbol"].notna()
            & e["history_date"].notna()
        )
        exposure_rows.append(e.loc[allowed, ["history_date", "symbol"]])
        is_buy = e["event_type"].isin(["buy", "net_buy", "buy_to_target"])
        direction_ok = e["direction_confidence"].isin(["L3", "L4"])
        buys = e.loc[allowed & is_buy & direction_ok, ["history_date", "symbol"]]
    if not positions.empty:
        required = {"security_code", "trading_date", "confidence", "total_quantity_best"}
        if not required.issubset(positions.columns):
            raise ValueError(
                f"positions is missing columns: {sorted(required - set(positions.columns))}"
            )
        p = pd.DataFrame(index=positions.index)
        p["symbol"] = positions["security_code"].map(_standard_symbol)
        p["history_date"] = _date_series(positions["trading_date"])
        quantity = pd.to_numeric(positions["total_quantity_best"], errors="coerce")
        confidence = positions["confidence"].astype("string").str.upper()
        valid = (
            p["symbol"].notna()
            & p["history_date"].notna()
            & quantity.gt(0)
            & confidence.eq("L4")
        )
        exposure_rows.append(p.loc[valid, ["history_date", "symbol"]])
    history = (
        pd.concat(exposure_rows, ignore_index=True)
        .drop_duplicates(["history_date", "symbol"])
        .sort_values(["history_date", "symbol"], kind="mergesort")
        if exposure_rows
        else pd.DataFrame(columns=["history_date", "symbol"])
    )
    return history.reset_index(drop=True), buys.drop_duplicates().reset_index(drop=True)


def _history_session_position(calendar: pd.DatetimeIndex, value: pd.Timestamp) -> int:
    exact = calendar.get_indexer([value])[0]
    if exact >= 0:
        return int(exact)
    return max(int(calendar.searchsorted(value, side="right")) - 1, 0)


def _topk_mean_and_count(values: np.ndarray, k: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Row-wise mean of the largest finite values and their available count."""

    if values.ndim != 2:
        raise ValueError("values must be a two-dimensional array")
    finite = np.isfinite(values)
    count = np.minimum(finite.sum(axis=1), k).astype(np.int16)
    if values.shape[1] == 0:
        return np.zeros(values.shape[0], dtype=float), count
    safe = np.where(finite, values, -np.inf)
    width = min(k, values.shape[1])
    top = np.partition(safe, values.shape[1] - width, axis=1)[:, -width:]
    top = np.where(np.isfinite(top), top, np.nan)
    total = np.nansum(top, axis=1)
    mean = np.divide(
        total,
        count,
        out=np.zeros(values.shape[0], dtype=float),
        where=count > 0,
    )
    return mean, count


def _pairwise_corr_topk(
    history_returns: pd.DataFrame,
    candidate_symbols: pd.Index,
    familiar_symbols: pd.Index,
    *,
    min_periods: int = 40,
    k: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized pairwise-complete Pearson correlations, then row top-k."""

    candidates = history_returns.reindex(columns=candidate_symbols).to_numpy(float)
    familiar = history_returns.reindex(columns=familiar_symbols).to_numpy(float)
    if candidates.size == 0 or familiar.size == 0:
        return np.zeros(len(candidate_symbols)), np.zeros(len(candidate_symbols), dtype=np.int16)
    candidate_valid = np.isfinite(candidates)
    familiar_valid = np.isfinite(familiar)
    x = np.where(candidate_valid, candidates, 0.0)
    y = np.where(familiar_valid, familiar, 0.0)
    mx = candidate_valid.astype(np.float64)
    my = familiar_valid.astype(np.float64)

    # Every N x M quantity below is computed on each pair's common sessions.
    n = mx.T @ my
    sum_x = x.T @ my
    sum_y = mx.T @ y
    sum_x2 = (x * x).T @ my
    sum_y2 = mx.T @ (y * y)
    sum_xy = x.T @ y
    with np.errstate(divide="ignore", invalid="ignore"):
        covariance = sum_xy - sum_x * sum_y / n
        variance_x = sum_x2 - sum_x * sum_x / n
        variance_y = sum_y2 - sum_y * sum_y / n
        correlation = covariance / np.sqrt(variance_x * variance_y)
    correlation[(n < min_periods) | (variance_x <= 0) | (variance_y <= 0)] = np.nan
    familiar_position = {symbol: j for j, symbol in enumerate(familiar_symbols)}
    for i, symbol in enumerate(candidate_symbols):
        j = familiar_position.get(symbol)
        if j is not None:
            correlation[i, j] = np.nan
    return _topk_mean_and_count(correlation, k=k)


def _shape_similarity_topk(
    current: np.ndarray,
    history: np.ndarray,
    *,
    min_common: int = 8,
    k: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Broadcast ten-dimensional percentile-shape similarity, then row top-k."""

    if current.ndim != 2 or history.ndim != 2 or current.shape[1] != history.shape[1]:
        raise ValueError("current and history must be 2-D arrays with equal width")
    if history.shape[0] == 0:
        return np.zeros(current.shape[0]), np.zeros(current.shape[0], dtype=np.int16)
    current_valid = np.isfinite(current)
    history_valid = np.isfinite(history)
    common = current_valid[:, None, :] & history_valid[None, :, :]
    count = common.sum(axis=2)
    distance = np.where(
        common,
        np.abs(current[:, None, :] - history[None, :, :]),
        0.0,
    ).sum(axis=2)
    similarity = 1.0 - np.divide(
        distance,
        count,
        out=np.full(distance.shape, np.nan, dtype=float),
        where=count > 0,
    )
    similarity[count < min_common] = np.nan
    return _topk_mean_and_count(similarity, k=k)


def add_familiarity(
    panel: pd.DataFrame,
    opens: pd.DataFrame,
    closes: pd.DataFrame,
    amounts: pd.DataFrame,
    *,
    events: pd.DataFrame | str | Path | None = None,
    positions: pd.DataFrame | str | Path | None = None,
    paths: ChoiceDataPaths | None = None,
    historical_buy_features: pd.DataFrame | None = None,
    candidate_pool_col: str = "pv_candidate",
) -> pd.DataFrame:
    """Attach the five PIT familiarity components (F5).

    Only previously disclosed reported/inferred L3/L4 events and prior L4
    positive holdings are visible.  Realized P&L and later winner labels are
    never read.  If daily history is too short to reconstruct an old buy's
    shape, that buy is omitted from shape similarity and the coverage count
    makes the limitation explicit.
    """

    required = {"decision_date", "symbol"}
    if not required.issubset(panel.columns):
        raise ValueError(f"panel is missing required columns: {sorted(required - set(panel.columns))}")
    paths = paths or ChoiceDataPaths()
    events_frame = _load_optional_frame(events, paths.events_file)
    positions_frame = _load_optional_frame(positions, paths.positions_file)
    history, buys = _prepare_disclosed_history(events_frame, positions_frame)
    opens, closes, amounts = _align_wide(opens, closes, amounts)
    calendar = closes.index

    result = panel.copy()
    result["decision_date"] = pd.to_datetime(result["decision_date"]).dt.normalize()
    result["symbol"] = result["symbol"].map(_standard_symbol).astype("string")
    for column in [
        "f5_exact_recency_raw",
        "f5_exact_freq252_raw",
        "f5_cluster_familiarity_raw",
        "f5_corr_familiarity_raw",
        "f5_shape_similarity_raw",
    ]:
        result[column] = np.nan
    result["f5_history_count"] = 0
    result["f5_corr_count"] = 0
    result["f5_shape_match_count"] = 0

    if historical_buy_features is None and not buys.empty:
        eligible_buy_dates = sorted(
            set(buys.loc[buys["history_date"].isin(opens.index), "history_date"])
        )
        historical_buy_features = build_base_panel(
            eligible_buy_dates,
            opens,
            closes,
            amounts,
        )
    historical_buy_features = (
        historical_buy_features.copy()
        if historical_buy_features is not None
        else pd.DataFrame()
    )
    historical_shapes = pd.DataFrame()
    if not historical_buy_features.empty:
        hf = historical_buy_features.copy()
        hf["decision_date"] = pd.to_datetime(hf["decision_date"]).dt.normalize()
        hf["symbol"] = hf["symbol"].map(_standard_symbol).astype("string")
        buy_keys = buys.rename(columns={"history_date": "decision_date"})
        historical_shapes = buy_keys.merge(
            hf[["decision_date", "symbol", *SHAPE_VECTOR_PERCENTILES]],
            on=["decision_date", "symbol"],
            how="inner",
        ).drop_duplicates(["decision_date", "symbol"])

    returns = closes.pct_change(fill_method=None)
    for decision, idx in result.groupby("decision_date", sort=False).groups.items():
        decision = pd.Timestamp(decision)
        decision_pos = int(calendar.searchsorted(decision, side="left"))
        prior = history.loc[history["history_date"].lt(decision)]
        result.loc[idx, "f5_history_count"] = int(len(prior))
        symbols = result.loc[idx, "symbol"].astype(str)
        if prior.empty:
            result.loc[idx, [
                "f5_exact_recency_raw",
                "f5_exact_freq252_raw",
                "f5_cluster_familiarity_raw",
                "f5_corr_familiarity_raw",
                "f5_shape_similarity_raw",
            ]] = 0.0
            continue

        last_seen = prior.groupby("symbol")["history_date"].max()
        last_positions = last_seen.map(lambda d: _history_session_position(calendar, d))
        distance = symbols.map(last_positions)
        recency = np.exp(-(decision_pos - distance) / 60.0).where(distance.notna(), 0.0)
        result.loc[idx, "f5_exact_recency_raw"] = recency.to_numpy()

        start252_pos = max(decision_pos - 252, 0)
        start252 = calendar[start252_pos] if len(calendar) else decision
        recent_history = prior.loc[prior["history_date"].ge(start252)]
        frequency = recent_history.groupby("symbol")["history_date"].nunique()
        result.loc[idx, "f5_exact_freq252_raw"] = np.log1p(
            symbols.map(frequency).fillna(0.0)
        ).to_numpy()

        if "statind_label" in result.columns:
            day = result.loc[idx]
            seen_symbols = set(prior["symbol"].astype(str))
            active = day.get(
                "eligible_for_ranking", pd.Series(True, index=day.index)
            ).fillna(False).astype(bool)
            group_size = day.loc[active].groupby("statind_label")["symbol"].nunique()
            familiar_by_group = (
                day.loc[active & day["symbol"].astype(str).isin(seen_symbols)]
                .groupby("statind_label")["symbol"]
                .nunique()
            )
            familiarity = day["statind_label"].map(familiar_by_group).fillna(0.0) / day[
                "statind_label"
            ].map(group_size)
            result.loc[idx, "f5_cluster_familiarity_raw"] = familiarity.fillna(0.0).to_numpy()
        else:
            result.loc[idx, "f5_cluster_familiarity_raw"] = 0.0

        history_dates = returns.index[returns.index < decision][-60:]
        familiar_symbols = pd.Index(sorted(set(prior["symbol"]).intersection(returns.columns)))
        corr_values = np.zeros(len(idx), dtype=float)
        corr_counts = np.zeros(len(idx), dtype=np.int16)
        if len(history_dates) >= 40 and len(familiar_symbols):
            history_returns = returns.loc[history_dates]
            corr_values, corr_counts = _pairwise_corr_topk(
                history_returns,
                pd.Index(symbols.to_numpy()),
                familiar_symbols,
                min_periods=40,
                k=3,
            )
        result.loc[idx, "f5_corr_familiarity_raw"] = corr_values
        result.loc[idx, "f5_corr_count"] = corr_counts

        shape_values = np.zeros(len(idx), dtype=float)
        shape_counts = np.zeros(len(idx), dtype=np.int16)
        prior_shapes = historical_shapes.loc[
            historical_shapes["decision_date"].lt(decision)
        ] if not historical_shapes.empty else historical_shapes
        if not prior_shapes.empty:
            history_matrix = prior_shapes[list(SHAPE_VECTOR_PERCENTILES)].apply(
                pd.to_numeric, errors="coerce"
            ).to_numpy(float)
            current_matrix = result.loc[idx, list(SHAPE_VECTOR_PERCENTILES)].apply(
                pd.to_numeric, errors="coerce"
            ).to_numpy(float)
            shape_values, shape_counts = _shape_similarity_topk(
                current_matrix,
                history_matrix,
                min_common=8,
                k=3,
            )
        result.loc[idx, "f5_shape_similarity_raw"] = shape_values
        result.loc[idx, "f5_shape_match_count"] = shape_counts

    raw_columns = [
        "f5_exact_recency_raw",
        "f5_exact_freq252_raw",
        "f5_cluster_familiarity_raw",
        "f5_corr_familiarity_raw",
        "f5_shape_similarity_raw",
    ]
    pct_columns: list[str] = []
    for raw_column in raw_columns:
        pct_column = raw_column.replace("_raw", "_pct")
        result[pct_column] = _candidate_rank(
            result, raw_column, pool_col=candidate_pool_col
        )
        pct_columns.append(pct_column)
    count = result[pct_columns].notna().sum(axis=1)
    result["f5_component_count"] = count
    result["f5_familiarity_score"] = result[pct_columns].mean(axis=1).where(count >= 3)
    result["f5_missing"] = count.lt(3)
    result.loc[result["f5_missing"], "f5_familiarity_score"] = 0.5
    result["f5_familiarity_candidate_pct"] = _candidate_rank(
        result, "f5_familiarity_score", pool_col=candidate_pool_col
    )
    result["familiarity_asof_rule"] = "disclosures and holdings strictly before T"
    return result


def add_forward_returns(
    panel: pd.DataFrame,
    opens: pd.DataFrame,
    *,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
) -> pd.DataFrame:
    """Attach open[T+h] / open[T] - 1 outcomes, never feature columns.

    The rejected sessions are absent from the horizon calendar.  Missing future
    opens remain NaN rather than being treated as zero returns.
    """

    required = {"decision_date", "symbol"}
    if not required.issubset(panel.columns):
        raise ValueError(f"panel is missing required columns: {sorted(required - set(panel.columns))}")
    resolved = tuple(int(h) for h in horizons)
    if not resolved or any(h <= 0 for h in resolved) or len(set(resolved)) != len(resolved):
        raise ValueError("horizons must be unique positive integers")
    opens = _normalize_wide(opens, name="opens")
    result = panel.copy()
    result["decision_date"] = pd.to_datetime(result["decision_date"]).dt.normalize()
    result["symbol"] = result["symbol"].map(_standard_symbol).astype("string")
    key = pd.MultiIndex.from_frame(result[["decision_date", "symbol"]])
    entry = _stack_all(opens).reindex(key).to_numpy(dtype=float)
    result["forward_entry_open"] = entry
    for horizon in resolved:
        exit_prices = _stack_all(opens.shift(-horizon)).reindex(key).to_numpy(dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            forward = exit_prices / entry - 1.0
        forward[~np.isfinite(forward)] = np.nan
        result[f"fwd_open_to_open_d{horizon}"] = forward
        exit_date_map = {
            date: opens.index[pos + horizon]
            for pos, date in enumerate(opens.index)
            if pos + horizon < len(opens.index)
        }
        result[f"fwd_exit_date_d{horizon}"] = result["decision_date"].map(exit_date_map)
    result["forward_return_definition"] = "open[T+h] / open[T] - 1"
    return result


__all__ = [
    "ChoiceDataPaths",
    "DEFAULT_HORIZONS",
    "RAW_FEATURES",
    "REJECTED_DATES",
    "SHAPE_VECTOR_PERCENTILES",
    "add_direction_and_leader",
    "add_event_attention",
    "add_familiarity",
    "add_forward_returns",
    "build_base_panel",
    "extend_statind_2026",
    "load_daily_inputs",
]
