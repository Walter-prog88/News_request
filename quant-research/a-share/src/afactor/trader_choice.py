"""Deterministic research helpers for the trader's final-choice layer.

The functions in this module deliberately stop short of machine learning.  They
construct auditable candidate universes, transform factors cross-sectionally,
and evaluate whether disclosed buys rank near the top.  Candidate membership is
computed from panel features only: labels are never injected into a pool.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


_DAY_NS = 86_400_000_000_000


@dataclass(frozen=True)
class ChoiceConfig:
    """Configuration for candidate construction and deterministic evaluation.

    The legacy P150 thresholds are intentionally ``init=False``.  This freezes
    the historical rule while leaving the broader, hypothesis-generating parent
    pool configurable.
    """

    date_col: str = "decision_date"
    symbol_col: str = "symbol"
    eligible_col: str = "eligible_for_ranking"
    score_col: str = "choice_score"
    rank_ks: tuple[int, ...] = (5, 10, 20, 50, 100)
    random_seed: int = 20260831
    random_basket_size: int = 5
    random_basket_repeats: int = 1
    controls_per_label: int = 3
    style_bins: int = 5
    categorical_style_columns: tuple[str, ...] = (
        "industry",
        "sw_industry",
        "fine_industry",
        "sector",
        "theme",
    )

    # Frozen legacy_P150 rule.
    legacy_mom10_min: float = field(default=0.70, init=False)
    legacy_vol20_min: float = field(default=0.70, init=False)
    legacy_liq20_min: float = field(default=0.90, init=False)
    legacy_gap_low: float = field(default=-0.06, init=False)
    legacy_gap_high: float = field(default=0.03, init=False)

    # Broad parent-pool hypotheses.  These are intentionally high recall.
    parent_liq_min: float = 0.75
    parent_gap_low: float = -0.08
    parent_gap_high: float = 0.05
    parent_accel_mom10_min: float = 0.55
    parent_accel_vol20_min: float = 0.55
    parent_rebound_mom60_min: float = 0.60
    parent_rebound_mom20_max: float = 0.55
    parent_rebound_mom5_min: float = 0.55
    parent_active_ma20_min: float = 0.55
    parent_active_prev_ret_min: float = 0.55
    parent_active_amount_min: float = 0.60

    def __post_init__(self) -> None:
        ks = tuple(int(k) for k in self.rank_ks)
        if not ks or any(k <= 0 for k in ks):
            raise ValueError("rank_ks must contain positive integers")
        if len(set(ks)) != len(ks):
            raise ValueError("rank_ks must not contain duplicates")
        if self.random_basket_size <= 0:
            raise ValueError("random_basket_size must be positive")
        if self.random_basket_repeats <= 0:
            raise ValueError("random_basket_repeats must be positive")
        if self.controls_per_label <= 0:
            raise ValueError("controls_per_label must be positive")
        if self.style_bins <= 0:
            raise ValueError("style_bins must be positive")
        object.__setattr__(self, "rank_ks", ks)


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], where: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{where} is missing required columns: {missing}")


def _normalize_dates(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, errors="coerce").dt.normalize()


def _bool_series(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values.dtype):
        return values.fillna(False).astype(bool)
    if pd.api.types.is_numeric_dtype(values.dtype):
        return pd.to_numeric(values, errors="coerce").fillna(0).ne(0)
    normalized = values.astype("string").str.strip().str.lower()
    return normalized.isin({"1", "true", "t", "yes", "y"}).fillna(False)


def _prepare_panel(panel: pd.DataFrame, config: ChoiceConfig, where: str) -> pd.DataFrame:
    _require_columns(panel, [config.date_col, config.symbol_col], where)
    result = panel.copy()
    result[config.date_col] = _normalize_dates(result[config.date_col])
    result[config.symbol_col] = result[config.symbol_col].astype("string").str.strip()
    if result[config.date_col].isna().any():
        raise ValueError(f"{where} contains invalid {config.date_col} values")
    if result[config.symbol_col].isna().any() or result[config.symbol_col].eq("").any():
        raise ValueError(f"{where} contains invalid {config.symbol_col} values")
    duplicated = result.duplicated([config.date_col, config.symbol_col], keep=False)
    if duplicated.any():
        examples = result.loc[duplicated, [config.date_col, config.symbol_col]].head(5)
        raise ValueError(
            f"{where} must be unique by ({config.date_col}, {config.symbol_col}); "
            f"examples={examples.to_dict('records')}"
        )
    return result


def _standard_symbol(code: str) -> str | pd.NA:
    if len(code) != 6 or not code.isdigit():
        return pd.NA
    if code.startswith("6"):
        return f"{code}.SH"
    if code.startswith(("0", "3")):
        return f"{code}.SZ"
    if code.startswith(("4", "8", "9")):
        return f"{code}.BJ"
    return pd.NA


def load_choice_labels(
    events_path: str | Path,
    config: ChoiceConfig = ChoiceConfig(),
) -> pd.DataFrame:
    """Load the primary L4, exact-day, numeric ordinary-stock buy labels.

    Returns only the stable research schema: ``decision_date``, ``symbol``,
    ``label_confidence``, and ``event_id``.  Interval dates, non-buy events,
    non-numeric securities, ETFs/funds, and non-L4 observations are excluded.
    """

    events = pd.read_csv(events_path, dtype={"security_code": "string"})
    required = [
        "event_id",
        "event_type",
        "security_code",
        "event_date_start",
        "event_date_end",
        "date_confidence",
        "direction_confidence",
    ]
    _require_columns(events, required, "events")

    code = events["security_code"].astype("string").str.strip()
    start = _normalize_dates(events["event_date_start"])
    end = _normalize_dates(events["event_date_end"])
    mask = (
        events["event_type"].astype("string").str.strip().str.lower().eq("buy")
        & code.str.fullmatch(r"\d{6}", na=False)
        & start.notna()
        & start.eq(end)
        & events["date_confidence"].astype("string").str.upper().eq("L4")
        & events["direction_confidence"].astype("string").str.upper().eq("L4")
    )
    if "notes" in events.columns:
        instrument_note = events["notes"].astype("string")
        mask &= ~instrument_note.str.contains(r"ETF|基金", case=False, na=False)

    labels = events.loc[mask, ["event_id"]].copy()
    labels[config.date_col] = start.loc[mask]
    labels[config.symbol_col] = code.loc[mask].map(_standard_symbol).astype("string")
    labels["label_confidence"] = "L4"
    labels["event_id"] = labels["event_id"].astype("string")
    labels = labels.dropna(subset=[config.date_col, config.symbol_col, "event_id"])
    labels = labels.drop_duplicates(
        [config.date_col, config.symbol_col, "event_id"], keep="first"
    )
    labels = labels.sort_values(
        [config.date_col, config.symbol_col, "event_id"], kind="mergesort"
    ).reset_index(drop=True)
    return labels[
        [config.date_col, config.symbol_col, "label_confidence", "event_id"]
    ]


def assign_candidate_pools(
    panel: pd.DataFrame,
    config: ChoiceConfig = ChoiceConfig(),
) -> pd.DataFrame:
    """Assign the full, frozen legacy, and broad multi-shape candidate pools.

    Pool membership depends only on contemporaneous panel features.  This API
    intentionally accepts no labels and does not inspect label-like columns.
    """

    required = [
        config.eligible_col,
        "gap",
        "mom5_pct",
        "mom10_pct",
        "mom20_pct",
        "mom60_pct",
        "ma20_dist_pct",
        "prev_ret_pct",
        "vol20_pct",
        "amt_ratio20_pct",
        "liq20_pct",
    ]
    result = _prepare_panel(panel, config, "panel")
    _require_columns(result, required, "panel")

    eligible = _bool_series(result[config.eligible_col])
    numeric = {}
    for column in required[1:]:
        numeric[column] = pd.to_numeric(result[column], errors="coerce")

    result["full_eligible"] = eligible
    result["legacy_P150"] = (
        eligible
        & numeric["mom10_pct"].ge(config.legacy_mom10_min)
        & numeric["vol20_pct"].ge(config.legacy_vol20_min)
        & numeric["liq20_pct"].ge(config.legacy_liq20_min)
        & numeric["gap"].between(
            config.legacy_gap_low, config.legacy_gap_high, inclusive="both"
        )
    )

    common_parent = (
        eligible
        & numeric["liq20_pct"].ge(config.parent_liq_min)
        & numeric["gap"].between(
            config.parent_gap_low, config.parent_gap_high, inclusive="both"
        )
    )
    result["parent_shape_acceleration"] = (
        common_parent
        & numeric["mom10_pct"].ge(config.parent_accel_mom10_min)
        & numeric["vol20_pct"].ge(config.parent_accel_vol20_min)
    )
    result["parent_shape_pullback_rebound"] = (
        common_parent
        & numeric["mom60_pct"].ge(config.parent_rebound_mom60_min)
        & numeric["mom20_pct"].le(config.parent_rebound_mom20_max)
        & numeric["mom5_pct"].ge(config.parent_rebound_mom5_min)
    )
    result["parent_shape_active_breakout"] = (
        common_parent
        & numeric["ma20_dist_pct"].ge(config.parent_active_ma20_min)
        & (
            numeric["prev_ret_pct"].ge(config.parent_active_prev_ret_min)
            | numeric["amt_ratio20_pct"].ge(config.parent_active_amount_min)
        )
    )
    result["high_recall_parent"] = result[
        [
            "legacy_P150",
            "parent_shape_acceleration",
            "parent_shape_pullback_rebound",
            "parent_shape_active_breakout",
        ]
    ].any(axis=1)
    return result


def add_same_day_percentiles(
    panel: pd.DataFrame,
    factor_columns: Sequence[str],
    config: ChoiceConfig = ChoiceConfig(),
    *,
    pool_col: str = "high_recall_parent",
    factor_directions: Mapping[str, int] | None = None,
    prefix: str = "pct__",
) -> pd.DataFrame:
    """Add cross-sectional factor percentiles within each event day and pool.

    Direction ``+1`` means larger is preferred; ``-1`` means smaller is
    preferred.  Non-pool rows retain missing percentiles.
    """

    columns = tuple(dict.fromkeys(str(column) for column in factor_columns))
    if not columns:
        raise ValueError("factor_columns must not be empty")
    result = _prepare_panel(panel, config, "panel")
    _require_columns(result, [pool_col, *columns], "panel")
    directions = dict(factor_directions or {})
    pool = _bool_series(result[pool_col])

    for column in columns:
        direction = int(directions.get(column, 1))
        if direction not in (-1, 1):
            raise ValueError(f"factor direction for {column!r} must be +1 or -1")
        values = pd.to_numeric(result[column], errors="coerce").where(pool)
        percentile = values.groupby(result[config.date_col], sort=False).rank(
            method="average",
            pct=True,
            ascending=(direction == 1),
        )
        result[f"{prefix}{column}"] = percentile.where(pool)
    return result


def score_factor_families(
    panel: pd.DataFrame,
    factor_families: Mapping[str, Sequence[str]],
    config: ChoiceConfig = ChoiceConfig(),
    *,
    pool_col: str = "high_recall_parent",
    factor_directions: Mapping[str, int] | None = None,
) -> pd.DataFrame:
    """Score equal-weight factors within families, then equal-weight families."""

    if not factor_families:
        raise ValueError("factor_families must not be empty")
    normalized: dict[str, tuple[str, ...]] = {}
    for raw_name, raw_columns in factor_families.items():
        name = str(raw_name).strip()
        columns = tuple(dict.fromkeys(str(column) for column in raw_columns))
        if not name:
            raise ValueError("factor family names must not be empty")
        if name in normalized:
            raise ValueError(f"duplicate factor family name: {name!r}")
        if not columns:
            raise ValueError(f"factor family {name!r} must contain factors")
        normalized[name] = columns

    all_factors = tuple(
        dict.fromkeys(column for columns in normalized.values() for column in columns)
    )
    result = add_same_day_percentiles(
        panel,
        all_factors,
        config,
        pool_col=pool_col,
        factor_directions=factor_directions,
    )
    family_score_columns = []
    for name, columns in normalized.items():
        percentile_columns = [f"pct__{column}" for column in columns]
        family_score = f"family_score__{name}"
        family_count = f"family_factor_count__{name}"
        result[family_score] = result[percentile_columns].mean(axis=1, skipna=True)
        result[family_count] = result[percentile_columns].notna().sum(axis=1)
        result.loc[result[family_count].eq(0), family_score] = np.nan
        family_score_columns.append(family_score)

    pool = _bool_series(result[pool_col])
    result[config.score_col] = result[family_score_columns].mean(axis=1, skipna=True)
    result["score_family_count"] = result[family_score_columns].notna().sum(axis=1)
    result.loc[~pool | result["score_family_count"].eq(0), config.score_col] = np.nan
    return result


def _prepare_labels(labels: pd.DataFrame, config: ChoiceConfig) -> pd.DataFrame:
    required = [config.date_col, config.symbol_col, "label_confidence", "event_id"]
    _require_columns(labels, required, "labels")
    result = labels[required].copy()
    result[config.date_col] = _normalize_dates(result[config.date_col])
    result[config.symbol_col] = result[config.symbol_col].astype("string").str.strip()
    result["event_id"] = result["event_id"].astype("string")
    if result[config.date_col].isna().any():
        raise ValueError(f"labels contains invalid {config.date_col} values")
    if result[config.symbol_col].isna().any() or result[config.symbol_col].eq("").any():
        raise ValueError(f"labels contains invalid {config.symbol_col} values")
    return result.sort_values(
        [config.date_col, config.symbol_col, "event_id"], kind="mergesort"
    ).reset_index(drop=True)


def attach_selection_status(
    panel: pd.DataFrame,
    labels: pd.DataFrame,
    config: ChoiceConfig = ChoiceConfig(),
) -> pd.DataFrame:
    """Mark exact disclosed buys as ``selected`` and every other row unselected."""

    result = _prepare_panel(panel, config, "panel")
    prepared_labels = _prepare_labels(labels, config)
    keys = prepared_labels[[config.date_col, config.symbol_col]].drop_duplicates()
    selected_index = pd.MultiIndex.from_frame(keys)
    panel_index = pd.MultiIndex.from_frame(
        result[[config.date_col, config.symbol_col]]
    )
    result["selection_status"] = np.where(
        panel_index.isin(selected_index), "selected", "unselected"
    )
    return result


def _resolve_ks(config: ChoiceConfig, ks: Sequence[int] | None) -> tuple[int, ...]:
    resolved = config.rank_ks if ks is None else tuple(int(k) for k in ks)
    if not resolved or any(k <= 0 for k in resolved):
        raise ValueError("ks must contain positive integers")
    if len(set(resolved)) != len(resolved):
        raise ValueError("ks must not contain duplicates")
    return tuple(sorted(resolved))


def _ndcg_at_k(ranked_symbols: Sequence[str], relevant: set[str], k: int) -> float:
    hits = np.fromiter(
        (1.0 if symbol in relevant else 0.0 for symbol in ranked_symbols[:k]),
        dtype=float,
    )
    if hits.size:
        discounts = np.log2(np.arange(2, hits.size + 2, dtype=float))
        dcg = float(np.sum(hits / discounts))
    else:
        dcg = 0.0
    ideal_hits = min(len(relevant), k)
    if ideal_hits == 0:
        return np.nan
    ideal_discounts = np.log2(np.arange(2, ideal_hits + 2, dtype=float))
    idcg = float(np.sum(1.0 / ideal_discounts))
    return dcg / idcg


def evaluate_ranking(
    scored_panel: pd.DataFrame,
    labels: pd.DataFrame,
    config: ChoiceConfig = ChoiceConfig(),
    *,
    pool_col: str = "high_recall_parent",
    score_col: str | None = None,
    ks: Sequence[int] | None = None,
) -> pd.DataFrame:
    """Evaluate disclosed buys on each event day under deterministic ranking.

    Ties are always resolved by score descending and symbol ascending.  Recall
    denominators include every disclosed buy, including labels outside the pool.
    """

    score_column = score_col or config.score_col
    rank_ks = _resolve_ks(config, ks)
    panel = _prepare_panel(scored_panel, config, "scored_panel")
    _require_columns(panel, [pool_col, score_column], "scored_panel")
    prepared_labels = _prepare_labels(labels, config)
    panel_score = pd.to_numeric(panel[score_column], errors="coerce")
    panel = panel.assign(_ranking_score=panel_score)
    records: list[dict[str, object]] = []

    for decision_date, day_labels in prepared_labels.groupby(config.date_col, sort=True):
        relevant = set(day_labels[config.symbol_col].dropna().astype(str).unique())
        day = panel.loc[panel[config.date_col].eq(decision_date)].copy()
        pool_mask = _bool_series(day[pool_col])
        pool_symbols = set(day.loc[pool_mask, config.symbol_col].astype(str))
        ranked = day.loc[pool_mask & day["_ranking_score"].notna()].sort_values(
            ["_ranking_score", config.symbol_col],
            ascending=[False, True],
            kind="mergesort",
        )
        ranked_symbols = ranked[config.symbol_col].astype(str).tolist()
        ranks = {symbol: rank for rank, symbol in enumerate(ranked_symbols, start=1)}
        relevant_ranks = sorted(ranks[symbol] for symbol in relevant if symbol in ranks)
        n_labels = len(relevant)
        n_in_pool = len(relevant & pool_symbols)
        n_scored = len(ranked_symbols)
        if relevant_ranks:
            mrr = 1.0 / relevant_ranks[0]
            if n_scored == 1:
                percentiles = [1.0]
            else:
                percentiles = [
                    1.0 - (rank - 1) / (n_scored - 1) for rank in relevant_ranks
                ]
            median_percentile = float(np.median(percentiles))
        else:
            mrr = 0.0
            median_percentile = np.nan

        record: dict[str, object] = {
            config.date_col: decision_date,
            "n_labels": n_labels,
            "n_pool": len(pool_symbols),
            "n_scored": n_scored,
            "n_labels_in_pool": n_in_pool,
            "pool_recall": n_in_pool / n_labels if n_labels else np.nan,
            "hit_day": bool(n_in_pool > 0),
            "full_basket": bool(n_labels > 0 and n_in_pool == n_labels),
            "mrr": mrr,
            "median_selected_percentile": median_percentile,
            "ranked_selected_symbols": "|".join(
                symbol for symbol in ranked_symbols if symbol in relevant
            ),
        }
        for k in rank_ks:
            top_k = set(ranked_symbols[:k])
            hits = len(relevant & top_k)
            record[f"recall_at_{k}"] = hits / n_labels if n_labels else np.nan
            record[f"hit_day_at_{k}"] = bool(hits > 0)
            record[f"full_basket_at_{k}"] = bool(n_labels > 0 and hits == n_labels)
            record[f"ndcg_at_{k}"] = _ndcg_at_k(ranked_symbols, relevant, k)
        record["ndcg"] = record[f"ndcg_at_{max(rank_ks)}"]
        records.append(record)

    return pd.DataFrame.from_records(records)


def _style_bucket_frame(
    panel: pd.DataFrame,
    style_columns: Sequence[str],
    config: ChoiceConfig,
    reference_mask: pd.Series,
    categorical_columns: set[str],
) -> tuple[pd.DataFrame, dict[str, str]]:
    buckets = pd.DataFrame(index=panel.index)
    output_names: dict[str, str] = {}
    for column in style_columns:
        output_name = f"style__{column}"
        output_names[column] = output_name
        series = panel[column]
        is_categorical = column in categorical_columns or not pd.api.types.is_numeric_dtype(
            series.dtype
        )
        if is_categorical:
            buckets[output_name] = series.astype("string").str.strip()
            continue
        values = pd.to_numeric(series, errors="coerce").where(reference_mask)
        percentiles = values.groupby(panel[config.date_col], sort=False).rank(
            method="average", pct=True
        )
        bucket = np.ceil(percentiles * config.style_bins).clip(1, config.style_bins)
        buckets[output_name] = bucket.astype("Int64")
    return buckets, output_names


def build_matched_controls(
    panel: pd.DataFrame,
    labels: pd.DataFrame,
    style_columns: Sequence[str],
    config: ChoiceConfig = ChoiceConfig(),
    *,
    pool_col: str = "high_recall_parent",
    categorical_style_columns: Sequence[str] | None = None,
    controls_per_label: int | None = None,
    random_seed: int | None = None,
) -> pd.DataFrame:
    """Build strict same-day, non-buy controls in identical style buckets.

    Matching never relaxes.  If the target is unavailable, has an incomplete
    style vector, or has too few controls, all requested control rows are NA.
    """

    styles = tuple(dict.fromkeys(str(column) for column in style_columns))
    if not styles:
        raise ValueError("style_columns must not be empty")
    n_controls = config.controls_per_label if controls_per_label is None else int(
        controls_per_label
    )
    if n_controls <= 0:
        raise ValueError("controls_per_label must be positive")
    seed = config.random_seed if random_seed is None else int(random_seed)

    prepared_panel = _prepare_panel(panel, config, "panel")
    _require_columns(prepared_panel, [pool_col, *styles], "panel")
    prepared_labels = _prepare_labels(labels, config)
    prepared_panel = attach_selection_status(prepared_panel, prepared_labels, config)
    pool = _bool_series(prepared_panel[pool_col])
    if "full_eligible" in prepared_panel.columns:
        reference = _bool_series(prepared_panel["full_eligible"])
    else:
        reference = pool
    categorical = set(config.categorical_style_columns)
    if categorical_style_columns is not None:
        categorical |= {str(column) for column in categorical_style_columns}
    buckets, output_names = _style_bucket_frame(
        prepared_panel, styles, config, reference, categorical
    )
    working = pd.concat([prepared_panel, buckets], axis=1)
    style_output_columns = [output_names[column] for column in styles]
    rng = np.random.default_rng(seed)
    records: list[dict[str, object]] = []

    for label in prepared_labels.itertuples(index=False, name=None):
        label_record = dict(zip(prepared_labels.columns, label))
        decision_date = label_record[config.date_col]
        selected_symbol = str(label_record[config.symbol_col])
        target = working.loc[
            working[config.date_col].eq(decision_date)
            & working[config.symbol_col].eq(selected_symbol)
        ]
        status = "matched"
        available = pd.DataFrame()
        style_values: dict[str, object] = {
            column: pd.NA for column in style_output_columns
        }
        if target.empty:
            status = "target_missing"
        else:
            target_row = target.iloc[0]
            style_values = {column: target_row[column] for column in style_output_columns}
            if any(pd.isna(value) for value in style_values.values()):
                status = "target_style_missing"
            else:
                candidate_mask = (
                    working[config.date_col].eq(decision_date)
                    & pool
                    & working["selection_status"].eq("unselected")
                )
                for column, value in style_values.items():
                    candidate_mask &= working[column].eq(value)
                available = working.loc[candidate_mask].sort_values(
                    config.symbol_col, kind="mergesort"
                )
                if len(available) < n_controls:
                    status = "insufficient_matches"

        if status == "matched":
            chosen_positions = rng.choice(len(available), size=n_controls, replace=False)
            chosen = available.iloc[chosen_positions]
            control_symbols: list[object] = chosen[config.symbol_col].astype(str).tolist()
        else:
            control_symbols = [pd.NA] * n_controls

        for control_rank, control_symbol in enumerate(control_symbols, start=1):
            record: dict[str, object] = {
                "event_id": label_record["event_id"],
                config.date_col: decision_date,
                "selected_symbol": selected_symbol,
                "label_confidence": label_record["label_confidence"],
                "control_rank": control_rank,
                "control_symbol": control_symbol,
                "match_status": status,
                "n_available_controls": len(available),
            }
            record.update(style_values)
            records.append(record)

    base_columns = [
        "event_id",
        config.date_col,
        "selected_symbol",
        "label_confidence",
        "control_rank",
        "control_symbol",
        "match_status",
        "n_available_controls",
        *style_output_columns,
    ]
    return pd.DataFrame.from_records(records, columns=base_columns)


def deterministic_random_basket(
    panel: pd.DataFrame,
    decision_dates: Iterable[object] | pd.DataFrame,
    config: ChoiceConfig = ChoiceConfig(),
    *,
    pool_col: str = "high_recall_parent",
    basket_size: int | None = None,
    n_baskets: int | None = None,
    seed: int | None = None,
) -> pd.DataFrame:
    """Draw reproducible same-day random baskets from a candidate pool."""

    size = config.random_basket_size if basket_size is None else int(basket_size)
    repeats = config.random_basket_repeats if n_baskets is None else int(n_baskets)
    resolved_seed = config.random_seed if seed is None else int(seed)
    if size <= 0 or repeats <= 0:
        raise ValueError("basket_size and n_baskets must be positive")

    prepared_panel = _prepare_panel(panel, config, "panel")
    _require_columns(prepared_panel, [pool_col], "panel")
    if isinstance(decision_dates, pd.DataFrame):
        _require_columns(decision_dates, [config.date_col], "decision_dates")
        raw_dates = decision_dates[config.date_col]
    else:
        raw_dates = pd.Series(list(decision_dates), dtype="object")
    dates = _normalize_dates(pd.Series(raw_dates)).dropna().drop_duplicates().sort_values()
    pool = _bool_series(prepared_panel[pool_col])
    records: list[dict[str, object]] = []

    for decision_date in dates:
        symbols = (
            prepared_panel.loc[
                prepared_panel[config.date_col].eq(decision_date) & pool,
                config.symbol_col,
            ]
            .astype(str)
            .sort_values(kind="mergesort")
            .tolist()
        )
        n_pool = len(symbols)
        n_draw = min(size, n_pool)
        day_code = int(pd.Timestamp(decision_date).value // _DAY_NS) & 0xFFFFFFFF
        for basket_id in range(1, repeats + 1):
            day_rng = np.random.default_rng(
                np.random.SeedSequence(
                    [resolved_seed & 0xFFFFFFFF, day_code, basket_id]
                )
            )
            if n_draw:
                positions = day_rng.choice(n_pool, size=n_draw, replace=False)
                chosen = [symbols[position] for position in positions]
                for basket_rank, symbol in enumerate(chosen, start=1):
                    records.append(
                        {
                            config.date_col: decision_date,
                            "basket_id": basket_id,
                            "basket_rank": basket_rank,
                            config.symbol_col: symbol,
                            "n_pool": n_pool,
                            "requested_basket_size": size,
                            "basket_complete": n_pool >= size,
                            "seed": resolved_seed,
                        }
                    )
            else:
                records.append(
                    {
                        config.date_col: decision_date,
                        "basket_id": basket_id,
                        "basket_rank": pd.NA,
                        config.symbol_col: pd.NA,
                        "n_pool": 0,
                        "requested_basket_size": size,
                        "basket_complete": False,
                        "seed": resolved_seed,
                    }
                )
    columns = [
        config.date_col,
        "basket_id",
        "basket_rank",
        config.symbol_col,
        "n_pool",
        "requested_basket_size",
        "basket_complete",
        "seed",
    ]
    return pd.DataFrame.from_records(records, columns=columns)


def evaluate_ablations(
    panel: pd.DataFrame,
    labels: pd.DataFrame,
    factor_families: Mapping[str, Sequence[str]],
    config: ChoiceConfig = ChoiceConfig(),
    *,
    pool_col: str = "high_recall_parent",
    factor_directions: Mapping[str, int] | None = None,
    ks: Sequence[int] | None = None,
) -> pd.DataFrame:
    """Evaluate the pool baseline, single families, cumulative families, and deletions."""

    if not factor_families:
        raise ValueError("factor_families must not be empty")
    names = list(factor_families)
    prepared_panel = _prepare_panel(panel, config, "panel")
    _require_columns(prepared_panel, [pool_col], "panel")

    def score_subset(subset: Sequence[str]) -> pd.DataFrame:
        if not subset:
            baseline = prepared_panel.copy()
            baseline[config.score_col] = np.where(
                _bool_series(baseline[pool_col]), 0.0, np.nan
            )
            return baseline
        selected = {name: factor_families[name] for name in subset}
        return score_factor_families(
            prepared_panel,
            selected,
            config,
            pool_col=pool_col,
            factor_directions=factor_directions,
        )

    experiments: list[tuple[str, str, list[str]]] = [
        ("baseline", "pool_only", [])
    ]
    experiments.extend(
        ("single_family", f"only_{name}", [name]) for name in names
    )
    experiments.extend(
        ("cumulative", f"through_{name}", names[: index + 1])
        for index, name in enumerate(names)
    )
    experiments.extend(
        (
            "leave_one_family_out",
            f"full_minus_{omitted}",
            [name for name in names if name != omitted],
        )
        for omitted in names
    )

    outputs = []
    for kind, experiment_name, subset in experiments:
        metrics = evaluate_ranking(
            score_subset(subset),
            labels,
            config,
            pool_col=pool_col,
            ks=ks,
        )
        metrics.insert(0, "families", "|".join(subset))
        metrics.insert(0, "experiment_name", experiment_name)
        metrics.insert(0, "experiment_kind", kind)
        outputs.append(metrics)
    return pd.concat(outputs, ignore_index=True, sort=False)


__all__ = [
    "ChoiceConfig",
    "load_choice_labels",
    "assign_candidate_pools",
    "add_same_day_percentiles",
    "score_factor_families",
    "attach_selection_status",
    "evaluate_ranking",
    "build_matched_controls",
    "deterministic_random_basket",
    "evaluate_ablations",
]
