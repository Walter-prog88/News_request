"""Structured industry/leader scores for the static-oracle experiment.

The function in this module deliberately has no buy-label, event, outcome, or
PnL argument.  It reads a frozen whitelist of features known to have a T-1
clock in :mod:`afactor.trader_choice_data` and
:mod:`afactor.trader_choice_industry`.  Extra columns such as ``is_buy``,
``event_id``, ``forward_return``, ``pnl``, same-day ``close`` or ``amount`` are
preserved in the returned frame but cannot participate in any score.

Three taxonomies are evaluated in parallel when present:

* analyst-report PIT fine industry (``report_industry_pit`` or the canonical
  ``report_fine_industry_pit`` alias);
* current/static Shenwan fine industry; and
* current/static Shenwan L1 industry.

The latter two are explicitly oracle classifications.  They are useful for
testing whether a known industry/industrial-chain structure could explain the
trader's choices, not as a historically reproducible membership source.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


# This is the complete numerical source whitelist.  No fuzzy matching or
# prefix-based discovery is used, so label/event/outcome fields cannot leak in.
STRUCTURED_SOURCE_COLUMNS: tuple[str, ...] = (
    "mom5",
    "mom10",
    "mom20",
    "mom60",
    "ma20_dist",
    "amt_ratio20",
    "liq20",
    "db_turnover_rate_t1",
    "db_turnover_rate_f_t1",
    "db_volume_ratio_t1",
    "db_circ_mv_t1",
    "sw_l1_ret5_t1",
    "sw_l1_ret20_t1",
    "sw_l1_amount_ratio20_t1",
    "sw_l1_amount_accel5_20_t1",
)

_MOMENTUM_COLUMNS = ("mom5", "mom20", "mom60", "ma20_dist")
_LEADER_MOMENTUM_COLUMNS = ("mom10", "mom20", "mom60", "ma20_dist")
_AMOUNT_COLUMNS = ("amt_ratio20",)
_LEADER_COLUMNS = (
    "liq20",
    "db_turnover_rate_t1",
    "db_turnover_rate_f_t1",
    "db_volume_ratio_t1",
    "db_circ_mv_t1",
)
_SW_INDEX_DIRECTION_COLUMNS = ("sw_l1_ret5_t1", "sw_l1_ret20_t1")
_SW_INDEX_AMOUNT_COLUMNS = (
    "sw_l1_amount_ratio20_t1",
    "sw_l1_amount_accel5_20_t1",
)


def _date_series(values: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce", utc=True)
    return parsed.dt.tz_convert(None).dt.normalize()


def _bool_series(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values.dtype):
        return values.fillna(False).astype(bool)
    if pd.api.types.is_numeric_dtype(values.dtype):
        return pd.to_numeric(values, errors="coerce").fillna(0).ne(0)
    normalized = values.astype("string").str.strip().str.lower()
    return normalized.isin({"1", "true", "t", "yes", "y"}).fillna(False)


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )


def _row_mean(frame: pd.DataFrame, columns: Sequence[str], minimum: int = 1) -> pd.Series:
    available = [column for column in columns if column in frame.columns]
    if not available:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    numeric = frame[available].apply(pd.to_numeric, errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    return numeric.mean(axis=1).where(numeric.notna().sum(axis=1).ge(minimum))


def _rank_groups(values: pd.Series) -> pd.Series:
    """Equal-weight percentile ranks over distinct groups, not stock rows."""

    return pd.to_numeric(values, errors="coerce").rank(
        method="average", pct=True, ascending=True
    )


def _rank_within_groups(
    values: pd.Series,
    date: pd.Series,
    taxonomy: pd.Series,
    valid_group: pd.Series,
) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").where(valid_group)
    return numeric.groupby([date, taxonomy], dropna=False, sort=False).rank(
        method="average", pct=True, ascending=True
    )


def _prepare_panel(panel: pd.DataFrame) -> pd.DataFrame:
    required = {"decision_date", "symbol"}
    missing = sorted(required.difference(panel.columns))
    if missing:
        raise ValueError(f"panel is missing required columns: {missing}")
    result = panel.copy()
    result["decision_date"] = _date_series(result["decision_date"])
    result["symbol"] = result["symbol"].astype("string").str.strip()
    if result[["decision_date", "symbol"]].isna().any(axis=None):
        raise ValueError("panel contains invalid decision_date or symbol")
    if result.duplicated(["decision_date", "symbol"], keep=False).any():
        raise ValueError("panel must be unique by (decision_date, symbol)")
    result["_structured_original_order"] = np.arange(len(result), dtype=np.int64)
    return result


def _active_mask(frame: pd.DataFrame) -> pd.Series:
    if "eligible_for_ranking" not in frame.columns:
        return pd.Series(True, index=frame.index, dtype=bool)
    return _bool_series(frame["eligible_for_ranking"])


def _taxonomy_specs(frame: pd.DataFrame) -> list[tuple[str, str, bool]]:
    specs: list[tuple[str, str, bool]] = []
    if "report_industry_pit" in frame.columns:
        specs.append(("report", "report_industry_pit", False))
    elif "report_fine_industry_pit" in frame.columns:
        specs.append(("report", "report_fine_industry_pit", False))
    if "sw_fine_static_oracle" in frame.columns:
        specs.append(("sw_fine", "sw_fine_static_oracle", True))
    if "sw_l1_static_oracle" in frame.columns:
        specs.append(("sw_l1", "sw_l1_static_oracle", True))
    return specs


def _direction_inputs(frame: pd.DataFrame, *, include_sw_index: bool) -> pd.DataFrame:
    momentum = _row_mean(frame, _MOMENTUM_COLUMNS, minimum=1)
    breadth_components: list[pd.Series] = []
    for column in ("mom5", "mom20", "ma20_dist"):
        values = _numeric(frame, column)
        breadth_components.append(values.gt(0).where(values.notna()).astype(float))
    breadth = pd.concat(breadth_components, axis=1).mean(axis=1, skipna=True)
    breadth = breadth.where(pd.concat(breadth_components, axis=1).notna().any(axis=1))
    amount = _row_mean(frame, _AMOUNT_COLUMNS, minimum=1)

    if include_sw_index:
        index_momentum = _row_mean(frame, _SW_INDEX_DIRECTION_COLUMNS, minimum=1)
        momentum = pd.concat([momentum, index_momentum], axis=1).mean(
            axis=1, skipna=True
        )
        index_amount = _row_mean(frame, _SW_INDEX_AMOUNT_COLUMNS, minimum=1)
        amount = pd.concat([amount, index_amount], axis=1).mean(axis=1, skipna=True)
    return pd.DataFrame(
        {
            "direction_strength": momentum,
            "breadth": breadth,
            "amount_acceleration": amount,
        },
        index=frame.index,
    )


def _add_taxonomy_scores(
    frame: pd.DataFrame,
    *,
    short_name: str,
    taxonomy_column: str,
    oracle: bool,
    active: pd.Series,
    min_group_size: int,
) -> tuple[pd.DataFrame, str, str]:
    taxonomy = frame[taxonomy_column].astype("string").str.strip().replace("", pd.NA)
    key = [frame["decision_date"], taxonomy]
    group_size = active.astype(int).groupby(key, dropna=False, sort=False).transform("sum")
    valid_group = active & taxonomy.notna() & group_size.ge(min_group_size)
    inputs = _direction_inputs(frame, include_sw_index=(short_name == "sw_l1"))

    group_rows: list[pd.DataFrame] = []
    valid_input = inputs.where(valid_group)
    grouped = valid_input.assign(
        decision_date=frame["decision_date"], taxonomy=taxonomy
    ).groupby(["decision_date", "taxonomy"], dropna=True, sort=False)
    aggregated = grouped[["direction_strength", "breadth", "amount_acceleration"]].mean()
    if not aggregated.empty:
        aggregated["direction_strength_pct"] = aggregated.groupby(
            level="decision_date", sort=False
        )["direction_strength"].transform(_rank_groups)
        aggregated["breadth_pct"] = aggregated.groupby(
            level="decision_date", sort=False
        )["breadth"].transform(_rank_groups)
        aggregated["amount_acceleration_pct"] = aggregated.groupby(
            level="decision_date", sort=False
        )["amount_acceleration"].transform(_rank_groups)
        direction_pcts = [
            "direction_strength_pct",
            "breadth_pct",
            "amount_acceleration_pct",
        ]
        aggregated["direction_component_count"] = aggregated[direction_pcts].notna().sum(axis=1)
        aggregated["direction_score"] = aggregated[direction_pcts].mean(axis=1).where(
            aggregated["direction_component_count"].ge(2)
        )
        group_rows.append(aggregated)
    group_table = group_rows[0] if group_rows else pd.DataFrame()

    prefix = f"structured_{short_name}"
    frame[f"{prefix}_group_size"] = group_size.where(taxonomy.notna())
    for raw_name in ("direction_strength", "breadth", "amount_acceleration"):
        output = f"f7_{short_name}_{raw_name}_raw"
        if group_table.empty:
            frame[output] = np.nan
        else:
            lookup = group_table[raw_name]
            row_key = pd.MultiIndex.from_arrays(
                [frame["decision_date"], taxonomy], names=lookup.index.names
            )
            frame[output] = lookup.reindex(row_key).to_numpy()
    direction_output = f"f7_{short_name}_direction_score"
    direction_count_output = f"f7_{short_name}_component_count"
    if group_table.empty:
        frame[direction_output] = np.nan
        frame[direction_count_output] = 0
    else:
        row_key = pd.MultiIndex.from_arrays(
            [frame["decision_date"], taxonomy], names=group_table.index.names
        )
        frame[direction_output] = group_table["direction_score"].reindex(row_key).to_numpy()
        count_values = group_table["direction_component_count"].reindex(row_key)
        frame[direction_count_output] = pd.array(count_values, dtype="Int64")

    # Stock leadership is ranked only within the same date/taxonomy group.
    leader_momentum = _row_mean(frame, _LEADER_MOMENTUM_COLUMNS, minimum=1)
    leader_components = pd.DataFrame(index=frame.index)
    leader_components["momentum"] = _rank_within_groups(
        leader_momentum, frame["decision_date"], taxonomy, valid_group
    )
    for source, component in (
        ("liq20", "liquidity"),
        ("db_turnover_rate_t1", "turnover"),
        ("db_turnover_rate_f_t1", "free_turnover"),
        ("db_volume_ratio_t1", "volume_ratio"),
        ("db_circ_mv_t1", "float_market_cap"),
    ):
        if source in frame.columns:
            leader_components[component] = _rank_within_groups(
                _numeric(frame, source), frame["decision_date"], taxonomy, valid_group
            )
    component_count = leader_components.notna().sum(axis=1)
    leader_output = f"f8_{short_name}_leader_score"
    frame[leader_output] = leader_components.mean(axis=1).where(component_count.ge(2))
    frame[f"f8_{short_name}_component_count"] = component_count.astype("Int64")
    frame[f"{prefix}_oracle_taxonomy"] = bool(oracle)
    return frame, direction_output, leader_output


def _tech_member_columns(frame: pd.DataFrame) -> list[str]:
    return sorted(
        column
        for column in frame.columns
        if column.startswith("tech_member__") and column.endswith("_static_oracle")
    )


def _add_tech_score(
    frame: pd.DataFrame,
    *,
    active: pd.Series,
    min_group_size: int,
) -> pd.DataFrame:
    member_columns = _tech_member_columns(frame)
    frame["f9_tech_index_score"] = np.nan
    frame["f9_tech_index_component_count"] = pd.Series(
        0, index=frame.index, dtype="Int64"
    )
    frame["f9_tech_index_memberships_used"] = ""
    if not member_columns:
        frame["f9_tech_index_missing"] = True
        return frame

    direction_inputs = _direction_inputs(frame, include_sw_index=False)
    leader_momentum = _row_mean(frame, _LEADER_MOMENTUM_COLUMNS, minimum=1)
    score_pieces: list[pd.Series] = []
    name_pieces: list[pd.Series] = []
    for membership_column in member_columns:
        member = _bool_series(frame[membership_column]) & active
        group_size = member.groupby(frame["decision_date"], sort=False).transform("sum")
        valid = member & group_size.ge(min_group_size)

        group_raw = direction_inputs.where(valid).assign(
            decision_date=frame["decision_date"]
        ).groupby("decision_date", sort=False)[
            ["direction_strength", "breadth", "amount_acceleration"]
        ].mean()
        if group_raw.empty:
            continue
        # Rank each tech index against the other technology indexes on the same
        # date.  We build the cross-index ranks after collecting raw group rows.
        code = membership_column[
            len("tech_member__") : -len("_static_oracle")
        ]
        group_raw["membership_column"] = membership_column
        group_raw["tech_code"] = code
        group_raw["_valid_member_count"] = valid.groupby(
            frame["decision_date"], sort=False
        ).sum()
        group_raw = group_raw.reset_index()
        name_pieces.append(group_raw)

    if not name_pieces:
        frame["f9_tech_index_missing"] = True
        return frame

    groups = pd.concat(name_pieces, ignore_index=True)
    for raw in ("direction_strength", "breadth", "amount_acceleration"):
        groups[f"{raw}_pct"] = groups.groupby("decision_date", sort=False)[raw].rank(
            method="average", pct=True
        )
    group_pct_columns = [
        "direction_strength_pct",
        "breadth_pct",
        "amount_acceleration_pct",
    ]
    groups["group_score"] = groups[group_pct_columns].mean(axis=1).where(
        groups[group_pct_columns].notna().sum(axis=1).ge(2)
    )

    membership_names = pd.Series("", index=frame.index, dtype="string")
    for group in groups.itertuples(index=False):
        membership_column = str(group.membership_column)
        member = _bool_series(frame[membership_column]) & active
        same_date = frame["decision_date"].eq(pd.Timestamp(group.decision_date))
        valid = member & same_date
        if int(valid.sum()) < min_group_size or pd.isna(group.group_score):
            continue
        taxonomy = pd.Series(membership_column, index=frame.index, dtype="string")
        leader_components = pd.DataFrame(index=frame.index)
        leader_components["momentum"] = _rank_within_groups(
            leader_momentum, frame["decision_date"], taxonomy, valid
        )
        for source, component in (
            ("liq20", "liquidity"),
            ("db_turnover_rate_t1", "turnover"),
            ("db_turnover_rate_f_t1", "free_turnover"),
            ("db_volume_ratio_t1", "volume_ratio"),
            ("db_circ_mv_t1", "float_market_cap"),
        ):
            if source in frame.columns:
                leader_components[component] = _rank_within_groups(
                    _numeric(frame, source),
                    frame["decision_date"],
                    taxonomy,
                    valid,
                )
        leader_count = leader_components.notna().sum(axis=1)
        leader_score = leader_components.mean(axis=1).where(leader_count.ge(2))
        combined = (0.70 * float(group.group_score) + 0.30 * leader_score).where(valid)
        score_pieces.append(combined.rename(membership_column))
        code = str(group.tech_code)
        existing = membership_names.loc[valid]
        membership_names.loc[valid] = np.where(
            existing.eq(""), code, existing + "|" + code
        )

    if score_pieces:
        scores = pd.concat(score_pieces, axis=1)
        frame["f9_tech_index_score"] = scores.mean(axis=1, skipna=True).where(
            scores.notna().any(axis=1)
        )
        frame["f9_tech_index_component_count"] = scores.notna().sum(axis=1).astype(
            "Int64"
        )
        frame["f9_tech_index_memberships_used"] = membership_names.fillna("")
    frame["f9_tech_index_missing"] = frame["f9_tech_index_score"].isna()
    return frame


def add_structured_industry_scores(
    panel: pd.DataFrame,
    *,
    min_group_size: int = 3,
) -> pd.DataFrame:
    """Add F7 structured direction, F8 group leader, and F9 tech scores.

    Parameters
    ----------
    panel:
        One row per ``(decision_date, symbol)``.  All numerical inputs are read
        only from :data:`STRUCTURED_SOURCE_COLUMNS`.
    min_group_size:
        Minimum eligible members required for an industry or technology group.

    Returns
    -------
    pandas.DataFrame
        The original columns plus taxonomy-specific diagnostics and the three
        final scores.  F7/F8 are means across available taxonomies; F9 is NA for
        nonmembers or when no individual static technology-membership columns
        exist.
    """

    if min_group_size < 2:
        raise ValueError("min_group_size must be at least 2")
    result = _prepare_panel(panel)
    active = _active_mask(result)
    direction_scores: list[str] = []
    leader_scores: list[str] = []
    specs = _taxonomy_specs(result)
    for short_name, taxonomy_column, oracle in specs:
        result, direction_output, leader_output = _add_taxonomy_scores(
            result,
            short_name=short_name,
            taxonomy_column=taxonomy_column,
            oracle=oracle,
            active=active,
            min_group_size=min_group_size,
        )
        direction_scores.append(direction_output)
        leader_scores.append(leader_output)

    if direction_scores:
        result["f7_structured_direction_score"] = result[direction_scores].mean(
            axis=1, skipna=True
        ).where(result[direction_scores].notna().any(axis=1))
        result["f7_structured_direction_taxonomy_count"] = result[
            direction_scores
        ].notna().sum(axis=1).astype("Int64")
    else:
        result["f7_structured_direction_score"] = np.nan
        result["f7_structured_direction_taxonomy_count"] = pd.Series(
            0, index=result.index, dtype="Int64"
        )
    if leader_scores:
        result["f8_structured_leader_score"] = result[leader_scores].mean(
            axis=1, skipna=True
        ).where(result[leader_scores].notna().any(axis=1))
        result["f8_structured_leader_taxonomy_count"] = result[
            leader_scores
        ].notna().sum(axis=1).astype("Int64")
    else:
        result["f8_structured_leader_score"] = np.nan
        result["f8_structured_leader_taxonomy_count"] = pd.Series(
            0, index=result.index, dtype="Int64"
        )

    result = _add_tech_score(
        result, active=active, min_group_size=min_group_size
    )
    used_sources = [column for column in STRUCTURED_SOURCE_COLUMNS if column in result]
    result["structured_factor_source_columns"] = "|".join(used_sources)
    result["structured_factor_taxonomies"] = "|".join(
        taxonomy_column for _, taxonomy_column, _ in specs
    )
    result["structured_factor_label_event_pnl_columns_used"] = False
    result["structured_factor_asof_rule"] = (
        "same-decision-date cross-section; approved T-1 feature whitelist only; "
        "static Shenwan/tech memberships are oracle classifications"
    )
    result["f7_structured_direction_missing"] = result[
        "f7_structured_direction_score"
    ].isna()
    result["f8_structured_leader_missing"] = result[
        "f8_structured_leader_score"
    ].isna()
    return (
        result.sort_values("_structured_original_order", kind="mergesort")
        .drop(columns="_structured_original_order")
        .reset_index(drop=True)
    )


__all__ = ["STRUCTURED_SOURCE_COLUMNS", "add_structured_industry_scores"]
