"""Label-free, deterministic rules for chain-first stock selection.

The module contains no file I/O and never reads disclosed-trade labels, future
returns, or P&L. It implements the frozen rule model as pure DataFrame
transformations:

* four complete chain-direction inputs are percentile-ranked and equally
  weighted within each decision day;
* stock inputs are percentile-ranked within ``date x chain``, aggregated into
  the four frozen families (strength, leadership, shape, evidence), and then
  equally weighted with missing families filled by a neutral 0.5;
* chains are ranked deterministically by score descending and chain id
  ascending; and
* each selected chain fills at most four fixed-weight slots with a staged
  greedy rule: highest score first, then a different subsegment where possible,
  then score order, subject to at most two names per subsegment and hard known
  correlation de-duplication above 0.80.

Unfilled slots remain cash. Unknown correlations are allowed but explicitly
counted in the audit output.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from afactor.trader_choice import ChoiceConfig, add_same_day_percentiles


MEMBER_FAMILY_ORDER: tuple[str, ...] = (
    "strength",
    "leadership",
    "shape",
    "evidence",
)
_MISSING_SUBSEGMENT = "__missing_subsegment__"


@dataclass(frozen=True)
class ChainRuleConfig:
    """Frozen columns and constraints for the rule-based chain portfolio."""

    date_col: str = "decision_date"
    chain_col: str = "chain_id"
    symbol_col: str = "symbol"
    chain_eligible_col: str = "chain_eligible"
    member_eligible_col: str = "eligible_for_ranking"
    subsegment_col: str = "subsegment"
    subsegment_aliases: tuple[str, ...] = ("chain_stage",)
    chain_score_col: str = "chain_direction_score"
    member_score_col: str = "within_chain_score"

    n_chains: int = 3
    names_per_chain: int = 4
    min_subsegments_per_chain: int = 2
    max_names_per_subsegment: int = 2
    correlation_threshold: float = 0.80

    def __post_init__(self) -> None:
        column_names = (
            self.date_col,
            self.chain_col,
            self.symbol_col,
            self.chain_eligible_col,
            self.member_eligible_col,
            self.subsegment_col,
            self.chain_score_col,
            self.member_score_col,
        )
        if any(not str(column).strip() for column in column_names):
            raise ValueError("column names must not be empty")
        if len({self.date_col, self.chain_col, self.symbol_col}) != 3:
            raise ValueError("date_col, chain_col, and symbol_col must be distinct")
        if self.n_chains <= 0 or self.names_per_chain <= 0:
            raise ValueError("n_chains and names_per_chain must be positive")
        if not 0 <= self.min_subsegments_per_chain <= self.names_per_chain:
            raise ValueError(
                "min_subsegments_per_chain must be between 0 and names_per_chain"
            )
        if not 1 <= self.max_names_per_subsegment <= self.names_per_chain:
            raise ValueError(
                "max_names_per_subsegment must be between 1 and names_per_chain"
            )
        if not 0 <= float(self.correlation_threshold) <= 1:
            raise ValueError("correlation_threshold must be between 0 and 1")

        aliases = tuple(
            dict.fromkeys(str(column).strip() for column in self.subsegment_aliases)
        )
        if any(not column for column in aliases):
            raise ValueError("subsegment_aliases must not contain empty names")
        aliases = tuple(
            column for column in aliases if column != self.subsegment_col
        )
        object.__setattr__(self, "subsegment_aliases", aliases)

    @property
    def slot_weight(self) -> float:
        """Fixed weight of one stock slot; unfilled slots are cash."""

        return 1.0 / (self.n_chains * self.names_per_chain)


@dataclass(frozen=True)
class ChainPortfolioResult:
    """Selected positions plus chain-level and date-level audit frames."""

    positions: pd.DataFrame
    chain_audit: pd.DataFrame
    date_summary: pd.DataFrame


def _require_columns(
    frame: pd.DataFrame, columns: Sequence[str], *, where: str
) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{where} is missing required columns: {missing}")


def _date_series(values: pd.Series) -> pd.Series:
    # Decision dates are calendar labels rather than instants; parsing through
    # UTC could shift a timezone-aware midnight into the prior session.
    return pd.to_datetime(values, errors="coerce").dt.normalize()


def _bool_series(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values.dtype):
        return values.fillna(False).astype(bool)
    if pd.api.types.is_numeric_dtype(values.dtype):
        return pd.to_numeric(values, errors="coerce").fillna(0).ne(0)
    normalized = values.astype("string").str.strip().str.lower()
    return normalized.isin({"1", "true", "t", "yes", "y"}).fillna(False)


def _clean_identifier(values: pd.Series) -> pd.Series:
    return values.astype("string").str.strip()


def _factor_columns(factor_columns: Sequence[str]) -> tuple[str, ...]:
    columns = tuple(dict.fromkeys(str(column).strip() for column in factor_columns))
    if not columns or any(not column for column in columns):
        raise ValueError("factor columns must contain non-empty names")
    return columns


def _factor_families(
    factor_families: Mapping[str, Sequence[str]],
) -> dict[str, tuple[str, ...]]:
    normalized = {
        str(name).strip().lower(): _factor_columns(columns)
        for name, columns in factor_families.items()
    }
    if set(normalized) != set(MEMBER_FAMILY_ORDER):
        raise ValueError(
            "factor_families must contain exactly: "
            + ", ".join(MEMBER_FAMILY_ORDER)
        )
    if len(normalized["shape"]) != 3:
        raise ValueError("the shape family must contain exactly three branches")
    flattened = [
        column
        for family in MEMBER_FAMILY_ORDER
        for column in normalized[family]
    ]
    if len(flattened) != len(set(flattened)):
        raise ValueError("a factor column cannot belong to multiple families")
    return {family: normalized[family] for family in MEMBER_FAMILY_ORDER}


def _prepare_chains(
    panel: pd.DataFrame, config: ChainRuleConfig, *, where: str
) -> pd.DataFrame:
    _require_columns(panel, [config.date_col, config.chain_col], where=where)
    result = panel.copy()
    result[config.date_col] = _date_series(result[config.date_col])
    result[config.chain_col] = _clean_identifier(result[config.chain_col])
    if result[config.date_col].isna().any():
        raise ValueError(f"{where} contains invalid {config.date_col} values")
    if result[config.chain_col].isna().any() or result[config.chain_col].eq("").any():
        raise ValueError(f"{where} contains invalid {config.chain_col} values")
    if result.duplicated([config.date_col, config.chain_col], keep=False).any():
        raise ValueError(
            f"{where} must be unique by ({config.date_col}, {config.chain_col})"
        )
    return result


def _prepare_members(
    panel: pd.DataFrame, config: ChainRuleConfig, *, where: str
) -> pd.DataFrame:
    _require_columns(
        panel,
        [config.date_col, config.chain_col, config.symbol_col],
        where=where,
    )
    result = panel.copy()
    result[config.date_col] = _date_series(result[config.date_col])
    result[config.chain_col] = _clean_identifier(result[config.chain_col])
    result[config.symbol_col] = _clean_identifier(result[config.symbol_col])
    keys = [config.date_col, config.chain_col, config.symbol_col]
    if result[keys].isna().any(axis=None):
        raise ValueError(f"{where} contains invalid key values")
    if result[config.chain_col].eq("").any() or result[config.symbol_col].eq("").any():
        raise ValueError(f"{where} contains blank chain or symbol values")
    if result.duplicated(keys, keep=False).any():
        raise ValueError(
            f"{where} must be unique by "
            f"({config.date_col}, {config.chain_col}, {config.symbol_col})"
        )
    return result


def _choice_config(
    config: ChainRuleConfig, *, item_col: str, eligible_col: str
) -> ChoiceConfig:
    return ChoiceConfig(
        date_col=config.date_col,
        symbol_col=item_col,
        eligible_col=eligible_col,
    )


def _add_deterministic_rank(
    frame: pd.DataFrame,
    *,
    group_columns: Sequence[str],
    item_col: str,
    score_col: str,
    eligible: pd.Series,
    rank_col: str,
) -> pd.DataFrame:
    result = frame.copy()
    result[rank_col] = pd.Series(pd.NA, index=result.index, dtype="Int64")
    score = pd.to_numeric(result[score_col], errors="coerce")
    rankable = eligible & score.notna()
    if not rankable.any():
        return result
    grouped = result.loc[rankable].groupby(list(group_columns), sort=True).groups
    for indices in grouped.values():
        ordered = result.loc[indices].assign(_rank_score=score.loc[indices]).sort_values(
            ["_rank_score", item_col],
            ascending=[False, True],
            kind="mergesort",
        )
        result.loc[ordered.index, rank_col] = np.arange(1, len(ordered) + 1)
    return result


def score_chain_directions(
    chain_panel: pd.DataFrame,
    factor_columns: Sequence[str],
    config: ChainRuleConfig = ChainRuleConfig(),
    *,
    factor_directions: Mapping[str, int] | None = None,
) -> pd.DataFrame:
    """Rank and equally weight the four complete chain-direction inputs."""

    columns = _factor_columns(factor_columns)
    if len(columns) != 4:
        raise ValueError("chain direction scoring requires exactly four factors")
    result = _prepare_chains(chain_panel, config, where="chain_panel")
    _require_columns(
        result, [config.chain_eligible_col, *columns], where="chain_panel"
    )
    result["_chain_rule_order"] = np.arange(len(result), dtype=np.int64)
    for column in columns:
        result[column] = pd.to_numeric(result[column], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
    eligible = _bool_series(result[config.chain_eligible_col])
    result["chain_factor_count"] = result[list(columns)].notna().sum(axis=1)
    complete = result["chain_factor_count"].eq(4)
    score_pool_col = "_chain_direction_complete_pool"
    while score_pool_col in result.columns:
        score_pool_col = f"_{score_pool_col}"
    result[score_pool_col] = eligible & complete
    result = add_same_day_percentiles(
        result,
        columns,
        _choice_config(
            config,
            item_col=config.chain_col,
            eligible_col=config.chain_eligible_col,
        ),
        pool_col=score_pool_col,
        factor_directions=factor_directions,
        prefix="chain_pct__",
    )
    percentile_columns = [f"chain_pct__{column}" for column in columns]
    result[config.chain_score_col] = result[percentile_columns].sum(
        axis=1, skipna=False
    ) / 4.0
    result.loc[~eligible | ~complete, config.chain_score_col] = np.nan
    result = _add_deterministic_rank(
        result,
        group_columns=[config.date_col],
        item_col=config.chain_col,
        score_col=config.chain_score_col,
        eligible=eligible,
        rank_col="chain_direction_rank",
    )
    return (
        result.sort_values("_chain_rule_order", kind="mergesort")
        .drop(columns=["_chain_rule_order", score_pool_col])
        .reset_index(drop=True)
    )


def score_within_chains(
    member_panel: pd.DataFrame,
    factor_families: Mapping[str, Sequence[str]],
    config: ChainRuleConfig = ChainRuleConfig(),
    *,
    factor_directions: Mapping[str, int] | None = None,
) -> pd.DataFrame:
    """Build the frozen four-family stock score within ``date x chain``.

    Every bottom-level factor is first converted to a within-chain percentile.
    Strength, leadership, and evidence use the mean of their available factor
    percentiles. Shape takes the maximum of its three branch percentiles.
    A wholly missing family receives a neutral 0.5 gate value, so the final
    score always uses the fixed four-family denominator for eligible rows.
    """

    families = _factor_families(factor_families)
    columns = tuple(
        column for family in MEMBER_FAMILY_ORDER for column in families[family]
    )
    prepared = _prepare_members(member_panel, config, where="member_panel")
    _require_columns(
        prepared, [config.member_eligible_col, *columns], where="member_panel"
    )
    prepared["_chain_rule_order"] = np.arange(len(prepared), dtype=np.int64)
    for column in columns:
        prepared[column] = pd.to_numeric(
            prepared[column], errors="coerce"
        ).replace([np.inf, -np.inf], np.nan)
    choice_config = _choice_config(
        config,
        item_col=config.symbol_col,
        eligible_col=config.member_eligible_col,
    )
    parts: list[pd.DataFrame] = []
    for _, subset in prepared.groupby(config.chain_col, sort=True):
        parts.append(
            add_same_day_percentiles(
                subset,
                columns,
                choice_config,
                pool_col=config.member_eligible_col,
                factor_directions=factor_directions,
                prefix="within_chain_pct__",
            )
        )
    if parts:
        result = pd.concat(parts, ignore_index=True)
    else:
        result = prepared.copy()
        for column in columns:
            result[f"within_chain_pct__{column}"] = np.nan

    percentile_columns = [f"within_chain_pct__{column}" for column in columns]
    result["within_chain_factor_count"] = result[percentile_columns].notna().sum(
        axis=1
    )
    raw_family_columns: list[str] = []
    gate_family_columns: list[str] = []
    for family in MEMBER_FAMILY_ORDER:
        inputs = [f"within_chain_pct__{column}" for column in families[family]]
        raw_column = f"family_score__{family}"
        gate_column = f"family_gate__{family}"
        if family == "shape":
            result[raw_column] = result[inputs].max(axis=1, skipna=True)
        else:
            result[raw_column] = result[inputs].mean(axis=1, skipna=True)
        result[gate_column] = result[raw_column].fillna(0.5)
        raw_family_columns.append(raw_column)
        gate_family_columns.append(gate_column)

    result["neutral_fill_count"] = result[raw_family_columns].isna().sum(axis=1)
    result["within_chain_family_count"] = 4 - result["neutral_fill_count"]
    eligible = _bool_series(result[config.member_eligible_col])
    result[config.member_score_col] = result[gate_family_columns].sum(axis=1) / 4.0
    result.loc[~eligible, config.member_score_col] = np.nan
    result = _add_deterministic_rank(
        result,
        group_columns=[config.date_col, config.chain_col],
        item_col=config.symbol_col,
        score_col=config.member_score_col,
        eligible=eligible,
        rank_col="within_chain_rank",
    )
    return (
        result.sort_values("_chain_rule_order", kind="mergesort")
        .drop(columns="_chain_rule_order")
        .reset_index(drop=True)
    )


def select_top_chains(
    scored_chains: pd.DataFrame,
    config: ChainRuleConfig = ChainRuleConfig(),
) -> pd.DataFrame:
    """Select up to ``n_chains`` per date with deterministic tie-breaking."""

    chains = _prepare_chains(scored_chains, config, where="scored_chains")
    _require_columns(
        chains,
        [config.chain_eligible_col, config.chain_score_col],
        where="scored_chains",
    )
    score = pd.to_numeric(chains[config.chain_score_col], errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    eligible = _bool_series(chains[config.chain_eligible_col]) & score.notna()
    ranked = chains.loc[eligible].assign(_selection_score=score.loc[eligible])
    ranked = ranked.sort_values(
        [config.date_col, "_selection_score", config.chain_col],
        ascending=[True, False, True],
        kind="mergesort",
    )
    selected = ranked.groupby(config.date_col, sort=True, as_index=False).head(
        config.n_chains
    ).copy()
    selected["chain_selection_rank"] = (
        selected.groupby(config.date_col, sort=False).cumcount() + 1
    ).astype("Int64")
    selected["chain_selected"] = True
    return selected.drop(columns="_selection_score").reset_index(drop=True)


def _prepare_correlations(
    correlations: pd.DataFrame | None,
    config: ChainRuleConfig,
) -> dict[tuple[pd.Timestamp, str, str], float]:
    if correlations is None:
        return {}
    required = [config.date_col, "symbol_a", "symbol_b", "correlation"]
    _require_columns(correlations, required, where="correlations")
    frame = correlations[required].copy()
    frame[config.date_col] = _date_series(frame[config.date_col])
    frame["symbol_a"] = _clean_identifier(frame["symbol_a"])
    frame["symbol_b"] = _clean_identifier(frame["symbol_b"])
    frame["correlation"] = pd.to_numeric(frame["correlation"], errors="coerce")
    invalid = (
        frame[config.date_col].isna()
        | frame["symbol_a"].isna()
        | frame["symbol_b"].isna()
        | frame["symbol_a"].eq("")
        | frame["symbol_b"].eq("")
        | frame["correlation"].isna()
        | ~np.isfinite(frame["correlation"])
        | frame["correlation"].abs().gt(1)
    )
    if invalid.any():
        raise ValueError("correlations contains invalid dates, symbols, or values")
    frame = frame.loc[frame["symbol_a"].ne(frame["symbol_b"])].copy()
    frame["_left"] = frame[["symbol_a", "symbol_b"]].min(axis=1)
    frame["_right"] = frame[["symbol_a", "symbol_b"]].max(axis=1)
    lookup: dict[tuple[pd.Timestamp, str, str], float] = {}
    for values in frame.to_dict("records"):
        key = (
            pd.Timestamp(values[config.date_col]),
            str(values["_left"]),
            str(values["_right"]),
        )
        value = float(values["correlation"])
        previous = lookup.get(key)
        if previous is not None and not np.isclose(previous, value, atol=1e-12):
            raise ValueError(f"correlations contains conflicting duplicate pair: {key}")
        lookup[key] = value
    return lookup


def _correlation_value(
    lookup: Mapping[tuple[pd.Timestamp, str, str], float],
    decision_date: pd.Timestamp,
    left: str,
    right: str,
) -> float | None:
    if left == right:
        return 1.0
    first, second = sorted((left, right))
    return lookup.get((pd.Timestamp(decision_date), first, second))


def _resolve_subsegment_column(
    members: pd.DataFrame, config: ChainRuleConfig
) -> str:
    if config.subsegment_col in members.columns:
        return config.subsegment_col
    for alias in config.subsegment_aliases:
        if alias in members.columns:
            return alias
    names = [config.subsegment_col, *config.subsegment_aliases]
    raise ValueError(f"scored_members is missing subsegment column; tried {names}")


def _greedy_chain_selection(
    candidates: pd.DataFrame,
    *,
    decision_date: pd.Timestamp,
    previously_selected: Sequence[str],
    correlation_lookup: Mapping[tuple[pd.Timestamp, str, str], float],
    config: ChainRuleConfig,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Apply the frozen full-universe staged greedy rule to one chain."""

    selected_indices: list[int] = []
    selected_symbols: list[str] = []
    subsegment_counts: dict[str, int] = {}
    correlation_rejected: set[str] = set()
    subsegment_cap_blocked: set[str] = set()
    checked_pairs: set[tuple[str, str]] = set()
    missing_pairs: set[tuple[str, str]] = set()
    selected_observed_correlations: list[float] = []

    def select_one(*, require_new_known_subsegment: bool) -> bool:
        for index, row in candidates.iterrows():
            if index in selected_indices:
                continue
            symbol = str(row[config.symbol_col])
            subsegment = str(row["_normalized_subsegment"])
            current_count = subsegment_counts.get(subsegment, 0)
            if current_count >= config.max_names_per_subsegment:
                subsegment_cap_blocked.add(symbol)
                continue
            if require_new_known_subsegment and (
                subsegment == _MISSING_SUBSEGMENT or current_count > 0
            ):
                continue

            observed_for_candidate: list[float] = []
            high_correlation = False
            for existing in [*previously_selected, *selected_symbols]:
                pair = tuple(sorted((symbol, str(existing))))
                checked_pairs.add(pair)
                value = _correlation_value(
                    correlation_lookup,
                    decision_date,
                    symbol,
                    str(existing),
                )
                if value is None:
                    missing_pairs.add(pair)
                    continue
                observed_for_candidate.append(value)
                if value > config.correlation_threshold:
                    high_correlation = True
                    break
            if high_correlation:
                correlation_rejected.add(symbol)
                continue

            selected_indices.append(int(index))
            selected_symbols.append(symbol)
            subsegment_counts[subsegment] = current_count + 1
            selected_observed_correlations.extend(observed_for_candidate)
            return True
        return False

    if len(candidates) and config.names_per_chain:
        select_one(require_new_known_subsegment=False)
    while (
        len(selected_indices) < config.names_per_chain
        and len(
            {
                segment
                for segment, count in subsegment_counts.items()
                if segment != _MISSING_SUBSEGMENT and count > 0
            }
        )
        < config.min_subsegments_per_chain
    ):
        if not select_one(require_new_known_subsegment=True):
            break
    while len(selected_indices) < config.names_per_chain:
        if not select_one(require_new_known_subsegment=False):
            break

    chosen = candidates.loc[selected_indices].copy()
    known_selected_subsegments = {
        segment
        for segment, count in subsegment_counts.items()
        if segment != _MISSING_SUBSEGMENT and count > 0
    }
    audit: dict[str, object] = {
        "selected_subsegment_count": len(known_selected_subsegments),
        "selected_missing_subsegment_names": subsegment_counts.get(
            _MISSING_SUBSEGMENT, 0
        ),
        "subsegment_target_met": (
            len(known_selected_subsegments) >= config.min_subsegments_per_chain
        ),
        "n_rejected_high_correlation": len(correlation_rejected),
        "n_blocked_subsegment_cap": len(subsegment_cap_blocked),
        "correlation_pairs_checked": len(checked_pairs),
        "correlation_pairs_missing": len(missing_pairs),
        "max_observed_selected_correlation": (
            max(selected_observed_correlations)
            if selected_observed_correlations
            else None
        ),
    }
    return chosen, audit


def select_chain_portfolio(
    scored_members: pd.DataFrame,
    selected_chains: pd.DataFrame,
    correlations: pd.DataFrame | None = None,
    config: ChainRuleConfig = ChainRuleConfig(),
) -> ChainPortfolioResult:
    """Construct the fixed-slot, diversified chain portfolio.

    Selected chains are processed in rank order. Each holding receives exactly
    ``1 / (n_chains * names_per_chain)`` weight. The rule may select fewer than
    the requested names; correlation, subsegment, unavailable-chain, and
    unselected-chain slots are not redistributed and therefore remain cash.
    """

    members = _prepare_members(scored_members, config, where="scored_members")
    _require_columns(
        members,
        [config.member_eligible_col, config.member_score_col],
        where="scored_members",
    )
    subsegment_source = _resolve_subsegment_column(members, config)
    if subsegment_source != config.subsegment_col:
        members[config.subsegment_col] = members[subsegment_source]
    chains = _prepare_chains(selected_chains, config, where="selected_chains")
    _require_columns(
        chains,
        [config.chain_score_col, "chain_selection_rank"],
        where="selected_chains",
    )
    chains["chain_selection_rank"] = pd.to_numeric(
        chains["chain_selection_rank"], errors="coerce"
    ).astype("Int64")
    if chains["chain_selection_rank"].isna().any() or chains[
        "chain_selection_rank"
    ].le(0).any():
        raise ValueError("selected_chains contains invalid chain_selection_rank")
    if chains.duplicated([config.date_col, "chain_selection_rank"], keep=False).any():
        raise ValueError("chain_selection_rank must be unique within each date")
    if chains.groupby(config.date_col).size().gt(config.n_chains).any():
        raise ValueError("selected_chains exceeds configured n_chains on a date")
    chain_score = pd.to_numeric(chains[config.chain_score_col], errors="coerce")
    if chain_score.isna().any() or not np.isfinite(chain_score).all():
        raise ValueError("selected_chains contains invalid chain scores")
    chains[config.chain_score_col] = chain_score

    normalized_subsegment = (
        members[config.subsegment_col]
        .astype("string")
        .str.strip()
        .str.lower()
        .replace("", pd.NA)
    )
    members["_normalized_subsegment"] = normalized_subsegment.fillna(
        _MISSING_SUBSEGMENT
    )
    members["_selection_score"] = pd.to_numeric(
        members[config.member_score_col], errors="coerce"
    ).replace([np.inf, -np.inf], np.nan)
    member_eligible = _bool_series(members[config.member_eligible_col])
    correlation_lookup = _prepare_correlations(correlations, config)

    position_parts: list[pd.DataFrame] = []
    audit_records: list[dict[str, object]] = []
    summary_records: list[dict[str, object]] = []
    chains = chains.sort_values(
        [config.date_col, "chain_selection_rank", config.chain_col],
        kind="mergesort",
    )

    for decision_date, day_chains in chains.groupby(config.date_col, sort=True):
        previously_selected: list[str] = []
        day_position_count = 0
        funded_chains = 0
        day_chain_allocations: list[float] = []

        for chain_record in day_chains.to_dict("records"):
            chain_id = str(chain_record[config.chain_col])
            chain_rank = int(chain_record["chain_selection_rank"])
            candidate_mask = (
                members[config.date_col].eq(decision_date)
                & members[config.chain_col].eq(chain_id)
                & member_eligible
                & members["_selection_score"].notna()
            )
            all_candidates = members.loc[candidate_mask].sort_values(
                ["_selection_score", config.symbol_col],
                ascending=[False, True],
                kind="mergesort",
            )
            available = all_candidates.loc[
                ~all_candidates[config.symbol_col]
                .astype(str)
                .isin(previously_selected)
            ].reset_index(drop=True)
            known_available = available.loc[
                available["_normalized_subsegment"].ne(_MISSING_SUBSEGMENT),
                "_normalized_subsegment",
            ]
            chosen, greedy_audit = _greedy_chain_selection(
                available,
                decision_date=pd.Timestamp(decision_date),
                previously_selected=previously_selected,
                correlation_lookup=correlation_lookup,
                config=config,
            )
            selected_count = len(chosen)
            allocated_weight = selected_count * config.slot_weight
            status = (
                "selected_full"
                if selected_count == config.names_per_chain
                else "selected_partial"
                if selected_count > 0
                else "no_feasible_candidate"
            )
            if selected_count:
                chosen = chosen.sort_values(
                    ["_selection_score", config.symbol_col],
                    ascending=[False, True],
                    kind="mergesort",
                ).copy()
                chosen["chain_selection_rank"] = chain_rank
                chosen[config.chain_score_col] = float(
                    chain_record[config.chain_score_col]
                )
                chosen["position_rank_in_chain"] = np.arange(
                    1, selected_count + 1
                )
                chosen["slot_weight"] = config.slot_weight
                chosen["position_weight"] = config.slot_weight
                chosen["selection_status"] = status
                position_parts.append(chosen)
                new_symbols = chosen[config.symbol_col].astype(str).tolist()
                previously_selected.extend(new_symbols)
                day_position_count += selected_count
                funded_chains += 1

            day_chain_allocations.append(allocated_weight)
            selected_subsegments = int(
                greedy_audit["selected_subsegment_count"]
            )
            audit_records.append(
                {
                    config.date_col: pd.Timestamp(decision_date),
                    config.chain_col: chain_id,
                    "chain_selection_rank": chain_rank,
                    config.chain_score_col: float(chain_record[config.chain_score_col]),
                    "selection_status": status,
                    "requested_names": config.names_per_chain,
                    "selected_names": selected_count,
                    "unfilled_slots": config.names_per_chain - selected_count,
                    "available_candidates": len(all_candidates),
                    "candidates_after_cross_chain_dedupe": len(available),
                    "available_subsegment_count": int(known_available.nunique()),
                    "selected_subsegment_count": selected_subsegments,
                    "subsegment_target_met": bool(
                        greedy_audit["subsegment_target_met"]
                    ),
                    "subsegment_shortfall": max(
                        0,
                        config.min_subsegments_per_chain - selected_subsegments,
                    ),
                    "selected_missing_subsegment_names": int(
                        greedy_audit["selected_missing_subsegment_names"]
                    ),
                    "n_rejected_high_correlation": int(
                        greedy_audit["n_rejected_high_correlation"]
                    ),
                    "n_blocked_subsegment_cap": int(
                        greedy_audit["n_blocked_subsegment_cap"]
                    ),
                    "correlation_pairs_checked": int(
                        greedy_audit["correlation_pairs_checked"]
                    ),
                    "correlation_pairs_missing": int(
                        greedy_audit["correlation_pairs_missing"]
                    ),
                    "max_observed_selected_correlation": greedy_audit[
                        "max_observed_selected_correlation"
                    ],
                    "subsegment_source_col": subsegment_source,
                    "slot_weight": config.slot_weight,
                    "target_chain_weight": config.names_per_chain
                    * config.slot_weight,
                    "allocated_weight": allocated_weight,
                    "unallocated_chain_weight": (
                        config.names_per_chain - selected_count
                    )
                    * config.slot_weight,
                }
            )

        invested_weight = day_position_count * config.slot_weight
        summary_records.append(
            {
                config.date_col: pd.Timestamp(decision_date),
                "portfolio_slots": config.n_chains * config.names_per_chain,
                "n_selected_chains": len(day_chains),
                "n_funded_chains": funded_chains,
                "requested_positions": len(day_chains) * config.names_per_chain,
                "selected_positions": day_position_count,
                "invested_weight": invested_weight,
                "cash_weight": max(0.0, 1.0 - invested_weight),
                "slot_weight": config.slot_weight,
                "max_stock_weight_observed": (
                    config.slot_weight if day_position_count else 0.0
                ),
                "max_chain_weight_observed": max(
                    day_chain_allocations, default=0.0
                ),
            }
        )

    appended_position_columns = [
        "chain_selection_rank",
        "position_rank_in_chain",
        "slot_weight",
        "position_weight",
        "selection_status",
    ]
    output_columns = list(members.columns)
    for column in appended_position_columns:
        if column not in output_columns:
            output_columns.append(column)
    internal_columns = {"_normalized_subsegment", "_selection_score"}
    output_columns = [
        column for column in output_columns if column not in internal_columns
    ]
    if position_parts:
        positions = pd.concat(position_parts, ignore_index=True)
        positions = positions.sort_values(
            [
                config.date_col,
                "chain_selection_rank",
                "position_rank_in_chain",
                config.symbol_col,
            ],
            kind="mergesort",
        )
        positions = positions.loc[:, output_columns].reset_index(drop=True)
    else:
        positions = pd.DataFrame(columns=output_columns)

    chain_audit = pd.DataFrame.from_records(audit_records)
    if not chain_audit.empty:
        chain_audit = chain_audit.sort_values(
            [config.date_col, "chain_selection_rank", config.chain_col],
            kind="mergesort",
        ).reset_index(drop=True)
    date_summary = pd.DataFrame.from_records(summary_records)
    if not date_summary.empty:
        date_summary = date_summary.sort_values(
            config.date_col, kind="mergesort"
        ).reset_index(drop=True)
    return ChainPortfolioResult(positions, chain_audit, date_summary)


__all__ = [
    "ChainPortfolioResult",
    "ChainRuleConfig",
    "MEMBER_FAMILY_ORDER",
    "score_chain_directions",
    "score_within_chains",
    "select_chain_portfolio",
    "select_top_chains",
]
