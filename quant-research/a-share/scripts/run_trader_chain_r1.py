#!/usr/bin/env python3
"""Run the frozen chain-first R1 historical and behavior experiments.

The historical arm is strictly point in time: chain evidence comes only from
analyst-report title/fine-industry text, starts on the next trading session,
and expires after the configured trading-session TTL.  The 2026 behavior arm
uses the explicitly contaminated static semantic mapping only as an explanatory
upper bound.  Feature construction and portfolio selection finish before the
behavior labels are loaded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from afactor.trader_chain_evaluation import (  # noqa: E402
    ChainEvaluationConfig,
    R1GateConfig,
    evaluate_chain_returns,
    summarize_paired_net_excess_inference,
)
from afactor.trader_chain_experiment import (  # noqa: E402
    CHAIN_DIRECTION_FACTORS,
    build_chain_direction_inputs,
    trailing_return_correlations,
)
from afactor.trader_chain_pit import (  # noqa: E402
    TraderChainPITConfig,
    build_trader_chain_pit_membership,
)
from afactor.trader_chain_rules import (  # noqa: E402
    ChainPortfolioResult,
    ChainRuleConfig,
    score_chain_directions,
    score_within_chains,
    select_chain_portfolio,
    select_top_chains,
)
from afactor.trader_choice_data import (  # noqa: E402
    ChoiceDataPaths,
    add_forward_returns,
    build_base_panel,
    load_daily_inputs,
)


LEDGER = ROOT / "research" / "ledger"
PROTOCOL = ROOT / "research" / "TRADER_CHAIN_RULE_PROTOCOL.md"
DEFAULT_ORACLE_PANEL = LEDGER / "trader_oracle_panel.parquet"
DEFAULT_CHOICE_PANEL = LEDGER / "trader_choice_panel.parquet"
DEFAULT_STATIC_MEMBERSHIP = LEDGER / "trader_oracle_theme_audit.csv"
DEFAULT_LABELS = LEDGER / "trader_choice_labels.csv"
DEFAULT_BASELINE_RANKS = LEDGER / "trader_oracle_label_ranks.csv"
SEED = 20260831
RANDOM_DRAWS = 500
BOOTSTRAP_ITERATIONS = 10_000
MINIMUM_CHAIN_MEMBERS = 5
TARGET_SLOTS = 12
CONTROL_CHAIN_SLOTS = 3

HISTORICAL_FEATURE_COLUMNS: tuple[str, ...] = (
    "decision_date",
    "symbol",
    "feature_cutoff_date",
    "decision_open",
    "eligible_for_ranking",
    "mom20",
    "mom60",
    "mom20_pct",
    "mom60_pct",
    "ma20_dist",
    "amt_ratio20",
    "amt_ratio20_pct",
    "liq20",
    "pv_pullback_raw",
    "pv_vcp_raw",
    "pv_acceleration_raw",
)
MEMBER_FAMILIES: dict[str, tuple[str, ...]] = {
    "strength": ("mom20", "mom60"),
    "leadership": ("liq20", "amt_ratio20"),
    "shape": ("pv_pullback_raw", "pv_vcp_raw", "pv_acceleration_raw"),
    "evidence": ("source_count", "evidence_age_sessions"),
}
STATIC_MEMBER_FAMILIES: dict[str, tuple[str, ...]] = {
    "strength": ("mom20", "mom60"),
    "leadership": ("liq20", "amt_ratio20"),
    "shape": ("pv_pullback_raw", "pv_vcp_raw", "pv_acceleration_raw"),
    "evidence": ("source_count", "evidence_purity"),
}


class R1RunError(RuntimeError):
    """Raised when a frozen experiment cannot be run unambiguously."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("all", "historical", "behavior", "inference"), default="all"
    )
    parser.add_argument("--astock-root", type=Path, default=ROOT)
    parser.add_argument("--ledger-dir", type=Path, default=LEDGER)
    parser.add_argument("--reports", type=Path)
    parser.add_argument("--oracle-panel", type=Path, default=DEFAULT_ORACLE_PANEL)
    parser.add_argument("--choice-panel", type=Path, default=DEFAULT_CHOICE_PANEL)
    parser.add_argument(
        "--static-membership", type=Path, default=DEFAULT_STATIC_MEMBERSHIP
    )
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument(
        "--baseline-ranks", type=Path, default=DEFAULT_BASELINE_RANKS
    )
    parser.add_argument("--random-draws", type=int, default=RANDOM_DRAWS)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


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


def _require_columns(
    frame: pd.DataFrame, columns: Sequence[str], *, where: str
) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise R1RunError(f"{where} is missing required columns: {missing}")


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", prefix=f".{path.stem}.", dir=path.parent,
        delete=False, encoding="utf-8", newline="",
    ) as handle:
        temporary = Path(handle.name)
        frame.to_csv(handle, index=False)
    os.replace(temporary, path)


def _atomic_json(payload: Mapping[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", prefix=f".{path.stem}.", dir=path.parent,
        delete=False, encoding="utf-8",
    ) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=_json_default)
        handle.write("\n")
    os.replace(temporary, path)


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        suffix=".parquet", prefix=f".{path.stem}.", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _json_default(value: object) -> object:
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return str(pd.Timestamp(value))
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if pd.isna(value):
        return None
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _file_signature(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "exists": path.exists(),
        "size": path.stat().st_size if path.exists() else None,
        "sha256": _sha256(path) if path.exists() else None,
    }


def _schedules(calendar: pd.DatetimeIndex) -> dict[str, tuple[list[pd.Timestamp], int, int]]:
    evaluation = calendar[
        (calendar >= pd.Timestamp("2021-01-01"))
        & (calendar <= pd.Timestamp("2025-12-31"))
    ]
    monthly = (
        pd.Series(evaluation, index=evaluation)
        .groupby(evaluation.to_period("M"))
        .first()
        .tolist()
    )
    weekly = list(evaluation[::5])
    return {
        "monthly": (monthly, 20, 12),
        "weekly": (weekly, 5, 52),
    }


def _prepare_historical_feature_side(
    decision_dates: Sequence[pd.Timestamp],
    *,
    ttl_sessions: int,
    opens: pd.DataFrame,
    closes: pd.DataFrame,
    amounts: pd.DataFrame,
    reports: Path,
    base_panel: pd.DataFrame | None = None,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, pd.DataFrame],
    dict[str, ChainPortfolioResult],
]:
    """Finish every historical feature and selection step without outcomes."""

    if base_panel is None:
        base = build_base_panel(decision_dates, opens, closes, amounts)
        _require_columns(base, HISTORICAL_FEATURE_COLUMNS, where="historical base panel")
        base = base.loc[:, HISTORICAL_FEATURE_COLUMNS].copy()
    else:
        _require_columns(
            base_panel, HISTORICAL_FEATURE_COLUMNS, where="cached historical base panel"
        )
        base = base_panel.loc[:, HISTORICAL_FEATURE_COLUMNS].copy()
    membership = build_trader_chain_pit_membership(
        base[["decision_date", "symbol"]],
        reports,
        opens.index,
        config=TraderChainPITConfig(ttl_sessions=ttl_sessions),
    )
    members = membership.merge(
        base,
        on=["decision_date", "symbol"],
        how="left",
        validate="many_to_one",
    )
    members["subsegment"] = (
        members["latest_report_fine_industry"]
        .astype("string")
        .str.strip()
        .replace("", pd.NA)
        .fillna("unknown")
    )
    chain_inputs = build_chain_direction_inputs(
        members, minimum_members=MINIMUM_CHAIN_MEMBERS
    )
    rule_config = ChainRuleConfig()
    chain_scores = score_chain_directions(
        chain_inputs, CHAIN_DIRECTION_FACTORS, rule_config
    )
    selected_chains = select_top_chains(chain_scores, rule_config)

    full_score = score_within_chains(
        members,
        MEMBER_FAMILIES,
        rule_config,
        factor_directions={"evidence_age_sessions": -1},
    )
    minus_shape_input = members.copy()
    for column in MEMBER_FAMILIES["shape"]:
        minus_shape_input[column] = np.nan
    minus_shape = score_within_chains(
        minus_shape_input,
        MEMBER_FAMILIES,
        rule_config,
        factor_directions={"evidence_age_sessions": -1},
    )
    minus_evidence_input = members.copy()
    for column in MEMBER_FAMILIES["evidence"]:
        minus_evidence_input[column] = np.nan
    minus_evidence = score_within_chains(
        minus_evidence_input,
        MEMBER_FAMILIES,
        rule_config,
        factor_directions={"evidence_age_sessions": -1},
    )
    scored_variants = {
        "R1": full_score,
        "R1_minus_shape": minus_shape,
        "R1_minus_evidence": minus_evidence,
    }

    result_parts: dict[str, list[pd.DataFrame]] = {
        key: [] for key in scored_variants
    }
    audit_parts: dict[str, list[pd.DataFrame]] = {
        key: [] for key in scored_variants
    }
    summary_parts: dict[str, list[pd.DataFrame]] = {
        key: [] for key in scored_variants
    }
    for date_number, decision_date in enumerate(decision_dates, start=1):
        date = pd.Timestamp(decision_date)
        day_chains = selected_chains.loc[
            selected_chains["decision_date"].eq(date)
        ].copy()
        chain_ids = set(day_chains["chain_id"].astype(str))
        correlation_symbols = full_score.loc[
            full_score["decision_date"].eq(date)
            & full_score["chain_id"].astype(str).isin(chain_ids)
            & _bool_series(full_score["eligible_for_ranking"])
            & pd.to_numeric(full_score["within_chain_score"], errors="coerce").notna(),
            "symbol",
        ].astype(str)
        correlations = trailing_return_correlations(
            date,
            correlation_symbols,
            closes,
            lookback_sessions=60,
            minimum_observations=40,
        )
        for variant, scored in scored_variants.items():
            day_members = scored.loc[scored["decision_date"].eq(date)].copy()
            selected = select_chain_portfolio(
                day_members, day_chains, correlations, rule_config
            )
            if not selected.positions.empty:
                selected.positions.insert(0, "model_variant", variant)
                result_parts[variant].append(selected.positions)
            if not selected.chain_audit.empty:
                selected.chain_audit.insert(0, "model_variant", variant)
                audit_parts[variant].append(selected.chain_audit)
            if not selected.date_summary.empty:
                selected.date_summary.insert(0, "model_variant", variant)
                summary_parts[variant].append(selected.date_summary)
        if date_number % 25 == 0 or date_number == len(decision_dates):
            print(
                f"feature-side selection {date_number}/{len(decision_dates)} "
                f"through {date.date()}",
                flush=True,
            )

    selections: dict[str, ChainPortfolioResult] = {}
    for variant in scored_variants:
        positions = (
            pd.concat(result_parts[variant], ignore_index=True)
            if result_parts[variant]
            else pd.DataFrame()
        )
        chain_audit = (
            pd.concat(audit_parts[variant], ignore_index=True)
            if audit_parts[variant]
            else pd.DataFrame()
        )
        date_summary = (
            pd.concat(summary_parts[variant], ignore_index=True)
            if summary_parts[variant]
            else pd.DataFrame()
        )
        selections[variant] = ChainPortfolioResult(
            positions=positions,
            chain_audit=chain_audit,
            date_summary=date_summary,
        )
    no_diversification = _select_no_diversification(
        full_score, selected_chains, rule_config
    )
    selections["R1_no_diversification"] = no_diversification
    return (
        base,
        membership,
        chain_scores,
        selected_chains,
        scored_variants,
        selections,
    )


def _select_no_diversification(
    scored_members: pd.DataFrame,
    selected_chains: pd.DataFrame,
    config: ChainRuleConfig,
) -> ChainPortfolioResult:
    """Frozen ablation: keep chain order and scores, remove segment/corr rules."""

    positions: list[pd.DataFrame] = []
    audits: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for date, chains in selected_chains.sort_values(
        ["decision_date", "chain_selection_rank", "chain_id"], kind="mergesort"
    ).groupby("decision_date", sort=True):
        used: set[str] = set()
        selected_count = 0
        for chain in chains.to_dict("records"):
            candidates = scored_members.loc[
                scored_members["decision_date"].eq(date)
                & scored_members["chain_id"].astype(str).eq(str(chain["chain_id"]))
                & _bool_series(scored_members["eligible_for_ranking"])
                & pd.to_numeric(
                    scored_members["within_chain_score"], errors="coerce"
                ).notna()
            ].copy()
            candidates = candidates.loc[
                ~candidates["symbol"].astype(str).isin(used)
            ].sort_values(
                ["within_chain_score", "symbol"],
                ascending=[False, True],
                kind="mergesort",
            )
            chosen = candidates.head(config.names_per_chain).copy()
            count = len(chosen)
            if count:
                chosen.insert(0, "model_variant", "R1_no_diversification")
                chosen["chain_selection_rank"] = int(chain["chain_selection_rank"])
                chosen["position_rank_in_chain"] = np.arange(1, count + 1)
                chosen["slot_weight"] = config.slot_weight
                chosen["position_weight"] = config.slot_weight
                chosen["selection_status"] = (
                    "selected_full" if count == config.names_per_chain else "selected_partial"
                )
                positions.append(chosen)
                used.update(chosen["symbol"].astype(str))
                selected_count += count
            audits.append(
                {
                    "model_variant": "R1_no_diversification",
                    "decision_date": date,
                    "chain_id": chain["chain_id"],
                    "chain_selection_rank": chain["chain_selection_rank"],
                    "selected_names": count,
                    "requested_names": config.names_per_chain,
                    "unfilled_slots": config.names_per_chain - count,
                    "selection_status": (
                        "selected_full"
                        if count == config.names_per_chain
                        else "selected_partial"
                        if count
                        else "no_feasible_candidate"
                    ),
                    "diversification_constraints_enabled": False,
                }
            )
        summaries.append(
            {
                "model_variant": "R1_no_diversification",
                "decision_date": date,
                "portfolio_slots": config.n_chains * config.names_per_chain,
                "n_selected_chains": len(chains),
                "selected_positions": selected_count,
                "invested_weight": selected_count * config.slot_weight,
                "cash_weight": 1.0 - selected_count * config.slot_weight,
            }
        )
    return ChainPortfolioResult(
        positions=pd.concat(positions, ignore_index=True) if positions else pd.DataFrame(),
        chain_audit=pd.DataFrame(audits),
        date_summary=pd.DataFrame(summaries),
    )


def _lookup_forward_returns(
    base: pd.DataFrame,
    opens: pd.DataFrame,
    horizon: int,
) -> tuple[pd.DataFrame, str]:
    outcomes = add_forward_returns(
        base[["decision_date", "symbol"]], opens, horizons=(horizon,)
    )
    column = f"fwd_open_to_open_d{horizon}"
    return outcomes[["decision_date", "symbol", column]], column


def _weighted_selected_return(
    positions: pd.DataFrame,
    outcomes: pd.DataFrame,
    return_column: str,
    *,
    target_slots: int = TARGET_SLOTS,
) -> tuple[float, int, int]:
    if positions.empty:
        return 0.0, 0, 0
    merged = positions[["decision_date", "symbol"]].merge(
        outcomes,
        on=["decision_date", "symbol"],
        how="left",
        validate="one_to_one",
    )
    values = pd.to_numeric(merged[return_column], errors="coerce")
    missing = int(values.isna().sum())
    if missing:
        return np.nan, len(merged), missing
    return float(values.sum() / target_slots), len(merged), 0


def _member_chain_return(
    day_members: pd.DataFrame,
    chain_id: str,
    outcome_lookup: pd.Series,
) -> tuple[float, int, int]:
    symbols = (
        day_members.loc[
            day_members["chain_id"].astype(str).eq(str(chain_id))
            & _bool_series(day_members["eligible_for_ranking"]),
            "symbol",
        ]
        .astype(str)
        .drop_duplicates()
    )
    values = pd.to_numeric(outcome_lookup.reindex(symbols), errors="coerce")
    valid = values.dropna()
    return (
        float(valid.mean()) if len(valid) else np.nan,
        int(len(symbols)),
        int(len(valid)),
    )


def _random_same_chain_draws(
    decision_date: pd.Timestamp,
    day_members: pd.DataFrame,
    day_chains: pd.DataFrame,
    outcome_lookup: pd.Series,
    *,
    repeats: int,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    chains = day_chains.sort_values("chain_selection_rank", kind="mergesort")
    candidates: dict[str, np.ndarray] = {}
    for chain_id in chains["chain_id"].astype(str):
        symbols = (
            day_members.loc[
                day_members["chain_id"].astype(str).eq(chain_id)
                & _bool_series(day_members["eligible_for_ranking"]),
                "symbol",
            ]
            .astype(str)
            .drop_duplicates()
            .sort_values()
            .to_numpy()
        )
        candidates[chain_id] = symbols
    rng = np.random.default_rng(seed)
    for draw in range(repeats):
        selected: list[str] = []
        for chain_id in chains["chain_id"].astype(str):
            population = candidates[chain_id]
            take = min(4, len(population))
            if take:
                selected.extend(rng.choice(population, size=take, replace=False).tolist())
        values = pd.to_numeric(outcome_lookup.reindex(selected), errors="coerce")
        missing = int(values.isna().sum())
        gross = float(values.sum() / TARGET_SLOTS) if not missing else np.nan
        rows.append(
            {
                "decision_date": decision_date,
                "draw": draw,
                "gross_return": gross,
                "slots_target": TARGET_SLOTS,
                "slots_filled": len(selected),
                "missing_outcomes": missing,
            }
        )
    return pd.DataFrame(rows)


def _historical_returns(
    base: pd.DataFrame,
    membership: pd.DataFrame,
    selected_chains: pd.DataFrame,
    selections: Mapping[str, ChainPortfolioResult],
    opens: pd.DataFrame,
    *,
    horizon: int,
    random_draws: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, float | None]:
    outcomes, return_column = _lookup_forward_returns(base, opens, horizon)
    outcome_series = outcomes.set_index(["decision_date", "symbol"])[return_column]
    members = membership.merge(
        base[["decision_date", "symbol", "eligible_for_ranking"]],
        on=["decision_date", "symbol"],
        how="left",
        validate="many_to_one",
    )
    daily_rows: list[dict[str, object]] = []
    draw_parts: list[pd.DataFrame] = []
    attribution_rows: list[dict[str, object]] = []
    dates = sorted(base["decision_date"].drop_duplicates())
    for date in dates:
        date = pd.Timestamp(date)
        day_base = base.loc[base["decision_date"].eq(date)]
        day_members = members.loc[members["decision_date"].eq(date)]
        day_chains = selected_chains.loc[
            selected_chains["decision_date"].eq(date)
        ].sort_values("chain_selection_rank", kind="mergesort")
        day_outcomes = outcome_series.loc[date] if date in outcome_series.index.levels[0] else pd.Series(dtype=float)

        eligible_symbols = day_base.loc[
            _bool_series(day_base["eligible_for_ranking"]), "symbol"
        ].astype(str)
        eligible_values = pd.to_numeric(
            day_outcomes.reindex(eligible_symbols), errors="coerce"
        )
        daily_rows.append(
            {
                "decision_date": date,
                "strategy": "all_eligible",
                "gross_return": float(eligible_values.dropna().mean())
                if eligible_values.notna().any()
                else np.nan,
                "slots_target": len(eligible_symbols),
                "slots_filled": len(eligible_symbols),
                "missing_outcomes": int(eligible_values.isna().sum()),
                "future_return_coverage": float(eligible_values.notna().mean())
                if len(eligible_values)
                else np.nan,
            }
        )

        active_symbols = (
            day_members.loc[_bool_series(day_members["eligible_for_ranking"]), "symbol"]
            .astype(str)
            .drop_duplicates()
        )
        active_values = pd.to_numeric(
            day_outcomes.reindex(active_symbols), errors="coerce"
        )
        daily_rows.append(
            {
                "decision_date": date,
                "strategy": "all_active_theme",
                "gross_return": float(active_values.dropna().mean())
                if active_values.notna().any()
                else np.nan,
                "slots_target": max(1, len(active_symbols)),
                "slots_filled": len(active_symbols),
                "missing_outcomes": int(active_values.isna().sum()),
                "future_return_coverage": float(active_values.notna().mean())
                if len(active_values)
                else np.nan,
            }
        )

        chain_returns: list[float] = []
        chain_valid_counts = 0
        chain_member_counts = 0
        for chain in day_chains.to_dict("records"):
            chain_id = str(chain["chain_id"])
            chain_return, member_count, valid_count = _member_chain_return(
                day_members, chain_id, day_outcomes
            )
            chain_returns.append(chain_return)
            chain_member_counts += member_count
            chain_valid_counts += valid_count
        control_gross = (
            float(np.sum(chain_returns) / CONTROL_CHAIN_SLOTS)
            if chain_returns and np.isfinite(chain_returns).all()
            else 0.0
            if not chain_returns
            else np.nan
        )
        for strategy in ("top3_chain_all_members", "direction_only"):
            daily_rows.append(
                {
                    "decision_date": date,
                    "strategy": strategy,
                    "gross_return": control_gross,
                    "slots_target": CONTROL_CHAIN_SLOTS,
                    "slots_filled": len(chain_returns),
                    "missing_outcomes": chain_member_counts - chain_valid_counts,
                    "future_return_coverage": (
                        chain_valid_counts / chain_member_counts
                        if chain_member_counts
                        else np.nan
                    ),
                }
            )

        random = _random_same_chain_draws(
            date,
            day_members,
            day_chains,
            day_outcomes,
            repeats=random_draws,
            seed=seed + int(date.strftime("%Y%m%d")),
        )
        draw_parts.append(random)
        valid_draws = random.loc[random["gross_return"].notna()]
        filled_values = random["slots_filled"].unique()
        if len(filled_values) != 1:
            raise R1RunError(
                f"random control has non-constant filled slots on {date.date()}"
            )
        daily_rows.append(
            {
                "decision_date": date,
                "strategy": "top3_chain_random_4",
                "gross_return": float(valid_draws["gross_return"].mean())
                if len(valid_draws)
                else np.nan,
                "slots_target": TARGET_SLOTS,
                "slots_filled": int(filled_values[0]),
                "missing_outcomes": int(random["missing_outcomes"].sum()),
                "future_return_coverage": len(valid_draws) / len(random),
                "random_draws": len(random),
                "random_draw_return_std": float(valid_draws["gross_return"].std(ddof=1))
                if len(valid_draws) > 1
                else np.nan,
            }
        )

        for variant, selection in selections.items():
            day_positions = selection.positions.loc[
                selection.positions["decision_date"].eq(date)
            ] if not selection.positions.empty else pd.DataFrame()
            gross, filled, missing = _weighted_selected_return(
                day_positions, outcomes, return_column
            )
            daily_rows.append(
                {
                    "decision_date": date,
                    "strategy": variant,
                    "gross_return": gross,
                    "slots_target": TARGET_SLOTS,
                    "slots_filled": filled,
                    "missing_outcomes": missing,
                    "future_return_coverage": 1.0 - missing / filled if filled else 1.0,
                }
            )

        r1_positions = selections["R1"].positions
        day_r1 = r1_positions.loc[r1_positions["decision_date"].eq(date)]
        for chain in day_chains.to_dict("records"):
            chain_id = str(chain["chain_id"])
            selected = day_r1.loc[day_r1["chain_id"].astype(str).eq(chain_id)]
            selected_values = pd.to_numeric(
                day_outcomes.reindex(selected["symbol"].astype(str)), errors="coerce"
            )
            r1_contribution = (
                float(selected_values.sum() / TARGET_SLOTS)
                if selected_values.notna().all()
                else np.nan
            )
            member_return, _, valid_count = _member_chain_return(
                day_members, chain_id, day_outcomes
            )
            control_contribution = (
                member_return / CONTROL_CHAIN_SLOTS
                if pd.notna(member_return)
                else np.nan
            )
            cost_rate = 0.0012 if date < pd.Timestamp("2023-08-28") else 0.0007
            r1_cost = cost_rate * len(selected) / TARGET_SLOTS
            control_cost = cost_rate / CONTROL_CHAIN_SLOTS
            attribution_rows.append(
                {
                    "decision_date": date,
                    "chain_id": chain_id,
                    "chain_name": chain.get("chain_name", ""),
                    "r1_selected_names": len(selected),
                    "control_valid_names": valid_count,
                    "r1_gross_contribution": r1_contribution,
                    "control_gross_contribution": control_contribution,
                    "r1_net_contribution": r1_contribution - r1_cost,
                    "control_net_contribution": control_contribution - control_cost,
                    "net_excess_contribution": (
                        r1_contribution
                        - r1_cost
                        - control_contribution
                        + control_cost
                    ),
                }
            )

    daily = pd.DataFrame(daily_rows)
    random_draw_table = pd.concat(draw_parts, ignore_index=True)
    attribution = pd.DataFrame(attribution_rows)
    valid_attribution = attribution.dropna(subset=["net_excess_contribution"])
    total_excess = valid_attribution["net_excess_contribution"].sum()
    if total_excess > 0:
        chain_sums = valid_attribution.groupby("chain_id")[
            "net_excess_contribution"
        ].sum()
        chain_contribution_share = float(chain_sums.max() / total_excess)
    else:
        chain_contribution_share = None
    return daily, random_draw_table, attribution, chain_contribution_share


def _coverage_table(
    base: pd.DataFrame,
    membership: pd.DataFrame,
    chain_scores: pd.DataFrame,
    selected_chains: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for date, day in base.groupby("decision_date", sort=True):
        active = membership.loc[membership["decision_date"].eq(date)]
        active_keys = active.merge(
            day[["symbol", "eligible_for_ranking"]],
            on="symbol",
            how="left",
            validate="many_to_one",
        )
        chains = chain_scores.loc[chain_scores["decision_date"].eq(date)]
        selected = selected_chains.loc[selected_chains["decision_date"].eq(date)]
        rows.append(
            {
                "decision_date": date,
                "eligible_stocks": int(_bool_series(day["eligible_for_ranking"]).sum()),
                "active_membership_rows": len(active),
                "active_unique_stocks": int(active["symbol"].nunique()),
                "eligible_active_membership_rows": int(
                    _bool_series(active_keys["eligible_for_ranking"]).sum()
                ),
                "eligible_active_unique_stocks": int(
                    active_keys.loc[
                        _bool_series(active_keys["eligible_for_ranking"]), "symbol"
                    ].nunique()
                ),
                "eligible_chains": int(_bool_series(chains["chain_eligible"]).sum()),
                "selected_chains": len(selected),
            }
        )
    return pd.DataFrame(rows)


def run_historical(args: argparse.Namespace) -> dict[str, object]:
    paths = ChoiceDataPaths(astock_root=args.astock_root)
    reports = args.reports or paths.reports_file
    opens, closes, amounts = load_daily_inputs(
        "2021-01-01",
        "2025-12-31",
        paths=paths,
        lookback_sessions=520,
        forward_sessions=20,
    )
    schedules = _schedules(opens.index)
    arm_specs = [
        ("monthly", 250),
        ("weekly", 250),
        ("monthly", 120),
        ("weekly", 120),
    ]
    all_daily: list[pd.DataFrame] = []
    all_summaries: list[pd.DataFrame] = []
    all_paired: list[pd.DataFrame] = []
    all_pair_summaries: list[pd.DataFrame] = []
    all_inference: list[pd.DataFrame] = []
    all_half_years: list[pd.DataFrame] = []
    all_gate_checks: list[pd.DataFrame] = []
    all_positions: list[pd.DataFrame] = []
    all_chain_audit: list[pd.DataFrame] = []
    all_chain_scores: list[pd.DataFrame] = []
    all_selected_chains: list[pd.DataFrame] = []
    all_coverage: list[pd.DataFrame] = []
    all_draws: list[pd.DataFrame] = []
    all_attribution: list[pd.DataFrame] = []
    gates: dict[str, object] = {}
    base_cache: dict[str, pd.DataFrame] = {}

    for frequency, ttl in arm_specs:
        dates, horizon, periods_per_year = schedules[frequency]
        arm = f"{frequency}_ttl{ttl}"
        print(f"starting historical arm {arm} ({len(dates)} dates)", flush=True)
        if frequency not in base_cache:
            cached = build_base_panel(dates, opens, closes, amounts)
            _require_columns(
                cached, HISTORICAL_FEATURE_COLUMNS, where="historical base panel"
            )
            base_cache[frequency] = cached.loc[:, HISTORICAL_FEATURE_COLUMNS].copy()
        (
            base,
            membership,
            chain_scores,
            selected_chains,
            _scored_variants,
            selections,
        ) = _prepare_historical_feature_side(
            dates,
            ttl_sessions=ttl,
            opens=opens,
            closes=closes,
            amounts=amounts,
            reports=reports,
            base_panel=base_cache[frequency],
        )
        daily, draws, attribution, chain_share = _historical_returns(
            base,
            membership,
            selected_chains,
            selections,
            opens,
            horizon=horizon,
            random_draws=args.random_draws,
            seed=args.seed,
        )
        evaluation = evaluate_chain_returns(
            daily,
            config=ChainEvaluationConfig(periods_per_year=periods_per_year),
            gate_config=R1GateConfig(),
            chain_contribution_share=chain_share,
        )
        inference = summarize_paired_net_excess_inference(
            evaluation.paired_daily,
            bootstrap_iterations=BOOTSTRAP_ITERATIONS,
            seed=args.seed,
        )
        frames = (
            (evaluation.daily, all_daily),
            (evaluation.strategy_summary, all_summaries),
            (evaluation.paired_daily, all_paired),
            (evaluation.paired_summary, all_pair_summaries),
            (inference, all_inference),
            (evaluation.half_year_summary, all_half_years),
            (evaluation.gate.checks, all_gate_checks),
            (chain_scores, all_chain_scores),
            (selected_chains, all_selected_chains),
            (_coverage_table(base, membership, chain_scores, selected_chains), all_coverage),
            (draws, all_draws),
            (attribution, all_attribution),
        )
        for frame, destination in frames:
            item = frame.copy()
            item.insert(0, "arm", arm)
            item.insert(1, "frequency", frequency)
            item.insert(2, "experiment_ttl_sessions", ttl)
            item.insert(3, "holding_horizon_sessions", horizon)
            destination.append(item)
        for variant, selection in selections.items():
            if not selection.positions.empty:
                positions = selection.positions.copy()
                positions.insert(0, "arm", arm)
                positions.insert(1, "frequency", frequency)
                positions.insert(2, "experiment_ttl_sessions", ttl)
                positions.insert(3, "holding_horizon_sessions", horizon)
                all_positions.append(positions)
            if not selection.chain_audit.empty:
                audit = selection.chain_audit.copy()
                audit.insert(0, "arm", arm)
                audit.insert(1, "frequency", frequency)
                audit.insert(2, "experiment_ttl_sessions", ttl)
                audit.insert(3, "holding_horizon_sessions", horizon)
                all_chain_audit.append(audit)
        gates[arm] = {
            "status": evaluation.gate.status,
            "all_protocol_checks_pass": evaluation.gate.all_protocol_checks_pass,
            "all_required_checks_evaluated": evaluation.gate.all_required_checks_evaluated,
            "daily_computable_checks_pass": evaluation.gate.daily_computable_checks_pass,
            "chain_contribution_share": chain_share,
            "official_return_gate": arm == "monthly_ttl250",
        }
        print(
            f"completed historical arm {arm}: gate={evaluation.gate.status}",
            flush=True,
        )

    outputs = {
        "historical_daily": pd.concat(all_daily, ignore_index=True),
        "historical_strategy_summary": pd.concat(all_summaries, ignore_index=True),
        "historical_paired_daily": pd.concat(all_paired, ignore_index=True),
        "historical_paired_summary": pd.concat(all_pair_summaries, ignore_index=True),
        "historical_inference": pd.concat(all_inference, ignore_index=True),
        "historical_half_year": pd.concat(all_half_years, ignore_index=True),
        "historical_gate_checks": pd.concat(all_gate_checks, ignore_index=True),
        "historical_positions": pd.concat(all_positions, ignore_index=True),
        "historical_chain_audit": pd.concat(all_chain_audit, ignore_index=True),
        "historical_chain_scores": pd.concat(all_chain_scores, ignore_index=True),
        "historical_selected_chains": pd.concat(all_selected_chains, ignore_index=True),
        "historical_coverage": pd.concat(all_coverage, ignore_index=True),
        "historical_random_draws": pd.concat(all_draws, ignore_index=True),
        "historical_chain_attribution": pd.concat(all_attribution, ignore_index=True),
    }
    for name, frame in outputs.items():
        suffix = ".parquet" if name in {"historical_positions", "historical_random_draws"} else ".csv"
        path = args.ledger_dir / f"trader_chain_r1_{name}{suffix}"
        (_atomic_parquet if suffix == ".parquet" else _atomic_csv)(frame, path)
    _atomic_json(gates, args.ledger_dir / "trader_chain_r1_historical_gates.json")
    return {
        "gates": gates,
        "outputs": {name: len(frame) for name, frame in outputs.items()},
        "reports": _file_signature(reports),
        "calendar_start": opens.index.min(),
        "calendar_end": opens.index.max(),
        "inference": {
            "sampling_unit": "decision_date",
            "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
            "bootstrap_confidence_level": 0.95,
            "sign_test_alternative": "greater",
            "seed": args.seed,
        },
    }


def run_inference(args: argparse.Namespace) -> dict[str, object]:
    """Backfill frozen decision-date inference from an existing paired ledger."""

    paired_path = args.ledger_dir / "trader_chain_r1_historical_paired_daily.csv"
    if not paired_path.exists():
        raise R1RunError(f"paired ledger does not exist: {paired_path}")
    paired = pd.read_csv(paired_path)
    grouping = [
        "arm",
        "frequency",
        "experiment_ttl_sessions",
        "holding_horizon_sessions",
    ]
    _require_columns(
        paired,
        [*grouping, "decision_date", "control_strategy", "paired_net_excess"],
        where="historical paired ledger",
    )
    outputs: list[pd.DataFrame] = []
    for keys, group in paired.groupby(grouping, sort=True, dropna=False):
        item = summarize_paired_net_excess_inference(
            group,
            bootstrap_iterations=BOOTSTRAP_ITERATIONS,
            seed=args.seed,
        )
        for index, (column, value) in enumerate(zip(grouping, keys)):
            item.insert(index, column, value)
        outputs.append(item)
    inference = pd.concat(outputs, ignore_index=True)
    output_path = args.ledger_dir / "trader_chain_r1_historical_inference.csv"
    _atomic_csv(inference, output_path)
    return {
        "outputs": {"historical_inference": len(inference)},
        "source": _file_signature(paired_path),
        "output": _file_signature(output_path),
        "sampling_unit": "decision_date",
        "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
        "bootstrap_confidence_level": 0.95,
        "sign_test_alternative": "greater",
        "seed": args.seed,
    }


def _read_behavior_feature_side(
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build H2 static-oracle features before any label file is read."""

    choice_columns = [
        "decision_date",
        "symbol",
        "feature_cutoff_date",
        "eligible_for_ranking",
        "full_eligible",
        "mom20",
        "mom60",
        "mom20_pct",
        "mom60_pct",
        "ma20_dist",
        "amt_ratio20",
        "amt_ratio20_pct",
        "liq20",
        "pv_pullback_raw",
        "pv_vcp_raw",
        "pv_acceleration_raw",
    ]
    oracle_columns = [
        "decision_date",
        "symbol",
        "report_fine_industry_pit",
        "sw_fine_static_oracle",
    ]
    base = pd.read_parquet(args.choice_panel, columns=choice_columns)
    oracle = pd.read_parquet(args.oracle_panel, columns=oracle_columns)
    base["decision_date"] = _date_series(base["decision_date"])
    oracle["decision_date"] = _date_series(oracle["decision_date"])
    if base.duplicated(["decision_date", "symbol"]).any():
        raise R1RunError("behavior choice panel keys are duplicated")
    if oracle.duplicated(["decision_date", "symbol"]).any():
        raise R1RunError("behavior oracle panel keys are duplicated")
    base = base.merge(
        oracle,
        on=["decision_date", "symbol"],
        how="left",
        validate="one_to_one",
    )
    if not _bool_series(base["eligible_for_ranking"]).equals(
        _bool_series(base["full_eligible"])
    ):
        raise R1RunError("behavior eligible_for_ranking and full_eligible drifted")

    mapping_columns = [
        "symbol",
        "theme_id",
        "theme_name",
        "oracle_static_source_count",
        "oracle_static_match_rules",
        "oracle_static_matched_keywords",
        "oracle_static_evidence_samples",
        "oracle_static_uses_future_metadata",
        "in_panel_universe",
        "mapping_audit_contains_labels",
    ]
    mapping = pd.read_csv(args.static_membership, usecols=mapping_columns)
    mapping = mapping.loc[_bool_series(mapping["in_panel_universe"])].copy()
    if _bool_series(mapping["mapping_audit_contains_labels"]).any():
        raise R1RunError("static membership audit unexpectedly contains labels")
    if not _bool_series(mapping["oracle_static_uses_future_metadata"]).all():
        raise R1RunError("H2 mapping must remain explicitly marked static/oracle")
    if mapping.duplicated(["symbol", "theme_id"]).any():
        raise R1RunError("static membership keys are duplicated")
    mapping = mapping.rename(
        columns={
            "theme_id": "chain_id",
            "theme_name": "chain_name",
            "oracle_static_source_count": "source_count",
        }
    )
    rules = mapping["oracle_static_match_rules"].astype("string").fillna("")
    mapping["evidence_purity"] = np.select(
        [rules.str.contains("direct", case=False), rules.str.contains("context", case=False)],
        [1.0, 0.5],
        default=0.0,
    )
    members = base.merge(mapping, on="symbol", how="inner", validate="many_to_many")
    if members.duplicated(["decision_date", "symbol", "chain_id"]).any():
        raise R1RunError("expanded static behavior membership keys are duplicated")
    report_segment = (
        members["report_fine_industry_pit"]
        .astype("string")
        .str.strip()
        .replace("", pd.NA)
    )
    sw_segment = (
        members["sw_fine_static_oracle"]
        .astype("string")
        .str.strip()
        .replace("", pd.NA)
    )
    members["subsegment"] = report_segment.fillna(sw_segment).fillna("unknown")

    chain_inputs = build_chain_direction_inputs(
        members, minimum_members=MINIMUM_CHAIN_MEMBERS
    )
    config = ChainRuleConfig()
    chain_scores = score_chain_directions(
        chain_inputs, CHAIN_DIRECTION_FACTORS, config
    )
    selected_chains = select_top_chains(chain_scores, config)
    scored = score_within_chains(
        members, STATIC_MEMBER_FAMILIES, config
    )
    return base, scored, selected_chains


def _behavior_select(
    scored: pd.DataFrame,
    selected_chains: pd.DataFrame,
    closes: pd.DataFrame,
) -> ChainPortfolioResult:
    config = ChainRuleConfig()
    positions: list[pd.DataFrame] = []
    audits: list[pd.DataFrame] = []
    summaries: list[pd.DataFrame] = []
    for date in sorted(scored["decision_date"].drop_duplicates()):
        day_chains = selected_chains.loc[selected_chains["decision_date"].eq(date)]
        chain_ids = set(day_chains["chain_id"].astype(str))
        day = scored.loc[scored["decision_date"].eq(date)].copy()
        symbols = day.loc[
            day["chain_id"].astype(str).isin(chain_ids)
            & _bool_series(day["eligible_for_ranking"])
            & pd.to_numeric(day["within_chain_score"], errors="coerce").notna(),
            "symbol",
        ]
        correlations = trailing_return_correlations(
            date, symbols, closes, lookback_sessions=60, minimum_observations=40
        )
        result = select_chain_portfolio(day, day_chains, correlations, config)
        if not result.positions.empty:
            positions.append(result.positions)
        if not result.chain_audit.empty:
            audits.append(result.chain_audit)
        if not result.date_summary.empty:
            summaries.append(result.date_summary)
    return ChainPortfolioResult(
        positions=pd.concat(positions, ignore_index=True) if positions else pd.DataFrame(),
        chain_audit=pd.concat(audits, ignore_index=True) if audits else pd.DataFrame(),
        date_summary=pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame(),
    )


def _behavior_rank_feature_side(
    base: pd.DataFrame,
    scored: pd.DataFrame,
    selected_chains: pd.DataFrame,
    portfolio: ChainPortfolioResult,
) -> pd.DataFrame:
    top = scored.merge(
        selected_chains[["decision_date", "chain_id", "chain_selection_rank"]],
        on=["decision_date", "chain_id"],
        how="inner",
        validate="many_to_one",
    )
    top = top.loc[
        _bool_series(top["eligible_for_ranking"])
        & pd.to_numeric(top["within_chain_score"], errors="coerce").notna()
    ].copy()
    top = top.sort_values(
        ["decision_date", "symbol", "within_chain_score", "chain_selection_rank", "chain_id"],
        ascending=[True, True, False, True, True],
        kind="mergesort",
    ).drop_duplicates(["decision_date", "symbol"], keep="first")
    score_lookup = top[[
        "decision_date",
        "symbol",
        "chain_id",
        "chain_name",
        "chain_selection_rank",
        "within_chain_score",
        "within_chain_rank",
        "neutral_fill_count",
        "subsegment",
    ]].rename(columns={"within_chain_score": "r1_behavior_score"})
    ranks: list[pd.DataFrame] = []
    for date, day in base.groupby("decision_date", sort=True):
        pool = day.loc[_bool_series(day["full_eligible"]), ["decision_date", "symbol"]]
        scored_day = score_lookup.loc[score_lookup["decision_date"].eq(date)].sort_values(
            ["r1_behavior_score", "chain_selection_rank", "symbol"],
            ascending=[False, True, True],
            kind="mergesort",
        ).copy()
        scored_day["r1_behavior_rank"] = np.arange(1, len(scored_day) + 1)
        scored_day["r1_behavior_top20"] = scored_day["r1_behavior_rank"].le(20)
        merged = pool.merge(
            scored_day,
            on=["decision_date", "symbol"],
            how="left",
            validate="one_to_one",
        )
        merged["r1_behavior_n_pool"] = len(pool)
        merged["r1_behavior_n_scored"] = len(scored_day)
        ranks.append(merged)
    result = pd.concat(ranks, ignore_index=True)
    selected_keys = pd.MultiIndex.from_frame(
        portfolio.positions[["decision_date", "symbol"]].drop_duplicates()
    ) if not portfolio.positions.empty else pd.MultiIndex.from_arrays([[], []])
    result_keys = pd.MultiIndex.from_frame(result[["decision_date", "symbol"]])
    result["r1_portfolio_selected"] = result_keys.isin(selected_keys)
    return result


def _load_behavior_labels_after_features(
    labels_path: Path,
    baseline_path: Path,
) -> pd.DataFrame:
    labels = pd.read_csv(labels_path)
    baseline = pd.read_csv(baseline_path)
    keys = ["event_id", "decision_date", "symbol", "label_confidence"]
    _require_columns(labels, keys, where="behavior labels")
    baseline_columns = [
        *keys,
        "baseline_rank",
        "baseline_penalty_rank",
        "baseline_top20_hit",
        "baseline_n_pool",
        "baseline_n_scored",
    ]
    _require_columns(baseline, baseline_columns, where="baseline ranks")
    labels["decision_date"] = _date_series(labels["decision_date"])
    baseline["decision_date"] = _date_series(baseline["decision_date"])
    if labels.duplicated(keys).any() or baseline.duplicated(keys).any():
        raise R1RunError("behavior labels or baseline rank keys are duplicated")
    return labels.merge(
        baseline[baseline_columns], on=keys, how="left", validate="one_to_one"
    )


def _behavior_gate(
    labels: pd.DataFrame,
    feature_ranks: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    audit = labels.merge(
        feature_ranks,
        on=["decision_date", "symbol"],
        how="left",
        validate="one_to_one",
    )
    audit["r1_in_full_pool"] = audit["r1_behavior_n_pool"].notna()
    audit["r1_behavior_scored"] = audit["r1_behavior_rank"].notna()
    audit["r1_behavior_top20"] = _bool_series(audit["r1_behavior_top20"])
    audit["r1_portfolio_selected"] = _bool_series(audit["r1_portfolio_selected"])
    audit["r1_behavior_penalty_rank"] = audit["r1_behavior_rank"].where(
        audit["r1_behavior_rank"].notna(),
        pd.to_numeric(audit["baseline_n_pool"], errors="coerce") + 1,
    )
    audit["penalty_rank_improvement"] = (
        pd.to_numeric(audit["baseline_penalty_rank"], errors="coerce")
        - pd.to_numeric(audit["r1_behavior_penalty_rank"], errors="coerce")
    )
    audit["incremental_top20_hit"] = audit["r1_behavior_top20"] & ~_bool_series(
        audit["baseline_top20_hit"]
    )
    daily = (
        audit.groupby("decision_date", sort=True)
        .agg(
            n_labels=("event_id", "size"),
            baseline_mean_penalty_rank=("baseline_penalty_rank", "mean"),
            r1_mean_penalty_rank=("r1_behavior_penalty_rank", "mean"),
            baseline_top20_hits=("baseline_top20_hit", lambda values: int(_bool_series(values).sum())),
            r1_top20_hits=("r1_behavior_top20", "sum"),
            r1_portfolio_hits=("r1_portfolio_selected", "sum"),
            incremental_top20_hits=("incremental_top20_hit", "sum"),
        )
        .reset_index()
    )
    daily["mean_penalty_rank_improved"] = (
        daily["r1_mean_penalty_rank"] < daily["baseline_mean_penalty_rank"]
    )
    total_hits = int(audit["r1_behavior_top20"].sum())
    hit_days = int(daily["r1_top20_hits"].gt(0).sum())
    improved_days = int(daily["mean_penalty_rank_improved"].sum())
    incremental = daily["incremental_top20_hits"]
    incremental_total = int(incremental.sum())
    max_incremental_share = (
        float(incremental.max() / incremental_total) if incremental_total else 0.0
    )
    checks = {
        "top20_hits": total_hits >= 4,
        "top20_hit_days": hit_days >= 4,
        "penalty_rank_improved_days": improved_days >= 6,
        "incremental_hit_concentration": max_incremental_share <= 0.5,
    }
    gate = {
        "status": "pass" if all(checks.values()) else "fail",
        "all_behavior_checks_pass": all(checks.values()),
        "checks": checks,
        "observed": {
            "top20_hits": total_hits,
            "top20_hit_days": hit_days,
            "penalty_rank_improved_days": improved_days,
            "incremental_top20_hits": incremental_total,
            "max_incremental_hit_day_share": max_incremental_share,
            "portfolio_hits": int(audit["r1_portfolio_selected"].sum()),
            "labels": len(audit),
            "decision_days": int(audit["decision_date"].nunique()),
        },
        "thresholds": {
            "minimum_top20_hits": 4,
            "minimum_top20_hit_days": 4,
            "minimum_penalty_rank_improved_days": 6,
            "maximum_incremental_hit_day_share": 0.5,
        },
    }
    return audit, daily, gate


def run_behavior(args: argparse.Namespace) -> dict[str, object]:
    # Feature-side construction is deliberately complete before this function
    # calls _load_behavior_labels_after_features below.
    base, scored, selected_chains = _read_behavior_feature_side(args)
    dates = pd.DatetimeIndex(sorted(base["decision_date"].drop_duplicates()))
    paths = ChoiceDataPaths(astock_root=args.astock_root)
    _opens, closes, _amounts = load_daily_inputs(
        dates.min(),
        dates.max(),
        paths=paths,
        lookback_sessions=520,
        forward_sessions=0,
    )
    portfolio = _behavior_select(scored, selected_chains, closes)
    feature_ranks = _behavior_rank_feature_side(
        base, scored, selected_chains, portfolio
    )
    # Labels and old rank outcomes are first read here, after features/ranks and
    # final positions are immutable in memory.
    labels = _load_behavior_labels_after_features(args.labels, args.baseline_ranks)
    audit, daily, gate = _behavior_gate(labels, feature_ranks)

    outputs = {
        "behavior_label_ranks": audit,
        "behavior_daily": daily,
        "behavior_positions": portfolio.positions,
        "behavior_chain_audit": portfolio.chain_audit,
        "behavior_selected_chains": selected_chains,
    }
    for name, frame in outputs.items():
        path = args.ledger_dir / f"trader_chain_r1_{name}.csv"
        _atomic_csv(frame, path)
    _atomic_json(gate, args.ledger_dir / "trader_chain_r1_behavior_gate.json")
    return {
        "gate": gate,
        "outputs": {name: len(frame) for name, frame in outputs.items()},
        "static_membership": _file_signature(args.static_membership),
        "feature_build_completed_before_label_load": True,
        "static_oracle_uses_future_metadata": True,
    }


def main() -> None:
    args = parse_args()
    if args.random_draws <= 0:
        raise R1RunError("--random-draws must be positive")
    args.ledger_dir.mkdir(parents=True, exist_ok=True)
    started = pd.Timestamp.utcnow()
    manifest_path = args.ledger_dir / "trader_chain_r1_manifest.json"
    result: dict[str, object] = {}
    if args.mode != "all" and manifest_path.exists():
        try:
            previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            previous = {}
        if previous.get("experiment") == "trader_chain_r1":
            old_results = previous.get("results", {})
            if isinstance(old_results, dict):
                result.update(old_results)
    if args.mode in {"all", "historical"}:
        result["historical"] = run_historical(args)
    if args.mode in {"all", "behavior"}:
        result["behavior"] = run_behavior(args)
    if args.mode == "inference":
        result["inference"] = run_inference(args)
    manifest = {
        "experiment": "trader_chain_r1",
        "protocol": _file_signature(PROTOCOL),
        "started_utc": started,
        "completed_utc": pd.Timestamp.utcnow(),
        "last_run_mode": args.mode,
        "seed": args.seed,
        "random_draws": args.random_draws,
        "rule_config": asdict(ChainRuleConfig()),
        "pit_main_ttl_sessions": 250,
        "pit_sensitivity_ttl_sessions": 120,
        "results": result,
        "machine_learning_allowed": bool(
            result.get("historical", {})
            .get("gates", {})
            .get("monthly_ttl250", {})
            .get("all_protocol_checks_pass", False)
        ),
    }
    _atomic_json(manifest, manifest_path)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
