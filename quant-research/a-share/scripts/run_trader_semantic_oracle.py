#!/usr/bin/env python3
"""Run the frozen static-semantic upper-bound experiment for trader choice.

This runner is intentionally diagnostic.  Current/static industry, technology
index, and public-text supply-chain classifications may be backfilled to the
eight historical decision days, but every numerical input remains T-1.  The
result can explain historical selection; it is not a forward trading model and
does not evaluate returns.

Feature construction is completed before the label ledger is read.  The
feature builders receive only an explicit allow-list of panel columns and
accept no label, event-id, forward-return, outcome, or PnL input.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import re
import sys
import tempfile
from collections import OrderedDict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from afactor.trader_choice import (  # noqa: E402
    ChoiceConfig,
    evaluate_ranking,
    score_factor_families,
)
from afactor.trader_choice_data import (  # noqa: E402
    ChoiceDataPaths,
    load_daily_inputs,
)
from afactor.trader_choice_industry import (  # noqa: E402
    IndustryFeaturePaths,
    add_industry_context_features,
)
from afactor.trader_choice_oracle import (  # noqa: E402
    STRUCTURED_SOURCE_COLUMNS,
    add_structured_industry_scores,
)
from afactor.trader_choice_semantic import (  # noqa: E402
    DEFAULT_STATIC_ORACLE_THEMES,
    StaticOracleSemanticConfig,
    add_static_oracle_semantic_features,
    build_static_oracle_theme_membership,
)


DEFAULT_LEDGER = ROOT / "research" / "ledger"
DEFAULT_PANEL = DEFAULT_LEDGER / "trader_choice_panel.parquet"
DEFAULT_LABELS = DEFAULT_LEDGER / "trader_choice_labels.csv"
DEFAULT_PROTOCOL = ROOT / "research" / "TRADER_SEMANTIC_ORACLE_PROTOCOL.md"

POOL_COLUMN = "full_eligible"
STANDARD_KS = (5, 10, 20, 50, 100)
COVERAGE_KS = (5, 10, 20, 50, 100, 200, 500, 1000)
FROZEN_LABEL_COUNT = 20
FROZEN_DECISION_DAY_COUNT = 8
FROZEN_PANEL_ROW_COUNT = 44_520
FROZEN_BASELINE_TOP20_HITS = 2
FROZEN_BASELINE_TOP20_HIT_DAYS = 2
MIN_TOP20_HITS = 4
MIN_TOP20_HIT_DAYS = 4
MIN_RANK_IMPROVED_DAYS = 6
MAX_INCREMENTAL_HIT_DAY_SHARE = 0.5

BASELINE_FAMILIES: OrderedDict[str, tuple[str, ...]] = OrderedDict(
    [
        ("price_volume", ("pv_score",)),
        ("dynamic_direction", ("f2_direction_score",)),
        ("within_direction_leader", ("f3_leader_score",)),
        ("event_attention", ("f4_event_attention_score",)),
        ("familiarity", ("f5_familiarity_score",)),
        ("portfolio_replacement", ("f6_portfolio_replacement_score",)),
    ]
)

NEW_FAMILIES: OrderedDict[str, tuple[str, ...]] = OrderedDict(
    [
        ("structured_direction", ("structured_direction_gate_score",)),
        ("structured_leader", ("structured_leader_gate_score",)),
        ("tech_index_theme", ("tech_gate_score",)),
        ("semantic_chain", ("semantic_gate_score",)),
    ]
)

BASE_PANEL_COLUMNS = (
    "decision_date",
    "symbol",
    "eligible_for_ranking",
    POOL_COLUMN,
    "feature_cutoff_date",
    "mom5",
    "mom10",
    "mom20",
    "mom60",
    "ma20_dist",
    "amt_ratio20",
    "liq20",
    "pv_score",
    "f2_direction_score",
    "f3_leader_score",
    "f4_event_attention_score",
    "f5_familiarity_score",
    "f6_portfolio_replacement_score",
)

ORACLE_PANEL_AUDIT_COLUMNS = (
    *BASE_PANEL_COLUMNS,
    "report_fine_industry_pit",
    "report_industry_publish_time",
    "report_industry_effective_date",
    "report_industry_age_sessions",
    "sw_l1_static_oracle",
    "sw_fine_static_oracle",
    "sw_index_required_date",
    "sw_index_code",
    "sw_l1_ret5_t1",
    "sw_l1_ret20_t1",
    "sw_l1_amount_ratio20_t1",
    "sw_l1_amount_accel5_20_t1",
    "daily_basic_required_date",
    "db_turnover_rate_t1",
    "db_turnover_rate_f_t1",
    "db_volume_ratio_t1",
    "db_circ_mv_t1",
    "tech_index_codes_static_oracle",
    "tech_index_names_static_oracle",
    "tech_index_count_static_oracle",
    "tech_index_member_static_oracle",
    "tech_index_snapshot_date",
    "f7_structured_direction_score",
    "f7_structured_direction_taxonomy_count",
    "f7_structured_direction_missing",
    "structured_direction_gate_score",
    "structured_direction_gate_neutral_filled",
    "f8_structured_leader_score",
    "f8_structured_leader_taxonomy_count",
    "f8_structured_leader_missing",
    "structured_leader_gate_score",
    "structured_leader_gate_neutral_filled",
    "f9_tech_index_score",
    "f9_tech_index_component_count",
    "f9_tech_index_memberships_used",
    "f9_tech_index_missing",
    "tech_gate_score",
    "tech_gate_explicit_nonmember",
    "tech_gate_member_component_missing",
    "tech_gate_neutral_filled",
    "oracle_static_theme_count",
    "oracle_static_theme_ids",
    "oracle_static_theme_names",
    "oracle_static_best_theme_id",
    "oracle_static_best_theme_name",
    "oracle_static_theme_strength_raw",
    "oracle_static_theme_strength_pct",
    "oracle_static_theme_leader_score",
    "oracle_static_theme_x_leader",
    "f7_oracle_static_semantic_score",
    "oracle_static_semantic_missing",
    "semantic_gate_score",
    "semantic_gate_explicit_nonmember",
    "semantic_gate_member_component_missing",
    "semantic_gate_neutral_filled",
    "structured_factor_source_columns",
    "structured_factor_taxonomies",
    "structured_factor_label_event_pnl_columns_used",
    "structured_factor_asof_rule",
    "oracle_static_mapping_version",
    "oracle_static_asof_rule",
    "oracle_static_uses_future_metadata",
    "oracle_static_feature_clock",
)

LABEL_COLUMNS = (
    "decision_date",
    "symbol",
    "label_confidence",
    "event_id",
)

FORBIDDEN_FEATURE_EXACT = {
    "event_id",
    "label",
    "label_confidence",
    "selection_status",
    "selected",
    "is_selected",
    "is_buy",
    "buy_flag",
    "outcome",
    "pnl",
    "realized_pnl",
    "realised_pnl",
}
FORBIDDEN_FEATURE_PREFIXES = (
    "fwd_",
    "forward_",
    "selected_fwd_",
    "control_fwd_",
    "paired_diff_",
)

OUTPUT_NAMES = OrderedDict(
    [
        ("panel", "trader_oracle_panel.parquet"),
        ("daily", "trader_oracle_experiment_daily.csv"),
        ("summary", "trader_oracle_experiment_summary.csv"),
        ("label_ranks", "trader_oracle_label_ranks.csv"),
        ("theme_audit", "trader_oracle_theme_audit.csv"),
        ("tech_audit", "trader_oracle_tech_audit.csv"),
        ("feature_coverage", "trader_oracle_feature_coverage.csv"),
        ("coverage_curve", "trader_oracle_coverage_curve.csv"),
        ("gate", "trader_oracle_gate.json"),
        ("manifest", "trader_oracle_manifest.json"),
    ]
)


class OracleRunError(RuntimeError):
    """Raised when an oracle run would be ambiguous or non-auditable."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--ledger-dir", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--astock-root", type=Path, default=ROOT)
    parser.add_argument(
        "--sw-index-root",
        type=Path,
        default=IndustryFeaturePaths().sw_index_root,
    )
    parser.add_argument("--reports", type=Path)
    parser.add_argument("--stock-sw-industry", type=Path)
    parser.add_argument("--daily-basic", type=Path)
    parser.add_argument("--tech-constituents", type=Path)
    parser.add_argument("--min-group-size", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260831)
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
        raise OracleRunError(f"{where} is missing required columns: {missing}")


def _read_panel(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"choice panel does not exist: {path}")
    try:
        frame = pd.read_parquet(path)
    except ImportError as exc:
        raise OracleRunError(
            "reading trader_choice_panel.parquet requires pyarrow or fastparquet"
        ) from exc
    _require_columns(frame, BASE_PANEL_COLUMNS, where=str(path))
    result = frame.loc[:, BASE_PANEL_COLUMNS].copy()
    result["decision_date"] = _date_series(result["decision_date"])
    result["feature_cutoff_date"] = _date_series(result["feature_cutoff_date"])
    result["symbol"] = result["symbol"].astype("string").str.strip()
    if result[["decision_date", "symbol"]].isna().any(axis=None):
        raise OracleRunError("panel contains invalid decision_date or symbol values")
    if result["symbol"].eq("").any():
        raise OracleRunError("panel contains blank symbols")
    duplicated = result.duplicated(["decision_date", "symbol"], keep=False)
    if duplicated.any():
        examples = result.loc[duplicated, ["decision_date", "symbol"]].head(5)
        raise OracleRunError(
            "panel must be unique by (decision_date, symbol); examples="
            f"{examples.to_dict('records')}"
        )
    bad_cutoff = result["feature_cutoff_date"].isna() | result[
        "feature_cutoff_date"
    ].ge(result["decision_date"])
    if bad_cutoff.any():
        examples = result.loc[
            bad_cutoff,
            ["decision_date", "symbol", "feature_cutoff_date"],
        ].head(5)
        raise OracleRunError(
            "every numerical feature cutoff must be strictly earlier than its "
            f"decision date; examples={examples.to_dict('records')}"
        )
    result["eligible_for_ranking"] = _bool_series(result["eligible_for_ranking"])
    result[POOL_COLUMN] = _bool_series(result[POOL_COLUMN])
    inconsistent = result[POOL_COLUMN].ne(result["eligible_for_ranking"])
    if inconsistent.any():
        raise OracleRunError(
            "full_eligible and eligible_for_ranking differ; the frozen oracle "
            "experiment requires one complete ranking universe"
        )
    forbidden = sorted(
        column
        for column in result.columns
        if column in FORBIDDEN_FEATURE_EXACT
        or column.startswith(FORBIDDEN_FEATURE_PREFIXES)
    )
    if forbidden:
        raise OracleRunError(
            f"feature input contains forbidden label/outcome columns: {forbidden}"
        )
    if len(result) != FROZEN_PANEL_ROW_COUNT:
        raise OracleRunError(
            "frozen static-oracle panel requires exactly "
            f"{FROZEN_PANEL_ROW_COUNT} rows; observed {len(result)}"
        )
    return result.sort_values(
        ["decision_date", "symbol"], kind="mergesort"
    ).reset_index(drop=True)


def _load_calendar(
    decision_dates: Iterable[object], astock_root: Path
) -> pd.DatetimeIndex:
    dates = pd.DatetimeIndex(pd.to_datetime(list(decision_dates))).normalize()
    if not len(dates):
        raise OracleRunError("panel contains no decision dates")
    paths = ChoiceDataPaths(astock_root=astock_root)
    opens, closes, amounts = load_daily_inputs(
        dates.min(),
        dates.max(),
        paths=paths,
        lookback_sessions=520,
        forward_sessions=0,
    )
    calendar = pd.DatetimeIndex(opens.index).normalize().sort_values().unique()
    del opens, closes, amounts
    absent = dates.unique().difference(calendar)
    if len(absent):
        raise OracleRunError(
            "daily calendar is missing decision sessions: "
            + ", ".join(str(date.date()) for date in absent)
        )
    return calendar


def _source_paths(args: argparse.Namespace) -> IndustryFeaturePaths:
    paths = IndustryFeaturePaths(
        astock_root=args.astock_root,
        sw_index_root=args.sw_index_root,
    )
    return paths


def _resolved_sources(
    args: argparse.Namespace, paths: IndustryFeaturePaths
) -> dict[str, Path]:
    sources = {
        "reports": args.reports or paths.reports_file,
        "stock_sw_industry": args.stock_sw_industry or paths.static_sw_file,
        "sw_index_root": args.sw_index_root,
        "daily_basic": args.daily_basic or paths.daily_basic_file,
        "tech_constituents": args.tech_constituents or paths.tech_constituents_file,
    }
    missing = [f"{name}={path}" for name, path in sources.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "oracle source paths do not exist: " + "; ".join(missing)
        )
    return sources


def _build_features(
    panel: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    sources: Mapping[str, Path],
    paths: IndustryFeaturePaths,
    *,
    min_group_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build every feature without accepting or reading a label frame."""

    if min_group_size < 2:
        raise OracleRunError("min-group-size must be at least 2")
    context = add_industry_context_features(
        panel,
        calendar,
        reports=sources["reports"],
        stock_sw_industry=sources["stock_sw_industry"],
        sw_index_history=sources["sw_index_root"],
        daily_basic=sources["daily_basic"],
        tech_constituents=sources["tech_constituents"],
        paths=paths,
    )
    membership = build_static_oracle_theme_membership(
        sources["stock_sw_industry"],
        analyst_reports=sources["reports"],
        technology_index_constituents=sources["tech_constituents"],
        themes=DEFAULT_STATIC_ORACLE_THEMES,
    )
    semantic_config = StaticOracleSemanticConfig(theme_min_members=min_group_size)
    context = add_static_oracle_semantic_features(context, membership, semantic_config)
    context = add_structured_industry_scores(context, min_group_size=min_group_size)
    _validate_feature_clocks(context)
    _require_columns(
        context,
        [
            "f7_structured_direction_score",
            "f8_structured_leader_score",
            "f9_tech_index_score",
            "f7_oracle_static_semantic_score",
            "tech_index_member_static_oracle",
            "oracle_static_theme_count",
        ],
        where="oracle feature panel",
    )

    # Static classifications are treated as known for this authorized oracle
    # test.  Clear nonmembers are zero.  Any other missing raw new-family value
    # is 0.5 neutral in the ranking column, while the raw score and reason flag
    # stay untouched for coverage audit.  Thus every eligible stock receives
    # exactly four new-family values in the complete model.
    eligible = _bool_series(context[POOL_COLUMN])
    tech_member = _bool_series(context["tech_index_member_static_oracle"])
    theme_member = (
        pd.to_numeric(context["oracle_static_theme_count"], errors="coerce")
        .fillna(0)
        .gt(0)
    )

    context["structured_direction_gate_score"] = pd.to_numeric(
        context["f7_structured_direction_score"], errors="coerce"
    )
    context["structured_direction_gate_neutral_filled"] = (
        eligible & context["structured_direction_gate_score"].isna()
    )
    context.loc[
        context["structured_direction_gate_neutral_filled"],
        "structured_direction_gate_score",
    ] = 0.5

    context["structured_leader_gate_score"] = pd.to_numeric(
        context["f8_structured_leader_score"], errors="coerce"
    )
    context["structured_leader_gate_neutral_filled"] = (
        eligible & context["structured_leader_gate_score"].isna()
    )
    context.loc[
        context["structured_leader_gate_neutral_filled"],
        "structured_leader_gate_score",
    ] = 0.5

    context["tech_gate_score"] = pd.to_numeric(
        context["f9_tech_index_score"], errors="coerce"
    )
    context["tech_gate_explicit_nonmember"] = eligible & ~tech_member
    context["tech_gate_member_component_missing"] = (
        eligible & tech_member & context["tech_gate_score"].isna()
    )
    context["tech_gate_neutral_filled"] = context["tech_gate_member_component_missing"]
    context.loc[context["tech_gate_explicit_nonmember"], "tech_gate_score"] = 0.0
    context.loc[context["tech_gate_neutral_filled"], "tech_gate_score"] = 0.5

    context["semantic_gate_score"] = pd.to_numeric(
        context["f7_oracle_static_semantic_score"], errors="coerce"
    )
    context["semantic_gate_explicit_nonmember"] = eligible & ~theme_member
    context["semantic_gate_member_component_missing"] = (
        eligible & theme_member & context["semantic_gate_score"].isna()
    )
    context["semantic_gate_neutral_filled"] = context[
        "semantic_gate_member_component_missing"
    ]
    context.loc[context["semantic_gate_explicit_nonmember"], "semantic_gate_score"] = (
        0.0
    )
    context.loc[context["semantic_gate_neutral_filled"], "semantic_gate_score"] = 0.5

    for column in (
        "structured_direction_gate_score",
        "structured_leader_gate_score",
        "tech_gate_score",
        "semantic_gate_score",
    ):
        values = pd.to_numeric(context[column], errors="coerce")
        if np.isinf(values.to_numpy(dtype=float, na_value=np.nan)).any():
            raise OracleRunError(f"{column} contains infinite values")
        if values.loc[eligible].isna().any():
            raise OracleRunError(
                f"{column} must have a fixed value for every eligible stock"
            )
    return context, membership


def _validate_feature_clocks(panel: pd.DataFrame) -> None:
    """Require the joined numerical sources to use the exact panel T-1 row."""

    _require_columns(
        panel,
        [
            "decision_date",
            "feature_cutoff_date",
            "sw_index_required_date",
            "daily_basic_required_date",
            "report_industry_effective_date",
        ],
        where="oracle feature clock audit",
    )
    cutoff = _date_series(panel["feature_cutoff_date"])
    decision = _date_series(panel["decision_date"])
    for column in ("sw_index_required_date", "daily_basic_required_date"):
        observed = _date_series(panel[column])
        bad = observed.isna() | observed.ne(cutoff)
        if bad.any():
            examples = panel.loc[
                bad, ["decision_date", "symbol", "feature_cutoff_date", column]
            ].head(5)
            raise OracleRunError(
                f"{column} must equal the exact T-1 feature_cutoff_date; "
                f"examples={examples.to_dict('records')}"
            )
    effective = _date_series(panel["report_industry_effective_date"])
    has_report = panel.get(
        "report_fine_industry_pit", pd.Series(pd.NA, index=panel.index)
    ).notna()
    bad_report = has_report & (effective.isna() | effective.gt(decision))
    if bad_report.any():
        examples = panel.loc[
            bad_report,
            [
                "decision_date",
                "symbol",
                "report_fine_industry_pit",
                "report_industry_effective_date",
            ],
        ].head(5)
        raise OracleRunError(
            "report industry labels must be effective no later than the decision "
            f"session; examples={examples.to_dict('records')}"
        )


def _load_labels(path: Path, panel_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Load labels only after all feature construction has completed."""

    if not path.exists():
        raise FileNotFoundError(f"choice label ledger does not exist: {path}")
    labels = pd.read_csv(path, dtype={"symbol": "string", "event_id": "string"})
    _require_columns(labels, LABEL_COLUMNS, where=str(path))
    labels = labels.loc[:, LABEL_COLUMNS].copy()
    labels["decision_date"] = _date_series(labels["decision_date"])
    labels["symbol"] = labels["symbol"].astype("string").str.strip()
    labels["event_id"] = labels["event_id"].astype("string").str.strip()
    labels["label_confidence"] = (
        labels["label_confidence"].astype("string").str.strip().str.upper()
    )
    if labels[list(LABEL_COLUMNS)].isna().any(axis=None):
        raise OracleRunError("label ledger contains missing or invalid values")
    if labels["symbol"].eq("").any() or labels["event_id"].eq("").any():
        raise OracleRunError("label ledger contains blank symbol or event_id")
    if not labels["label_confidence"].eq("L4").all():
        raise OracleRunError("oracle evaluation accepts only frozen L4 labels")
    if labels.duplicated(["decision_date", "symbol"], keep=False).any():
        raise OracleRunError(
            "frozen oracle labels must be unique by (decision_date, symbol)"
        )
    if labels["event_id"].duplicated().any():
        raise OracleRunError("frozen oracle event_id values must be unique")
    labels = labels.sort_values(
        ["decision_date", "symbol", "event_id"], kind="mergesort"
    ).reset_index(drop=True)
    decision_days = labels["decision_date"].nunique()
    if len(labels) != FROZEN_LABEL_COUNT or decision_days != FROZEN_DECISION_DAY_COUNT:
        raise OracleRunError(
            "frozen static-oracle protocol requires exactly "
            f"{FROZEN_LABEL_COUNT} labels on {FROZEN_DECISION_DAY_COUNT} days; "
            f"observed {len(labels)} labels on {decision_days} days"
        )
    absent = pd.DatetimeIndex(labels["decision_date"].unique()).difference(panel_dates)
    if len(absent):
        raise OracleRunError(
            "labels contain dates absent from the panel: "
            + ", ".join(str(date.date()) for date in absent)
        )
    return labels


def _experiments() -> list[dict[str, object]]:
    baseline = OrderedDict(BASELINE_FAMILIES)
    new = OrderedDict(NEW_FAMILIES)
    experiments: list[dict[str, object]] = [
        {
            "experiment_order": 0,
            "experiment_kind": "baseline",
            "experiment_name": "baseline_f1_f6",
            "families": baseline,
        }
    ]
    order = 1
    for family_name in NEW_FAMILIES:
        experiments.append(
            {
                "experiment_order": order,
                "experiment_kind": "new_single_family",
                "experiment_name": f"only_{family_name}",
                "families": OrderedDict([(family_name, NEW_FAMILIES[family_name])]),
            }
        )
        order += 1
    for family_name in NEW_FAMILIES:
        families = OrderedDict(baseline)
        families[family_name] = NEW_FAMILIES[family_name]
        experiments.append(
            {
                "experiment_order": order,
                "experiment_kind": "baseline_plus_one",
                "experiment_name": f"baseline_plus_{family_name}",
                "families": families,
            }
        )
        order += 1
    all_families = OrderedDict(baseline)
    all_families.update(new)
    experiments.append(
        {
            "experiment_order": order,
            "experiment_kind": "complete_fixed",
            "experiment_name": "complete_f1_f10",
            "families": all_families,
        }
    )
    order += 1
    for omitted_family in NEW_FAMILIES:
        families = OrderedDict(
            (name, columns)
            for name, columns in all_families.items()
            if name != omitted_family
        )
        experiments.append(
            {
                "experiment_order": order,
                "experiment_kind": "complete_minus_one_new_family",
                "experiment_name": f"complete_minus_{omitted_family}",
                "families": families,
            }
        )
        order += 1
    experiments.append(
        {
            "experiment_order": order,
            "experiment_kind": "new_only_all",
            "experiment_name": "new_f7_f10_only",
            "families": new,
        }
    )
    if len(experiments) != 15:
        raise AssertionError(
            f"frozen experiment matrix must contain 15 rows, got {len(experiments)}"
        )
    return experiments


def _run_experiments(
    panel: pd.DataFrame,
    labels: pd.DataFrame,
    config: ChoiceConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    daily_parts: list[pd.DataFrame] = []
    scored_models: dict[str, pd.DataFrame] = {}
    for experiment in _experiments():
        families = experiment["families"]
        if not isinstance(families, Mapping):
            raise AssertionError("experiment families must be a mapping")
        scored = score_factor_families(
            panel,
            families,
            config,
            pool_col=POOL_COLUMN,
        )
        daily = evaluate_ranking(
            scored,
            labels,
            config,
            pool_col=POOL_COLUMN,
            score_col=config.score_col,
            ks=COVERAGE_KS,
        )
        daily.insert(0, "families", "|".join(families))
        daily.insert(0, "experiment_name", experiment["experiment_name"])
        daily.insert(0, "experiment_kind", experiment["experiment_kind"])
        daily.insert(0, "experiment_order", experiment["experiment_order"])
        family_count = (
            scored.loc[_bool_series(scored[POOL_COLUMN])]
            .groupby("decision_date", sort=True)["score_family_count"]
            .agg(
                min_score_family_count="min",
                median_score_family_count="median",
                max_score_family_count="max",
            )
            .reset_index()
        )
        daily = daily.merge(
            family_count,
            on="decision_date",
            how="left",
            validate="one_to_one",
        )
        daily["score_coverage"] = np.where(
            daily["n_pool"].gt(0), daily["n_scored"] / daily["n_pool"], np.nan
        )
        for k in COVERAGE_KS:
            daily[f"hits_at_{k}"] = np.rint(
                pd.to_numeric(daily[f"recall_at_{k}"], errors="coerce")
                * pd.to_numeric(daily["n_labels"], errors="coerce")
            ).astype("Int64")
        daily_parts.append(daily)
        if experiment["experiment_name"] in {
            "baseline_f1_f6",
            "complete_f1_f10",
        }:
            scored_models[str(experiment["experiment_name"])] = scored
    experiment_daily = pd.concat(daily_parts, ignore_index=True)
    experiment_daily = experiment_daily.sort_values(
        ["experiment_order", "decision_date"], kind="mergesort"
    ).reset_index(drop=True)
    summary = _aggregate_experiments(experiment_daily)
    return experiment_daily, summary, scored_models


def _aggregate_experiments(daily: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    keys = [
        "experiment_order",
        "experiment_kind",
        "experiment_name",
        "families",
    ]
    for key, group in daily.groupby(keys, sort=True, dropna=False):
        row = dict(zip(keys, key))
        weights = pd.to_numeric(group["n_labels"], errors="coerce").fillna(0.0)
        total_labels = int(weights.sum())
        row.update(
            {
                "n_days": int(group["decision_date"].nunique()),
                "n_labels": total_labels,
                "median_candidates": float(group["n_pool"].median()),
                "median_scored": float(group["n_scored"].median()),
                "mean_score_coverage": float(group["score_coverage"].mean()),
                "median_score_coverage": float(group["score_coverage"].median()),
                "median_min_score_family_count": float(
                    group["min_score_family_count"].median()
                ),
                "median_score_family_count": float(
                    group["median_score_family_count"].median()
                ),
                "median_max_score_family_count": float(
                    group["max_score_family_count"].median()
                ),
                "micro_pool_recall": float(
                    (group["pool_recall"] * weights).sum() / weights.sum()
                )
                if weights.sum()
                else np.nan,
                "hit_day_rate": float(group["hit_day"].mean()),
                "mean_mrr": float(group["mrr"].mean()),
                "median_selected_percentile": float(
                    group["median_selected_percentile"].median()
                ),
            }
        )
        for k in COVERAGE_KS:
            hits = int(pd.to_numeric(group[f"hits_at_{k}"], errors="coerce").sum())
            hit_days = int(_bool_series(group[f"hit_day_at_{k}"]).sum())
            row[f"total_hits_at_{k}"] = hits
            row[f"micro_recall_at_{k}"] = (
                hits / total_labels if total_labels else np.nan
            )
            row[f"hit_days_at_{k}"] = hit_days
            row[f"hit_day_rate_at_{k}"] = (
                hit_days / row["n_days"] if row["n_days"] else np.nan
            )
        rows.append(row)
    result = (
        pd.DataFrame(rows)
        .sort_values("experiment_order", kind="mergesort")
        .reset_index(drop=True)
    )
    baseline = result.loc[result["experiment_name"].eq("baseline_f1_f6")]
    if len(baseline) != 1:
        raise OracleRunError("baseline experiment summary is missing or duplicated")
    baseline_row = baseline.iloc[0]
    for k in COVERAGE_KS:
        result[f"delta_hits_at_{k}_vs_baseline"] = (
            result[f"total_hits_at_{k}"] - baseline_row[f"total_hits_at_{k}"]
        )
        result[f"delta_hit_days_at_{k}_vs_baseline"] = (
            result[f"hit_days_at_{k}"] - baseline_row[f"hit_days_at_{k}"]
        )
    return result


def _rank_labels(
    scored: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    model_prefix: str,
    config: ChoiceConfig,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for date, day_labels in labels.groupby("decision_date", sort=True):
        day = scored.loc[scored["decision_date"].eq(date)].copy()
        pool = _bool_series(day[POOL_COLUMN])
        pooled = day.loc[pool]
        ranked = pooled.loc[
            pd.to_numeric(pooled[config.score_col], errors="coerce").notna()
        ].copy()
        ranked["_score"] = pd.to_numeric(ranked[config.score_col], errors="coerce")
        ranked = ranked.sort_values(
            ["_score", "symbol"],
            ascending=[False, True],
            kind="mergesort",
        )
        ranks = {
            symbol: rank
            for rank, symbol in enumerate(ranked["symbol"].astype(str), start=1)
        }
        scores = ranked.set_index("symbol")["_score"].to_dict()
        pool_symbols = set(pooled["symbol"].astype(str))
        n_pool = int(len(pooled))
        n_scored = int(len(ranked))
        for label in day_labels.itertuples(index=False):
            symbol = str(label.symbol)
            rank = ranks.get(symbol)
            rank_percentile = (
                1.0
                if rank is not None and n_scored == 1
                else 1.0 - (rank - 1) / (n_scored - 1)
                if rank is not None and n_scored > 1
                else np.nan
            )
            rows.append(
                {
                    "event_id": label.event_id,
                    "decision_date": date,
                    "symbol": symbol,
                    "label_confidence": label.label_confidence,
                    f"{model_prefix}_in_pool": symbol in pool_symbols,
                    f"{model_prefix}_scored": rank is not None,
                    f"{model_prefix}_score": scores.get(symbol, np.nan),
                    f"{model_prefix}_rank": rank if rank is not None else pd.NA,
                    f"{model_prefix}_rank_percentile": rank_percentile,
                    f"{model_prefix}_penalty_rank": (
                        float(rank) if rank is not None else float(n_pool + 1)
                    ),
                    f"{model_prefix}_top20_hit": bool(rank is not None and rank <= 20),
                    f"{model_prefix}_n_pool": n_pool,
                    f"{model_prefix}_n_scored": n_scored,
                }
            )
    return pd.DataFrame(rows)


def _build_label_rank_audit(
    feature_panel: pd.DataFrame,
    labels: pd.DataFrame,
    scored_models: Mapping[str, pd.DataFrame],
    config: ChoiceConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline = _rank_labels(
        scored_models["baseline_f1_f6"],
        labels,
        model_prefix="baseline",
        config=config,
    )
    complete = _rank_labels(
        scored_models["complete_f1_f10"],
        labels,
        model_prefix="complete",
        config=config,
    )
    join_keys = ["event_id", "decision_date", "symbol", "label_confidence"]
    ranks = baseline.merge(
        complete,
        on=join_keys,
        how="outer",
        validate="one_to_one",
    )
    ranks["rank_improvement"] = (
        ranks["baseline_penalty_rank"] - ranks["complete_penalty_rank"]
    )
    ranks["new_top20_incremental_hit"] = _bool_series(
        ranks["complete_top20_hit"]
    ) & ~_bool_series(ranks["baseline_top20_hit"])
    audit_columns = [
        "decision_date",
        "symbol",
        "f7_structured_direction_score",
        "structured_direction_gate_score",
        "structured_direction_gate_neutral_filled",
        "f8_structured_leader_score",
        "structured_leader_gate_score",
        "structured_leader_gate_neutral_filled",
        "f9_tech_index_score",
        "tech_gate_score",
        "tech_gate_explicit_nonmember",
        "tech_gate_member_component_missing",
        "tech_gate_neutral_filled",
        "f7_oracle_static_semantic_score",
        "semantic_gate_score",
        "semantic_gate_explicit_nonmember",
        "semantic_gate_member_component_missing",
        "semantic_gate_neutral_filled",
        "report_fine_industry_pit",
        "sw_l1_static_oracle",
        "sw_fine_static_oracle",
        "tech_index_codes_static_oracle",
        "tech_index_names_static_oracle",
        "oracle_static_theme_ids",
        "oracle_static_theme_names",
        "oracle_static_best_theme_id",
        "oracle_static_best_theme_name",
    ]
    available_audit = [column for column in audit_columns if column in feature_panel]
    ranks = ranks.merge(
        feature_panel[available_audit],
        on=["decision_date", "symbol"],
        how="left",
        validate="one_to_one",
    )
    ranks = ranks.sort_values(
        ["decision_date", "symbol", "event_id"], kind="mergesort"
    ).reset_index(drop=True)

    daily_rows: list[dict[str, object]] = []
    for date, group in ranks.groupby("decision_date", sort=True):
        baseline_mean = float(group["baseline_penalty_rank"].mean())
        complete_mean = float(group["complete_penalty_rank"].mean())
        baseline_median = float(group["baseline_penalty_rank"].median())
        complete_median = float(group["complete_penalty_rank"].median())
        mean_improved = complete_mean < baseline_mean
        median_improved = complete_median < baseline_median
        daily_rows.append(
            {
                "decision_date": date,
                "n_labels": int(len(group)),
                "baseline_mean_penalty_rank": baseline_mean,
                "complete_mean_penalty_rank": complete_mean,
                "mean_rank_improved": mean_improved,
                "baseline_median_penalty_rank": baseline_median,
                "complete_median_penalty_rank": complete_median,
                "median_rank_improved": median_improved,
                "mean_or_median_rank_improved": mean_improved or median_improved,
                "incremental_top20_hits": int(
                    _bool_series(group["new_top20_incremental_hit"]).sum()
                ),
            }
        )
    return ranks, pd.DataFrame(daily_rows)


def _feature_coverage(panel: pd.DataFrame) -> pd.DataFrame:
    specs = [
        (
            "structured_direction",
            "f7_structured_direction_score",
            "structured_direction_gate_score",
            None,
            "structured_direction_gate_neutral_filled",
        ),
        (
            "structured_leader",
            "f8_structured_leader_score",
            "structured_leader_gate_score",
            None,
            "structured_leader_gate_neutral_filled",
        ),
        (
            "tech_index_theme",
            "f9_tech_index_score",
            "tech_gate_score",
            "tech_gate_explicit_nonmember",
            "tech_gate_neutral_filled",
        ),
        (
            "semantic_chain",
            "f7_oracle_static_semantic_score",
            "semantic_gate_score",
            "semantic_gate_explicit_nonmember",
            "semantic_gate_neutral_filled",
        ),
    ]
    rows: list[dict[str, object]] = []
    for date, day in panel.groupby("decision_date", sort=True):
        pool = _bool_series(day[POOL_COLUMN])
        n_pool = int(pool.sum())
        for family, raw_column, gate_column, nonmember_column, missing_column in specs:
            raw_available = (
                pool & pd.to_numeric(day[raw_column], errors="coerce").notna()
            )
            gate_available = (
                pool & pd.to_numeric(day[gate_column], errors="coerce").notna()
            )
            explicit_nonmember = (
                pool & _bool_series(day[nonmember_column])
                if nonmember_column is not None
                else pd.Series(False, index=day.index)
            )
            member_component_missing = (
                pool & _bool_series(day[missing_column])
                if missing_column is not None
                else pool & ~raw_available
            )
            rows.append(
                {
                    "decision_date": date,
                    "family": family,
                    "n_pool": n_pool,
                    "raw_available": int(raw_available.sum()),
                    "raw_coverage": raw_available.sum() / n_pool if n_pool else np.nan,
                    "explicit_nonmember": int(explicit_nonmember.sum()),
                    "zero_filled": int(explicit_nonmember.sum()),
                    "member_or_source_component_missing": int(
                        member_component_missing.sum()
                    ),
                    "neutral_filled": int(member_component_missing.sum()),
                    "gate_available": int(gate_available.sum()),
                    "gate_coverage": gate_available.sum() / n_pool
                    if n_pool
                    else np.nan,
                    "nonmember_zero_policy": bool(nonmember_column is not None),
                    "missing_neutral_policy": True,
                }
            )
    return pd.DataFrame(rows)


def _coverage_curve(experiment_daily: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    keys = [
        "experiment_order",
        "experiment_kind",
        "experiment_name",
        "families",
    ]
    for key, group in experiment_daily.groupby(keys, sort=True, dropna=False):
        base = dict(zip(keys, key))
        total_labels = int(pd.to_numeric(group["n_labels"], errors="coerce").sum())
        for k in COVERAGE_KS:
            hits = int(pd.to_numeric(group[f"hits_at_{k}"], errors="coerce").sum())
            hit_days = int(_bool_series(group[f"hit_day_at_{k}"]).sum())
            effective = np.minimum(k, pd.to_numeric(group["n_scored"], errors="coerce"))
            compression = effective / pd.to_numeric(
                group["n_pool"], errors="coerce"
            ).replace(0, np.nan)
            row = dict(base)
            row.update(
                {
                    "k": k,
                    "n_days": int(group["decision_date"].nunique()),
                    "n_labels": total_labels,
                    "total_hits": hits,
                    "micro_recall": hits / total_labels if total_labels else np.nan,
                    "hit_days": hit_days,
                    "hit_day_rate": hit_days / len(group) if len(group) else np.nan,
                    "median_pool": float(group["n_pool"].median()),
                    "median_scored": float(group["n_scored"].median()),
                    "median_score_coverage": float(group["score_coverage"].median()),
                    "median_score_family_count": float(
                        group["median_score_family_count"].median()
                    ),
                    "median_effective_top_k": float(np.median(effective)),
                    "median_top_k_fraction_of_pool": float(np.nanmedian(compression)),
                    "compression_denominator": POOL_COLUMN,
                }
            )
            rows.append(row)
    return (
        pd.DataFrame(rows)
        .sort_values(["experiment_order", "k"], kind="mergesort")
        .reset_index(drop=True)
    )


def _theme_definition_table() -> pd.DataFrame:
    rows = []
    for theme in DEFAULT_STATIC_ORACLE_THEMES:
        rows.append(
            {
                "theme_id": theme.theme_id,
                "theme_name": theme.theme_name,
                "theme_rule_keywords": "|".join(theme.keywords),
                "theme_rule_cooccurrence": json.dumps(
                    theme.cooccurrence_rules, ensure_ascii=False
                ),
                "theme_rule_exclusions": "|".join(theme.exclude_keywords),
            }
        )
    return pd.DataFrame(rows)


def _theme_audit(membership: pd.DataFrame, feature_panel: pd.DataFrame) -> pd.DataFrame:
    definitions = _theme_definition_table()
    universe = (
        feature_panel.groupby("symbol", sort=True)["decision_date"]
        .nunique()
        .rename("panel_decision_day_count")
        .reset_index()
    )
    audit = membership.merge(
        definitions,
        on=["theme_id", "theme_name"],
        how="outer",
        validate="many_to_one",
    )
    audit = audit.merge(universe, on="symbol", how="left", validate="many_to_one")
    audit["in_panel_universe"] = audit["panel_decision_day_count"].fillna(0).gt(0)
    audit["mapping_audit_contains_labels"] = False
    return audit.sort_values(
        ["theme_id", "symbol"], kind="mergesort", na_position="last"
    ).reset_index(drop=True)


def _tech_audit(feature_panel: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "symbol",
        "tech_index_codes_static_oracle",
        "tech_index_names_static_oracle",
        "tech_index_count_static_oracle",
        "tech_index_member_static_oracle",
        "tech_index_snapshot_date",
        "tech_index_static_asof_rule",
    ]
    available = [column for column in columns if column in feature_panel]
    audit = feature_panel[available].drop_duplicates("symbol", keep="first")
    if "tech_index_member_static_oracle" in audit:
        audit = audit.loc[_bool_series(audit["tech_index_member_static_oracle"])]
    return audit.sort_values("symbol", kind="mergesort").reset_index(drop=True)


def _oracle_panel_export(feature_panel: pd.DataFrame) -> pd.DataFrame:
    """Materialize a compact, label-free cross-section for rank verification."""

    dynamic_tech_columns = sorted(
        column
        for column in feature_panel.columns
        if column.startswith("tech_member__") and column.endswith("_static_oracle")
    )
    columns = list(
        dict.fromkeys(
            column
            for column in (*ORACLE_PANEL_AUDIT_COLUMNS, *dynamic_tech_columns)
            if column in feature_panel.columns
        )
    )
    _require_columns(
        feature_panel,
        [
            *BASE_PANEL_COLUMNS,
            "f7_structured_direction_score",
            "structured_direction_gate_score",
            "f8_structured_leader_score",
            "structured_leader_gate_score",
            "f9_tech_index_score",
            "tech_gate_score",
            "f7_oracle_static_semantic_score",
            "semantic_gate_score",
            "sw_l1_static_oracle",
            "oracle_static_theme_ids",
            "tech_index_codes_static_oracle",
        ],
        where="oracle panel export",
    )
    export = (
        feature_panel.loc[:, columns]
        .sort_values(["decision_date", "symbol"], kind="mergesort")
        .reset_index(drop=True)
    )
    if len(export) != FROZEN_PANEL_ROW_COUNT:
        raise OracleRunError(
            f"oracle panel export must retain {FROZEN_PANEL_ROW_COUNT} rows"
        )
    forbidden = sorted(
        column
        for column in export.columns
        if column in FORBIDDEN_FEATURE_EXACT
        or column.startswith(FORBIDDEN_FEATURE_PREFIXES)
    )
    if forbidden:
        raise OracleRunError(
            f"oracle panel export contains forbidden columns: {forbidden}"
        )
    return export


def _mapping_audit(
    membership: pd.DataFrame,
    feature_panel: pd.DataFrame,
) -> dict[str, object]:
    terms: list[str] = []
    for theme in DEFAULT_STATIC_ORACLE_THEMES:
        terms.extend(theme.keywords)
        terms.extend(theme.exclude_keywords)
        terms.extend(
            term
            for rule in theme.cooccurrence_rules
            for group in rule
            for term in group
        )
    code_terms = sorted(
        term
        for term in terms
        if re.fullmatch(r"\d{6}(?:\.(?:SH|SZ|BJ))?", str(term), re.I)
    )
    forbidden_mapping_columns = sorted(
        column
        for column in membership.columns
        if column in FORBIDDEN_FEATURE_EXACT
        or column.startswith(FORBIDDEN_FEATURE_PREFIXES)
    )
    forbidden_panel_columns = sorted(
        column
        for column in feature_panel.columns
        if column in FORBIDDEN_FEATURE_EXACT
        or column.startswith(FORBIDDEN_FEATURE_PREFIXES)
    )
    builder_parameters = {
        "add_industry_context_features": list(
            inspect.signature(add_industry_context_features).parameters
        ),
        "build_static_oracle_theme_membership": list(
            inspect.signature(build_static_oracle_theme_membership).parameters
        ),
        "add_static_oracle_semantic_features": list(
            inspect.signature(add_static_oracle_semantic_features).parameters
        ),
        "add_structured_industry_scores": list(
            inspect.signature(add_structured_industry_scores).parameters
        ),
    }
    builder_forbidden_parameters = {
        name: sorted(
            set(parameters)
            & {"labels", "events", "buys", "selected_symbols", "outcomes", "pnl"}
        )
        for name, parameters in builder_parameters.items()
    }
    structured_flag_ok = bool(
        "structured_factor_label_event_pnl_columns_used" in feature_panel
        and ~_bool_series(
            feature_panel["structured_factor_label_event_pnl_columns_used"]
        ).any()
    )
    source_whitelist_forbidden = sorted(
        column
        for column in STRUCTURED_SOURCE_COLUMNS
        if column in FORBIDDEN_FEATURE_EXACT
        or column.startswith(FORBIDDEN_FEATURE_PREFIXES)
    )
    allowed_mapping_sources = {
        "sw_static_snapshot",
        "analyst_report_static_text",
        "technology_index_static_snapshot",
    }
    observed_mapping_sources: set[str] = set()
    if "oracle_static_sources" in membership:
        for value in membership["oracle_static_sources"].dropna().astype(str):
            observed_mapping_sources.update(item for item in value.split("|") if item)
    unexpected_mapping_sources = sorted(
        observed_mapping_sources.difference(allowed_mapping_sources)
    )
    checks = {
        "no_six_digit_theme_terms": not code_terms,
        "membership_has_no_label_or_outcome_columns": not forbidden_mapping_columns,
        "feature_input_has_no_label_or_outcome_columns": not forbidden_panel_columns,
        "builder_signatures_accept_no_labels_events_or_outcomes": not any(
            builder_forbidden_parameters.values()
        ),
        "structured_source_whitelist_has_no_label_or_outcome_columns": not source_whitelist_forbidden,
        "structured_runtime_reports_no_label_event_pnl_use": structured_flag_ok,
        "mapping_sources_are_frozen_public_metadata_only": not unexpected_mapping_sources,
        "labels_loaded_after_feature_construction": True,
    }
    return {
        "pass": all(checks.values()),
        "checks": checks,
        "theme_definition_count": len(DEFAULT_STATIC_ORACLE_THEMES),
        "membership_rows": int(len(membership)),
        "membership_symbols": int(membership["symbol"].nunique())
        if "symbol" in membership
        else 0,
        "forbidden_theme_terms": code_terms,
        "forbidden_membership_columns": forbidden_mapping_columns,
        "forbidden_feature_columns": forbidden_panel_columns,
        "builder_parameters": builder_parameters,
        "builder_forbidden_parameters": builder_forbidden_parameters,
        "structured_source_columns": list(STRUCTURED_SOURCE_COLUMNS),
        "structured_source_whitelist_forbidden": source_whitelist_forbidden,
        "allowed_mapping_sources": sorted(allowed_mapping_sources),
        "observed_mapping_sources": sorted(observed_mapping_sources),
        "unexpected_mapping_sources": unexpected_mapping_sources,
    }


def _gate_payload(
    summary: pd.DataFrame,
    rank_daily: pd.DataFrame,
    label_ranks: pd.DataFrame,
    mapping_audit: Mapping[str, object],
) -> dict[str, object]:
    indexed = summary.set_index("experiment_name")
    baseline = indexed.loc["baseline_f1_f6"]
    complete = indexed.loc["complete_f1_f10"]
    baseline_hits = int(baseline["total_hits_at_20"])
    baseline_hit_days = int(baseline["hit_days_at_20"])
    if (
        baseline_hits != FROZEN_BASELINE_TOP20_HITS
        or baseline_hit_days != FROZEN_BASELINE_TOP20_HIT_DAYS
    ):
        raise OracleRunError(
            "frozen F1-F6 baseline drifted: expected Recall@20 "
            f"{FROZEN_BASELINE_TOP20_HITS}/{FROZEN_LABEL_COUNT} and hit-day "
            f"{FROZEN_BASELINE_TOP20_HIT_DAYS}/{FROZEN_DECISION_DAY_COUNT}; "
            f"observed {baseline_hits}/{FROZEN_LABEL_COUNT} and "
            f"{baseline_hit_days}/{FROZEN_DECISION_DAY_COUNT}"
        )
    complete_hits = int(complete["total_hits_at_20"])
    complete_hit_days = int(complete["hit_days_at_20"])
    mean_improved_days = int(_bool_series(rank_daily["mean_rank_improved"]).sum())
    median_improved_days = int(_bool_series(rank_daily["median_rank_improved"]).sum())
    rank_gate_improved_days = mean_improved_days
    incremental_by_day = label_ranks.groupby("decision_date", sort=True)[
        "new_top20_incremental_hit"
    ].apply(lambda values: int(_bool_series(values).sum()))
    incremental_total = int(incremental_by_day.sum())
    max_incremental = int(incremental_by_day.max()) if len(incremental_by_day) else 0
    concentration = (
        max_incremental / incremental_total if incremental_total > 0 else None
    )
    checks = {
        "recall_at_20": complete_hits >= MIN_TOP20_HITS,
        "hit_day_at_20": complete_hit_days >= MIN_TOP20_HIT_DAYS,
        "rank_improved_days": rank_gate_improved_days >= MIN_RANK_IMPROVED_DAYS,
        "incremental_hit_day_concentration": bool(
            concentration is not None and concentration <= MAX_INCREMENTAL_HIT_DAY_SHARE
        ),
        "mapping_no_whitelist_or_label_injection": bool(
            mapping_audit.get("pass", False)
        ),
    }
    return {
        "gate_name": "static_semantic_oracle_v1",
        "evaluation_model": "complete_f1_f10",
        "baseline_model": "baseline_f1_f6",
        "overall_pass": all(checks.values()),
        "checks": checks,
        "thresholds": {
            "recall_at_20_min_hits": MIN_TOP20_HITS,
            "recall_at_20_denominator": FROZEN_LABEL_COUNT,
            "hit_day_at_20_min_days": MIN_TOP20_HIT_DAYS,
            "hit_day_at_20_denominator": FROZEN_DECISION_DAY_COUNT,
            "rank_improved_min_days": MIN_RANK_IMPROVED_DAYS,
            "rank_improvement_metric": (
                "daily mean penalty rank across every L4 buy; missing rank = n_pool + 1"
            ),
            "max_incremental_top20_hit_share_from_one_day": MAX_INCREMENTAL_HIT_DAY_SHARE,
        },
        "observed": {
            "baseline_top20_hits": baseline_hits,
            "baseline_top20_hit_days": baseline_hit_days,
            "complete_top20_hits": complete_hits,
            "complete_recall_at_20": complete_hits / FROZEN_LABEL_COUNT,
            "complete_top20_hit_days": complete_hit_days,
            "complete_hit_day_rate_at_20": complete_hit_days
            / FROZEN_DECISION_DAY_COUNT,
            "mean_rank_improved_days": mean_improved_days,
            "median_rank_improved_days": median_improved_days,
            "rank_gate_improved_days": rank_gate_improved_days,
            "incremental_top20_hits": incremental_total,
            "max_incremental_top20_hits_from_one_day": max_incremental,
            "max_incremental_top20_hit_day_share": concentration,
            "incremental_top20_hits_by_day": {
                str(pd.Timestamp(date).date()): int(value)
                for date, value in incremental_by_day.items()
            },
        },
        "mapping_audit": dict(mapping_audit),
        "interpretation": (
            "Pass means static classifications can explain historical choice. "
            "It does not establish point-in-time availability, profitability, "
            "or a deployable forward strategy."
        ),
        "oracle_static_not_forward": True,
        "return_or_pnl_evaluation_performed": False,
    }


def _sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _file_signature(
    path: Path, *, full_hash_limit: int = 64 << 20
) -> dict[str, object]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    stat = path.stat()
    signature: dict[str, object] = {
        "path": str(path),
        "exists": True,
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }
    if stat.st_size <= full_hash_limit:
        signature["sha256"] = _sha256(path)
    else:
        with path.open("rb") as handle:
            head = handle.read(1 << 20)
            handle.seek(max(0, stat.st_size - (1 << 20)))
            tail = handle.read(1 << 20)
        signature["sha256_head_tail"] = hashlib.sha256(head + tail).hexdigest()
    return signature


def _directory_signature(directory: Path) -> dict[str, object]:
    if not directory.exists():
        return {"path": str(directory), "exists": False}
    entries = []
    for path in sorted(item for item in directory.rglob("*") if item.is_file()):
        signature = _file_signature(path)
        signature["relative_path"] = path.relative_to(directory).as_posix()
        entries.append(signature)
    canonical = json.dumps(entries, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return {
        "path": str(directory),
        "exists": True,
        "file_count": len(entries),
        "total_size": int(sum(int(entry["size"]) for entry in entries)),
        "sha256_inventory": hashlib.sha256(canonical).hexdigest(),
        "files": entries,
    }


def _json_default(value: object) -> object:
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if np.isnan(value) else float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            frame.to_csv(
                handle,
                index=False,
                date_format="%Y-%m-%d",
                float_format="%.12g",
                na_rep="",
            )
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _atomic_json(payload: Mapping[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = (
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            default=_json_default,
        )
        + "\n"
    )
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(text)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{path.name}.",
            suffix=".tmp.parquet",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
        try:
            frame.to_parquet(temporary, index=False)
        except ImportError as exc:
            raise OracleRunError(
                "writing trader_oracle_panel.parquet requires pyarrow or fastparquet"
            ) from exc
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _manifest(
    args: argparse.Namespace,
    sources: Mapping[str, Path],
    output_paths: Mapping[str, Path],
    frames: Mapping[str, pd.DataFrame],
    gate: Mapping[str, object],
) -> dict[str, object]:
    experiments = []
    for item in _experiments():
        families = item["families"]
        experiments.append(
            {
                "experiment_order": item["experiment_order"],
                "experiment_kind": item["experiment_kind"],
                "experiment_name": item["experiment_name"],
                "families": {
                    name: list(columns)
                    for name, columns in families.items()  # type: ignore[union-attr]
                },
            }
        )
    source_signatures: dict[str, object] = {}
    for name, path in sources.items():
        source_signatures[name] = (
            _directory_signature(path) if path.is_dir() else _file_signature(path)
        )
    code_paths = [
        ROOT / "src" / "afactor" / "trader_choice.py",
        ROOT / "src" / "afactor" / "trader_choice_data.py",
        ROOT / "src" / "afactor" / "trader_choice_industry.py",
        ROOT / "src" / "afactor" / "trader_choice_semantic.py",
        ROOT / "src" / "afactor" / "trader_choice_oracle.py",
        Path(__file__).resolve(),
        DEFAULT_PROTOCOL,
    ]
    fill_counts: dict[str, dict[str, int]] = {}
    coverage = frames.get("feature_coverage")
    if coverage is not None and not coverage.empty:
        for family, group in coverage.groupby("family", sort=True):
            fill_counts[str(family)] = {
                "raw_na_total": int(
                    (
                        pd.to_numeric(group["n_pool"], errors="coerce")
                        - pd.to_numeric(group["raw_available"], errors="coerce")
                    ).sum()
                ),
                "neutral_fill_reason_count": int(
                    pd.to_numeric(
                        group["member_or_source_component_missing"], errors="coerce"
                    ).sum()
                ),
                "neutral_0_5_filled": int(
                    pd.to_numeric(group["neutral_filled"], errors="coerce").sum()
                ),
                "explicit_nonmember_zero_filled": int(
                    pd.to_numeric(group["zero_filled"], errors="coerce").sum()
                ),
            }
    return {
        "manifest_schema": 1,
        "generated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "experiment_name": "trader_static_semantic_oracle",
        "oracle_static_not_forward": True,
        "feature_build_completed_before_label_load": True,
        "feature_builders_received_labels": False,
        "feature_input_allowlist": list(BASE_PANEL_COLUMNS),
        "feature_clock": (
            "current/static classification backfill authorized; all numerical "
            "features require feature_cutoff_date < decision_date"
        ),
        "new_family_missing_policy": {
            "raw_score_columns": "remain NA when source/group/T-1 components are insufficient",
            "structured_direction_gate_score": "raw NA = 0.5 neutral",
            "structured_leader_gate_score": "raw NA = 0.5 neutral",
            "tech_gate_score": "clear static-index nonmember = 0; member raw NA = 0.5 neutral",
            "semantic_gate_score": "clear nine-chain nonmember = 0; member raw NA = 0.5 neutral",
            "raw_scores_and_missing_flags_preserved": True,
            "eligible_new_family_denominator_is_fixed": True,
            "fill_counts": fill_counts,
        },
        "ranking": {
            "pool": POOL_COLUMN,
            "tie_break": "score desc, symbol asc",
            "family_weighting": "within-family equal, between-family equal",
            "standard_ks": list(STANDARD_KS),
            "coverage_ks": list(COVERAGE_KS),
        },
        "frozen_sample": {
            "panel_rows": FROZEN_PANEL_ROW_COUNT,
            "n_labels": FROZEN_LABEL_COUNT,
            "n_decision_days": FROZEN_DECISION_DAY_COUNT,
            "baseline_top20_hits": FROZEN_BASELINE_TOP20_HITS,
            "baseline_top20_hit_days": FROZEN_BASELINE_TOP20_HIT_DAYS,
        },
        "config": {
            "seed": args.seed,
            "min_group_size": args.min_group_size,
            "experiments": experiments,
        },
        "implementation_caveats": [
            (
                "structured_direction averages approved raw momentum inputs before "
                "group ranking; this is not literally a percentile transform of "
                "every continuous input first"
            ),
            (
                "structured_leader has no separately materialized 20-session "
                "leadership-persistence component"
            ),
            (
                "tech_index_count_static_oracle is audited but is not an explicit "
                "weighted component of f9_tech_index_score"
            ),
        ],
        "inputs": {
            "panel": _file_signature(args.panel),
            "labels": _file_signature(args.labels),
            "sources": source_signatures,
            "code_and_protocol": [_file_signature(path) for path in code_paths],
        },
        "row_counts": {name: int(len(frame)) for name, frame in frames.items()},
        "gate_overall_pass": bool(gate["overall_pass"]),
        "outputs": {
            name: _file_signature(path)
            for name, path in output_paths.items()
            if name != "manifest" and path.exists()
        },
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    # Phase 1: panel and public/static sources only.  Labels are deliberately
    # not read until _build_features returns.
    panel = _read_panel(args.panel)
    calendar = _load_calendar(panel["decision_date"].unique(), args.astock_root)
    industry_paths = _source_paths(args)
    sources = _resolved_sources(args, industry_paths)
    feature_panel, membership = _build_features(
        panel,
        calendar,
        sources,
        industry_paths,
        min_group_size=args.min_group_size,
    )

    # Phase 2: evaluation.  From this line onward labels may be used, but no
    # feature builder is called again.
    labels = _load_labels(
        args.labels,
        pd.DatetimeIndex(feature_panel["decision_date"].unique()),
    )
    config = ChoiceConfig(rank_ks=STANDARD_KS, random_seed=args.seed)
    experiment_daily, experiment_summary, scored_models = _run_experiments(
        feature_panel, labels, config
    )
    label_ranks, rank_daily = _build_label_rank_audit(
        feature_panel, labels, scored_models, config
    )
    # Rank-change fields describe only the fixed complete model versus the
    # baseline.  Keep them NA on the other ablations instead of repeating a
    # comparison that those rows did not perform.
    rank_daily_for_join = rank_daily.rename(
        columns={"n_labels": "rank_audit_n_labels"}
    )
    rank_lookup = rank_daily_for_join.set_index("decision_date")
    complete_rows = experiment_daily["experiment_name"].eq("complete_f1_f10")
    for column in rank_daily_for_join.columns.difference(["decision_date"]):
        mapped = experiment_daily.loc[complete_rows, "decision_date"].map(
            rank_lookup[column]
        )
        if pd.api.types.is_bool_dtype(rank_daily_for_join[column].dtype):
            output = pd.Series(
                pd.NA, index=experiment_daily.index, dtype="boolean"
            )
            output.loc[complete_rows] = mapped.astype("boolean").to_numpy()
        elif pd.api.types.is_numeric_dtype(rank_daily_for_join[column].dtype):
            output = pd.Series(np.nan, index=experiment_daily.index, dtype=float)
            output.loc[complete_rows] = pd.to_numeric(
                mapped, errors="coerce"
            ).to_numpy()
        else:
            output = pd.Series(pd.NA, index=experiment_daily.index, dtype="object")
            output.loc[complete_rows] = mapped.to_numpy()
        experiment_daily[column] = output
    feature_coverage = _feature_coverage(feature_panel)
    coverage_curve = _coverage_curve(experiment_daily)
    theme_audit = _theme_audit(membership, feature_panel)
    tech_audit = _tech_audit(feature_panel)
    oracle_panel = _oracle_panel_export(feature_panel)
    mapping_audit = _mapping_audit(membership, feature_panel)
    gate = _gate_payload(experiment_summary, rank_daily, label_ranks, mapping_audit)

    output_paths = {
        name: args.ledger_dir / filename for name, filename in OUTPUT_NAMES.items()
    }
    frames = {
        "oracle_panel": oracle_panel,
        "experiment_daily": experiment_daily,
        "experiment_summary": experiment_summary,
        "label_ranks": label_ranks,
        "theme_audit": theme_audit,
        "tech_audit": tech_audit,
        "feature_coverage": feature_coverage,
        "coverage_curve": coverage_curve,
    }
    _atomic_parquet(oracle_panel, output_paths["panel"])
    _atomic_csv(experiment_daily, output_paths["daily"])
    _atomic_csv(experiment_summary, output_paths["summary"])
    _atomic_csv(label_ranks, output_paths["label_ranks"])
    _atomic_csv(theme_audit, output_paths["theme_audit"])
    _atomic_csv(tech_audit, output_paths["tech_audit"])
    _atomic_csv(feature_coverage, output_paths["feature_coverage"])
    _atomic_csv(coverage_curve, output_paths["coverage_curve"])
    _atomic_json(gate, output_paths["gate"])
    manifest = _manifest(args, sources, output_paths, frames, gate)
    _atomic_json(manifest, output_paths["manifest"])
    return {
        "gate_pass": bool(gate["overall_pass"]),
        "top20_hits": gate["observed"]["complete_top20_hits"],
        "top20_hit_days": gate["observed"]["complete_top20_hit_days"],
        "rank_improved_days": gate["observed"]["rank_gate_improved_days"],
        "ledger_dir": str(args.ledger_dir),
    }


def main() -> None:
    args = parse_args()
    try:
        result = run(args)
    except (FileNotFoundError, OracleRunError, ValueError, KeyError, TypeError) as exc:
        raise SystemExit(f"semantic oracle failed: {exc}") from exc
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
