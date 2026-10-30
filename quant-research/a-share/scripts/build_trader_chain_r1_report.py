#!/usr/bin/env python3
"""Build the canonical portable-report artifact for trader-chain R1."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "research" / "ledger"
DEFAULT_ARTIFACT = ROOT / "research" / "trader_chain_r1_artifact.json"
REPORT_GENERATED_AT = "2026-08-31T05:22:45Z"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger-dir", type=Path, default=LEDGER)
    parser.add_argument("--output", type=Path, default=DEFAULT_ARTIFACT)
    return parser.parse_args()


def _atomic_json(payload: Mapping[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".json",
        prefix=f".{path.stem}.",
        dir=path.parent,
        encoding="utf-8",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def _number(value: object) -> float | int | None:
    if pd.isna(value):
        return None
    numeric = float(value)
    if not np.isfinite(numeric):
        return None
    return int(numeric) if numeric.is_integer() else numeric


def _records(frame: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for record in frame.to_dict("records"):
        rows.append(
            {
                key: (
                    value.strftime("%Y-%m-%d")
                    if isinstance(value, pd.Timestamp)
                    else bool(value)
                    if isinstance(value, (bool, np.bool_))
                    else _number(value)
                    if isinstance(value, (int, float, np.integer, np.floating))
                    else None
                    if pd.isna(value)
                    else str(value)
                )
                for key, value in record.items()
            }
        )
    return rows


def _bool_series(values: pd.Series) -> pd.Series:
    """Parse booleans without treating the string ``False`` as truthy."""

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


def _source(
    source_id: str,
    label: str,
    path: str,
    description: str,
    *,
    sql: str | None = None,
    tables_used: list[str] | None = None,
    filters: list[str] | None = None,
    definitions: Mapping[str, str] | None = None,
) -> dict[str, object]:
    source: dict[str, object] = {
        "id": source_id,
        "label": label,
        "path": path,
    }
    if sql is not None:
        query: dict[str, object] = {
            "engine": "duckdb",
            "language": "sql",
            "sql": sql,
            "description": description,
            "executed_at": REPORT_GENERATED_AT,
            "tables_used": tables_used or [path],
        }
        if filters:
            query["filters"] = filters
        if definitions:
            query["metric_definitions"] = dict(definitions)
        source["query"] = query
    return source


def build_artifact(ledger: Path) -> dict[str, object]:
    strategy = pd.read_csv(ledger / "trader_chain_r1_historical_strategy_summary.csv")
    paired = pd.read_csv(ledger / "trader_chain_r1_historical_paired_summary.csv")
    paired_daily = pd.read_csv(ledger / "trader_chain_r1_historical_paired_daily.csv")
    half_year = pd.read_csv(ledger / "trader_chain_r1_historical_half_year.csv")
    gate_checks = pd.read_csv(ledger / "trader_chain_r1_historical_gate_checks.csv")
    coverage = pd.read_csv(ledger / "trader_chain_r1_historical_coverage.csv")
    daily = pd.read_csv(ledger / "trader_chain_r1_historical_daily.csv")
    behavior_daily = pd.read_csv(ledger / "trader_chain_r1_behavior_daily.csv")
    behavior_ranks = pd.read_csv(ledger / "trader_chain_r1_behavior_label_ranks.csv")
    behavior_gate_payload = json.loads(
        (ledger / "trader_chain_r1_behavior_gate.json").read_text(encoding="utf-8")
    )
    chain_audit = pd.read_csv(ledger / "trader_chain_r1_historical_chain_audit.csv")
    positions = pd.read_parquet(ledger / "trader_chain_r1_historical_positions.parquet")
    attribution = pd.read_csv(ledger / "trader_chain_r1_historical_chain_attribution.csv")
    inference = pd.read_csv(ledger / "trader_chain_r1_historical_inference.csv")
    labels = pd.read_csv(ledger / "trader_choice_labels.csv")
    theme_audit = pd.read_csv(ledger / "trader_oracle_theme_audit.csv")

    net = strategy.loc[strategy["return_basis"].eq("net_historical_cost")].copy()
    main = net.loc[net["arm"].eq("monthly_ttl250")].copy()
    main_pair = paired.loc[
        paired["arm"].eq("monthly_ttl250")
        & paired["control_strategy"].eq("top3_chain_all_members")
    ].iloc[0]
    behavior_hits = int(_bool_series(behavior_ranks["r1_behavior_top20"]).sum())
    mapped_symbols = set(
        theme_audit.loc[_bool_series(theme_audit["in_panel_universe"]), "symbol"]
        .astype(str)
        .str.strip()
    )
    theme_coverage_hits = int(labels["symbol"].astype(str).str.strip().isin(mapped_symbols).sum())
    label_eligibility = labels[["event_id", "symbol"]].merge(
        behavior_ranks[["event_id", "r1_in_full_pool"]],
        on="event_id",
        how="left",
        validate="one_to_one",
    )
    eligible_theme_coverage_hits = int(
        (
            label_eligibility["symbol"].astype(str).str.strip().isin(mapped_symbols)
            & _bool_series(label_eligibility["r1_in_full_pool"])
        ).sum()
    )

    display_order = {
        "all_eligible": (1, "全市场合格池"),
        "all_active_theme": (2, "全部活跃链成员"),
        "top3_chain_all_members": (3, "Top-3链全成员"),
        "top3_chain_random_4": (4, "同链随机4只"),
        "R1_no_diversification": (5, "R1去分散约束"),
        "R1_minus_shape": (6, "R1去形态族"),
        "R1_minus_evidence": (7, "R1去证据族"),
        "R1": (8, "完整R1"),
    }
    main_chart = main.loc[main["strategy"].isin(display_order)].copy()
    main_chart["order"] = main_chart["strategy"].map(
        lambda value: display_order[str(value)][0]
    )
    main_chart["strategy_label"] = main_chart["strategy"].map(
        lambda value: display_order[str(value)][1]
    )
    main_chart = main_chart.sort_values("order")
    main_chart = main_chart[
        [
            "strategy_label",
            "annualized_return",
            "compound_return",
            "max_drawdown",
            "mean_invested_fraction",
        ]
    ]

    arm_labels = {
        "monthly_ttl250": "月频·TTL250",
        "monthly_ttl120": "月频·TTL120",
        "weekly_ttl250": "周频·TTL250",
        "weekly_ttl120": "周频·TTL120",
    }
    sensitivity_strategies = {
        "all_eligible": "全市场",
        "top3_chain_all_members": "Top-3链全成员",
        "top3_chain_random_4": "同链随机4只",
        "R1": "完整R1",
    }
    sensitivity = net.loc[net["strategy"].isin(sensitivity_strategies)].copy()
    sensitivity["arm_label"] = sensitivity["arm"].map(arm_labels)
    sensitivity["series"] = sensitivity["strategy"].map(sensitivity_strategies)
    sensitivity["arm_order"] = sensitivity["arm"].map(
        {name: index for index, name in enumerate(arm_labels, start=1)}
    )
    sensitivity["series_order"] = sensitivity["strategy"].map(
        {name: index for index, name in enumerate(sensitivity_strategies, start=1)}
    )
    sensitivity = sensitivity.sort_values(["arm_order", "series_order"])[
        ["arm_label", "series", "annualized_return", "max_drawdown"]
    ]

    half = half_year.loc[half_year["arm"].eq("monthly_ttl250")].copy()
    half["control"] = half["control_strategy"].map(
        {
            "top3_chain_all_members": "Top-3链全成员",
            "top3_chain_random_4": "同链随机4只",
        }
    )
    half = half[
        [
            "half_year",
            "control",
            "net_relative_compound_excess",
            "paired_net_win_rate",
            "n_periods",
        ]
    ].sort_values(["half_year", "control"])

    main_inference = inference.loc[inference["arm"].eq("monthly_ttl250")].copy()
    main_inference["control"] = main_inference["control_strategy"].map(
        {
            "top3_chain_all_members": "Top-3链全成员",
            "top3_chain_random_4": "同链随机4只",
        }
    )
    main_inference["positive_days"] = (
        main_inference["sign_positive_decision_dates"].astype(int).astype(str)
        + "/"
        + main_inference["sign_nonzero_decision_dates"].astype(int).astype(str)
    )
    inference_lookup = main_inference.set_index("control_strategy")
    member_inference = inference_lookup.loc["top3_chain_all_members"]
    random_inference = inference_lookup.loc["top3_chain_random_4"]
    main_inference = main_inference[
        [
            "control",
            "n_evaluable_decision_dates",
            "mean_paired_net_excess",
            "bootstrap_ci_lower",
            "bootstrap_ci_upper",
            "positive_days",
            "sign_test_pvalue_greater",
        ]
    ]

    behavior = behavior_daily.copy()
    behavior["decision_date"] = pd.to_datetime(behavior["decision_date"])
    behavior = behavior[
        [
            "decision_date",
            "n_labels",
            "baseline_top20_hits",
            "r1_top20_hits",
            "r1_portfolio_hits",
            "baseline_mean_penalty_rank",
            "r1_mean_penalty_rank",
            "mean_penalty_rank_improved",
        ]
    ]
    behavior_gate_specs = [
        ("top20_hits", "Top-20命中笔数", "top20_hits", "minimum_top20_hits", ">="),
        ("top20_hit_days", "出现Top-20命中的买入日", "top20_hit_days", "minimum_top20_hit_days", ">="),
        (
            "penalty_rank_improved_days",
            "平均惩罚排名优于旧基线的买入日",
            "penalty_rank_improved_days",
            "minimum_penalty_rank_improved_days",
            ">=",
        ),
        (
            "incremental_hit_concentration",
            "新增Top-20命中的最大单日占比",
            "max_incremental_hit_day_share",
            "maximum_incremental_hit_day_share",
            "<=",
        ),
    ]
    behavior_gate_rows: list[dict[str, object]] = []
    for check_id, label, observed_key, threshold_key, operator in behavior_gate_specs:
        passed = bool(behavior_gate_payload["checks"][check_id])
        status = "通过" if passed else "未通过"
        if check_id == "incremental_hit_concentration" and int(
            behavior_gate_payload["observed"]["incremental_top20_hits"]
        ) == 0:
            status = "通过（无新增命中）"
        behavior_gate_rows.append(
            {
                "check": label,
                "observed": behavior_gate_payload["observed"][observed_key],
                "threshold": f"{operator} {behavior_gate_payload['thresholds'][threshold_key]}",
                "status": status,
            }
        )
    behavior_gate = pd.DataFrame(behavior_gate_rows)

    main_gate_ids = [
        "minimum_decisions",
        "covers_each_year",
        "covers_all_half_years",
        "net_excess_positive__top3_chain_all_members",
        "net_excess_positive__top3_chain_random_4",
        "positive_half_years_both_controls",
        "main_control_annualized_net_excess",
        "main_control_information_ratio",
        "main_control_mdd_degradation",
        "double_cost_excess_positive__top3_chain_all_members",
        "double_cost_excess_positive__top3_chain_random_4",
        "half_year_contribution_concentration",
        "chain_contribution_concentration",
    ]
    gate_labels = {
        "minimum_decisions": "双对照配对决策日数量",
        "covers_each_year": "覆盖冻结区间每个年份",
        "covers_all_half_years": "两类对照覆盖全部10个半年",
        "net_excess_positive__top3_chain_all_members": "相对Top-3链全成员累计净超额为正",
        "net_excess_positive__top3_chain_random_4": "相对同链随机4只累计净超额为正",
        "positive_half_years_both_controls": "同时跑赢两类对照的半年数",
        "main_control_annualized_net_excess": "相对主对照年化净超额",
        "main_control_information_ratio": "逐决策期配对净超额信息比率",
        "main_control_mdd_degradation": "决策期收益序列回撤相对主对照恶化",
        "double_cost_excess_positive__top3_chain_all_members": "双倍成本下相对主对照净超额为正",
        "double_cost_excess_positive__top3_chain_random_4": "双倍成本下相对随机对照净超额为正",
        "half_year_contribution_concentration": "最大单半年净超额贡献占比",
        "chain_contribution_concentration": "最大单产业链净超额贡献占比",
    }
    gate = gate_checks.loc[
        gate_checks["arm"].eq("monthly_ttl250")
        & gate_checks["check_id"].isin(main_gate_ids),
        ["check_id", "description", "observed", "threshold", "evaluated", "status"],
    ].copy()
    gate["check_order"] = gate["check_id"].map(
        {name: index for index, name in enumerate(main_gate_ids)}
    )
    gate = gate.sort_values("check_order").drop(columns="check_order")
    gate["description"] = gate["check_id"].map(gate_labels).fillna(gate["description"])
    gate["status_label"] = gate["status"].map(
        {"pass": "通过", "fail": "未通过", "not_evaluated": "不可评价"}
    )
    all_required_checks_evaluated = bool(_bool_series(gate["evaluated"]).all())

    main_coverage = coverage.loc[coverage["arm"].eq("monthly_ttl250")]
    main_daily = daily.loc[daily["arm"].eq("monthly_ttl250")]
    main_paired_daily = paired_daily.loc[
        paired_daily["arm"].eq("monthly_ttl250")
        & paired_daily["control_strategy"].eq("top3_chain_all_members")
    ]
    main_attribution = attribution.loc[attribution["arm"].eq("monthly_ttl250")]
    attribution_check = (
        main_attribution.groupby("decision_date", sort=True)["net_excess_contribution"]
        .sum()
        .rename("attributed_excess")
        .reset_index()
        .merge(
            main_paired_daily[["decision_date", "paired_net_excess"]],
            on="decision_date",
            how="inner",
            validate="one_to_one",
        )
    )
    max_attribution_residual = float(
        (attribution_check["attributed_excess"] - attribution_check["paired_net_excess"])
        .abs()
        .max()
    )
    main_audit = chain_audit.loc[
        chain_audit["arm"].eq("monthly_ttl250")
        & chain_audit["model_variant"].eq("R1")
    ]
    main_r1_positions = positions.loc[
        positions["arm"].eq("monthly_ttl250")
        & positions["model_variant"].eq("R1")
    ]
    quality = pd.DataFrame(
        [
            {
                "check": "月频决策日",
                "observed": int(
                    main_daily.loc[main_daily["strategy"].eq("R1"), "decision_date"].nunique()
                ),
                "interpretation": "覆盖2021–2025全部60个月",
            },
            {
                "check": "R1可评价期",
                "observed": int(
                    main_daily.loc[
                        main_daily["strategy"].eq("R1")
                        & main_daily["return_evaluable"].astype(bool)
                    ].shape[0]
                ),
                "interpretation": "60/60；无未来价格被填0",
            },
            {
                "check": "平均合格活跃链股票",
                "observed": round(
                    float(main_coverage["eligible_active_unique_stocks"].mean()), 1
                ),
                "interpretation": "研报PIT代理的日均覆盖",
            },
            {
                "check": "R1满槽率",
                "observed": round(
                    float(main_audit["selection_status"].eq("selected_full").mean()), 4
                ),
                "interpretation": "180/180个链×日均选满4只",
            },
            {
                "check": "相关性未知对",
                "observed": int(main_audit["correlation_pairs_missing"].sum()),
                "interpretation": "主月频选择中为0",
            },
            {
                "check": "随机篮子完整率",
                "observed": round(
                    float(
                        main_daily.loc[
                            main_daily["strategy"].eq("top3_chain_random_4"),
                            "future_return_coverage",
                        ].mean()
                    ),
                    6,
                ),
                "interpretation": "500次/日随机对照的可评价比例",
            },
            {
                "check": "链级归因最大残差",
                "observed": max_attribution_residual,
                "interpretation": "链贡献汇总与主对照逐日配对净超额之差",
            },
            {
                "check": "双倍成本最好相对累计超额",
                "observed": float(
                    paired.loc[
                        paired["arm"].eq("monthly_ttl250"),
                        "double_cost_relative_compound_excess",
                    ].max()
                ),
                "interpretation": "两类冻结对照中较不差者；仍为负",
            },
            {
                "check": "主R1中性填充持仓行",
                "observed": int(main_r1_positions["neutral_fill_count"].gt(0).sum()),
                "interpretation": f"0/{len(main_r1_positions)}；完整R1选中股票四族均可评分",
            },
        ]
    )

    r1_main = main.set_index("strategy").loc["R1"]
    control_main = main.set_index("strategy").loc["top3_chain_all_members"]
    random_main = main.set_index("strategy").loc["top3_chain_random_4"]
    summary_metrics = pd.DataFrame(
        [
            {
                "r1_annualized_return": float(r1_main["annualized_return"]),
                "annualized_excess": float(main_pair["annualized_net_excess"]),
                "r1_max_drawdown": float(r1_main["max_drawdown"]),
                "control_annualized_return": float(control_main["annualized_return"]),
                "random_annualized_return": float(random_main["annualized_return"]),
            }
        ]
    )
    behavior_summary = pd.DataFrame(
        [
            {
                "top20_recall": behavior_hits / len(behavior_ranks),
                "top20_hits": behavior_hits,
                "labels": len(behavior_ranks),
                "hit_days": int(behavior["r1_top20_hits"].gt(0).sum()),
                "decision_days": len(behavior),
                "theme_membership_hits": theme_coverage_hits,
                "eligible_theme_membership_hits": eligible_theme_coverage_hits,
            }
        ]
    )

    sources = [
        _source(
            "historical_summary",
            "历史策略收益汇总",
            "research/ledger/trader_chain_r1_historical_strategy_summary.csv",
            "R1、对照和固定消融在四个频率/TTL臂上的成本后收益与风险汇总。",
            sql=(
                "SELECT * FROM read_csv_auto("
                "'research/ledger/trader_chain_r1_historical_strategy_summary.csv', "
                "header = true) WHERE return_basis = 'net_historical_cost'"
            ),
            filters=["主评价期2021-01-01至2025-12-31", "普通沪深A股", "ADV20至少2000万元"],
            definitions={
                "annualized_return": "按每臂周期频率对成本后组合复利年化。",
                "max_drawdown": "按独立决策期收益顺序复利所得净值的最深回撤，不是持有期逐日盯市回撤。",
            },
        ),
        _source(
            "paired_summary",
            "R1逐日配对超额汇总",
            "research/ledger/trader_chain_r1_historical_paired_summary.csv",
            "R1相对Top-3链全成员和同链随机4只的逐决策日配对结果。",
            sql=(
                "SELECT * FROM read_csv_auto("
                "'research/ledger/trader_chain_r1_historical_paired_summary.csv', "
                "header = true)"
            ),
            filters=["同一决策日内配对", "历史往返成本12bp/7bp"],
            definitions={
                "annualized_net_excess": "R1成本后年化收益减主对照成本后年化收益。",
                "paired_net_information_ratio": "逐期配对净超额均值除以标准差并年化。",
            },
        ),
        _source(
            "paired_inference",
            "决策日粒度配对推断",
            "research/ledger/trader_chain_r1_historical_inference.csv",
            "按决策日重抽样的均值bootstrap区间和非零差值单边精确符号检验。",
            sql=(
                "SELECT * FROM read_csv_auto("
                "'research/ledger/trader_chain_r1_historical_inference.csv', "
                "header = true) WHERE arm = 'monthly_ttl250'"
            ),
            filters=["决策日为抽样单位", "固定seed", "双侧95% percentile bootstrap区间"],
            definitions={
                "mean_paired_net_excess": "同一决策日R1成本后收益减对照成本后收益的算术均值。",
                "sign_test_pvalue_greater": "剔除零差值后，正向备择的精确二项检验p值。",
            },
        ),
        _source(
            "half_year_summary",
            "半年稳定性汇总",
            "research/ledger/trader_chain_r1_historical_half_year.csv",
            "R1相对两类冻结对照在10个自然半年区间的成本后复利超额。",
            sql=(
                "SELECT * FROM read_csv_auto("
                "'research/ledger/trader_chain_r1_historical_half_year.csv', "
                "header = true)"
            ),
        ),
        _source(
            "historical_gate",
            "主收益闸门审计",
            "research/ledger/trader_chain_r1_historical_gate_checks.csv",
            "冻结协议中收益闸门各检查项的观察值、阈值和状态。",
            sql=(
                "SELECT * FROM read_csv_auto("
                "'research/ledger/trader_chain_r1_historical_gate_checks.csv', "
                "header = true) WHERE arm = 'monthly_ttl250'"
            ),
        ),
        _source(
            "behavior_results",
            "2026静态行为解释结果",
            "research/ledger/trader_chain_r1_behavior_daily.csv",
            "20笔L4买入在8个披露日上的Top-20命中、组合命中和penalty-rank比较。",
            sql=(
                "SELECT * FROM read_csv_auto("
                "'research/ledger/trader_chain_r1_behavior_daily.csv', header = true)"
            ),
            filters=["静态九链oracle，仅用于解释", "20笔L4精确日普通股票买入"],
        ),
        _source(
            "behavior_gate_checks",
            "行为解释完整闸门",
            "research/ledger/trader_chain_r1_behavior_gate.json",
            "冻结行为闸门四项检查，包括新增命中单日集中度。",
            sql=(
                "WITH gate AS (SELECT * FROM read_json_auto("
                "'research/ledger/trader_chain_r1_behavior_gate.json')) "
                "SELECT 'top20_hits' AS check_id, observed.top20_hits AS observed, "
                "thresholds.minimum_top20_hits AS threshold, checks.top20_hits AS passed FROM gate "
                "UNION ALL SELECT 'top20_hit_days', observed.top20_hit_days, "
                "thresholds.minimum_top20_hit_days, checks.top20_hit_days FROM gate "
                "UNION ALL SELECT 'penalty_rank_improved_days', observed.penalty_rank_improved_days, "
                "thresholds.minimum_penalty_rank_improved_days, checks.penalty_rank_improved_days FROM gate "
                "UNION ALL SELECT 'incremental_hit_concentration', "
                "observed.max_incremental_hit_day_share, "
                "thresholds.maximum_incremental_hit_day_share, checks.incremental_hit_concentration FROM gate"
            ),
        ),
        _source(
            "behavior_ranks",
            "2026静态行为标签排名",
            "research/ledger/trader_chain_r1_behavior_label_ranks.csv",
            "20笔L4买入的静态oracle R1排名、Top-20命中和最终组合命中。",
            sql=(
                "SELECT * FROM read_csv_auto("
                "'research/ledger/trader_chain_r1_behavior_label_ranks.csv', "
                "header = true)"
            ),
            filters=["20笔L4精确日普通股票买入"],
        ),
        _source(
            "behavior_evidence",
            "行为解释与九链覆盖联合审计",
            "research/ledger/trader_chain_r1_behavior_label_ranks.csv",
            "区分静态九链主题成员覆盖、合格且有成员映射的可评分覆盖与最终Top-20命中。",
            sql=(
                "WITH labels AS (SELECT * FROM read_csv_auto("
                "'research/ledger/trader_choice_labels.csv', header = true)), "
                "members AS (SELECT DISTINCT symbol FROM read_csv_auto("
                "'research/ledger/trader_oracle_theme_audit.csv', header = true) "
                "WHERE in_panel_universe = true), ranks AS (SELECT * FROM read_csv_auto("
                "'research/ledger/trader_chain_r1_behavior_label_ranks.csv', header = true)) "
                "SELECT COUNT(*) AS labels, COUNT(*) FILTER (WHERE members.symbol IS NOT NULL) "
                "AS theme_membership_hits, COUNT(*) FILTER (WHERE members.symbol IS NOT NULL "
                "AND ranks.r1_in_full_pool = true) AS eligible_theme_membership_hits, "
                "COUNT(*) FILTER (WHERE ranks.r1_behavior_top20 = true) AS r1_top20_hits "
                "FROM labels LEFT JOIN members USING (symbol) LEFT JOIN ranks USING (event_id)"
            ),
            tables_used=[
                "research/ledger/trader_choice_labels.csv",
                "research/ledger/trader_oracle_theme_audit.csv",
                "research/ledger/trader_chain_r1_behavior_label_ranks.csv",
            ],
            filters=["静态九链oracle，仅用于解释", "映射表不含交易标签"],
        ),
        _source(
            "quality_audit",
            "数据与组合质量审计",
            "research/ledger/trader_chain_r1_historical_chain_audit.csv",
            "覆盖、组合填槽、子环节分散和相关性去重的审计结果。",
            sql=(
                "WITH audit AS (SELECT * FROM read_csv_auto("
                "'research/ledger/trader_chain_r1_historical_chain_audit.csv', "
                "header = true)), coverage AS (SELECT * FROM read_csv_auto("
                "'research/ledger/trader_chain_r1_historical_coverage.csv', "
                "header = true)), daily AS (SELECT * FROM read_csv_auto("
                "'research/ledger/trader_chain_r1_historical_daily.csv', "
                "header = true)), paired AS (SELECT * FROM read_csv_auto("
                "'research/ledger/trader_chain_r1_historical_paired_daily.csv', "
                "header = true)), paired_summary AS (SELECT * FROM read_csv_auto("
                "'research/ledger/trader_chain_r1_historical_paired_summary.csv', "
                "header = true)), attribution AS (SELECT * FROM read_csv_auto("
                "'research/ledger/trader_chain_r1_historical_chain_attribution.csv', "
                "header = true)), positions AS (SELECT * FROM read_parquet("
                "'research/ledger/trader_chain_r1_historical_positions.parquet')), "
                "attr_day AS (SELECT decision_date, SUM(net_excess_contribution) "
                "AS attributed_excess FROM attribution WHERE arm = 'monthly_ttl250' GROUP BY 1), "
                "pair_day AS (SELECT decision_date, paired_net_excess FROM paired WHERE "
                "arm = 'monthly_ttl250' AND control_strategy = 'top3_chain_all_members') "
                "SELECT (SELECT COUNT(DISTINCT decision_date) FROM daily WHERE arm = "
                "'monthly_ttl250' AND strategy = 'R1') AS decision_dates, (SELECT COUNT(*) "
                "FROM daily WHERE arm = 'monthly_ttl250' AND strategy = 'R1' AND "
                "return_evaluable = true) AS evaluable_dates, (SELECT AVG("
                "eligible_active_unique_stocks) FROM coverage WHERE arm = 'monthly_ttl250') "
                "AS mean_eligible_active_stocks, (SELECT AVG(CASE WHEN selection_status = "
                "'selected_full' THEN 1.0 ELSE 0.0 END) FROM audit WHERE arm = "
                "'monthly_ttl250' AND model_variant = 'R1') AS full_slot_rate, (SELECT SUM("
                "correlation_pairs_missing) FROM audit WHERE arm = 'monthly_ttl250' AND "
                "model_variant = 'R1') AS missing_correlation_pairs, (SELECT AVG("
                "future_return_coverage) FROM daily WHERE arm = 'monthly_ttl250' AND strategy = "
                "'top3_chain_random_4') AS random_future_coverage, (SELECT MAX(ABS("
                "attr_day.attributed_excess - pair_day.paired_net_excess)) FROM attr_day JOIN "
                "pair_day USING (decision_date)) AS max_attribution_residual, (SELECT MAX("
                "double_cost_relative_compound_excess) FROM paired_summary WHERE arm = "
                "'monthly_ttl250') AS least_negative_double_cost_excess, (SELECT COUNT(*) "
                "FROM positions WHERE arm = 'monthly_ttl250' AND model_variant = 'R1' AND "
                "neutral_fill_count > 0) AS r1_neutral_fill_rows"
            ),
            tables_used=[
                "research/ledger/trader_chain_r1_historical_chain_audit.csv",
                "research/ledger/trader_chain_r1_historical_coverage.csv",
                "research/ledger/trader_chain_r1_historical_daily.csv",
                "research/ledger/trader_chain_r1_historical_paired_daily.csv",
                "research/ledger/trader_chain_r1_historical_paired_summary.csv",
                "research/ledger/trader_chain_r1_historical_chain_attribution.csv",
                "research/ledger/trader_chain_r1_historical_positions.parquet",
            ],
        ),
        _source(
            "frozen_protocol",
            "R1冻结实验协议",
            "research/TRADER_CHAIN_RULE_PROTOCOL.md",
            "在收益与标签结果生成前冻结的数据时钟、规则、对照、消融和闸门。",
        ),
    ]
    manifest_sources = [
        {"id": source["id"], "label": source["label"], "path": source["path"]}
        for source in sources
    ]

    cards = [
        {
            "id": "r1_return_card",
            "description": "月频TTL250主实验，60个决策日，历史成本后。",
            "dataset": "summary_metrics",
            "sourceId": "historical_summary",
            "metrics": [
                {
                    "label": "完整R1成本后年化收益",
                    "field": "r1_annualized_return",
                    "format": "percent",
                    "signed": True,
                }
            ],
        },
        {
            "id": "excess_card",
            "description": "完整R1减Top-3链全成员对照的年化收益差。",
            "dataset": "summary_metrics",
            "sourceId": "paired_summary",
            "metrics": [
                {
                    "label": "相对主对照年化净超额",
                    "field": "annualized_excess",
                    "format": "percent",
                    "signed": True,
                }
            ],
        },
        {
            "id": "drawdown_card",
            "description": "月频TTL250主实验；按20日决策期收益序列复利，不是持有期逐日盯市回撤。",
            "dataset": "summary_metrics",
            "sourceId": "historical_summary",
            "metrics": [
                {
                    "label": "决策期序列最大回撤",
                    "field": "r1_max_drawdown",
                    "format": "percent",
                    "signed": True,
                }
            ],
        },
        {
            "id": "behavior_card",
            "description": "使用静态oracle映射的R1；20笔L4买入、8个决策日。",
            "dataset": "behavior_summary",
            "sourceId": "behavior_ranks",
            "metrics": [
                {
                    "label": "披露买入Top-20命中率",
                    "field": "top20_recall",
                    "format": "percent",
                },
                {"label": "命中笔数", "field": "top20_hits", "format": "number"},
            ],
        },
    ]

    charts = [
        {
            "id": "main_strategy_returns",
            "title": "月频主实验成本后年化收益",
            "subtitle": "2021–2025，TTL 250个交易日，60个决策日",
            "type": "bar",
            "dataset": "main_strategy_returns",
            "sourceId": "historical_summary",
            "valueFormat": "percent",
            "encodings": {
                "x": {"field": "strategy_label", "type": "nominal", "label": "策略"},
                "y": {
                    "field": "annualized_return",
                    "type": "quantitative",
                    "label": "成本后年化收益",
                    "format": "percent",
                },
                "tooltip": [
                    {
                        "field": "max_drawdown",
                        "type": "quantitative",
                        "label": "最大回撤",
                        "format": "percent",
                    },
                    {
                        "field": "compound_return",
                        "type": "quantitative",
                        "label": "累计收益",
                        "format": "percent",
                    },
                ],
            },
            "yAxisTitle": "成本后年化收益",
            "layout": "full",
        },
        {
            "id": "sensitivity_returns",
            "title": "持有周期与证据寿命敏感性",
            "subtitle": "四个预注册实验臂的成本后年化收益",
            "type": "bar",
            "dataset": "sensitivity_returns",
            "sourceId": "historical_summary",
            "valueFormat": "percent",
            "encodings": {
                "x": {"field": "arm_label", "type": "ordinal", "label": "实验臂"},
                "y": {
                    "field": "annualized_return",
                    "type": "quantitative",
                    "label": "成本后年化收益",
                    "format": "percent",
                },
                "color": {"field": "series", "type": "nominal", "label": "策略"},
                "tooltip": [
                    {
                        "field": "max_drawdown",
                        "type": "quantitative",
                        "label": "最大回撤",
                        "format": "percent",
                    }
                ],
            },
            "yAxisTitle": "成本后年化收益",
            "layout": "full",
        },
        {
            "id": "half_year_excess",
            "title": "主月频逐半年配对净超额",
            "subtitle": "R1相对两类冻结对照，2021H1–2025H2",
            "type": "bar",
            "dataset": "half_year_excess",
            "sourceId": "half_year_summary",
            "valueFormat": "percent",
            "encodings": {
                "x": {"field": "half_year", "type": "ordinal", "label": "半年"},
                "y": {
                    "field": "net_relative_compound_excess",
                    "type": "quantitative",
                    "label": "配对净超额",
                    "format": "percent",
                },
                "color": {"field": "control", "type": "nominal", "label": "对照"},
                "tooltip": [
                    {
                        "field": "paired_net_win_rate",
                        "type": "quantitative",
                        "label": "逐期胜率",
                        "format": "percent",
                    },
                    {
                        "field": "n_periods",
                        "type": "quantitative",
                        "label": "决策日数",
                        "format": "number",
                    },
                ],
            },
            "yAxisTitle": "配对净超额",
            "layout": "full",
        },
    ]

    tables = [
        {
            "id": "inference_table",
            "title": "主月频决策日粒度推断",
            "subtitle": "10,000次percentile bootstrap；符号检验为正向单边精确二项检验",
            "dataset": "main_inference",
            "sourceId": "paired_inference",
            "defaultSort": {"field": "control", "direction": "asc"},
            "columns": [
                {"field": "control", "label": "冻结对照", "type": "text"},
                {"field": "n_evaluable_decision_dates", "label": "决策日", "format": "number"},
                {"field": "mean_paired_net_excess", "label": "平均净超额", "format": "percent"},
                {"field": "bootstrap_ci_lower", "label": "95%下界", "format": "percent"},
                {"field": "bootstrap_ci_upper", "label": "95%上界", "format": "percent"},
                {"field": "positive_days", "label": "正超额日", "type": "text"},
                {"field": "sign_test_pvalue_greater", "label": "正向符号p值", "format": "number"},
            ],
        },
        {
            "id": "behavior_daily_table",
            "title": "披露买入逐日解释结果",
            "subtitle": "静态oracle解释臂；penalty rank越小越好",
            "dataset": "behavior_daily",
            "sourceId": "behavior_results",
            "defaultSort": {"field": "decision_date", "direction": "asc"},
            "columns": [
                {"field": "decision_date", "label": "决策日", "type": "date"},
                {"field": "n_labels", "label": "买入笔数", "format": "number"},
                {"field": "baseline_top20_hits", "label": "旧基线Top-20", "format": "number"},
                {"field": "r1_top20_hits", "label": "R1 Top-20", "format": "number"},
                {"field": "r1_portfolio_hits", "label": "R1组合命中", "format": "number"},
                {"field": "baseline_mean_penalty_rank", "label": "旧平均惩罚排名", "format": "number"},
                {"field": "r1_mean_penalty_rank", "label": "R1平均惩罚排名", "format": "number"},
                {"field": "mean_penalty_rank_improved", "label": "排名改善", "type": "boolean"},
            ],
        },
        {
            "id": "behavior_gate_table",
            "title": "完整行为解释闸门（4项）",
            "subtitle": "新增命中为0，因此集中度项通过但不提供正向证据",
            "dataset": "behavior_gate_checks",
            "sourceId": "behavior_gate_checks",
            "defaultSort": {"field": "check", "direction": "asc"},
            "columns": [
                {"field": "check", "label": "检查项", "type": "text"},
                {"field": "observed", "label": "观察值", "type": "text"},
                {"field": "threshold", "label": "阈值", "type": "text"},
                {"field": "status", "label": "状态", "type": "text"},
            ],
        },
        {
            "id": "gate_table",
            "title": "完整主月频收益闸门（13项）",
            "subtitle": "冻结阈值与实际观察值；负总超额使两项贡献集中度不可评价",
            "dataset": "gate_checks",
            "sourceId": "historical_gate",
            "defaultSort": {"field": "description", "direction": "asc"},
            "columns": [
                {"field": "description", "label": "检查项", "type": "text"},
                {"field": "observed", "label": "观察值", "type": "text"},
                {"field": "threshold", "label": "阈值", "type": "text"},
                {"field": "status_label", "label": "状态", "type": "text"},
            ],
        },
        {
            "id": "quality_table",
            "title": "数据与组合可用性检查",
            "subtitle": "主月频TTL250实验；用于排除现金、缺价和相关性缺失伪影",
            "dataset": "quality_checks",
            "sourceId": "quality_audit",
            "defaultSort": {"field": "check", "direction": "asc"},
            "columns": [
                {"field": "check", "label": "检查", "type": "text"},
                {"field": "observed", "label": "观察值", "type": "text"},
                {"field": "interpretation", "label": "解释", "type": "text"},
            ],
        },
    ]

    blocks = [
        {"id": "title", "type": "markdown", "body": "# 产业链内分散规则 R1：复刻与收益验证"},
        {
            "id": "technical_summary",
            "type": "markdown",
            "body": (
                "## 技术结论：当前模型既没有复刻买入，也没有形成可交易超额\n\n"
                "- **收益假设被否定。** 月频主实验中，完整 R1 成本后年化 **-10.02%**，"
                "Top-3 链全成员为 **+9.29%**，同链随机 4 只为 **+9.57%**。相对主对照年化净超额为 **-19.31 个百分点**。\n"
                "- **行为解释也未过门。** 使用静态 oracle 映射的 R1 只命中 **1/20** 笔、覆盖 **1/8** 个买入日，"
                "逐日平均 penalty rank 仅 **1/8** 日优于旧 F1–F6。\n"
                "- **失败主要发生在链内选股层。** 产业链方向对照仍有正收益；加入强度、领导力、形态和研报证据后，"
                "收益转负。分散约束不是根因，只是进一步拖累约 2 个年化百分点。\n"
                "- **裁决：不进入机器学习。** 四个预注册频率/TTL实验臂全部失败，当前九链研报代理和四族排序"
                "不应继续围绕20笔披露买入调参。"
            ),
        },
        {"id": "headline_metrics", "type": "metric-strip", "cardIds": [card["id"] for card in cards]},
        {
            "id": "finding_stock_rank",
            "type": "markdown",
            "sourceId": "historical_summary",
            "body": (
                "## 产业链方向尚可，链内排名把正收益变成负收益\n\n"
                "主月频中，全市场合格池年化 **+9.07%**，Top-3 链全成员 **+9.29%**，说明方向层至多是"
                "接近市场、略有改善；它不是主要损失来源。冻结消融 `direction_only` 在实现上就是Top-3链全成员"
                "组合，因此两者数值相同、不在图中重复。完整 R1 为 **-10.02%**，去掉分散约束仍为 **-7.95%**。"
                "因此，即使不做细行业与相关性约束，链内高分股票仍大幅落后同链平均。\n\n"
                "固定消融显示两个族与失效相关：去掉研报证据族后改善到 **-2.22%**，去掉形态族后为 **-4.79%**。"
                "这不能证明因果，也不能据此反向选择新权重；它只说明当前编码没有产生稳定的正向排序。"
            ),
        },
        {"id": "main_chart", "type": "chart", "chartId": "main_strategy_returns", "layout": "full"},
        {
            "id": "sensitivity_finding",
            "type": "markdown",
            "sourceId": "historical_summary",
            "body": (
                "## 缩短研报寿命或改成周频都没有修复模型\n\n"
                "完整 R1 在四个预注册实验臂的成本后年化收益介于 **-5.55% 与 -12.19%**；"
                "两类同链对照则均为正。TTL 从 250 缩到 120 个交易日只改善月频，周频反而更差。"
                "因此，旧研报长期滞留不是唯一问题，5 日持有也没有捕捉到预期的短周期概念动量。"
            ),
        },
        {"id": "sensitivity_chart", "type": "chart", "chartId": "sensitivity_returns", "layout": "full"},
        {
            "id": "half_year_finding",
            "type": "markdown",
            "sourceId": "half_year_summary",
            "body": (
                "## 失败不是单一坏年份造成：10个半年中仅3个同时跑赢两类对照\n\n"
                "冻结闸门要求至少 7/10 个半年为正，实际只有 **3/10**。主月频配对净超额信息比率为 **-0.90**，"
                "按20日决策期收益序列复利的最大回撤为 **-69.2%**，比主对照恶化 **19.7 个百分点**。"
                "这不是由某一个异常年份或交易成本造成的"
                "边缘失败。"
            ),
        },
        {"id": "half_year_chart", "type": "chart", "chartId": "half_year_excess", "layout": "full"},
        {
            "id": "inference_finding",
            "type": "markdown",
            "sourceId": "paired_inference",
            "body": (
                "## 日期粒度推断同样拒绝正超额解释\n\n"
                f"相对Top-3链全成员，平均决策期净超额为 **{member_inference['mean_paired_net_excess']:.2%}**，"
                f"日期bootstrap 95%区间为 **[{member_inference['bootstrap_ci_lower']:.2%}, "
                f"{member_inference['bootstrap_ci_upper']:.2%}]**，正向单边符号检验 "
                f"`p={member_inference['sign_test_pvalue_greater']:.3f}`；相对同链随机4只为 "
                f"**{random_inference['mean_paired_net_excess']:.2%}**，区间 "
                f"**[{random_inference['bootstrap_ci_lower']:.2%}, {random_inference['bootstrap_ci_upper']:.2%}]**，"
                f"`p={random_inference['sign_test_pvalue_greater']:.3f}`。抽样单位始终是决策日，"
                "不把股票行误当独立样本。"
            ),
        },
        {"id": "inference_detail", "type": "table", "tableId": "inference_table", "layout": "full"},
        {"id": "gate_detail", "type": "table", "tableId": "gate_table", "layout": "full"},
        {
            "id": "behavior_finding",
            "type": "markdown",
            "sourceId": "behavior_evidence",
            "body": (
                "## 静态九链能解释‘属于哪个主题’，但不能解释‘为什么买它’\n\n"
                f"九链静态映射覆盖 **{theme_coverage_hits}/20** 笔披露买入，但其中合格且有映射的可评分覆盖只有 "
                f"**{eligible_theme_coverage_hits}/20**；R1 只在 1 笔上进入 Top-20，最终12只组合也只命中1笔。"
                "关键区别是：**19/20 的主题成员覆盖是中间层覆盖，不是末端选择命中。**"
                "一只股票可以合理归入产业链，却仍可能在同链数十到上百只股票中排名靠后。"
            ),
        },
        {"id": "behavior_gate_detail", "type": "table", "tableId": "behavior_gate_table", "layout": "full"},
        {"id": "behavior_table", "type": "table", "tableId": "behavior_daily_table", "layout": "full"},
        {
            "id": "scope_definitions",
            "type": "markdown",
            "sourceId": "frozen_protocol",
            "body": (
                "## 评价对象与指标定义\n\n"
                "**H1 历史 PIT 代理**覆盖 2021–2025；2020 仅作预热。月频在每月首个交易日开盘买入并持有20个交易日；"
                "周频每5个交易日买入并持有5个交易日。收益定义为 `open[T+h] / open[T] - 1`。\n\n"
                "**H2 行为解释上限**只使用 2026 静态九链映射评估20笔L4精确日买入。它允许未来元数据回填，"
                "所以只回答解释力，不报告可部署收益。\n\n"
                "主对照是当日方向得分最高三条链的全部合格成员，链等权、链内等权；第二对照在相同三条链内"
                "每链随机4只、每日500次。成本在2023年8月28日前按往返12bp，之后7bp；该成本按每个独立"
                "决策期固定收取并按投资比例缩放，不是基于真实换手、冲击或滑点的成交仿真。"
            ),
        },
        {
            "id": "model_specification",
            "type": "markdown",
            "sourceId": "frozen_protocol",
            "body": (
                "## 模型与实验设计\n\n"
                "1. **方向层：** 九条产业链按20/60日动量中位数、站上MA20宽度和量能扩张四项等权，取Top-3。\n"
                "2. **链内层：** 相对强度、领导力/可交易性、三支回调再启动形态、研报来源数/新鲜度四族等权。"
                "缺失族固定填0.5，不改变分母。\n"
                "3. **组合层：** 每链最多4只、单子环节最多2只、尽量覆盖至少2个子环节；已知60日相关系数大于0.80"
                "的候选拒绝，空槽留现金。\n"
                "4. **防过拟合：** 规则、两类对照、四个消融、120日TTL敏感性和收益/行为闸门均在结果生成前冻结。"
            ),
        },
        {
            "id": "robustness",
            "type": "markdown",
            "sourceId": "quality_audit",
            "body": (
                "## 数据质量检查支持把失败归因于规则，而不是工程伪影\n\n"
                "主月频60/60期可评价，R1在180个链×日上全部选满4只；子环节约束全部满足，相关性审计没有未知配对。"
                "同链随机对照的未来收益完整率为99.93%。链级净超额归因与组合级逐日配对结果精确对账。"
                "双倍成本下结论不变。由于R1总超额为负，‘单半年／单产业链占正总超额不超过50%’两项没有"
                "可解释分母，因此13项协议检查并非全部可评价（"
                f"`all_required_checks_evaluated={str(all_required_checks_evaluated).lower()}`）；这不会把已失败的收益闸门改为通过。"
            ),
        },
        {"id": "quality_detail", "type": "table", "tableId": "quality_table", "layout": "full"},
        {
            "id": "limitations",
            "type": "markdown",
            "body": (
                "## 结论边界：这否定的是当前代理，不是交易者本人\n\n"
                "- 历史产业链成员只来自分析师研报标题和细行业，日均约531只合格活跃股票；它缺少历史题材成分、新闻/社交注意力、"
                "产业链上下游强弱和公司实际业务收入暴露。\n"
                "- 20笔披露买入只覆盖8天，不能识别仓位大小、卖出、盘中执行或人工否决。静态行为臂还明确使用未来元数据。\n"
                "- 当前模型没有估值、盈利修正、订单/政策催化、龙虎榜资金结构、涨停可买性和更细的市场状态。\n"
                "- 本地股票截面尚未完成历史退市股覆盖审计，可能存在幸存者偏差；最大回撤来自决策期收益序列，"
                "不是持有期间逐日盯市路径；固定往返bp也没有模拟真实换手、冲击和滑点。\n"
                "- 所以本实验不能解释交易者账户所称高收益由哪些已实现交易贡献，也不能证明其方法不是量化；它只证明"
                "‘九链研报映射 + 当前四族等权 + 规则分散’不是答案。"
            ),
        },
        {
            "id": "next_steps",
            "type": "markdown",
            "body": (
                "## 下一步应更换信息层，而不是继续调当前权重\n\n"
                "1. **暂停当前R1与机器学习。** 不用20笔披露交易调阈值、主题词或权重。\n"
                "2. **新建R2协议，先补时间点一致的关系数据。** 优先顺序是历史概念/指数成分、公告与新闻事件、"
                "分析师盈利修正、公司产业链收入暴露，再考虑模型。\n"
                "3. **把链内目标改成未来相对本链收益。** 先做每个族的历史IC、符号稳定性和分层收益；只有简单规则"
                "在多数半年稳定后，才允许pairwise/LambdaMART。\n"
                "4. **复刻交易者与寻找可盈利策略继续分轨。** 复刻线需要更多买卖与仓位标签；盈利线必须只看历史PIT数据，"
                "不能用披露买入作训练标签。"
            ),
        },
        {
            "id": "further_questions",
            "type": "markdown",
            "body": (
                "## 仍需回答的关键问题\n\n"
                "- 交易者的高收益是否集中在少数大仓位、卖出时点或未完整披露的交易，而不是这20笔买入？\n"
                "- 他是否先用量化生成主题/形态候选，再以催化、基本面或熟悉度人工压缩到3–5只？\n"
                "- ETF长线仓与个股短线仓是否采用完全不同的风险预算和退出规则？"
            ),
        },
    ]

    generated_at = REPORT_GENERATED_AT
    return {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "产业链内分散规则 R1：复刻与收益验证",
            "description": "九链产业链方向、链内四族排序与分散组合的历史PIT收益和2026披露买入解释力验证。",
            "generatedAt": generated_at,
            "cards": cards,
            "charts": charts,
            "tables": tables,
            "sources": manifest_sources,
            "blocks": blocks,
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": {
                "summary_metrics": _records(summary_metrics),
                "behavior_summary": _records(behavior_summary),
                "main_strategy_returns": _records(main_chart),
                "sensitivity_returns": _records(sensitivity),
                "half_year_excess": _records(half),
                "main_inference": _records(main_inference),
                "behavior_daily": _records(behavior),
                "behavior_gate_checks": _records(behavior_gate),
                "gate_checks": _records(gate),
                "quality_checks": _records(quality),
            },
        },
        "sources": sources,
    }


def main() -> None:
    args = parse_args()
    artifact = build_artifact(args.ledger_dir)
    _atomic_json(artifact, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
