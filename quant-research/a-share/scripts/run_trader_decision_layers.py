#!/usr/bin/env python3
"""Run the trader-decision-layer discovery and forward-scoring workflow.

The evidence repository remains the label source of truth.  This script builds
features directly from point-in-time A-share data, writes auditable long-form
ledgers, and keeps imitation metrics separate from forward-return diagnostics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from afactor.trader_choice import (  # noqa: E402
    ChoiceConfig,
    assign_candidate_pools,
    build_matched_controls,
    evaluate_ablations,
    evaluate_ranking,
    load_choice_labels,
    score_factor_families,
)
from afactor.trader_choice_data import (  # noqa: E402
    ChoiceDataPaths,
    add_direction_and_leader,
    add_event_attention,
    add_familiarity,
    add_forward_returns,
    build_base_panel,
    extend_statind_2026,
    load_daily_inputs,
)
from afactor.trader_choice_portfolio import add_portfolio_replacement  # noqa: E402


DEFAULT_TRADER_ROOT = Path("/path/to/user/Documents/ChatGPT/Stock model")
DEFAULT_LEDGER = ROOT / "research" / "ledger"
DEFAULT_REPORT = ROOT / "research" / "TRADER_DECISION_LAYER_REPORT.md"
ROUND_TRIP_COST = 0.0007  # buy 1bp + sell 1bp + current 5bp stamp duty
RANDOM_REPEATS = 10_000
EXCHANGE_TZ = "Asia/Shanghai"
FORWARD_START_DATE = pd.Timestamp("2026-09-01")
OUTCOME_PREFIXES = ("fwd_", "selected_fwd_", "control_fwd_", "paired_diff_")


FAMILIES_5 = OrderedDict(
    [
        ("price_volume", ("pv_score",)),
        ("dynamic_direction", ("f2_direction_score",)),
        ("within_direction_leader", ("f3_leader_score",)),
        ("event_attention", ("f4_event_attention_score",)),
        ("familiarity", ("f5_familiarity_score",)),
    ]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("discovery", "score-day", "forward-eval"), default="discovery"
    )
    parser.add_argument("--as-of", help="Decision date for score-day mode (YYYY-MM-DD).")
    parser.add_argument("--astock-root", type=Path, default=ROOT)
    parser.add_argument("--trader-root", type=Path, default=DEFAULT_TRADER_ROOT)
    parser.add_argument("--ledger-dir", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--random-repeats", type=int, default=RANDOM_REPEATS)
    parser.add_argument("--seed", type=int, default=20260831)
    return parser.parse_args()


def _sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _file_signature(path: Path, *, full_hash_limit: int = 64 << 20) -> dict[str, object]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    stat = path.stat()
    signature: dict[str, object] = {
        "path": str(path),
        "exists": True,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
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


def _directory_signature(directory: Path, pattern: str = "*.csv") -> dict[str, object]:
    """Hash a source directory into one deterministic inventory signature."""

    if not directory.exists():
        return {"path": str(directory), "exists": False}
    digest = hashlib.sha256()
    file_count = 0
    total_size = 0
    for path in sorted(directory.glob(pattern)):
        if not path.is_file():
            continue
        relative = path.relative_to(directory).as_posix()
        size = path.stat().st_size
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\0")
        digest.update(_sha256(path).encode("ascii"))
        digest.update(b"\n")
        file_count += 1
        total_size += size
    return {
        "path": str(directory),
        "exists": True,
        "pattern": pattern,
        "file_count": file_count,
        "total_size": total_size,
        "sha256_inventory": digest.hexdigest(),
    }


def _frame_signature(name: str, frame: pd.DataFrame) -> dict[str, object]:
    """Hash the exact aligned values consumed by the feature builders."""

    metadata = {
        "name": name,
        "rows": int(frame.shape[0]),
        "columns": int(frame.shape[1]),
        "first_date": str(pd.Timestamp(frame.index.min()).date()) if len(frame) else None,
        "last_date": str(pd.Timestamp(frame.index.max()).date()) if len(frame) else None,
        "column_names": [str(column) for column in frame.columns],
        "dtypes": [str(dtype) for dtype in frame.dtypes],
    }
    digest = hashlib.sha256(
        json.dumps(metadata, ensure_ascii=False, sort_keys=True).encode("utf-8")
    )
    digest.update(
        pd.util.hash_pandas_object(frame, index=True, categorize=True)
        .to_numpy(dtype="uint64", copy=False)
        .tobytes()
    )
    metadata["sha256"] = digest.hexdigest()
    return metadata


def _validate_forward_as_of(
    as_of: pd.Timestamp, *, now: pd.Timestamp | None = None
) -> pd.Timestamp:
    """Allow a forward score only for today's exchange date, never a backfill."""

    current = pd.Timestamp.now(tz=EXCHANGE_TZ) if now is None else pd.Timestamp(now)
    if current.tzinfo is None:
        current = current.tz_localize(EXCHANGE_TZ)
    else:
        current = current.tz_convert(EXCHANGE_TZ)
    exchange_date = current.normalize().tz_localize(None)
    if as_of < FORWARD_START_DATE:
        raise ValueError(
            f"Forward scoring starts on {FORWARD_START_DATE.date()}; "
            "use discovery mode for historical dates"
        )
    if as_of != exchange_date:
        raise ValueError(
            f"score-day only accepts today's Shanghai date {exchange_date.date()}; "
            "historical and future as-of dates are rejected"
        )
    return current


def _append_forward_log(
    directory: Path, snapshot_path: Path, manifest_path: Path, manifest: Mapping[str, object]
) -> str:
    """Append a hash-chained local creation record for a forward snapshot."""

    log_path = directory / "append_log.jsonl"
    previous_hash: str | None = None
    if log_path.exists():
        lines = [line for line in log_path.read_text(encoding="utf-8").splitlines() if line]
        if lines:
            previous_hash = str(json.loads(lines[-1])["entry_sha256"])
    entry: dict[str, object] = {
        "as_of": manifest["as_of"],
        "created_at_utc": manifest["snapshot_created_at_utc"],
        "snapshot": snapshot_path.name,
        "snapshot_sha256": manifest["snapshot_sha256"],
        "manifest": manifest_path.name,
        "manifest_sha256": _sha256(manifest_path),
        "previous_entry_sha256": previous_hash,
    }
    canonical = json.dumps(entry, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    entry["entry_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")
    return str(entry["entry_sha256"])


def _previous_sessions(calendar: pd.DatetimeIndex) -> dict[pd.Timestamp, pd.Timestamp]:
    return {pd.Timestamp(calendar[i]): pd.Timestamp(calendar[i - 1]) for i in range(1, len(calendar))}


def _build_feature_panel(
    decision_dates: Iterable[object],
    paths: ChoiceDataPaths,
    config: ChoiceConfig,
    *,
    include_outcomes: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dates = pd.DatetimeIndex(pd.to_datetime(list(decision_dates))).normalize().sort_values().unique()
    if not len(dates):
        raise ValueError("No decision dates were supplied")
    opens, closes, amounts = load_daily_inputs(
        dates.min(),
        dates.max(),
        paths=paths,
        lookback_sessions=520,
        forward_sessions=5 if include_outcomes else 0,
    )
    panel = build_base_panel(dates, opens, closes, amounts)
    panel = assign_candidate_pools(panel, config)
    # Keep the explicit fixed-size multi-shape pool as a separate diagnostic.
    panel["pv_top3pct"] = panel["pv_candidate"].fillna(False).astype(bool)
    statind = extend_statind_2026(closes, paths=paths)
    panel = add_direction_and_leader(
        panel,
        closes,
        amounts,
        statind_labels=statind,
        candidate_pool_col="full_eligible",
    )
    panel = add_event_attention(
        panel, closes.index, paths=paths, candidate_pool_col="full_eligible"
    )
    panel = add_familiarity(
        panel,
        opens,
        closes,
        amounts,
        paths=paths,
        candidate_pool_col="full_eligible",
    )
    if include_outcomes:
        panel = add_forward_returns(panel, opens, horizons=(1, 3, 5))

    # Score F1-F5 on the full eligible set so live holdings outside a narrower
    # candidate pool still have a valid comparison score for F6.
    panel = score_factor_families(
        panel, FAMILIES_5, config, pool_col="full_eligible"
    )
    panel = add_portfolio_replacement(
        panel,
        closes,
        paths.positions_file,
        _previous_sessions(closes.index),
    )
    panel["f6_portfolio_replacement_score"] = panel[
        "family_score__portfolio_replacement"
    ]
    return panel, opens, closes, amounts


def _pool_audit(
    panel: pd.DataFrame,
    labels: pd.DataFrame,
    config: ChoiceConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pools = OrderedDict(
        [
            ("full_eligible", "choice_score"),
            ("legacy_P150", "legacy_fixed_score"),
            ("high_recall_parent", "choice_score"),
            ("pv_top3pct", "pv_score"),
        ]
    )
    daily_parts: list[pd.DataFrame] = []
    count_parts: list[pd.DataFrame] = []
    for pool, score in pools.items():
        daily = evaluate_ranking(
            panel, labels, config, pool_col=pool, score_col=score
        )
        daily.insert(0, "pool", pool)
        daily_parts.append(daily)
        counts = (
            panel.groupby("decision_date", sort=True)[pool]
            .sum()
            .rename("candidate_count")
            .reset_index()
        )
        counts.insert(0, "pool", pool)
        count_parts.append(counts)
    return pd.concat(daily_parts, ignore_index=True), pd.concat(count_parts, ignore_index=True)


def _assert_frozen_regression(panel: pd.DataFrame, labels: pd.DataFrame) -> None:
    """Guard the current eight-day L4 evidence baseline against silent drift."""

    frozen_dates = pd.DatetimeIndex(
        [
            "2026-06-30",
            "2026-08-05",
            "2026-08-06",
            "2026-08-14",
            "2026-08-18",
            "2026-08-21",
            "2026-08-26",
            "2026-08-27",
        ]
    )
    observed_dates = pd.DatetimeIndex(labels["decision_date"].unique()).sort_values()
    if len(labels) != 20 or not observed_dates.equals(frozen_dates):
        return
    joined = labels.merge(
        panel[["decision_date", "symbol", "full_eligible", "legacy_P150"]],
        on=["decision_date", "symbol"],
        how="left",
    )
    checks = {
        "legacy candidate rows": (int(panel["legacy_P150"].sum()), 1127),
        "legacy L4 hits": (int(joined["legacy_P150"].fillna(False).sum()), 7),
        "legacy hit days": (
            int(joined.groupby("decision_date")["legacy_P150"].any().sum()),
            4,
        ),
        "full eligible L4 hits": (int(joined["full_eligible"].fillna(False).sum()), 19),
    }
    failed = {name: values for name, values in checks.items() if values[0] != values[1]}
    if failed:
        raise AssertionError(f"Frozen trader-choice regression changed: {failed}")


def _aggregate_daily(
    daily: pd.DataFrame,
    group_columns: Sequence[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for key, group in daily.groupby(list(group_columns), dropna=False, sort=False):
        key_tuple = key if isinstance(key, tuple) else (key,)
        row = dict(zip(group_columns, key_tuple))
        weights = pd.to_numeric(group["n_labels"], errors="coerce").fillna(0.0)
        row.update(
            {
                "n_days": int(group["decision_date"].nunique()),
                "n_labels": int(weights.sum()),
                "median_candidates": float(group["n_pool"].median()),
                "macro_pool_recall": float(group["pool_recall"].mean()),
                "micro_pool_recall": float(
                    np.average(group["pool_recall"], weights=weights)
                )
                if weights.sum()
                else np.nan,
                "hit_day_rate": float(group["hit_day"].mean()),
                "full_basket_rate": float(group["full_basket"].mean()),
                "mean_mrr": float(group["mrr"].mean()),
                "median_selected_percentile": float(
                    group["median_selected_percentile"].median()
                ),
            }
        )
        for k in (5, 10, 20, 50, 100):
            column = f"recall_at_{k}"
            row[f"macro_recall_at_{k}"] = float(group[column].mean())
            row[f"micro_recall_at_{k}"] = float(
                np.average(group[column], weights=weights)
            ) if weights.sum() else np.nan
            row[f"hit_day_at_{k}"] = float(group[f"hit_day_at_{k}"].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def _strict_controls(
    panel: pd.DataFrame,
    labels: pd.DataFrame,
    config: ChoiceConfig,
) -> pd.DataFrame:
    return build_matched_controls(
        panel,
        labels,
        style_columns=("statind_label", "liq20", "vol20"),
        config=config,
        pool_col="full_eligible",
        categorical_style_columns=("statind_label",),
        controls_per_label=5,
        random_seed=config.random_seed,
    )


def _matched_profit(
    panel: pd.DataFrame,
    labels: pd.DataFrame,
    controls: pd.DataFrame,
) -> pd.DataFrame:
    outcomes = [f"fwd_open_to_open_d{h}" for h in (1, 3, 5)]
    base = panel[["decision_date", "symbol", *outcomes]].copy()
    selected = labels.merge(base, on=["decision_date", "symbol"], how="left")
    selected = selected.rename(columns={column: f"selected_{column}" for column in outcomes})
    matched = controls.merge(
        selected[["event_id", *[f"selected_{c}" for c in outcomes]]],
        on="event_id",
        how="left",
    )
    control_values = base.rename(
        columns={"symbol": "control_symbol", **{c: f"control_{c}" for c in outcomes}}
    )
    matched = matched.merge(
        control_values,
        on=["decision_date", "control_symbol"],
        how="left",
    )
    for column in outcomes:
        matched[f"paired_diff_{column}"] = (
            matched[f"selected_{column}"] - matched[f"control_{column}"]
        )
    return matched


def _within_direction_random(
    panel: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    repeats: int,
    seed: int,
) -> pd.DataFrame:
    if repeats <= 0:
        raise ValueError("random repeats must be positive")
    outcomes = [f"fwd_open_to_open_d{h}" for h in (1, 3, 5)]
    columns = ["decision_date", "symbol", "statind_label", "full_eligible", *outcomes]
    working = panel[columns].copy()
    selected = labels.merge(working, on=["decision_date", "symbol"], how="left")
    selected_keys = set(zip(labels["decision_date"], labels["symbol"]))
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for date, selected_day in selected.groupby("decision_date", sort=True):
        day = working.loc[
            working["decision_date"].eq(date) & working["full_eligible"].fillna(False)
        ]
        valid = selected_day["statind_label"].notna().all()
        cluster_counts = selected_day["statind_label"].value_counts() if valid else pd.Series(dtype=int)
        pools: dict[float, pd.DataFrame] = {}
        if valid:
            for cluster, count in cluster_counts.items():
                pool = day.loc[day["statind_label"].eq(cluster)].copy()
                pool = pool.loc[
                    ~pool.apply(lambda r: (r["decision_date"], r["symbol"]) in selected_keys, axis=1)
                ]
                if len(pool) < int(count):
                    valid = False
                    break
                pools[float(cluster)] = pool
        for outcome in outcomes:
            actual = pd.to_numeric(selected_day[outcome], errors="coerce").mean()
            simulations = np.full(repeats, np.nan)
            if valid and pd.notna(actual):
                sums = np.zeros(repeats, dtype=float)
                complete = np.ones(repeats, dtype=bool)
                total = int(cluster_counts.sum())
                for cluster, count in cluster_counts.items():
                    values = pd.to_numeric(pools[float(cluster)][outcome], errors="coerce").to_numpy()
                    good = np.flatnonzero(np.isfinite(values))
                    if len(good) < int(count):
                        complete[:] = False
                        break
                    # Fixed small basket; looping repetitions is clearer and
                    # memory-safe while preserving no-replacement sampling.
                    for repeat in range(repeats):
                        take = rng.choice(good, size=int(count), replace=False)
                        sums[repeat] += float(values[take].sum())
                simulations[complete] = sums[complete] / total
            finite = simulations[np.isfinite(simulations)]
            rows.append(
                {
                    "decision_date": date,
                    "horizon": outcome,
                    "n_selected": len(selected_day),
                    "actual_basket_return": actual,
                    "random_mean": float(finite.mean()) if len(finite) else np.nan,
                    "random_p10": float(np.quantile(finite, 0.10)) if len(finite) else np.nan,
                    "random_p90": float(np.quantile(finite, 0.90)) if len(finite) else np.nan,
                    "actual_percentile": float(np.mean(finite <= actual)) if len(finite) else np.nan,
                    "valid_random_draws": len(finite),
                    "match_status": "matched" if len(finite) else "insufficient_within_direction",
                    "round_trip_cost": ROUND_TRIP_COST,
                    "actual_net_return": actual - ROUND_TRIP_COST if pd.notna(actual) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def _coverage_table(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for date, day in panel.groupby("decision_date", sort=True):
        row: dict[str, object] = {
            "decision_date": date,
            "eligible": int(day["full_eligible"].sum()),
            "legacy_P150": int(day["legacy_P150"].sum()),
            "high_recall_parent": int(day["high_recall_parent"].sum()),
            "pv_top3pct": int(day["pv_top3pct"].sum()),
            "statind_coverage": float(day["statind_label"].notna().mean()),
            "portfolio_context_available": bool(day["portfolio_context_available"].any()),
            "f5_history_count": int(day["f5_history_count"].max()),
        }
        for source in ("reports", "forecast", "express", "toplist"):
            row[f"{source}_covered"] = bool(day[f"event_{source}_covered"].all())
            cutoff = day[f"event_{source}_cutoff"].dropna()
            row[f"{source}_cutoff"] = cutoff.iloc[0] if len(cutoff) else pd.NaT
        rows.append(row)
    return pd.DataFrame(rows)


def _format_pct(value: object) -> str:
    return "—" if pd.isna(value) else f"{float(value) * 100:.1f}%"


def _binomial_greater_p(wins: int, trials: int) -> float:
    if trials <= 0:
        return np.nan
    return sum(math.comb(trials, k) for k in range(wins, trials + 1)) / (2**trials)


def _write_report(
    path: Path,
    pool_summary: pd.DataFrame,
    ablation_summary: pd.DataFrame,
    profit: pd.DataFrame,
    matched_profit: pd.DataFrame,
    coverage: pd.DataFrame,
) -> None:
    pool_lines = []
    for row in pool_summary.itertuples(index=False):
        pool_lines.append(
            f"| {row.pool} | {row.median_candidates:.0f} | "
            f"{_format_pct(row.micro_pool_recall)} | {_format_pct(row.hit_day_rate)} |"
        )
    ablation_view = ablation_summary.loc[
        ablation_summary["experiment_kind"].isin(
            ["baseline", "single_family", "cumulative"]
        )
    ]
    ablation_lines = []
    for row in ablation_view.itertuples(index=False):
        ablation_lines.append(
            f"| {row.pool} | {row.experiment_name} | {row.median_candidates:.0f} | "
            f"{_format_pct(row.micro_recall_at_5)} | {_format_pct(row.micro_recall_at_10)} | "
            f"{_format_pct(row.micro_recall_at_20)} |"
        )
    profit_lines = []
    for horizon, group in profit.groupby("horizon", sort=False):
        profit_lines.append(
            f"| {horizon} | {_format_pct(group.actual_basket_return.mean())} | "
            f"{_format_pct(group.random_mean.mean())} | "
            f"{_format_pct(group.actual_percentile.mean())} | {int(group.valid_random_draws.sum())} |"
        )
    pool_lookup = pool_summary.set_index("pool")
    single = ablation_summary.loc[
        ablation_summary["pool"].eq("full_eligible")
        & ablation_summary["experiment_kind"].eq("single_family")
    ].set_index("experiment_name")
    final_row = ablation_summary.loc[
        ablation_summary["pool"].eq("full_eligible")
        & ablation_summary["experiment_name"].eq("through_portfolio_replacement")
    ].iloc[0]
    random_findings = []
    matched_findings = []
    matched_day_counts = []
    for horizon, group in profit.groupby("horizon", sort=False):
        valid = group.dropna(subset=["actual_percentile"])
        wins = int(valid["actual_percentile"].gt(0.5).sum())
        excess = (valid["actual_basket_return"] - valid["random_mean"]).mean()
        random_findings.append(
            f"{horizon.split('d')[-1]}日 {excess * 100:+.2f}pp，"
            f"胜 {wins}/{len(valid)} 日，单边符号检验 p={_binomial_greater_p(wins, len(valid)):.3f}"
        )
        day_differences = matched_profit.loc[
            matched_profit["match_status"].eq("matched")
        ].groupby("decision_date")[f"paired_diff_{horizon}"].mean().dropna()
        matched_findings.append(
            f"{horizon.split('d')[-1]}日 {day_differences.mean() * 100:+.2f}pp"
            if len(day_differences)
            else f"{horizon.split('d')[-1]}日 NA"
        )
        matched_day_counts.append(
            f"{horizon.split('d')[-1]}日 {len(day_differences)} 个"
        )
    matched_rows = matched_profit.loc[matched_profit["match_status"].eq("matched")]
    matched_label_count = matched_rows["event_id"].nunique()
    total_label_count = matched_profit["event_id"].nunique()
    stale = []
    for source in ("reports", "forecast", "express", "toplist"):
        dates = coverage.loc[~coverage[f"{source}_covered"], "decision_date"]
        if len(dates):
            stale.append(f"{source}: " + ", ".join(pd.Timestamp(d).strftime("%Y-%m-%d") for d in dates))
    text = f"""# 交易者决策层：首轮结果

生成日期：{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}。本报告全部为探索性结果；统计独立单位为买入日。模仿度与未来收益分别裁决。

## 1. 候选池先验检查

| 池 | 每日候选中位数 | L4逐笔召回 | 有命中日期占比 |
|---|---:|---:|---:|
{os.linesep.join(pool_lines)}

`legacy_P150` 是旧阈值规则（名称仅为历史沿用，并不表示每日固定 104–150 只）；`pv_top3pct` 是参考 VCP/leader/reacceleration 构造的固定 3% 多形态池；`high_recall_parent` 是较宽的多形态并集。任何真实买入都没有被强行补入候选池。

## 2. 分层选择消融

| 排名池 | 累加实验 | 候选中位数 | Recall@5 | Recall@10 | Recall@20 |
|---|---|---:|---:|---:|---:|
{os.linesep.join(ablation_lines)}

族顺序固定为：量价 → 动态方向 → 方向内龙头 → 事件/关注度 → 熟悉池；组合替换仅在严格 T-1 完整持仓日进入。当前只有 8 个独立 L4 买入日，因此没有训练机器学习权重，以上均为日内百分位和固定等权。

## 3. 同统计方向代理随机篮子收益对照

| 持有口径 | 实际篮子均值 | 同 PIT 收益聚类随机均值 | 实际平均分位 | 有效模拟数 |
|---|---:|---:|---:|---:|
{os.linesep.join(profit_lines)}

收益为 `open[T+h]/open[T]-1`；净收益另扣 7bp 往返成本。这里的“方向”只是严格 PIT 的收益聚类代理，不是已经验证的行业或题材。该检验回答“统计方向代理给定后，他挑的个股是否优于随机”，不用于反向优化选择分数。

## 4. 首轮裁决

1. **旧候选池不是高召回入口。** `legacy_P150` 每日中位 {pool_lookup.loc['legacy_P150', 'median_candidates']:.0f} 只，只覆盖 {_format_pct(pool_lookup.loc['legacy_P150', 'micro_pool_recall'])} 的 L4 买入；扩大到中位 {pool_lookup.loc['high_recall_parent', 'median_candidates']:.0f} 只也只覆盖 {_format_pct(pool_lookup.loc['high_recall_parent', 'micro_pool_recall'])}。参考 VCP/leader/reacceleration 的固定 3% 池覆盖更低（{_format_pct(pool_lookup.loc['pv_top3pct', 'micro_pool_recall'])}）。因此不能在这 100–600 只里直接学习最后 3–5 只，否则一半以上答案已在第一层丢失。
2. **量价适合作为画像，不足以还原最终决策。** 单独量价使真实买入的中位日内分位升至 {_format_pct(single.loc['only_price_volume', 'median_selected_percentile'])}，但 Recall@10 仍是 {_format_pct(single.loc['only_price_volume', 'micro_recall_at_10'])}。动态方向、方向内龙头、事件关注单独的中位分位分别为 {_format_pct(single.loc['only_dynamic_direction', 'median_selected_percentile'])}、{_format_pct(single.loc['only_within_direction_leader', 'median_selected_percentile'])}、{_format_pct(single.loc['only_event_attention', 'median_selected_percentile'])}，均未产生 Top-20 命中。
3. **熟悉池是唯一可见的行为线索，但证据只有一例。** 单独熟悉度的中位分位为 {_format_pct(single.loc['only_familiarity', 'median_selected_percentile'])}；加入后完整模型首次出现 Top-5 命中，最终也只有 {int(round(final_row.micro_recall_at_5 * final_row.n_labels))}/{int(final_row.n_labels)}。逐笔核对表明该命中来自此前已披露持有过的蓝色光标，不能凭一例外推为稳定因子。
4. **组合替换没有观察到增量。** 它只在严格 T-1 有完整普通股持仓的 3 个买入日可计算；加入前后 Recall@5/10/20 不变。ETF 没有被当作待卖股票，持仓快照也没有向前携带。
5. **统计方向代理给定后的个股收益只有弱线索，尚未过证据门槛。** 相对同一 PIT 收益聚类随机篮子的均值差为：{'；'.join(random_findings)}；与更严格的同聚类、流动性、波动匹配股相比则为：{'；'.join(matched_findings)}。严格匹配覆盖 {matched_label_count}/{total_label_count} 笔标签；可计算的独立日期分别为 {'、'.join(matched_day_counts)}。两类对照方向不一致，不能认定已有稳定选股 alpha。

综合判断：现有证据与“**量价/形态做宽筛，题材语义和人工判断完成最后选择，纪律化执行买卖**”相容，但不支持“当前这些可见因子已经构成可复现的纯量化模型”。下一道最有价值的数据闸门是带历史生效日期的概念/题材成员与新闻语义，而不是继续堆叠量价窗口。

## 5. 数据边界

过期而记为未知的源：{'；'.join(stale) if stale else '无'}。

- 当前申万快照不进入主模型；方向主口径为前季末 120 日收益聚类、下一季度生效的 PIT 统计行业。
- 8 月 28 日天华新能尚未同时具备规范事件与全市场行情，因此没有混入主样本。
- 无帖子或缺图日期不产生负标签；事件字段在源截止后是 NA，不是 0。

## 6. 相似方法的使用边界

- [IBD 官方 CAN SLIM / SwingTrader 指南](https://get.investors.com/wp-content/uploads/2025/01/How-To-Use-SwingTrader-2025.pdf)：只借鉴“基本面/质量预筛 + 技术执行”的分层流程，不移植美股仓位和止损阈值。
- [Mark Minervini 官方 SEPA 说明](https://www.minervini.com/)：只映射高动量筛选、领导力和图形确认；未公开的 VCP 专有阈值不作假设。
- [George & Hwang 52周高点动量论文](https://onlinelibrary.wiley.com/doi/10.1111/j.1540-6261.2004.00695.x)：支持检验既往领导地位，不支持把深回撤百分比当成论文结论。
- [Opening-range breakout 原始论文](https://www.sciencedirect.com/science/article/pii/S1544612312000438)：只作为后续盘中确认/执行层假设；其原油期货窗口和阈值不进入 A 股选股模型。

完整逐日结果见 `research/ledger/trader_choice_*`；冻结协议见 `research/TRADER_DECISION_LAYER_PROTOCOL.md`。
"""
    path.write_text(text, encoding="utf-8")


def _manifest(
    paths: ChoiceDataPaths,
    config: ChoiceConfig,
    args: argparse.Namespace,
    *,
    daily_inputs: Mapping[str, pd.DataFrame] | None = None,
) -> dict[str, object]:
    inputs = [
        paths.events_file,
        paths.positions_file,
        paths.reports_file,
        paths.forecast_file,
        paths.express_file,
        paths.toplist_file,
        ROOT / "src" / "afactor" / "trader_choice.py",
        ROOT / "src" / "afactor" / "trader_choice_data.py",
        ROOT / "src" / "afactor" / "trader_choice_portfolio.py",
        ROOT / "scripts" / "run_trader_decision_layers.py",
        ROOT / "research" / "TRADER_DECISION_LAYER_PROTOCOL.md",
        paths.daily_research_dir / "open_auction_px.parquet",
        paths.daily_research_dir / "close_px.parquet",
        paths.daily_research_dir / "amount.parquet",
        paths.daily_sealed_dir / "open_auction_px.parquet",
        paths.daily_sealed_dir / "close_px.parquet",
        paths.daily_sealed_dir / "amount.parquet",
        paths.statind_labels_file,
    ]
    config_payload = {
        "rank_ks": config.rank_ks,
        "seed": config.random_seed,
        "families": {
            **{key: list(value) for key, value in FAMILIES_5.items()},
            "portfolio_replacement": ["f6_portfolio_replacement_score"],
        },
        "round_trip_cost": ROUND_TRIP_COST,
        "random_repeats": args.random_repeats,
    }
    manifest: dict[str, object] = {
        "manifest_schema": 2,
        "generated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "mode": args.mode,
        "inputs": [_file_signature(path) for path in inputs],
        "input_directories": [_directory_signature(paths.recent_parts_dir)],
        "config": config_payload,
        "config_sha256": hashlib.sha256(
            json.dumps(config_payload, sort_keys=True).encode("utf-8")
        ).hexdigest(),
    }
    if daily_inputs is not None:
        manifest["exact_daily_panels"] = [
            _frame_signature(name, frame) for name, frame in daily_inputs.items()
        ]
    return manifest


def run_discovery(args: argparse.Namespace) -> None:
    paths = ChoiceDataPaths(args.astock_root, args.trader_root)
    config = ChoiceConfig(random_seed=args.seed)
    labels = load_choice_labels(paths.events_file, config)
    panel, opens, closes, amounts = _build_feature_panel(
        labels["decision_date"].unique(), paths, config
    )
    _assert_frozen_regression(panel, labels)

    pool_daily, pool_counts = _pool_audit(panel, labels, config)
    pool_summary = _aggregate_daily(pool_daily, ["pool"])

    families_6 = OrderedDict(FAMILIES_5)
    families_6["portfolio_replacement"] = ("f6_portfolio_replacement_score",)
    ablation_parts = []
    for pool in ("full_eligible", "high_recall_parent"):
        daily = evaluate_ablations(
            panel, labels, families_6, config, pool_col=pool
        )
        daily.insert(0, "pool", pool)
        ablation_parts.append(daily)
    ablation_daily = pd.concat(ablation_parts, ignore_index=True)
    ablation_summary = _aggregate_daily(
        ablation_daily, ["pool", "experiment_kind", "experiment_name", "families"]
    )

    controls = _strict_controls(panel, labels, config)
    matched_profit = _matched_profit(panel, labels, controls)
    random_profit = _within_direction_random(
        panel, labels, repeats=args.random_repeats, seed=args.seed
    )
    coverage = _coverage_table(panel)

    args.ledger_dir.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(args.ledger_dir / "trader_choice_panel.parquet", index=False)
    labels.to_csv(args.ledger_dir / "trader_choice_labels.csv", index=False)
    pool_daily.to_csv(args.ledger_dir / "trader_choice_pool_daily.csv", index=False)
    pool_counts.to_csv(args.ledger_dir / "trader_choice_pool_counts.csv", index=False)
    pool_summary.to_csv(args.ledger_dir / "trader_choice_pool_summary.csv", index=False)
    ablation_daily.to_csv(args.ledger_dir / "trader_choice_ablation_daily.csv", index=False)
    ablation_summary.to_csv(args.ledger_dir / "trader_choice_ablation_summary.csv", index=False)
    controls.to_csv(args.ledger_dir / "trader_choice_matched_controls.csv", index=False)
    matched_profit.to_csv(args.ledger_dir / "trader_choice_matched_profit.csv", index=False)
    random_profit.to_csv(args.ledger_dir / "trader_choice_within_direction_random.csv", index=False)
    coverage.to_csv(args.ledger_dir / "trader_choice_coverage.csv", index=False)
    manifest = _manifest(
        paths,
        config,
        args,
        daily_inputs={"open": opens, "close": closes, "amount": amounts},
    )
    (args.ledger_dir / "trader_choice_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    _write_report(
        args.report,
        pool_summary,
        ablation_summary,
        random_profit,
        matched_profit,
        coverage,
    )
    print(f"labels={len(labels)} decision_days={labels.decision_date.nunique()}")
    print(f"panel_rows={len(panel)} report={args.report}")


def run_score_day(args: argparse.Namespace) -> None:
    if not args.as_of:
        raise SystemExit("score-day requires --as-of YYYY-MM-DD")
    as_of = pd.Timestamp(args.as_of).normalize()
    current = _validate_forward_as_of(as_of)
    paths = ChoiceDataPaths(args.astock_root, args.trader_root)
    config = ChoiceConfig(random_seed=args.seed)
    destination = args.ledger_dir / "trader_choice_forward"
    path = destination / f"{as_of.date()}.parquet"
    manifest_path = path.with_suffix(".manifest.json")
    if path.exists() or manifest_path.exists():
        raise FileExistsError(
            "Forward artifacts are append-only and already exist for "
            f"{as_of.date()}"
        )
    panel, opens, closes, amounts = _build_feature_panel(
        [as_of], paths, config, include_outcomes=False
    )
    families_6 = OrderedDict(FAMILIES_5)
    families_6["portfolio_replacement"] = ("f6_portfolio_replacement_score",)
    scored = score_factor_families(
        panel, families_6, config, pool_col="full_eligible"
    ).sort_values([config.score_col, "symbol"], ascending=[False, True])
    outcome_columns = [
        column for column in scored.columns if column.startswith(OUTCOME_PREFIXES)
    ]
    if outcome_columns:
        raise AssertionError(
            "Forward snapshot contains forbidden outcome columns: "
            + ", ".join(outcome_columns)
        )
    destination.mkdir(parents=True, exist_ok=True)
    scored.to_parquet(path, index=False)
    manifest = _manifest(
        paths,
        config,
        args,
        daily_inputs={"open": opens, "close": closes, "amount": amounts},
    )
    manifest["as_of"] = str(as_of.date())
    manifest["snapshot_created_at_utc"] = current.tz_convert("UTC").isoformat()
    manifest["snapshot_created_at_exchange"] = current.isoformat()
    manifest["snapshot_sha256"] = _sha256(path)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    entry_hash = _append_forward_log(destination, path, manifest_path, manifest)
    print(f"forward_snapshot={path} rows={len(scored)} log_entry={entry_hash}")


def _load_verified_forward(directory: Path) -> pd.DataFrame:
    files = sorted(directory.glob("*.parquet"))
    if not files:
        raise FileNotFoundError("No forward snapshots exist")
    log_path = directory / "append_log.jsonl"
    if not log_path.exists():
        raise ValueError("Forward append log is missing")
    entries: dict[str, dict[str, object]] = {}
    previous_hash: str | None = None
    for line_number, line in enumerate(log_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line:
            continue
        entry = json.loads(line)
        observed_hash = str(entry.pop("entry_sha256"))
        canonical = json.dumps(
            entry, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        expected_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        if observed_hash != expected_hash:
            raise ValueError(f"Forward log entry {line_number} hash mismatch")
        if entry.get("previous_entry_sha256") != previous_hash:
            raise ValueError(f"Forward log entry {line_number} chain mismatch")
        previous_hash = observed_hash
        entry["entry_sha256"] = observed_hash
        entries[str(entry["snapshot"])] = entry
    if set(entries) != {path.name for path in files}:
        raise ValueError("Forward snapshot files and append log do not match")
    frames = []
    for path in files:
        manifest_path = path.with_suffix(".manifest.json")
        if not manifest_path.exists():
            raise ValueError(f"Forward manifest is missing: {manifest_path}")
        entry = entries[path.name]
        if _sha256(path) != entry["snapshot_sha256"]:
            raise ValueError(f"Forward snapshot hash mismatch: {path}")
        if _sha256(manifest_path) != entry["manifest_sha256"]:
            raise ValueError(f"Forward manifest hash mismatch: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("as_of") != path.stem:
            raise ValueError(f"Forward manifest date mismatch: {manifest_path}")
        frame = pd.read_parquet(path)
        outcome_columns = [
            column for column in frame.columns if column.startswith(OUTCOME_PREFIXES)
        ]
        if outcome_columns:
            raise ValueError(f"Forward snapshot contains outcome columns: {path}")
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def run_forward_eval(args: argparse.Namespace) -> None:
    paths = ChoiceDataPaths(args.astock_root, args.trader_root)
    config = ChoiceConfig(random_seed=args.seed)
    labels = load_choice_labels(paths.events_file, config)
    panel = _load_verified_forward(args.ledger_dir / "trader_choice_forward")
    covered = labels.loc[labels["decision_date"].isin(panel["decision_date"])].copy()
    result = evaluate_ranking(
        panel, covered, config, pool_col="full_eligible", score_col=config.score_col
    )
    path = args.ledger_dir / "trader_choice_forward_eval.csv"
    result.to_csv(path, index=False)
    print(f"forward_labels={len(covered)} evaluation={path}")


def main() -> None:
    args = parse_args()
    if args.mode == "discovery":
        run_discovery(args)
    elif args.mode == "score-day":
        run_score_day(args)
    else:
        run_forward_eval(args)


if __name__ == "__main__":
    main()
