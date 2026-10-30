#!/usr/bin/env python3
"""Validate the trader's public method claims against disclosed positions.

This is a descriptive, post-hoc audit.  It deliberately keeps three questions
separate:

1. Were disclosed buys/holdings spread across broad Shenwan industries?
2. Were the chosen industries themselves strong on T-1 price/volume data?
3. Were the chosen stocks leaders within their selected industries?

Static Shenwan and theme memberships are oracle classifications.  Numerical
features come from the already-frozen ``trader_oracle_panel.parquet`` and use
T-1 data only.  The script never treats undisclosed stocks as true negatives.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "research" / "ledger"


def _float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _median(values: Iterable[float]) -> float | None:
    clean = list(values)
    return statistics.median(clean) if clean else None


def _mean(values: Iterable[float]) -> float | None:
    clean = list(values)
    return statistics.fmean(clean) if clean else None


def _fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.12g}"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _load_oracle_panel(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    try:
        import duckdb  # type: ignore[import-not-found]
    except ImportError as exc:
        raise SystemExit(
            "duckdb is required to read the frozen parquet panel; install it in "
            "the active research environment"
        ) from exc

    query = """
        SELECT
            CAST(CAST(decision_date AS DATE) AS VARCHAR) AS decision_date,
            LEFT(symbol, 6) AS security_code,
            sw_l1_static_oracle,
            oracle_static_best_theme_name,
            f7_structured_direction_score,
            f8_structured_leader_score,
            pv_score,
            f7_oracle_static_semantic_score
        FROM read_parquet(?)
    """
    columns = [
        "decision_date",
        "security_code",
        "sw_l1_static_oracle",
        "oracle_static_best_theme_name",
        "direction_score",
        "leader_score",
        "price_volume_score",
        "semantic_score",
    ]
    connection = duckdb.connect()
    try:
        records = connection.execute(query, [str(path)]).fetchall()
    finally:
        connection.close()
    return {
        (str(record[0]), str(record[1])): dict(zip(columns, record, strict=True))
        for record in records
    }


def _security_name_map(position_rows: list[dict[str, str]]) -> dict[str, str]:
    result: dict[str, str] = {}
    for row in position_rows:
        code = row.get("security_code", "")
        name = row.get("security_name", "")
        if code.isdigit() and len(code) == 6 and name:
            result[code] = name
    return result


def _build_trade_rows(
    label_rows: list[dict[str, str]],
    oracle: dict[tuple[str, str], dict[str, Any]],
    names: dict[str, str],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in label_rows:
        date = row["decision_date"]
        symbol = row["symbol"]
        code = symbol[:6]
        source = oracle.get((date, code), {})
        direction = _float(source.get("direction_score"))
        leader = _float(source.get("leader_score"))
        output.append(
            {
                "event_id": row["event_id"],
                "decision_date": date,
                "symbol": symbol,
                "security_name": names.get(code, ""),
                "sw_l1_static_oracle": source.get("sw_l1_static_oracle") or "",
                "oracle_static_best_theme_name": source.get(
                    "oracle_static_best_theme_name"
                )
                or "",
                "direction_strength_percentile": _fmt(direction),
                "direction_top_half": "" if direction is None else int(direction >= 0.5),
                "direction_top_30pct": "" if direction is None else int(direction >= 0.7),
                "within_direction_leader_percentile": _fmt(leader),
                "leader_top_half": "" if leader is None else int(leader >= 0.5),
                "leader_top_30pct": "" if leader is None else int(leader >= 0.7),
                "price_volume_score": _fmt(_float(source.get("price_volume_score"))),
                "semantic_chain_score": _fmt(_float(source.get("semantic_score"))),
                "classification_clock": "static_oracle",
                "numeric_feature_clock": "T-1",
            }
        )
    return sorted(output, key=lambda row: (row["decision_date"], row["symbol"]))


def _build_holding_rows(
    position_rows: list[dict[str, str]],
    oracle: dict[tuple[str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for row in position_rows:
        date = row.get("trading_date", "")
        code = row.get("security_code", "")
        name = row.get("security_name", "")
        quantity = _float(row.get("total_quantity_best"))
        key = (date, code)
        if (
            row.get("confidence") != "L4"
            or quantity is None
            or quantity <= 0
            or not (code.isdigit() and len(code) == 6)
            or "ETF" in name.upper()
            or key not in oracle
            or key in seen
        ):
            continue
        seen.add(key)
        source = oracle[key]
        output.append(
            {
                "trading_date": date,
                "security_code": code,
                "security_name": name,
                "sw_l1_static_oracle": source.get("sw_l1_static_oracle") or "",
                "oracle_static_best_theme_name": source.get(
                    "oracle_static_best_theme_name"
                )
                or "",
                "direction_strength_percentile": _fmt(
                    _float(source.get("direction_score"))
                ),
                "within_direction_leader_percentile": _fmt(
                    _float(source.get("leader_score"))
                ),
                "price_volume_score": _fmt(_float(source.get("price_volume_score"))),
                "classification_clock": "static_oracle",
                "numeric_feature_clock": "T-1",
            }
        )
    return sorted(output, key=lambda row: (row["trading_date"], row["security_code"]))


def _group_summary(rows: list[dict[str, Any]], date_field: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[date_field])].append(row)
    output: list[dict[str, Any]] = []
    for date, members in sorted(grouped.items()):
        sw = [str(row["sw_l1_static_oracle"]) for row in members if row["sw_l1_static_oracle"]]
        themes = [
            str(row["oracle_static_best_theme_name"])
            for row in members
            if row["oracle_static_best_theme_name"]
        ]
        direction = [
            value
            for value in (
                _float(row["direction_strength_percentile"]) for row in members
            )
            if value is not None
        ]
        leader = [
            value
            for value in (
                _float(row["within_direction_leader_percentile"]) for row in members
            )
            if value is not None
        ]
        repeated_themes = sorted(
            {theme for theme in themes if themes.count(theme) > 1}
        )
        output.append(
            {
                "date": date,
                "n_rows": len(members),
                "n_sw_mapped": len(sw),
                "n_unique_sw_l1": len(set(sw)),
                "sw_l1_diversification_ratio": _fmt(
                    len(set(sw)) / len(sw) if sw else None
                ),
                "all_sw_l1_distinct": "" if not sw else int(len(set(sw)) == len(sw)),
                "n_theme_mapped": len(themes),
                "n_unique_best_theme": len(set(themes)),
                "theme_diversification_ratio": _fmt(
                    len(set(themes)) / len(themes) if themes else None
                ),
                "repeated_best_themes": "|".join(repeated_themes),
                "mean_direction_strength_percentile": _fmt(_mean(direction)),
                "median_direction_strength_percentile": _fmt(_median(direction)),
                "n_direction_top_half": sum(value >= 0.5 for value in direction),
                "n_leader_scored": len(leader),
                "mean_within_direction_leader_percentile": _fmt(_mean(leader)),
                "median_within_direction_leader_percentile": _fmt(_median(leader)),
                "n_leader_top_half": sum(value >= 0.5 for value in leader),
            }
        )
    return output


def _metric(
    metric_id: str,
    value: Any,
    denominator: Any = "",
    interpretation: str = "",
) -> dict[str, Any]:
    return {
        "metric_id": metric_id,
        "value": value,
        "denominator": denominator,
        "interpretation": interpretation,
    }


def _build_summary(
    trades: list[dict[str, Any]],
    holdings: list[dict[str, Any]],
    trade_days: list[dict[str, Any]],
    holding_days: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    trade_direction = [
        value
        for value in (_float(row["direction_strength_percentile"]) for row in trades)
        if value is not None
    ]
    trade_leader = [
        value
        for value in (
            _float(row["within_direction_leader_percentile"]) for row in trades
        )
        if value is not None
    ]
    holding_direction = [
        value
        for value in (
            _float(row["direction_strength_percentile"]) for row in holdings
        )
        if value is not None
    ]
    holding_leader = [
        value
        for value in (
            _float(row["within_direction_leader_percentile"]) for row in holdings
        )
        if value is not None
    ]
    multi_trade_days = [row for row in trade_days if int(row["n_rows"]) > 1]
    multi_trade_count = sum(int(row["n_rows"]) for row in multi_trade_days)
    multi_trade_unique_sw = sum(int(row["n_unique_sw_l1"]) for row in multi_trade_days)
    holding_count = sum(int(row["n_rows"]) for row in holding_days)
    holding_unique_sw = sum(int(row["n_unique_sw_l1"]) for row in holding_days)
    holding_theme_mapped = sum(int(row["n_theme_mapped"]) for row in holding_days)
    holding_unique_theme = sum(int(row["n_unique_best_theme"]) for row in holding_days)
    return [
        _metric("confirmed_buy_rows", len(trades), "rows", "L4 ordinary-stock buys"),
        _metric("confirmed_buy_days", len(trade_days), "days"),
        _metric("buy_direction_median_percentile", _fmt(_median(trade_direction)), len(trade_direction), "not consistently top sectors"),
        _metric("buy_direction_top_half_count", sum(value >= 0.5 for value in trade_direction), len(trade_direction)),
        _metric("buy_direction_top_30pct_count", sum(value >= 0.7 for value in trade_direction), len(trade_direction)),
        _metric("buy_leader_median_percentile", _fmt(_median(trade_leader)), len(trade_leader), "strong within-direction preference"),
        _metric("buy_leader_top_half_count", sum(value >= 0.5 for value in trade_leader), len(trade_leader)),
        _metric("buy_leader_top_30pct_count", sum(value >= 0.7 for value in trade_leader), len(trade_leader)),
        _metric("multi_buy_sw_unique_slot_ratio", _fmt(multi_trade_unique_sw / multi_trade_count), multi_trade_count, "broad-industry diversification"),
        _metric("multi_buy_days_all_sw_distinct", sum(int(row["all_sw_l1_distinct"]) for row in multi_trade_days), len(multi_trade_days)),
        _metric("holding_rows_on_oracle_dates", len(holdings), "L4 rows"),
        _metric("holding_direction_median_percentile", _fmt(_median(holding_direction)), len(holding_direction), "broad-sector strength not required"),
        _metric("holding_direction_top_half_count", sum(value >= 0.5 for value in holding_direction), len(holding_direction)),
        _metric("holding_leader_median_percentile", _fmt(_median(holding_leader)), len(holding_leader), "persistent within-sector leadership"),
        _metric("holding_leader_top_half_count", sum(value >= 0.5 for value in holding_leader), len(holding_leader)),
        _metric("holding_sw_unique_slot_ratio", _fmt(holding_unique_sw / holding_count), holding_count),
        _metric("holding_theme_unique_slot_ratio", _fmt(holding_unique_theme / holding_theme_mapped), holding_theme_mapped, "theme risk is more concentrated than security count"),
        _metric("classification_clock", "static_oracle", "", "not forward reproducible"),
        _metric("numeric_feature_clock", "T-1", "", "no future price/volume inputs"),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--label-ranks", type=Path, default=LEDGER / "trader_oracle_label_ranks.csv"
    )
    parser.add_argument(
        "--oracle-panel", type=Path, default=LEDGER / "trader_oracle_panel.parquet"
    )
    parser.add_argument(
        "--position-state",
        type=Path,
        default=Path.home()
        / "Documents"
        / "ChatGPT"
        / "Stock model"
        / "data"
        / "normalized"
        / "position_state.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=LEDGER)
    args = parser.parse_args()

    labels = _read_csv(args.label_ranks)
    if len(labels) != 20 or len({row["decision_date"] for row in labels}) != 8:
        raise SystemExit("frozen L4 label audit must contain 20 buys on 8 dates")
    positions = _read_csv(args.position_state)
    oracle = _load_oracle_panel(args.oracle_panel)
    names = _security_name_map(positions)
    trades = _build_trade_rows(labels, oracle, names)
    holdings = _build_holding_rows(positions, oracle)
    trade_days = _group_summary(trades, "decision_date")
    holding_days = _group_summary(holdings, "trading_date")
    summary = _build_summary(trades, holdings, trade_days, holding_days)

    _write_csv(
        args.output_dir / "trader_method_video_validation_trades.csv",
        trades,
        list(trades[0]),
    )
    _write_csv(
        args.output_dir / "trader_method_video_validation_holdings.csv",
        holdings,
        list(holdings[0]),
    )
    _write_csv(
        args.output_dir / "trader_method_video_validation_trade_days.csv",
        trade_days,
        list(trade_days[0]),
    )
    _write_csv(
        args.output_dir / "trader_method_video_validation_holding_days.csv",
        holding_days,
        list(holding_days[0]),
    )
    _write_csv(
        args.output_dir / "trader_method_video_validation_summary.csv",
        summary,
        ["metric_id", "value", "denominator", "interpretation"],
    )
    print(
        f"wrote {len(trades)} trades, {len(holdings)} holding rows, "
        f"{len(trade_days)} trade days, and {len(holding_days)} holding days"
    )


if __name__ == "__main__":
    main()
