"""Build a self-contained Chinese research report from saved runs."""

from __future__ import annotations

import html
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
from jinja2 import Environment, FileSystemLoader, select_autoescape
from plotly.offline import get_plotlyjs
from plotly.subplots import make_subplots

from .artifacts import discover_runs, read_trial_ledger, verify_trial_ledger
from .paths import ARTIFACTS_DIR, GENERATED_REPORTS_DIR, REPORTS_DIR, ensure_runtime_dirs
from .signal_ledger import parse_signal_ledger
from .statistics import paired_neural_increment_test

DISPLAY_METRICS = {
    "cagr": "年化收益",
    "annual_return": "年化收益",
    "annual_volatility": "年化波动",
    "sharpe": "Sharpe",
    "sortino": "Sortino",
    "max_drawdown": "最大回撤",
    "calmar": "Calmar",
    "annual_alpha": "对等敞口回归Alpha",
    "active_return_vs_equal_exposure": "对等敞口主动收益",
    "jensen_alpha_vs_market": "市场Jensen Alpha",
    "information_ratio": "信息比率",
    "turnover": "年化换手",
    "average_exposure": "平均仓位",
    "interval_win_rate": "区间胜率",
    "probability_brier_score": "概率Brier分数",
    "probability_calibration_ece": "概率校准ECE",
}


def _pick_column(frame: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    return next((column for column in candidates if column in frame.columns), None)


def _normalise_equity(returns: pd.Series) -> pd.Series:
    clean = pd.to_numeric(returns, errors="coerce").fillna(0.0).clip(lower=-0.999999)
    return (1.0 + clean).cumprod()


def _preferred_runs(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select the newest immutable artifact for each requested experiment id.

    Older content-addressed versions remain on disk and in the trial ledger.  A
    report should not silently count those historical reruns as independent
    strategies or choose one by directory-name ordering.
    """

    preferred: dict[str, dict[str, Any]] = {}

    def timestamp_key(raw: Any) -> int:
        if not raw:
            return pd.Timestamp.min.value
        timestamp = pd.Timestamp(raw)
        if timestamp.tzinfo is not None:
            timestamp = timestamp.tz_convert("UTC").tz_localize(None)
        return timestamp.value

    for record in records:
        metadata = record["metadata"]
        experiment_id = str(metadata.get("requested_run_id") or record["run_id"])
        raw_timestamp = metadata.get("written_at") or metadata.get("created_at")
        timestamp = timestamp_key(raw_timestamp)
        candidate_key = (timestamp, str(record["run_id"]))
        current = preferred.get(experiment_id)
        if current is None:
            preferred[experiment_id] = record
            continue
        current_metadata = current["metadata"]
        current_raw_timestamp = current_metadata.get("written_at") or current_metadata.get(
            "created_at"
        )
        current_timestamp = timestamp_key(current_raw_timestamp)
        current_key = (current_timestamp, str(current["run_id"]))
        if candidate_key > current_key:
            preferred[experiment_id] = record
    return [preferred[key] for key in sorted(preferred)]


def run_figure(frame: pd.DataFrame, run_name: str) -> str:
    strategy_return = _pick_column(
        frame, ("strategy_net_return", "net_return", "strategy_return", "strategy_gross_return")
    )
    benchmark_return = _pick_column(frame, ("benchmark_return", "asset_return", "execution_return"))
    equal_exposure_return = _pick_column(frame, ("equal_exposure_return",))
    position = _pick_column(frame, ("position", "exposure", "target_position"))

    figure = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.72, 0.28],
        vertical_spacing=0.08,
        subplot_titles=("累计净值（对数轴）", "仓位"),
    )
    if strategy_return:
        figure.add_trace(
            go.Scatter(
                x=frame.index,
                y=_normalise_equity(frame[strategy_return]),
                name="策略",
                line={"width": 2.2},
            ),
            row=1,
            col=1,
        )
    if benchmark_return:
        figure.add_trace(
            go.Scatter(
                x=frame.index,
                y=_normalise_equity(frame[benchmark_return]),
                name="买入持有",
                line={"width": 1.5, "dash": "dot"},
            ),
            row=1,
            col=1,
        )
    if equal_exposure_return:
        figure.add_trace(
            go.Scatter(
                x=frame.index,
                y=_normalise_equity(frame[equal_exposure_return]),
                name="等敞口基准",
                line={"width": 1.5, "dash": "dash"},
            ),
            row=1,
            col=1,
        )
    if position:
        figure.add_trace(
            go.Scatter(
                x=frame.index,
                y=pd.to_numeric(frame[position], errors="coerce"),
                name="仓位",
                line={"width": 1.2, "shape": "hv"},
                fill="tozeroy",
            ),
            row=2,
            col=1,
        )
    figure.update_yaxes(type="log", row=1, col=1)
    figure.update_yaxes(range=[-0.05, 1.05], tickformat=".0%", row=2, col=1)
    figure.update_layout(
        title=run_name,
        height=650,
        margin={"l": 55, "r": 30, "t": 70, "b": 45},
        legend={"orientation": "h", "y": 1.06, "x": 1, "xanchor": "right"},
        hovermode="x unified",
        template="plotly_white",
    )
    return figure.to_html(full_html=False, include_plotlyjs=False, config={"displaylogo": False})


def _format_metric(key: str, value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, (int, float)):
        if key in {
            "cagr",
            "annual_return",
            "annual_volatility",
            "max_drawdown",
            "annual_alpha",
            "active_return_vs_equal_exposure",
            "jensen_alpha_vs_market",
            "turnover",
            "average_exposure",
            "interval_win_rate",
        }:
            return f"{value:.2%}"
        return f"{value:.3f}"
    return str(value)


def build_report(output: str | Path | None = None) -> Path:
    ensure_runtime_dirs()
    output_path = Path(output) if output else GENERATED_REPORTS_DIR / "index-timing-report.html"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    all_runs = discover_runs()
    runs = _preferred_runs(all_runs)
    rendered_runs: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, str]] = []
    for record in runs:
        daily = pd.read_parquet(record["daily_path"])
        if not isinstance(daily.index, pd.DatetimeIndex) and "date" in daily:
            daily = daily.set_index(pd.to_datetime(daily.pop("date")))
        metric_rows = [
            {
                "label": label,
                "value": _format_metric(key, record["metrics"].get(key)),
            }
            for key, label in DISPLAY_METRICS.items()
            if key in record["metrics"]
        ]
        assumptions = record["metadata"].get("assumptions", [])
        rendered_runs.append(
            {
                "run_id": record["run_id"],
                "study": record["metadata"].get("study", "unknown"),
                "asset": record["metadata"].get("asset_id", "unknown"),
                "status": record["metrics"].get("status", "Not evaluated"),
                "expanded": record["run_id"] == "validation__caitong__sp500",
                "figure": run_figure(daily, record["run_id"]),
                "metrics": metric_rows,
                "assumptions": assumptions,
                "metadata_json": html.escape(
                    json.dumps(record["metadata"], ensure_ascii=False, indent=2)
                ),
            }
        )
        comparison_rows.append(
            {
                "run_id": record["run_id"],
                "study": str(record["metadata"].get("study", "unknown")),
                "asset": str(record["metadata"].get("asset_id", "unknown")),
                "evidence": str(
                    record["metadata"].get("replication_level", "unclassified")
                ),
                "formal": (
                    "是" if record["metadata"].get("formal_evidence_eligible") else "否"
                ),
                "cagr": _format_metric("cagr", record["metrics"].get("cagr")),
                "sharpe": _format_metric("sharpe", record["metrics"].get("sharpe")),
                "drawdown": _format_metric(
                    "max_drawdown", record["metrics"].get("max_drawdown")
                ),
                "alpha": _format_metric(
                    "annual_alpha", record["metrics"].get("annual_alpha")
                ),
                "status": str(record["metrics"].get("status", "Not evaluated")),
            }
        )

    by_id: dict[str, dict[str, Any]] = {}
    for record in runs:
        by_id[record["run_id"]] = record
        requested = record["metadata"].get("requested_run_id")
        if requested:
            by_id[str(requested)] = record
    increment_rows: list[dict[str, str]] = []
    asset_ids = sorted(
        {str(record["metadata"].get("asset_id", "unknown")) for record in runs}
    )
    for asset_id in asset_ids:
        comparisons = (
            (
                "财通固定切分：MLP - ElasticNet",
                f"caitong__{asset_id}__mlp__h5__seed0",
                f"caitong__{asset_id}__elastic_net_logistic__h5__seed0",
            ),
            (
                "中邮固定切分：LSTM - PCA",
                f"zhongyou__{asset_id}__lstm__train2008__seed0",
                f"zhongyou__{asset_id}__pca__train2008__seed0",
            ),
            (
                "中邮Walk-forward：LSTM - PCA",
                f"validation__zhongyou__lstm__{asset_id}",
                f"validation__zhongyou__pca__{asset_id}",
            ),
        )
        for label, neural_id, baseline_id in comparisons:
            if neural_id not in by_id or baseline_id not in by_id:
                continue
            neural_metrics = by_id[neural_id]["metrics"]
            baseline_metrics = by_id[baseline_id]["metrics"]
            delta_alpha = float(neural_metrics.get("annual_alpha", 0.0)) - float(
                baseline_metrics.get("annual_alpha", 0.0)
            )
            delta_sharpe = float(neural_metrics.get("sharpe", 0.0)) - float(
                baseline_metrics.get("sharpe", 0.0)
            )
            economic_pass = delta_alpha >= 0.005 and delta_sharpe >= 0.10
            significance = "收益路径不足，无法配对检验"
            neural_daily = pd.read_parquet(by_id[neural_id]["daily_path"])
            baseline_daily = pd.read_parquet(by_id[baseline_id]["daily_path"])
            required = {"net_return", "equal_exposure_return"}
            if required.issubset(neural_daily) and required.issubset(baseline_daily):
                increment = paired_neural_increment_test(
                    neural_daily["net_return"],
                    baseline_daily["net_return"],
                    neural_daily["equal_exposure_return"],
                    baseline_benchmark_returns=baseline_daily[
                        "equal_exposure_return"
                    ],
                    risk_free_returns=(
                        neural_daily["cash_return"]
                        if "cash_return" in neural_daily
                        else None
                    ),
                    n_boot=1_000,
                    seed=0,
                )
                significance = (
                    "配对20日块Bootstrap "
                    f"p={increment['alpha_increment_pvalue_one_sided']:.3f}"
                    "（快速；仍需10种子）"
                )
            increment_rows.append(
                {
                    "comparison": label,
                    "asset": asset_id,
                    "delta_alpha": f"{delta_alpha:.2%}",
                    "delta_sharpe": f"{delta_sharpe:.3f}",
                    "economic_pass": "通过" if economic_pass else "未通过",
                    "significance": significance,
                }
            )

    env = Environment(
        loader=FileSystemLoader(REPORTS_DIR / "templates"),
        autoescape=select_autoescape(("html", "xml")),
    )
    template = env.get_template("report.html.j2")
    ledger_records = read_trial_ledger()
    trial_audit = {
        "verified": verify_trial_ledger(),
        "total": len(ledger_records),
        "success": sum(record["status"] == "success" for record in ledger_records),
        "failed": sum(record["status"] == "failed" for record in ledger_records),
        "skipped": sum(record["status"] == "skipped" for record in ledger_records),
        "formal_eligible": sum(
            bool(record.get("metadata", {}).get("formal_evidence_eligible"))
            for record in ledger_records
        ),
    }
    signal_path = ARTIFACTS_DIR / "forward_signals.jsonl"
    signal_records, signal_tip = parse_signal_ledger(
        signal_path.read_text(encoding="utf-8") if signal_path.exists() else ""
    )
    eligible_signal_records = [
        record
        for record in signal_records
        if bool(record.get("prospective_evidence_eligible"))
    ]
    eligible_run_sessions = {
        (str(record.get("run_id")), str(record.get("source_row_date")))
        for record in eligible_signal_records
    }
    sessions_per_run: dict[str, set[str]] = {}
    for run_id, source_date in eligible_run_sessions:
        sessions_per_run.setdefault(run_id, set()).add(source_date)
    signal_audit = {
        "verified": True,
        "total": len(signal_records),
        "prospective": len(eligible_signal_records),
        "unique_run_sessions": len(eligible_run_sessions),
        "maximum_sessions_per_run": max(
            (len(sessions) for sessions in sessions_per_run.values()), default=0
        ),
        "mature_run_count": sum(
            len(sessions) >= 252 for sessions in sessions_per_run.values()
        ),
        "chain_tip": signal_tip[:12],
    }
    output_path.write_text(
        template.render(
            runs=rendered_runs,
            comparison_rows=comparison_rows,
            increment_rows=increment_rows,
            trial_audit=trial_audit,
            signal_audit=signal_audit,
            immutable_artifact_versions=len(all_runs),
            generated_at=datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z"),
            plotly_js=get_plotlyjs(),
        ),
        encoding="utf-8",
    )

    summary_path = output_path.with_suffix(".md")
    lines = [
        "# 指数择时研究结果摘要",
        "",
        f"生成时间：{datetime.now().astimezone().isoformat(timespec='minutes')}",
        "",
        "> 只有 `formal_evidence_eligible=true` 的运行才可用于正式有效性结论。",
        "",
        (
            f"试验台账：{trial_audit['total']}项（成功{trial_audit['success']}、"
            f"失败{trial_audit['failed']}、跳过{trial_audit['skipped']}、"
            f"正式证据资格{trial_audit['formal_eligible']}），哈希链已校验。"
        ),
        (
            f"前瞻信号日志：{signal_audit['total']}条，可登记记录"
            f"{signal_audit['prospective']}条、唯一模型×交易日"
            f"{signal_audit['unique_run_sessions']}个、单模型最多"
            f"{signal_audit['maximum_sessions_per_run']}日、达到252日的模型"
            f"{signal_audit['mature_run_count']}个，链尾 `{signal_audit['chain_tip']}`。"
        ),
        (
            f"报告展示{len(runs)}个实验规格的最新不可变版本；磁盘保留"
            f"{len(all_runs)}个完整历史版本。"
        ),
        "",
    ]
    if not runs:
        lines.append("尚无运行结果。先执行 `index-timing demo` 或真实复现命令。")
    if runs:
        lines.extend(
            [
                "| 运行 | 资产 | CAGR | Sharpe | 最大回撤 | 对等敞口回归Alpha | 正式证据 | 结论 |",
                "|---|---:|---:|---:|---:|---:|---:|---|",
            ]
        )
        for record in runs:
            metrics = record["metrics"]
            metadata = record["metadata"]
            row_template = (
                "| {run} | {asset} | {cagr} | {sharpe} | {dd} | {alpha} | "
                "{formal} | {status} |"
            )
            lines.append(
                row_template.format(
                    run=record["run_id"],
                    asset=metadata.get("asset_id", "unknown"),
                    cagr=_format_metric("cagr", metrics.get("cagr")),
                    sharpe=_format_metric("sharpe", metrics.get("sharpe")),
                    dd=_format_metric("max_drawdown", metrics.get("max_drawdown")),
                    alpha=_format_metric("annual_alpha", metrics.get("annual_alpha")),
                    formal="是" if metadata.get("formal_evidence_eligible") else "否",
                    status=metrics.get("status", "Not evaluated"),
                )
            )
        if not any(
            record["metadata"].get("formal_evidence_eligible") for record in runs
        ):
            lines.extend(
                [
                    "",
                    "当前尚无完成10随机种子、20次GHMM初始化和10,000次Bootstrap的正式运行，因此不能声称策略已被重新证明有效。",
                ]
            )
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path
