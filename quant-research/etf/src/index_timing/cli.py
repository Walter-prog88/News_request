"""Command-line entry point for the research project."""

from __future__ import annotations

import fcntl
import json
import os
import subprocess
from datetime import date, datetime
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from .artifacts import (
    append_trial_ledger_record,
    dataframe_sha256,
    discover_runs,
    read_trial_ledger,
    trial_spec_sha256,
)
from .data_service import (
    data_status,
    select_assets,
    sync_selected_assets,
    sync_selected_cash,
    sync_selected_macro,
)
from .paths import ARTIFACTS_DIR, ROOT, ensure_runtime_dirs
from .reporting import _preferred_runs, build_report
from .runner import (
    _runtime_metadata,
    generate_demo,
    run_caitong_reconstruction,
    validate_caitong,
)
from .signal_ledger import (
    chained_signal_record,
    parse_signal_ledger,
    prospective_signal_assessment,
)
from .validation_reporting import update_cross_run_validation

app = typer.Typer(no_args_is_help=True, help="指数择时研报重构、独立验证与美股迁移")
data_app = typer.Typer(no_args_is_help=True, help="抓取、校验和查看不可变数据快照")
report_app = typer.Typer(no_args_is_help=True, help="生成研究报告")
signal_app = typer.Typer(no_args_is_help=True, help="维护不可回写的前瞻信号日志")
app.add_typer(data_app, name="data")
app.add_typer(report_app, name="report")
app.add_typer(signal_app, name="signal")
console = Console()


def _record_validation_trial(
    *,
    status: str,
    trial_spec: dict[str, object],
    pvalue: float | None = None,
    artifact_run_id: str | None = None,
    metadata: dict[str, object] | None = None,
) -> None:
    """Append one attempted hypothesis unless that exact outcome is recorded."""

    spec_hash = trial_spec_sha256(trial_spec)
    for record in read_trial_ledger():
        if (
            record["trial_spec_hash"] == spec_hash
            and record["status"] == status
            and record["artifact_run_id"] == artifact_run_id
        ):
            return
    append_trial_ledger_record(
        status=status,  # type: ignore[arg-type]
        trial_spec=trial_spec,
        pvalue=pvalue,
        artifact_run_id=artifact_run_id,
        metadata=metadata,
    )


@data_app.command("sync")
def data_sync(
    study: Annotated[str, typer.Option(help="all/us/caitong/zhongyou")] = "all",
    asset: Annotated[str | None, typer.Option(help="仅同步一个asset_id")] = None,
    start: Annotated[str | None, typer.Option(help="覆盖配置起始日期 YYYY-MM-DD")] = None,
    end: Annotated[str | None, typer.Option(help="含当日 YYYY-MM-DD")] = None,
    include_cash: Annotated[
        bool, typer.Option(help="同时同步资产配置中的DGS3MO/DR007现金利率")
    ] = True,
) -> None:
    """Download configured price/cash data and write hashed Parquet snapshots."""

    snapshots, failures = sync_selected_assets(
        study=study,
        asset_id=asset,
        start=start,
        end=end,
        include_cash=include_cash,
    )
    for snapshot in snapshots:
        console.print(
            f"[green]saved[/green] {snapshot.dataset_id} · {snapshot.row_count:,} rows · "
            f"{snapshot.sha256[:12]}"
        )
    for asset_id, reason in failures.items():
        console.print(f"[red]failed[/red] {asset_id}: {reason}")
    if failures and not snapshots:
        raise typer.Exit(code=1)


@data_app.command("sync-cash")
def data_sync_cash(
    study: Annotated[str, typer.Option(help="all/us/caitong/zhongyou")] = "all",
    asset: Annotated[str | None, typer.Option(help="按一个asset_id读取cash_series配置")] = None,
    series: Annotated[
        str | None, typer.Option(help="直接指定DGS3MO或DR007；省略则按资产配置")
    ] = None,
    start: Annotated[str | None, typer.Option(help="起始日期 YYYY-MM-DD")] = None,
    end: Annotated[str | None, typer.Option(help="含当日 YYYY-MM-DD")] = None,
    csv_path: Annotated[
        Path | None, typer.Option(help="DR007锁定CSV；也可设置INDEX_TIMING_DR007_CSV")
    ] = None,
) -> None:
    """Synchronize configured cash yields as immutable point-in-time snapshots."""

    snapshots, failures = sync_selected_cash(
        study=study,
        asset_id=asset,
        series_id=series,
        start=start,
        end=end,
        csv_path=csv_path,
    )
    for snapshot in snapshots:
        console.print(
            f"[green]saved[/green] {snapshot.dataset_id} · {snapshot.row_count:,} rows · "
            f"{snapshot.sha256[:12]}"
        )
    for series_id, reason in failures.items():
        console.print(f"[red]failed[/red] {series_id}: {reason}")
    if failures and not snapshots:
        raise typer.Exit(code=1)


@data_app.command("sync-macro")
def data_sync_macro(
    series: Annotated[
        str | None, typer.Option(help="仅同步macro_us.yml中的一个FRED series id")
    ] = None,
    start: Annotated[str | None, typer.Option(help="起始日期 YYYY-MM-DD")] = None,
    end: Annotated[str | None, typer.Option(help="含当日 YYYY-MM-DD")] = None,
) -> None:
    """Synchronize configured FRED/ALFRED factors as immutable PIT snapshots."""

    snapshots, failures, skipped = sync_selected_macro(
        series_id=series,
        start=start,
        end=end,
    )
    for snapshot in snapshots:
        console.print(
            f"[green]saved[/green] {snapshot.dataset_id} · {snapshot.row_count:,} rows · "
            f"{snapshot.sha256[:12]}"
        )
    for series_id, reason in skipped.items():
        console.print(f"[yellow]skipped[/yellow] {series_id}: {reason}")
    for series_id, reason in failures.items():
        console.print(f"[red]failed[/red] {series_id}: {reason}")
    if failures and not snapshots:
        raise typer.Exit(code=1)


@data_app.command("status")
def show_data_status() -> None:
    frame = data_status()
    if frame.empty:
        console.print("No snapshots. Run `index-timing data sync`.")
        return
    table = Table(*frame.columns)
    for row in frame.astype(str).itertuples(index=False, name=None):
        table.add_row(*row)
    console.print(table)


@app.command("reproduce")
def reproduce(
    study: Annotated[str, typer.Option(help="all/caitong/zhongyou")] = "all",
    asset: Annotated[str | None, typer.Option(help="仅运行一个asset_id")] = None,
    model: Annotated[str, typer.Option(help="财通: mlp/elastic_net_logistic/gru/lstm")] = "mlp",
    representation: Annotated[str, typer.Option(help="中邮: raw/pca/lstm")] = "lstm",
    train_window: Annotated[str, typer.Option(help="中邮冲突训练起点: 2008/2016")] = "2008",
    horizon: Annotated[int, typer.Option(help="财通标签期: 1/5/10/20")] = 5,
    seed: Annotated[int, typer.Option(help="单次研报重构随机种子")] = 0,
    quick: Annotated[bool, typer.Option(help="缩短神经网络epoch用于链路检查")] = False,
    microcap_patch: Annotated[bool, typer.Option(help="单列运行中邮微盘事后开度补丁")] = False,
    feature_set: Annotated[
        str,
        typer.Option(
            help=(
                "财通特征消融: technical/technical_plus_valuation_fund/"
                "technical_plus_market_global/full"
            )
        ),
    ] = "full",
) -> None:
    """Run report-constrained reconstruction and save daily artifacts."""

    if study not in {"all", "caitong", "zhongyou"}:
        raise typer.BadParameter("study must be all, caitong or zhongyou")
    selected = select_assets(study="all" if study == "all" else study, asset_id=asset)
    failures = 0
    for spec in selected:
        requested_studies = [name for name in ("caitong", "zhongyou") if name in spec.studies]
        if study != "all":
            requested_studies = [study] if study in spec.studies else []
        for requested in requested_studies:
            try:
                if requested == "caitong":
                    path = run_caitong_reconstruction(
                        spec.asset_id,
                        model_kind=model,  # type: ignore[arg-type]
                        seed=seed,
                        horizon=horizon,
                        quick=quick,
                        feature_set=feature_set,
                    )
                else:
                    from .runner_zhongyou import run_zhongyou_reconstruction

                    path = run_zhongyou_reconstruction(
                        spec.asset_id,
                        representation=representation,  # type: ignore[arg-type]
                        seed=seed,
                        train_window=train_window,
                        quick=quick,
                        microcap_patch=microcap_patch,
                    )
                console.print(f"[green]complete[/green] {path}")
            except Exception as exc:
                failures += 1
                console.print(
                    f"[red]failed[/red] {requested}/{spec.asset_id}: "
                    f"{type(exc).__name__}: {exc}"
                )
    if failures:
        raise typer.Exit(code=1)


@app.command("validate")
def validate(
    study: Annotated[str, typer.Option(help="all/caitong/zhongyou")] = "all",
    asset: Annotated[str | None, typer.Option(help="仅验证一个asset_id")] = None,
    seeds: Annotated[int, typer.Option(min=1, max=10, help="使用前N个预注册种子")] = 10,
    bootstrap: Annotated[int, typer.Option(min=100, help="正式规格为10000")] = 10_000,
    representation: Annotated[str, typer.Option(help="中邮: raw/pca/lstm")] = "lstm",
    quick: Annotated[bool, typer.Option(help="快速链路检查，不可作为正式证据")] = False,
    feature_set: Annotated[
        str,
        typer.Option(
            help=(
                "财通特征消融: technical/technical_plus_valuation_fund/"
                "technical_plus_market_global/full"
            )
        ),
    ] = "full",
) -> None:
    """Run purged annual walk-forward validation."""

    if study not in {"all", "caitong", "zhongyou"}:
        raise typer.BadParameter("study must be all, caitong or zhongyou")
    selected = select_assets(study="all" if study == "all" else study, asset_id=asset)
    failures = 0
    runtime = _runtime_metadata()
    for spec in selected:
        requested_studies = [name for name in ("caitong", "zhongyou") if name in spec.studies]
        if study != "all":
            requested_studies = [study] if study in spec.studies else []
        for requested in requested_studies:
            trial_spec: dict[str, object] = {
                "family": "index_timing_independent_validation_v1",
                "study": requested,
                "asset_id": spec.asset_id,
                "seeds": list(range(seeds)),
                "bootstrap_replications": bootstrap,
                "quick_mode": quick,
                "feature_set": feature_set if requested == "caitong" else None,
                "representation": representation if requested == "zhongyou" else None,
                "source_tree_sha256": runtime["source_tree_sha256"],
                "uv_lock_sha256": runtime["uv_lock_sha256"],
            }
            try:
                if requested == "caitong":
                    path = validate_caitong(
                        spec.asset_id,
                        seeds=range(seeds),
                        n_boot=bootstrap,
                        quick=quick,
                        feature_set=feature_set,
                    )
                else:
                    from .runner_zhongyou import validate_zhongyou

                    path = validate_zhongyou(
                        spec.asset_id,
                        representation=representation,  # type: ignore[arg-type]
                        seeds=range(seeds),
                        n_boot=bootstrap,
                        quick=quick,
                    )
                artifact_metadata = json.loads(
                    (path / "metadata.json").read_text(encoding="utf-8")
                )
                artifact_metrics = json.loads(
                    (path / "metrics.json").read_text(encoding="utf-8")
                )
                trial_spec.update(
                    {
                        "signal_market_snapshot_sha256": artifact_metadata.get(
                            "signal_market_snapshot_sha256"
                        ),
                        "execution_market_snapshot_sha256": artifact_metadata.get(
                            "execution_market_snapshot_sha256"
                        ),
                        "cash_rate_snapshot_sha256": artifact_metadata.get(
                            "cash_rate_snapshot_sha256"
                        ),
                        "macro_snapshot_sha256": sorted(
                            source.get("snapshot_sha256")
                            for source in artifact_metadata.get("macro_series_audit", [])
                            if source.get("snapshot_sha256")
                        ),
                        "robustness_parameter_grid": artifact_metadata.get(
                            "robustness_parameter_grid"
                        ),
                    }
                )
                _record_validation_trial(
                    status="success",
                    trial_spec=trial_spec,
                    pvalue=float(artifact_metrics["bootstrap_pvalue"]),
                    artifact_run_id=path.name,
                    metadata={
                        "formal_evidence_eligible": artifact_metadata.get(
                            "formal_evidence_eligible", False
                        )
                    },
                )
                console.print(f"[green]complete[/green] {path}")
            except Exception as exc:
                failures += 1
                _record_validation_trial(
                    status="failed",
                    trial_spec=trial_spec,
                    metadata={"error": f"{type(exc).__name__}: {exc}"},
                )
                console.print(
                    f"[red]failed[/red] {requested}/{spec.asset_id}: "
                    f"{type(exc).__name__}: {exc}"
                )
    correction = update_cross_run_validation(spa_reps=100 if quick else 1_000)
    if not correction.empty:
        console.print(
            f"[green]multiplicity[/green] BH-FDR updated across {len(correction)} saved trials"
        )
    if failures:
        raise typer.Exit(code=1)


@report_app.command("build")
def report_build(
    output: Annotated[Path | None, typer.Option(help="HTML输出路径")] = None,
) -> None:
    path = build_report(output)
    console.print(f"[green]report[/green] {path}")


@app.command("app")
def launch_app() -> None:
    """Launch the local Streamlit research dashboard."""

    dashboard = ROOT / "app" / "dashboard.py"
    raise typer.Exit(
        subprocess.call(
            [str(ROOT / ".tools" / "uv"), "run", "streamlit", "run", str(dashboard)],
            cwd=ROOT,
        )
    )


@app.command("demo")
def demo() -> None:
    path = generate_demo()
    report = build_report()
    console.print(f"[green]demo[/green] {path}")
    console.print(f"[green]report[/green] {report}")


@signal_app.command("log")
def signal_log(
    as_of: Annotated[str | None, typer.Option(help="默认今天；格式YYYY-MM-DD")] = None,
) -> None:
    """Append versioned, tamper-evident signals without rewriting history.

    A historical artifact is marked ineligible for prospective evidence unless
    its source row is the as-of date and it exposes a close-generated target for
    the next open.
    """

    ensure_runtime_dirs()
    log_path = ARTIFACTS_DIR / "forward_signals.jsonl"
    requested_date = date.fromisoformat(as_of) if as_of else None
    appended = 0
    with log_path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            handle.seek(0)
            existing_records, previous_hash = parse_signal_ledger(handle.read())
            existing = {
                (
                    record["run_id"],
                    record["as_of"],
                    record.get("daily_sha256"),
                    record.get("signal_sha256", record.get("daily_sha256")),
                )
                for record in existing_records
            }
            handle.seek(0, os.SEEK_END)
            for run in _preferred_runs(discover_runs()):
                import pandas as pd

                daily = pd.read_parquet(run["daily_path"])
                if daily.empty:
                    continue
                daily_sha256 = dataframe_sha256(daily)
                forward_path = run["path"] / "forward_signals.parquet"
                source = pd.read_parquet(forward_path) if forward_path.exists() else daily
                if source.empty:
                    continue
                source_sha256 = dataframe_sha256(source)
                source_date = pd.Timestamp(source.index[-1]).date()
                target_date = requested_date or source_date
                key = (
                    run["run_id"],
                    target_date.isoformat(),
                    daily_sha256,
                    source_sha256,
                )
                if key in existing:
                    continue
                row = source.iloc[-1]

                def clean(value: object) -> object:
                    if value is None or bool(pd.isna(value)):
                        return None
                    return value.item() if hasattr(value, "item") else value

                state = clean(row.get("state"))
                if state is None:
                    state = clean(row.get("map_state"))
                target_source = next(
                    (
                        column
                        for column in (
                            "signal_position",
                            "target_position",
                            "signal",
                            "position",
                            "exposure",
                        )
                        if column in row.index and clean(row.get(column)) is not None
                    ),
                    None,
                )
                target_position = clean(row.get(target_source)) if target_source else None
                logged_at = datetime.now().astimezone()
                metadata = run.get("metadata", {})
                assessment = prospective_signal_assessment(
                    source_date=source_date,
                    as_of=target_date,
                    calendar_name=metadata.get("calendar"),
                    artifact_written_at=metadata.get("written_at"),
                    snapshot_retrieved_at=metadata.get(
                        "signal_market_snapshot_retrieved_at"
                    ),
                    logged_at=logged_at,
                    target_source_column=target_source,
                    quick_mode=metadata.get("quick_mode"),
                    feature_set_degraded=bool(
                        metadata.get("feature_set_degraded", False)
                    ),
                    cash_rate_status=metadata.get("cash_rate_status"),
                    registered_specification=bool(
                        metadata.get("registered_experiment_spec")
                    ),
                )
                prospective_eligible = bool(assessment["eligible"])
                payload = {
                    "version": 3,
                    "run_id": run["run_id"],
                    "as_of": target_date.isoformat(),
                    "daily_sha256": daily_sha256,
                    "signal_sha256": source_sha256,
                    "signal_source_artifact": (
                        "forward_signals.parquet"
                        if forward_path.exists()
                        else "daily.parquet_legacy_fallback"
                    ),
                    "source_row_date": source_date.isoformat(),
                    "target_position": target_position,
                    "target_source_column": target_source,
                    "planned_execution": "next_trading_session_open",
                    "planned_execution_session": assessment[
                        "planned_execution_session"
                    ],
                    "planned_execution_open": assessment["planned_execution_open"],
                    "signal_session_close": assessment["signal_session_close"],
                    "probability": clean(
                        row.get("ensemble_probability", row.get("probability"))
                    ),
                    "state": state,
                    "logged_at": logged_at.isoformat(),
                    "prospective_evidence_eligible": prospective_eligible,
                    "prospective_eligibility_reasons": assessment["reasons"],
                    "evidence_class": (
                        "prospective_signal"
                        if prospective_eligible
                        else "historical_artifact_snapshot_not_prospective"
                    ),
                    "tamper_evident": True,
                }
                record = chained_signal_record(payload, previous_hash=previous_hash)
                handle.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                        default=str,
                        allow_nan=False,
                    )
                    + "\n"
                )
                previous_hash = record["record_hash"]
                existing.add(key)
                appended += 1
            handle.flush()
            os.fsync(handle.fileno())
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    console.print(f"[green]appended[/green] {appended} signal records to {log_path}")


@signal_app.command("verify")
def signal_verify() -> None:
    """Verify the complete forward-signal hash chain."""

    log_path = ARTIFACTS_DIR / "forward_signals.jsonl"
    raw = log_path.read_text(encoding="utf-8") if log_path.exists() else ""
    records, tip = parse_signal_ledger(raw)
    console.print(
        f"[green]verified[/green] {len(records)} signal records · chain tip {tip[:12]}"
    )


if __name__ == "__main__":
    app()
