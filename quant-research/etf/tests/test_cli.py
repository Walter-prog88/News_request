import json
from datetime import UTC, date, datetime

import pandas as pd
from typer.testing import CliRunner

from index_timing.artifacts import dataframe_sha256
from index_timing.cli import app
from index_timing.signal_ledger import prospective_signal_assessment


def test_cli_exposes_registered_workflows() -> None:
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("data", "reproduce", "validate", "report", "app", "signal"):
        assert command in result.stdout


def test_signal_log_uses_unsettled_forward_signal_and_versions_artifact(
    tmp_path, monkeypatch
) -> None:
    import index_timing.cli as cli

    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()
    daily_path = tmp_path / "daily.parquet"
    daily = pd.DataFrame(
        {"position": [0.25], "map_state": [3]},
        index=pd.DatetimeIndex(["2026-07-10"], name="date"),
    )
    daily.to_parquet(daily_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    forward = pd.DataFrame(
        {"target_position": [0.50, 0.75], "map_state": [2, 4]},
        index=pd.DatetimeIndex(["2026-07-10", "2026-07-13"], name="date"),
    )
    forward.to_parquet(run_dir / "forward_signals.parquet")
    run = {
        "run_id": "zhongyou__test",
        "path": run_dir,
        "daily_path": daily_path,
        "metadata": {"requested_run_id": "zhongyou__test"},
    }
    monkeypatch.setattr(cli, "ARTIFACTS_DIR", artifacts_dir)
    monkeypatch.setattr(cli, "discover_runs", lambda: [run])

    log_path = artifacts_dir / "forward_signals.jsonl"
    legacy_record = {
        "run_id": run["run_id"],
        "as_of": "2026-07-13",
        "immutable": True,
    }
    log_path.write_text(json.dumps(legacy_record) + "\n", encoding="utf-8")

    first = CliRunner().invoke(app, ["signal", "log", "--as-of", "2026-07-13"])
    assert first.exit_code == 0
    records = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert records[0] == legacy_record
    assert len(records) == 2
    assert records[1]["daily_sha256"] == dataframe_sha256(daily)
    assert records[1]["signal_sha256"] == dataframe_sha256(forward)
    assert records[1]["source_row_date"] == "2026-07-13"
    assert records[1]["state"] == 4
    assert records[1]["target_position"] == 0.75
    assert records[1]["target_source_column"] == "target_position"
    assert records[1]["prospective_evidence_eligible"] is False
    assert len(records[1]["record_hash"]) == 64

    duplicate = CliRunner().invoke(app, ["signal", "log", "--as-of", "2026-07-13"])
    assert duplicate.exit_code == 0
    assert len(log_path.read_text().splitlines()) == 2

    changed = daily.copy()
    changed.loc[:, "position"] = 0.75
    changed.to_parquet(daily_path)
    revised = CliRunner().invoke(app, ["signal", "log", "--as-of", "2026-07-13"])
    assert revised.exit_code == 0
    records = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert len(records) == 3
    assert records[-1]["daily_sha256"] == dataframe_sha256(changed)
    assert records[-1]["daily_sha256"] != records[1]["daily_sha256"]

    verified = CliRunner().invoke(app, ["signal", "verify"])
    assert verified.exit_code == 0
    assert "verified" in verified.stdout

    records[0]["as_of"] = "2026-07-12"
    log_path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8"
    )
    tampered = CliRunner().invoke(app, ["signal", "verify"])
    assert tampered.exit_code != 0


def test_prospective_signal_window_uses_exchange_time_not_host_date() -> None:
    assessment = prospective_signal_assessment(
        source_date=date(2026, 7, 10),
        as_of=date(2026, 7, 10),
        calendar_name="XNYS",
        artifact_written_at=datetime(2026, 7, 10, 20, 10, tzinfo=UTC),
        snapshot_retrieved_at=datetime(2026, 7, 10, 20, 5, tzinfo=UTC),
        logged_at=datetime(2026, 7, 12, 12, 0, tzinfo=UTC),
        target_source_column="target_position",
        quick_mode=False,
        feature_set_degraded=False,
        cash_rate_status="snapshot_available_at",
        registered_specification=True,
    )

    assert assessment["eligible"] is True
    assert assessment["planned_execution_session"] == "2026-07-13"

    quick = prospective_signal_assessment(
        source_date=date(2026, 7, 10),
        as_of=date(2026, 7, 10),
        calendar_name="XNYS",
        artifact_written_at=datetime(2026, 7, 10, 20, 10, tzinfo=UTC),
        snapshot_retrieved_at=datetime(2026, 7, 10, 20, 5, tzinfo=UTC),
        logged_at=datetime(2026, 7, 12, 12, 0, tzinfo=UTC),
        target_source_column="target_position",
        quick_mode=True,
        feature_set_degraded=False,
        cash_rate_status="snapshot_available_at",
        registered_specification=True,
    )

    assert quick["eligible"] is False
    assert "quick_or_unclassified_model" in quick["reasons"]
