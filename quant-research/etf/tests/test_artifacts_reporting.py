from __future__ import annotations

import json

import pandas as pd

import index_timing.artifacts as artifacts
import index_timing.validation_reporting as validation_reporting
from index_timing.reporting import _preferred_runs, build_report
from index_timing.runner import _source_tree_sha256


def test_artifact_roundtrip_and_report(tmp_path, monkeypatch) -> None:
    run_root = tmp_path / "runs"
    monkeypatch.setattr(artifacts, "RUNS_DIR", run_root)
    index = pd.bdate_range("2024-01-01", periods=10, name="date")
    daily = pd.DataFrame(
        {
            "net_return": [0.001] * len(index),
            "benchmark_return": [0.0005] * len(index),
            "position": [0.5] * len(index),
        },
        index=index,
    )
    run_dir = artifacts.write_run_artifacts(
        "test-run",
        daily=daily,
        metrics={"cagr": 0.12, "sharpe": 1.1, "status": "Promising"},
        metadata={"study": "test", "asset_id": "synthetic", "assumptions": ["test"]},
    )
    assert pd.read_parquet(run_dir / "daily.parquet").equals(daily)
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["daily_rows"] == 10
    assert len(metadata["daily_sha256"]) == 64
    assert len(metadata["artifact_identity"]) == 64
    assert metadata["artifact_run_id"] == "test-run"

    output = build_report(tmp_path / "report.html")
    rendered = output.read_text(encoding="utf-8")
    assert "指数择时研报复现与独立验证" in rendered
    assert "test-run" in rendered


def test_cross_run_fdr_updates_saved_decisions(tmp_path, monkeypatch) -> None:
    records = []
    for run_id, pvalue in (("a", 0.01), ("b", 0.20)):
        run_dir = tmp_path / run_id
        run_dir.mkdir()
        index = pd.bdate_range("2020-01-01", periods=80, name="date")
        daily = pd.DataFrame(
            {
                "net_return": [0.001] * len(index),
                "equal_exposure_return": [0.0002] * len(index),
            },
            index=index,
        )
        daily_path = run_dir / "daily.parquet"
        daily.to_parquet(daily_path)
        metrics = {"bootstrap_pvalue": pvalue, "sharpe": 1.0, "status": "Promising"}
        evidence = {
            "years_history": 12.0,
            "net_alpha": 0.02,
            "alpha_ci_low": 0.001,
            "fdr_reject": False,
            "sharpe_improvement": 0.2,
            "max_drawdown_improvement": 0.2,
            "positive_fold_fraction": 0.7,
            "double_cost_alpha": 0.01,
            "delayed_alpha": 0.01,
            "robust_seed_parameter_fraction": 0.8,
            "best_year_alpha_share": 0.4,
        }
        metadata = {
            "asset_id": "same_asset",
            "replication_level": "independent_validation",
            "validation_evidence": evidence,
        }
        (run_dir / "metrics.json").write_text(json.dumps(metrics))
        (run_dir / "metadata.json").write_text(json.dumps(metadata))
        records.append(
            {
                "run_id": run_id,
                "path": run_dir,
                "metrics": metrics,
                "metadata": metadata,
                "daily_path": daily_path,
            }
        )
    monkeypatch.setattr(validation_reporting, "discover_runs", lambda: records)
    monkeypatch.setattr(validation_reporting, "read_trial_ledger", lambda: [])
    monkeypatch.setattr(
        validation_reporting,
        "spa_test",
        lambda *args, **kwargs: {"available": True, "pvalues": {"consistent": 0.1}},
    )
    correction = validation_reporting.update_cross_run_validation(spa_reps=10)
    assert bool(correction.loc["a", "reject"])
    assert not bool(correction.loc["b", "reject"])


def test_artifact_write_is_idempotent_and_content_addressed(tmp_path, monkeypatch) -> None:
    run_root = tmp_path / "runs"
    monkeypatch.setattr(artifacts, "RUNS_DIR", run_root)
    index = pd.bdate_range("2024-01-01", periods=4, name="date")
    daily = pd.DataFrame({"net_return": [0.0, 0.01, -0.01, 0.02]}, index=index)
    table = pd.DataFrame({"fold": [1], "alpha": [0.01]})

    base = artifacts.write_run_artifacts(
        "immutable-run",
        daily=daily,
        metrics={"alpha": 0.01},
        metadata={"model": "mlp", "created_at": "2026-01-01T00:00:00+00:00"},
        tables={"folds": table},
    )
    original_metrics = (base / "metrics.json").read_text(encoding="utf-8")
    original_written_at = json.loads(
        (base / "metadata.json").read_text(encoding="utf-8")
    )["written_at"]

    same = artifacts.write_run_artifacts(
        "immutable-run",
        daily=daily,
        metrics={"alpha": 0.01},
        metadata={"model": "mlp", "created_at": "2026-06-01T00:00:00+00:00"},
        tables={"folds": table},
        overwrite=True,
    )
    assert same == base
    assert json.loads((same / "metadata.json").read_text(encoding="utf-8"))[
        "written_at"
    ] == original_written_at

    changed = artifacts.write_run_artifacts(
        "immutable-run",
        daily=daily,
        metrics={"alpha": 0.02},
        metadata={"model": "mlp"},
        tables={"folds": table},
        overwrite=False,
    )
    assert changed != base
    assert changed.name.startswith("immutable-run__artifact_")
    assert (base / "metrics.json").read_text(encoding="utf-8") == original_metrics
    assert json.loads((changed / "metrics.json").read_text(encoding="utf-8"))["alpha"] == 0.02
    changed_metadata = json.loads((changed / "metadata.json").read_text(encoding="utf-8"))
    assert changed_metadata["run_id"] == changed.name
    assert changed_metadata["requested_run_id"] == "immutable-run"

    repeated_changed = artifacts.write_run_artifacts(
        "immutable-run",
        daily=daily,
        metrics={"alpha": 0.02},
        metadata={"model": "mlp", "written_at": "ignored"},
        tables={"folds": table},
    )
    assert repeated_changed == changed
    assert sorted(path.name for path in run_root.iterdir()) == sorted([base.name, changed.name])


def test_report_prefers_latest_artifact_for_each_requested_run() -> None:
    def record(run_id: str, written_at: str, requested: str) -> dict:
        return {
            "run_id": run_id,
            "metadata": {
                "requested_run_id": requested,
                "written_at": written_at,
            },
        }

    records = [
        record("run", "2026-07-13T10:00:00+09:00", "run"),
        record("run__artifact_old", "2026-07-13T11:00:00+09:00", "run"),
        record("other", "2026-07-13T09:00:00+09:00", "other"),
        record("run__artifact_new", "2026-07-13T12:00:00+09:00", "run"),
    ]

    preferred = _preferred_runs(records)

    assert [item["run_id"] for item in preferred] == ["other", "run__artifact_new"]


def test_source_tree_hash_ignores_python_cache_files(tmp_path) -> None:
    (tmp_path / "src" / "package").mkdir(parents=True)
    (tmp_path / "configs").mkdir()
    (tmp_path / "src" / "package" / "model.py").write_text("VALUE = 1\n")
    (tmp_path / "configs" / "experiment.yml").write_text("seed: 0\n")
    (tmp_path / "pyproject.toml").write_text("[project]\nname='test'\n")
    (tmp_path / "uv.lock").write_text("version = 1\n")

    original, complete = _source_tree_sha256(tmp_path)
    cache = tmp_path / "src" / "package" / "__pycache__"
    cache.mkdir()
    (cache / "model.cpython-311.pyc").write_bytes(b"interpreter cache")
    with_cache, still_complete = _source_tree_sha256(tmp_path)

    assert complete and still_complete
    assert with_cache == original

    (tmp_path / "src" / "package" / "model.py").write_text("VALUE = 2\n")
    changed, _ = _source_tree_sha256(tmp_path)
    assert changed != original
