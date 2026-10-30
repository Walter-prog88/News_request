from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_trader_decision_layers", ROOT / "scripts" / "run_trader_decision_layers.py"
)
assert SPEC and SPEC.loader
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def _write_forward_artifact(directory: Path, frame: pd.DataFrame) -> Path:
    directory.mkdir(parents=True)
    path = directory / "2026-09-03.parquet"
    frame.to_parquet(path, index=False)
    manifest = {
        "as_of": "2026-09-03",
        "snapshot_created_at_utc": "2026-09-03T01:31:00+00:00",
        "snapshot_sha256": RUNNER._sha256(path),
    }
    manifest_path = path.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    RUNNER._append_forward_log(directory, path, manifest_path, manifest)
    return path


def test_forward_as_of_rejects_backfill_and_future_dates() -> None:
    now = pd.Timestamp("2026-09-03 09:31", tz="Asia/Shanghai")

    observed = RUNNER._validate_forward_as_of(pd.Timestamp("2026-09-03"), now=now)
    assert observed == now
    with pytest.raises(ValueError, match="historical and future"):
        RUNNER._validate_forward_as_of(pd.Timestamp("2026-09-02"), now=now)
    with pytest.raises(ValueError, match="historical and future"):
        RUNNER._validate_forward_as_of(pd.Timestamp("2026-09-04"), now=now)
    with pytest.raises(ValueError, match="starts on"):
        RUNNER._validate_forward_as_of(pd.Timestamp("2026-08-31"), now=now)


def test_forward_loader_verifies_log_and_rejects_outcome_columns(tmp_path: Path) -> None:
    clean = tmp_path / "clean"
    frame = pd.DataFrame(
        {
            "decision_date": [pd.Timestamp("2026-09-03")],
            "symbol": ["000001.SZ"],
            "full_eligible": [True],
            "choice_score": [0.8],
        }
    )
    _write_forward_artifact(clean, frame)
    observed = RUNNER._load_verified_forward(clean)
    assert observed["symbol"].tolist() == ["000001.SZ"]

    contaminated = tmp_path / "contaminated"
    _write_forward_artifact(
        contaminated, frame.assign(fwd_open_to_open_d1=0.1)
    )
    with pytest.raises(ValueError, match="outcome columns"):
        RUNNER._load_verified_forward(contaminated)

