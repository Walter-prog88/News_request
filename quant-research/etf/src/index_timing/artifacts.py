"""Auditable run artifact persistence."""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import shutil
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from .paths import ARTIFACTS_DIR, RUNS_DIR, ensure_runtime_dirs

ARTIFACT_IDENTITY_VERSION = 1
TRIAL_LEDGER_VERSION = 1
TRIAL_LEDGER_GENESIS_HASH = "0" * 64
TRIAL_STATUSES = frozenset({"success", "failed", "skipped"})
DERIVED_VALIDATION_FILENAME = "derived_validation.json"

_PERSISTENCE_METADATA_KEYS = frozenset(
    {
        "artifact_identity",
        "artifact_identity_version",
        "artifact_run_id",
        "daily_max_date",
        "daily_min_date",
        "daily_rows",
        "daily_sha256",
        "requested_run_id",
        "run_id",
    }
)
_RUNTIME_TIME_SUFFIXES = (
    "created_at",
    "finished_at",
    "generated_at",
    "retrieved_at",
    "started_at",
    "written_at",
)


class TrialLedgerIntegrityError(ValueError):
    """Raised when an append-only trial ledger fails hash-chain validation."""


def _json_default(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if pd.isna(value):
        return None
    raise TypeError(f"Cannot JSON encode {type(value)!r}")


def _jsonable(value: Any) -> Any:
    """Convert supported research values into a deterministic JSON value."""

    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (set, frozenset)):
        items = [_jsonable(item) for item in value]
        return sorted(items, key=lambda item: _canonical_json(item))
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if isinstance(value, pd.Timedelta):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, float):
        if math.isnan(value):
            return {"__float__": "nan"}
        if math.isinf(value):
            return {"__float__": "inf" if value > 0 else "-inf"}
        return value
    if value is pd.NA or value is pd.NaT:
        return None
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _jsonable(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _without_runtime_fields(value: Any) -> Any:
    """Strip persistence/runtime clocks while retaining research-time information."""

    if isinstance(value, Mapping):
        retained: dict[str, Any] = {}
        for raw_key, item in value.items():
            key = str(raw_key)
            lowered = key.lower()
            if lowered in _PERSISTENCE_METADATA_KEYS:
                continue
            if lowered.endswith(_RUNTIME_TIME_SUFFIXES):
                continue
            retained[key] = _without_runtime_fields(item)
        return retained
    if isinstance(value, (list, tuple)):
        return [_without_runtime_fields(item) for item in value]
    if isinstance(value, (set, frozenset)):
        items = [_without_runtime_fields(item) for item in value]
        return sorted(items, key=lambda item: _canonical_json(item))
    return value


def dataframe_sha256(frame: pd.DataFrame) -> str:
    normalised = frame.sort_index().sort_index(axis=1)
    payload = pd.util.hash_pandas_object(normalised, index=True).values.tobytes()
    return hashlib.sha256(payload).hexdigest()


def _dataframe_fingerprint(frame: pd.DataFrame) -> dict[str, Any]:
    normalised = frame.sort_index().sort_index(axis=1)
    index_dtypes = (
        [str(level.dtype) for level in normalised.index.levels]
        if isinstance(normalised.index, pd.MultiIndex)
        else [str(normalised.index.dtype)]
    )
    return {
        "sha256": dataframe_sha256(normalised),
        "rows": len(normalised),
        "columns": [
            {"label": _jsonable(label), "dtype": str(dtype)}
            for label, dtype in zip(normalised.columns, normalised.dtypes, strict=True)
        ],
        "index": {
            "class": type(normalised.index).__name__,
            "names": _jsonable(list(normalised.index.names)),
            "dtypes": index_dtypes,
        },
    }


def artifact_identity(
    *,
    daily: pd.DataFrame,
    metrics: Mapping[str, Any],
    metadata: Mapping[str, Any],
    tables: Mapping[str, pd.DataFrame] | None = None,
) -> str:
    """Return the stable content identity for one complete research artifact.

    Runtime clocks and persistence-derived metadata are intentionally excluded.
    Data, result metrics, provenance, model metadata and auxiliary tables remain
    identity-bearing.
    """

    payload = {
        "version": ARTIFACT_IDENTITY_VERSION,
        "daily": _dataframe_fingerprint(daily),
        "metrics": _without_runtime_fields(metrics),
        "metadata": _without_runtime_fields(metadata),
        "tables": {
            name: _dataframe_fingerprint(frame)
            for name, frame in sorted((tables or {}).items())
        },
    }
    return _sha256_json(payload)


def _read_artifact_identity(run_dir: Path) -> str | None:
    daily_path = run_dir / "daily.parquet"
    metrics_path = run_dir / "metrics.json"
    metadata_path = run_dir / "metadata.json"
    if not (daily_path.exists() and metrics_path.exists() and metadata_path.exists()):
        return None
    try:
        daily = pd.read_parquet(daily_path)
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        tables = {
            path.stem: pd.read_parquet(path)
            for path in sorted(run_dir.glob("*.parquet"))
            if path.name != "daily.parquet"
        }
        return artifact_identity(
            daily=daily,
            metrics=metrics,
            metadata=metadata,
            tables=tables,
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None


def _content_addressed_run_dir(run_id: str, identity: str) -> Path:
    base = RUNS_DIR / run_id
    if not base.exists() or _read_artifact_identity(base) == identity:
        return base

    for length in range(12, len(identity) + 1, 4):
        candidate = RUNS_DIR / f"{run_id}__artifact_{identity[:length]}"
        if not candidate.exists() or _read_artifact_identity(candidate) == identity:
            return candidate
    raise RuntimeError(f"Unable to resolve a collision-free artifact path for {run_id!r}")


def write_run_artifacts(
    run_id: str,
    *,
    daily: pd.DataFrame,
    metrics: dict[str, Any],
    metadata: dict[str, Any],
    tables: dict[str, pd.DataFrame] | None = None,
    overwrite: bool = True,
) -> Path:
    """Persist one immutable research run without replacing historical content.

    ``overwrite`` remains only for source compatibility; neither value permits an
    overwrite.  An identical artifact is idempotent.  Different content requested
    under an occupied ``run_id`` receives a content-addressed directory suffix.
    """

    ensure_runtime_dirs()
    del overwrite

    ordered = daily.copy()
    if isinstance(ordered.index, pd.DatetimeIndex):
        ordered = ordered.sort_index()
        ordered.index.name = ordered.index.name or "date"
    identity = artifact_identity(
        daily=ordered,
        metrics=metrics,
        metadata=metadata,
        tables=tables,
    )
    run_dir = _content_addressed_run_dir(run_id, identity)
    if run_dir.exists():
        return run_dir

    run_dir.mkdir(parents=True, exist_ok=False)
    try:
        ordered.to_parquet(run_dir / "daily.parquet")
        for name, frame in (tables or {}).items():
            preserve_index = not isinstance(frame.index, pd.RangeIndex)
            frame.to_parquet(run_dir / f"{name}.parquet", index=preserve_index)

        enriched_metadata = {
            **metadata,
            "run_id": run_dir.name,
            "requested_run_id": run_id,
            "artifact_run_id": run_dir.name,
            "artifact_identity": identity,
            "artifact_identity_version": ARTIFACT_IDENTITY_VERSION,
            "written_at": datetime.now().astimezone().isoformat(),
            "daily_sha256": dataframe_sha256(ordered),
            "daily_rows": len(ordered),
            "daily_min_date": ordered.index.min() if len(ordered) else None,
            "daily_max_date": ordered.index.max() if len(ordered) else None,
        }
        (run_dir / "metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2, default=_json_default),
            encoding="utf-8",
        )
        (run_dir / "metadata.json").write_text(
            json.dumps(enriched_metadata, ensure_ascii=False, indent=2, default=_json_default),
            encoding="utf-8",
        )
    except Exception:
        shutil.rmtree(run_dir)
        raise
    return run_dir


def trial_spec_sha256(trial_spec: Mapping[str, Any]) -> str:
    """Hash a fully specified validation trial independent of mapping key order."""

    return _sha256_json(trial_spec)


def _trial_record_hash(record: Mapping[str, Any]) -> str:
    return _sha256_json({key: value for key, value in record.items() if key != "record_hash"})


def _verify_trial_records(records: list[dict[str, Any]]) -> None:
    previous_hash = TRIAL_LEDGER_GENESIS_HASH
    required = {
        "version",
        "sequence",
        "recorded_at",
        "status",
        "trial_spec",
        "trial_spec_hash",
        "pvalue",
        "artifact_run_id",
        "metadata",
        "previous_hash",
        "record_hash",
    }
    for expected_sequence, record in enumerate(records, start=1):
        missing = required.difference(record)
        if missing:
            raise TrialLedgerIntegrityError(
                f"Trial ledger record {expected_sequence} lacks {sorted(missing)}"
            )
        if record["version"] != TRIAL_LEDGER_VERSION:
            raise TrialLedgerIntegrityError(
                f"Unsupported trial ledger version at record {expected_sequence}"
            )
        if record["sequence"] != expected_sequence:
            raise TrialLedgerIntegrityError(
                f"Trial ledger sequence mismatch at record {expected_sequence}"
            )
        if record["status"] not in TRIAL_STATUSES:
            raise TrialLedgerIntegrityError(
                f"Invalid trial status at record {expected_sequence}: {record['status']!r}"
            )
        if not isinstance(record["trial_spec"], Mapping):
            raise TrialLedgerIntegrityError(
                f"trial_spec is not an object at record {expected_sequence}"
            )
        artifact_run_id = record["artifact_run_id"]
        if artifact_run_id is not None and (
            not isinstance(artifact_run_id, str) or not artifact_run_id.strip()
        ):
            raise TrialLedgerIntegrityError(
                f"Invalid artifact_run_id at record {expected_sequence}"
            )
        if record["status"] == "success" and artifact_run_id is None:
            raise TrialLedgerIntegrityError(
                f"Successful trial lacks artifact_run_id at record {expected_sequence}"
            )
        if record["previous_hash"] != previous_hash:
            raise TrialLedgerIntegrityError(
                f"Trial ledger previous_hash mismatch at record {expected_sequence}"
            )
        if trial_spec_sha256(record["trial_spec"]) != record["trial_spec_hash"]:
            raise TrialLedgerIntegrityError(
                f"Trial specification hash mismatch at record {expected_sequence}"
            )
        try:
            pvalue = float(record["pvalue"])
        except (TypeError, ValueError) as exc:
            raise TrialLedgerIntegrityError(
                f"Invalid pvalue at record {expected_sequence}"
            ) from exc
        if not 0.0 <= pvalue <= 1.0:
            raise TrialLedgerIntegrityError(
                f"pvalue outside [0, 1] at record {expected_sequence}"
            )
        if record["status"] == "failed" and pvalue != 1.0:
            raise TrialLedgerIntegrityError(
                f"Failed trial must carry pvalue=1 at record {expected_sequence}"
            )
        expected_hash = _trial_record_hash(record)
        if record["record_hash"] != expected_hash:
            raise TrialLedgerIntegrityError(
                f"Trial ledger record_hash mismatch at record {expected_sequence}"
            )
        previous_hash = record["record_hash"]


def _parse_trial_ledger(raw: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(raw.splitlines(), start=1):
        if not line.strip():
            raise TrialLedgerIntegrityError(f"Blank line in trial ledger at line {line_number}")
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise TrialLedgerIntegrityError(
                f"Malformed JSON in trial ledger at line {line_number}"
            ) from exc
        if not isinstance(record, dict):
            raise TrialLedgerIntegrityError(
                f"Trial ledger line {line_number} is not a JSON object"
            )
        records.append(record)
    _verify_trial_records(records)
    return records


def read_trial_ledger(
    ledger_path: Path | None = None,
    *,
    verify: bool = True,
) -> list[dict[str, Any]]:
    """Read the trial ledger, verifying its complete hash chain by default."""

    path = ledger_path or (ARTIFACTS_DIR / "trial_ledger.jsonl")
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
        try:
            raw = handle.read()
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    if verify:
        return _parse_trial_ledger(raw)
    records: list[dict[str, Any]] = []
    for line in raw.splitlines():
        if line.strip():
            record = json.loads(line)
            if not isinstance(record, dict):
                raise TrialLedgerIntegrityError("Trial ledger contains a non-object record")
            records.append(record)
    return records


def verify_trial_ledger(ledger_path: Path | None = None) -> bool:
    """Validate the full ledger chain, raising on any corruption."""

    read_trial_ledger(ledger_path, verify=True)
    return True


def append_trial_ledger_record(
    *,
    status: Literal["success", "failed", "skipped"],
    trial_spec: Mapping[str, Any],
    pvalue: float | None = None,
    artifact_run_id: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    ledger_path: Path | None = None,
    recorded_at: datetime | None = None,
) -> dict[str, Any]:
    """Append one validation trial after validating the existing hash chain.

    Failed trials are always assigned ``pvalue=1`` so failures remain inside the
    multiple-testing universe.  Skipped trials default to the same conservative
    value.  Successful trials require both an observed p-value and artifact id.
    """

    if status not in TRIAL_STATUSES:
        raise ValueError(f"status must be one of {sorted(TRIAL_STATUSES)}")
    if status == "success":
        if pvalue is None:
            raise ValueError("Successful trials require a pvalue")
        if not artifact_run_id:
            raise ValueError("Successful trials require artifact_run_id")
    elif status == "failed":
        pvalue = 1.0
    elif pvalue is None:
        pvalue = 1.0

    numeric_pvalue = float(pvalue)
    if not 0.0 <= numeric_pvalue <= 1.0:
        raise ValueError("pvalue must be inside [0, 1]")
    if artifact_run_id is not None and (
        not isinstance(artifact_run_id, str) or not artifact_run_id.strip()
    ):
        raise ValueError("artifact_run_id must be non-empty when supplied")

    path = ledger_path or (ARTIFACTS_DIR / "trial_ledger.jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    normalised_spec = _jsonable(trial_spec)
    normalised_metadata = _jsonable(metadata or {})
    timestamp = (recorded_at or datetime.now().astimezone()).isoformat()

    with path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            handle.seek(0)
            records = _parse_trial_ledger(handle.read())
            previous_hash = (
                records[-1]["record_hash"] if records else TRIAL_LEDGER_GENESIS_HASH
            )
            record: dict[str, Any] = {
                "version": TRIAL_LEDGER_VERSION,
                "sequence": len(records) + 1,
                "recorded_at": timestamp,
                "status": status,
                "trial_spec": normalised_spec,
                "trial_spec_hash": trial_spec_sha256(normalised_spec),
                "pvalue": numeric_pvalue,
                "artifact_run_id": artifact_run_id,
                "metadata": normalised_metadata,
                "previous_hash": previous_hash,
            }
            record["record_hash"] = _trial_record_hash(record)
            handle.seek(0, os.SEEK_END)
            handle.write(_canonical_json(record) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    return record


def discover_runs() -> list[dict[str, Any]]:
    ensure_runtime_dirs()
    records: list[dict[str, Any]] = []
    for run_dir in sorted(RUNS_DIR.iterdir()):
        if not run_dir.is_dir():
            continue
        metrics_path = run_dir / "metrics.json"
        metadata_path = run_dir / "metadata.json"
        daily_path = run_dir / "daily.parquet"
        if not (metrics_path.exists() and metadata_path.exists() and daily_path.exists()):
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        derived_path = run_dir / DERIVED_VALIDATION_FILENAME
        if derived_path.exists():
            derived = json.loads(derived_path.read_text(encoding="utf-8"))
            metrics.update(derived.get("metrics_overrides", {}))
            metadata.update(derived.get("metadata_overrides", {}))
        records.append(
            {
                "run_id": run_dir.name,
                "path": run_dir,
                "metrics": metrics,
                "metadata": metadata,
                "daily_path": daily_path,
                "derived_validation_path": derived_path,
            }
        )
    return records
