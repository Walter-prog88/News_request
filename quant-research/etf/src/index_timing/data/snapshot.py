"""Immutable Parquet snapshots with a machine-auditable SHA256 manifest."""

from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd

from ..types import DatasetSnapshot


def _safe_component(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return cleaned or "dataset"


def file_sha256(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _date_bounds(frame: pd.DataFrame) -> tuple[date | None, date | None]:
    if frame.empty:
        return None, None
    if isinstance(frame.index, pd.DatetimeIndex):
        dates = frame.index
    else:
        candidate = next(
            (column for column in ("date", "observation_date", "available_at") if column in frame),
            None,
        )
        if candidate is None:
            return None, None
        dates = pd.DatetimeIndex(pd.to_datetime(frame[candidate], errors="coerce").dropna())
    if dates.empty:
        return None, None
    return dates.min().date(), dates.max().date()


def write_snapshot(
    frame: pd.DataFrame,
    *,
    raw_dir: str | Path,
    provider: str,
    symbol: str,
    dataset_id: str | None = None,
    retrieved_at: datetime | None = None,
    as_of: date | None = None,
    vintage: str | None = None,
    is_proxy: bool = False,
    warnings: list[str] | None = None,
) -> DatasetSnapshot:
    """Write a content-hashed Parquet snapshot and append its manifest record."""

    now = retrieved_at or datetime.now(UTC)
    if now.tzinfo is None:
        now = now.replace(tzinfo=UTC)
    root = Path(raw_dir).expanduser()
    provider_dir = root / _safe_component(provider) / _safe_component(symbol)
    provider_dir.mkdir(parents=True, exist_ok=True)
    dataset_id = dataset_id or f"{provider}:{symbol}:{now:%Y%m%dT%H%M%SZ}"
    stem = f"{now:%Y%m%dT%H%M%S%fZ}_{_safe_component(symbol)}"
    final_path = provider_dir / f"{stem}.parquet"
    temporary_path = final_path.with_suffix(".parquet.tmp")
    frame.to_parquet(temporary_path, index=True)
    os.replace(temporary_path, final_path)
    sha256 = file_sha256(final_path)
    minimum, maximum = _date_bounds(frame)
    snapshot = DatasetSnapshot(
        dataset_id=dataset_id,
        provider=provider,
        symbol=symbol,
        retrieved_at=now,
        as_of=as_of or now.date(),
        path=final_path.resolve(),
        sha256=sha256,
        row_count=len(frame),
        min_date=minimum,
        max_date=maximum,
        vintage=vintage,
        is_proxy=is_proxy,
        warnings=warnings or [],
    )
    manifest_path = root / "manifest.jsonl"
    with manifest_path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(snapshot.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)
        )
        handle.write("\n")
    return snapshot


def read_manifest(raw_dir: str | Path, *, verify_hashes: bool = False) -> list[DatasetSnapshot]:
    manifest_path = Path(raw_dir).expanduser() / "manifest.jsonl"
    if not manifest_path.exists():
        return []
    snapshots: list[DatasetSnapshot] = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                snapshot = DatasetSnapshot.model_validate(json.loads(line))
            except Exception as exc:
                raise ValueError(f"Invalid snapshot manifest record at line {line_number}") from exc
            if verify_hashes:
                if not snapshot.path.exists():
                    raise FileNotFoundError(snapshot.path)
                if file_sha256(snapshot.path) != snapshot.sha256:
                    raise ValueError(f"Snapshot checksum mismatch: {snapshot.path}")
            snapshots.append(snapshot)
    return snapshots


def load_snapshot(snapshot: DatasetSnapshot, *, verify_hash: bool = True) -> pd.DataFrame:
    if verify_hash and file_sha256(snapshot.path) != snapshot.sha256:
        raise ValueError(f"Snapshot checksum mismatch: {snapshot.path}")
    return pd.read_parquet(snapshot.path)
