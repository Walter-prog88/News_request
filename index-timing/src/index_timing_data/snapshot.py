from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .client import TwoShareClient
from .models import ApiTable, QuerySpec


class SnapshotError(RuntimeError):
    """Snapshot publication or verification failure."""


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_bytes(value)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_csv(path: Path, table: ApiTable) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(table.fields)
            writer.writerows(table.items)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@dataclass(slots=True)
class SnapshotWriter:
    client: TwoShareClient

    def capture(self, specs: Iterable[QuerySpec], output_dir: str | Path) -> Path:
        destination = Path(output_dir).resolve()
        if destination.exists():
            raise SnapshotError(f"snapshot already exists: {destination}")
        if not destination.parent.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)

        staging = destination.with_name(f".{destination.name}.tmp-{uuid.uuid4().hex}")
        staging.mkdir()
        created_at = datetime.now(UTC).isoformat()
        entries: list[dict[str, Any]] = []

        try:
            for spec in specs:
                table = self.client.query(spec)
                raw_relative = Path("raw") / f"{spec.dataset_id}.json"
                csv_relative = Path("normalized") / f"{spec.dataset_id}.csv"
                receipt_relative = Path("receipts") / f"{spec.dataset_id}.json"

                _write_bytes(
                    staging / raw_relative,
                    canonical_json_bytes(table.response_payload()),
                )
                _write_csv(staging / csv_relative, table)

                receipt = {
                    "schema_version": 1,
                    "dataset_id": spec.dataset_id,
                    "api_name": spec.api_name,
                    "base_url": self.client.base_url,
                    "request": spec.public_dict(),
                    "http_status": table.http_status,
                    "response_code": table.response_code,
                    "message": table.message,
                    "received_at_utc": table.received_at_utc,
                    "row_count": table.row_count,
                    "response_fields": list(table.fields),
                    "raw_path": raw_relative.as_posix(),
                    "raw_sha256": sha256_file(staging / raw_relative),
                    "normalized_path": csv_relative.as_posix(),
                    "normalized_sha256": sha256_file(staging / csv_relative),
                    "credential_persisted": False,
                }
                _write_bytes(staging / receipt_relative, canonical_json_bytes(receipt))
                entries.append(
                    {
                        "dataset_id": spec.dataset_id,
                        "api_name": spec.api_name,
                        "row_count": table.row_count,
                        "fields": list(table.fields),
                        "metadata": dict(spec.metadata),
                        "receipt_path": receipt_relative.as_posix(),
                        "receipt_sha256": sha256_file(staging / receipt_relative),
                    }
                )

            manifest = {
                "schema_version": 1,
                "snapshot_id": destination.name,
                "provider": "2share_tushare_compatible_proxy",
                "base_url": self.client.base_url,
                "created_at_utc": created_at,
                "credential_persisted": False,
                "dataset_count": len(entries),
                "datasets": entries,
            }
            _write_bytes(staging / "manifest.json", canonical_json_bytes(manifest))

            members = sorted(
                path
                for path in staging.rglob("*")
                if path.is_file() and path.name != "SHA256SUMS"
            )
            sums = "".join(
                f"{sha256_file(path)}  {path.relative_to(staging).as_posix()}\n"
                for path in members
            )
            _write_bytes(staging / "SHA256SUMS", sums.encode("utf-8"))
            os.replace(staging, destination)
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise

        return destination


def verify_snapshot(snapshot_dir: str | Path) -> dict[str, Any]:
    root = Path(snapshot_dir).resolve()
    sums_path = root / "SHA256SUMS"
    manifest_path = root / "manifest.json"
    if not root.is_dir() or not sums_path.is_file() or not manifest_path.is_file():
        raise SnapshotError(f"not a snapshot bundle: {root}")

    checked = 0
    for line_number, raw_line in enumerate(
        sums_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not raw_line:
            continue
        try:
            expected, relative_text = raw_line.split("  ", 1)
        except ValueError as exc:
            raise SnapshotError(f"invalid SHA256SUMS line {line_number}") from exc
        relative = Path(relative_text)
        if relative.is_absolute() or ".." in relative.parts:
            raise SnapshotError(f"unsafe member path at line {line_number}")
        member = root / relative
        if not member.is_file():
            raise SnapshotError(f"missing snapshot member: {relative_text}")
        actual = sha256_file(member)
        if actual != expected:
            raise SnapshotError(f"SHA-256 mismatch: {relative_text}")
        checked += 1

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("credential_persisted") is not False:
        raise SnapshotError("manifest credential policy drift")
    if manifest.get("dataset_count") != len(manifest.get("datasets") or []):
        raise SnapshotError("manifest dataset count mismatch")
    return {
        "snapshot_id": manifest.get("snapshot_id"),
        "dataset_count": manifest.get("dataset_count"),
        "checked_files": checked,
        "verified": True,
    }
