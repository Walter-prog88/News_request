"""Tamper-evident forward-signal ledger with legacy-prefix support."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from datetime import date, datetime
from typing import Any

import pandas as pd

SIGNAL_LEDGER_GENESIS_HASH = "0" * 64


class SignalLedgerIntegrityError(ValueError):
    """Raised when a forward-signal chain no longer verifies."""


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _jsonable(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
        allow_nan=False,
    )


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def signal_record_hash(record: Mapping[str, Any]) -> str:
    """Hash one signal record excluding its self-referential hash field."""

    return _sha256_text(
        _canonical_json({key: value for key, value in record.items() if key != "record_hash"})
    )


def parse_signal_ledger(raw: str) -> tuple[list[dict[str, Any]], str]:
    """Parse and verify a ledger, returning records and the chain-tip hash.

    Records written before hash chaining are retained as a legacy prefix.  The
    first chained record commits to the canonical hash of that entire prefix,
    so later modification of any legacy observation is detectable.
    """

    records: list[dict[str, Any]] = []
    legacy: list[dict[str, Any]] = []
    previous_hash = SIGNAL_LEDGER_GENESIS_HASH
    chain_started = False
    for line_number, line in enumerate(raw.splitlines(), start=1):
        if not line.strip():
            raise SignalLedgerIntegrityError(
                f"blank line in signal ledger at line {line_number}"
            )
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SignalLedgerIntegrityError(
                f"malformed JSON in signal ledger at line {line_number}"
            ) from exc
        if not isinstance(record, dict):
            raise SignalLedgerIntegrityError(
                f"signal ledger line {line_number} is not an object"
            )
        is_chained = "record_hash" in record or "previous_hash" in record
        if not is_chained:
            if chain_started:
                raise SignalLedgerIntegrityError(
                    "legacy signal record appears after the hash chain started"
                )
            legacy.append(record)
            records.append(record)
            continue
        if not chain_started:
            chain_started = True
            previous_hash = (
                _sha256_text("\n".join(_canonical_json(item) for item in legacy) + "\n")
                if legacy
                else SIGNAL_LEDGER_GENESIS_HASH
            )
        if record.get("previous_hash") != previous_hash:
            raise SignalLedgerIntegrityError(
                f"previous_hash mismatch at signal record {line_number}"
            )
        expected = signal_record_hash(record)
        if record.get("record_hash") != expected:
            raise SignalLedgerIntegrityError(
                f"record_hash mismatch at signal record {line_number}"
            )
        previous_hash = expected
        records.append(record)
    if not chain_started and legacy:
        previous_hash = _sha256_text(
            "\n".join(_canonical_json(item) for item in legacy) + "\n"
        )
    return records, previous_hash


def chained_signal_record(
    payload: Mapping[str, Any], *, previous_hash: str
) -> dict[str, Any]:
    """Attach the previous pointer and deterministic self hash."""

    record = {**payload, "previous_hash": previous_hash}
    record["record_hash"] = signal_record_hash(record)
    return record


def prospective_signal_assessment(
    *,
    source_date: date,
    as_of: date,
    calendar_name: str | None,
    artifact_written_at: str | datetime | None,
    snapshot_retrieved_at: str | datetime | None,
    logged_at: datetime,
    target_source_column: str | None,
    quick_mode: bool | None,
    feature_set_degraded: bool,
    cash_rate_status: str | None,
    registered_specification: bool,
) -> dict[str, Any]:
    """Assess whether a logged close signal predates its next executable open.

    This uses the asset exchange calendar instead of the host computer's date,
    which is essential when a US close is logged from Asia on the following
    local calendar day.  A record can still be stored when ineligible; reasons
    are explicit and immutable.
    """

    reasons: list[str] = []
    planned_execution_session: str | None = None
    signal_session_close: str | None = None
    planned_execution_open: str | None = None
    permitted_sources = {"signal_position", "target_position", "signal"}
    if source_date != as_of:
        reasons.append("source_date_differs_from_requested_as_of")
    if target_source_column not in permitted_sources:
        reasons.append("not_a_close_generated_target_column")
    if quick_mode is not False:
        reasons.append("quick_or_unclassified_model")
    if feature_set_degraded:
        reasons.append("feature_set_degraded")
    snapshot_cash_statuses = {
        "snapshot_available_at",
        "snapshot_observation_date_lagged",
        "snapshot_index_lagged",
    }
    if cash_rate_status not in snapshot_cash_statuses:
        reasons.append("cash_rate_not_snapshot_backed")
    if not registered_specification:
        reasons.append("missing_registered_experiment_specification")
    if not calendar_name:
        reasons.append("missing_exchange_calendar")
    else:
        try:
            import exchange_calendars as xcals

            calendar = xcals.get_calendar(calendar_name)
            session = pd.Timestamp(source_date)
            if not calendar.is_session(session):
                reasons.append("source_date_is_not_exchange_session")
            else:
                close_at = pd.Timestamp(calendar.session_close(session)).tz_convert("UTC")
                next_session = calendar.next_session(session)
                next_open = pd.Timestamp(calendar.session_open(next_session)).tz_convert("UTC")
                signal_session_close = close_at.isoformat()
                planned_execution_session = pd.Timestamp(next_session).date().isoformat()
                planned_execution_open = next_open.isoformat()

                def utc_timestamp(raw: str | datetime | None) -> pd.Timestamp | None:
                    if raw is None:
                        return None
                    timestamp = pd.Timestamp(raw)
                    if timestamp.tzinfo is None:
                        timestamp = timestamp.tz_localize("UTC")
                    return timestamp.tz_convert("UTC")

                artifact_at = utc_timestamp(artifact_written_at)
                snapshot_at = utc_timestamp(snapshot_retrieved_at)
                logged_timestamp = utc_timestamp(logged_at)
                if artifact_at is None or artifact_at < close_at:
                    reasons.append("artifact_not_created_after_signal_close")
                if snapshot_at is None or snapshot_at < close_at:
                    reasons.append("market_snapshot_not_retrieved_after_signal_close")
                if logged_timestamp is None or logged_timestamp >= next_open:
                    reasons.append("logged_at_or_after_next_execution_open")
                if artifact_at is not None and logged_timestamp is not None:
                    if logged_timestamp < artifact_at:
                        reasons.append("log_timestamp_precedes_artifact")
        except (KeyError, TypeError, ValueError) as exc:
            reasons.append(f"calendar_assessment_failed:{type(exc).__name__}")
    return {
        "eligible": not reasons,
        "reasons": reasons,
        "signal_session_close": signal_session_close,
        "planned_execution_session": planned_execution_session,
        "planned_execution_open": planned_execution_open,
    }


__all__ = [
    "SIGNAL_LEDGER_GENESIS_HASH",
    "SignalLedgerIntegrityError",
    "chained_signal_record",
    "parse_signal_ledger",
    "prospective_signal_assessment",
    "signal_record_hash",
]
