from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from index_timing.artifacts import (
    TRIAL_LEDGER_GENESIS_HASH,
    TrialLedgerIntegrityError,
    append_trial_ledger_record,
    read_trial_ledger,
    trial_spec_sha256,
    verify_trial_ledger,
)


def test_trial_ledger_appends_and_verifies_hash_chain(tmp_path) -> None:
    path = tmp_path / "trial_ledger.jsonl"
    first_spec = {"asset": "SPY", "model": {"kind": "mlp", "seed": 7}}
    first = append_trial_ledger_record(
        status="success",
        trial_spec=first_spec,
        pvalue=0.04,
        artifact_run_id="validation__caitong__sp500",
        metadata={"note": "outer walk-forward"},
        ledger_path=path,
        recorded_at=datetime(2026, 7, 13, tzinfo=UTC),
    )
    failed = append_trial_ledger_record(
        status="failed",
        trial_spec={"asset": "QQQ", "model": "lstm"},
        pvalue=0.01,
        artifact_run_id=None,
        metadata={"error": "training diverged"},
        ledger_path=path,
        recorded_at=datetime(2026, 7, 14, tzinfo=UTC),
    )
    skipped = append_trial_ledger_record(
        status="skipped",
        trial_spec={"asset": "STAR50", "reason": "short history"},
        ledger_path=path,
        recorded_at=datetime(2026, 7, 15, tzinfo=UTC),
    )

    records = read_trial_ledger(path)
    assert records == [first, failed, skipped]
    assert verify_trial_ledger(path)
    assert first["previous_hash"] == TRIAL_LEDGER_GENESIS_HASH
    assert failed["previous_hash"] == first["record_hash"]
    assert skipped["previous_hash"] == failed["record_hash"]
    assert failed["pvalue"] == 1.0
    assert skipped["pvalue"] == 1.0
    assert first["trial_spec_hash"] == trial_spec_sha256(first_spec)
    assert trial_spec_sha256({"b": 2, "a": 1}) == trial_spec_sha256({"a": 1, "b": 2})


def test_trial_ledger_detects_tampering_and_refuses_next_append(tmp_path) -> None:
    path = tmp_path / "trial_ledger.jsonl"
    append_trial_ledger_record(
        status="success",
        trial_spec={"asset": "SPY", "seed": 1},
        pvalue=0.05,
        artifact_run_id="run-1",
        ledger_path=path,
    )
    append_trial_ledger_record(
        status="failed",
        trial_spec={"asset": "SPY", "seed": 2},
        ledger_path=path,
    )
    lines = path.read_text(encoding="utf-8").splitlines()
    tampered = json.loads(lines[0])
    tampered["pvalue"] = 0.5
    lines[0] = json.dumps(tampered, ensure_ascii=False, sort_keys=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(TrialLedgerIntegrityError, match="record_hash mismatch"):
        read_trial_ledger(path)
    with pytest.raises(TrialLedgerIntegrityError, match="record_hash mismatch"):
        append_trial_ledger_record(
            status="skipped",
            trial_spec={"asset": "QQQ"},
            ledger_path=path,
        )


def test_successful_trial_requires_pvalue_and_artifact_id(tmp_path) -> None:
    path = tmp_path / "trial_ledger.jsonl"
    with pytest.raises(ValueError, match="pvalue"):
        append_trial_ledger_record(
            status="success",
            trial_spec={"asset": "SPY"},
            artifact_run_id="run",
            ledger_path=path,
        )
    with pytest.raises(ValueError, match="artifact_run_id"):
        append_trial_ledger_record(
            status="success",
            trial_spec={"asset": "SPY"},
            pvalue=0.1,
            ledger_path=path,
        )
