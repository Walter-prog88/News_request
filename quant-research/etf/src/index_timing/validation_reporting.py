"""Cross-run multiple-testing correction and model-selection diagnostics."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .artifacts import (
    DERIVED_VALIDATION_FILENAME,
    discover_runs,
    read_trial_ledger,
)
from .statistics import benjamini_hochberg, deflated_sharpe_ratio, spa_test
from .validation import ValidationEvidence, classify_validation


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def update_cross_run_validation(*, fdr_q: float = 0.10, spa_reps: int = 1_000) -> pd.DataFrame:
    """Correct all saved independent-validation trials as one research family."""

    runs = [
        record
        for record in discover_runs()
        if record["metadata"].get("replication_level") == "independent_validation"
        and record["metrics"].get("bootstrap_pvalue") is not None
    ]
    ledger = read_trial_ledger()
    if not runs and not ledger:
        return pd.DataFrame()
    pvalues = {
        record["run_id"]: float(record["metrics"]["bootstrap_pvalue"])
        for record in runs
    }
    ledger_trial_keys: dict[str, str] = {}
    for trial in ledger:
        if trial["status"] == "success" and trial["artifact_run_id"]:
            key = str(trial["artifact_run_id"])
        else:
            key = (
                f"ledger__{trial['status']}__{trial['trial_spec_hash'][:12]}"
                f"__seq{trial['sequence']}"
            )
        # Identical successful artifacts already discovered above retain the
        # same key.  Failed/skipped attempts enter conservatively with p=1.
        pvalues.setdefault(key, float(trial["pvalue"]))
        ledger_trial_keys[str(trial["trial_spec_hash"])] = key
    correction = benjamini_hochberg(pvalues, q=fdr_q)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in runs:
        groups[str(record["metadata"].get("asset_id", "unknown"))].append(record)

    spa_by_asset: dict[str, dict[str, Any]] = {}
    sharpes_by_asset: dict[str, list[float]] = {}
    trial_ids_by_asset: dict[str, set[str]] = defaultdict(set)
    for record in runs:
        trial_ids_by_asset[str(record["metadata"].get("asset_id", "unknown"))].add(
            record["run_id"]
        )
    for trial in ledger:
        asset_id = str(trial["trial_spec"].get("asset_id", "unknown"))
        trial_ids_by_asset[asset_id].add(
            ledger_trial_keys[str(trial["trial_spec_hash"])]
        )
    for asset_id, records in groups.items():
        trial_sharpes = [
            float(record["metrics"]["sharpe"])
            for record in records
            if np.isfinite(record["metrics"].get("sharpe", np.nan))
        ]
        sharpes_by_asset[asset_id] = trial_sharpes
        active_parts: list[pd.Series] = []
        for record in records:
            daily = pd.read_parquet(record["daily_path"])
            if "net_return" not in daily or "equal_exposure_return" not in daily:
                continue
            active_parts.append(
                (daily["net_return"] - daily["equal_exposure_return"]).rename(
                    record["run_id"]
                )
            )
        if not active_parts:
            spa_by_asset[asset_id] = {"available": False, "reason": "missing aligned returns"}
            continue
        comparison = pd.concat(active_parts, axis=1).dropna()
        zero_benchmark = pd.Series(0.0, index=comparison.index, name="zero_active_return")
        spa_by_asset[asset_id] = spa_test(
            zero_benchmark,
            comparison,
            reps=spa_reps,
            block_size=20,
            seed=0,
        )

    for record in runs:
        run_id = record["run_id"]
        metadata = record["metadata"]
        asset_id = str(metadata.get("asset_id", "unknown"))
        daily = pd.read_parquet(record["daily_path"])
        risk_free = daily["cash_return"] if "cash_return" in daily else 0.0
        dsr = deflated_sharpe_ratio(
            daily["net_return"] - risk_free,
            n_trials=max(1, len(trial_ids_by_asset[asset_id])),
            trial_sharpes=sharpes_by_asset[asset_id],
        )
        metrics_overrides: dict[str, Any] = {
            "deflated_sharpe_probability": dsr["dsr"],
            "expected_max_trial_sharpe": dsr["expected_max_sharpe"],
        }
        metadata_overrides: dict[str, Any] = {}
        metadata_overrides["global_bh_fdr"] = {
            "research_family_trials": len(pvalues),
            "successful_artifacts": len(runs),
            "ledger_failed_or_skipped": sum(
                trial["status"] != "success" for trial in ledger
            ),
            "q": fdr_q,
            "raw_pvalue": float(correction.loc[run_id, "p_value"]),
            "adjusted_pvalue": float(correction.loc[run_id, "adjusted_p_value"]),
            "reject": bool(correction.loc[run_id, "reject"]),
        }
        metadata_overrides["deflated_sharpe"] = dsr
        metadata_overrides["spa_reality_check"] = spa_by_asset[asset_id]
        evidence_payload = metadata.get("validation_evidence")
        if evidence_payload:
            evidence_payload = dict(evidence_payload)
            evidence_payload["fdr_reject"] = bool(correction.loc[run_id, "reject"])
            evidence = ValidationEvidence(**evidence_payload)
            decision = classify_validation(
                evidence,
                formal_eligible=bool(metadata.get("formal_evidence_eligible", False)),
            )
            metrics_overrides["status"] = decision.status.value
            metadata_overrides["validation_evidence"] = evidence_payload
            metadata_overrides["validation_checks"] = decision.checks
            metadata_overrides["validation_summary"] = decision.summary
        _write_json(
            record["path"] / DERIVED_VALIDATION_FILENAME,
            {
                "metrics_overrides": metrics_overrides,
                "metadata_overrides": metadata_overrides,
            },
        )
    return correction


__all__ = ["update_cross_run_validation"]
