"""Semantic gates for actual financial values; source timestamps stay unchanged.

SEC acceptance is not a first-public timestamp.  A next-filing-day estimate
which precedes acceptance is an inconsistent estimate, not evidence of an
earlier issuer release. Keep it for review rather than invent a replacement.
Raw Company Facts may contain prospective obligations and are not subject to
the actual-period gate in this module.
"""
from __future__ import annotations

import pandas as pd


ACTUAL_PERIOD_FLAG = "period_end_after_filing_date"
AVAILABILITY_FLAG = "availability_precedes_acceptance"
QUARANTINE_FLAGS = {ACTUAL_PERIOD_FLAG, AVAILABILITY_FLAG}


def violations(frame):
    """Return reproducible masks without changing values, dates or provenance."""
    empty = pd.Series(False, index=frame.index, dtype=bool)
    period = empty.copy()
    if {"period_end", "filed"} <= set(frame.columns):
        end = pd.to_datetime(frame.period_end, errors="coerce", format="ISO8601")
        filed = pd.to_datetime(frame.filed, errors="coerce", format="ISO8601")
        period = end.notna() & filed.notna() & end.gt(filed)
    availability = empty.copy()
    if {"available_at", "acceptance_at", "availability_basis"} <= set(frame.columns):
        available = pd.to_datetime(frame.available_at, utc=True, errors="coerce", format="ISO8601")
        accepted = pd.to_datetime(frame.acceptance_at, utc=True, errors="coerce", format="ISO8601")
        estimate = frame.availability_basis.fillna("").str.startswith("estimated_filed_next_day")
        availability = estimate & available.notna() & accepted.notna() & available.lt(accepted)
    return {ACTUAL_PERIOD_FLAG: period, AVAILABILITY_FLAG: availability}


def quarantine_values(frame):
    """Append quality flags while preserving disclosure versions and timestamps."""
    result = frame.copy()
    if result.empty:
        return result
    if "row_status" not in result:
        raise ValueError("Financial values require row_status")
    for flag, mask in violations(result).items():
        if not mask.any():
            continue
        def append(status):
            parts = [] if status == "ok" else str(status).split("|")
            if flag not in parts:
                parts.append(flag)
            return "|".join(parts)
        result.loc[mask, "row_status"] = result.loc[mask, "row_status"].map(append)
    return result


def quarantined_status(status):
    return bool(set(str(status).split("|")) & QUARANTINE_FLAGS)
