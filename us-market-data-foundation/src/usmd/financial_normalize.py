"""Version-preserving SEC Company Facts normalization.

This module never fetches data or writes to the lake.  A fact is a disclosure
version, not a unique economic period: comparison figures and amendments are
retained. ``fy`` and ``fp`` describe the filing and must not identify the period
of an individual comparison fact. Period classification uses actual dates.

The conservative availability estimate is midnight in New York after ``filed``.
SEC acceptance timestamps are retained separately: acceptance is not evidence
of the first moment that content became publicly available. None of these rows
claims exact intraday point-in-time eligibility.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime, time, timedelta, timezone
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd


CONFIG_PATH = Path(__file__).resolve().parents[2] / "config/financial_metrics.json"
NORMALIZER_VERSION = "1.2"
NEW_YORK = ZoneInfo("America/New_York")
PERIODIC_FORMS = {"10-K", "10-K/A", "10-Q", "10-Q/A", "20-F", "20-F/A", "40-F", "40-F/A"}
ANNUAL_FORMS = {"10-K", "10-K/A", "20-F", "20-F/A", "40-F", "40-F/A"}

FACT_COLUMNS = [
    "fact_id", "issuer_id", "cik", "symbol", "taxonomy", "tag", "unit", "value",
    "value_raw_json", "start", "end", "duration_days", "fy", "fp", "form", "accn",
    "filed", "frame", "acceptance_raw", "acceptance_at", "available_at",
    "availability_basis", "pit_eligible", "source_id", "retrieved_at", "row_status",
]
VALUE_COLUMNS = [
    "value_id", "issuer_id", "cik", "symbol", "metric_id", "statement", "unit",
    "value", "period_start", "period_end", "duration_days", "period_kind",
    "fiscal_year", "fiscal_period", "form", "accn", "filed", "acceptance_raw",
    "acceptance_at", "available_at", "availability_basis", "pit_eligible",
    "source_id", "source_tag", "normalization_method", "derivation_compatibility",
    "input_fact_ids", "input_accessions", "tag_conflict", "conflict_fact_ids",
    "row_status", "retrieved_at", "normalizer_version",
]


@lru_cache(maxsize=1)
def load_metrics():
    """Return the reviewed metric dictionary; never synthesize an absent tag."""
    return json.loads(CONFIG_PATH.read_text())["metrics"]


def _json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)


def _id(prefix, value):
    return prefix + hashlib.sha256(_json(value).encode()).hexdigest()[:24]


def _date(value):
    try:
        return date.fromisoformat(str(value))
    except (ValueError, TypeError):
        return None


def _finite_number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value if math.isfinite(value) else None


def _filings(submissions):
    """Accept full submissions JSON, historical pages, or flattened records."""
    if isinstance(submissions, (list, tuple)):
        for item in submissions:
            yield from _filings(item)
        return
    if not isinstance(submissions, dict):
        return
    if "filings" in submissions:
        yield from _filings(submissions["filings"])
    if "recent" in submissions:
        yield from _filings(submissions["recent"])
    accessions = submissions.get("accessionNumber", submissions.get("accn"))
    if isinstance(accessions, list):
        for i, accession in enumerate(accessions):
            row = {key: values[i] for key, values in submissions.items()
                   if isinstance(values, list) and len(values) > i}
            row["accessionNumber"] = accession
            yield row
    elif accessions:
        yield submissions


def _acceptance_index(submissions):
    result = {}
    for filing in _filings(submissions):
        accession = filing.get("accessionNumber", filing.get("accn"))
        raw = filing.get("acceptanceDateTime", filing.get("acceptance_datetime"))
        stamp = None
        if raw:
            try:
                parsed = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
                # Do not guess a timezone for undocumented naive timestamps.
                if parsed.tzinfo is not None:
                    stamp = parsed.astimezone(timezone.utc).isoformat()
            except ValueError:
                pass
        previous = result.get(accession)
        if previous is None or (stamp and not previous[1]):
            result[accession] = (raw, stamp)
    return result


def _availability(filed):
    filed_day = _date(filed)
    if filed_day is None:
        return None, "unavailable_missing_filed_date"
    conservative = datetime.combine(filed_day + timedelta(days=1), time.min, NEW_YORK)
    return conservative.astimezone(timezone.utc).isoformat(), "estimated_filed_next_day"


def _period_kind(start, end, form=None):
    end_day = _date(end)
    if end_day is None:
        return "invalid", None
    if start is None:
        return "instant", None
    start_day = _date(start)
    if start_day is None or end_day < start_day:
        return "invalid", None
    duration = (end_day - start_day).days + 1
    # Retail calendars can have 12/12/12/16 (or 17) week quarters.
    if 70 <= duration <= 120:
        return "direct_quarter", duration
    if 150 <= duration <= 210:
        return "cumulative_6m", duration
    if 240 <= duration <= 300:
        return "cumulative_9m", duration
    if 350 <= duration <= 380:
        # Amazon and others report rolling 12-month supplemental cash flows in
        # 10-Q filings. A 365-day duration alone does not establish a fiscal year.
        return ("annual" if form in ANNUAL_FORMS else "reported_12m"), duration
    return "reported_other", duration


def _base_facts(companyfacts, submissions, issuer, retrieved_at, since):
    cik = str(issuer.get("cik", companyfacts.get("cik", ""))).zfill(10)
    actual_cik = str(companyfacts.get("cik", "")).zfill(10)
    if companyfacts.get("cik") is not None and actual_cik != cik:
        raise ValueError(f"Company Facts CIK {actual_cik} does not match issuer {cik}")
    cutoff = _date(since)
    if cutoff is None:
        raise ValueError("since must be an ISO date")
    issuer_id = issuer.get("issuer_id", f"SEC:{cik}")
    symbol = issuer.get("symbol", issuer.get("primary_symbol"))
    source_id = issuer.get("companyfacts_source_id", issuer.get("source_id", f"sec_companyfacts:{cik}"))
    acceptance = _acceptance_index(submissions)
    result = {}
    for taxonomy, tags in companyfacts.get("facts", {}).items():
        for tag, concept in tags.items():
            for unit, facts in concept.get("units", {}).items():
                for fact in facts:
                    end = fact.get("end")
                    if _date(end) is not None and _date(end) < cutoff:
                        continue
                    start = fact.get("start")
                    kind, duration = _period_kind(start, end, fact.get("form"))
                    value = _finite_number(fact.get("val"))
                    available_at, basis = _availability(fact.get("filed"))
                    problems = []
                    if value is None:
                        problems.append("invalid_value")
                    if kind == "invalid":
                        problems.append("invalid_period")
                    if available_at is None:
                        problems.append("missing_filed_date")
                    if not fact.get("accn"):
                        problems.append("missing_accession")
                    acceptance_raw, acceptance_at = acceptance.get(fact.get("accn"), (None, None))
                    identity = {"cik": cik, "taxonomy": taxonomy, "tag": tag, "unit": unit, "fact": fact}
                    fact_id = _id("fact_", identity)
                    result[fact_id] = {
                        "fact_id": fact_id, "issuer_id": issuer_id, "cik": cik, "symbol": symbol,
                        "taxonomy": taxonomy, "tag": tag, "unit": unit, "value": value,
                        "value_raw_json": _json(fact.get("val")), "start": start, "end": end,
                        "duration_days": duration, "fy": fact.get("fy"), "fp": fact.get("fp"),
                        "form": fact.get("form"), "accn": fact.get("accn"), "filed": fact.get("filed"),
                        "frame": fact.get("frame"), "acceptance_raw": acceptance_raw,
                        "acceptance_at": acceptance_at, "available_at": available_at,
                        "availability_basis": basis, "pit_eligible": False, "source_id": source_id,
                        "retrieved_at": retrieved_at, "row_status": "|".join(problems) or "ok",
                    }
    return list(result.values())


def _normal_values(facts, metrics, retrieved_at, issuer):
    mapping = defaultdict(list)
    for metric in metrics:
        tags = metric.get("tag_overrides_by_profile", {}).get(issuer.get("profile"), metric["tags"])
        tags = metric.get("tag_overrides_by_cik", {}).get(str(issuer.get("cik", "")).zfill(10), tags)
        for rank, tag in enumerate(tags):
            mapping[(metric.get("taxonomy", "us-gaap"), tag)].append((metric, rank))
        # A differently scoped tag is never a global fallback. Only an exact
        # disclosure checked against its original consolidated statement may
        # enter the reviewed metric. Preserve the real tag in source_tag.
        equivalent_tags = {rule["tag"] for rule in metric.get("verified_fact_equivalences", [])
            if rule.get("verified") is True and rule.get("official_sources")
            and str(rule.get("cik", "")).zfill(10) == str(issuer.get("cik", "")).zfill(10)}
        for tag in sorted(equivalent_tags - set(tags)):
            mapping[(metric.get("taxonomy", "us-gaap"), tag)].append((metric, len(tags)))
    groups = defaultdict(list)
    for fact in facts:
        if fact["form"] not in PERIODIC_FORMS or fact["row_status"] != "ok":
            continue
        for metric, rank in mapping.get((fact["taxonomy"], fact["tag"]), []):
            direct_tags = metric.get("tag_overrides_by_profile", {}).get(issuer.get("profile"), metric["tags"])
            direct_tags = metric.get("tag_overrides_by_cik", {}).get(str(issuer.get("cik", "")).zfill(10), direct_tags)
            if fact["tag"] not in direct_tags:
                matched = any(rule.get("verified") is True and rule.get("official_sources")
                    and str(rule.get("cik", "")).zfill(10) == fact["cik"]
                    and rule.get("tag") == fact["tag"] and rule.get("accn") == fact["accn"]
                    and rule.get("unit") == fact["unit"]
                    and any(item.get("start") == fact["start"] and item.get("end") == fact["end"]
                            and item.get("value") == fact["value"] for item in rule.get("facts", []))
                    for rule in metric.get("verified_fact_equivalences", []))
                if not matched:
                    continue
            # Different currencies and units remain raw facts, never mislabeled USD.
            if fact["unit"] not in metric["units"]:
                continue
            kind, _ = _period_kind(fact["start"], fact["end"], fact["form"])
            if (metric["period_type"] == "instant") != (kind == "instant"):
                continue
            key = (metric["metric_id"], fact["accn"], fact["start"], fact["end"], fact["unit"])
            groups[key].append((metric, rank, fact))
    result = []
    for candidates in groups.values():
        # Precedence is explicit in config. Preserve all competing values in facts.
        candidates.sort(key=lambda item: (item[1], item[2]["fact_id"]))
        metric, rank, fact = candidates[0]
        kind, duration = _period_kind(fact["start"], fact["end"], fact["form"])
        competing = {float(item[2]["value"]) for item in candidates}
        conflict = len(competing) > 1
        same_tag_values = {float(item[2]["value"]) for item in candidates if item[1] == rank}
        ambiguous = len(same_tag_values) > 1
        row = {key: fact[key] for key in (
            "issuer_id", "cik", "symbol", "unit", "value", "form", "accn", "filed",
            "acceptance_raw", "acceptance_at", "available_at", "availability_basis", "pit_eligible", "source_id",
        )}
        row.update({
            "value_id": _id("value_", [metric["metric_id"], fact["fact_id"]]),
            "metric_id": metric["metric_id"], "statement": metric["statement"],
            "period_start": fact["start"], "period_end": fact["end"], "duration_days": duration,
            "period_kind": kind, "fiscal_year": fact["fy"], "fiscal_period": fact["fp"],
            "source_tag": fact["taxonomy"] + ":" + fact["tag"], "normalization_method": "reported",
            "derivation_compatibility": "not_applicable", "input_fact_ids": _json([fact["fact_id"]]),
            "input_accessions": _json([fact["accn"]]), "tag_conflict": conflict,
            "conflict_fact_ids": _json(sorted({item[2]["fact_id"] for item in candidates})) if conflict else "[]",
            "row_status": "ambiguous_same_tag_value" if ambiguous else ("tag_conflict" if conflict else "ok"),
            "retrieved_at": retrieved_at, "normalizer_version": NORMALIZER_VERSION,
        })
        result.append(row)
    return result


def _latest_input(rows, event):
    from usmd.financial_quality import QUARANTINE_FLAGS
    eligible = [row for row in rows if row["available_at"] <= event]
    if not eligible:
        return None
    latest_at = max(row["available_at"] for row in eligible)
    latest = [row for row in eligible if row["available_at"] == latest_at]
    chosen = max(latest, key=lambda row: (row["filed"], row["accn"], row["value_id"]))
    # A conflicting latest input invalidates a previous clean subtraction. It
    # must emit a flagged derived vintage, not disappear and leave stale output.
    # Likewise, inconsistent same-day accessions cannot be ordered by assumption.
    if len({row["value"] for row in latest}) > 1 or any(row["row_status"] != "ok" for row in latest):
        chosen = chosen.copy()
        facts = {fid for row in latest for fid in json.loads(row["input_fact_ids"])}
        facts.update(fid for row in latest for fid in json.loads(row["conflict_fact_ids"]))
        quarantine = sorted({flag for row in latest for flag in row['row_status'].split('|') if flag in QUARANTINE_FLAGS})
        conflict = len({row['value'] for row in latest}) > 1 or any(row.get('tag_conflict') or row['row_status'] in {'tag_conflict','ambiguous_same_tag_value'} for row in latest)
        status = '|'.join((['tag_conflict'] if conflict else []) + quarantine) or 'ok'
        chosen.update({"row_status": status, "tag_conflict": conflict,
                       "input_fact_ids": _json(sorted(facts)), "conflict_fact_ids": _json(sorted(facts)),
                       "input_accessions": _json(sorted({accn for row in latest for accn in json.loads(row["input_accessions"])}))})
    return chosen


def _derive_quarters(values, metrics, retrieved_at):
    """Derive only additive flows with matching fiscal start, tag and unit.

    Recompute at every input vintage change; a later restatement of the earlier
    cumulative period must not leak into an earlier as-of result. Cross-filing
    differences are marked unverified because an equal tag cannot prove that
    both disclosures share the same restatement basis. Conflicted inputs emit
    flagged derived vintages to invalidate old clean results; their arithmetic
    is only a diagnostic candidate, never a default-query financial value.
    """
    from usmd.financial_quality import QUARANTINE_FLAGS, quarantined_status
    additive = {item["metric_id"] for item in metrics if item.get("additive")}
    groups = defaultdict(list)
    for row in values:
        if row["metric_id"] in additive and (row["row_status"] in {"ok", "tag_conflict", "ambiguous_same_tag_value"} or quarantined_status(row['row_status'])):
            groups[(row["metric_id"], row["source_tag"], row["unit"], row["period_start"])].append(row)
    preceding_kind = {"cumulative_6m": "direct_quarter", "cumulative_9m": "cumulative_6m", "annual": "cumulative_9m"}
    result = {}
    for group in groups.values():
        targets = defaultdict(list)
        for row in group:
            if row["period_kind"] in preceding_kind:
                targets[(row["period_kind"], row["period_end"])].append(row)
        for (kind, target_end), current_versions in targets.items():
            target_date = _date(target_end)
            previous_versions = [row for row in group
                                 if row["period_kind"] == preceding_kind[kind]
                                 and 70 <= (target_date - _date(row["period_end"])).days <= 120]
            if not previous_versions or len({row["period_end"] for row in previous_versions}) != 1:
                continue
            events = sorted({row["available_at"] for row in current_versions + previous_versions})
            for event in events:
                current = _latest_input(current_versions, event)
                previous = _latest_input(previous_versions, event)
                if current is None or previous is None:
                    continue
                facts = sorted(set(json.loads(current["input_fact_ids"]) + json.loads(previous["input_fact_ids"])))
                accessions = sorted(set(json.loads(current["input_accessions"]) + json.loads(previous["input_accessions"])))
                input_conflict = bool(current['tag_conflict'] or previous['tag_conflict'])
                quarantine = sorted({flag for source in (current,previous) for flag in source['row_status'].split('|') if flag in QUARANTINE_FLAGS})
                derived_status = '|'.join((['tag_conflict'] if input_conflict else [])+quarantine) or 'ok'
                value_id = _id("value_", ["cumulative_difference", current["metric_id"], facts])
                if value_id in result:
                    continue
                row = current.copy()
                period_start = (_date(previous["period_end"]) + timedelta(days=1)).isoformat()
                row.update({
                    "value_id": value_id, "value": current["value"] - previous["value"],
                    "period_start": period_start, "duration_days": (target_date - _date(period_start)).days + 1,
                    "period_kind": "derived_quarter", "available_at": max(current["available_at"], previous["available_at"]),
                    "availability_basis": "estimated_filed_next_day_max_inputs", "pit_eligible": False,
                    "normalization_method": "cumulative_difference",
                    "derivation_compatibility": "same_filing" if len(accessions) == 1 else "cross_filing_unverified",
                    "input_fact_ids": _json(facts), "input_accessions": _json(accessions),
                    "tag_conflict": input_conflict, "row_status": derived_status,
                    "conflict_fact_ids": _json(facts) if input_conflict else "[]",
                    "retrieved_at": retrieved_at,
                })
                result[value_id] = row
    return list(result.values())


def normalize_company(companyfacts: dict, submissions: list[dict], issuer: dict,
                      retrieved_at: str, since: str = "2015-01-01") -> dict[str, pd.DataFrame]:
    """Build raw and standardized tables without discarding filing vintages.

    ``issuer`` requires a CIK, with optional ``issuer_id``, ``symbol`` and
    ``companyfacts_source_id`` (the exact raw snapshot provenance ID). The SEC
    dictionary must match that CIK. Unsupported tags/units and malformed facts
    remain inspectable in ``financial_facts``; they do not become valid metrics.
    """
    if not isinstance(companyfacts, dict) or not isinstance(companyfacts.get("facts"), dict):
        raise ValueError("Expected a SEC Company Facts object with facts")
    metrics = load_metrics()
    facts = _base_facts(companyfacts, submissions, issuer, retrieved_at, since)
    # Apply actual-value gates before subtraction: invalid reported inputs must
    # never create a clean derived quarter. Raw prospective facts stay intact.
    from usmd.financial_quality import quarantine_values
    values = quarantine_values(pd.DataFrame(_normal_values(facts, metrics, retrieved_at, issuer),
                                            columns=VALUE_COLUMNS)).to_dict("records")
    values.extend(_derive_quarters(values, metrics, retrieved_at))
    fact_frame = pd.DataFrame(facts, columns=FACT_COLUMNS)
    value_frame = pd.DataFrame(values, columns=VALUE_COLUMNS)
    value_frame = quarantine_values(value_frame)
    if not fact_frame.empty:
        fact_frame = fact_frame.sort_values(["end", "tag", "filed", "fact_id"], na_position="last").reset_index(drop=True)
    if not value_frame.empty:
        value_frame = value_frame.sort_values(["period_end", "metric_id", "available_at", "value_id"], na_position="last").reset_index(drop=True)
    return {"financial_facts": fact_frame, "financial_values": value_frame}
