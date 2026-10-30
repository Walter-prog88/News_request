"""Conservative, filing-local financial tie-outs; findings never alter facts.

Only original reported USD values are eligible. Parent-only equity cannot
establish consolidated balance-sheet equality. The current metric dictionary
does not establish a consistent cash/restricted-cash and FX reconciliation, so
cash-flow checks explicitly remain not_tested.
"""
from __future__ import annotations

import hashlib
import json
import math

import pandas as pd


GROUP_COLUMNS = ["issuer_id", "accn", "unit", "period_start", "period_end", "period_kind"]
CHECK_COLUMNS = [
    "check_id", "issuer_id", "symbol", "accn", "unit", "period_start", "period_end",
    "period_kind", "check", "status", "observed", "expected", "difference",
    "tolerance", "input_value_ids", "input_fact_ids", "detail",
]
BALANCE_METRICS = ("total_assets", "total_liabilities", "equity_including_noncontrolling")
GROSS_METRICS = ("gross_profit", "revenue", "cost_of_revenue")
CASH_METRICS = ("cash_change", "operating_cash_flow", "investing_cash_flow", "financing_cash_flow")
REPORTED_DURATIONS = {"annual", "direct_quarter", "cumulative_6m", "cumulative_9m", "reported_other"}


def _json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _missing(value):
    return value is None or pd.isna(value) or value == ""


def _ids(value):
    try:
        result = json.loads(value) if isinstance(value, str) else value
    except (TypeError, ValueError):
        return []
    return result if isinstance(result, list) and all(isinstance(x, str) and x for x in result) else []


def _row_problem(row):
    if row["normalization_method"] != "reported" or row["period_kind"] == "derived_quarter":
        return "not_original_reported"
    if row["row_status"] != "ok":
        return "row_status_" + str(row["row_status"])
    conflict = row["tag_conflict"]
    if pd.isna(conflict) or conflict not in (False, 0):
        return "tag_conflict_or_unknown"
    try:
        finite = not isinstance(row["value"], bool) and math.isfinite(float(row["value"]))
    except (TypeError, ValueError, OverflowError):
        finite = False
    if not finite:
        return "nonfinite_value"
    if _missing(row["value_id"]) or not _ids(row["input_fact_ids"]):
        return "missing_input_provenance"
    return None


def _inputs(group, metrics):
    """Return unique usable metrics and all relevant candidate provenance."""
    relevant = [row for metric in metrics for row in group.get(metric, [])]
    value_ids = sorted({str(row['value_id']) for row in relevant if not _missing(row['value_id'])})
    fact_ids = sorted({x for row in relevant for x in _ids(row['input_fact_ids'])})
    found, problems = {}, []
    for metric in metrics:
        candidates = group.get(metric, [])
        if not candidates:
            problems.append(metric + ":missing")
            continue
        # Duplicate economic values with separate identities cannot silently
        # choose a vintage/tag. Repeated identical records can be deduplicated.
        unique = {}
        for candidate in candidates:
            identity = tuple(None if pd.isna(candidate[key]) else candidate[key] for key in (
                "value_id", "value", "normalization_method", "row_status", "tag_conflict", "input_fact_ids"))
            unique.setdefault(identity, candidate)
        candidates = list(unique.values())
        if len(candidates) != 1:
            problems.append(metric + ":ambiguous_multiple_records")
            continue
        candidate = candidates[0]
        problem = _row_problem(candidate)
        if problem:
            problems.append(metric + ":" + problem)
        else:
            found[metric] = float(candidate["value"])
    return found, problems, value_ids, fact_ids


def _metric_rows(group):
    """Index the small set of checked metrics once per filing-period group."""
    columns = ["metric_id", "value_id", "value", "normalization_method", "row_status", "tag_conflict", "input_fact_ids", "period_kind"]
    wanted = set(BALANCE_METRICS + GROSS_METRICS + CASH_METRICS)
    result = {}
    for values in group[columns].itertuples(index=False, name=None):
        if values[0] in wanted:
            result.setdefault(values[0], []).append(dict(zip(columns, values)))
    return result


def _group_problem(context):
    for key in ("issuer_id", "accn", "unit", "period_end", "period_kind"):
        if _missing(context[key]):
            return "missing_group_identity:" + key
    if context["unit"] != "USD":
        return "unsupported_currency_unit; the absolute tolerance is one USD"
    kind = context["period_kind"]
    if kind == "instant":
        if not _missing(context["period_start"]):
            return "instant_has_period_start"
    elif kind not in REPORTED_DURATIONS or _missing(context["period_start"]):
        return "not_an_original_reported_duration"
    return None


def _check(group, context, name, metrics, detail, formula=None, limitation=None):
    found, problems, value_ids, fact_ids = _inputs(group, metrics)
    group_problem = _group_problem(context)
    if group_problem:
        problems.append(group_problem)
    if limitation:
        problems.append(limitation)
    observed = found.get(metrics[0])
    expected = difference = tolerance = None
    status = "not_tested"
    if not problems and formula is not None:
        expected = formula(found)
        difference = observed - expected
        # abs only sets the numerical tolerance. It never changes input signs.
        scale = max(abs(value) for value in found.values())
        tolerance = max(1.0, scale * 1e-4)
        status = "pass" if abs(difference) <= tolerance else "review"
    row = {**context, "check": name, "status": status, "observed": observed,
           "expected": expected, "difference": difference, "tolerance": tolerance,
           "input_value_ids": _json(value_ids), "input_fact_ids": _json(fact_ids),
           "detail": detail + ("; " + "; ".join(problems) if problems else "")}
    identity = {key: row[key] for key in GROUP_COLUMNS + ["check", "input_value_ids"]}
    row["check_id"] = "tieout_" + hashlib.sha256(_json(identity).encode()).hexdigest()[:24]
    return row


def check_tieouts(values: pd.DataFrame) -> pd.DataFrame:
    """Return pass/review/not_tested evidence without mutating ``values``.

    Each check uses one issuer, accession, unit, actual period and period kind.
    All filing vintages remain separate; derived quarters and conflicted tags
    cannot turn a missing reported component into a passing check. A non-test
    records its reason and any available source IDs. Numerical discrepancies
    are review findings rather than reasons to delete or repair disclosures.
    """
    if values.empty:
        return pd.DataFrame(columns=CHECK_COLUMNS)
    required = set(GROUP_COLUMNS) | {
        "metric_id", "value", "value_id", "normalization_method", "tag_conflict",
        "row_status", "input_fact_ids",
    }
    missing = required - set(values)
    if missing:
        raise ValueError("Missing financial tie-out columns: " + ", ".join(sorted(missing)))
    results = []
    for keys, group in values.groupby(GROUP_COLUMNS, dropna=False, sort=False):
        context = {key: None if pd.isna(value) else value for key, value in zip(GROUP_COLUMNS, keys)}
        symbols = sorted(set(group.symbol.dropna().astype(str))) if "symbol" in group else []
        context["symbol"] = "|".join(symbols) or None
        metric_rows = _metric_rows(group)
        if context["period_kind"] == "instant":
            results.append(_check(
                metric_rows, context, "balance_sheet_equation", BALANCE_METRICS,
                "total_assets = total_liabilities + equity_including_noncontrolling; parent-only equity is insufficient",
                lambda found: found["total_liabilities"] + found["equity_including_noncontrolling"],
            ))
        else:
            results.append(_check(
                metric_rows, context, "gross_profit_equation", GROSS_METRICS,
                "gross_profit = revenue - cost_of_revenue; original signs retained",
                lambda found: found["revenue"] - found["cost_of_revenue"],
            ))
            results.append(_check(
                metric_rows, context, "cash_flow_equation", CASH_METRICS,
                "cash_change = operating_cash_flow + investing_cash_flow + financing_cash_flow + FX",
                limitation="not_tested: mapped FX component and a verified matching cash/restricted-cash scope are unavailable; missing FX is not zero",
            ))
    return pd.DataFrame(results, columns=CHECK_COLUMNS)


FRESHNESS_COLUMNS = CHECK_COLUMNS + [
    "latest_report_date", "latest_normalized_period_end", "latest_filed",
    "latest_filing_accn", "latest_filing_has_values", "latest_filing_has_report_period",
]


def check_freshness(values: pd.DataFrame, filings: pd.DataFrame,
                    issuers: list[dict]) -> pd.DataFrame:
    """Flag source/mapping freshness gaps independently for each legal issuer.

    ``observed`` is the signed number of days by which the latest periodic
    report date exceeds the latest valid, non-share financial period; expected
    is zero. A cover-page share count cannot advance financial freshness. This
    is a quality review, never a transport-failure or full-field-coverage claim.
    Latest filings also require normalized values for that accession and report
    period; an unprocessed amendment cannot pass on an older filing's values.
    """
    for label, frame, required in (
        ("values", values, {"issuer_id", "row_status", "unit", "period_end", "accn"}),
        ("filings", filings, {"issuer_id", "form", "report_date", "filed", "accn"}),
    ):
        missing = required - set(frame)
        if not frame.empty and missing:
            raise ValueError(f"Missing financial freshness {label} columns: " + ", ".join(sorted(missing)))
    normalized = values.copy()
    if not normalized.empty:
        normalized = normalized.loc[normalized.row_status.eq("ok") & normalized.unit.notna()
                                    & normalized.unit.ne("shares")].copy()
        if "tag_conflict" in normalized:
            normalized = normalized.loc[normalized.tag_conflict.eq(False)].copy()
        normalized["_period_end"] = pd.to_datetime(normalized.period_end, errors="coerce")
        normalized = normalized.loc[normalized._period_end.notna()]
    periodic = filings.copy()
    if not periodic.empty:
        periodic = periodic.loc[periodic.form.isin(["10-K", "10-K/A", "10-Q", "10-Q/A"])].copy()
        periodic["_report_date"] = pd.to_datetime(periodic.report_date, errors="coerce")
        periodic["_filed"] = pd.to_datetime(periodic.filed, errors="coerce")
        periodic = periodic.loc[periodic._report_date.notna()]
    results, seen = [], set()
    iso = lambda value: None if value is None or pd.isna(value) else value.date().isoformat()
    for issuer in issuers:
        issuer_id = issuer["issuer_id"]
        if issuer_id in seen:
            continue
        seen.add(issuer_id)
        v = normalized.loc[normalized.issuer_id.eq(issuer_id)] if not normalized.empty else normalized
        f = periodic.loc[periodic.issuer_id.eq(issuer_id)] if not periodic.empty else periodic
        norm_end = v._period_end.max() if not v.empty else None
        report_end = f._report_date.max() if not f.empty else None
        latest = f.loc[f._report_date.eq(report_end)] if not f.empty else f
        filed = latest._filed.max() if not latest.empty else None
        if filed is not None and not pd.isna(filed):
            latest = latest.loc[latest._filed.eq(filed)]
        accessions = sorted({str(x) for x in latest.accn.dropna() if str(x)}) if not latest.empty else []
        latest_values = v.loc[v.accn.isin(accessions)] if not v.empty else v
        has_values = bool(accessions) and not latest_values.empty
        report_values = latest_values.loc[latest_values._period_end.eq(report_end)] if has_values else latest_values
        has_report = bool(accessions) and not report_values.empty and set(accessions) <= set(report_values.accn)
        observed = float((report_end - norm_end).days) if report_end is not None and norm_end is not None else None
        reasons = []
        if report_end is None:
            reasons.append("no_valid_periodic_filing_report_date")
        if norm_end is None:
            reasons.append("no_valid_nonshare_normalized_financial_period")
        if not accessions:
            reasons.append("missing_latest_periodic_accession")
        if filed is None or pd.isna(filed):
            reasons.append("missing_latest_periodic_filed_date")
        status = "not_tested" if reasons else ("review" if observed > 0 or not has_report else "pass")
        if status == "review":
            if observed > 0:
                reasons.append("reported_period_newer_than_normalized_financial_period")
            if not has_report:
                reasons.append("latest_filing_missing_normalized_values_for_its_report_period")
        evidence = v.loc[v._period_end.eq(norm_end)] if not v.empty else v
        if not report_values.empty:
            evidence = pd.concat([evidence, report_values], ignore_index=True)
        value_ids = sorted(set(evidence.value_id.dropna().astype(str))) if "value_id" in evidence else []
        fact_ids = sorted({item for ids in evidence.input_fact_ids for item in _ids(ids)}) if "input_fact_ids" in evidence else []
        context = {"issuer_id": issuer_id, "symbol": issuer.get("symbol"), "accn": "|".join(accessions) or None,
                   "unit": "days", "period_start": None, "period_end": iso(report_end), "period_kind": None,
                   "check": "filing_fact_freshness", "status": status, "observed": observed,
                   "expected": 0.0, "difference": observed, "tolerance": 0.0,
                   "input_value_ids": _json(value_ids), "input_fact_ids": _json(fact_ids),
                   "latest_report_date": iso(report_end), "latest_normalized_period_end": iso(norm_end),
                   "latest_filed": iso(filed), "latest_filing_accn": "|".join(accessions) or None,
                   "latest_filing_has_values": has_values, "latest_filing_has_report_period": has_report,
                   "detail": (f"Latest periodic report={iso(report_end)}; latest valid non-share financial period={iso(norm_end)}; "
                              f"latest accession has normalized values={has_values}, report-period values={has_report}. "
                              "Source/mapping freshness review, not an acquisition failure or proof of complete field coverage. "
                              + "; ".join(reasons))}
        context["check_id"] = "freshness_" + hashlib.sha256(_json([issuer_id, iso(report_end), iso(norm_end), accessions]).encode()).hexdigest()[:24]
        results.append(context)
    return pd.DataFrame(results, columns=FRESHNESS_COLUMNS)
