"""Offline parser for SEC Financial Statement Data Sets (FSDS) ZIP files.

The parser keeps as-filed identifiers and values close to the SEC flat files.
It is an ingestion layer only: missing facts remain missing and no accounting
meaning is inferred from a tag or a segment.
"""
from __future__ import annotations

import csv
from datetime import date, datetime
import hashlib
import io
import json
from pathlib import Path
from typing import Iterable, Iterator, Mapping
import zipfile


CORE_TAGS = frozenset({
    # Net income / profit and loss comparisons.
    "ProfitLoss", "NetIncomeLoss", "ProfitLossAttributableToParent",
    "NetIncomeLossAttributableToParent", "IncomeLossFromContinuingOperations",
    "IncomeLossFromContinuingOperationsIncludingPortionAttributableToNoncontrollingInterest",
    # Operating cash flow.
    "NetCashProvidedByUsedInOperatingActivities",
    "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
    "NetCashProvidedByUsedInOperatingActivitiesContinuingOperationsIncludingDiscontinuedOperations",
    # Assets.
    "Assets", "AssetsCurrent", "AssetsNoncurrent",
    # Non-controlling interest comparisons.
    "NetIncomeLossAttributableToNoncontrollingInterest",
    "ProfitLossAttributableToNoncontrollingInterest",
    "MinorityInterest", "NoncontrollingInterestInConsolidatedEntity",
    "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
})


def _clean(value: str | None) -> str | None:
    return value if value not in (None, "") else None


def _rows(raw: io.TextIOBase, required_columns: frozenset[str] = frozenset()) -> Iterator[dict[str, str | None]]:
    reader = csv.DictReader(raw, delimiter="\t")
    if not reader.fieldnames:
        raise ValueError("SEC FSDS member has no tab-separated header")
    names = [name.strip() for name in reader.fieldnames]
    if any(not name for name in names):
        raise ValueError("SEC FSDS member has an empty column name")
    if len(names) != len(set(names)):
        raise ValueError("SEC FSDS member has duplicate column names")
    if not required_columns.issubset(names):
        raise ValueError(f"SEC FSDS member lacks required columns: {sorted(required_columns - set(names))}")
    for row in reader:
        if None in row:
            raise ValueError("SEC FSDS member has more fields than its header")
        if [row.get(name) for name in names] == names:
            raise ValueError("SEC FSDS member contains a repeated header row")
        yield {name: _clean(row.get(name)) for name in names}


def _read_member(zf: zipfile.ZipFile, name: str, required_columns: frozenset[str] = frozenset()) -> Iterator[dict[str, str | None]]:
    try:
        member = zf.getinfo(name)
    except KeyError as exc:
        raise ValueError(f"SEC FSDS ZIP lacks required member {name}") from exc
    if member.is_dir():
        raise ValueError(f"SEC FSDS member is a directory: {name}")
    with zf.open(member, "r") as binary:
        with io.TextIOWrapper(binary, encoding="utf-8-sig", newline="") as text:
            yield from _rows(text, required_columns)


def _date_key(value: str | None) -> str | None:
    if not value:
        return None
    value = str(value).strip()
    if len(value) == 8 and value.isdigit():
        try:
            return datetime.strptime(value, "%Y%m%d").date().isoformat()
        except ValueError as exc:
            raise ValueError(f"Invalid SEC compact date: {value}") from exc
    try:
        return date.fromisoformat(value[:10]).isoformat()
    except ValueError as exc:
        raise ValueError(f"Invalid SEC date: {value}") from exc


def _cutoff(value: str | date | None) -> str | None:
    if value is None:
        return None
    result = value.isoformat() if isinstance(value, date) else str(value)
    return _date_key(result)


def _source_hash(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as raw:
        for block in iter(lambda: raw.read(1024 * 1024), b""):
            digest.update(block)
            size += len(block)
    return digest.hexdigest(), size


def _filing_index(rows: Iterable[dict[str, str | None]], cutoff: str | None):
    result: dict[str, dict[str, str | None]] = {}
    for row in rows:
        adsh = row.get("adsh")
        if not adsh:
            continue
        if not row.get("cik") or not row.get("form") or not row.get("period") or not row.get("filed"):
            raise ValueError("SEC sub.txt row lacks a required filing field")
        filed = row.get("filed")
        filed_key = _date_key(filed)
        if cutoff is not None and (filed_key is None or filed_key > cutoff):
            continue
        if adsh in result:
            raise ValueError(f"SEC sub.txt has duplicate accession {adsh}")
        row["filed_raw"] = filed
        row["period_raw"] = row.get("period")
        row["accepted_raw"] = row.get("accepted")
        # Preserve the complete sub row while exposing the required identity
        # fields in the same object; unknown SEC columns remain available.
        result[adsh] = row
    return result


def parse_fsds_zip(
    zip_path: str | Path,
    *,
    cutoff: str | date | None = None,
    version: str | None = None,
    source_url: str | None = None,
) -> dict[str, object]:
    """Parse one local SEC FSDS ZIP into filings, core facts and provenance.

    ``cutoff`` is applied to the filing date (``sub.txt:filed``), not the
    observation date. Facts retain SEC ``segments`` verbatim, including null
    values; no zero-filling or ticker lookup occurs.
    """
    path = Path(zip_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    cutoff_value = _cutoff(cutoff)
    sha256, byte_count = _source_hash(path)
    with zipfile.ZipFile(path) as zf:
        members = [info for info in zf.infolist() if not info.is_dir()]
        names = {info.filename for info in members}
        if len(names) != len(members):
            raise ValueError("SEC FSDS ZIP contains duplicate members")
        required = {"sub.txt", "num.txt", "tag.txt", "pre.txt"}
        missing = required - names
        if missing:
            raise ValueError(f"SEC FSDS ZIP lacks required members: {sorted(missing)}")
        filings_by_adsh = _filing_index(_read_member(
            zf, "sub.txt", frozenset({"adsh", "cik", "name", "sic", "form", "period", "filed", "accepted"})
        ), cutoff_value)

        facts: list[dict[str, str | None]] = []
        observation_after_cutoff_count = 0
        for row in _read_member(zf, "num.txt", frozenset({"adsh", "tag", "version", "ddate", "value"})):
            adsh, tag = row.get("adsh"), row.get("tag")
            filing = filings_by_adsh.get(adsh or "")
            if filing is None or tag not in CORE_TAGS:
                continue
            fact = dict(row)
            fact["ddate_raw"] = row.get("ddate")
            fact["observation_after_cutoff"] = bool(
                cutoff_value and _date_key(row.get("ddate")) and _date_key(row.get("ddate")) > cutoff_value
            )
            if fact["observation_after_cutoff"]:
                observation_after_cutoff_count += 1
            fact.update({
                "cik": filing.get("cik"), "name": filing.get("name"),
                "form": filing.get("form"), "period": filing.get("period"),
                "filed": filing.get("filed"), "accepted": filing.get("accepted"),
                "sic": filing.get("sic"),
                "source_zip_sha256": sha256,
            })
            facts.append(fact)

        tags = [row for row in _read_member(zf, "tag.txt", frozenset({"tag", "version"})) if row.get("tag") in CORE_TAGS]
        presentation = [
            row for row in _read_member(zf, "pre.txt", frozenset({"adsh", "tag", "version"}))
            if row.get("adsh") in filings_by_adsh and row.get("tag") in CORE_TAGS
        ]

    filings = list(filings_by_adsh.values())
    manifest = {
        "source_path": str(path), "source_url": source_url,
        "source_zip_sha256": sha256, "source_bytes": byte_count,
        "version": version, "cutoff_filed": cutoff_value,
        "members_required": sorted(required), "filing_count": len(filings),
        "fact_count": len(facts), "tag_count": len(tags),
        "presentation_count": len(presentation),
        "observation_after_cutoff_count": observation_after_cutoff_count,
    }
    return {"filings": filings, "facts": facts, "tags": tags,
            "presentation": presentation, "manifest": manifest}


def build_manifest(
    zip_paths: Iterable[str | Path], *, cutoff: str | date | None = None,
    version: str | None = None, source_urls: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Create JSON-ready manifests for local packages without parsing facts."""
    cutoff_value = _cutoff(cutoff)
    packages = []
    for item in zip_paths:
        path = Path(item)
        sha256, byte_count = _source_hash(path)
        packages.append({
            "source_path": str(path), "source_url": (source_urls or {}).get(str(path)),
            "source_zip_sha256": sha256, "source_bytes": byte_count,
            "version": version, "cutoff_filed": cutoff_value,
        })
    return {"package_count": len(packages), "packages": packages}


def write_manifest(path: str | Path, manifest: Mapping[str, object]) -> None:
    """Write a deterministic local manifest; callers control Lake publication."""
    Path(path).write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
