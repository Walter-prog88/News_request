#!/usr/bin/env python3
"""Extract structured facts from research report Markdown files.

Produces two parquet files:
  data/facts/report_facts.parquet    — one row per report
  data/facts/stock_consensus.parquet — one row per stock (aggregated)

Usage:
    # Full run (all 2218 reports):
    .venv/bin/python scripts/extract_report_facts.py

    # Validate on first 100 reports:
    .venv/bin/python scripts/extract_report_facts.py --limit 100

    # Save CSV copies for quick inspection:
    .venv/bin/python scripts/extract_report_facts.py --csv
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

LOGGER = logging.getLogger(__name__)
FACTS_DIR = PROJECT_ROOT / "data" / "facts"
MARKDOWN_DIR = PROJECT_ROOT / "data" / "processed" / "markdown"

# ---------------------------------------------------------------------------
# Rating normalisation
# ---------------------------------------------------------------------------

_POSITIVE = frozenset({
    "买入", "强烈推荐", "强推", "推荐", "买入评级",
    "Buy", "Overweight", "Strong Buy", "Add", "增持",
})
_NEGATIVE = frozenset({
    "减持", "卖出", "回避", "Sell", "Underperform", "Underweight",
})


def rating_label(raw: str | None) -> str:
    """Map raw rating string to Positive / Neutral / Negative."""
    if not raw:
        return "Unknown"
    r = raw.strip()
    if r in _POSITIVE:
        return "Positive"
    if r in _NEGATIVE:
        return "Negative"
    return "Neutral"


# ---------------------------------------------------------------------------
# Front-matter parser (lightweight, no PyYAML required)
# ---------------------------------------------------------------------------

_FM_RE = re.compile(r"^---\s*\n(.*?)\n---", re.DOTALL)


def parse_frontmatter(text: str) -> dict:
    m = _FM_RE.match(text)
    if not m:
        return {}
    fm: dict = {}
    for line in m.group(1).splitlines():
        if ":" not in line:
            continue
        key, _, val = line.partition(":")
        fm[key.strip()] = val.strip().strip("'\"")
    return fm


# ---------------------------------------------------------------------------
# EPS table parser
# ---------------------------------------------------------------------------

_TABLE_ROW_RE = re.compile(r"\|([^|]+)\|([^|]+)\|([^|]+)\|([^|]+)\|")
_YEAR_RE = re.compile(r"(20\d\d)")
_NUM_RE = re.compile(r"[\d,]+\.?\d*")


def _clean_num(s: str) -> float | None:
    s = s.replace(",", "").strip()
    m = _NUM_RE.search(s)
    if m:
        try:
            return float(m.group())
        except ValueError:
            pass
    return None


def parse_eps_table(body: str) -> list[dict]:
    """
    Parse the standard 业绩预测 table.

    Expected columns: 预测期 | EPS（元） | 净利润（万元） | PE | ROE（%） | …
    Returns list of {year, eps, pe} dicts sorted by year.
    """
    rows = []
    in_table = False
    header_checked = False
    eps_col = pe_col = -1

    for line in body.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            if in_table:
                break
            continue

        cols = [c.strip() for c in stripped.split("|") if c.strip()]
        if not cols:
            continue

        # Detect header row
        if not header_checked:
            header_lower = " ".join(cols).lower()
            if "eps" in header_lower or "每股收益" in header_lower:
                in_table = True
                header_checked = True
                for i, c in enumerate(cols):
                    cl = c.lower()
                    if "eps" in cl or "每股收益" in cl:
                        eps_col = i
                    elif cl.strip() == "pe":
                        pe_col = i
            continue

        if not in_table:
            continue
        # Skip separator row
        if all(c.replace("-", "").replace("|", "") == "" for c in cols):
            continue

        period = cols[0] if cols else ""
        ym = _YEAR_RE.search(period)
        if not ym:
            continue
        year = int(ym.group(1))

        eps = _clean_num(cols[eps_col]) if eps_col != -1 and eps_col < len(cols) else None
        pe = _clean_num(cols[pe_col]) if pe_col != -1 and pe_col < len(cols) else None

        if eps is not None:
            rows.append({"year": year, "eps": eps, "pe": pe})

    return sorted(rows, key=lambda r: r["year"])


# ---------------------------------------------------------------------------
# Target price extraction — frontmatter-first, regex fallback
# ---------------------------------------------------------------------------

_TARGET_PRICE_RE = re.compile(
    r"(?:目标价[格至为到]?|人民币|target\s*price)[^\d]*?([\d]+\.?\d*)\s*元?",
    re.IGNORECASE,
)


def _parse_float(s: str | None) -> float | None:
    if not s:
        return None
    try:
        return float(str(s).strip())
    except ValueError:
        return None


def extract_target_prices(fm: dict, text: str) -> tuple[float | None, float | None]:
    """Return (tp_low, tp_high) — prefer frontmatter fields, fall back to regex."""
    tp_low = _parse_float(fm.get("tp_low"))
    tp_high = _parse_float(fm.get("tp_high"))
    if tp_low is not None:
        return tp_low, tp_high

    # Regex fallback for legacy files without frontmatter tp_low
    m = _TARGET_PRICE_RE.search(text[:3000])
    if m:
        try:
            val = float(m.group(1))
            return val, None
        except ValueError:
            pass
    return None, None


# ---------------------------------------------------------------------------
# Per-file extraction
# ---------------------------------------------------------------------------

def extract_one(path: Path) -> dict | None:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None

    fm = parse_frontmatter(text)
    if not fm.get("ts_code"):
        return None

    # Body starts after second ---
    body_start = text.find("---", 3)
    body = text[body_start + 3:] if body_start != -1 else text

    eps_rows = parse_eps_table(body)
    tp_low, tp_high = extract_target_prices(fm, text)

    row: dict = {
        "ts_code": fm.get("ts_code"),
        "name": fm.get("name"),
        "report_date": fm.get("report_date"),
        "org_name": fm.get("org_name"),
        "author_name": fm.get("author_name"),
        "rating": fm.get("rating"),
        "rating_label": rating_label(fm.get("rating")),
        "tp_low": tp_low,
        "tp_high": tp_high,
        "source_file": str(path),
    }

    # Up to 3 forecast years
    for i, r in enumerate(eps_rows[:3], 1):
        row[f"eps_y{i}"] = r["eps"]
        row[f"eps_y{i}_year"] = r["year"]
        row[f"pe_y{i}"] = r["pe"]

    return row


# ---------------------------------------------------------------------------
# Aggregation → stock_consensus
# ---------------------------------------------------------------------------

_BUY_LABELS = frozenset({"Positive"})
_SELL_LABELS = frozenset({"Negative"})
_RECENT_DAYS = 30


def build_consensus(df: pd.DataFrame, today: datetime) -> pd.DataFrame:
    """Aggregate report_facts by ts_code into consensus signals."""
    df = df.copy()
    df["report_date"] = pd.to_datetime(df["report_date"], errors="coerce", utc=False)
    cutoff = pd.Timestamp(today.replace(tzinfo=None) - timedelta(days=_RECENT_DAYS))

    rows = []
    for ts_code, grp in df.groupby("ts_code"):
        grp = grp.sort_values("report_date", ascending=False)
        recent = grp[grp["report_date"] >= cutoff]
        prev = grp[grp["report_date"] < cutoff]

        n = len(grp)
        n_recent = len(recent)
        pos_pct = (grp["rating_label"] == "Positive").mean() * 100
        neg_pct = (grp["rating_label"] == "Negative").mean() * 100

        recent_pos = (recent["rating_label"] == "Positive").mean() * 100 if n_recent else float("nan")
        prev_pos = (prev["rating_label"] == "Positive").mean() * 100 if len(prev) else float("nan")

        if n_recent >= 2 and len(prev) >= 2:
            trend = "升级" if recent_pos > prev_pos + 5 else ("降级" if recent_pos < prev_pos - 5 else "稳定")
        else:
            trend = "数据不足"

        # Consensus EPS (next forecast year = minimum y1_year in recent reports)
        y1_data = grp[grp["eps_y1"].notna() & grp["eps_y1_year"].notna()]
        eps_y1_mean = eps_y1_std = eps_y1_year = float("nan")
        pe_y1_mean = float("nan")
        if not y1_data.empty:
            eps_y1_mean = y1_data["eps_y1"].mean()
            eps_y1_std = y1_data["eps_y1"].std()
            eps_y1_year = int(y1_data["eps_y1_year"].mode().iloc[0])
            pe_y1_mean = y1_data["pe_y1"].dropna().mean()

        # Target price consensus and change signal
        tp_data = grp["tp_low"].dropna().sort_index()  # sorted by report_date (already done above)
        grp_tp = grp.dropna(subset=["tp_low"]).sort_values("report_date", ascending=False)
        tp_mean = round(grp_tp["tp_low"].mean(), 2) if not grp_tp.empty else None
        tp_count = int(grp_tp["tp_low"].count())

        # Target price change: latest vs previous report with tp_low (per-institution where possible)
        tp_change = tp_change_pct = None
        if len(grp_tp) >= 2:
            tp_latest = grp_tp["tp_low"].iloc[0]
            tp_prev = grp_tp["tp_low"].iloc[1]
            tp_change = round(tp_latest - tp_prev, 2)
            if tp_prev != 0:
                tp_change_pct = round((tp_latest - tp_prev) / tp_prev * 100, 1)

        # Among institutions that issued >=2 reports, count how many raised/cut TP recently
        tp_raised = tp_cut = 0
        for _, org_grp in grp_tp.groupby("org_name"):
            if len(org_grp) >= 2:
                org_sorted = org_grp.sort_values("report_date", ascending=False)
                delta = org_sorted["tp_low"].iloc[0] - org_sorted["tp_low"].iloc[1]
                if delta > 0:
                    tp_raised += 1
                elif delta < 0:
                    tp_cut += 1

        rows.append({
            "ts_code": ts_code,
            "name": grp["name"].iloc[0],
            "coverage_count": n,
            "recent_30d_count": n_recent,
            "latest_report_date": grp["report_date"].max().date() if not grp["report_date"].isna().all() else None,
            "positive_pct": round(pos_pct, 1),
            "negative_pct": round(neg_pct, 1),
            "neutral_pct": round(100 - pos_pct - neg_pct, 1),
            "recent_positive_pct": round(recent_pos, 1) if not pd.isna(recent_pos) else None,
            "rating_trend": trend,
            "eps_y1_mean": round(eps_y1_mean, 2) if not pd.isna(eps_y1_mean) else None,
            "eps_y1_std": round(eps_y1_std, 2) if not pd.isna(eps_y1_std) else None,
            "eps_y1_year": int(eps_y1_year) if not pd.isna(eps_y1_year) else None,
            "pe_y1_mean": round(pe_y1_mean, 1) if not pd.isna(pe_y1_mean) else None,
            "tp_mean": tp_mean,
            "tp_count": tp_count,
            "tp_change": tp_change,
            "tp_change_pct": tp_change_pct,
            "tp_raised_count": tp_raised,
            "tp_cut_count": tp_cut,
        })

    return pd.DataFrame(rows).sort_values("coverage_count", ascending=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(limit: int | None = None, save_csv: bool = False) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    FACTS_DIR.mkdir(parents=True, exist_ok=True)

    paths = sorted(MARKDOWN_DIR.glob("rc_*.md"))
    if limit:
        paths = paths[:limit]
    LOGGER.info("Processing %d research report files...", len(paths))

    records = []
    failed = 0
    for path in paths:
        row = extract_one(path)
        if row:
            records.append(row)
        else:
            failed += 1

    if not records:
        LOGGER.error("No records extracted. Check markdown directory.")
        return

    df = pd.DataFrame(records)
    out_facts = FACTS_DIR / "report_facts.parquet"
    df.to_parquet(out_facts, index=False)
    LOGGER.info("Saved %d rows → %s", len(df), out_facts)

    if save_csv:
        df.to_csv(FACTS_DIR / "report_facts.csv", index=False, encoding="utf-8-sig")

    today = datetime.now(timezone.utc)
    consensus = build_consensus(df, today)
    out_consensus = FACTS_DIR / "stock_consensus.parquet"
    consensus.to_parquet(out_consensus, index=False)
    LOGGER.info("Saved %d stocks → %s", len(consensus), out_consensus)

    if save_csv:
        consensus.to_csv(FACTS_DIR / "stock_consensus.csv", index=False, encoding="utf-8-sig")

    # ── Quick validation summary ──────────────────────────────────────────
    eps_coverage = df["eps_y1"].notna().mean() * 100
    tp_coverage = df["tp_low"].notna().mean() * 100
    rating_dist = df["rating_label"].value_counts(normalize=True).mul(100).round(1).to_dict()

    LOGGER.info("=== Extraction Summary ===")
    LOGGER.info("  Total reports:       %d  (failed: %d)", len(df), failed)
    LOGGER.info("  EPS y1 coverage:     %.1f%%", eps_coverage)
    LOGGER.info("  tp_low coverage:     %.1f%%", tp_coverage)
    LOGGER.info("  Rating distribution: %s", rating_dist)
    LOGGER.info("  Stocks covered:      %d", len(consensus))

    # Show top 10 most-covered stocks with TP info
    display_cols = ["ts_code", "name", "coverage_count", "positive_pct", "rating_trend",
                    "tp_mean", "tp_change", "tp_change_pct", "tp_raised_count", "tp_cut_count"]
    available_cols = [c for c in display_cols if c in consensus.columns]
    top10 = consensus.head(10)[available_cols]
    LOGGER.info("Top 10 most-covered stocks (TP focus):\n%s", top10.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract facts from research report Markdown files")
    parser.add_argument("--limit", type=int, default=None, help="Process only first N files (validation mode)")
    parser.add_argument("--csv", action="store_true", help="Also save CSV copies for inspection")
    args = parser.parse_args()
    main(limit=args.limit, save_csv=args.csv)

