"""Import Tushare research report summaries into the RAG knowledge base.

Reads ../tushare_quant/quant_data_6y/report_rc.parquet and converts
individual-stock research reports to Markdown files ready for FAISS indexing.

Strategy:
  - Individual-stock reports only (exclude sector/macro "非个股")
  - 2024-01-01 onwards (recent enough to be useful)
  - Top-5 most recent unique reports per stock (keeps index manageable)
  - Each unique (ts_code, report_date, report_title, org_name) → one .md file

Usage:
    python scripts/import_research_reports.py
    python scripts/import_research_reports.py --since 20230101 --top-n 10
    python scripts/import_research_reports.py --stock-code 300308
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import re
import sys
from pathlib import Path

import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PARQUET_PATH = PROJECT_ROOT.parent / "tushare_quant" / "quant_data_6y" / "report_rc.parquet"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "markdown"

REPORT_KEY = ["ts_code", "report_date", "report_title", "org_name"]


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _safe_filename(ts_code: str, report_date: str, org_name: str, title: str) -> str:
    """Build a filesystem-safe filename from report metadata."""
    code = ts_code.replace(".", "_")
    org = re.sub(r"[^\w一-鿿]", "", org_name)[:8]
    title_slug = re.sub(r"[^\w一-鿿]", "_", title)[:30]
    return f"rc_{code}_{report_date}_{org}_{title_slug}.md"


def _fmt_num(v, digits: int = 2) -> str:
    if pd.isna(v):
        return "N/A"
    try:
        return f"{float(v):,.{digits}f}"
    except (TypeError, ValueError):
        return "N/A"


def build_markdown(report_date: str, ts_code: str, name: str,
                   report_title: str, report_type: str, org_name: str,
                   author_name: str, rating: str,
                   quarter_rows: pd.DataFrame) -> str:
    """Render one research report as Markdown with YAML front matter."""
    date_fmt = f"{report_date[:4]}-{report_date[4:6]}-{report_date[6:]}"
    symbol = ts_code.split(".")[0]

    body_lines = [
        f"# 研报摘要：{name}（{ts_code}）",
        "",
        f"**标题**：{report_title}",
        "",
        f"**机构**：{org_name}　**分析师**：{author_name}　**评级**：{rating}　**日期**：{date_fmt}",
        f"**股票**：{name}（{ts_code} / {symbol}）　**报告类型**：{report_type}",
        "",
    ]

    # Forecast table — one row per quarter, sorted chronologically
    valid_rows = quarter_rows.dropna(subset=["quarter"]).sort_values("quarter")
    if not valid_rows.empty:
        body_lines += [
            "## 业绩预测",
            "",
            "| 预测期 | EPS（元） | 净利润（万元） | PE | ROE（%） | 目标净利（万元） |",
            "|--------|----------|--------------|-----|---------|----------------|",
        ]
        for _, row in valid_rows.iterrows():
            body_lines.append(
                f"| {row['quarter']} "
                f"| {_fmt_num(row.get('eps'))} "
                f"| {_fmt_num(row.get('np'), 0)} "
                f"| {_fmt_num(row.get('pe'))} "
                f"| {_fmt_num(row.get('roe'))} "
                f"| {_fmt_num(row.get('tp'), 0)} |"
            )
        body_lines.append("")

    body = "\n".join(body_lines)

    # Extract per-report target price from the first non-null row
    tp_low = tp_high = None
    if not quarter_rows.empty:
        tp_low_vals = quarter_rows["min_price"].dropna()
        tp_high_vals = quarter_rows["max_price"].dropna()
        if not tp_low_vals.empty:
            tp_low = round(float(tp_low_vals.iloc[0]), 2)
        if not tp_high_vals.empty:
            tp_high = round(float(tp_high_vals.iloc[0]), 2)

    front_matter = {
        "source": "research_report",
        "title": f"{name}研报：{report_title}",  # used by retriever for display/scoring
        "ts_code": ts_code,
        "name": name,
        "symbol": symbol,
        "report_date": date_fmt,
        "report_title": report_title,
        "report_type": report_type,
        "org_name": org_name,
        "author_name": author_name,
        "rating": rating,
        "content_hash": _content_hash(body),
    }
    if tp_low is not None:
        front_matter["tp_low"] = tp_low
    if tp_high is not None:
        front_matter["tp_high"] = tp_high

    fm_str = yaml.dump(front_matter, allow_unicode=True, default_flow_style=False, sort_keys=False)
    return f"---\n{fm_str}---\n\n{body}"


def load_and_filter(parquet_path: Path, since: str, top_n: int,
                    stock_code: str | None) -> pd.DataFrame:
    """Load parquet, filter, deduplicate, select top-N per stock."""
    LOGGER.info("Reading %s ...", parquet_path)
    df = pd.read_parquet(parquet_path)
    LOGGER.info("Loaded %d rows", len(df))

    # Normalize stock code
    if stock_code:
        code = stock_code.upper()
        if "." not in code:
            code = code + ".SZ" if int(code) < 600000 else code + ".SH"
        df = df[df["ts_code"] == code]
        LOGGER.info("Filtered to %s: %d rows", code, len(df))

    # Individual-stock reports only, recent dates
    df = df[
        (df["report_type"] != "非个股") &
        (df["report_date"] >= since)
    ]
    LOGGER.info("After type/date filter: %d rows", len(df))

    # Deduplicate to unique reports
    unique_reports = df.drop_duplicates(subset=REPORT_KEY)
    unique_reports = unique_reports.sort_values("report_date", ascending=False)

    # Top-N most recent per stock
    top = unique_reports.groupby("ts_code").head(top_n)
    LOGGER.info("Unique reports to import: %d (covering %d stocks)",
                len(top), top["ts_code"].nunique())
    return top, df  # also return full df for quarter lookup


def run(since: str = "20240101", top_n: int = 5, stock_code: str | None = None) -> None:
    if not PARQUET_PATH.exists():
        LOGGER.error("Parquet not found: %s", PARQUET_PATH)
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    top_reports, full_df = load_and_filter(PARQUET_PATH, since, top_n, stock_code)

    # Group full_df by report key for quarter lookup
    full_indexed = full_df.set_index(REPORT_KEY)

    written = 0
    skipped = 0
    seen_hashes: set[str] = set()

    for _, meta in top_reports.iterrows():
        key = (meta["ts_code"], meta["report_date"], meta["report_title"], meta["org_name"])

        # Get all quarter rows for this report
        try:
            quarter_rows = full_df[
                (full_df["ts_code"] == key[0]) &
                (full_df["report_date"] == key[1]) &
                (full_df["report_title"] == key[2]) &
                (full_df["org_name"] == key[3])
            ]
        except Exception:
            quarter_rows = pd.DataFrame()

        md = build_markdown(
            report_date=meta["report_date"],
            ts_code=meta["ts_code"],
            name=meta.get("name", ""),
            report_title=meta["report_title"],
            report_type=meta.get("report_type", ""),
            org_name=meta["org_name"],
            author_name=meta.get("author_name", ""),
            rating=meta.get("rating", ""),
            quarter_rows=quarter_rows,
        )

        h = _content_hash(md)
        if h in seen_hashes:
            skipped += 1
            continue
        seen_hashes.add(h)

        fname = _safe_filename(
            meta["ts_code"], meta["report_date"], meta["org_name"], meta["report_title"]
        )
        out_path = OUTPUT_DIR / fname
        out_path.write_text(md, encoding="utf-8")
        written += 1

    LOGGER.info("Done — written: %d, skipped duplicates: %d", written, skipped)
    LOGGER.info("Output directory: %s", OUTPUT_DIR)
    LOGGER.info("Next step: python scripts/build_index.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Import research reports into RAG knowledge base")
    parser.add_argument("--since", default="20240101", help="Start date YYYYMMDD (default: 20240101)")
    parser.add_argument("--top-n", type=int, default=5, help="Max recent reports per stock (default: 5)")
    parser.add_argument("--stock-code", default=None, help="Import only one stock, e.g. 300308")
    args = parser.parse_args()
    run(since=args.since, top_n=args.top_n, stock_code=args.stock_code)

