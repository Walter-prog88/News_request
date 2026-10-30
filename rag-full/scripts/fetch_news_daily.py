#!/usr/bin/env python3
"""Daily news fetcher for the Fin-RAG knowledge base.

Builds a two-tier dynamic watchlist:
  Tier-1: stocks with the most Buy-rated research reports in the KB (top 30)
  Tier-2: today's 涨停股池 and 龙虎榜 hot stocks (top 20)

Then fetches recent news for each stock, saves as Markdown files into
data/processed/markdown/, cleans up files older than NEWS_RETENTION_DAYS,
and triggers an incremental FAISS index update.

Usage:
    # Normal daily run (fetch last 2 days of news):
    .venv/bin/python scripts/fetch_news_daily.py

    # Initial backfill (fetch last 21 days):
    .venv/bin/python scripts/fetch_news_daily.py --days 21

    # Dry-run: show what would happen without writing files:
    .venv/bin/python scripts/fetch_news_daily.py --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import re
import subprocess
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.indexer.build_vectorstore import load_config, resolve_project_path

LOGGER = logging.getLogger(__name__)

NEWS_RETENTION_DAYS = 90
TIER1_TOP_N = 30
TIER2_TOP_N = 20
BUY_RATINGS = frozenset({
    "买入", "增持", "强烈推荐", "强推", "推荐",
    "Buy", "Overweight", "Strong Buy", "Add",
})
_FRONT_MATTER_RE = re.compile(r"^---\s*\n(.*?)\n---", re.DOTALL)


# ---------------------------------------------------------------------------
# Watchlist builders
# ---------------------------------------------------------------------------

def get_institutional_watchlist(markdown_dir: Path, top_n: int = TIER1_TOP_N) -> list[str]:
    """Return top-N ts_codes with the most Buy-rated research reports in the KB."""
    counts: Counter = Counter()
    for path in markdown_dir.glob("rc_*.md"):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        m = _FRONT_MATTER_RE.match(text)
        if not m:
            continue
        ts_code = rating = None
        for line in m.group(1).splitlines():
            if line.startswith("ts_code:"):
                ts_code = line.split(":", 1)[1].strip().strip("'\"")
            elif line.startswith("rating:"):
                rating = line.split(":", 1)[1].strip().strip("'\"")
        if ts_code and rating and rating in BUY_RATINGS:
            counts[ts_code] += 1
    top = [code for code, _ in counts.most_common(top_n)]
    LOGGER.info("Tier-1 watchlist: %d stocks (Buy-rated research reports)", len(top))
    return top


def _plain_code(code: str) -> str:
    return code.zfill(6).replace(".SZ", "").replace(".SH", "")


def _normalize_ts_code(code: str) -> str:
    code = code.zfill(6)
    if code[:3] in ("000", "001", "002", "003", "300", "301", "302"):
        return f"{code}.SZ"
    if code[:3] in ("600", "601", "603", "605", "688"):
        return f"{code}.SH"
    return code


def get_hot_watchlist(top_n: int = TIER2_TOP_N) -> list[str]:
    """Return hot ts_codes from today's 涨停股池 and 龙虎榜."""
    import akshare as ak
    today = datetime.now().strftime("%Y%m%d")
    codes: set[str] = set()

    try:
        df = ak.stock_zt_pool_em(date=today)
        if not df.empty:
            for code in df["代码"].astype(str).head(top_n // 2):
                codes.add(_normalize_ts_code(code))
    except Exception as exc:
        LOGGER.warning("涨停股池 unavailable: %s", exc)

    try:
        df = ak.stock_lhb_detail_em(start_date=today, end_date=today)
        if not df.empty:
            for code in df["代码"].astype(str).head(top_n // 2):
                codes.add(_normalize_ts_code(code))
    except Exception as exc:
        LOGGER.warning("龙虎榜 unavailable: %s", exc)

    result = list(codes)[:top_n]
    LOGGER.info("Tier-2 watchlist: %d hot stocks (涨停+龙虎榜)", len(result))
    return result


# ---------------------------------------------------------------------------
# News fetching and Markdown conversion
# ---------------------------------------------------------------------------

def fetch_news_for_stock(ts_code: str, days: int) -> list[dict]:
    """Fetch recent news for one stock via AKShare. Returns list of article dicts."""
    import akshare as ak
    plain = _plain_code(ts_code)
    try:
        df = ak.stock_news_em(symbol=plain)
        if df.empty:
            return []
    except Exception as exc:
        LOGGER.warning("News fetch failed for %s: %s", ts_code, exc)
        return []

    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    articles = []
    for _, row in df.iterrows():
        pub_str = str(row.get("发布时间", "")).strip()[:19]
        try:
            pub_dt = datetime.strptime(pub_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        if pub_dt < cutoff:
            continue
        title = str(row.get("新闻标题", "")).strip()
        content = str(row.get("新闻内容", "")).strip()
        if not title or not content:
            continue
        articles.append({
            "title": title,
            "content": content,
            "source_name": str(row.get("文章来源", "")).strip(),
            "url": str(row.get("新闻链接", "")).strip(),
            "published_at": pub_dt.strftime("%Y-%m-%d %H:%M:%S"),
        })
    return articles


def _yaml_str(value: str) -> str:
    """Single-quote a string for YAML, escaping internal single quotes."""
    return "'" + value.replace("'", "''") + "'"


def article_to_markdown(article: dict, ts_code: str) -> tuple[str, str]:
    """Return (markdown_text, content_hash) for one news article."""
    raw = f"{article['title']}\n\n{article['content']}"
    content_hash = hashlib.md5(raw.encode()).hexdigest()[:16]
    date_str = article["published_at"][:10]
    md = (
        f"---\n"
        f"source: news\n"
        f"doc_type: news\n"
        f"ts_code: {ts_code}\n"
        f"title: {_yaml_str(article['title'])}\n"
        f"published_at: '{article['published_at']}'\n"
        f"news_source: {_yaml_str(article['source_name'])}\n"
        f"url: {article['url']}\n"
        f"content_hash: {content_hash}\n"
        f"---\n\n"
        f"# {article['title']}\n\n"
        f"**来源**：{article['source_name']}　"
        f"**日期**：{date_str}　"
        f"**股票**：{ts_code}\n\n"
        f"{article['content']}\n"
    )
    return md, content_hash


def load_existing_hashes(markdown_dir: Path) -> set[str]:
    """Collect content_hashes from existing news markdown files."""
    hashes: set[str] = set()
    for path in markdown_dir.glob("news_*.md"):
        try:
            for line in path.read_text(encoding="utf-8", errors="ignore").splitlines()[:20]:
                if line.startswith("content_hash:"):
                    hashes.add(line.split(":", 1)[1].strip())
                    break
        except OSError:
            continue
    return hashes


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

def cleanup_old_news(markdown_dir: Path, retention_days: int = NEWS_RETENTION_DAYS) -> int:
    """Delete news_*.md files whose embedded date is older than retention_days."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)
    deleted = 0
    for path in markdown_dir.glob("news_*.md"):
        # filename format: news_XXXXXX_SZ_YYYYMMDD_hash.md
        parts = path.stem.split("_")
        try:
            date_str = next(p for p in parts if len(p) == 8 and p.isdigit())
            file_date = datetime.strptime(date_str, "%Y%m%d").replace(tzinfo=timezone.utc)
        except (StopIteration, ValueError):
            continue
        if file_date < cutoff:
            path.unlink()
            deleted += 1
    return deleted


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(days: int = 2, dry_run: bool = False, skip_index: bool = False) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    config = load_config(PROJECT_ROOT / "configs/sources.yaml")
    markdown_dir = resolve_project_path(
        PROJECT_ROOT / "configs/sources.yaml",
        config.get("project", {}).get("processed_markdown_dir", "data/processed/markdown"),
    )

    # Build watchlist
    tier1 = get_institutional_watchlist(markdown_dir)
    tier2 = get_hot_watchlist()
    watchlist = list(dict.fromkeys(tier1 + tier2))
    LOGGER.info("Combined watchlist: %d stocks", len(watchlist))

    # Cleanup old news before writing new
    if not dry_run:
        deleted = cleanup_old_news(markdown_dir)
        if deleted:
            LOGGER.info("Removed %d outdated news files (>%d days)", deleted, NEWS_RETENTION_DAYS)

    # Load existing hashes to skip duplicates
    existing_hashes = load_existing_hashes(markdown_dir)

    saved = skipped = errors = 0
    for ts_code in watchlist:
        articles = fetch_news_for_stock(ts_code, days=days)
        for article in articles:
            md, content_hash = article_to_markdown(article, ts_code)
            if content_hash in existing_hashes:
                skipped += 1
                continue
            date_str = article["published_at"][:10].replace("-", "")
            plain = ts_code.replace(".", "_")
            fname = f"news_{plain}_{date_str}_{content_hash}.md"
            if not dry_run:
                try:
                    (markdown_dir / fname).write_text(md, encoding="utf-8")
                    existing_hashes.add(content_hash)
                    saved += 1
                except OSError as exc:
                    LOGGER.error("Failed to write %s: %s", fname, exc)
                    errors += 1
            else:
                LOGGER.info("[dry-run] Would save: %s", fname)
                saved += 1

    LOGGER.info(
        "News fetch done: saved=%d skipped=%d errors=%d (dry_run=%s)",
        saved, skipped, errors, dry_run,
    )

    if not dry_run and saved > 0 and not skip_index:
        LOGGER.info("Triggering incremental index update...")
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts" / "update_index_incremental.py")],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            LOGGER.error("Index update failed:\n%s", result.stderr[-2000:])
        else:
            LOGGER.info("Index update complete.")
            if result.stdout:
                LOGGER.info(result.stdout[-1000:])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch daily stock news into the RAG knowledge base")
    parser.add_argument("--days", type=int, default=2, help="Fetch news from last N days (default: 2)")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without writing files")
    parser.add_argument("--skip-index", action="store_true", help="Skip FAISS index update after fetching")
    args = parser.parse_args()
    main(days=args.days, dry_run=args.dry_run, skip_index=args.skip_index)

