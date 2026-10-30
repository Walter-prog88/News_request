"""HTML to standardized Markdown parser."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup

from src.parser.markdown_cleaner import content_hash, write_markdown


LOGGER = logging.getLogger(__name__)


def load_metadata(metadata_path: str | Path) -> dict[str, Any]:
    """Load crawler sidecar metadata JSON."""
    with Path(metadata_path).open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_html_text(html: str) -> tuple[str, str]:
    """Extract title and readable text from HTML."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "iframe", "nav", "footer"]):
        tag.decompose()

    title = ""
    if soup.find("h1"):
        title = soup.find("h1").get_text(" ", strip=True)
    elif soup.title and soup.title.string:
        title = soup.title.string.strip()

    main = (
        soup.find("article")
        or soup.find("main")
        or soup.find("div", class_="article")
        or soup.find("div", class_="content")
        or soup.body
        or soup
    )

    lines: list[str] = []
    for element in main.find_all(["h1", "h2", "h3", "p", "li", "td"], recursive=True):
        text = element.get_text(" ", strip=True)
        if not text:
            continue
        if element.name == "h1":
            lines.append(f"# {text}")
        elif element.name == "h2":
            lines.append(f"## {text}")
        elif element.name == "h3":
            lines.append(f"### {text}")
        elif element.name == "li":
            lines.append(f"- {text}")
        else:
            lines.append(text)

    if not lines:
        lines = [main.get_text("\n", strip=True)]
    return title, "\n\n".join(lines)


def parse_html_file(
    raw_path: str | Path,
    metadata: dict[str, Any],
    output_dir: str | Path,
) -> Path | None:
    """Parse one raw HTML file and write standardized Markdown."""
    raw_path = Path(raw_path)
    html = raw_path.read_text(encoding="utf-8", errors="ignore")
    extracted_title, body = extract_html_text(html)
    if not body.strip():
        LOGGER.warning("No text extracted from HTML: %s", raw_path)
        return None

    title = metadata.get("title") or extracted_title or raw_path.stem
    digest = content_hash(body)
    output_path = Path(output_dir) / f"{metadata.get('source', 'html')}_{digest[:12]}.md"
    front_matter = {
        "source": metadata.get("source"),
        "title": title,
        "url": metadata.get("url"),
        "published_at": metadata.get("published_at"),
        "fetched_at": metadata.get("fetched_at"),
        "content_type": metadata.get("content_type") or "text/html",
        "raw_path": str(raw_path),
        "raw_sha256": metadata.get("sha256"),
        "parser": "html_parser",
        "parsed_at": datetime.now(timezone.utc).isoformat(),
        **(metadata.get("extra") or {}),
    }
    return write_markdown(output_path, front_matter, body)

