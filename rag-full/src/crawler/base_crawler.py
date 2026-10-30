"""Base crawler utilities for polite public-source collection.

This module provides shared request, delay, raw-file persistence, and metadata
sidecar logic. Source-specific crawlers should subclass ``BaseCrawler`` and
return metadata dictionaries produced by ``save_response``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import mimetypes
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urljoin, urlparse

import requests
import yaml


LOGGER = logging.getLogger(__name__)


@dataclass
class CrawlResult:
    """Metadata for a raw crawled document."""

    source: str
    url: str
    raw_path: str
    metadata_path: str
    title: str | None = None
    published_at: str | None = None
    content_type: str | None = None
    sha256: str | None = None
    fetched_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable metadata dictionary."""
        return {
            "source": self.source,
            "url": self.url,
            "raw_path": self.raw_path,
            "metadata_path": self.metadata_path,
            "title": self.title,
            "published_at": self.published_at,
            "content_type": self.content_type,
            "sha256": self.sha256,
            "fetched_at": self.fetched_at,
            "extra": self.extra,
        }


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file."""
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_logging(logs_dir: str | Path = "logs", level: int = logging.INFO) -> None:
    """Configure console and file logging for scripts."""
    Path(logs_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(logs_dir) / "pipeline.log"
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path, encoding="utf-8"),
        ],
    )


def sha256_bytes(data: bytes) -> str:
    """Return SHA-256 hex digest for bytes."""
    return hashlib.sha256(data).hexdigest()


def safe_filename(value: str, max_len: int = 80) -> str:
    """Convert arbitrary text into a filesystem-safe ASCII-ish filename."""
    value = re.sub(r"https?://", "", value)
    value = re.sub(r"[^\w\-.]+", "_", value, flags=re.UNICODE)
    value = value.strip("._")
    return value[:max_len] or "document"


def guess_extension(url: str, content_type: str | None) -> str:
    """Infer a file extension from URL and content type."""
    path_suffix = Path(urlparse(url).path).suffix.lower()
    if path_suffix in {".html", ".htm", ".pdf", ".json", ".txt", ".md"}:
        return path_suffix
    if content_type:
        mime = content_type.split(";")[0].strip().lower()
        guessed = mimetypes.guess_extension(mime)
        if guessed:
            if guessed == ".jpe":
                return ".jpg"
            return guessed
        if "pdf" in mime:
            return ".pdf"
        if "html" in mime:
            return ".html"
        if "json" in mime:
            return ".json"
    return ".bin"


class BaseCrawler:
    """Base class for source-specific crawlers.

    Parameters
    ----------
    config:
        Whole project config loaded from ``configs/sources.yaml``.
    source_key:
        Key under ``sources`` for the current crawler.
    """

    def __init__(self, config: dict[str, Any], source_key: str):
        self.config = config
        self.source_key = source_key
        self.source_config = config.get("sources", {}).get(source_key, {})
        self.project_config = config.get("project", {})
        self.request_config = config.get("request", {})
        self.source_name = self.source_config.get("source_name", source_key)
        self.raw_dir = Path(self.project_config.get("raw_dir", "data/raw"))
        self.raw_dir.mkdir(parents=True, exist_ok=True)

        self.delay_seconds = float(self.request_config.get("delay_seconds", 1.5))
        self.timeout_seconds = int(self.request_config.get("timeout_seconds", 25))
        self.max_docs_per_source = int(self.request_config.get("max_docs_per_source", 5))
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": self.request_config.get(
                    "user_agent", "Fin-RAG-Knowledge-Base/0.1"
                ),
                "Accept": "text/html,application/pdf,application/json;q=0.9,*/*;q=0.8",
            }
        )

    def sleep(self) -> None:
        """Sleep between requests to avoid high-frequency access."""
        if self.delay_seconds > 0:
            time.sleep(self.delay_seconds)

    def request(self, url: str, method: str = "GET", **kwargs: Any) -> requests.Response:
        """Perform a delayed HTTP request and raise for non-2xx responses."""
        self.sleep()
        LOGGER.info("Fetching %s %s", method, url)
        response = self.session.request(
            method,
            url,
            timeout=self.timeout_seconds,
            **kwargs,
        )
        response.raise_for_status()
        return response

    def allowed_url(self, url: str) -> bool:
        """Return True when URL belongs to one of the configured domains."""
        domains = set(self.source_config.get("allowed_domains", []))
        if not domains:
            return True
        netloc = urlparse(url).netloc.lower()
        return any(netloc == domain or netloc.endswith("." + domain) for domain in domains)

    def keyword_allowed(self, text: str) -> bool:
        """Apply optional include/exclude keyword filters."""
        include_keywords = self.source_config.get("include_keywords", [])
        exclude_keywords = self.source_config.get("exclude_keywords", [])
        if exclude_keywords and any(k in text for k in exclude_keywords):
            return False
        if include_keywords and not any(k in text for k in include_keywords):
            return False
        return True

    def normalize_url(self, base_url: str, href: str | None) -> str | None:
        """Resolve a link and discard unsupported schemes."""
        if not href:
            return None
        href = href.strip()
        if href.startswith(("javascript:", "mailto:", "#")):
            return None
        return urljoin(base_url, href)

    def save_response(
        self,
        response: requests.Response,
        *,
        source: str | None = None,
        url: str | None = None,
        title: str | None = None,
        published_at: str | None = None,
        extra: dict[str, Any] | None = None,
        extension: str | None = None,
    ) -> CrawlResult:
        """Persist raw response bytes and a JSON metadata sidecar."""
        source = source or self.source_name
        url = url or response.url
        content = response.content
        digest = sha256_bytes(content)
        content_type = response.headers.get("Content-Type")
        ext = extension or guess_extension(url, content_type)
        source_dir = self.raw_dir / safe_filename(source)
        source_dir.mkdir(parents=True, exist_ok=True)

        stem_parts = [
            safe_filename(source, 32),
            datetime.now().strftime("%Y%m%d_%H%M%S"),
            digest[:12],
        ]
        stem = "_".join(stem_parts)
        raw_path = source_dir / f"{stem}{ext}"
        meta_path = source_dir / f"{stem}.meta.json"

        if not raw_path.exists():
            raw_path.write_bytes(content)

        result = CrawlResult(
            source=source,
            url=url,
            raw_path=str(raw_path),
            metadata_path=str(meta_path),
            title=title,
            published_at=published_at,
            content_type=content_type,
            sha256=digest,
            extra=extra or {},
        )
        meta_path.write_text(
            json.dumps(result.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self.append_manifest(result)
        LOGGER.info("Saved raw document: %s", raw_path)
        return result

    def save_bytes(
        self,
        content: bytes,
        *,
        source: str | None,
        url: str,
        title: str | None = None,
        published_at: str | None = None,
        content_type: str | None = None,
        extra: dict[str, Any] | None = None,
        extension: str | None = None,
    ) -> CrawlResult:
        """Persist arbitrary bytes, useful for JSON API payloads."""
        class _ResponseLike:
            def __init__(self, content: bytes, url: str, content_type: str | None):
                self.content = content
                self.url = url
                self.headers = {"Content-Type": content_type or "application/octet-stream"}

        return self.save_response(
            _ResponseLike(content, url, content_type),  # type: ignore[arg-type]
            source=source,
            url=url,
            title=title,
            published_at=published_at,
            extra=extra,
            extension=extension,
        )

    def append_manifest(self, result: CrawlResult) -> None:
        """Append one JSONL record to the global raw manifest."""
        manifest_path = self.raw_dir / "manifest.jsonl"
        with manifest_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")

    def crawl(self) -> list[CrawlResult]:
        """Run crawler. Subclasses must implement this method."""
        raise NotImplementedError


def unique_by_url(items: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deduplicate dictionaries by their ``url`` field while preserving order."""
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for item in items:
        url = item.get("url")
        if not url or url in seen:
            continue
        seen.add(url)
        unique.append(item)
    return unique

