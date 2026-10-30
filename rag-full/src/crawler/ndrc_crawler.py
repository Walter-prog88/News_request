"""Crawler for National Development and Reform Commission policy documents.

The crawler works with static NDRC list pages. It keeps collection intentionally
small by respecting ``max_pages`` and ``max_docs_per_source`` in the YAML config.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from bs4 import BeautifulSoup

from src.crawler.base_crawler import BaseCrawler, CrawlResult, unique_by_url


LOGGER = logging.getLogger(__name__)


class NDRCCrawler(BaseCrawler):
    """Collect NDRC policy files and notices."""

    def __init__(self, config: dict[str, Any]):
        super().__init__(config, "ndrc")

    def extract_candidates(self, html: str, base_url: str) -> list[dict[str, str]]:
        """Extract candidate NDRC document links from one list page."""
        soup = BeautifulSoup(html, "html.parser")
        candidates: list[dict[str, str]] = []
        for link in soup.find_all("a"):
            title = link.get_text(" ", strip=True)
            url = self.normalize_url(base_url, link.get("href"))
            if not url or not self.allowed_url(url):
                continue
            if not self.keyword_allowed(title):
                continue
            if not re.search(r"/t20\d{6}_\d+\.html$", url):
                continue
            if not ("/xxgk/" in url or "/zcfb/" in url or url.lower().endswith(".pdf")):
                continue
            candidates.append(
                {
                    "title": title or url,
                    "url": url,
                    "published_at": self._extract_date(title, url, link),
                }
            )
        return unique_by_url(candidates)

    @staticmethod
    def _extract_date(title: str, url: str, link: Any) -> str | None:
        """Extract date from title, URL, or surrounding list item text."""
        haystacks = [title, url]
        if link.parent:
            haystacks.append(link.parent.get_text(" ", strip=True))
        for value in haystacks:
            match = re.search(r"(20\d{2})[-年./]?(\d{2})[-月./]?(\d{2})", value)
            if match:
                year, month, day = match.groups()
                return f"{year}-{month}-{day}"
        return None

    def crawl(self) -> list[CrawlResult]:
        """Crawl configured NDRC list pages and linked documents."""
        if not self.source_config.get("enabled", True):
            LOGGER.info("ndrc source disabled")
            return []

        list_urls = self.source_config.get("list_urls", [])
        max_pages = int(self.request_config.get("max_pages", 3))
        max_docs = int(self.source_config.get("max_docs_per_source", self.max_docs_per_source))
        candidates: list[dict[str, str]] = []
        results: list[CrawlResult] = []

        for list_url in list_urls[:max_pages]:
            try:
                response = self.request(list_url)
                self.save_response(
                    response,
                    source=self.source_name,
                    url=list_url,
                    title="NDRC list page",
                    extra={"document_role": "list_page"},
                )
                html = response.content.decode("utf-8", errors="ignore")
                candidates.extend(self.extract_candidates(html, list_url))
            except Exception as exc:
                LOGGER.warning("Failed to crawl NDRC list page %s: %s", list_url, exc)

        for candidate in unique_by_url(candidates)[:max_docs]:
            try:
                response = self.request(candidate["url"])
                results.append(
                    self.save_response(
                        response,
                        source=self.source_name,
                        url=candidate["url"],
                        title=candidate.get("title"),
                        published_at=candidate.get("published_at"),
                        extra={"document_role": "policy_document"},
                    )
                )
            except Exception as exc:
                LOGGER.warning("Failed to download NDRC document %s: %s", candidate["url"], exc)

        LOGGER.info("NDRC crawler saved %d documents", len(results))
        return results

