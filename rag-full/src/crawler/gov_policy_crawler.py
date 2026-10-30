"""Crawler for China government policy pages on gov.cn.

The crawler downloads a small number of list pages, extracts candidate policy
links, and saves the linked HTML/PDF documents to ``data/raw``.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from bs4 import BeautifulSoup

from src.crawler.base_crawler import BaseCrawler, CrawlResult, unique_by_url


LOGGER = logging.getLogger(__name__)


class GovPolicyCrawler(BaseCrawler):
    """Collect recent policy files from gov.cn list pages."""

    def __init__(self, config: dict[str, Any]):
        super().__init__(config, "gov_policy")

    def extract_candidates(self, html: str, base_url: str) -> list[dict[str, str]]:
        """Extract policy candidate links from one list page."""
        soup = BeautifulSoup(html, "html.parser")
        candidates: list[dict[str, str]] = []
        for link in soup.find_all("a"):
            text = link.get_text(" ", strip=True)
            url = self.normalize_url(base_url, link.get("href"))
            if not url or not self.allowed_url(url):
                continue
            if not self.keyword_allowed(text):
                continue
            if not ("/zhengce/" in url or url.lower().endswith(".pdf")):
                continue
            published_at = self._extract_date_from_context(link)
            candidates.append({"title": text or url, "url": url, "published_at": published_at})
        return unique_by_url(candidates)

    @staticmethod
    def _extract_date_from_context(link: Any) -> str | None:
        """Extract a date string near a link when list pages expose one."""
        context = link.parent.get_text(" ", strip=True) if link.parent else ""
        match = re.search(r"(20\d{2})[-年./](\d{1,2})[-月./](\d{1,2})", context)
        if not match:
            return None
        year, month, day = match.groups()
        return f"{year}-{int(month):02d}-{int(day):02d}"

    def crawl(self) -> list[CrawlResult]:
        """Crawl configured gov.cn policy list pages and linked documents."""
        if not self.source_config.get("enabled", True):
            LOGGER.info("gov_policy source disabled")
            return []

        list_urls = self.source_config.get("list_urls", [])
        max_pages = int(self.request_config.get("max_pages", 3))
        max_docs = int(self.source_config.get("max_docs_per_source", self.max_docs_per_source))
        results: list[CrawlResult] = []
        candidates: list[dict[str, str]] = []

        for list_url in list_urls[:max_pages]:
            try:
                response = self.request(list_url)
                self.save_response(
                    response,
                    source=self.source_name,
                    url=list_url,
                    title="gov.cn list page",
                    extra={"document_role": "list_page"},
                )
                html = response.content.decode("utf-8", errors="ignore")
                candidates.extend(self.extract_candidates(html, list_url))
            except Exception as exc:
                LOGGER.warning("Failed to crawl gov.cn list page %s: %s", list_url, exc)

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
                LOGGER.warning("Failed to download gov.cn document %s: %s", candidate["url"], exc)

        LOGGER.info("gov_policy crawler saved %d documents", len(results))
        return results

