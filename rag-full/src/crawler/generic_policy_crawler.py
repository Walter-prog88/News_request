"""Generic crawler for public policy list pages.

Used for sources whose list pages expose ordinary HTML links, such as CSRC and
MIIT policy/announcement pages.
"""

from __future__ import annotations

import ast
import json
import logging
import re
from typing import Any

from bs4 import BeautifulSoup

from src.crawler.base_crawler import BaseCrawler, CrawlResult, unique_by_url


LOGGER = logging.getLogger(__name__)


class GenericPolicyCrawler(BaseCrawler):
    """Collect policy documents from configured static list pages."""

    def extract_candidates(self, html: str, base_url: str) -> list[dict[str, str]]:
        """Extract candidate document links from one list page."""
        soup = BeautifulSoup(html, "html.parser")
        candidates: list[dict[str, str]] = []
        url_patterns = [
            re.compile(pattern)
            for pattern in self.source_config.get("url_patterns", [])
        ]
        exclude_url_patterns = [
            re.compile(pattern)
            for pattern in self.source_config.get("exclude_url_patterns", [])
        ]

        for link in soup.find_all("a"):
            title = link.get_text(" ", strip=True)
            url = self.normalize_url(base_url, link.get("href"))
            if not url or not self.allowed_url(url):
                continue
            if not self.keyword_allowed(f"{title} {url}"):
                continue
            if url_patterns and not any(pattern.search(url) for pattern in url_patterns):
                continue
            if exclude_url_patterns and any(pattern.search(url) for pattern in exclude_url_patterns):
                continue

            candidates.append(
                {
                    "title": title or url,
                    "url": url,
                    "published_at": self._extract_date(title, url, link),
                }
            )
        return unique_by_url(candidates)

    def extract_dynamic_units(self, html: str, base_url: str) -> list[dict[str, Any]]:
        """Extract dynamic list-building API calls embedded in script tags."""
        soup = BeautifulSoup(html, "html.parser")
        units: list[dict[str, Any]] = []
        for script in soup.find_all("script"):
            src = self.normalize_url(base_url, script.get("url") or script.get("src"))
            query_data = script.get("querydata") or script.get("queryData")
            if not src or not query_data or "/api-gateway/" not in src:
                continue
            try:
                params = ast.literal_eval(query_data)
            except Exception:
                LOGGER.debug("Unable to parse dynamic queryData for %s", src)
                continue
            if isinstance(params, dict):
                units.append({"url": src, "params": params})
        return units

    @staticmethod
    def _extract_date(title: str, url: str, link: Any) -> str | None:
        """Extract date from title, URL, or surrounding list item text."""
        haystacks = [title, url]
        if link.parent:
            haystacks.append(link.parent.get_text(" ", strip=True))
        for value in haystacks:
            match = re.search(r"(20\d{2})[-年./]?(\d{1,2})[-月./]?(\d{1,2})", value)
            if match:
                year, month, day = match.groups()
                return f"{year}-{int(month):02d}-{int(day):02d}"
        return None

    def crawl(self) -> list[CrawlResult]:
        """Crawl configured list pages and linked policy documents."""
        if not self.source_config.get("enabled", True):
            LOGGER.info("%s source disabled", self.source_key)
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
                    title=f"{self.source_name} list page",
                    extra={"document_role": "list_page"},
                )
                response.encoding = response.apparent_encoding or response.encoding
                html = response.text
                candidates.extend(self.extract_candidates(html, list_url))
                for unit in self.extract_dynamic_units(html, list_url):
                    try:
                        api_response = self.request(unit["url"], params=unit["params"])
                        self.save_response(
                            api_response,
                            source=self.source_name,
                            url=api_response.url,
                            title=f"{self.source_name} dynamic list unit",
                            extra={"document_role": "api_response"},
                            extension=".json",
                        )
                        payload = json.loads(api_response.content.decode("utf-8", errors="ignore"))
                        unit_html = ((payload.get("data") or {}).get("html") or "")
                        candidates.extend(self.extract_candidates(unit_html, list_url))
                    except Exception as exc:
                        LOGGER.warning("Failed to crawl %s dynamic unit %s: %s", self.source_key, unit["url"], exc)
            except Exception as exc:
                LOGGER.warning("Failed to crawl %s list page %s: %s", self.source_key, list_url, exc)

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
                LOGGER.warning("Failed to download %s document %s: %s", self.source_key, candidate["url"], exc)

        LOGGER.info("%s crawler saved %d documents", self.source_key, len(results))
        return results


class CSRCCrawler(GenericPolicyCrawler):
    """Collect China Securities Regulatory Commission policy documents."""

    def __init__(self, config: dict[str, Any]):
        super().__init__(config, "csrc")


class MIITCrawler(GenericPolicyCrawler):
    """Collect Ministry of Industry and Information Technology policy documents."""

    def __init__(self, config: dict[str, Any]):
        super().__init__(config, "miit")

