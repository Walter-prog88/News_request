"""Crawler for SSE/SZSE listed-company announcements.

This module uses public exchange announcement endpoints where available and
keeps page size small by default. It saves both API JSON responses and linked
announcement PDFs to ``data/raw``.
"""

from __future__ import annotations

import json
import logging
import random
import re
import time
from typing import Any
from urllib.parse import urljoin

from src.crawler.base_crawler import BaseCrawler, CrawlResult


LOGGER = logging.getLogger(__name__)


class ExchangeAnnouncementCrawler(BaseCrawler):
    """Collect a small sample of SSE and SZSE listed-company announcements."""

    def __init__(self, config: dict[str, Any]):
        super().__init__(config, "exchange_announcements")

    @staticmethod
    def _loads_json_or_jsonp(text: str) -> dict[str, Any]:
        """Parse JSON or simple JSONP text."""
        stripped = text.strip()
        if stripped.startswith("{") or stripped.startswith("["):
            return json.loads(stripped)
        match = re.search(r"^[^(]+\((.*)\)\s*;?$", stripped, flags=re.S)
        if match:
            return json.loads(match.group(1))
        raise ValueError("response is neither JSON nor JSONP")

    def crawl_sse(self) -> list[CrawlResult]:
        """Crawl Shanghai Stock Exchange announcement PDFs via JSON endpoint."""
        sse_config = self.source_config.get("sse", {})
        if not sse_config.get("enabled", False):
            return []

        results: list[CrawlResult] = []
        api_url = sse_config["api_url"]
        max_pages = int(self.source_config.get("max_pages", 2))
        page_size = int(self.source_config.get("page_size", 5))
        max_docs = int(self.source_config.get("max_docs_per_source", self.max_docs_per_source))
        headers = {"Referer": sse_config.get("referer", "https://www.sse.com.cn/")}

        for stock_code in sse_config.get("stock_codes", []):
            for page_no in range(1, max_pages + 1):
                params = {
                    "isPagination": "true",
                    "pageHelp.pageSize": page_size,
                    "pageHelp.pageNo": page_no,
                    "pageHelp.beginPage": page_no,
                    "pageHelp.endPage": page_no,
                    "productId": stock_code,
                    "securityType": sse_config.get("security_type", ""),
                    "reportType": sse_config.get("report_type", "ALL"),
                    "beginDate": self.source_config.get("begin_date", ""),
                    "endDate": self.source_config.get("end_date", ""),
                    "_": int(time.time() * 1000),
                }
                try:
                    response = self.request(api_url, params=params, headers=headers)
                    self.save_response(
                        response,
                        source="sse_announcement_api",
                        url=response.url,
                        title=f"SSE announcement API {stock_code} page {page_no}",
                        extra={"document_role": "api_response", "stock_code": stock_code},
                        extension=".json",
                    )
                    payload = self._loads_json_or_jsonp(response.text)
                    rows = payload.get("result", []) or payload.get("data", [])
                    for row in rows:
                        if len(results) >= max_docs:
                            return results
                        pdf_path = row.get("URL") or row.get("url")
                        if not pdf_path:
                            continue
                        pdf_url = urljoin("https://www.sse.com.cn", pdf_path)
                        pdf_response = self.request(pdf_url, headers=headers)
                        results.append(
                            self.save_response(
                                pdf_response,
                                source="sse_announcement",
                                url=pdf_url,
                                title=row.get("TITLE") or row.get("title"),
                                published_at=row.get("SSEDATE") or row.get("BULLETIN_YEAR"),
                                extra={
                                    "exchange": "SSE",
                                    "stock_code": row.get("SECURITY_CODE") or stock_code,
                                    "stock_name": row.get("SECURITY_NAME"),
                                    "announcement_type": row.get("BULLETIN_TYPE"),
                                },
                                extension=".pdf",
                            )
                        )
                except Exception as exc:
                    LOGGER.warning("Failed to crawl SSE %s page %s: %s", stock_code, page_no, exc)
        return results

    def crawl_szse(self) -> list[CrawlResult]:
        """Crawl Shenzhen Stock Exchange announcement PDFs via JSON endpoint."""
        szse_config = self.source_config.get("szse", {})
        if not szse_config.get("enabled", False):
            return []

        results: list[CrawlResult] = []
        api_url = szse_config["api_url"]
        max_pages = int(self.source_config.get("max_pages", 2))
        page_size = int(self.source_config.get("page_size", 5))
        max_docs = int(self.source_config.get("max_docs_per_source", self.max_docs_per_source))
        headers = {
            "Referer": szse_config.get("referer", "https://www.szse.cn/"),
            "Content-Type": "application/json",
        }

        for stock_code in szse_config.get("stock_codes", []):
            for page_no in range(1, max_pages + 1):
                payload = {
                    "seDate": [
                        self.source_config.get("begin_date", ""),
                        self.source_config.get("end_date", ""),
                    ],
                    "stock": [stock_code],
                    "channelCode": szse_config.get("channel_code", ["fixed_disc"]),
                    "pageSize": page_size,
                    "pageNum": page_no,
                }
                params = {"random": random.random()}
                try:
                    response = self.request(
                        api_url,
                        method="POST",
                        params=params,
                        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                        headers=headers,
                    )
                    self.save_response(
                        response,
                        source="szse_announcement_api",
                        url=response.url,
                        title=f"SZSE announcement API {stock_code} page {page_no}",
                        extra={"document_role": "api_response", "stock_code": stock_code},
                        extension=".json",
                    )
                    rows = response.json().get("data", [])
                    for row in rows:
                        if len(results) >= max_docs:
                            return results
                        pdf_url = self._szse_pdf_url(row, szse_config)
                        if not pdf_url:
                            continue
                        pdf_response = self.request(pdf_url, headers={"Referer": headers["Referer"]})
                        results.append(
                            self.save_response(
                                pdf_response,
                                source="szse_announcement",
                                url=pdf_url,
                                title=row.get("title") or row.get("announcementTitle"),
                                published_at=row.get("publishTime") or row.get("publishDate"),
                                extra={
                                    "exchange": "SZSE",
                                    "stock_code": row.get("secCode") or stock_code,
                                    "stock_name": row.get("secName"),
                                    "announcement_type": row.get("channelName"),
                                },
                                extension=".pdf",
                            )
                        )
                except Exception as exc:
                    LOGGER.warning("Failed to crawl SZSE %s page %s: %s", stock_code, page_no, exc)
        return results

    @staticmethod
    def _szse_pdf_url(row: dict[str, Any], szse_config: dict[str, Any]) -> str | None:
        """Resolve SZSE PDF URL from known announcement row fields."""
        for key in ("attachPath", "adjunctUrl", "url"):
            value = row.get(key)
            if not value:
                continue
            if value.startswith("http"):
                return value
            if value.startswith("/"):
                return szse_config.get("download_base_url", "https://disc.static.szse.cn/download") + value
            return urljoin("https://www.szse.cn/", value)
        return None

    def crawl(self) -> list[CrawlResult]:
        """Run SSE and SZSE crawlers according to config."""
        if not self.source_config.get("enabled", True):
            LOGGER.info("exchange_announcements source disabled")
            return []

        results: list[CrawlResult] = []
        results.extend(self.crawl_sse())
        results.extend(self.crawl_szse())
        LOGGER.info("Exchange announcement crawler saved %d PDFs", len(results))
        return results

