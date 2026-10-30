"""Run configured crawlers for policies and exchange announcements."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.crawler.base_crawler import load_yaml, setup_logging
from src.crawler.exchange_announcement_crawler import ExchangeAnnouncementCrawler
from src.crawler.generic_policy_crawler import CSRCCrawler, MIITCrawler
from src.crawler.gov_policy_crawler import GovPolicyCrawler
from src.crawler.ndrc_crawler import NDRCCrawler


LOGGER = logging.getLogger(__name__)


CRAWLER_REGISTRY = {
    "gov_policy": GovPolicyCrawler,
    "ndrc": NDRCCrawler,
    "csrc": CSRCCrawler,
    "miit": MIITCrawler,
    "exchange_announcements": ExchangeAnnouncementCrawler,
}


def main() -> None:
    """Run one or more crawlers from YAML config."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sources.yaml")
    parser.add_argument(
        "--sources",
        nargs="*",
        default=["gov_policy", "ndrc", "csrc", "miit", "exchange_announcements"],
        choices=sorted(CRAWLER_REGISTRY),
    )
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)
    config_path = PROJECT_ROOT / args.config
    config = load_yaml(config_path)
    setup_logging(PROJECT_ROOT / config.get("project", {}).get("logs_dir", "logs"))

    total = 0
    for source_name in args.sources:
        crawler_cls = CRAWLER_REGISTRY[source_name]
        LOGGER.info("Starting crawler: %s", source_name)
        crawler = crawler_cls(config)
        results = crawler.crawl()
        total += len(results)
        LOGGER.info("Crawler %s saved %d primary documents", source_name, len(results))

    LOGGER.info("Crawl finished. Primary documents saved: %d", total)


if __name__ == "__main__":
    main()

