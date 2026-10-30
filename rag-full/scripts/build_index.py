"""Build FAISS vector index from processed Markdown."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.crawler.base_crawler import load_yaml, setup_logging
from src.indexer.build_vectorstore import build_vectorstore


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sources.yaml")
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)
    config = load_yaml(PROJECT_ROOT / args.config)
    setup_logging(PROJECT_ROOT / config.get("project", {}).get("logs_dir", "logs"))
    index_dir = build_vectorstore(PROJECT_ROOT / args.config)
    logging.getLogger(__name__).info("Index built at %s", index_dir)


if __name__ == "__main__":
    main()

