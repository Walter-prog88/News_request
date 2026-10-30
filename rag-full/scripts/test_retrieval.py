"""Run a simple FAISS retrieval smoke test."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.crawler.base_crawler import load_yaml, setup_logging
from src.indexer.retriever import print_results, retrieve


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sources.yaml")
    parser.add_argument("--query", default="设备更新 政策 金融支持")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)
    config = load_yaml(PROJECT_ROOT / args.config)
    setup_logging(PROJECT_ROOT / config.get("project", {}).get("logs_dir", "logs"))
    results = retrieve(args.query, config_path=PROJECT_ROOT / args.config, top_k=args.top_k)
    print_results(results)


if __name__ == "__main__":
    main()

