"""Convert raw HTML/PDF files and local notes into standardized Markdown."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.crawler.base_crawler import load_yaml, setup_logging
from src.parser.html_parser import parse_html_file
from src.parser.markdown_cleaner import deduplicate_markdown_dir, standardize_local_markdown
from src.parser.pdf_parser import parse_pdf_file


LOGGER = logging.getLogger(__name__)


def resolve_path(value: str | Path) -> Path:
    """Resolve project-relative paths."""
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def should_parse(metadata: dict, include_list_pages: bool) -> bool:
    """Return whether a raw file should enter the RAG corpus."""
    role = (metadata.get("extra") or {}).get("document_role")
    if role in {"api_response"}:
        return False
    if role == "list_page" and not include_list_pages:
        return False
    return True


def convert_raw_documents(config: dict, include_list_pages: bool = False) -> int:
    """Convert raw HTML/PDF documents referenced by metadata sidecars."""
    project_config = config.get("project", {})
    raw_dir = resolve_path(project_config.get("raw_dir", "data/raw"))
    output_dir = resolve_path(project_config.get("processed_markdown_dir", "data/processed/markdown"))
    converted = 0

    for meta_path in sorted(raw_dir.rglob("*.meta.json")):
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        if not should_parse(metadata, include_list_pages):
            continue

        raw_path = resolve_path(metadata["raw_path"])
        if not raw_path.exists():
            LOGGER.warning("Raw file missing for metadata %s: %s", meta_path, raw_path)
            continue

        content_type = (metadata.get("content_type") or "").lower()
        suffix = raw_path.suffix.lower()
        try:
            if suffix == ".pdf" or "pdf" in content_type:
                if parse_pdf_file(raw_path, metadata, output_dir):
                    converted += 1
            elif suffix in {".html", ".htm"} or "html" in content_type:
                if parse_html_file(raw_path, metadata, output_dir):
                    converted += 1
            else:
                LOGGER.info("Skipping unsupported raw file: %s (%s)", raw_path, content_type)
        except Exception as exc:
            LOGGER.warning("Failed to parse %s: %s", raw_path, exc)
    return converted


def convert_local_notes(config: dict) -> int:
    """Standardize configured local Markdown research notes."""
    local_config = config.get("local_notes", {})
    if not local_config.get("enabled", False):
        return 0

    output_dir = resolve_path(config.get("project", {}).get("processed_markdown_dir", "data/processed/markdown"))
    converted = 0
    for path_value in local_config.get("paths", []):
        path = resolve_path(path_value)
        if standardize_local_markdown(path, output_dir, source_name="local_note"):
            converted += 1
    return converted


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sources.yaml")
    parser.add_argument("--include-list-pages", action="store_true")
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)
    config = load_yaml(PROJECT_ROOT / args.config)
    setup_logging(PROJECT_ROOT / config.get("project", {}).get("logs_dir", "logs"))

    raw_count = convert_raw_documents(config, include_list_pages=args.include_list_pages)
    local_count = convert_local_notes(config)
    manifest = deduplicate_markdown_dir(
        resolve_path(config.get("project", {}).get("processed_markdown_dir", "data/processed/markdown")),
        resolve_path(config.get("project", {}).get("duplicate_dir", "data/processed/duplicates")),
    )
    LOGGER.info(
        "Markdown conversion finished. raw=%d local=%d unique=%d duplicates=%d",
        raw_count,
        local_count,
        manifest["unique_documents"],
        len(manifest["duplicates"]),
    )


if __name__ == "__main__":
    main()

