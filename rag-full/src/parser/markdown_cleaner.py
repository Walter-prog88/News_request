"""Markdown normalization, YAML front matter, and hash deduplication.

The parser modules call ``write_markdown`` so every processed document has
standard YAML metadata. This module can also standardize local research notes.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


LOGGER = logging.getLogger(__name__)
FRONT_MATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n(.*)$", flags=re.S)


def content_hash(text: str) -> str:
    """Return SHA-256 hash for normalized text content."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def clean_markdown_text(text: str) -> str:
    """Clean extracted text while preserving paragraph boundaries."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[\t\f\v]+", " ", text)
    text = re.sub(r"[ \u3000]{2,}", " ", text)
    lines = [line.strip() for line in text.split("\n")]

    cleaned_lines: list[str] = []
    blank_seen = False
    for line in lines:
        if not line:
            if not blank_seen:
                cleaned_lines.append("")
            blank_seen = True
            continue
        blank_seen = False
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines).strip() + "\n"


def split_front_matter(markdown_text: str) -> tuple[dict[str, Any], str]:
    """Split Markdown into metadata and body."""
    match = FRONT_MATTER_RE.match(markdown_text)
    if not match:
        return {}, markdown_text
    metadata = yaml.safe_load(match.group(1)) or {}
    return metadata, match.group(2)


def markdown_with_front_matter(metadata: dict[str, Any], body: str) -> str:
    """Build a Markdown document with YAML front matter."""
    metadata = dict(metadata)
    body = clean_markdown_text(body)
    metadata.setdefault("content_hash", content_hash(body))
    metadata.setdefault("processed_at", datetime.now(timezone.utc).isoformat())
    yaml_text = yaml.safe_dump(
        metadata,
        allow_unicode=True,
        sort_keys=False,
        default_flow_style=False,
    ).strip()
    return f"---\n{yaml_text}\n---\n\n{body}"


def write_markdown(output_path: str | Path, metadata: dict[str, Any], body: str) -> Path:
    """Write standardized Markdown with front matter."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown_with_front_matter(metadata, body), encoding="utf-8")
    LOGGER.info("Wrote markdown: %s", output_path)
    return output_path


def standardize_local_markdown(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    source_name: str = "local_note",
) -> Path | None:
    """Convert a local Markdown note into the project Markdown schema."""
    input_path = Path(input_path)
    if not input_path.exists():
        LOGGER.warning("Local note does not exist: %s", input_path)
        return None

    raw_text = input_path.read_text(encoding="utf-8", errors="ignore")
    existing_metadata, body = split_front_matter(raw_text)
    title = existing_metadata.get("title") or input_path.stem.replace("_", " ")
    metadata = {
        **existing_metadata,
        "source": source_name,
        "title": title,
        "url": None,
        "published_at": existing_metadata.get("published_at"),
        "content_type": "text/markdown",
        "raw_path": str(input_path),
        "parser": "markdown_cleaner",
    }
    digest = content_hash(clean_markdown_text(body))
    output_path = Path(output_dir) / f"{source_name}_{digest[:12]}.md"
    return write_markdown(output_path, metadata, body)


def deduplicate_markdown_dir(
    markdown_dir: str | Path,
    duplicate_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Move exact duplicate Markdown documents aside based on body hash."""
    markdown_dir = Path(markdown_dir)
    duplicate_dir = Path(duplicate_dir) if duplicate_dir else markdown_dir / "_duplicates"
    duplicate_dir.mkdir(parents=True, exist_ok=True)

    seen: dict[str, str] = {}
    duplicates: list[dict[str, str]] = []

    for path in sorted(markdown_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8", errors="ignore")
        metadata, body = split_front_matter(text)
        digest = metadata.get("content_hash") or content_hash(clean_markdown_text(body))
        if digest in seen:
            target = duplicate_dir / path.name
            if target.exists():
                target = duplicate_dir / f"{path.stem}_{digest[:8]}{path.suffix}"
            shutil.move(str(path), str(target))
            duplicates.append(
                {
                    "duplicate_path": str(target),
                    "kept_path": seen[digest],
                    "content_hash": digest,
                }
            )
            LOGGER.info("Moved duplicate markdown %s -> %s", path, target)
        else:
            seen[digest] = str(path)

    manifest = {
        "deduplicated_at": datetime.now(timezone.utc).isoformat(),
        "unique_documents": len(seen),
        "duplicates": duplicates,
    }
    manifest_path = markdown_dir.parent / "dedup_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest

