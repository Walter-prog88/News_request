"""PDF to standardized Markdown parser.

The parser tries pdfplumber first because it often handles financial PDFs well,
then falls back to PyMuPDF when needed.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.parser.markdown_cleaner import content_hash, write_markdown


LOGGER = logging.getLogger(__name__)


def extract_pdf_text_pdfplumber(path: str | Path) -> str:
    """Extract text from PDF using pdfplumber."""
    import pdfplumber

    chunks: list[str] = []
    with pdfplumber.open(path) as pdf:
        for page_no, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                chunks.append(f"\n\n## Page {page_no}\n\n{text}")
    return "\n".join(chunks)


def extract_pdf_text_pymupdf(path: str | Path) -> str:
    """Extract text from PDF using PyMuPDF."""
    import fitz

    chunks: list[str] = []
    with fitz.open(path) as doc:
        for page_no, page in enumerate(doc, start=1):
            text = page.get_text("text") or ""
            if text.strip():
                chunks.append(f"\n\n## Page {page_no}\n\n{text}")
    return "\n".join(chunks)


def extract_pdf_text(path: str | Path) -> str:
    """Extract text from PDF with fallback parser."""
    try:
        text = extract_pdf_text_pdfplumber(path)
        if text.strip():
            return text
    except Exception as exc:
        LOGGER.warning("pdfplumber failed for %s: %s", path, exc)
    return extract_pdf_text_pymupdf(path)


def parse_pdf_file(
    raw_path: str | Path,
    metadata: dict[str, Any],
    output_dir: str | Path,
) -> Path | None:
    """Parse one raw PDF file and write standardized Markdown."""
    raw_path = Path(raw_path)
    body = extract_pdf_text(raw_path)
    if not body.strip():
        LOGGER.warning("No text extracted from PDF: %s", raw_path)
        return None

    title = metadata.get("title") or raw_path.stem
    digest = content_hash(body)
    output_path = Path(output_dir) / f"{metadata.get('source', 'pdf')}_{digest[:12]}.md"
    front_matter = {
        "source": metadata.get("source"),
        "title": title,
        "url": metadata.get("url"),
        "published_at": metadata.get("published_at"),
        "fetched_at": metadata.get("fetched_at"),
        "content_type": metadata.get("content_type") or "application/pdf",
        "raw_path": str(raw_path),
        "raw_sha256": metadata.get("sha256"),
        "parser": "pdf_parser",
        "parsed_at": datetime.now(timezone.utc).isoformat(),
        **(metadata.get("extra") or {}),
    }
    return write_markdown(output_path, front_matter, body)

