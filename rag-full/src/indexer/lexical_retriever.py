"""Chinese-friendly lexical retrieval over processed Markdown.

This retriever is intentionally lightweight and deterministic. It is useful as:

1. A fallback when semantic embeddings are unavailable.
2. A precision layer for company documents where exact stock code / company
   name matches matter more than broad semantic similarity.
3. A transparent baseline for debugging RAG evidence quality.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


LOGGER = logging.getLogger(__name__)
FRONT_MATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n(.*)$", flags=re.S)


@dataclass
class LexicalDocument:
    """One processed Markdown document."""

    path: Path
    metadata: dict[str, Any]
    body: str


def split_front_matter_light(markdown_text: str) -> tuple[dict[str, Any], str]:
    """Parse simple YAML front matter without requiring PyYAML."""
    match = FRONT_MATTER_RE.match(markdown_text)
    if not match:
        return {}, markdown_text

    metadata: dict[str, Any] = {}
    for line in match.group(1).splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        value = value.strip().strip("'\"")
        metadata[key.strip()] = value if value != "null" else None
    return metadata, match.group(2)


def normalize_text(text: str) -> str:
    """Normalize text for matching."""
    return re.sub(r"\s+", " ", text or "").strip()


def expand_query_terms(query: str) -> list[str]:
    """Extract and expand query terms."""
    raw_terms = [term for term in re.split(r"[\s,，;；|/]+", query.strip()) if term]
    expansions: list[str] = []
    query_upper = query.upper()

    if "AI" in query_upper or "人工智能" in query:
        expansions.extend(["人工智能", "AI", "算力", "数据中心", "大模型", "智能算力"])
    if "光模块" in query:
        expansions.extend(["光模块", "光通信", "光网络", "高速光", "CPO", "800G", "1.6T"])
    if "半导体" in query or "芯片" in query:
        expansions.extend(["半导体", "芯片", "集成电路", "软件"])
    if "年报" in query or "年度报告" in query:
        expansions.extend(["年度报告", "经营情况", "核心竞争力", "主营业务"])

    seen: set[str] = set()
    terms: list[str] = []
    for term in raw_terms + expansions:
        term = term.strip()
        if len(term) == 0 or term in seen:
            continue
        seen.add(term)
        terms.append(term)
    return terms


def load_markdown_documents(markdown_dir: str | Path) -> list[LexicalDocument]:
    """Load processed Markdown documents."""
    markdown_dir = Path(markdown_dir)
    docs: list[LexicalDocument] = []
    for path in sorted(markdown_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8", errors="ignore")
        metadata, body = split_front_matter_light(text)
        docs.append(LexicalDocument(path=path, metadata=metadata, body=body))
    return docs


def paragraph_chunks(body: str, max_chars: int = 1100) -> list[str]:
    """Split body into paragraph-like chunks."""
    raw_parts = [part.strip() for part in re.split(r"\n\s*\n", body) if part.strip()]
    chunks: list[str] = []
    current = ""
    for part in raw_parts:
        if not current:
            current = part
        elif len(current) + len(part) + 2 <= max_chars:
            current += "\n\n" + part
        else:
            chunks.append(current)
            current = part
    if current:
        chunks.append(current)
    return chunks or [body[:max_chars]]


def score_text(text: str, terms: list[str], weight: float = 1.0) -> float:
    """Score text with exact term matches and mild length normalization."""
    text_norm = normalize_text(text)
    if not text_norm:
        return 0.0
    score = 0.0
    for term in terms:
        count = text_norm.count(term)
        if count:
            score += weight * (1.0 + math.log1p(count))
            if len(term) >= 4:
                score += weight * 0.3
    return score


def best_snippet(body: str, terms: list[str]) -> tuple[str, float, list[str]]:
    """Return best paragraph chunk and score."""
    best = ""
    best_score = 0.0
    best_terms: list[str] = []
    for chunk in paragraph_chunks(body):
        matched = [term for term in terms if term in chunk]
        score = score_text(chunk, terms, weight=1.0)
        if score > best_score:
            best = chunk
            best_score = score
            best_terms = matched
    if not best:
        best = body[:1100]
    return best[:1400], best_score, best_terms


def source_allowed(
    source: str | None,
    *,
    include_sources: list[str] | None = None,
    exclude_sources: list[str] | None = None,
) -> bool:
    """Filter by source names."""
    source_value = (source or "").lower()
    if include_sources and not any(item.lower() in source_value for item in include_sources):
        return False
    if exclude_sources and any(item.lower() in source_value for item in exclude_sources):
        return False
    return True


def lexical_search(
    query: str,
    *,
    markdown_dir: str | Path,
    top_k: int = 5,
    include_sources: list[str] | None = None,
    exclude_sources: list[str] | None = None,
    required_terms: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Search processed Markdown files with exact-term weighted scoring."""
    terms = expand_query_terms(query)
    required_terms = [term for term in (required_terms or []) if term]
    results: list[dict[str, Any]] = []

    for doc in load_markdown_documents(markdown_dir):
        source = doc.metadata.get("source")
        if not source_allowed(source, include_sources=include_sources, exclude_sources=exclude_sources):
            continue

        title = str(doc.metadata.get("title") or doc.path.stem)
        url = doc.metadata.get("url")
        haystack = f"{title} {url or ''} {doc.body}"
        if required_terms and not any(term in haystack for term in required_terms):
            continue

        snippet, snippet_score, matched_terms = best_snippet(doc.body, terms)
        title_score = score_text(title, terms, weight=5.0)
        metadata_score = score_text(str(url or ""), terms, weight=2.0)
        exact_required_bonus = 4.0 if required_terms else 0.0
        score = title_score + metadata_score + snippet_score + exact_required_bonus

        if score <= 0:
            continue
        results.append(
            {
                "score": float(score),
                "title": title,
                "source": source,
                "url": url,
                "source_file": str(doc.path),
                "content": snippet,
                "matched_terms": matched_terms,
                "retriever": "lexical",
            }
        )

    return sorted(results, key=lambda item: item["score"], reverse=True)[:top_k]

