"""Build a LangChain FAISS vector store from processed Markdown files."""

from __future__ import annotations

import hashlib
import logging
import math
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any

import yaml

from src.parser.markdown_cleaner import clean_markdown_text, content_hash, split_front_matter


# ---------------------------------------------------------------------------
# Metadata enrichment helpers
# ---------------------------------------------------------------------------

def _normalize_ts_code(stock_code: Any, exchange: str | None = None) -> str | None:
    """Normalize raw stock_code (str or list) + exchange into 'XXXXXX.SZ/SH'."""
    if isinstance(stock_code, list):
        stock_code = stock_code[0] if stock_code else None
    if not stock_code:
        return None
    code = str(stock_code).strip().zfill(6)
    if exchange:
        ex = exchange.upper()
        if ex in ("SZSE", "SZ"):
            return f"{code}.SZ"
        if ex in ("SSE", "SH", "SHSE"):
            return f"{code}.SH"
    if code[:3] in ("000", "001", "002", "003", "300", "301", "302"):
        return f"{code}.SZ"
    if code[:3] in ("600", "601", "603", "605", "688"):
        return f"{code}.SH"
    return None


_ANNUAL_REPORT_KEYWORDS = ("年度报告", "年报")
_SEMI_ANNUAL_KEYWORDS = ("半年度报告", "半年报")
_QUARTERLY_KEYWORDS = ("季度报告", "一季报", "三季报", "季报")
_POLICY_SOURCES = frozenset(("ndrc", "gov_policy", "csrc", "miit"))


def _derive_doc_type(source: str | None, title: str | None) -> str:
    """Classify a document into a coarse doc_type for downstream filtering."""
    src = (source or "").lower()
    ttl = (title or "").lower()
    if src == "research_report":
        return "research_report"
    if src in _POLICY_SOURCES:
        return "policy"
    if src == "local_note":
        return "local_note"
    if any(k in ttl for k in _ANNUAL_REPORT_KEYWORDS):
        return "annual_report"
    if any(k in ttl for k in _SEMI_ANNUAL_KEYWORDS):
        return "semi_annual_report"
    if any(k in ttl for k in _QUARTERLY_KEYWORDS):
        return "quarterly_report"
    return "announcement"


def _enrich_doc_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Add normalized ts_code and doc_type to a chunk's metadata dict in-place."""
    if "ts_code" not in metadata:
        raw_code = metadata.get("stock_code")
        exchange = metadata.get("exchange")
        normalized = _normalize_ts_code(raw_code, exchange)
        if normalized:
            metadata["ts_code"] = normalized
    source = metadata.get("source")
    title = metadata.get("title")
    if "doc_type" not in metadata:
        metadata["doc_type"] = _derive_doc_type(source, title)
    return metadata


LOGGER = logging.getLogger(__name__)


try:
    from langchain_core.embeddings import Embeddings
except Exception:
    Embeddings = object  # type: ignore[assignment]


class HashingEmbeddings(Embeddings):
    """Deterministic local fallback embeddings.

    This is not a semantic model. It exists so the pipeline can run offline when
    HuggingFace model downloads are unavailable. Use the configured multilingual
    sentence-transformer for real retrieval quality.
    """

    def __init__(self, dimension: int = 384):
        self.dimension = dimension

    def _embed(self, text: str) -> list[float]:
        tokens = re.findall(r"[\u4e00-\u9fff]|[a-zA-Z0-9_]+", text.lower())
        vector = [0.0] * self.dimension
        for token in tokens:
            digest = hashlib.md5(token.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "big") % self.dimension
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[idx] += sign
        norm = math.sqrt(sum(v * v for v in vector))
        if norm > 0:
            vector = [v / norm for v in vector]
        return vector

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of documents."""
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        """Embed one query."""
        return self._embed(text)

    def __call__(self, text: str) -> list[float]:
        """Compatibility hook for older LangChain FAISS call sites."""
        return self.embed_query(text)


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load YAML config."""
    with Path(config_path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_project_path(config_path: str | Path, value: str | Path) -> Path:
    """Resolve config-relative project paths."""
    path = Path(value)
    if path.is_absolute():
        return path
    project_root = Path(config_path).resolve().parent.parent
    return project_root / path


def create_embeddings(config: dict[str, Any]):
    """Create configured embedding model with deterministic fallback."""
    embedding_config = config.get("embedding", {})
    provider = embedding_config.get("provider", "huggingface")
    model_name = embedding_config.get(
        "model_name", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    device = os.getenv("EMBEDDING_DEVICE", embedding_config.get("device", "cpu"))

    if provider == "huggingface":
        try:
            try:
                from langchain_huggingface import HuggingFaceEmbeddings
            except Exception:
                from langchain_community.embeddings import HuggingFaceEmbeddings

            LOGGER.info("Using HuggingFace embeddings: %s on %s", model_name, device)
            return HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={"device": device},
                encode_kwargs={
                    "normalize_embeddings": bool(
                        embedding_config.get("normalize_embeddings", True)
                    )
                },
            )
        except Exception as exc:
            LOGGER.warning("Failed to initialize HuggingFace embeddings: %s", exc)

    dimension = int(embedding_config.get("hashing_dimension", 384))
    LOGGER.warning("Using HashingEmbeddings fallback with dimension=%s", dimension)
    return HashingEmbeddings(dimension=dimension)


def load_markdown_documents(markdown_dir: str | Path, config: dict[str, Any] | None = None):
    """Load Markdown files as LangChain Document objects.

    When *config* is provided, documents whose title matches any pattern in
    ``chunking.exclude_title_patterns`` are skipped and their FAISS entries will
    be removed during the next incremental update (because they disappear from
    ``current_docs``).  This is the primary mechanism for excluding English
    annual reports without touching source files.
    """
    try:
        from langchain_core.documents import Document
    except Exception:
        from langchain.schema import Document

    markdown_dir = Path(markdown_dir)
    exclude_patterns: list[str] = []
    if config:
        exclude_patterns = config.get("chunking", {}).get("exclude_title_patterns", [])

    documents = []
    seen_hashes: set[str] = set()
    skipped_excluded = 0

    for path in sorted(markdown_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8", errors="ignore")
        metadata, body = split_front_matter(text)
        body = clean_markdown_text(body)
        if not body.strip():
            continue

        title = str(metadata.get("title") or "")
        if any(pat in title for pat in exclude_patterns):
            skipped_excluded += 1
            LOGGER.debug("Excluding document (title pattern match): %s", path.name)
            continue

        digest = metadata.get("content_hash") or content_hash(body)
        if digest in seen_hashes:
            LOGGER.info("Skipping duplicate markdown during indexing: %s", path)
            continue
        seen_hashes.add(digest)

        doc_metadata: dict[str, Any] = {
            "source_file": str(path),
            "content_hash": digest,
            **{k: v for k, v in metadata.items() if v is not None},
        }
        _enrich_doc_metadata(doc_metadata)
        documents.append(Document(page_content=body, metadata=doc_metadata))

    if skipped_excluded:
        LOGGER.info("Excluded %d documents matching title patterns", skipped_excluded)
    LOGGER.info("Loaded %d unique Markdown documents", len(documents))
    return documents


def _cosine_sim(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


def _split_sentences(text: str) -> list[str]:
    """Split Chinese text into sentences on natural punctuation boundaries."""
    parts = re.split(r"(?<=[。！？\n])\s*", text)
    return [p.strip() for p in parts if p.strip() and len(p.strip()) > 5]


def semantic_split_documents(
    documents: list[Any],
    config: dict[str, Any],
    embeddings: Any,
) -> list[Any]:
    """Split documents at semantic breakpoints instead of fixed character counts.

    Algorithm:
        1. Split each document into sentences.
        2. Embed all sentences in one batch per document.
        3. Compute cosine similarity between consecutive sentences.
        4. Start a new chunk wherever similarity drops below the threshold
           OR the current chunk would exceed max_chunk_size characters.
        5. Apply the same max_chunks_per_doc cap as the recursive splitter.

    This keeps semantically cohesive passages together, avoiding mid-sentence
    cuts that leave chunks with incomplete information.
    """
    try:
        from langchain_core.documents import Document
    except Exception:
        from langchain.schema import Document

    chunk_config = config.get("chunking", {})
    max_chunk_size = int(chunk_config.get("chunk_size", 900))
    threshold = float(chunk_config.get("semantic_breakpoint_threshold", 0.45))
    max_per_doc = chunk_config.get("max_chunks_per_doc")
    cap = int(max_per_doc) if max_per_doc and int(max_per_doc) > 0 else None

    all_chunks: list[Any] = []
    chunk_id = 0

    for doc in documents:
        sentences = _split_sentences(doc.page_content)
        if len(sentences) <= 1:
            # Too short to split further — keep as-is
            meta = {**doc.metadata, "chunk_id": chunk_id}
            all_chunks.append(Document(page_content=doc.page_content, metadata=meta))
            chunk_id += 1
            continue

        vecs = embeddings.embed_documents(sentences)

        groups: list[str] = []
        current: list[str] = [sentences[0]]

        for i in range(1, len(sentences)):
            sim = _cosine_sim(vecs[i - 1], vecs[i])
            current_len = sum(len(s) for s in current)
            if sim < threshold or current_len + len(sentences[i]) > max_chunk_size:
                groups.append("".join(current))
                current = [sentences[i]]
            else:
                current.append(sentences[i])
        if current:
            groups.append("".join(current))

        # Apply per-doc cap with same head/tail sampling strategy
        if cap and len(groups) > cap:
            head_n = max(1, cap // 4)
            tail_n = cap - head_n
            head = groups[:head_n]
            rest = groups[head_n:]
            step = max(1, len(rest) // tail_n)
            groups = head + rest[::step][:tail_n]

        for text in groups:
            meta = {**doc.metadata, "chunk_id": chunk_id}
            all_chunks.append(Document(page_content=text, metadata=meta))
            chunk_id += 1

    LOGGER.info("Semantic split: %d docs → %d chunks", len(documents), len(all_chunks))
    return all_chunks


def split_documents(documents: list[Any], config: dict[str, Any]) -> list[Any]:
    """Split documents into retrieval chunks.

    When ``chunking.max_chunks_per_doc`` is set, long documents are sampled so
    that no single file can occupy more than that many index slots.  We keep the
    first quarter (intro/highlights) verbatim and sample the remainder evenly,
    so both the beginning and the body of large annual reports are covered.
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    chunk_config = config.get("chunking", {})
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=int(chunk_config.get("chunk_size", 900)),
        chunk_overlap=int(chunk_config.get("chunk_overlap", 150)),
        separators=["\n\n", "\n", "。", "；", ";", "，", ",", " ", ""],
    )
    raw_chunks = splitter.split_documents(documents)

    max_per_doc = chunk_config.get("max_chunks_per_doc")
    if max_per_doc and int(max_per_doc) > 0:
        cap = int(max_per_doc)
        from collections import defaultdict
        groups: dict[str, list[Any]] = defaultdict(list)
        for chunk in raw_chunks:
            groups[chunk.metadata.get("source_file", "")].append(chunk)

        capped: list[Any] = []
        for group_chunks in groups.values():
            if len(group_chunks) <= cap:
                capped.extend(group_chunks)
            else:
                head_n = max(1, cap // 4)
                tail_n = cap - head_n
                head = group_chunks[:head_n]
                rest = group_chunks[head_n:]
                step = max(1, len(rest) // tail_n)
                tail = rest[::step][:tail_n]
                capped.extend(head + tail)
        LOGGER.info(
            "Chunk cap=%d: %d raw chunks → %d after capping", cap, len(raw_chunks), len(capped)
        )
        chunks = capped
    else:
        chunks = raw_chunks

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
    LOGGER.info("Split into %d chunks", len(chunks))
    return chunks


def build_faiss_in_batches(chunks: list[Any], embeddings: Any, config: dict[str, Any]):
    """Build FAISS with progress logs instead of one opaque embed call."""
    from langchain_community.vectorstores import FAISS

    if not chunks:
        raise RuntimeError("No chunks available for indexing")

    embedding_config = config.get("embedding", {})
    batch_size = int(embedding_config.get("index_batch_size", 512))
    total = len(chunks)
    started_at = time.monotonic()

    vectorstore = None
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch = chunks[start:end]
        LOGGER.info("Embedding chunks %d-%d / %d", start + 1, end, total)

        if vectorstore is None:
            vectorstore = FAISS.from_documents(batch, embeddings)
        else:
            vectorstore.add_documents(batch)

        elapsed = time.monotonic() - started_at
        processed = end
        rate = processed / elapsed if elapsed > 0 else 0.0
        remaining = (total - processed) / rate if rate > 0 else 0.0
        LOGGER.info(
            "Indexed %d/%d chunks (%.1f%%), %.1f chunks/sec, eta %.1f min",
            processed,
            total,
            processed * 100 / total,
            rate,
            remaining / 60,
        )

    return vectorstore


def save_vectorstore_atomically(vectorstore: Any, index_dir: Path) -> None:
    """Persist FAISS files without replacing the live index until save succeeds."""
    index_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=str(index_dir.parent), prefix=f".{index_dir.name}."
    ) as tmp:
        tmp_dir = Path(tmp)
        vectorstore.save_local(str(tmp_dir))
        for filename in ("index.faiss", "index.pkl"):
            src = tmp_dir / filename
            if not src.exists():
                raise RuntimeError(f"Expected FAISS output missing: {src}")
            os.replace(src, index_dir / filename)


def build_vectorstore(config_path: str | Path = "configs/sources.yaml") -> Path:
    """Build and persist FAISS vector store from processed Markdown."""
    config = load_config(config_path)
    project_config = config.get("project", {})
    markdown_dir = resolve_project_path(
        config_path, project_config.get("processed_markdown_dir", "data/processed/markdown")
    )
    index_dir = resolve_project_path(config_path, project_config.get("index_dir", "data/index/faiss"))
    index_dir.mkdir(parents=True, exist_ok=True)

    documents = load_markdown_documents(markdown_dir, config=config)
    if not documents:
        raise RuntimeError(f"No Markdown documents found in {markdown_dir}")

    chunk_strategy = config.get("chunking", {}).get("strategy", "recursive")
    if chunk_strategy == "semantic":
        # Semantic splitting needs the embedding model upfront to find breakpoints.
        LOGGER.info("Using semantic chunking strategy (strategy=semantic)")
        embeddings = create_embeddings(config)
        chunks = semantic_split_documents(documents, config, embeddings)
    else:
        chunks = split_documents(documents, config)
        embeddings = create_embeddings(config)

    vectorstore = build_faiss_in_batches(chunks, embeddings, config)
    save_vectorstore_atomically(vectorstore, index_dir)
    LOGGER.info("Saved FAISS vector store to %s", index_dir)
    return index_dir


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    build_vectorstore()

