"""Load a FAISS vector store and run retrieval queries."""

from __future__ import annotations

import logging
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.indexer.build_vectorstore import create_embeddings, load_config, resolve_project_path


LOGGER = logging.getLogger(__name__)
_VECTORSTORE_CACHE: dict[tuple[str, str], Any] = {}
_RERANKER_CACHE: dict[str, Any] = {}
_RERANKER_MODEL = "BAAI/bge-reranker-base"

# Sources whose knowledge is slow-decaying (2-year half-life vs 6-month for others).
_SLOW_DECAY_SOURCES = frozenset(("ndrc", "gov_policy", "csrc", "miit", "local_note"))
_HALF_LIFE_POLICY_DAYS = 730
_HALF_LIFE_DEFAULT_DAYS = 180
_DATE_FIELDS = ("published_at", "report_date", "fetched_at")


def _parse_doc_date(metadata: dict[str, Any]) -> datetime | None:
    for field in _DATE_FIELDS:
        raw = metadata.get(field)
        if not raw:
            continue
        text = str(raw).strip()[:19]
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
            try:
                return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
            except ValueError:
                continue
    return None


def _recency_factor(doc_date: datetime | None, half_life_days: int) -> float:
    """Exponential decay in [0, 1].  Undated documents get a neutral 0.75."""
    if doc_date is None:
        return 0.75
    today = datetime.now(timezone.utc)
    days_old = max(0, (today - doc_date).days)
    return math.exp(-days_old * math.log(2) / half_life_days)


def _apply_time_weights(docs_and_scores: list[tuple[Any, float]]) -> list[tuple[Any, float]]:
    """Re-rank FAISS L2-distance results by blending semantic similarity with recency.

    FAISS returns L2 distances (lower = more similar).  We convert to a
    similarity measure, blend with a recency factor, then re-sort descending so
    the most relevant *and* most recent documents surface first.

    Blend formula:
        sim        = 1 / (1 + raw_dist)     ∈ (0, 1]
        blended    = sim × (0.6 + 0.4 × recency)
    Policy docs use a 2-year half-life; research reports and announcements use 6 months.
    """
    scored: list[tuple[Any, float, float]] = []
    for doc, raw_dist in docs_and_scores:
        source = str(doc.metadata.get("source", "")).lower()
        half_life = _HALF_LIFE_POLICY_DAYS if source in _SLOW_DECAY_SOURCES else _HALF_LIFE_DEFAULT_DAYS
        doc_date = _parse_doc_date(doc.metadata)
        recency = _recency_factor(doc_date, half_life)
        sim = 1.0 / (1.0 + float(raw_dist))
        blended = sim * (0.6 + 0.4 * recency)
        scored.append((doc, raw_dist, blended))

    scored.sort(key=lambda x: x[2], reverse=True)
    return [(doc, raw_dist) for doc, raw_dist, _ in scored]


def _load_reranker(model_name: str = _RERANKER_MODEL) -> Any:
    """Load and cache a CrossEncoder reranker. Returns None if unavailable."""
    if model_name in _RERANKER_CACHE:
        return _RERANKER_CACHE[model_name]
    try:
        from sentence_transformers import CrossEncoder
        reranker = CrossEncoder(model_name)
        _RERANKER_CACHE[model_name] = reranker
        LOGGER.info("Reranker loaded: %s", model_name)
        return reranker
    except Exception as exc:
        LOGGER.warning("Reranker unavailable (%s); falling back to time-weighted FAISS.", exc)
        _RERANKER_CACHE[model_name] = None
        return None


def _apply_reranker_with_time(
    query: str,
    docs_and_scores: list[tuple[Any, float]],
    reranker: Any,
) -> list[tuple[Any, float]]:
    """Cross-encoder second-pass rerank, then blend with recency.

    Pipeline:
        raw CrossEncoder logit → sigmoid → norm_score ∈ (0, 1)
        blended = norm_score × (0.6 + 0.4 × recency)

    Recency uses the same half-life rules as _apply_time_weights so that
    time decay is always the final step regardless of which path is taken.
    """
    pairs = [(query, doc.page_content) for doc, _ in docs_and_scores]
    raw_scores = reranker.predict(pairs)

    result: list[tuple[Any, float, float]] = []
    for (doc, raw_dist), raw_score in zip(docs_and_scores, raw_scores):
        norm_score = 1.0 / (1.0 + math.exp(-float(raw_score)))
        source = str(doc.metadata.get("source", "")).lower()
        half_life = _HALF_LIFE_POLICY_DAYS if source in _SLOW_DECAY_SOURCES else _HALF_LIFE_DEFAULT_DAYS
        recency = _recency_factor(_parse_doc_date(doc.metadata), half_life)
        blended = norm_score * (0.6 + 0.4 * recency)
        result.append((doc, raw_dist, blended))

    result.sort(key=lambda x: x[2], reverse=True)
    return [(doc, raw_dist) for doc, raw_dist, _ in result]


def load_vectorstore(
    config: dict[str, Any],
    config_path: str | Path = "configs/sources.yaml",
    *,
    use_cache: bool = True,
):
    """Load persisted FAISS vector store."""
    from langchain_community.vectorstores import FAISS

    index_dir = resolve_project_path(
        config_path, config.get("project", {}).get("index_dir", "data/index/faiss")
    )
    if not (index_dir / "index.faiss").exists():
        raise FileNotFoundError(f"FAISS index not found: {index_dir}")

    cache_key = (str(index_dir.resolve()), os.getenv("EMBEDDING_DEVICE", ""))
    if use_cache and cache_key in _VECTORSTORE_CACHE:
        return _VECTORSTORE_CACHE[cache_key]

    embeddings = create_embeddings(config)
    vectorstore = FAISS.load_local(
        str(index_dir),
        embeddings,
        allow_dangerous_deserialization=True,
    )
    if use_cache:
        _VECTORSTORE_CACHE[cache_key] = vectorstore
    return vectorstore


def retrieve(
    query: str,
    *,
    config_path: str | Path = "configs/sources.yaml",
    top_k: int = 5,
    time_weighted: bool = True,
    use_reranker: bool = True,
    use_hyde: bool = False,
) -> list[dict[str, Any]]:
    """Retrieve top-k chunks for a query.

    Pipeline when use_reranker=True:
        (HyDE rewrite →) FAISS top-30 → cross-encoder rerank + recency blend → top-k

    Pipeline when use_reranker=False and time_weighted=True:
        (HyDE rewrite →) FAISS top-15 → recency blend → top-k

    Args:
        time_weighted: Blend scores with document recency (ignored when
            use_reranker=True because recency is applied inside the reranker path).
        use_reranker: Apply BAAI/bge-reranker-base second-pass reranking.
            Falls back to time-weighted FAISS if the model is unavailable.
        use_hyde: Apply HyDE query rewriting before embedding. The LLM generates
            a hypothetical answer passage whose vector is closer to real document
            chunks than the raw question is.
    """
    if use_hyde:
        from src.agent.llm_synthesizer import hyde_rewrite
        query = hyde_rewrite(query)

    config = load_config(config_path)
    vectorstore = load_vectorstore(config, config_path=config_path)

    reranker = _load_reranker() if use_reranker else None
    if reranker is not None:
        fetch_k = max(30, top_k * 6)
    elif time_weighted:
        fetch_k = top_k * 3
    else:
        fetch_k = top_k

    docs_and_scores = vectorstore.similarity_search_with_score(query, k=fetch_k)

    if reranker is not None:
        docs_and_scores = _apply_reranker_with_time(query, docs_and_scores, reranker)
    elif time_weighted:
        docs_and_scores = _apply_time_weights(docs_and_scores)

    results: list[dict[str, Any]] = []
    for doc, score in docs_and_scores[:top_k]:
        results.append(
            {
                "score": float(score),
                "title": doc.metadata.get("title"),
                "source": doc.metadata.get("source"),
                "url": doc.metadata.get("url"),
                "source_file": doc.metadata.get("source_file"),
                "chunk_id": doc.metadata.get("chunk_id"),
                "content": doc.page_content,
                "published_at": doc.metadata.get("published_at"),
                "report_date": doc.metadata.get("report_date"),
                "ts_code": doc.metadata.get("ts_code"),
                "doc_type": doc.metadata.get("doc_type"),
            }
        )
    return results


def print_results(results: list[dict[str, Any]]) -> None:
    """Print retrieval results for CLI tests."""
    for i, item in enumerate(results, start=1):
        print("=" * 80)
        print(f"[{i}] score={item['score']:.4f}  ts_code={item.get('ts_code')}  doc_type={item.get('doc_type')}")
        print(f"title: {item.get('title')}")
        print(f"source: {item.get('source')}")
        date = item.get("published_at") or item.get("report_date") or ""
        print(f"date: {date}  url: {item.get('url')}")
        print(f"file: {item.get('source_file')}  chunk={item.get('chunk_id')}")
        preview = item.get("content", "").replace("\n", " ")[:400]
        print(f"preview: {preview}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    print_results(retrieve("中际旭创 光模块 800G", top_k=5))

