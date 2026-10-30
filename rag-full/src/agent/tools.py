"""Tool wrappers used by the deterministic research workflow."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from src.indexer.lexical_retriever import lexical_search
from src.agent.python_executor import analyze_eps_revision_return_spread, run_python_analysis
from src.market.data_loader import get_stock_info, normalize_ts_code
from src.market.factor_signal import calculate_factor_score, get_industry_peers
from src.market.risk import risk_check
from src.market.technical import calculate_technical_signals


LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

SOURCE_GROUPS = {
    "research_report": ["research_report"],
    "announcement": ["announcement"],
    "policy": ["ndrc", "gov_policy", "csrc", "miit"],
    "local_note": ["local_note"],
    "news": ["news"],
}


def _source_allowed(
    source: str | None,
    *,
    include_sources: list[str] | None,
    exclude_sources: list[str] | None,
) -> bool:
    source_value = (source or "").lower()
    if include_sources and not any(value.lower() in source_value for value in include_sources):
        return False
    if exclude_sources and any(value.lower() in source_value for value in exclude_sources):
        return False
    return True


def _has_required_term(item: dict[str, Any], required_terms: list[str] | None) -> bool:
    required_terms = [term for term in (required_terms or []) if term]
    if not required_terms:
        return True
    haystack = " ".join(
        str(item.get(key) or "")
        for key in ("title", "url", "source_file", "content")
    )
    return any(term in haystack for term in required_terms)


def _dedupe_key(item: dict[str, Any]) -> str:
    return str(item.get("source_file") or item.get("url") or item.get("title") or id(item))


def _merge_ranked_candidates(
    lexical_items: list[dict[str, Any]],
    semantic_items: list[dict[str, Any]],
    top_k: int,
) -> list[dict[str, Any]]:
    """Merge lexical and vector candidates with reciprocal rank fusion."""
    candidates: dict[str, dict[str, Any]] = {}
    rrf_k = 60.0

    def update(item: dict[str, Any], retriever: str, rank: int) -> None:
        key = _dedupe_key(item)
        current = candidates.get(key)
        contribution = 1.0 / (rrf_k + rank)
        if current is None:
            current = {**item}
            current["_rank_score"] = 0.0
            current["_retrievers"] = set()
            candidates[key] = current

        current["_rank_score"] += contribution
        current["_retrievers"].add(retriever)

        if retriever == "lexical":
            current["lexical_score"] = float(item.get("score", 0.0))
            if item.get("matched_terms"):
                current["matched_terms"] = item.get("matched_terms")
            # Lexical snippets are usually tighter for exact company/code hits.
            if not current.get("content") or "semantic" not in current["_retrievers"]:
                current["content"] = item.get("content")
        else:
            current["semantic_score"] = float(item.get("score", 0.0))
            if "lexical" not in current["_retrievers"]:
                current["content"] = item.get("content")

    for rank, item in enumerate(lexical_items, start=1):
        update(item, "lexical", rank)
    seen_semantic_keys: set[str] = set()
    semantic_rank = 0
    for item in semantic_items:
        key = _dedupe_key(item)
        if key in seen_semantic_keys:
            continue
        seen_semantic_keys.add(key)
        semantic_rank += 1
        update(item, "semantic", semantic_rank)

    results: list[dict[str, Any]] = []
    for item in sorted(candidates.values(), key=lambda value: value["_rank_score"], reverse=True):
        retrievers = sorted(item.pop("_retrievers"))
        rank_score = item.pop("_rank_score")
        item["retriever"] = "hybrid:" + "+".join(retrievers)
        item["score"] = rank_score * 100.0
        results.append(item)
    return results[:top_k]


def rag_search(
    query: str,
    top_k: int = 5,
    *,
    include_sources: list[str] | None = None,
    exclude_sources: list[str] | None = None,
    required_terms: list[str] | None = None,
    source_quotas: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    """Search Markdown lexically and FAISS semantically, then merge results."""
    if source_quotas:
        return _rag_search_with_source_quotas(
            query,
            top_k=top_k,
            source_quotas=source_quotas,
            include_sources=include_sources,
            exclude_sources=exclude_sources,
            required_terms=required_terms,
        )
    return _rag_search_once(
        query,
        top_k=top_k,
        include_sources=include_sources,
        exclude_sources=exclude_sources,
        required_terms=required_terms,
    )


def _rag_search_once(
    query: str,
    top_k: int,
    *,
    include_sources: list[str] | None,
    exclude_sources: list[str] | None,
    required_terms: list[str] | None,
) -> list[dict[str, Any]]:
    """Run one hybrid search with optional source filters."""
    markdown_dir = PROJECT_ROOT / "data" / "processed" / "markdown"
    candidate_k = max(top_k * 4, 20)

    lexical_items = lexical_search(
        query,
        markdown_dir=markdown_dir,
        top_k=candidate_k,
        include_sources=include_sources,
        exclude_sources=exclude_sources,
        required_terms=required_terms,
    )

    semantic_items: list[dict[str, Any]] = []
    try:
        index_dir = PROJECT_ROOT / "data" / "index" / "faiss"
        if (index_dir / "index.faiss").exists():
            from src.indexer.retriever import retrieve

            for item in retrieve(
                query,
                config_path=PROJECT_ROOT / "configs" / "sources.yaml",
                top_k=candidate_k,
            ):
                if not _source_allowed(
                    item.get("source"),
                    include_sources=include_sources,
                    exclude_sources=exclude_sources,
                ):
                    continue
                if not _has_required_term(item, required_terms):
                    continue
                semantic_items.append({**item, "retriever": "semantic"})
    except Exception as exc:
        LOGGER.warning("FAISS retrieval failed; using lexical-only results: %s", exc)

    if semantic_items:
        return _merge_ranked_candidates(lexical_items, semantic_items, top_k)
    return lexical_items[:top_k]


def _sources_for_group(group: str) -> list[str]:
    return SOURCE_GROUPS.get(group, [group])


def _source_filter_allowed_by_parent(
    source_values: list[str],
    include_sources: list[str] | None,
) -> bool:
    if not include_sources:
        return True
    return any(
        parent.lower() in value.lower() or value.lower() in parent.lower()
        for value in source_values
        for parent in include_sources
    )


def _rag_search_with_source_quotas(
    query: str,
    *,
    top_k: int,
    source_quotas: dict[str, int],
    include_sources: list[str] | None,
    exclude_sources: list[str] | None,
    required_terms: list[str] | None,
) -> list[dict[str, Any]]:
    """Run source-aware retrieval so long documents cannot consume all slots."""
    results: list[dict[str, Any]] = []
    seen: set[str] = set()

    for group, quota in source_quotas.items():
        if quota <= 0:
            continue
        group_sources = _sources_for_group(group)
        if not _source_filter_allowed_by_parent(group_sources, include_sources):
            continue
        group_results = _rag_search_once(
            query,
            top_k=quota,
            include_sources=group_sources,
            exclude_sources=exclude_sources,
            required_terms=required_terms,
        )
        for item in group_results:
            key = _dedupe_key(item)
            if key in seen:
                continue
            seen.add(key)
            item = {**item, "source_group": group}
            results.append(item)
            if len(results) >= top_k:
                return results

    if len(results) < top_k:
        fill_results = _rag_search_once(
            query,
            top_k=max(top_k * 2, 10),
            include_sources=include_sources,
            exclude_sources=exclude_sources,
            required_terms=required_terms,
        )
        for item in fill_results:
            key = _dedupe_key(item)
            if key in seen:
                continue
            seen.add(key)
            results.append(item)
            if len(results) >= top_k:
                break
    return results[:top_k]


def _company_source_quotas(top_k: int) -> dict[str, int]:
    if top_k <= 1:
        return {"research_report": top_k}
    research_quota = min(top_k - 1, max(1, round(top_k * 0.7)))
    return {
        "research_report": research_quota,
        "announcement": top_k - research_quota,
    }


def _policy_source_quotas(top_k: int) -> dict[str, int]:
    if top_k <= 2:
        return {"policy": top_k}
    local_note_quota = 1 if top_k <= 4 else 2
    return {
        "policy": top_k - local_note_quota,
        "local_note": local_note_quota,
    }


def search_policy_docs(theme: str, question: str, top_k: int = 5) -> list[dict[str, Any]]:
    """Search policy and industry documents for a theme."""
    expanded_terms = []
    if "AI" in theme.upper() or "人工智能" in theme or "AI" in question.upper():
        expanded_terms.extend(["人工智能", "算力", "数据中心", "集成电路", "软件"])
    if "光模块" in theme or "光模块" in question:
        expanded_terms.extend(["光通信", "通信设备", "光网络", "CPO", "800G"])
    expanded_query = " ".join([theme, question, "政策 产业 支持"] + expanded_terms)
    return rag_search(
        expanded_query,
        top_k=top_k,
        include_sources=["ndrc", "gov_policy", "csrc", "miit", "local_note"],
        exclude_sources=["announcement"],
        source_quotas=_policy_source_quotas(top_k),
    )


def search_company_docs(stock_code: str, theme: str, top_k: int = 5) -> list[dict[str, Any]]:
    """Search company-specific announcements, annual reports, and research reports."""
    info = get_stock_info(stock_code)
    company_name = info.get("name") or normalize_ts_code(stock_code)
    ts_code = normalize_ts_code(stock_code)
    symbol = ts_code.split(".")[0]
    return rag_search(
        f"{company_name} {ts_code} {symbol} {theme} 公告 年报 季报 研报 核心竞争力 经营情况 EPS 业绩",
        top_k=top_k,
        include_sources=["announcement", "research_report"],
        required_terms=[company_name, ts_code, symbol],
        source_quotas=_company_source_quotas(top_k),
    )


def search_news_docs(stock_code: str, theme: str, top_k: int = 5) -> list[dict[str, Any]]:
    """Search recent news articles for a specific stock (source=news)."""
    info = get_stock_info(stock_code)
    company_name = info.get("name") or normalize_ts_code(stock_code)
    ts_code = normalize_ts_code(stock_code)
    symbol = ts_code.split(".")[0]
    return _rag_search_once(
        f"{company_name} {ts_code} {theme} 最新消息 动态",
        top_k=top_k,
        include_sources=["news"],
        exclude_sources=None,
        required_terms=[company_name, ts_code, symbol],
    )


def search_company_risk_docs(stock_code: str, theme: str, top_k: int = 3) -> list[dict[str, Any]]:
    """Search company announcements for risk-related passages."""
    info = get_stock_info(stock_code)
    company_name = info.get("name") or normalize_ts_code(stock_code)
    ts_code = normalize_ts_code(stock_code)
    symbol = ts_code.split(".")[0]
    return rag_search(
        f"{company_name} {ts_code} {symbol} {theme} 风险 不确定 市场竞争 价格波动 经营风险",
        top_k=top_k,
        include_sources=["announcement"],
        required_terms=[company_name, ts_code, symbol],
    )


def get_analyst_consensus(stock_code: str) -> dict[str, Any]:
    """Return structured analyst consensus facts + verdict for a stock."""
    from src.market.analyst_consensus import get_analyst_consensus as _get
    from src.market.data_loader import normalize_ts_code
    return _get(normalize_ts_code(stock_code))


def get_market_data(stock_code: str) -> dict[str, Any]:
    """Return market data summary for a stock."""
    return calculate_technical_signals(stock_code)


def get_factor_data(stock_code: str) -> dict[str, Any]:
    """Return factor scoring summary for a stock."""
    return calculate_factor_score(stock_code)


def get_peer_comparison(stock_code: str) -> str:
    """Return Markdown table of industry peers for LLM context."""
    return get_industry_peers(stock_code)


def get_risk_data(stock_code: str) -> dict[str, Any]:
    """Return risk-check result for a stock."""
    return risk_check(stock_code)


def run_table_analysis(code: str, timeout_seconds: int = 8) -> dict[str, Any]:
    """Run controlled pandas/numpy code against local project datasets."""
    return run_python_analysis(code, timeout_seconds=timeout_seconds)


def get_eps_revision_spread_analysis(timeout_seconds: int = 15) -> dict[str, Any]:
    """Return a dynamic Python analysis for EPS revision predictive spread."""
    return analyze_eps_revision_return_spread(timeout_seconds=timeout_seconds)

