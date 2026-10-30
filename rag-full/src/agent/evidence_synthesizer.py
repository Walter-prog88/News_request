"""Rule-based synthesis of retrieved evidence.

This module turns raw retrieval hits into concise research bullets. It is not a
replacement for an LLM, but it gives the report a useful structure before a
language model is introduced.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from typing import Any


LOGGER = logging.getLogger(__name__)

# Universal positive signals — industry-agnostic.
POSITIVE_TERMS = [
    "增长",
    "增加",
    "领先",
    "优势",
    "核心竞争力",
    "需求",
    "政策",
    "支持",
]

# Words that appear frequently in document titles but carry no domain signal.
_TITLE_STOPWORDS = frozenset({
    "公司", "报告", "研究", "分析", "年度", "季度", "股份", "有限",
    "集团", "深度", "摘要", "综合", "内容", "情况", "说明", "概况",
    "问题", "简介", "介绍", "关于", "来源", "类型", "文件", "文档",
    "半年", "三季", "一季", "全年", "月报", "周报", "日报", "更新",
    "点评", "跟踪", "系列", "专题", "研报", "公告", "声明", "发布",
    "行业", "板块", "市场", "投资", "策略", "评级", "观点", "解读",
})

RISK_TERMS = [
    "风险",
    "不确定",
    "下滑",
    "减少",
    "波动",
    "竞争",
    "价格",
    "资本开支",
    "回撤",
    "高位",
]

SUPPORT_EXCLUDE_TERMS = [
    "做空",
    "高危",
    "危机",
    "砍单",
    "下调",
    "见顶",
    "跑输",
    "回撤",
    "高位",
    "不确定",
    "风险",
    "VIX",
    "SOX",
]


def extract_domain_terms(docs: list[dict[str, Any]], top_k: int = 12) -> list[str]:
    """Extract high-frequency domain terms from retrieved document titles.

    Uses jieba segmentation on concatenated titles so that industry keywords
    (e.g. "白酒", "光伏", "创新药") bubble up automatically, with no
    prior knowledge of which sector the stock belongs to.
    """
    try:
        import jieba
        jieba.setLogLevel(logging.WARNING)
    except ImportError:
        LOGGER.warning("jieba not installed; domain term extraction disabled")
        return []

    titles = " ".join(
        str(item.get("title") or "")
        for item in docs
        if item.get("title")
    )
    if not titles.strip():
        return []

    words = [w for w in jieba.cut(titles) if len(w) >= 2 and w not in _TITLE_STOPWORDS]
    freq = Counter(words)
    return [w for w, _ in freq.most_common(top_k)]


def clean_sentence(sentence: str) -> str:
    """Remove Markdown noise from a candidate evidence sentence."""
    sentence = re.sub(r"^[#>*\-\s]+", "", sentence.strip())
    sentence = sentence.replace("```", "").replace("`", "")
    sentence = re.sub(r"\s+", " ", sentence)
    return sentence.strip()


def looks_like_noise(sentence: str) -> bool:
    """Identify table/code fragments that should not become report bullets."""
    if not sentence:
        return True
    if sentence.count("|") >= 3:
        return True
    if sentence.count("✓") + sentence.count("+") >= 3:
        return True
    if len(sentence) < 12:
        return True
    return False


def split_sentences(text: str) -> list[str]:
    """Split Chinese/English text into sentence-like units."""
    text = (text or "").strip()
    if not text:
        return []

    sentences: list[str] = []
    for line in re.split(r"[\r\n]+", text):
        line = clean_sentence(line)
        if looks_like_noise(line):
            continue
        for part in re.split(r"(?<=[。！？；;.!?])\s*", line):
            part = clean_sentence(part)
            if not looks_like_noise(part):
                sentences.append(part)
    return sentences


def score_sentence(sentence: str, terms: list[str]) -> int:
    """Score a sentence by term hits."""
    return sum(sentence.count(term) for term in terms)


def best_sentences(
    items: list[dict[str, Any]],
    terms: list[str],
    *,
    max_items: int = 3,
    exclude_terms: list[str] | None = None,
    required_any: list[str] | None = None,
    official_source_bonus: bool = False,
) -> list[dict[str, str]]:
    """Extract best evidence sentences from retrieved items."""
    candidates: list[dict[str, Any]] = []
    exclude_terms = exclude_terms or []
    required_any = required_any or []
    for item in items:
        title = item.get("title") or "未命名文档"
        source = item.get("source") or "unknown"
        url = item.get("url") or item.get("source_file") or ""
        for sentence in split_sentences(item.get("content") or ""):
            if exclude_terms and any(term in sentence for term in exclude_terms):
                continue
            if required_any and not any(term in sentence for term in required_any):
                continue
            score = score_sentence(sentence, terms)
            if official_source_bonus and source in {"ndrc", "gov_policy"}:
                score += 2
            if score <= 0:
                continue
            candidates.append(
                {
                    "score": score,
                    "sentence": sentence[:260],
                    "title": title,
                    "source": source,
                    "url": url,
                }
            )

    candidates = sorted(candidates, key=lambda row: row["score"], reverse=True)
    seen: set[str] = set()
    selected: list[dict[str, str]] = []
    for row in candidates:
        key = row["sentence"][:80]
        if key in seen:
            continue
        seen.add(key)
        selected.append(
            {
                "text": row["sentence"],
                "title": row["title"],
                "source": row["source"],
                "url": row["url"],
            }
        )
        if len(selected) >= max_items:
            break
    return selected


def synthesize_evidence(
    policy_docs: list[dict[str, Any]],
    company_docs: list[dict[str, Any]],
    *,
    theme: str,
    company_name: str,
    risk_docs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Create structured bullets from policy and company evidence.

    Domain terms are extracted dynamically from the titles of retrieved
    documents, so the function generalises to any stock/sector without
    requiring hardcoded industry keywords.
    """
    all_docs = policy_docs + company_docs + (risk_docs or [])
    domain_terms = extract_domain_terms(all_docs)
    theme_words = [w for w in re.split(r"[\s,，;；|/]+", theme) if w]

    # Build scoring term lists: theme/company words first, then domain terms
    # extracted from retrieved titles, then universal signals.
    theme_terms = list(dict.fromkeys(theme_words + domain_terms))
    company_terms = list(dict.fromkeys([company_name] + theme_words + domain_terms))

    # required_any acts as a soft relevance gate.
    # Policy sentences must mention the theme or a domain keyword.
    # Company sentences must at least mention the company name.
    # Falls back to no filtering (None) when the term list is empty.
    policy_required: list[str] | None = (
        list(dict.fromkeys(["政策", "支持"] + theme_words + domain_terms[:4])) or None
    )
    company_required = list(dict.fromkeys([company_name, "增长"] + domain_terms[:5]))

    theme_support = best_sentences(
        policy_docs,
        theme_terms + POSITIVE_TERMS,
        max_items=3,
        exclude_terms=SUPPORT_EXCLUDE_TERMS,
        required_any=policy_required,
        official_source_bonus=True,
    )
    company_support = best_sentences(
        company_docs,
        company_terms + POSITIVE_TERMS,
        max_items=4,
        exclude_terms=SUPPORT_EXCLUDE_TERMS,
        required_any=company_required,
    )
    risk_points = best_sentences(
        (risk_docs or []) + company_docs + policy_docs,
        RISK_TERMS,
        max_items=3,
        required_any=RISK_TERMS,
    )

    gaps: list[str] = []
    if not policy_docs:
        gaps.append("缺少政策/产业资料证据，主题持续性判断的外部依据不足。")
    if not company_docs:
        gaps.append("缺少公司公告、年报或投资者关系记录，个股基本面证据不足。")
    if not any("投资者关系" in str(item.get("title") or "") for item in company_docs):
        gaps.append("尚未接入投资者关系记录，管理层对订单、产能和需求的最新表述缺失。")
    if not any("政策" in str(item.get("title") or item.get("content") or "") for item in policy_docs):
        gaps.append(f"当前政策证据偏少，建议补充与「{theme}」相关的产业政策文件。")

    return {
        "theme_support": theme_support,
        "company_support": company_support,
        "risk_points": risk_points,
        "evidence_gaps": gaps,
    }


def render_synthesis_markdown(synthesis: dict[str, Any]) -> str:
    """Render synthesis as Markdown."""
    lines: list[str] = []

    sections = [
        ("主题支持要点", synthesis.get("theme_support") or []),
        ("公司支撑要点", synthesis.get("company_support") or []),
        ("风险关注要点", synthesis.get("risk_points") or []),
    ]
    for title, items in sections:
        lines.append(f"### {title}")
        if not items:
            lines.append("- 暂无足够证据。")
        else:
            for item in items:
                lines.append(f"- {item['text']}（来源：{item['title']}）")
        lines.append("")

    lines.append("### 证据缺口")
    gaps = synthesis.get("evidence_gaps") or []
    if not gaps:
        lines.append("- 暂无显著证据缺口。")
    else:
        for gap in gaps:
            lines.append(f"- {gap}")
    lines.append("")
    return "\n".join(lines)

