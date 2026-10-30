"""LLM-powered investment analysis generation.

Supports multiple backends via a unified interface:
  - siliconflow  (https://api.siliconflow.cn/v1 — recommended, free tier)
  - deepseek     (https://api.deepseek.com)
  - ollama       (http://localhost:11434/v1 — local, free)
  - anthropic    (claude via official SDK)
  - openai       (or any other OpenAI-compatible endpoint)

Backend is configured in configs/sources.yaml under the `llm:` key.
"""

from __future__ import annotations

import logging
import os
import re
import json
from pathlib import Path
from typing import Any, Iterator

import yaml

from src.agent.structured_decision import DECISION_TOOL, normalize_decision, parse_json_object

LOGGER = logging.getLogger(__name__)

# Auto-load .env from project root so SILICONFLOW_API_KEY etc. are available
# without needing `export` in the terminal every time.
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
except ImportError:
    pass  # python-dotenv not installed, env vars must be set manually

_CONFIG_CACHE: dict | None = None

_SYSTEM_PROMPT = """你是一位专业的A股量化投研分析师，具备以下核心能力：

1. 政策与产业分析
   - 准确解读政府政策（发改委、财政部、证监会）对产业链和个股的具体影响
   - 区分政策表态与落地执行，识别政策红利力度和持续性

2. 公司基本面分析
   - 深度解读年报、季报、公告中的经营数据：营收、毛利率、订单、产能、核心客户
   - 识别公告中的积极信号（产品放量、大客户突破、产能扩张）和风险信号（价格下滑、竞争加剧）

3. 量化因子信号解读
   - EP/BP/ROE（价值因子）：反映当前估值水平
   - EPS修正因子（fac_eps_revision_60d）：分析师预期变化，领先指标
   - 动量因子（近20/60日涨跌幅）：趋势强度
   - 波动率/回撤：短期风险量化

4. 综合研判
   - 量化信号与基本面证据须相互印证
   - 主动识别证据不足的维度

分析准则：
- 严格基于给定证据材料，每个核心观点必须引用具体来源（格式：【来源：文档标题】）
- 语言专业简洁，500字以内，不使用Markdown标题格式
- 分析末尾必须单独一行输出结论，格式严格为以下三选一：
  【结论】Buy
  【结论】Watch
  【结论】Avoid"""


_STRUCTURED_SYSTEM_PROMPT = """你是一位专业的A股量化投研分析师。

你必须基于给定证据输出结构化投资决策，不得编造证据。每个关键判断应能对应到输入材料中的文档标题、结构化行情数据、因子数据或同业对比。

决策口径：
- Buy：基本面、预期修正、趋势或估值风险收益比同时有较强支撑，且主要风险可控。
- Watch：有关注价值，但证据不完整、短期涨幅过大、估值偏高或风险收益比一般。
- Avoid：风险信号、趋势压力、基本面恶化或证据缺口明显，不适合积极介入。

请通过 submit_investment_decision 函数提交最终结构化结果。"""


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def _load_llm_config() -> dict[str, Any]:
    """Load llm section from sources.yaml. Cached after first read."""
    global _CONFIG_CACHE
    if _CONFIG_CACHE is not None:
        return _CONFIG_CACHE

    config_path = Path(__file__).resolve().parents[2] / "configs" / "sources.yaml"
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as f:
            full = yaml.safe_load(f) or {}
        _CONFIG_CACHE = full.get("llm", {})
    else:
        _CONFIG_CACHE = {}
    return _CONFIG_CACHE


def get_llm_config() -> dict[str, Any]:
    """Return effective LLM config with defaults."""
    cfg = _load_llm_config()
    return {
        "provider": cfg.get("provider", "siliconflow"),
        "model": cfg.get("model", "Qwen/Qwen2.5-72B-Instruct"),
        "base_url": cfg.get("base_url", "https://api.siliconflow.cn/v1"),
        "max_tokens": int(cfg.get("max_tokens", 1024)),
        "api_key_env": cfg.get("api_key_env", "SILICONFLOW_API_KEY"),
    }


# ---------------------------------------------------------------------------
# Evidence context builder  (provider-agnostic)
# ---------------------------------------------------------------------------

def _fmt(v: float | None, pct: bool = False) -> str:
    if v is None:
        return "N/A"
    return f"{v * 100:.2f}%" if pct else f"{v:.4f}"


def _build_evidence_context(
    policy_docs: list[dict[str, Any]],
    company_docs: list[dict[str, Any]],
    market: dict[str, Any],
    factor: dict[str, Any],
    risk: dict[str, Any],
    company_name: str,
    news_docs: list[dict[str, Any]] | None = None,
) -> str:
    parts: list[str] = []

    if news_docs:
        parts.append("【近期新闻动态（短线信号）】")
        for i, doc in enumerate(news_docs[:5], 1):
            title = doc.get("title") or "未命名"
            pub = doc.get("published_at") or ""
            content = (doc.get("content") or "").replace("\n", " ")[:300]
            parts.append(f"  [{i}] {title}（{pub[:10]}）\n      {content}")

    if policy_docs:
        parts.append("\n【政策与产业证据】")
        for i, doc in enumerate(policy_docs[:5], 1):
            title = doc.get("title") or "未命名"
            source = doc.get("source") or "unknown"
            content = (doc.get("content") or "").replace("\n", " ")[:400]
            parts.append(f"  [{i}] {title}（{source}）\n      {content}")
    else:
        parts.append("\n【政策与产业证据】暂无相关政策证据，政策面判断依据不足")

    if company_docs:
        parts.append("\n【公司公告与年报证据】")
        for i, doc in enumerate(company_docs[:5], 1):
            title = doc.get("title") or "未命名"
            source = doc.get("source") or "unknown"
            content = (doc.get("content") or "").replace("\n", " ")[:400]
            parts.append(f"  [{i}] {title}（{source}）\n      {content}")
    else:
        parts.append("\n【公司公告与年报证据】暂无公司证据，个股基本面判断依据不足")

    parts.append("\n【量化信号摘要】")
    parts.append(f"  股票：{company_name}")
    parts.append(f"  最新收盘价：{market.get('close', 'N/A')}")
    parts.append(f"  近5日涨幅：{_fmt(market.get('return_5d'), True)}")
    parts.append(f"  近20日涨幅：{_fmt(market.get('return_20d'), True)}")
    parts.append(f"  近60日涨幅：{_fmt(market.get('return_60d'), True)}")
    parts.append(f"  是否站上20日均线：{market.get('above_ma20')}")
    parts.append(f"  是否站上60日均线：{market.get('above_ma60')}")
    parts.append(f"  综合因子得分：{_fmt(factor.get('composite_score'), True)}（{factor.get('signal', 'N/A')}）")
    parts.append(f"  风险等级：{risk.get('severity', 'N/A')}")
    parts.append(f"  20日年化波动率：{_fmt(market.get('volatility_20d_annualized'), True)}")
    parts.append(f"  近60日最大回撤：{_fmt(market.get('max_drawdown_60d'), True)}")
    parts.append(f"  近120日最大回撤：{_fmt(market.get('max_drawdown_120d'), True)}")
    flags = risk.get("flags") or []
    if flags:
        parts.append(f"  风险标签：{' | '.join(flags[:4])}")

    return "\n".join(parts)


def _build_user_message(question: str, context: str, peer_context: str = "") -> str:
    peer_section = f"\n{peer_context}" if peer_context else ""
    return (
        f"研究问题：{question}\n\n"
        f"{context}"
        f"{peer_section}\n\n"
        "请基于以上证据材料，给出专业的投研分析，并在最后一行单独输出结论标签。"
    )


_VERDICT_RE = re.compile(r"【结论】\s*(Buy|Watch|Avoid)", re.IGNORECASE)


def hyde_rewrite(query: str) -> str:
    """HyDE: generate a hypothetical answer document, use it as the retrieval query.

    Instead of embedding the user's question directly, we ask the LLM to write
    a short passage that *would* answer the question. That passage lives in the
    same vector space as real document chunks, so the FAISS search finds closer
    neighbours than the raw question would.
    """
    cfg = get_llm_config()
    prompt = (
        "假设你是一位A股金融分析师，请根据以下投资问题，"
        "写一段可能出现在研报或公告中、能回答该问题的文字（100字以内，使用专业术语，直接输出内容无需解释）：\n"
        f"{query}"
    )
    try:
        client = _make_openai_client(cfg)
        resp = client.chat.completions.create(
            model=cfg["model"],
            max_tokens=150,
            messages=[{"role": "user", "content": prompt}],
            stream=False,
        )
        result = (resp.choices[0].message.content or "").strip()
        LOGGER.info("HyDE rewrite: %r → %r", query[:60], result[:60])
        return result if result else query
    except Exception as exc:
        LOGGER.warning("HyDE rewrite failed (%s); falling back to original query", exc)
        return query


def parse_verdict(text: str) -> str:
    """Extract Buy/Watch/Avoid from LLM output. Returns 'Watch' if not found."""
    m = _VERDICT_RE.search(text)
    if m:
        return m.group(1).capitalize()
    # Fallback: scan for bare keywords at end of text
    tail = text[-200:].lower()
    if "buy" in tail or "看多" in tail:
        return "Buy"
    if "avoid" in tail or "回避" in tail or "看空" in tail:
        return "Avoid"
    return "Watch"


def strip_verdict_line(text: str) -> str:
    """Remove the 【结论】 line from display text."""
    return _VERDICT_RE.sub("", text).rstrip()


# ---------------------------------------------------------------------------
# Backend: OpenAI-compatible  (SiliconFlow / DeepSeek / Ollama / OpenAI)
# ---------------------------------------------------------------------------

def _make_openai_client(cfg: dict[str, Any]):
    try:
        from openai import OpenAI
    except ImportError:
        raise RuntimeError("openai 包未安装，请运行：pip install openai")

    api_key_env = cfg["api_key_env"]
    api_key = os.environ.get(api_key_env, "") if api_key_env else "ollama"

    # Ollama doesn't need a real key; SiliconFlow/DeepSeek do
    if not api_key and api_key_env:
        raise RuntimeError(
            f"未检测到 {api_key_env} 环境变量。\n"
            f"请运行：export {api_key_env}=你的API密钥"
        )

    return OpenAI(api_key=api_key or "local", base_url=cfg["base_url"])


def _openai_generate(question: str, context: str, cfg: dict[str, Any],
                     peer_context: str = "") -> str:
    client = _make_openai_client(cfg)
    response = client.chat.completions.create(
        model=cfg["model"],
        max_tokens=cfg["max_tokens"],
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_message(question, context, peer_context)},
        ],
        stream=False,
    )
    return response.choices[0].message.content or ""


def _openai_structured_decision(question: str, context: str, cfg: dict[str, Any],
                                peer_context: str = "") -> dict[str, Any]:
    """Generate a structured decision via OpenAI-compatible Function Calling."""
    client = _make_openai_client(cfg)
    response = client.chat.completions.create(
        model=cfg["model"],
        max_tokens=cfg["max_tokens"],
        messages=[
            {"role": "system", "content": _STRUCTURED_SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_message(question, context, peer_context)},
        ],
        tools=[DECISION_TOOL],
        tool_choice={"type": "function", "function": {"name": "submit_investment_decision"}},
        stream=False,
    )
    message = response.choices[0].message
    tool_calls = getattr(message, "tool_calls", None) or []
    if tool_calls:
        arguments = tool_calls[0].function.arguments or "{}"
        return normalize_decision(json.loads(arguments))

    # Some OpenAI-compatible providers silently ignore tools. Accept JSON text
    # as a second-best structured path before falling back to regex verdicts.
    parsed = parse_json_object(message.content or "")
    if parsed:
        return normalize_decision(parsed)
    raise RuntimeError("LLM did not return a function call or JSON object")


def _openai_stream(question: str, context: str, cfg: dict[str, Any],
                   peer_context: str = "") -> Iterator[str]:
    client = _make_openai_client(cfg)
    stream = client.chat.completions.create(
        model=cfg["model"],
        max_tokens=cfg["max_tokens"],
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_message(question, context, peer_context)},
        ],
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


# ---------------------------------------------------------------------------
# Backend: Anthropic  (kept for users who have Claude API)
# ---------------------------------------------------------------------------

def _anthropic_generate(question: str, context: str, cfg: dict[str, Any]) -> str:
    try:
        import anthropic
    except ImportError:
        raise RuntimeError("anthropic 包未安装，请运行：pip install anthropic")

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("未检测到 ANTHROPIC_API_KEY 环境变量")

    client = anthropic.Anthropic()
    with client.messages.stream(
        model=cfg["model"],
        max_tokens=cfg["max_tokens"],
        system=[{"type": "text", "text": _SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": _build_user_message(question, context)}],
    ) as stream:
        final = stream.get_final_message()
        usage = final.usage
        LOGGER.info(
            "Anthropic LLM complete — input=%d (cached=%d) output=%d",
            usage.input_tokens,
            getattr(usage, "cache_read_input_tokens", 0),
            usage.output_tokens,
        )
        return final.content[0].text


def _anthropic_stream(question: str, context: str, cfg: dict[str, Any]) -> Iterator[str]:
    try:
        import anthropic
    except ImportError:
        raise RuntimeError("anthropic 包未安装，请运行：pip install anthropic")

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("未检测到 ANTHROPIC_API_KEY 环境变量")

    client = anthropic.Anthropic()
    with client.messages.stream(
        model=cfg["model"],
        max_tokens=cfg["max_tokens"],
        system=[{"type": "text", "text": _SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": _build_user_message(question, context)}],
    ) as stream:
        for text in stream.text_stream:
            yield text


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_investment_analysis(
    question: str,
    policy_docs: list[dict[str, Any]],
    company_docs: list[dict[str, Any]],
    market: dict[str, Any],
    factor: dict[str, Any],
    risk: dict[str, Any],
    company_name: str,
    *,
    model: str | None = None,
    peer_context: str = "",
    news_docs: list[dict[str, Any]] | None = None,
) -> str:
    """Generate LLM investment analysis (non-streaming). Returns complete text."""
    cfg = get_llm_config()
    if model:
        cfg = {**cfg, "model": model}

    context = _build_evidence_context(policy_docs, company_docs, market, factor, risk, company_name, news_docs)
    LOGGER.info("Generating analysis via %s / %s", cfg["provider"], cfg["model"])

    try:
        if cfg["provider"] == "anthropic":
            return _anthropic_generate(question, context, cfg)
        else:
            return _openai_generate(question, context, cfg, peer_context)
    except Exception as exc:
        LOGGER.warning("LLM analysis failed: %s", exc)
        return f"⚠️ AI 分析生成失败：{exc}"


def _fallback_evidence_refs(
    policy_docs: list[dict[str, Any]],
    company_docs: list[dict[str, Any]],
    news_docs: list[dict[str, Any]] | None = None,
) -> list[str]:
    refs: list[str] = []
    for doc in (news_docs or [])[:2] + policy_docs[:3] + company_docs[:3]:
        title = str(doc.get("title") or "").strip()
        if title and title not in refs:
            refs.append(title)
    return refs


def generate_structured_investment_decision(
    question: str,
    policy_docs: list[dict[str, Any]],
    company_docs: list[dict[str, Any]],
    market: dict[str, Any],
    factor: dict[str, Any],
    risk: dict[str, Any],
    company_name: str,
    *,
    model: str | None = None,
    peer_context: str = "",
    news_docs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Generate a structured investment decision.

    OpenAI-compatible providers use Function Calling first. If the provider does
    not support tools, the function falls back to the existing text generation
    path and normalizes the regex verdict into the same JSON shape.
    """
    cfg = get_llm_config()
    if model:
        cfg = {**cfg, "model": model}

    context = _build_evidence_context(policy_docs, company_docs, market, factor, risk, company_name, news_docs)
    LOGGER.info("Generating structured decision via %s / %s", cfg["provider"], cfg["model"])

    if cfg["provider"] != "anthropic":
        try:
            return _openai_structured_decision(question, context, cfg, peer_context)
        except Exception as exc:
            LOGGER.warning("Structured Function Calling failed; falling back to text verdict: %s", exc)

    raw = generate_investment_analysis(
        question,
        policy_docs,
        company_docs,
        market,
        factor,
        risk,
        company_name,
        model=model,
        peer_context=peer_context,
        news_docs=news_docs,
    )
    verdict = parse_verdict(raw)
    analysis = strip_verdict_line(raw)
    confidence = 0.5 if raw.startswith("⚠️") else 0.6
    return normalize_decision(
        {
            "verdict": verdict,
            "confidence": confidence,
            "analysis": analysis,
            "key_reasons": [],
            "risk_factors": risk.get("flags") or [],
            "evidence_refs": _fallback_evidence_refs(policy_docs, company_docs, news_docs),
            "missing_info": ["结构化 Function Calling 未成功，已使用文本分析回退。"],
        },
        fallback_text=analysis,
    )


def stream_investment_analysis(
    question: str,
    policy_docs: list[dict[str, Any]],
    company_docs: list[dict[str, Any]],
    market: dict[str, Any],
    factor: dict[str, Any],
    risk: dict[str, Any],
    company_name: str,
    *,
    model: str | None = None,
    peer_context: str = "",
    news_docs: list[dict[str, Any]] | None = None,
) -> Iterator[str]:
    """Stream LLM investment analysis token-by-token (for Streamlit)."""
    cfg = get_llm_config()
    if model:
        cfg = {**cfg, "model": model}

    context = _build_evidence_context(policy_docs, company_docs, market, factor, risk, company_name, news_docs)
    LOGGER.info("Streaming analysis via %s / %s", cfg["provider"], cfg["model"])

    try:
        if cfg["provider"] == "anthropic":
            yield from _anthropic_stream(question, context, cfg)
        else:
            yield from _openai_stream(question, context, cfg, peer_context)
    except Exception as exc:
        LOGGER.warning("LLM streaming failed: %s", exc)
        yield f"\n⚠️ AI 分析生成失败：{exc}"

