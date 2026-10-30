"""Structured investment decision helpers.

The deterministic workflow still owns data collection. This module adds the
Function Calling layer: the LLM must return a machine-readable decision object
instead of a free-text verdict line that has to be parsed with regex.
"""

from __future__ import annotations

import json
import logging
from typing import Any


LOGGER = logging.getLogger(__name__)

VALID_VERDICTS = {"Buy", "Watch", "Avoid"}


INVESTMENT_DECISION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "verdict": {
            "type": "string",
            "enum": ["Buy", "Watch", "Avoid"],
            "description": "Final investment label.",
        },
        "confidence": {
            "type": "number",
            "minimum": 0,
            "maximum": 1,
            "description": "Confidence of the final verdict.",
        },
        "analysis": {
            "type": "string",
            "description": "Concise professional analysis in Chinese, within 500 Chinese characters.",
        },
        "key_reasons": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Main positive or neutral reasons supporting the verdict.",
        },
        "risk_factors": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Main risk factors and cautions.",
        },
        "evidence_refs": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Referenced source titles or structured data names.",
        },
        "missing_info": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Important missing evidence or uncertainty.",
        },
    },
    "required": [
        "verdict",
        "confidence",
        "analysis",
        "key_reasons",
        "risk_factors",
        "evidence_refs",
        "missing_info",
    ],
    "additionalProperties": False,
}


DECISION_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "submit_investment_decision",
        "description": "Submit the final structured investment decision for the research report.",
        "parameters": INVESTMENT_DECISION_SCHEMA,
    },
}


def _as_string_list(value: Any, max_items: int = 8) -> list[str]:
    if not isinstance(value, list):
        return []
    cleaned = [str(item).strip() for item in value if str(item).strip()]
    return cleaned[:max_items]


def normalize_decision(data: dict[str, Any] | None, *, fallback_text: str = "") -> dict[str, Any]:
    """Validate and normalize an LLM-produced decision dictionary."""
    data = data or {}
    verdict = str(data.get("verdict") or "").strip().capitalize()
    if verdict not in VALID_VERDICTS:
        verdict = "Watch"

    confidence_raw = data.get("confidence", 0.5)
    try:
        confidence = float(confidence_raw)
    except (TypeError, ValueError):
        confidence = 0.5
    confidence = max(0.0, min(1.0, confidence))

    analysis = str(data.get("analysis") or fallback_text or "").strip()
    if not analysis:
        analysis = "AI 未返回可用的结构化分析，已回退到中性观察结论。"

    return {
        "verdict": verdict,
        "confidence": confidence,
        "analysis": analysis,
        "key_reasons": _as_string_list(data.get("key_reasons")),
        "risk_factors": _as_string_list(data.get("risk_factors")),
        "evidence_refs": _as_string_list(data.get("evidence_refs"), max_items=12),
        "missing_info": _as_string_list(data.get("missing_info")),
    }


def parse_json_object(text: str) -> dict[str, Any] | None:
    """Parse a JSON object from model text, including fenced-code fallbacks."""
    text = (text or "").strip()
    if not text:
        return None
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                value = json.loads(text[start:end + 1])
                return value if isinstance(value, dict) else None
            except json.JSONDecodeError:
                LOGGER.debug("Failed to parse embedded JSON decision", exc_info=True)
    return None


def render_decision_markdown(decision: dict[str, Any]) -> str:
    """Render a structured decision as the report's analysis section."""
    normalized = normalize_decision(decision)
    lines = [normalized["analysis"].strip(), ""]
    lines.append(
        f"结构化结论：**{normalized['verdict']}** "
        f"（置信度 {normalized['confidence']:.0%}）"
    )

    if normalized["key_reasons"]:
        lines.append("")
        lines.append("关键依据：")
        for reason in normalized["key_reasons"]:
            lines.append(f"- {reason}")

    if normalized["risk_factors"]:
        lines.append("")
        lines.append("主要风险：")
        for risk in normalized["risk_factors"]:
            lines.append(f"- {risk}")

    if normalized["missing_info"]:
        lines.append("")
        lines.append("证据缺口：")
        for gap in normalized["missing_info"]:
            lines.append(f"- {gap}")

    if normalized["evidence_refs"]:
        lines.append("")
        refs = "；".join(normalized["evidence_refs"])
        lines.append(f"引用证据：{refs}")

    return "\n".join(lines).strip() + "\n"

