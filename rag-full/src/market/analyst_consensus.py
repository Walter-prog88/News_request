"""Load analyst consensus facts and compute a structured verdict.

Reads from data/facts/stock_consensus.parquet produced by
scripts/extract_report_facts.py.

Verdict logic (rule-based, no LLM):
  score accumulates from coverage, rating distribution, trend, and EPS stability.
  利好  score >= 2
  偏多  score in [1, 2)
  中性  score in (-1, 1)
  偏空  score in [-2, -1]
  利空  score < -2
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CONSENSUS_PATH = PROJECT_ROOT / "data" / "facts" / "stock_consensus.parquet"


@lru_cache(maxsize=1)
def _load_consensus():
    """Load consensus parquet once and cache in memory."""
    import pandas as pd
    if not _CONSENSUS_PATH.exists():
        LOGGER.warning("Consensus file not found: %s. Run scripts/extract_report_facts.py first.", _CONSENSUS_PATH)
        return pd.DataFrame()
    return pd.read_parquet(_CONSENSUS_PATH).set_index("ts_code")


def _safe(val, default=None):
    """Return default if val is NaN or None."""
    import math
    if val is None:
        return default
    try:
        if math.isnan(float(val)):
            return default
    except (TypeError, ValueError):
        pass
    return val


def _compute_verdict(row: dict) -> dict[str, Any]:
    """
    Rule-based verdict from consensus fields.
    Returns dict with: verdict, score, reasons, cautions.
    """
    coverage = _safe(row.get("coverage_count"), 0)
    pos_pct = _safe(row.get("positive_pct"), 0.0)
    neg_pct = _safe(row.get("negative_pct"), 0.0)
    trend = row.get("rating_trend") or "数据不足"
    recent_pos = _safe(row.get("recent_positive_pct"))
    eps_mean = _safe(row.get("eps_y1_mean"))
    eps_std = _safe(row.get("eps_y1_std"))

    if coverage < 2:
        return {
            "verdict": "数据不足",
            "score": 0,
            "reasons": [f"仅 {coverage} 家机构覆盖，无法形成共识判断。"],
            "cautions": [],
        }

    score = 0.0
    reasons: list[str] = []
    cautions: list[str] = []

    # ── 1. 评级分布 ──────────────────────────────────────────────────────
    if pos_pct >= 80:
        score += 2
        reasons.append(f"{pos_pct:.0f}% 机构持买入/增持评级，机构认可度高")
    elif pos_pct >= 65:
        score += 1
        reasons.append(f"{pos_pct:.0f}% 机构持正面评级，整体偏乐观")
    elif pos_pct < 35:
        score -= 2
        reasons.append(f"仅 {pos_pct:.0f}% 机构持正面评级，机构态度偏谨慎")
    elif pos_pct < 50:
        score -= 1
        reasons.append(f"{pos_pct:.0f}% 机构持正面评级，多空分歧较大")
    else:
        reasons.append(f"{pos_pct:.0f}% 机构持正面评级，观点中性")

    if neg_pct > 20:
        score -= 1
        cautions.append(f"{neg_pct:.0f}% 机构持卖出/减持评级，存在明确看空声音")
    elif neg_pct > 10:
        cautions.append(f"{neg_pct:.0f}% 机构持负面评级，需关注分歧")

    # ── 2. 评级趋势 ──────────────────────────────────────────────────────
    if trend == "升级":
        score += 1
        if recent_pos is not None:
            reasons.append(
                f"近 30 天评级趋势升级（近期正面率 {recent_pos:.0f}% vs 历史 {pos_pct:.0f}%），机构态度改善"
            )
        else:
            reasons.append("近期出现评级上调，机构态度改善")
    elif trend == "降级":
        score -= 1
        if recent_pos is not None:
            cautions.append(
                f"近 30 天评级趋势降级（近期正面率 {recent_pos:.0f}% vs 历史 {pos_pct:.0f}%），需警惕情绪转向"
            )
        else:
            cautions.append("近期出现评级下调，需关注风险")
    elif trend == "稳定":
        reasons.append("近期评级趋势稳定，机构观点无明显变化")

    # ── 3. EPS 预测分歧度 ────────────────────────────────────────────────
    if eps_mean and eps_mean > 0 and eps_std is not None:
        cv = eps_std / eps_mean  # coefficient of variation
        if cv < 0.08:
            score += 0.5
            reasons.append(f"机构 EPS 预测分歧小（CV={cv:.1%}），盈利预期一致性高")
        elif cv > 0.25:
            score -= 0.5
            cautions.append(f"机构 EPS 预测分歧大（CV={cv:.1%}），盈利可见度存疑")

    # ── 最终判断 ─────────────────────────────────────────────────────────
    if score >= 2:
        verdict = "利好"
    elif score >= 1:
        verdict = "偏多"
    elif score <= -2:
        verdict = "利空"
    elif score <= -1:
        verdict = "偏空"
    else:
        verdict = "中性"

    return {
        "verdict": verdict,
        "score": round(score, 1),
        "reasons": reasons,
        "cautions": cautions,
    }


def get_analyst_consensus(ts_code: str) -> dict[str, Any]:
    """Return consensus facts + verdict for one stock. Empty dict if no data."""
    df = _load_consensus()
    if df.empty or ts_code not in df.index:
        return {}

    row = df.loc[ts_code].to_dict()
    verdict_info = _compute_verdict(row)

    return {
        "ts_code": ts_code,
        "name": row.get("name"),
        "coverage_count": int(_safe(row.get("coverage_count"), 0)),
        "recent_30d_count": int(_safe(row.get("recent_30d_count"), 0)),
        "latest_report_date": str(row.get("latest_report_date") or ""),
        "positive_pct": _safe(row.get("positive_pct")),
        "negative_pct": _safe(row.get("negative_pct")),
        "neutral_pct": _safe(row.get("neutral_pct")),
        "recent_positive_pct": _safe(row.get("recent_positive_pct")),
        "rating_trend": row.get("rating_trend") or "数据不足",
        "eps_y1_mean": _safe(row.get("eps_y1_mean")),
        "eps_y1_std": _safe(row.get("eps_y1_std")),
        "eps_y1_year": _safe(row.get("eps_y1_year")),
        "pe_y1_mean": _safe(row.get("pe_y1_mean")),
        "target_price_mean": _safe(row.get("target_price_mean")),
        "target_price_count": int(_safe(row.get("target_price_count"), 0)),
        **verdict_info,
    }


def render_consensus_section(c: dict[str, Any]) -> str:
    """Render the analyst consensus as a Markdown section."""
    if not c:
        return "_暂无机构研究数据（运行 scripts/extract_report_facts.py 可生成）_\n"

    verdict = c.get("verdict", "数据不足")
    verdict_icon = {"利好": "🟢", "偏多": "🔵", "中性": "⚪", "偏空": "🟡", "利空": "🔴", "数据不足": "⚫"}.get(verdict, "⚪")

    lines: list[str] = []

    # ── 核心指标一览 ──────────────────────────────────────────────────────
    cov = c.get("coverage_count", 0)
    rec = c.get("recent_30d_count", 0)
    latest = c.get("latest_report_date", "")
    pos = c.get("positive_pct")
    neg = c.get("negative_pct")
    neu = c.get("neutral_pct")

    lines.append(
        f"**机构覆盖**：{cov} 家（近30天 {rec} 篇）　"
        f"**最新报告**：{latest}"
    )
    if pos is not None:
        lines.append(
            f"**评级分布**：正面 {pos:.0f}%　中性 {neu:.0f}%　负面 {neg:.0f}%　"
            f"**趋势**：{c.get('rating_trend', '—')}"
        )

    # ── EPS & PE ──────────────────────────────────────────────────────────
    eps = c.get("eps_y1_mean")
    eps_std = c.get("eps_y1_std")
    yr = c.get("eps_y1_year")
    pe = c.get("pe_y1_mean")
    if eps is not None:
        eps_str = f"¥{eps:.2f}"
        if eps_std:
            eps_str += f" ± {eps_std:.2f}"
        lines.append(
            f"**共识 EPS**：{yr:.0f}年预测均值 {eps_str}　"
            + (f"**当前正向 PE**：{pe:.1f}x" if pe else "")
        )

    lines.append("")

    # ── 综合结论 ──────────────────────────────────────────────────────────
    lines.append(f"### 综合结论：{verdict_icon} {verdict}")
    lines.append("")

    reasons = c.get("reasons") or []
    cautions = c.get("cautions") or []

    if reasons:
        lines.append("**支撑因素**")
        for r in reasons:
            lines.append(f"- {r}")
        lines.append("")

    if cautions:
        lines.append("**注意事项**")
        for ca in cautions:
            lines.append(f"- {ca}")
        lines.append("")

    lines.append(
        "> 以上结论基于知识库内研报的评级分布与趋势，为统计事实而非投资建议。"
        f"（评分：{c.get('score', 0):+.1f}）"
    )
    lines.append("")

    return "\n".join(lines)

