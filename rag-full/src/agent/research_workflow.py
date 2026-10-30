"""Deterministic investment research workflow.

This is the first complete end-to-end layer above RAG and market data. It does
not require an LLM: the workflow calls retrieval and market tools, applies
simple decision rules, and emits a structured Markdown report with sources.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from src.agent.evidence_synthesizer import render_synthesis_markdown, synthesize_evidence
from src.agent.llm_synthesizer import generate_structured_investment_decision
from src.agent.structured_decision import render_decision_markdown
from src.agent.trace import WorkflowTrace
from src.agent.tools import (
    get_eps_revision_spread_analysis,
    get_analyst_consensus,
    get_factor_data,
    get_market_data,
    get_peer_comparison,
    get_risk_data,
    search_company_docs,
    search_company_risk_docs,
    search_news_docs,
    search_policy_docs,
)
from src.market.analyst_consensus import render_consensus_section
from src.market.data_loader import get_stock_info, normalize_ts_code
from src.market.technical import get_price_history


LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def fmt_pct(value: float | None) -> str:
    """Format decimal percentage values."""
    if value is None:
        return "N/A"
    return f"{value * 100:.2f}%"


def fmt_num(value: float | None, digits: int = 2) -> str:
    """Format numbers with missing-value handling."""
    if value is None:
        return "N/A"
    return f"{value:.{digits}f}"


def decide_conclusion(market: dict[str, Any], factor: dict[str, Any], risk: dict[str, Any]) -> dict[str, str]:
    """Convert signals into Buy / Watch / Avoid."""
    score = 0

    factor_score = factor.get("composite_score")
    if factor_score is not None:
        if factor_score >= 0.65:
            score += 2
        elif factor_score >= 0.50:
            score += 1
        elif factor_score < 0.35:
            score -= 1

    if market.get("trend_score", 0) >= 2:
        score += 1
    if market.get("return_20d") is not None and market["return_20d"] > 0.25:
        score -= 1
    if risk.get("severity") == "high":
        score -= 2
    elif risk.get("severity") == "medium":
        score -= 1

    trend_score = market.get("trend_score", 0)
    if score >= 3 and risk.get("severity") == "low":
        label = "Buy"
        reason = "因子和趋势同时偏强，且风险未显著压制。"
    elif score >= 0 or (trend_score >= 2 and factor_score is not None and factor_score >= 0.45 and risk.get("severity") != "high"):
        label = "Watch"
        reason = "主题和趋势仍有关注价值，但短期风险收益比需要更多证据确认，适合跟踪或分批关注。"
    else:
        label = "Avoid"
        reason = "风险信号或趋势压力较强，当前不适合积极介入。"
    return {"label": label, "reason": reason, "score": str(score)}


def summarize_evidence(items: list[dict[str, Any]], max_items: int = 5) -> str:
    """Render retrieval evidence as Markdown bullets."""
    if not items:
        return "- 暂无可用 RAG 证据。需要先运行采集、转换和建库脚本。\n"

    lines = []
    for idx, item in enumerate(items[:max_items], start=1):
        title = item.get("title") or "未命名文档"
        source = item.get("source") or "unknown"
        url = item.get("url") or item.get("source_file") or ""
        preview = (item.get("content") or "").replace("\n", " ")[:180]
        lines.append(f"- [{idx}] {title}（source={source}）")
        if url:
            lines.append(f"  - 来源：{url}")
        if item.get("matched_terms"):
            matched = "、".join(item.get("matched_terms", [])[:8])
            lines.append(f"  - 匹配词：{matched}")
        if item.get("retriever"):
            lines.append(f"  - 检索器：{item.get('retriever')}，score={fmt_num(item.get('score'), 2)}")
        if item.get("source_group"):
            lines.append(f"  - 来源组：{item.get('source_group')}")
        if preview:
            lines.append(f"  - 摘要：{preview}")
    return "\n".join(lines) + "\n"


def render_factor_table(factor: dict[str, Any], max_rows: int = 8) -> str:
    """Render factor details as a Markdown table."""
    rows = factor.get("factor_details", [])[:max_rows]
    if not rows:
        return "暂无因子数据。\n"
    lines = [
        "| 因子 | 原始值 | 截面分位 | 方向后得分 | RankIC | RankICIR |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        percentile = row.get("percentile")
        oriented = row.get("oriented_score")
        lines.append(
            "| {factor} | {value} | {pct} | {score} | {rank_ic} | {rank_icir} |".format(
                factor=row.get("factor"),
                value=fmt_num(row.get("value"), 4),
                pct=fmt_pct(percentile),
                score=fmt_pct(oriented),
                rank_ic=fmt_num(row.get("rank_ic"), 4),
                rank_icir=fmt_num(row.get("rank_icir"), 4),
            )
        )
    return "\n".join(lines) + "\n"


def render_python_analysis(python_analysis: dict[str, Any] | None) -> str:
    """Render controlled Python analysis output."""
    if not python_analysis:
        return "_未运行 Python 动态分析。_\n"
    if not python_analysis.get("ok"):
        return f"_Python 动态分析失败：{python_analysis.get('error') or 'unknown error'}_\n"

    result = python_analysis.get("result") or {}
    if not isinstance(result, dict):
        return f"`result`：{result}\n"

    spread = result.get("avg_spread")
    up = result.get("avg_top20_future_5d")
    down = result.get("avg_bottom20_future_5d")
    pos_ratio = result.get("positive_spread_ratio")
    start = result.get("start_date") or "N/A"
    end = result.get("end_date") or "N/A"
    n_days = result.get("n_days") or 0

    lines = [
        "**受控 Python 工具验证：EPS 修正因子的历史区分度**",
        "",
        f"- 样本区间：{start} ~ {end}，有效交易日：{n_days}",
        f"- EPS 修正 Top20% 组合未来5日平均收益：{fmt_pct(up)}",
        f"- EPS 修正 Bottom20% 组合未来5日平均收益：{fmt_pct(down)}",
        f"- 多空收益差（Top20% - Bottom20%）：{fmt_pct(spread)}",
        f"- 收益差为正的交易日占比：{fmt_pct(pos_ratio)}",
        "",
        "> 该分析由本地受控 Python 执行器运行，只读 `panel_with_factors.parquet`，用于补充说明因子信号是否具备历史统计支撑。",
    ]
    return "\n".join(lines) + "\n"


def render_trace_summary(trace: dict[str, Any] | None) -> str:
    """Render compact workflow trace metrics."""
    if not trace:
        return "_未记录 workflow trace。_\n"
    lines = [
        f"- Run ID：`{trace.get('run_id')}`",
        f"- 工具调用次数：{trace.get('tool_call_count', 0)}",
        f"- 失败工具数：{trace.get('failed_tool_count', 0)}",
        f"- 记录耗时合计：{trace.get('total_recorded_latency_ms', 0)} ms",
    ]
    events = trace.get("events") or []
    if events:
        lines.append("- 工具明细：" + " → ".join(str(event.get("tool")) for event in events[:16]))
    return "\n".join(lines) + "\n"


def collect_research_payload(
    stock_code: str,
    theme: str,
    question: str,
    *,
    top_k: int = 5,
    include_llm_analysis: bool = True,
    llm_model: str | None = None,  # None = read from configs/sources.yaml llm.model
) -> dict[str, Any]:
    """Run all tools once and return structured research payload.

    Args:
        include_llm_analysis: When True (default), calls the configured LLM to generate an
            AI narrative. Set to False in Streamlit to stream the analysis
            separately via stream_investment_analysis().
        llm_model: Optional model ID override for LLM analysis.
    """
    trace = WorkflowTrace(stock_code=stock_code, theme=theme, question=question)
    ts_code = normalize_ts_code(stock_code)
    info = trace.call("get_stock_info", get_stock_info, ts_code)
    market = trace.call("get_market_data", get_market_data, ts_code)
    factor = trace.call("get_factor_data", get_factor_data, ts_code)
    risk = trace.call("get_risk_data", get_risk_data, ts_code)
    policy_docs = trace.call("search_policy_docs", search_policy_docs, theme, question, top_k=top_k)
    company_docs = trace.call("search_company_docs", search_company_docs, ts_code, theme, top_k=top_k)
    company_risk_docs = trace.call("search_company_risk_docs", search_company_risk_docs, ts_code, theme, top_k=3)
    news_docs = trace.call("search_news_docs", search_news_docs, ts_code, theme, top_k=5)
    analyst_consensus = trace.call("get_analyst_consensus", get_analyst_consensus, ts_code)
    conclusion = decide_conclusion(market, factor, risk)
    price_history = trace.call("get_price_history", get_price_history, ts_code)
    company_name = info.get("name") or ts_code
    synthesis = synthesize_evidence(
        policy_docs,
        company_docs,
        theme=theme,
        company_name=company_name,
        risk_docs=company_risk_docs,
    )
    trace.add_event("synthesize_evidence", synthesis)

    peer_context = trace.call("get_peer_comparison", get_peer_comparison, ts_code)
    python_analysis = trace.call("get_eps_revision_spread_analysis", get_eps_revision_spread_analysis)

    llm_analysis = ""
    llm_decision = None
    llm_verdict = None
    if include_llm_analysis:
        LOGGER.info("Generating LLM investment analysis via %s...", llm_model)
        llm_decision = trace.call(
            "generate_structured_investment_decision",
            generate_structured_investment_decision,
            question,
            policy_docs,
            company_docs,
            market,
            factor,
            risk,
            company_name,
            model=llm_model,
            peer_context=peer_context,
            news_docs=news_docs,
        )
        llm_verdict = llm_decision.get("verdict")
        llm_analysis = render_decision_markdown(llm_decision)

    payload = {
        "stock_code": ts_code,
        "theme": theme,
        "question": question,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "info": info,
        "market": market,
        "factor": factor,
        "risk": risk,
        "policy_docs": policy_docs,
        "company_docs": company_docs,
        "company_risk_docs": company_risk_docs,
        "news_docs": news_docs,
        "analyst_consensus": analyst_consensus,
        "conclusion": conclusion,
        "price_history": price_history,
        "synthesis": synthesis,
        "llm_analysis": llm_analysis,
        "llm_decision": llm_decision,
        "llm_verdict": llm_verdict,
        "peer_context": peer_context,
        "python_analysis": python_analysis,
        "trace": trace.to_dict(),
    }
    # Override rule-based conclusion with LLM verdict when available
    if llm_verdict:
        payload["conclusion"]["label"] = llm_verdict
        confidence = llm_decision.get("confidence") if llm_decision else None
        confidence_text = f"，置信度 {confidence:.0%}" if isinstance(confidence, (int, float)) else ""
        payload["conclusion"]["reason"] = f"由 AI Function Calling 结构化判断{confidence_text}（见第6节分析）"
    payload["report"] = render_research_report(payload)
    return payload


def render_research_report(payload: dict[str, Any]) -> str:
    """Render a structured payload as Markdown."""
    ts_code = payload["stock_code"]
    theme = payload["theme"]
    question = payload["question"]
    info = payload["info"]
    market = payload["market"]
    factor = payload["factor"]
    risk = payload["risk"]
    policy_docs = payload["policy_docs"]
    company_docs = payload["company_docs"]
    news_docs = payload.get("news_docs") or []
    analyst_consensus = payload.get("analyst_consensus") or {}
    conclusion = payload["conclusion"]
    synthesis = payload.get("synthesis", {})
    python_analysis = payload.get("python_analysis")
    trace = payload.get("trace")

    company_name = info.get("name") or ts_code
    risk_flags = risk.get("flags") or ["未触发显著技术风险标签。"]
    llm_analysis = payload.get("llm_analysis", "")

    if llm_analysis and not llm_analysis.startswith("⚠️"):
        section4 = llm_analysis + "\n"
        section4_note = "本节由配置的 LLM 基于检索证据自动生成，每个观点均引用具体来源。"
    else:
        section4 = render_synthesis_markdown(synthesis)
        section4_note = "本节为规则化证据综合（LLM 分析未启用或生成失败，请检查 configs/sources.yaml 与对应 API Key）。"

    report = f"""# AI 投研报告：{company_name}（{ts_code}） - {theme}

生成时间：{payload["generated_at"]}

用户问题：{question}

## 1. 最终结论

结论：**{conclusion["label"]}**

判断理由：{conclusion["reason"]}

规则评分：{conclusion["score"]}

## 2. 公司基本情况

- 股票代码：{ts_code}
- 公司名称：{company_name}
- 所属行业：{info.get("industry") or "N/A"}
- 地区：{info.get("area") or "N/A"}
- 市场：{info.get("market") or "N/A"}
- 上市日期：{info.get("list_date") or "N/A"}

## 3. 最近行情表现

- 数据日期：{market.get("as_of")}
- 最新收盘价：{fmt_num(market.get("close"))}
- 近5日收益：{fmt_pct(market.get("return_5d"))}
- 近20日收益：{fmt_pct(market.get("return_20d"))}
- 近60日收益：{fmt_pct(market.get("return_60d"))}
- 20日成交量比：{fmt_num(market.get("volume_ratio_20d"))}
- 是否站上20日均线：{market.get("above_ma20")}
- 是否站上60日均线：{market.get("above_ma60")}

## 4. 近期新闻动态

{summarize_evidence(news_docs) if news_docs else "_暂无近期新闻（运行 scripts/fetch_news_daily.py 后可获取）_"}

## 5. 机构研究共识

{render_consensus_section(analyst_consensus)}

## 6. AI 投研综合叙述：{theme}

> {section4_note}

{section4}
{payload.get("peer_context", "")}

## 7. 政策和产业支持证据

{summarize_evidence(policy_docs)}

## 8. 公司公告 / 年报 / 投资者关系证据

{summarize_evidence(company_docs)}

## 9. 因子信号

- 因子日期：{factor.get("as_of")}
- 综合因子得分：{fmt_pct(factor.get("composite_score"))}
- 因子信号：{factor.get("signal")}

{render_factor_table(factor)}

## 10. Python 动态分析

{render_python_analysis(python_analysis)}

## 11. 风险提示

- 风险等级：{risk.get("severity")}
- 20日年化波动率：{fmt_pct(market.get("volatility_20d_annualized"))}
- 近60日最大回撤：{fmt_pct(market.get("max_drawdown_60d"))}
- 近120日最大回撤：{fmt_pct(market.get("max_drawdown_120d"))}

风险标签：

{chr(10).join(f"- {flag}" for flag in risk_flags)}

## 12. 证据来源

机构共识覆盖：{analyst_consensus.get('coverage_count', 0)} 家研报　近30天：{analyst_consensus.get('recent_30d_count', 0)} 篇

近期新闻数量：{len(news_docs)}

政策/产业证据数量：{len(policy_docs)}

公司证据数量：{len(company_docs)}

结构化行情与因子数据来源：`../tushare_quant/quant_data_6y/panel_with_factors.parquet`

## 13. Workflow Trace

{render_trace_summary(trace)}

## 14. 口径说明

- 当前结论是规则驱动的研究辅助结论，不构成投资建议。
- 行情和因子数据来自本地历史面板，最新日期取决于 parquet 文件更新时间。
- RAG 证据依赖 `data/processed/markdown/` 和 `data/index/faiss/`，如未建库会退回本地 Markdown 关键词检索。
- 当前 workflow 已接入结构化 Function Calling 决策层、受控 Python 动态分析工具和调用 trace；实时行情和实时公告仍需单独接入。
"""
    return report


def generate_research_report(
    stock_code: str,
    theme: str,
    question: str,
    *,
    top_k: int = 5,
) -> str:
    """Run the full workflow and return a Markdown report."""
    return collect_research_payload(stock_code, theme, question, top_k=top_k)["report"]


def save_report(report: str, stock_code: str, theme: str, output_dir: str | Path = PROJECT_ROOT / "reports") -> Path:
    """Save report Markdown to disk."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_theme = "".join(ch if ch.isalnum() else "_" for ch in theme)[:40]
    path = output_dir / f"research_{normalize_ts_code(stock_code).replace('.', '_')}_{safe_theme}.md"
    path.write_text(report, encoding="utf-8")
    LOGGER.info("Saved research report: %s", path)
    return path

