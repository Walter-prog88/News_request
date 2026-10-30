"""Streamlit demo for the FinResearch Agent workflow."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.llm_synthesizer import (
    get_llm_config, parse_verdict, strip_verdict_line, stream_investment_analysis,
)
from src.agent.research_workflow import (
    collect_research_payload,
    fmt_pct,
    render_research_report,
    save_report,
)
from src.market.data_loader import get_stock_info


def render_evidence_cards(title: str, items: list[dict], empty_text: str) -> None:
    """Render RAG evidence as expandable cards."""
    st.subheader(title)
    if not items:
        st.info(empty_text)
        return

    for idx, item in enumerate(items, start=1):
        card_title = item.get("title") or "未命名文档"
        source = item.get("source") or "unknown"
        with st.expander(f"{idx}. {card_title}  ·  {source}", expanded=idx <= 2):
            url = item.get("url") or item.get("source_file")
            if url:
                st.write(f"来源：{url}")
            meta_cols = st.columns(3)
            meta_cols[0].metric("Score", f"{float(item.get('score', 0.0)):.2f}")
            meta_cols[1].metric("Retriever", item.get("retriever") or "unknown")
            meta_cols[2].metric("Source Group", item.get("source_group") or "default")
            matched_terms = item.get("matched_terms") or []
            st.write("匹配词：" + ("、".join(matched_terms[:8]) if matched_terms else "N/A"))
            preview = (item.get("content") or "").replace("\n", " ")[:900]
            st.write(preview or "无摘要")


def render_factor_table(factor: dict) -> None:
    """Render factor details — shows raw and neutralized percentiles side by side."""
    rows = factor.get("factor_details", [])
    if not rows:
        st.info("暂无因子数据")
        return

    df = pd.DataFrame(rows)
    is_neutralized = factor.get("neutralized", False)

    # Build display columns
    cols = ["factor", "value"]
    if is_neutralized and "percentile_raw" in df.columns:
        cols += ["percentile_raw", "percentile", "oriented_score", "rank_ic", "rank_icir"]
    else:
        cols += ["percentile", "oriented_score", "rank_ic", "rank_icir"]

    display = df[[c for c in cols if c in df.columns]].copy()

    def fmt_pct_cell(x):
        return "N/A" if (x is None or (isinstance(x, float) and pd.isna(x))) else f"{x * 100:.1f}%"

    def fmt_num_cell(x):
        return "N/A" if (x is None or (isinstance(x, float) and pd.isna(x))) else f"{x:.4f}"

    for col in ["percentile", "percentile_raw", "oriented_score"]:
        if col in display.columns:
            display[col] = display[col].map(fmt_pct_cell)
    for col in ["value", "rank_ic", "rank_icir"]:
        if col in display.columns:
            display[col] = display[col].map(fmt_num_cell)

    rename = {
        "factor": "因子",
        "value": "原始值",
        "percentile_raw": "原始分位",
        "percentile": "中性化分位" if is_neutralized else "截面分位",
        "oriented_score": "方向后得分",
        "rank_ic": "RankIC",
        "rank_icir": "RankICIR",
    }
    display = display.rename(columns=rename)
    st.dataframe(display)

    if is_neutralized:
        st.caption("中性化分位：已剔除行业 + 市值效应，反映个股相对于同行业同规模公司的纯 alpha 信号")


def render_synthesis_section(title: str, items: list[dict]) -> None:
    """Render synthesized evidence bullets (rule-based fallback)."""
    st.markdown(f"#### {title}")
    if not items:
        st.info("暂无足够证据。")
        return
    for item in items:
        st.write(f"- {item['text']}")
        st.caption(f"来源：{item['title']} · {item['source']}")


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(page_title="FinResearch Agent", layout="wide")
st.title("FinResearch Agent")
st.caption("RAG 证据 + 配置化 LLM 分析 + 本地行情/因子信号 + 结构化投研报告")

# ---------------------------------------------------------------------------
# Sidebar inputs
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("输入")
    stock_code = st.text_input("股票代码", value="300308")
    theme = st.text_input("主题关键词", value="AI 光模块")
    question = st.text_area(
        "研究问题",
        value="中际旭创现在还值得关注吗？AI 光模块主题是否还能持续？",
        height=110,
    )
    top_k = st.slider("RAG 证据数量", min_value=1, max_value=8, value=5)

    st.markdown("---")
    _llm_cfg = get_llm_config()
    st.markdown("**AI 分析配置**")
    st.caption(f"后端：{_llm_cfg['provider']}  |  模型：{_llm_cfg['model']}")
    _key_env = _llm_cfg["api_key_env"]
    api_key_set = bool(os.environ.get(_key_env)) if _key_env else True
    if api_key_set:
        st.success(f"✓ {_key_env or '本地模型'} 已就绪")
    else:
        st.warning(f"⚠ 未设置 {_key_env}\n\n`export {_key_env}=你的key`")
    llm_model = None  # use config default

    run = st.button("生成投研报告")

# ---------------------------------------------------------------------------
# Stock info caption
# ---------------------------------------------------------------------------

if stock_code:
    try:
        _info = get_stock_info(stock_code)
        st.caption(
            f"当前股票：{_info.get('name') or '未知'} "
            f"({_info.get('ts_code')}) / {_info.get('industry') or '行业未知'}"
        )
    except Exception:
        pass

if not run:
    st.info('输入股票代码、主题和问题后点击"生成投研报告"。')
    st.markdown(
        """
### 当前能力

- **LLM 分析层**：基于 RAG 证据，由配置模型生成带引用的投研叙事（流式输出）
- **本地 6 年行情/因子面板**：技术信号、因子截面分位、风险标签
- **RAG 知识库**：政策文件 + 交易所公告 + 本地研究笔记
- **规则结论**：Buy / Watch / Avoid + 量化评分
"""
    )
    st.stop()

# ---------------------------------------------------------------------------
# Step 1: Fast data collection (no LLM call here)
# ---------------------------------------------------------------------------

with st.spinner("正在检索 RAG 证据、计算行情因子..."):
    try:
        payload = collect_research_payload(
            stock_code,
            theme,
            question,
            top_k=top_k,
            include_llm_analysis=False,  # stream LLM separately below
        )
    except Exception as exc:
        st.error(f"数据获取失败：{exc}")
        st.stop()

info = payload["info"]
market = payload["market"]
factor = payload["factor"]
risk = payload["risk"]
conclusion = payload["conclusion"]
history = payload["price_history"].copy()
synthesis = payload.get("synthesis", {})
company_name = info.get("name") or payload["stock_code"]

# Top metrics row
col1, col2, col3, col4 = st.columns(4)
col1.metric("规则结论", conclusion["label"])
col2.metric("近20日收益", fmt_pct(market.get("return_20d")))
col3.metric("近60日收益", fmt_pct(market.get("return_60d")))
col4.metric("风险等级", risk.get("severity", "unknown"))
st.markdown(f"**判断理由：** {conclusion['reason']}")

# ---------------------------------------------------------------------------
# Step 2: Stream LLM analysis
# ---------------------------------------------------------------------------

st.markdown("---")
st.subheader("LLM 投研分析")

if api_key_set:
    # st.write_stream is only available in Streamlit >= 1.31.
    # For older versions, we collect chunks manually and update a placeholder.
    _chunks: list[str] = []
    _placeholder = st.empty()
    for _chunk in stream_investment_analysis(
        question=question,
        policy_docs=payload["policy_docs"],
        company_docs=payload["company_docs"],
        market=market,
        factor=factor,
        risk=risk,
        company_name=company_name,
        model=llm_model,
        peer_context=payload.get("peer_context", ""),
    ):
        _chunks.append(_chunk)
        _placeholder.markdown("".join(_chunks))
    raw_llm = "".join(_chunks)
    llm_verdict = parse_verdict(raw_llm)
    llm_analysis_text = strip_verdict_line(raw_llm)
    # Override rule-based conclusion with LLM verdict
    if llm_verdict:
        conclusion["label"] = llm_verdict
        conclusion["reason"] = "由 AI 基于检索证据综合判断（见上方分析）"
else:
    st.info(
        f"设置 {_key_env} 环境变量后重启 Streamlit 以启用 LLM 分析。\n\n"
        '规则化证据综合可在下方"行情概览"标签页查看。'
    )
    llm_analysis_text = ""

# ---------------------------------------------------------------------------
# Step 3: Generate and save full report (now with LLM analysis)
# ---------------------------------------------------------------------------

payload["llm_analysis"] = llm_analysis_text
payload["report"] = render_research_report(payload)

try:
    path = save_report(payload["report"], stock_code, theme)
    st.success(f"完整报告已保存：{path}")
except Exception as exc:
    st.warning(f"报告保存失败：{exc}")
    path = None

report = payload["report"]

# ---------------------------------------------------------------------------
# Step 4: Tabs with detailed data
# ---------------------------------------------------------------------------

tab_overview, tab_evidence, tab_factors, tab_report = st.tabs(
    ["行情概览", "RAG 证据", "因子与风险", "Markdown 报告"]
)

with tab_overview:
    # Rule-based evidence summary as supplement to LLM analysis
    with st.expander("规则化证据综合（补充参考）", expanded=False):
        s1, s2 = st.columns(2)
        with s1:
            render_synthesis_section("主题支持要点", synthesis.get("theme_support") or [])
        with s2:
            render_synthesis_section("公司支撑要点", synthesis.get("company_support") or [])
        risk_points = synthesis.get("risk_points") or []
        if risk_points:
            render_synthesis_section("风险关注要点", risk_points)
        gaps = synthesis.get("evidence_gaps") or []
        if gaps:
            st.markdown("**证据缺口**")
            for gap in gaps:
                st.write(f"- {gap}")

    st.subheader(f"{company_name} 行情走势")
    chart_df = history.set_index("trade_date")[["close", "ma20", "ma60"]]
    st.line_chart(chart_df, use_container_width=False)

    st.subheader("成交量")
    volume_df = history.set_index("trade_date")[["volume"]]
    st.bar_chart(volume_df, use_container_width=False)

    c1, c2, c3 = st.columns(3)
    c1.metric("最新收盘价", f"{market.get('close', 0.0):.2f}")
    c2.metric("20日成交量比", f"{market.get('volume_ratio_20d', 0.0):.2f}")
    c3.metric("20日年化波动率", fmt_pct(market.get("volatility_20d_annualized")))

with tab_evidence:
    render_evidence_cards(
        "政策和产业支持证据",
        payload["policy_docs"],
        "暂无政策/产业证据。请先运行采集、转换和建库脚本。",
    )
    render_evidence_cards(
        "公司公告 / 年报 / 投资者关系证据",
        payload["company_docs"],
        "暂无公司证据。需要抓取对应股票公告或投资者关系记录。",
    )

with tab_factors:
    st.subheader("因子信号")
    c1, c2, c3 = st.columns(3)
    c1.metric("因子日期", factor.get("as_of", "N/A"))
    c2.metric("综合因子得分", fmt_pct(factor.get("composite_score")))
    c3.metric("因子信号", factor.get("signal", "unknown"))
    render_factor_table(factor)

    st.subheader("风险提示")
    flags = risk.get("flags") or ["未触发显著技术风险标签。"]
    for flag in flags:
        st.warning(flag)
    st.write(f"近60日最大回撤：{fmt_pct(market.get('max_drawdown_60d'))}")
    st.write(f"近120日最大回撤：{fmt_pct(market.get('max_drawdown_120d'))}")

with tab_report:
    if path:
        st.download_button(
            "下载 Markdown 报告",
            data=report,
            file_name=Path(path).name,
            mime="text/markdown",
        )
    st.markdown(report)

