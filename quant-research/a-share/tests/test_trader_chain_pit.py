from __future__ import annotations

import re

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from afactor.trader_chain_pit import (
    PIT_CHAIN_DEFINITIONS,
    TraderChainPITConfig,
    build_trader_chain_pit_membership,
    prepare_report_chain_evidence,
)


def _calendar() -> pd.DatetimeIndex:
    return pd.bdate_range("2026-08-03", periods=10)


def _reports(calendar: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "stockCode": [1, 1, 2, 3, 4],
            "publishDate": [
                calendar[0],
                calendar[2],
                calendar[1],
                calendar[1],
                calendar[1],
            ],
            "indvInduName": [
                "电子制造旧分类",
                "印制电路板新分类",
                "软件服务",
                "钢铁",
                "其他制造",
            ],
            "title": [
                "高多层PCB和覆铜板产能进入放量期",
                "PCB订单继续增长",
                "生成式AI软件平台加快商业化落地",
                "冷轧板和冷板价格上涨",
                "散热温控技术用于数据中心服务器",
            ],
            "stockName": ["样本甲", "样本甲", "样本乙", "样本丙", "样本丁"],
        }
    )


def test_frozen_dictionary_is_exactly_nine_general_market_rules() -> None:
    assert len(PIT_CHAIN_DEFINITIONS) == 9
    assert len({definition.theme_id for definition in PIT_CHAIN_DEFINITIONS}) == 9
    for definition in PIT_CHAIN_DEFINITIONS:
        terms = (
            *definition.keywords,
            *(
                term
                for rule in definition.cooccurrence_rules
                for group in rule
                for term in group
            ),
        )
        assert not any(re.fullmatch(r"\d{6}(?:\.(?:SH|SZ|BJ))?", term) for term in terms)


def test_report_evidence_is_strict_t_plus_one_and_keeps_context_guards() -> None:
    calendar = _calendar()
    evidence = prepare_report_chain_evidence(_reports(calendar), calendar)

    pcb = evidence.loc[
        evidence["symbol"].eq("000001.SZ") & evidence["chain_id"].eq("pcb")
    ].sort_values("report_publish_time")
    assert pcb.iloc[0]["report_publish_date"] == calendar[0]
    assert pcb.iloc[0]["evidence_effective_date"] == calendar[1]
    assert pcb.iloc[1]["evidence_effective_date"] == calendar[3]
    assert "覆铜板" in pcb.iloc[0]["matched_keywords"]
    assert "strict T+1" in pcb.iloc[0]["asof_rule"]

    liquid = evidence.loc[evidence["chain_id"].eq("liquid_cooling")]
    assert set(liquid["symbol"]) == {"000004.SZ"}


def test_membership_accumulates_active_sources_and_expires_by_session_ttl() -> None:
    calendar = _calendar()
    panel = pd.DataFrame(
        {
            "decision_date": [
                calendar[0],  # publication day: unavailable
                calendar[1],  # first source age 0
                calendar[2],  # second report published: still unavailable
                calendar[3],  # second source age 0; first source expired
                calendar[4],  # second source age 1
                calendar[5],  # second source expired
            ],
            "symbol": ["000001.SZ"] * 6,
        }
    )
    result = build_trader_chain_pit_membership(
        panel,
        _reports(calendar),
        calendar,
        config=TraderChainPITConfig(ttl_sessions=2),
    )
    pcb = result.loc[result["chain_id"].eq("pcb")].set_index("decision_date")

    assert calendar[0] not in pcb.index
    assert pcb.at[calendar[1], "membership"]
    assert pcb.at[calendar[1], "evidence_age"] == 0
    assert pcb.at[calendar[1], "source_count"] == 1
    assert pcb.at[calendar[1], "latest_report_fine_industry"] == "电子制造旧分类"
    assert pcb.at[calendar[2], "latest_report_fine_industry"] == "电子制造旧分类"
    assert pcb.at[calendar[3], "evidence_age_sessions"] == 0
    assert pcb.at[calendar[3], "source_count"] == 1
    assert pcb.at[calendar[3], "latest_report_fine_industry"] == "印制电路板新分类"
    assert pcb.at[calendar[4], "evidence_age"] == 1
    assert calendar[5] not in pcb.index


def test_source_count_counts_distinct_active_reports_and_age_uses_freshest() -> None:
    calendar = _calendar()
    reports = _reports(calendar)
    # Keep both PCB sources active and duplicate one exact source row.  The
    # duplicate must not inflate source_count.
    reports = pd.concat([reports, reports.iloc[[1]]], ignore_index=True)
    panel = pd.DataFrame(
        {"decision_date": [calendar[3]], "symbol": ["000001.SZ"]}
    )
    result = build_trader_chain_pit_membership(
        panel,
        reports,
        calendar,
        config=TraderChainPITConfig(ttl_sessions=10),
    ).set_index("chain_id")

    assert result.at["pcb", "source_count"] == 2
    assert result.at["pcb", "evidence_age"] == 0
    assert result.at["pcb", "latest_evidence_effective_date"] == calendar[3]


def test_panel_labels_are_ignored_and_report_label_columns_are_rejected() -> None:
    calendar = _calendar()
    panel = pd.DataFrame(
        {"decision_date": [calendar[3]], "symbol": ["000002.SZ"]}
    )
    baseline = build_trader_chain_pit_membership(
        panel, _reports(calendar), calendar
    )
    labelled_panel = panel.assign(
        is_selected=[True], event_id=["E1"], realised_pnl=[99.0]
    )
    changed = build_trader_chain_pit_membership(
        labelled_panel, _reports(calendar), calendar
    )
    assert_frame_equal(baseline, changed)

    contaminated_reports = _reports(calendar).assign(event_id="E1")
    with pytest.raises(ValueError, match="forbidden to read"):
        prepare_report_chain_evidence(contaminated_reports, calendar)


def test_config_validation_and_missing_text_are_explicit() -> None:
    with pytest.raises(ValueError, match="positive or None"):
        TraderChainPITConfig(ttl_sessions=0)

    calendar = _calendar()
    reports = _reports(calendar).drop(columns=["title", "indvInduName"])
    with pytest.raises(ValueError, match="title or fine-industry"):
        prepare_report_chain_evidence(reports, calendar)
