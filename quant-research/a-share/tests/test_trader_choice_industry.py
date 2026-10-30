from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from afactor.trader_choice_industry import (
    add_industry_context_features,
    add_report_industry_pit,
)


def _fixture_data():
    calendar = pd.bdate_range("2026-06-01", periods=35)
    decision = calendar[-1]
    prior = calendar[-2]
    panel = pd.DataFrame(
        {
            "decision_date": [decision, decision],
            "symbol": ["000001.SZ", "600000.SH"],
            "unrelated_factor": [1.0, 2.0],
        }
    )
    reports = pd.DataFrame(
        {
            "stockCode": ["000001", "000001", "600000"],
            "publishDate": [
                calendar[-4],
                decision,  # must not be visible until the following session
                calendar[-3],
            ],
            "indvInduName": ["银行旧标签", "未来标签", "银行细分"],
        }
    )
    static_sw = pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "600000.SH"],
            "sw_industry": ["银行", "银行"],
            "fine_industry": ["银行", "银行"],
        }
    )
    index_history = pd.DataFrame(
        {
            "trade_date": calendar,
            "sw_index_code": "801780",
            "sw_l1_name": "银行",
            "close": np.linspace(100.0, 134.0, len(calendar)),
            "amount": np.linspace(1_000.0, 2_000.0, len(calendar)),
        }
    )
    daily_basic = pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "600000.SH", "000001.SZ", "600000.SH"],
            "trade_date": [prior, prior, decision, decision],
            "turnover_rate": [1.1, 2.2, 91.0, 92.0],
            "turnover_rate_f": [1.2, 2.3, 91.1, 92.1],
            "volume_ratio": [1.3, 2.4, 91.2, 92.2],
            "circ_mv": [101.0, 202.0, 9_101.0, 9_202.0],
        }
    )
    tech = pd.DataFrame(
        {
            "日期": ["2026-08-21", "2026-08-21"],
            "成分券代码": ["000001", "000001"],
            "index_code": ["930713", "H30202"],
            "index_name": ["中证人工智能", "中证软件"],
        }
    )
    return (
        calendar,
        decision,
        prior,
        panel,
        reports,
        static_sw,
        index_history,
        daily_basic,
        tech,
    )


def test_report_industry_is_strict_t_plus_one_and_has_age():
    calendar, decision, _, panel, reports, *_ = _fixture_data()
    result = add_report_industry_pit(panel, reports, calendar).set_index("symbol")

    assert result.at["000001.SZ", "report_fine_industry_pit"] == "银行旧标签"
    assert result.at["600000.SH", "report_fine_industry_pit"] == "银行细分"
    effective = result.at["000001.SZ", "report_industry_effective_date"]
    assert effective == calendar[-3]
    assert result.at["000001.SZ", "report_industry_age_sessions"] == 2
    assert "strict T+1" in result.at["000001.SZ", "report_industry_asof_rule"]
    assert result.at["000001.SZ", "decision_date"] == decision


def test_all_context_features_are_label_free_and_do_not_use_t_close_or_amount():
    (
        calendar,
        _,
        prior,
        panel,
        reports,
        static_sw,
        index_history,
        daily_basic,
        tech,
    ) = _fixture_data()
    baseline = add_industry_context_features(
        panel,
        calendar,
        reports=reports,
        stock_sw_industry=static_sw,
        sw_index_history=index_history,
        daily_basic=daily_basic,
        tech_constituents=tech,
    )

    # Changing the decision-session index close/amount and daily_basic values
    # cannot change features whose clock stops at the exact previous session.
    changed_index = index_history.copy()
    changed_index.loc[changed_index["trade_date"].eq(calendar[-1]), ["close", "amount"]] = [
        99_999.0,
        88_888.0,
    ]
    changed_basic = daily_basic.copy()
    current = pd.to_datetime(changed_basic["trade_date"]).eq(calendar[-1])
    changed_basic.loc[
        current, ["turnover_rate", "turnover_rate_f", "volume_ratio", "circ_mv"]
    ] = -999.0
    # A selected/buy label is deliberately present but is neither consumed nor
    # allowed to affect any generated feature.
    labelled_panel = panel.assign(is_disclosed_buy=[True, False])
    changed = add_industry_context_features(
        labelled_panel,
        calendar,
        reports=reports,
        stock_sw_industry=static_sw,
        sw_index_history=changed_index,
        daily_basic=changed_basic,
        tech_constituents=tech,
    )

    generated = [
        column
        for column in baseline.columns
        if column not in panel.columns and column != "is_disclosed_buy"
    ]
    assert_frame_equal(baseline[generated], changed[generated])
    by_symbol = baseline.set_index("symbol")
    assert by_symbol.at["000001.SZ", "daily_basic_required_date"] == prior
    assert by_symbol.at["000001.SZ", "db_turnover_rate_t1"] == 1.1
    assert by_symbol.at["000001.SZ", "tech_index_count_static_oracle"] == 2
    assert by_symbol.at["600000.SH", "tech_index_count_static_oracle"] == 0
    assert by_symbol["sw_index_feature_available"].all()
    assert by_symbol["sw_static_oracle_available"].all()
    assert "oracle experiment only" in by_symbol.iloc[0]["sw_static_asof_rule"]

