from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from afactor.trader_choice_oracle import add_structured_industry_scores


def _manual_panel() -> pd.DataFrame:
    date = pd.Timestamp("2026-08-21")
    symbols = [
        "000001.SZ",
        "000002.SZ",
        "000003.SZ",
        "600001.SH",
        "600002.SH",
        "600003.SH",
    ]
    # Industry A is uniformly stronger than B.  Within each group all leader
    # inputs share the same high/mid/low ordering, making the expected ranks
    # exactly hand-computable.
    return pd.DataFrame(
        {
            "decision_date": date,
            "symbol": symbols,
            "eligible_for_ranking": True,
            "report_industry_pit": ["A", "A", "A", "B", "B", "B"],
            "sw_fine_static_oracle": ["A", "A", "A", "B", "B", "B"],
            "sw_l1_static_oracle": ["A", "A", "A", "B", "B", "B"],
            "mom5": [0.30, 0.20, 0.10, -0.10, -0.20, -0.30],
            "mom10": [0.40, 0.30, 0.20, -0.10, -0.20, -0.30],
            "mom20": [0.50, 0.40, 0.30, -0.10, -0.20, -0.30],
            "mom60": [0.60, 0.50, 0.40, -0.10, -0.20, -0.30],
            "ma20_dist": [0.30, 0.20, 0.10, -0.10, -0.20, -0.30],
            "amt_ratio20": [2.0, 1.8, 1.6, 0.9, 0.8, 0.7],
            "liq20": [300.0, 200.0, 100.0, 60.0, 50.0, 40.0],
            "db_turnover_rate_t1": [3.0, 2.0, 1.0, 0.6, 0.5, 0.4],
            "db_turnover_rate_f_t1": [3.3, 2.2, 1.1, 0.66, 0.55, 0.44],
            "db_volume_ratio_t1": [1.8, 1.5, 1.2, 0.9, 0.8, 0.7],
            "db_circ_mv_t1": [300.0, 200.0, 100.0, 60.0, 50.0, 40.0],
            "sw_l1_ret5_t1": [0.20, 0.20, 0.20, -0.10, -0.10, -0.10],
            "sw_l1_ret20_t1": [0.30, 0.30, 0.30, -0.20, -0.20, -0.20],
            "sw_l1_amount_ratio20_t1": [1.8, 1.8, 1.8, 0.8, 0.8, 0.8],
            "sw_l1_amount_accel5_20_t1": [1.6, 1.6, 1.6, 0.7, 0.7, 0.7],
            "tech_member__AI_static_oracle": [True, True, True, False, False, False],
            "tech_member__SOFT_static_oracle": [False, False, False, True, True, True],
        }
    )


def test_hand_calculated_direction_leader_and_tech_scores():
    result = add_structured_industry_scores(_manual_panel()).set_index("symbol")

    # Two equal-weight groups: weaker B gets percentile .5 and stronger A 1.0
    # for every direction component and taxonomy.
    assert result.at["000001.SZ", "f7_structured_direction_score"] == 1.0
    assert result.at["600001.SH", "f7_structured_direction_score"] == 0.5
    assert result.at["000001.SZ", "f7_structured_direction_taxonomy_count"] == 3

    # All six leader components have the same ordering within each 3-stock
    # group, so the exact within-group percentiles are 1, 2/3 and 1/3.
    assert result.at["000001.SZ", "f8_structured_leader_score"] == 1.0
    assert np.isclose(
        result.at["000002.SZ", "f8_structured_leader_score"], 2.0 / 3.0
    )
    assert np.isclose(
        result.at["000003.SZ", "f8_structured_leader_score"], 1.0 / 3.0
    )

    # Tech score = 70% group direction rank + 30% within-index leader rank.
    assert result.at["000001.SZ", "f9_tech_index_score"] == 1.0
    assert np.isclose(result.at["000003.SZ", "f9_tech_index_score"], 0.8)
    assert np.isclose(result.at["600001.SH", "f9_tech_index_score"], 0.65)
    assert not result["structured_factor_label_event_pnl_columns_used"].any()


def test_labels_events_pnl_and_same_day_raw_fields_cannot_change_scores():
    panel = _manual_panel()
    baseline = add_structured_industry_scores(panel)
    contaminated = panel.assign(
        is_disclosed_buy=[True, False, False, True, False, False],
        event_attention=[9_999.0, -9_999.0, 1.0, 2.0, 3.0, 4.0],
        forward_return=[8.0, 7.0, 6.0, 5.0, 4.0, 3.0],
        pnl=[100.0, -100.0, 90.0, -90.0, 80.0, -80.0],
        close=[1e9, -1e9, 3.0, 4.0, 5.0, 6.0],
        amount=[-1e9, 1e9, 3.0, 4.0, 5.0, 6.0],
    )
    changed = add_structured_industry_scores(contaminated)
    generated = [column for column in baseline.columns if column not in panel.columns]
    assert_frame_equal(baseline[generated], changed[generated])


def test_no_taxonomy_or_tech_membership_stays_missing():
    panel = _manual_panel().drop(
        columns=[
            "report_industry_pit",
            "sw_fine_static_oracle",
            "sw_l1_static_oracle",
            "tech_member__AI_static_oracle",
            "tech_member__SOFT_static_oracle",
        ]
    )
    result = add_structured_industry_scores(panel)
    assert result["f7_structured_direction_score"].isna().all()
    assert result["f8_structured_leader_score"].isna().all()
    assert result["f9_tech_index_score"].isna().all()

