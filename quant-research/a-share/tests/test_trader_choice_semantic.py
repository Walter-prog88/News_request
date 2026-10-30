from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest
from pandas.testing import assert_series_equal


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from afactor.trader_choice_semantic import (  # noqa: E402
    StaticOracleSemanticConfig,
    ThemeDefinition,
    add_static_oracle_semantic_features,
    build_static_oracle_semantic_long,
    build_static_oracle_theme_membership,
)


def _membership_sources() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sw = pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "000002.SZ", "600001.SH", "600002.SH"],
            "name": ["样本甲", "样本乙", "样本丙", "样本丁"],
            "sw_industry": ["电子", "计算机", "机械设备", "医药生物"],
            "fine_industry": ["半导体", "软件服务", "机床制造", "其他制造"],
        }
    )
    reports = pd.DataFrame(
        {
            "stockCode": [1, 2, 600001, 600002],
            "stockName": ["样本甲", "样本乙", "样本丙", "样本丁"],
            "indvInduName": ["半导体", "软件服务", "机床制造", "其他制造"],
            "title": [
                "GPU芯片与高多层PCB覆铜板进入放量阶段",
                "大模型与算力业务加速",
                "人形机器人伺服系统扩产",
                "散热温控技术用于数据中心服务器",
            ],
        }
    )
    indices = pd.DataFrame(
        {
            "成分券代码": [1, 2],
            "成分券名称": ["样本甲", "样本乙"],
            "index_name": ["中证全指半导体", "中证人工智能"],
            "指数名称": ["半导体", "人工智能"],
        }
    )
    return sw, reports, indices


def test_static_mapping_is_multilabel_and_keeps_public_source_provenance() -> None:
    sw, reports, indices = _membership_sources()
    result = build_static_oracle_theme_membership(sw, reports, indices)

    first = result.loc[result["symbol"].eq("000001.SZ")]
    second = result.loc[result["symbol"].eq("000002.SZ")]
    assert set(first["theme_id"]) == {
        "advanced_materials",
        "ai_compute_network",
        "pcb",
        "semiconductor",
    }
    assert "ai_compute_network" in set(second["theme_id"])
    assert "ai_applications" not in set(second["theme_id"])
    semiconductor = first.loc[
        first["theme_id"].eq("semiconductor")
    ].iloc[0]
    assert semiconductor["oracle_static_source_count"] == 3
    assert semiconductor["oracle_static_uses_future_metadata"]
    assert "oracle/static" in semiconductor["oracle_static_asof_rule"]


def test_mapping_rejects_transaction_or_label_sources_and_stock_code_rules() -> None:
    sw, _, _ = _membership_sources()
    contaminated = sw.assign(event_id="E1")
    with pytest.raises(ValueError, match="forbidden to read"):
        build_static_oracle_theme_membership(contaminated)

    with pytest.raises(ValueError, match="forbidden stock code"):
        ThemeDefinition("bad_theme", "bad", ("000001",))

    company_rule = ThemeDefinition("bad_company", "bad", ("样本甲",))
    with pytest.raises(ValueError, match="company-name"):
        build_static_oracle_theme_membership(sw, themes=(company_rule,))


def test_context_rule_needs_cooccurrence_and_exclusion_blocks_generic_cold_plate() -> None:
    reports = pd.DataFrame(
        {
            "stockCode": [1, 2, 3],
            "title": [
                "散热温控技术用于数据中心服务器",
                "散热产品收入增长",
                "冷轧板和冷板价格继续上涨",
            ],
            "indvInduName": ["其他制造", "其他制造", "钢铁"],
        }
    )
    result = build_static_oracle_theme_membership(pd.DataFrame(), reports)
    liquid = result.loc[result["theme_id"].eq("liquid_cooling")]
    assert set(liquid["symbol"]) == {"000001.SZ"}
    assert "context" in liquid.iloc[0]["oracle_static_match_rules"]


def test_frozen_supply_chain_terms_and_ai_application_context_gate() -> None:
    reports = pd.DataFrame(
        {
            "stockCode": list(range(1, 10)),
            "indvInduName": ["其他制造"] * 9,
            "title": [
                "电子玻纤布与ABF载板、PCBA电子装联产能扩张",
                "MDI聚氨酯与TPU、光学薄膜OCA、电子胶导热材料和人造金刚石靶材放量",
                "数据中心算力电源和UPS订单增长",
                "通用工业电源收入增长",
                "数据中心温控系统中的泵阀与换热器需求提升",
                "通用泵阀与换热器业务稳健",
                "生成式AI软件平台订单和收入增长",
                "生成式AI服务器收入增长",
                "计算机视觉用于工业质检，订单加速落地",
            ],
        }
    )
    result = build_static_oracle_theme_membership(pd.DataFrame(), reports)
    themes_by_symbol = result.groupby("symbol")["theme_id"].agg(set).to_dict()

    assert "pcb" in themes_by_symbol["000001.SZ"]
    assert "advanced_materials" in themes_by_symbol["000002.SZ"]
    assert "ai_compute_network" in themes_by_symbol["000003.SZ"]
    assert "ai_compute_network" not in themes_by_symbol.get("000004.SZ", set())
    assert "liquid_cooling" in themes_by_symbol["000005.SZ"]
    assert "liquid_cooling" not in themes_by_symbol.get("000006.SZ", set())
    assert "ai_applications" in themes_by_symbol["000007.SZ"]
    assert "ai_compute_network" in themes_by_symbol["000008.SZ"]
    assert "ai_applications" not in themes_by_symbol["000008.SZ"]
    assert "ai_applications" in themes_by_symbol["000009.SZ"]


def _panel() -> pd.DataFrame:
    rows = []
    values = {
        "000001.SZ": (0.08, 0.20, 1.8, 0.15, 0.18, 0.35, 200.0),
        "000002.SZ": (0.04, 0.10, 1.4, 0.08, 0.10, 0.22, 120.0),
        "600001.SH": (-0.02, -0.05, 0.8, -0.03, -0.04, 0.12, 80.0),
        "600002.SH": (-0.04, -0.08, 0.7, -0.06, -0.07, 0.10, 70.0),
    }
    for symbol, value in values.items():
        mom5, mom20, amount, ma20, mom10, mom60, liq = value
        rows.append(
            {
                "decision_date": "2026-08-20",
                "feature_cutoff_date": "2026-08-19",
                "symbol": symbol,
                "eligible_for_ranking": True,
                "mom5": mom5,
                "mom20": mom20,
                "amt_ratio20": amount,
                "ma20_dist": ma20,
                "mom10": mom10,
                "mom60": mom60,
                "liq20": liq,
                "db_turnover_rate_t1": 3.0 * amount,
                "db_volume_ratio_t1": amount,
                "db_circ_mv_t1": 1000.0 * liq,
            }
        )
    return pd.DataFrame(rows)


def _manual_membership() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": [
                "000001.SZ",
                "000001.SZ",
                "000002.SZ",
                "600001.SH",
                "600002.SH",
            ],
            "theme_id": [
                "strong_theme",
                "weak_theme",
                "strong_theme",
                "weak_theme",
                "weak_theme",
            ],
            "theme_name": ["强主题", "弱主题", "强主题", "弱主题", "弱主题"],
        }
    )


def test_theme_strength_leader_and_interaction_use_tminus1_panel_fields() -> None:
    config = StaticOracleSemanticConfig(theme_min_members=2)
    panel = _panel()
    long = build_static_oracle_semantic_long(panel, _manual_membership(), config)
    by_key = long.set_index(["symbol", "theme_id"])

    assert by_key.loc[
        ("000001.SZ", "strong_theme"), "oracle_static_theme_strength_pct"
    ] == 1.0
    assert by_key.loc[
        ("600001.SH", "weak_theme"), "oracle_static_theme_strength_pct"
    ] == 0.5
    assert (
        by_key.loc[
            ("000001.SZ", "strong_theme"), "oracle_static_theme_leader_score"
        ]
        > by_key.loc[
            ("000002.SZ", "strong_theme"), "oracle_static_theme_leader_score"
        ]
    )
    assert (
        by_key.loc[
            ("000001.SZ", "strong_theme"), "oracle_static_theme_x_leader"
        ]
        > by_key.loc[
            ("600001.SH", "weak_theme"), "oracle_static_theme_x_leader"
        ]
    )

    attached = add_static_oracle_semantic_features(
        panel.assign(is_selected=[True, False, False, False], close_t=999.0),
        _manual_membership(),
        config,
    ).set_index("symbol")
    changed_labels = add_static_oracle_semantic_features(
        panel.assign(is_selected=[False, True, True, True], close_t=-999.0),
        _manual_membership(),
        config,
    ).set_index("symbol")
    assert_series_equal(
        attached["f7_oracle_static_semantic_score"],
        changed_labels["f7_oracle_static_semantic_score"],
        check_names=False,
    )
    assert attached.loc["000001.SZ", "oracle_static_best_theme_id"] == "strong_theme"
    assert attached.loc["000001.SZ", "f7_oracle_static_semantic_candidate_pct"] == 1.0
    stock_long = long.loc[long["symbol"].eq("000001.SZ")]
    assert attached.loc["000001.SZ", "f7_oracle_static_semantic_score"] == (
        stock_long["oracle_static_theme_x_leader"].max()
    )
    assert attached.loc[
        "000001.SZ", "oracle_static_stock_theme_aggregation"
    ].startswith("max")
    assert "oracle_static_leader_pct__db_turnover_rate_t1" in long.columns


def test_feature_clock_rejects_same_day_or_future_cutoffs() -> None:
    panel = _panel()
    panel.loc[0, "feature_cutoff_date"] = panel.loc[0, "decision_date"]
    config = StaticOracleSemanticConfig(theme_min_members=2)
    with pytest.raises(ValueError, match="strictly earlier"):
        add_static_oracle_semantic_features(panel, _manual_membership(), config)
