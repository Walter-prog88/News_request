from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_series_equal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from afactor.trader_choice_data import build_base_panel
from afactor.trader_choice_portfolio import add_portfolio_replacement


def _wide_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    dates = pd.bdate_range("2025-01-02", periods=75)
    columns = ["000001.SZ", "600000.SH"]
    base = np.arange(len(dates), dtype=float)[:, None]
    closes = pd.DataFrame(10.0 + base * np.array([[0.03, 0.02]]), index=dates, columns=columns)
    opens = closes.shift(1).fillna(closes.iloc[0]) * 1.001
    amounts = pd.DataFrame(50_000_000.0, index=dates, columns=columns)
    return opens, closes, amounts, dates[-1]


def test_base_features_ignore_t_close_and_t_amount_but_use_t_open() -> None:
    opens, closes, amounts, decision = _wide_inputs()
    first = build_base_panel([decision], opens, closes, amounts).set_index("symbol")

    changed_close = closes.copy()
    changed_amount = amounts.copy()
    changed_close.loc[decision] *= 10.0
    changed_amount.loc[decision] *= 100.0
    second = build_base_panel([decision], opens, changed_close, changed_amount).set_index("symbol")
    feature_columns = [
        "mom5",
        "mom10",
        "mom20",
        "mom60",
        "prev_ret",
        "ma20_dist",
        "drawdown20",
        "vol20",
        "amt_ratio20",
        "liq20",
        "pv_score",
    ]
    for column in feature_columns:
        assert_series_equal(first[column], second[column], check_names=False)

    changed_open = opens.copy()
    changed_open.loc[decision, "000001.SZ"] *= 1.02
    third = build_base_panel([decision], changed_open, closes, amounts).set_index("symbol")
    assert third.loc["000001.SZ", "gap"] != first.loc["000001.SZ", "gap"]


def test_portfolio_layer_ignores_etfs_and_never_carries_snapshot(tmp_path: Path) -> None:
    dates = pd.bdate_range("2025-01-02", periods=70)
    decision = dates[-1]
    next_session = decision + pd.offsets.BDay(1)
    symbols = ["000001.SZ", "600000.SH"]
    closes = pd.DataFrame(
        {
            "000001.SZ": np.linspace(10, 13, len(dates)),
            "600000.SH": np.linspace(9, 10, len(dates)),
        },
        index=dates,
    )
    family_columns = [
        "price_volume",
        "dynamic_direction",
        "within_direction_leader",
        "event_attention",
        "familiarity",
    ]
    panel_rows = []
    for date in (decision, next_session):
        for index, symbol in enumerate(symbols):
            row = {"decision_date": date, "symbol": symbol}
            for family in family_columns:
                row[f"family_score__{family}"] = 0.8 - 0.3 * index
            panel_rows.append(row)
    panel = pd.DataFrame(panel_rows)
    positions = pd.DataFrame(
        {
            "trading_date": [dates[-2], dates[-2]],
            "security_code": ["600000", "NAME:黄金ETF华安"],
            "total_quantity_best": [1000, 2000],
            "confidence": ["L4", "L4"],
        }
    )
    path = tmp_path / "positions.csv"
    positions.to_csv(path, index=False)
    result = add_portfolio_replacement(
        panel,
        closes,
        path,
        {decision: dates[-2], next_session: decision},
    )
    first_day = result[result["decision_date"].eq(decision)]
    second_day = result[result["decision_date"].eq(next_session)]
    assert first_day["portfolio_context_available"].all()
    assert first_day["portfolio_holding_count"].eq(1).all()
    assert set(first_day["predicted_sell_symbol"].dropna()) == {"600000.SH"}
    assert not second_day["portfolio_context_available"].any()
    assert second_day["family_score__portfolio_replacement"].isna().all()
