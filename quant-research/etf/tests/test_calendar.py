from __future__ import annotations

import pandas as pd

from index_timing import data_service
from index_timing.data import align_to_trading_calendar, load_snapshot
from index_timing.types import AssetSpec, Market


def test_calendar_drops_holiday_placeholder_without_filling_missing_session() -> None:
    frame = pd.DataFrame(
        {"close": [100.0, 100.0, 101.0]},
        index=pd.DatetimeIndex(["2024-07-03", "2024-07-04", "2024-07-08"], name="date"),
    )
    aligned, warnings = align_to_trading_calendar(frame, "XNYS")

    assert pd.Timestamp("2024-07-04") not in aligned.index
    assert pd.Timestamp("2024-07-05") not in aligned.index
    assert any("非交易日" in warning for warning in warnings)
    assert any("未进行价格填充" in warning for warning in warnings)


def test_sync_asset_aligns_calendar_before_snapshot(monkeypatch, tmp_path) -> None:
    class FakeProvider:
        last_warnings = ["provider warning"]

        def fetch(self, symbol, start, end):
            del symbol, start, end
            return pd.DataFrame(
                {
                    "open": [100.0, 100.0, 101.0],
                    "high": [101.0, 101.0, 102.0],
                    "low": [99.0, 99.0, 100.0],
                    "close": [100.0, 100.0, 101.0],
                    "volume": [10.0, 0.0, 11.0],
                },
                index=pd.DatetimeIndex(
                    ["2024-07-03", "2024-07-04", "2024-07-08"],
                    name="date",
                ),
            )

    asset = AssetSpec(
        asset_id="test_spy",
        name="Test SPY",
        studies=["us_extension"],
        signal_symbol="SPY",
        provider="yahoo",
        market=Market.UNITED_STATES,
        currency="USD",
        calendar="XNYS",
        one_way_cost_bps=2.0,
    )
    monkeypatch.setattr(data_service, "RAW_DIR", tmp_path)
    monkeypatch.setattr(data_service, "get_provider", lambda provider: FakeProvider())

    snapshot = data_service.sync_asset(
        asset,
        start="2024-07-03",
        end="2024-07-08",
    )[0]
    persisted = load_snapshot(snapshot)

    assert list(persisted.index) == [
        pd.Timestamp("2024-07-03"),
        pd.Timestamp("2024-07-08"),
    ]
    assert snapshot.row_count == 2
    assert any("非交易日" in warning for warning in snapshot.warnings)
    assert any("未进行价格填充" in warning for warning in snapshot.warnings)
    assert "provider warning" in snapshot.warnings


def test_calendar_retains_history_before_library_coverage() -> None:
    index = pd.DatetimeIndex(["2000-01-04", "2007-01-04"])
    frame = pd.DataFrame({"close": [100.0, 110.0]}, index=index)

    aligned, warnings = align_to_trading_calendar(frame, "XSHG")

    assert pd.Timestamp("2000-01-04") in aligned.index
    assert any("覆盖区间外历史" in warning for warning in warnings)
