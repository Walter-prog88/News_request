from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

import index_timing.data.providers as providers
import index_timing.data_service as data_service
from index_timing.backtest import cash_rate_to_daily_returns, run_cost_scenarios
from index_timing.cli import app
from index_timing.data import normalize_ohlcv
from index_timing.data.providers import FREDDataProvider


def _prices() -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-05", periods=5)
    prices = np.arange(100.0, 105.0)
    return normalize_ohlcv(
        pd.DataFrame(
            {
                "date": dates,
                "open": prices,
                "high": prices + 1.0,
                "low": prices - 1.0,
                "close": prices,
                "volume": 1_000.0,
            }
        )
    )


def test_keyless_fred_csv_supports_dgs3mo_and_marks_availability(monkeypatch) -> None:
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    seen: list[str] = []

    def fake_read_csv(url: str) -> pd.DataFrame:
        seen.append(url)
        return pd.DataFrame(
            {
                "observation_date": ["2024-01-02", "2024-01-03"],
                "DGS3MO": [5.40, "."],
            }
        )

    monkeypatch.setattr(providers.pd, "read_csv", fake_read_csv)
    frame = FREDDataProvider().fetch("DGS3MO", start="2024-01-01", end="2024-01-31")

    assert "id=DGS3MO" in seen[0]
    assert "cosd=2024-01-01" in seen[0]
    assert frame["value"].tolist() == [5.4]
    assert frame.loc[0, "available_at"] == pd.Timestamp("2024-01-03")


def test_cash_snapshot_roundtrip_uses_available_at_without_an_extra_lag(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(data_service, "RAW_DIR", tmp_path)

    class FakeFred:
        def fetch(self, symbol: str, start=None, end=None) -> pd.DataFrame:
            assert symbol == "DGS3MO"
            return pd.DataFrame(
                {
                    "observation_date": pd.to_datetime(["2024-01-05", "2024-01-08"]),
                    "value": [5.0, 5.1],
                    "available_at": pd.to_datetime(["2024-01-06", "2024-01-09"]),
                }
            )

    monkeypatch.setattr(data_service, "get_provider", lambda name: FakeFred())
    snapshot = data_service.sync_cash_series(
        "DGS3MO", start="2024-01-01", end="2024-01-31"
    )
    cash = data_service.load_cash_rate("DGS3MO")

    assert snapshot.dataset_id.startswith("cash:DGS3MO:")
    assert cash.status == "snapshot_available_at"
    assert cash.lag == 0
    assert cash.values is not None
    assert cash.values.index[0] == pd.Timestamp("2024-01-06")
    assert cash.metadata["cash_rate_snapshot_sha256"] == snapshot.sha256
    assert cash.metadata["cash_rate_snapshot_provider"] == "fred"
    assert cash.metadata["cash_rate_snapshot_min_date"] == "2024-01-05"
    assert cash.metadata["cash_rate_snapshot_max_date"] == "2024-01-08"
    assert isinstance(cash.metadata["cash_rate_snapshot_warnings"], list)
    assert cash.assumptions == ()


def test_non_trading_day_availability_is_consumed_on_next_session() -> None:
    prices = _prices()
    # Friday's observation first becomes available on Saturday.  The Monday
    # session may use it, while Friday may not.
    annual_yield = pd.Series([10.0], index=pd.DatetimeIndex(["2024-01-06"]))
    daily_cash = cash_rate_to_daily_returns(
        annual_yield,
        prices.index,
        lag=0,
        unit="percent",
    )
    expected = (1.10 ** (1.0 / 365.0)) - 1.0

    assert daily_cash.loc["2024-01-05"] == 0.0
    assert daily_cash.loc["2024-01-08"] == pytest.approx(expected)
    scenarios = run_cost_scenarios(
        prices,
        pd.Series(0.0, index=prices.index),
        one_way_cost_bps=2.0,
        cash_rate=annual_yield,
        cash_rate_lag=0,
        cash_rate_unit="percent",
    )
    assert all(
        frame.loc[pd.Timestamp("2024-01-08"), "cash_return"] == pytest.approx(expected)
        for frame in scenarios.values()
    )


def test_missing_cash_snapshot_has_explicit_zero_assumption(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(data_service, "RAW_DIR", tmp_path)
    cash = data_service.load_cash_rate("DR007")

    assert cash.values is None
    assert cash.assumptions == ("cash_rate_unavailable_zero",)
    assert cash.metadata["cash_rate_status"] == "cash_rate_unavailable_zero"
    assert cash.metadata["cash_rate_assumption"] == "cash_rate_unavailable_zero"


def test_dr007_locked_csv_interface(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(data_service, "RAW_DIR", tmp_path / "raw")
    csv_path = tmp_path / "dr007.csv"
    csv_path.write_text("date,DR007\n2024-01-02,1.80\n2024-01-03,1.85\n", encoding="utf-8")

    snapshot = data_service.sync_cash_series(
        "DR007",
        start=date(2024, 1, 1),
        end=date(2024, 1, 31),
        csv_path=csv_path,
    )
    cash = data_service.load_cash_rate("DR007")

    assert snapshot.provider == "csv"
    assert cash.values is not None
    assert cash.values.loc["2024-01-03"] == 1.80


def test_cli_exposes_explicit_cash_sync() -> None:
    result = CliRunner().invoke(app, ["data", "sync-cash", "--help"])
    assert result.exit_code == 0
    assert "DGS3MO" in result.stdout
    assert "DR007" in result.stdout
