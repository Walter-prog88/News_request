from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

import pandas as pd

import index_timing.data_service as data_service
from index_timing.data import (
    AKShareDataProvider,
    CSVDataProvider,
    DataProvider,
    TushareDataProvider,
    add_release_dates,
    assert_point_in_time,
    file_sha256,
    normalize_ohlcv,
    point_in_time_join,
    read_manifest,
    validate_ohlcv,
    write_snapshot,
)


def test_normalize_ohlcv_handles_chinese_columns_duplicates_and_sorting() -> None:
    raw = pd.DataFrame(
        {
            "日期": ["2024-01-03", "2024-01-02", "2024-01-03"],
            "开盘": [12, 10, 13],
            "最高": [13, 11, 14],
            "最低": [11, 9, 12],
            "收盘": [12.5, 10.5, 13.5],
            "成交量": [100, 90, 110],
        }
    )
    normalized = normalize_ohlcv(raw)
    validate_ohlcv(normalized)
    assert list(normalized.columns) == ["open", "high", "low", "close", "adj_close", "volume"]
    assert list(normalized.index) == [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")]
    assert normalized.loc["2024-01-03", "close"] == 13.5
    assert normalized.loc["2024-01-03", "adj_close"] == 13.5


def test_point_in_time_join_never_uses_future_release() -> None:
    observations = add_release_dates(
        pd.DataFrame(
            {
                "observation_date": pd.to_datetime(["2024-01-01", "2024-02-01"]),
                "value": [10.0, 20.0],
            }
        ),
        lag_days=5,
    )
    decisions = pd.date_range("2024-01-03", "2024-02-08", freq="D")
    joined = point_in_time_join(decisions, observations, value_columns=["value"])
    assert pd.isna(joined.loc["2024-01-05", "value"])
    assert joined.loc["2024-01-06", "value"] == 10.0
    assert joined.loc["2024-02-05", "value"] == 10.0
    assert joined.loc["2024-02-06", "value"] == 20.0
    assert_point_in_time(joined)


def test_csv_provider_implements_protocol(tmp_path) -> None:
    path = tmp_path / "prices.csv"
    path.write_text(
        "trade_date,Open,High,Low,Close,Volume\n"
        "2024-01-02,10,11,9,10.5,100\n"
        "2024-01-03,11,12,10,11.5,110\n",
        encoding="utf-8",
    )
    provider = CSVDataProvider(root=tmp_path)
    assert isinstance(provider, DataProvider)
    result = provider.fetch("prices.csv", start="2024-01-03")
    assert len(result) == 1
    assert result.iloc[0]["close"] == 11.5


def test_akshare_falls_back_to_same_index_and_records_cleanup(monkeypatch) -> None:
    def primary(**_kwargs):
        raise ConnectionError("remote disconnected")

    def official(**_kwargs):
        return pd.DataFrame(
            {
                "日期": ["2024-01-02", "2024-01-03"],
                "开盘": [10.0, None],
                "最高": [11.0, None],
                "最低": [9.0, None],
                "收盘": [10.5, 10.5],
                "成交量": [100.0, None],
            }
        )

    fake = SimpleNamespace(
        index_zh_a_hist=primary,
        stock_zh_index_hist_csindex=official,
    )
    monkeypatch.setitem(__import__("sys").modules, "akshare", fake)
    provider = AKShareDataProvider()
    result = provider.fetch("000300")

    assert list(result.index) == [pd.Timestamp("2024-01-02")]
    assert any("stock_zh_index_hist_csindex" in warning for warning in provider.last_warnings)
    assert any("2024-01-03" in warning for warning in provider.last_warnings)


def test_akshare_extends_truncated_primary_with_same_exchange_index(monkeypatch) -> None:
    def primary(**_kwargs):
        return pd.DataFrame(
            {
                "日期": ["2024-01-02", "2024-01-03"],
                "开盘": [10.0, 11.0],
                "最高": [11.0, 12.0],
                "最低": [9.0, 10.0],
                "收盘": [10.5, 11.5],
                "成交量": [100.0, 110.0],
            }
        )

    def long_history(symbol: str):
        assert symbol == "sz399006"
        return pd.DataFrame(
            {
                "date": pd.to_datetime(["2020-01-02", "2024-01-02"]),
                "open": [5.0, 9.9],
                "high": [5.5, 10.9],
                "low": [4.5, 8.9],
                "close": [5.2, 10.4],
                "volume": [50.0, 99.0],
            }
        )

    fake = SimpleNamespace(
        index_zh_a_hist=primary,
        stock_zh_index_daily=long_history,
    )
    monkeypatch.setitem(__import__("sys").modules, "akshare", fake)
    provider = AKShareDataProvider()
    result = provider.fetch("399006", start="2020-01-01", end="2024-01-03")

    assert list(result.index) == [
        pd.Timestamp("2020-01-02"),
        pd.Timestamp("2024-01-02"),
        pd.Timestamp("2024-01-03"),
    ]
    assert result.loc["2024-01-02", "close"] == 10.5
    assert any("补全历史" in warning for warning in provider.last_warnings)


def test_optional_tushare_provider_uses_index_endpoint_without_logging_token(
    monkeypatch,
) -> None:
    seen: dict[str, object] = {}

    class FakePro:
        def index_daily(self, **kwargs):
            seen.update(kwargs)
            return pd.DataFrame(
                {
                    "trade_date": ["20240103", "20240102"],
                    "open": [11.0, 10.0],
                    "high": [12.0, 11.0],
                    "low": [10.0, 9.0],
                    "close": [11.5, 10.5],
                    "vol": [110.0, 100.0],
                }
            )

    fake = SimpleNamespace(pro_api=lambda token: FakePro())
    monkeypatch.setitem(__import__("sys").modules, "tushare", fake)
    result = TushareDataProvider(token="secret").fetch(
        "000300.SH", start="2024-01-02", end="2024-01-03"
    )

    assert seen["ts_code"] == "000300.SH"
    assert seen["start_date"] == "20240102"
    assert list(result.index) == [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")]


def test_snapshot_writes_hash_and_verifiable_manifest(tmp_path) -> None:
    frame = normalize_ohlcv(
        pd.DataFrame(
            {
                "date": ["2024-01-02", "2024-01-03"],
                "open": [10, 11],
                "high": [11, 12],
                "low": [9, 10],
                "close": [10.5, 11.5],
                "volume": [100, 110],
            }
        )
    )
    snapshot = write_snapshot(
        frame,
        raw_dir=tmp_path,
        provider="csv",
        symbol="test/index",
        retrieved_at=datetime(2024, 1, 4, 12, tzinfo=UTC),
        is_proxy=True,
        warnings=["test proxy"],
    )
    assert snapshot.path.exists()
    assert snapshot.sha256 == file_sha256(snapshot.path)
    manifest = read_manifest(tmp_path, verify_hashes=True)
    assert len(manifest) == 1
    assert manifest[0].row_count == 2
    assert manifest[0].min_date.isoformat() == "2024-01-02"
    assert manifest[0].is_proxy is True


def test_latest_asset_snapshot_prefers_asof_coverage_over_later_sync_clock(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(data_service, "RAW_DIR", tmp_path)

    def frame(end: str) -> pd.DataFrame:
        dates = pd.DatetimeIndex(["2020-01-02", end])
        return normalize_ohlcv(
            pd.DataFrame(
                {
                    "date": dates,
                    "open": [10.0, 11.0],
                    "high": [11.0, 12.0],
                    "low": [9.0, 10.0],
                    "close": [10.5, 11.5],
                    "volume": [100.0, 110.0],
                }
            )
        )

    better_coverage = write_snapshot(
        frame("2024-01-03"),
        raw_dir=tmp_path,
        provider="test",
        symbol="TEST",
        dataset_id="asset:signal:older-clock",
        retrieved_at=datetime(2026, 7, 13, 10, tzinfo=UTC),
        as_of=pd.Timestamp("2026-07-13").date(),
    )
    write_snapshot(
        frame("2023-01-03"),
        raw_dir=tmp_path,
        provider="test",
        symbol="TEST",
        dataset_id="asset:signal:newer-clock",
        retrieved_at=datetime(2026, 7, 13, 11, tzinfo=UTC),
        as_of=pd.Timestamp("2026-07-13").date(),
    )

    assert data_service.latest_snapshot("asset").dataset_id == better_coverage.dataset_id


def test_latest_asset_snapshot_prefers_full_history_then_newer_clean_snapshot(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(data_service, "RAW_DIR", tmp_path)

    def frame(dates: list[str]) -> pd.DataFrame:
        values = list(range(10, 10 + len(dates)))
        return normalize_ohlcv(
            pd.DataFrame(
                {
                    "date": pd.DatetimeIndex(dates),
                    "open": values,
                    "high": [value + 1 for value in values],
                    "low": [value - 1 for value in values],
                    "close": [value + 0.5 for value in values],
                    "volume": [100.0] * len(dates),
                }
            )
        )

    write_snapshot(
        frame(["2020-01-02", "2024-01-03"]),
        raw_dir=tmp_path,
        provider="test",
        symbol="TEST",
        dataset_id="asset:signal:old-extra-row",
        retrieved_at=datetime(2026, 7, 13, 10, tzinfo=UTC),
        as_of=pd.Timestamp("2026-07-13").date(),
    )
    clean = write_snapshot(
        frame(["2020-01-02", "2024-01-03"]),
        raw_dir=tmp_path,
        provider="test",
        symbol="TEST",
        dataset_id="asset:signal:new-clean",
        retrieved_at=datetime(2026, 7, 13, 11, tzinfo=UTC),
        as_of=pd.Timestamp("2026-07-13").date(),
    )
    write_snapshot(
        frame(["2021-01-04", "2024-01-03"]),
        raw_dir=tmp_path,
        provider="test",
        symbol="TEST",
        dataset_id="asset:signal:newer-truncated",
        retrieved_at=datetime(2026, 7, 13, 12, tzinfo=UTC),
        as_of=pd.Timestamp("2026-07-13").date(),
    )

    assert data_service.latest_snapshot("asset").dataset_id == clean.dataset_id
