"""Common market-data contracts and canonical OHLCV normalization."""

from __future__ import annotations

from datetime import date
from typing import Any, Protocol, runtime_checkable

import numpy as np
import pandas as pd

OHLCV_COLUMNS = ("open", "high", "low", "close", "adj_close", "volume")


@runtime_checkable
class DataProvider(Protocol):
    """Minimal contract implemented by all market-data adapters.

    Providers return a sorted, unique, timezone-naive ``DatetimeIndex`` named
    ``date`` and the columns in :data:`OHLCV_COLUMNS`.  Macro providers expose
    an additional ``fetch_series`` method, but still implement ``fetch`` so a
    provider registry can treat them uniformly.
    """

    name: str

    def fetch(
        self,
        symbol: str,
        start: date | str | None = None,
        end: date | str | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Fetch a dataset for ``symbol`` over the requested inclusive range."""


def _flatten_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if isinstance(result.columns, pd.MultiIndex):
        # yfinance returns either (field, ticker) or (ticker, field).  Retain
        # the level containing the standard price field names.
        field_names = {"open", "high", "low", "close", "adj close", "volume"}
        scores = [
            sum(str(value).strip().lower() in field_names for value in level)
            for level in result.columns.levels
        ]
        field_level = int(np.argmax(scores))
        result.columns = result.columns.get_level_values(field_level)
    return result


def normalize_ohlcv(
    frame: pd.DataFrame,
    *,
    start: date | str | None = None,
    end: date | str | None = None,
) -> pd.DataFrame:
    """Normalize common provider layouts to the project's OHLCV schema.

    ``adj_close`` falls back to ``close`` and absent volume is represented by
    ``NaN``.  Rows without a usable close price are discarded.  The function
    intentionally does not forward-fill prices, because that would invent
    tradable observations across suspensions and holidays.
    """

    if frame is None or frame.empty:
        return pd.DataFrame(
            {column: pd.Series(dtype="float64") for column in OHLCV_COLUMNS},
            index=pd.DatetimeIndex([], name="date"),
        )

    result = _flatten_columns(frame)
    aliases = {
        "date": "date",
        "datetime": "date",
        "timestamp": "date",
        "trade_date": "date",
        "日期": "date",
        "open": "open",
        "开盘": "open",
        "high": "high",
        "最高": "high",
        "low": "low",
        "最低": "low",
        "close": "close",
        "收盘": "close",
        "adj close": "adj_close",
        "adj_close": "adj_close",
        "adjusted_close": "adj_close",
        "复权收盘": "adj_close",
        "volume": "volume",
        "vol": "volume",
        "成交量": "volume",
    }
    result = result.rename(
        columns={column: aliases.get(str(column).strip().lower(), str(column)) for column in result}
    )

    if "date" in result.columns:
        dates = pd.to_datetime(result.pop("date"), errors="coerce")
    else:
        dates = pd.to_datetime(result.index, errors="coerce")
    if isinstance(dates, pd.Series):
        date_index = pd.DatetimeIndex(dates.array)
    else:
        date_index = pd.DatetimeIndex(dates)
    if date_index.tz is not None:
        date_index = date_index.tz_convert(None)
    result.index = date_index.normalize()
    result.index.name = "date"
    result = result.loc[~result.index.isna()]

    if "close" not in result:
        raise ValueError("OHLCV data must contain a close column")
    for column in ("open", "high", "low"):
        if column not in result:
            result[column] = result["close"]
    if "adj_close" not in result:
        result["adj_close"] = result["close"]
    if "volume" not in result:
        result["volume"] = np.nan

    result = result.loc[:, list(OHLCV_COLUMNS)].apply(pd.to_numeric, errors="coerce")
    result = result.dropna(subset=["close"])
    result = result[~result.index.duplicated(keep="last")].sort_index()
    if start is not None:
        result = result.loc[result.index >= pd.Timestamp(start)]
    if end is not None:
        result = result.loc[result.index <= pd.Timestamp(end)]
    return result.astype("float64")


def validate_ohlcv(frame: pd.DataFrame) -> None:
    """Raise a useful error if ``frame`` violates the canonical schema."""

    missing = set(OHLCV_COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(f"OHLCV frame is missing columns: {sorted(missing)}")
    if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.name != "date":
        raise ValueError("OHLCV frame must use a DatetimeIndex named 'date'")
    if not frame.index.is_monotonic_increasing or frame.index.has_duplicates:
        raise ValueError("OHLCV dates must be sorted and unique")
    if frame["close"].isna().any():
        raise ValueError("close prices cannot be missing")
