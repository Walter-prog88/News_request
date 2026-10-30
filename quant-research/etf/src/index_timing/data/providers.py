"""Public, local and optional proprietary data-provider adapters."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import numpy as np
import pandas as pd

from .base import normalize_ohlcv
from .point_in_time import add_release_dates


def _compact_date(value: date | str | None) -> str | None:
    if value is None:
        return None
    return pd.Timestamp(value).strftime("%Y%m%d")


@dataclass(slots=True)
class YahooDataProvider:
    """Yahoo Finance adapter backed by yfinance."""

    auto_adjust: bool = True
    name: str = "yahoo"

    def fetch(
        self,
        symbol: str,
        start: date | str | None = None,
        end: date | str | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        try:
            import yfinance as yf
        except ImportError as exc:  # pragma: no cover - dependency error path
            raise RuntimeError("Yahoo provider requires the 'yfinance' package") from exc

        # yfinance treats end as exclusive, while the project contract is inclusive.
        exclusive_end = None if end is None else pd.Timestamp(end) + pd.Timedelta(days=1)
        frame = yf.download(
            symbol,
            start=start,
            end=exclusive_end,
            auto_adjust=self.auto_adjust,
            actions=False,
            progress=False,
            threads=False,
            **kwargs,
        )
        return normalize_ohlcv(frame, start=start, end=end)


@dataclass(slots=True)
class AKShareDataProvider:
    """AKShare adapter for mainland and Hong Kong index history.

    ``endpoint`` can be supplied to force a particular AKShare function.  The
    default chooses an index endpoint and is intentionally conservative rather
    than silently substituting a different instrument.
    """

    endpoint: str = "index_zh_a_hist"
    name: str = "akshare"
    last_warnings: list[str] = field(default_factory=list, init=False)

    def fetch(
        self,
        symbol: str,
        start: date | str | None = None,
        end: date | str | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        try:
            import akshare as ak
        except ImportError as exc:  # pragma: no cover - dependency error path
            raise RuntimeError("AKShare provider requires the 'akshare' package") from exc

        self.last_warnings = []
        endpoint = str(kwargs.pop("endpoint", self.endpoint))
        if not hasattr(ak, endpoint):
            raise ValueError(f"Unknown AKShare endpoint: {endpoint}")
        function = getattr(ak, endpoint)
        if endpoint == "index_zh_a_hist":
            period = kwargs.pop("period", "daily")
            primary_frame: pd.DataFrame | None = None
            primary_error: Exception | None = None
            try:
                primary_frame = function(
                    symbol=symbol,
                    period=period,
                    start_date=_compact_date(start) or "19900101",
                    end_date=_compact_date(end) or pd.Timestamp.today().strftime("%Y%m%d"),
                    **kwargs,
                )
            except Exception as exc:
                primary_error = exc

            candidates: list[tuple[str, pd.DataFrame]] = []
            if primary_frame is not None:
                primary_normalized = normalize_ohlcv(
                    primary_frame, start=start, end=end
                )
                if not primary_normalized.empty:
                    candidates.append((endpoint, primary_normalized))

            upper_symbol = symbol.upper()
            if upper_symbol == "HSTECH" and hasattr(
                ak, "stock_hk_index_daily_sina"
            ):
                fallback_frame = ak.stock_hk_index_daily_sina(symbol=symbol)
                fallback_normalized = normalize_ohlcv(
                    fallback_frame, start=start, end=end
                )
                if not fallback_normalized.empty:
                    candidates.insert(
                        0, ("stock_hk_index_daily_sina", fallback_normalized)
                    )
            elif symbol.isdigit() and hasattr(ak, "stock_zh_index_daily"):
                # Eastmoney/CSI endpoints sometimes return only a recent
                # window even when an earlier start was requested.  The Sina
                # endpoint exposes long same-instrument histories for exchange
                # index codes (SH 000xxx / SZ 399xxx).  Use it only when it
                # materially extends the requested coverage, then prefer the
                # primary source on overlapping dates.
                exchange_symbol = (
                    f"sz{symbol}" if symbol.startswith("399") else f"sh{symbol}"
                )
                try:
                    fallback_frame = ak.stock_zh_index_daily(symbol=exchange_symbol)
                    fallback_normalized = normalize_ohlcv(
                        fallback_frame, start=start, end=end
                    )
                    primary_min = (
                        candidates[-1][1].index.min() if candidates else pd.Timestamp.max
                    )
                    extends_history = bool(
                        not fallback_normalized.empty
                        and fallback_normalized.index.min() < primary_min
                    )
                    if extends_history or not candidates:
                        candidates.insert(
                            0, ("stock_zh_index_daily", fallback_normalized)
                        )
                except Exception:
                    pass

            if hasattr(ak, "stock_zh_index_hist_csindex"):
                try:
                    fallback_frame = ak.stock_zh_index_hist_csindex(symbol=symbol)
                    fallback_normalized = normalize_ohlcv(
                        fallback_frame, start=start, end=end
                    )
                    current_min = min(
                        (candidate.index.min() for _, candidate in candidates),
                        default=pd.Timestamp.max,
                    )
                    current_max = max(
                        (candidate.index.max() for _, candidate in candidates),
                        default=pd.Timestamp.min,
                    )
                    extends_coverage = bool(
                        not fallback_normalized.empty
                        and (
                            fallback_normalized.index.min() < current_min
                            or fallback_normalized.index.max() > current_max
                        )
                    )
                    if extends_coverage or not candidates:
                        candidates.insert(
                            0,
                            ("stock_zh_index_hist_csindex", fallback_normalized),
                        )
                except Exception:
                    pass
            if not candidates:
                if primary_error is not None:
                    raise primary_error
                raise RuntimeError(f"AKShare returned no usable history for {symbol}")
            normalized = pd.concat([candidate for _, candidate in candidates])
            normalized = normalized[~normalized.index.duplicated(keep="last")].sort_index()
            fallback_names = [name for name, _ in candidates if name != endpoint]
            if fallback_names:
                action = "补全历史" if primary_frame is not None else "接口失败后回退"
                self.last_warnings.append(
                    f"默认{endpoint}{action}；使用同一指数的"
                    f"{'+'.join(fallback_names)}公开历史接口"
                )
        elif endpoint in {"stock_hk_index_daily_em", "stock_hk_index_daily_sina"}:
            frame = function(symbol=symbol, **kwargs)
            normalized = normalize_ohlcv(frame, start=start, end=end)
        else:
            frame = function(symbol=symbol, **kwargs)
            normalized = normalize_ohlcv(frame, start=start, end=end)
        incomplete = normalized[["open", "high", "low", "close"]].isna().any(axis=1)
        if incomplete.any():
            dates = ",".join(date.strftime("%Y-%m-%d") for date in normalized.index[incomplete])
            self.last_warnings.append(
                f"剔除{int(incomplete.sum())}条休市占位/不完整OHLC记录：{dates}"
            )
            normalized = normalized.loc[~incomplete]
        return normalized


@dataclass(slots=True)
class CSVDataProvider:
    """Read user-supplied price files without granting implicit substitutions."""

    root: Path | str = Path(".")
    date_column: str | None = None
    name: str = "csv"

    def fetch(
        self,
        symbol: str,
        start: date | str | None = None,
        end: date | str | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        path = Path(symbol).expanduser()
        if not path.is_absolute():
            path = Path(self.root).expanduser() / path
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path, **kwargs)
        date_column = self.date_column
        if date_column is not None and date_column in frame:
            frame = frame.rename(columns={date_column: "date"})
        return normalize_ohlcv(frame, start=start, end=end)


@dataclass(slots=True)
class FREDDataProvider:
    """FRED series adapter with explicit point-in-time availability metadata."""

    api_key: str | None = None
    release_lag_days: int = 1
    name: str = "fred"

    def fetch_series(
        self,
        symbol: str,
        start: date | str | None = None,
        end: date | str | None = None,
        *,
        vintage: date | str | None = None,
        release_lag_days: int | None = None,
        **_: Any,
    ) -> pd.DataFrame:
        key = self.api_key or os.getenv("FRED_API_KEY")
        if key:
            try:
                from fredapi import Fred
            except ImportError as exc:  # pragma: no cover - dependency error path
                raise RuntimeError("FRED provider requires the 'fredapi' package") from exc
            fred = Fred(api_key=key)
            if vintage is None:
                series = fred.get_series(symbol, observation_start=start, observation_end=end)
            else:
                # This method returns the values known on the requested vintage date.
                series = fred.get_series_as_of_date(symbol, vintage)
                if isinstance(series, pd.DataFrame):
                    date_col = "date" if "date" in series else "realtime_start"
                    value_col = "value"
                    series = pd.Series(series[value_col].to_numpy(), index=series[date_col])
                series.index = pd.to_datetime(series.index)
                if start is not None:
                    series = series.loc[series.index >= pd.Timestamp(start)]
                if end is not None:
                    series = series.loc[series.index <= pd.Timestamp(end)]
        else:
            if vintage is not None:
                raise RuntimeError(
                    "ALFRED vintage retrieval requires FRED_API_KEY; use a locked CSV export "
                    "when no key is available."
                )
            # Current, non-vintage FRED observations are publicly downloadable
            # without a key.  This is sufficient for DGS3MO, which is not revised.
            parameters = {"id": symbol}
            if start is not None:
                parameters["cosd"] = str(pd.Timestamp(start).date())
            if end is not None:
                parameters["coed"] = str(pd.Timestamp(end).date())
            url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?{urlencode(parameters)}"
            downloaded = pd.read_csv(url)
            date_col = next(
                (column for column in ("observation_date", "DATE", "date") if column in downloaded),
                None,
            )
            value_col = symbol if symbol in downloaded else None
            if value_col is None:
                candidates = [column for column in downloaded if column != date_col]
                value_col = candidates[0] if len(candidates) == 1 else None
            if date_col is None or value_col is None:
                raise ValueError(
                    f"Unexpected FRED CSV schema for {symbol}: "
                    f"{downloaded.columns.tolist()}"
                )
            series = pd.Series(
                downloaded[value_col].to_numpy(),
                index=pd.to_datetime(downloaded[date_col], errors="raise"),
            )
        result = pd.DataFrame(
            {"observation_date": pd.to_datetime(series.index), "value": series.to_numpy()}
        )
        result["value"] = pd.to_numeric(result["value"], errors="coerce")
        result = result.dropna(subset=["value"])
        return add_release_dates(
            result,
            observation_col="observation_date",
            lag_days=self.release_lag_days if release_lag_days is None else release_lag_days,
        )

    def fetch_all_releases(
        self,
        symbol: str,
        start: date | str | None = None,
        end: date | str | None = None,
    ) -> pd.DataFrame:
        """Fetch ALFRED releases without replacing history by final revised values."""

        key = self.api_key or os.getenv("FRED_API_KEY")
        if not key:
            raise RuntimeError("FRED_API_KEY is required for ALFRED all-release history")
        try:
            from fredapi import Fred
        except ImportError as exc:  # pragma: no cover - dependency error path
            raise RuntimeError("ALFRED retrieval requires the 'fredapi' package") from exc
        releases = Fred(api_key=key).get_series_all_releases(symbol)
        required = {"realtime_start", "date", "value"}
        if not required.issubset(releases.columns):
            raise ValueError(
                f"Unexpected ALFRED schema for {symbol}: {releases.columns.tolist()}"
            )
        result = pd.DataFrame(
            {
                "observation_date": pd.to_datetime(releases["date"], errors="raise"),
                "realtime_start": pd.to_datetime(
                    releases["realtime_start"], errors="raise"
                ),
                "value": pd.to_numeric(releases["value"], errors="coerce"),
            }
        ).dropna(subset=["value"])
        if start is not None:
            result = result.loc[result["observation_date"] >= pd.Timestamp(start)]
        if end is not None:
            result = result.loc[result["realtime_start"] <= pd.Timestamp(end)]
        # Release timestamps do not encode an intraday publication time.  The
        # conservative rule consumes them from the following calendar day,
        # which naturally rolls to the next market session in the as-of join.
        result["available_at"] = result["realtime_start"] + pd.Timedelta(days=1)
        # Preserve every raw release/revision in the immutable snapshot.  The
        # consumer derives a latest-observation-as-of stream without losing
        # same-day revisions or letting an old-period revision replace a newer
        # observation.
        return result.sort_values(["available_at", "observation_date"]).reset_index(
            drop=True
        )

    def fetch(
        self,
        symbol: str,
        start: date | str | None = None,
        end: date | str | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        return self.fetch_series(symbol, start, end, **kwargs)


@dataclass(slots=True)
class TushareDataProvider:
    """Optional Tushare Pro index adapter; credentials stay in the environment."""

    token: str | None = None
    endpoint: str = "index_daily"
    name: str = "tushare"

    def fetch(
        self,
        symbol: str,
        start: date | str | None = None,
        end: date | str | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        try:
            import tushare as ts
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "Tushare is optional; install 'tushare' before using this provider"
            ) from exc
        token = self.token or os.getenv("TUSHARE_TOKEN")
        if not token:
            raise RuntimeError("TUSHARE_TOKEN is required for the Tushare provider")
        pro = ts.pro_api(token)
        endpoint = str(kwargs.pop("endpoint", self.endpoint))
        if not hasattr(pro, endpoint):
            raise ValueError(f"Unknown Tushare Pro endpoint: {endpoint}")
        frame = getattr(pro, endpoint)(
            ts_code=symbol,
            start_date=_compact_date(start) or "19900101",
            end_date=_compact_date(end) or pd.Timestamp.today().strftime("%Y%m%d"),
            **kwargs,
        )
        return normalize_ohlcv(frame, start=start, end=end)


@dataclass(slots=True)
class WindDataProvider:
    """Optional WindPy adapter; unavailable installations fail explicitly."""

    fields: str = "open,high,low,close,volume"
    options: str = "PriceAdj=F"
    name: str = "wind"

    def fetch(
        self,
        symbol: str,
        start: date | str | None = None,
        end: date | str | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        try:
            from WindPy import w
        except ImportError as exc:  # pragma: no cover - proprietary dependency
            raise RuntimeError(
                "WindPy is optional and must be installed from a licensed Wind terminal"
            ) from exc

        if not w.isconnected():
            response = w.start()
            if getattr(response, "ErrorCode", 0) != 0:
                raise RuntimeError(f"Wind connection failed: {response}")
        response = w.wsd(
            symbol,
            kwargs.pop("fields", self.fields),
            str(pd.Timestamp(start).date()) if start is not None else "1990-01-01",
            str(pd.Timestamp(end).date()) if end is not None else str(pd.Timestamp.today().date()),
            kwargs.pop("options", self.options),
        )
        if response.ErrorCode != 0:
            raise RuntimeError(f"Wind request failed with error {response.ErrorCode}")
        frame = pd.DataFrame(
            np.asarray(response.Data).T,
            index=pd.to_datetime(response.Times),
            columns=[str(code).lower() for code in response.Fields],
        )
        return normalize_ohlcv(frame, start=start, end=end)


ProviderAdapter = (
    YahooDataProvider
    | AKShareDataProvider
    | CSVDataProvider
    | FREDDataProvider
    | TushareDataProvider
    | WindDataProvider
)


def get_provider(name: str, **kwargs: Any) -> ProviderAdapter:
    """Construct a provider by stable configuration name."""

    providers = {
        "yahoo": YahooDataProvider,
        "akshare": AKShareDataProvider,
        "csv": CSVDataProvider,
        "fred": FREDDataProvider,
        "tushare": TushareDataProvider,
        "wind": WindDataProvider,
    }
    normalized = name.strip().lower()
    if normalized not in providers:
        raise ValueError(f"Unknown data provider {name!r}; choose from {sorted(providers)}")
    return providers[normalized](**kwargs)
