"""Technical signal calculation for one stock."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.market.data_loader import DEFAULT_TUSHARE_DATA_DIR, get_stock_panel


def _safe_pct_change(series: pd.Series, periods: int) -> float | None:
    """Return percentage change over a fixed period."""
    if len(series) <= periods:
        return None
    start = series.iloc[-periods - 1]
    end = series.iloc[-1]
    if pd.isna(start) or pd.isna(end) or start == 0:
        return None
    return float(end / start - 1)


def _max_drawdown(close: pd.Series) -> float | None:
    """Calculate max drawdown from close prices."""
    close = close.dropna()
    if close.empty:
        return None
    running_max = close.cummax()
    drawdown = close / running_max - 1
    return float(drawdown.min())


def calculate_technical_signals(
    stock_code: str,
    *,
    data_dir: str | Path = DEFAULT_TUSHARE_DATA_DIR,
    lookback_days: int = 260,
) -> dict:
    """Calculate recent returns, volume, moving-average, volatility, and drawdown."""
    df = get_stock_panel(stock_code, data_dir=data_dir, lookback_days=lookback_days)
    close = df["close"].astype(float)
    volume = df["volume"].astype(float)
    ret_1d = close.pct_change()

    latest = df.iloc[-1]
    ma20 = close.rolling(20).mean().iloc[-1]
    ma60 = close.rolling(60).mean().iloc[-1]
    vol20 = ret_1d.rolling(20).std().iloc[-1] * np.sqrt(252)
    volume_ratio = volume.iloc[-1] / volume.rolling(20).mean().iloc[-1]

    trend_score = 0
    if pd.notna(ma20) and close.iloc[-1] > ma20:
        trend_score += 1
    if pd.notna(ma60) and close.iloc[-1] > ma60:
        trend_score += 1
    ret20 = _safe_pct_change(close, 20)
    if ret20 is not None and ret20 > 0:
        trend_score += 1

    return {
        "as_of": latest["trade_date"].strftime("%Y-%m-%d"),
        "close": float(latest["close"]),
        "return_5d": _safe_pct_change(close, 5),
        "return_20d": ret20,
        "return_60d": _safe_pct_change(close, 60),
        "volume_ratio_20d": float(volume_ratio) if pd.notna(volume_ratio) else None,
        "ma20": float(ma20) if pd.notna(ma20) else None,
        "ma60": float(ma60) if pd.notna(ma60) else None,
        "above_ma20": bool(pd.notna(ma20) and close.iloc[-1] > ma20),
        "above_ma60": bool(pd.notna(ma60) and close.iloc[-1] > ma60),
        "volatility_20d_annualized": float(vol20) if pd.notna(vol20) else None,
        "max_drawdown_60d": _max_drawdown(close.tail(60)),
        "max_drawdown_120d": _max_drawdown(close.tail(120)),
        "trend_score": trend_score,
    }


def get_price_history(
    stock_code: str,
    *,
    data_dir: str | Path = DEFAULT_TUSHARE_DATA_DIR,
    lookback_days: int = 180,
) -> pd.DataFrame:
    """Return price history with moving averages for charting."""
    df = get_stock_panel(stock_code, data_dir=data_dir, lookback_days=lookback_days)
    history = df[["trade_date", "close", "volume"]].copy()
    history["trade_date"] = pd.to_datetime(history["trade_date"])
    history["ma20"] = history["close"].rolling(20).mean()
    history["ma60"] = history["close"].rolling(60).mean()
    history["return_1d"] = history["close"].pct_change()
    return history

