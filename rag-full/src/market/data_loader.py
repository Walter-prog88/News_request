"""Load local market and factor data for the research workflow.

The first implementation reuses the existing ``tushare_quant`` parquet files.
This keeps the demo deterministic and avoids relying on real-time APIs.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path

import pandas as pd


LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TUSHARE_DATA_DIR = Path(
    os.environ.get("RAG_MARKET_DATA_DIR")
    or PROJECT_ROOT.parent / "tushare_quant" / "quant_data_6y"
).expanduser()


def normalize_ts_code(stock_code: str) -> str:
    """Normalize user stock code input to Tushare ``ts_code`` format."""
    value = stock_code.strip().upper()
    if "." in value:
        return value
    if value.startswith(("6", "9")):
        return f"{value}.SH"
    return f"{value}.SZ"


@lru_cache(maxsize=4)
def load_panel(data_dir: str | Path = DEFAULT_TUSHARE_DATA_DIR) -> pd.DataFrame:
    """Load factor panel from parquet."""
    path = Path(data_dir) / "panel_with_factors.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Factor panel not found: {path}")
    LOGGER.info("Loading factor panel: %s", path)
    panel = pd.read_parquet(path)
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    return panel.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)


@lru_cache(maxsize=4)
def load_stock_basic(data_dir: str | Path = DEFAULT_TUSHARE_DATA_DIR) -> pd.DataFrame:
    """Load stock basic information."""
    path = Path(data_dir) / "stock_basic.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Stock basic file not found: {path}")
    return pd.read_parquet(path)


@lru_cache(maxsize=4)
def load_factor_summary(data_dir: str | Path = DEFAULT_TUSHARE_DATA_DIR) -> pd.DataFrame:
    """Load factor IC summary if available."""
    path = Path(data_dir) / "factor_ic_summary.csv"
    if not path.exists():
        LOGGER.warning("Factor IC summary not found: %s", path)
        return pd.DataFrame()
    return pd.read_csv(path)


def get_stock_info(stock_code: str, data_dir: str | Path = DEFAULT_TUSHARE_DATA_DIR) -> dict:
    """Return company basic information for a stock code."""
    ts_code = normalize_ts_code(stock_code)
    stock_basic = load_stock_basic(data_dir)
    row = stock_basic[stock_basic["ts_code"] == ts_code]
    if row.empty:
        return {"ts_code": ts_code, "name": None, "industry": None, "market": None}
    record = row.iloc[0].to_dict()
    return {
        "ts_code": record.get("ts_code"),
        "symbol": record.get("symbol"),
        "name": record.get("name"),
        "area": record.get("area"),
        "industry": record.get("industry"),
        "market": record.get("market"),
        "list_date": record.get("list_date"),
    }


def get_stock_panel(
    stock_code: str,
    *,
    data_dir: str | Path = DEFAULT_TUSHARE_DATA_DIR,
    lookback_days: int | None = None,
) -> pd.DataFrame:
    """Return sorted panel rows for one stock."""
    ts_code = normalize_ts_code(stock_code)
    panel = load_panel(data_dir)
    stock_panel = panel[panel["ts_code"] == ts_code].copy()
    if stock_panel.empty:
        raise ValueError(f"No panel data found for {ts_code}")
    stock_panel = stock_panel.sort_values("trade_date").reset_index(drop=True)
    if lookback_days:
        stock_panel = stock_panel.tail(lookback_days).reset_index(drop=True)
    return stock_panel

