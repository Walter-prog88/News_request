"""Risk checks for a stock research workflow."""

from __future__ import annotations

from src.market.technical import calculate_technical_signals


def risk_check(stock_code: str) -> dict:
    """Generate simple risk flags from technical signals."""
    technical = calculate_technical_signals(stock_code)
    flags: list[str] = []

    ret20 = technical.get("return_20d")
    ret60 = technical.get("return_60d")
    vol20 = technical.get("volatility_20d_annualized")
    dd60 = technical.get("max_drawdown_60d")
    volume_ratio = technical.get("volume_ratio_20d")

    if ret20 is not None and ret20 > 0.25:
        flags.append("近20日涨幅较大，存在短期交易拥挤和回撤风险")
    if ret60 is not None and ret60 > 0.50:
        flags.append("近60日涨幅过高，需警惕主题交易过热")
    if vol20 is not None and vol20 > 0.60:
        flags.append("20日年化波动率偏高，持仓波动压力较大")
    if dd60 is not None and dd60 < -0.20:
        flags.append("近60日最大回撤超过20%，趋势稳定性不足")
    if volume_ratio is not None and volume_ratio > 2.5:
        flags.append("成交量显著放大，可能存在情绪化交易")
    if not technical.get("above_ma20"):
        flags.append("股价低于20日均线，短期趋势偏弱")

    severity = "low"
    if len(flags) >= 3:
        severity = "high"
    elif len(flags) >= 1:
        severity = "medium"

    return {
        "severity": severity,
        "flags": flags,
        "technical": technical,
    }

