from __future__ import annotations

import numpy as np
import pandas as pd
import pandas.testing as pdt

from index_timing.features.caitong import (
    PointInTimeSource,
    assemble_caitong_features,
    build_technical_features,
    make_caitong_targets,
    point_in_time_join,
)


def _market(rows: int = 100) -> pd.DataFrame:
    index = pd.bdate_range("2020-01-01", periods=rows)
    close = pd.Series(np.arange(1.0, rows + 1.0), index=index)
    return pd.DataFrame(
        {
            "open": close - 0.2,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": 1_000.0 + np.arange(rows) * 10.0,
            "amount": (1_000.0 + np.arange(rows) * 10.0) * close,
            "turnover": 0.01 + np.arange(rows) / 100_000.0,
        },
        index=index,
    )


def test_technical_features_use_registered_windows_and_are_causal() -> None:
    market = _market()
    actual = build_technical_features(market)

    for window in (5, 10, 20, 60):
        assert f"return_{window}d" in actual
        assert f"close_to_ma_{window}d" in actual
        assert f"volume_to_ma_{window}d" in actual
        assert f"rsi_{window}d" in actual
    assert actual.loc[market.index[5], "return_5d"] == 5.0
    assert np.isclose(actual.loc[market.index[4], "close_to_ma_5d"], 5.0 / 3.0 - 1.0)

    changed_future = market.copy()
    changed_future.loc[market.index[-1], ["close", "volume", "amount"]] *= 100.0
    changed = build_technical_features(changed_future)
    pdt.assert_frame_equal(actual.iloc[:-1], changed.iloc[:-1])


def test_point_in_time_join_uses_release_date_not_observation_period() -> None:
    index = pd.date_range("2023-01-01", periods=6, freq="D")
    base = pd.DataFrame({"base": np.arange(6.0)}, index=index)
    macro = pd.DataFrame(
        {
            "observation_date": ["2022-10-01", "2022-11-01"],
            "released": ["2023-01-03", "2023-01-05"],
            "pmi": [49.0, 51.0],
        }
    )
    source = PointInTimeSource(
        name="official_macro",
        group="low_frequency_macro",
        data=macro,
        release_timestamp_col="released",
        value_columns=("pmi",),
    )
    joined, columns = point_in_time_join(base, source, keep_available_at=True)

    assert columns == ("official_macro__pmi",)
    assert np.isnan(joined.loc["2023-01-02", "official_macro__pmi"])
    assert joined.loc["2023-01-03", "official_macro__pmi"] == 49.0
    assert joined.loc["2023-01-04", "official_macro__pmi"] == 49.0
    assert joined.loc["2023-01-05", "official_macro__pmi"] == 51.0
    available = joined["official_macro__available_at"].dropna()
    assert (available <= available.index).all()


def test_assembled_groups_preserve_ablation_order() -> None:
    market = _market()
    index = market.index[[10, 40, 70]]
    sources = [
        PointInTimeSource(
            name="valuation",
            group="valuation_fund",
            data=pd.DataFrame({"pe": [10.0, 11.0, 12.0]}, index=index),
        ),
        PointInTimeSource(
            name="rates",
            group="rates_commodities_overseas",
            data=pd.DataFrame({"yield_10y": [2.0, 2.1, 2.2]}, index=index),
        ),
        PointInTimeSource(
            name="macro",
            group="low_frequency_macro",
            data=pd.DataFrame({"pmi": [49.0, 50.0, 51.0]}, index=index),
        ),
    ]
    bundle = assemble_caitong_features(market, pit_sources=sources)

    technical = bundle.columns_for("technical")
    first_ablation = bundle.columns_for("technical_plus_valuation_fund")
    market_global = bundle.columns_for("technical_plus_market_global")
    full = bundle.columns_for("full")
    assert first_ablation[: len(technical)] == technical
    assert first_ablation[-1] == "valuation__pe"
    assert market_global[-1] == "rates__yield_10y"
    assert full[-1] == "macro__pmi"


def test_forward_targets_have_explicit_realization_dates() -> None:
    index = pd.bdate_range("2023-01-02", periods=25)
    close = pd.Series(np.arange(100.0, 125.0), index=index)
    targets = make_caitong_targets(close)

    assert targets.loc[index[0], "target_up_5d"] == 1.0
    assert targets.loc[index[0], "target_available_at_5d"] == index[5]
    assert np.isclose(targets.loc[index[0], "forward_return_5d"], 105.0 / 100.0 - 1.0)
    assert targets["target_up_5d"].tail(5).isna().all()

