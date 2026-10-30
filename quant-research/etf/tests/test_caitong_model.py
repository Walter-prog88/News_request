from __future__ import annotations

import numpy as np
import pandas as pd
import pandas.testing as pdt

from index_timing.experiments.caitong import (
    PAPER_TRAIN_END,
    run_caitong_experiment,
    summarise_signal_intervals,
)
from index_timing.models.caitong import CaitongClassifier, CaitongModelConfig


def _classification_data(rows: int = 160) -> tuple[pd.DataFrame, pd.Series]:
    index = pd.bdate_range("2020-01-01", periods=rows)
    angle = np.linspace(0.0, 16.0 * np.pi, rows)
    X = pd.DataFrame(
        {
            "momentum": np.sin(angle),
            "valuation": np.cos(angle),
            "macro": np.linspace(-1.0, 1.0, rows),
        },
        index=index,
    )
    y = pd.Series((np.sin(angle + 0.4) > 0.0).astype(float), index=index)
    return X, y


def test_registered_primary_model_parameters() -> None:
    config = CaitongModelConfig()
    assert config.kind == "mlp"
    assert config.hidden_sizes == (128, 64)
    assert config.learning_rate == 1e-3
    assert config.weight_decay == 1e-4
    assert config.patience == 10


def test_elastic_net_scaler_is_fitted_only_on_supplied_training_rows() -> None:
    X, y = _classification_data()
    train_X, train_y = X.iloc[:100], y.iloc[:100]
    model = CaitongClassifier(
        CaitongModelConfig(kind="elastic_net_logistic", seed=7)
    ).fit(train_X, train_y)

    assert model.scaler_ is not None
    np.testing.assert_allclose(model.scaler_.mean_, train_X.mean().to_numpy(), atol=1e-12)
    before = model.scaler_.mean_.copy()
    changed_test = X.iloc[100:].copy() + 1_000_000.0
    probabilities = model.predict_proba(changed_test)
    np.testing.assert_array_equal(model.scaler_.mean_, before)
    assert probabilities.between(0.0, 1.0).all()
    assert model.metadata["scaler_fit_end"] == train_X.index[-1].isoformat()


def test_mlp_training_is_deterministic_for_fixed_seed() -> None:
    X, y = _classification_data(120)
    config = CaitongModelConfig(seed=19, max_epochs=4, patience=2, batch_size=32)
    first = CaitongClassifier(config).fit(X, y).predict_proba(X)
    second = CaitongClassifier(config).fit(X, y).predict_proba(X)

    pdt.assert_series_equal(first, second, check_exact=False, atol=1e-7, rtol=1e-7)


def test_walk_forward_uses_explicit_validation_without_fitting_its_scale() -> None:
    X, y = _classification_data(180)
    train_X, train_y = X.iloc[:100], y.iloc[:100]
    validation_X = X.iloc[120:].copy() + 10_000.0
    validation_y = y.iloc[120:]
    model = CaitongClassifier(
        CaitongModelConfig(seed=5, max_epochs=2, patience=1)
    ).fit(
        train_X,
        train_y,
        validation_X=validation_X,
        validation_y=validation_y,
    )

    assert model.scaler_ is not None
    np.testing.assert_allclose(model.scaler_.mean_, train_X.mean().to_numpy(), atol=1e-12)
    assert model.metadata["explicit_walk_forward_validation"] is True
    assert model.metadata["validation_start"] == validation_X.index[0].isoformat()
    assert model.metadata["validation_rows"] == len(validation_X)


def test_fixed_split_purges_labels_and_applies_ma5_then_next_day_exposure() -> None:
    index = pd.bdate_range("2020-01-02", "2023-09-29")
    angle = np.linspace(0.0, 40.0 * np.pi, len(index))
    close = pd.Series(100.0 + 8.0 * np.sin(angle) + np.linspace(0.0, 5.0, len(index)), index=index)
    features = pd.DataFrame(
        {
            "momentum": pd.Series(close, index=index).pct_change(5),
            "cycle": np.sin(angle - 0.5),
            "slow_cycle": np.cos(angle / 3.0),
        },
        index=index,
    )

    result = run_caitong_experiment(
        features,
        close,
        model_kind="elastic_net_logistic",
        seed=3,
        test_end="2023-09-29",
    )

    assert result.daily.index.min() == pd.Timestamp("2023-06-01")
    assert result.daily["probability_ma5"].iloc[:4].isna().all()
    assert result.daily["signal"].iloc[:4].isna().all()
    pdt.assert_series_equal(
        result.daily["exposure"], result.daily["signal"].shift(1), check_names=False
    )
    assert pd.Timestamp(result.metadata["latest_training_label_realization"]) <= PAPER_TRAIN_END
    assert result.metadata["primary_specification"] is True
    assert result.metadata["is_author_original_model"] is False
    assert result.summary["horizon_days"] == 5
    assert result.summary["calibration_rows"] > 0
    assert 0.0 <= result.summary["brier_score"] <= 1.0
    assert result.calibration["observations"].sum() == result.summary["calibration_rows"]


def test_interval_scoring_distinguishes_bullish_and_adjustment() -> None:
    index = pd.bdate_range("2023-01-02", periods=6)
    daily = pd.DataFrame(
        {
            "signal": [1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
            "forward_1d_asset_return": [0.01, 0.02, -0.01, -0.02, -0.01, np.nan],
        },
        index=index,
    )
    intervals = summarise_signal_intervals(daily)

    assert intervals["regime"].tolist() == ["bullish", "adjustment", "bullish"]
    assert intervals["win"].tolist() == [True, True, False]
