"""Fixed-split Caitong report reconstruction experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from index_timing.features.caitong import CaitongFeatureBundle, make_caitong_targets
from index_timing.models.caitong import CaitongClassifier, CaitongModelConfig, ModelKind

PAPER_TRAIN_END = pd.Timestamp("2023-05-31")
PAPER_TEST_START = pd.Timestamp("2023-06-01")
PAPER_REPLICATION_END = pd.Timestamp("2026-02-27")
PRIMARY_HORIZON = 5
SENSITIVITY_HORIZONS: tuple[int, ...] = (1, 10, 20)
PROBABILITY_SMOOTHING_DAYS = 5
SIGNAL_THRESHOLD = 0.5

CAITONG_PROXY_ASSUMPTIONS: tuple[str, ...] = (
    "原文未披露完整网络结构；128/64两层MLP是预注册代理而非作者原模型。",
    "主标签定义为未来5个交易日收益大于0，1/10/20日仅作敏感性检验。",
    "训练截止2023-05-31；会跨越训练截止日才能实现的标签已从训练样本剔除。",
    "每日概率使用5日移动平均，以0.5划分看多与调整区间。",
    "信号在收盘形成，策略敞口从下一交易日收益开始生效。",
)


@dataclass(slots=True)
class CaitongExperimentResult:
    """Daily predictions, interval diagnostics and an audit trail."""

    daily: pd.DataFrame
    intervals: pd.DataFrame
    calibration: pd.DataFrame
    summary: dict[str, float | int | str | None]
    metadata: dict[str, Any]
    model: CaitongClassifier


def _as_bundle(features: CaitongFeatureBundle | pd.DataFrame) -> CaitongFeatureBundle:
    if isinstance(features, CaitongFeatureBundle):
        return features
    if not isinstance(features, pd.DataFrame):
        raise TypeError("features must be a DataFrame or CaitongFeatureBundle")
    return CaitongFeatureBundle(
        frame=features,
        groups={
            "technical": tuple(features.columns),
            "valuation_fund": (),
            "rates_commodities_overseas": (),
            "low_frequency_macro": (),
        },
        assumptions=("调用方提供的未分组特征全部按technical组处理。",),
        source_metadata=(),
    )


def summarise_signal_intervals(daily: pd.DataFrame) -> pd.DataFrame:
    """Score contiguous probability regimes using their subsequent returns.

    A bullish interval wins when its compounded next-day index return is
    positive; an adjustment interval wins when that return is non-positive.
    """

    required = {"signal", "forward_1d_asset_return"}
    missing = sorted(required - set(daily.columns))
    if missing:
        raise ValueError(f"daily is missing interval fields: {missing}")
    valid = daily.loc[daily["signal"].notna()].copy()
    columns = [
        "interval_id",
        "regime",
        "start",
        "end",
        "observations",
        "asset_return",
        "win",
    ]
    if valid.empty:
        return pd.DataFrame(columns=columns)

    valid["interval_id"] = valid["signal"].ne(valid["signal"].shift()).cumsum()
    records: list[dict[str, object]] = []
    for interval_id, section in valid.groupby("interval_id", sort=True):
        signal = int(section["signal"].iloc[0])
        next_returns = section["forward_1d_asset_return"].dropna()
        compounded = (
            float((1.0 + next_returns).prod() - 1.0) if not next_returns.empty else np.nan
        )
        win = bool(compounded > 0.0) if signal == 1 else bool(compounded <= 0.0)
        records.append(
            {
                "interval_id": int(interval_id),
                "regime": "bullish" if signal == 1 else "adjustment",
                "start": section.index[0],
                "end": section.index[-1],
                "observations": int(len(section)),
                "asset_return": compounded,
                "win": win if np.isfinite(compounded) else None,
            }
        )
    return pd.DataFrame.from_records(records, columns=columns)


def _win_rate(intervals: pd.DataFrame, regime: str | None = None) -> float:
    if regime is not None:
        intervals = intervals.loc[intervals["regime"] == regime]
    outcomes = intervals["win"].dropna()
    return float(outcomes.astype(float).mean()) if not outcomes.empty else np.nan


def probability_calibration_table(
    daily: pd.DataFrame,
    *,
    probability_column: str = "probability",
    target_column: str = "target_up_5d",
    bins: int = 10,
) -> pd.DataFrame:
    """Return an auditable reliability table for realised directional labels."""

    if bins < 2:
        raise ValueError("calibration bins must be at least two")
    required = {probability_column, target_column}
    missing = sorted(required - set(daily.columns))
    if missing:
        raise ValueError(f"daily is missing calibration fields: {missing}")
    valid = daily[[probability_column, target_column]].apply(
        pd.to_numeric, errors="coerce"
    ).dropna()
    columns = [
        "bin",
        "lower",
        "upper",
        "observations",
        "mean_probability",
        "observed_frequency",
        "absolute_gap",
    ]
    if valid.empty:
        return pd.DataFrame(columns=columns)
    edges = np.linspace(0.0, 1.0, bins + 1)
    buckets = pd.cut(
        valid[probability_column].clip(0.0, 1.0),
        bins=edges,
        include_lowest=True,
        labels=False,
    )
    records: list[dict[str, float | int]] = []
    for bucket in range(bins):
        section = valid.loc[buckets == bucket]
        if section.empty:
            continue
        mean_probability = float(section[probability_column].mean())
        observed_frequency = float(section[target_column].mean())
        records.append(
            {
                "bin": bucket,
                "lower": float(edges[bucket]),
                "upper": float(edges[bucket + 1]),
                "observations": len(section),
                "mean_probability": mean_probability,
                "observed_frequency": observed_frequency,
                "absolute_gap": abs(mean_probability - observed_frequency),
            }
        )
    return pd.DataFrame.from_records(records, columns=columns)


def run_caitong_experiment(
    features: CaitongFeatureBundle | pd.DataFrame,
    close: pd.Series,
    *,
    feature_set: str = "full",
    model_kind: ModelKind = "mlp",
    horizon: int = PRIMARY_HORIZON,
    seed: int = 0,
    model_config: CaitongModelConfig | None = None,
    train_end: str | pd.Timestamp = PAPER_TRAIN_END,
    test_start: str | pd.Timestamp = PAPER_TEST_START,
    test_end: str | pd.Timestamp | None = PAPER_REPLICATION_END,
    train_index: pd.DatetimeIndex | None = None,
    validation_index: pd.DatetimeIndex | None = None,
) -> CaitongExperimentResult:
    """Fit once on the paper split and produce causal daily test predictions."""

    if horizon not in {PRIMARY_HORIZON, *SENSITIVITY_HORIZONS}:
        raise ValueError("horizon must be one of 1, 5, 10 or 20 trading days")
    train_end = pd.Timestamp(train_end)
    test_start = pd.Timestamp(test_start)
    test_end_timestamp = pd.Timestamp(test_end) if test_end is not None else None
    if not train_end < test_start:
        raise ValueError("train_end must be earlier than test_start")

    bundle = _as_bundle(features)
    selected_columns = bundle.columns_for(feature_set)
    if not selected_columns:
        raise ValueError(f"feature_set={feature_set!r} contains no available columns")

    prices = pd.to_numeric(close, errors="coerce").astype(float).copy()
    prices.index = pd.DatetimeIndex(pd.to_datetime(prices.index), name="date")
    if not prices.index.is_monotonic_increasing or prices.index.has_duplicates:
        raise ValueError("close index must be sorted and unique")
    feature_frame = bundle.frame.reindex(prices.index).loc[:, selected_columns].astype(float)
    target_horizons = (horizon,) if horizon == 1 else (horizon, 1)
    targets = make_caitong_targets(prices, horizons=target_horizons)
    target_column = f"target_up_{horizon}d"
    available_column = f"target_available_at_{horizon}d"

    if (train_index is None) != (validation_index is None):
        raise ValueError("train_index and validation_index must be supplied together")
    if train_index is not None:
        fit_membership = feature_frame.index.isin(pd.DatetimeIndex(train_index))
        label_cutoff = pd.Timestamp(pd.DatetimeIndex(train_index).max())
    else:
        fit_membership = feature_frame.index <= train_end
        label_cutoff = train_end
    train_mask = (
        fit_membership
        & targets[target_column].notna()
        & (targets[available_column] <= label_cutoff)
        & feature_frame.notna().any(axis=1)
    )
    train_X = feature_frame.loc[train_mask]
    train_y = targets.loc[train_mask, target_column]
    usable_columns = tuple(column for column in train_X if train_X[column].notna().any())
    dropped_columns = tuple(column for column in train_X if column not in usable_columns)
    if not usable_columns:
        raise ValueError("No selected feature has a non-missing training observation")
    train_X = train_X.loc[:, usable_columns]
    prediction_X = feature_frame.loc[:, usable_columns]

    if model_config is None:
        model_config = CaitongModelConfig(kind=model_kind, seed=seed)
    elif model_config.kind != model_kind or model_config.seed != seed:
        raise ValueError("model_config kind/seed must match model_kind/seed")
    validation_X: pd.DataFrame | None = None
    validation_y: pd.Series | None = None
    if validation_index is not None:
        validation_dates = pd.DatetimeIndex(validation_index)
        validation_cutoff = pd.Timestamp(validation_dates.max())
        validation_mask = (
            feature_frame.index.isin(validation_dates)
            & targets[target_column].notna()
            & (targets[available_column] <= validation_cutoff)
            & feature_frame.notna().any(axis=1)
        )
        validation_X = feature_frame.loc[validation_mask, usable_columns]
        validation_y = targets.loc[validation_mask, target_column]
    model = CaitongClassifier(model_config).fit(
        train_X,
        train_y,
        validation_X=validation_X,
        validation_y=validation_y,
    )

    probability = model.predict_proba(prediction_X)
    test_mask = probability.index >= test_start
    if test_end_timestamp is not None:
        test_mask &= probability.index <= test_end_timestamp
    daily_index = probability.index[test_mask]
    daily = pd.DataFrame(index=daily_index)
    daily["close"] = prices.reindex(daily_index)
    daily["probability"] = probability.reindex(daily_index)
    daily["probability_ma5"] = daily["probability"].rolling(
        PROBABILITY_SMOOTHING_DAYS, min_periods=PROBABILITY_SMOOTHING_DAYS
    ).mean()
    daily["signal"] = np.where(
        daily["probability_ma5"].notna(),
        (daily["probability_ma5"] > SIGNAL_THRESHOLD).astype(float),
        np.nan,
    )
    daily["regime"] = daily["signal"].map({1.0: "bullish", 0.0: "adjustment"})
    all_asset_returns = prices.pct_change(fill_method=None)
    all_forward_returns = prices.shift(-1) / prices - 1.0
    daily["asset_return"] = all_asset_returns.reindex(daily_index)
    daily["forward_1d_asset_return"] = all_forward_returns.reindex(daily_index)
    daily["exposure"] = daily["signal"].shift(1)
    daily["strategy_gross_return"] = daily["exposure"] * daily["asset_return"]
    daily["turnover"] = daily["exposure"].diff().abs()
    daily[f"forward_return_{horizon}d"] = targets[
        f"forward_return_{horizon}d"
    ].reindex(daily_index)
    daily[target_column] = targets[target_column].reindex(daily_index)

    intervals = summarise_signal_intervals(daily)
    realised_predictions = daily[["signal", target_column]].dropna()
    directional_accuracy = (
        float((realised_predictions["signal"] == realised_predictions[target_column]).mean())
        if not realised_predictions.empty
        else np.nan
    )
    calibration = probability_calibration_table(
        daily,
        probability_column="probability",
        target_column=target_column,
    )
    calibrated_rows = int(calibration["observations"].sum()) if len(calibration) else 0
    expected_calibration_error = (
        float(
            (
                calibration["absolute_gap"]
                * calibration["observations"]
                / calibrated_rows
            ).sum()
        )
        if calibrated_rows
        else np.nan
    )
    calibration_values = daily[["probability", target_column]].dropna()
    brier_score = (
        float(
            np.mean(
                np.square(
                    calibration_values["probability"]
                    - calibration_values[target_column]
                )
            )
        )
        if len(calibration_values)
        else np.nan
    )
    summary: dict[str, float | int | str | None] = {
        "model_kind": model_kind,
        "feature_set": feature_set,
        "horizon_days": horizon,
        "train_rows": len(train_X),
        "test_rows": len(daily),
        "intervals": len(intervals),
        "interval_win_rate": _win_rate(intervals),
        "bullish_interval_win_rate": _win_rate(intervals, "bullish"),
        "adjustment_interval_win_rate": _win_rate(intervals, "adjustment"),
        "daily_directional_accuracy": directional_accuracy,
        "brier_score": brier_score,
        "expected_calibration_error_10bin": expected_calibration_error,
        "calibration_rows": calibrated_rows,
    }
    metadata: dict[str, Any] = {
        "study": "caitong_broad_index_timing_proxy",
        "replication_level": "method_reconstruction",
        "is_author_original_model": False,
        "train_end": train_end.date().isoformat(),
        "test_start": test_start.date().isoformat(),
        "paper_replication_end": PAPER_REPLICATION_END.date().isoformat(),
        "requested_test_end": test_end_timestamp.date().isoformat()
        if test_end_timestamp is not None
        else None,
        "walk_forward_fit_rows": len(train_X) if train_index is not None else None,
        "walk_forward_validation_rows": (
            len(validation_X) if validation_X is not None else None
        ),
        "target_horizon_days": horizon,
        "primary_specification": horizon == PRIMARY_HORIZON,
        "probability_smoothing_days": PROBABILITY_SMOOTHING_DAYS,
        "signal_threshold": SIGNAL_THRESHOLD,
        "feature_set": feature_set,
        "feature_groups": {group: list(columns) for group, columns in bundle.groups.items()},
        "selected_features": list(usable_columns),
        "dropped_all_missing_train_features": list(dropped_columns),
        "latest_training_label_realization": (
            pd.Timestamp(targets.loc[train_mask, available_column].max()).date().isoformat()
        ),
        "feature_sources": list(bundle.source_metadata),
        "model": model.metadata,
        "assumptions": list((*CAITONG_PROXY_ASSUMPTIONS, *bundle.assumptions)),
    }
    return CaitongExperimentResult(
        daily=daily,
        intervals=intervals,
        calibration=calibration,
        summary=summary,
        metadata=metadata,
        model=model,
    )


def run_horizon_sensitivity(
    features: CaitongFeatureBundle | pd.DataFrame,
    close: pd.Series,
    *,
    feature_set: str = "full",
    model_kind: ModelKind = "mlp",
    seed: int = 0,
    include_primary: bool = True,
    **split_overrides: object,
) -> dict[int, CaitongExperimentResult]:
    """Run the registered 1/5/10/20-day target comparison."""

    horizons = (1, 5, 10, 20) if include_primary else SENSITIVITY_HORIZONS
    return {
        horizon: run_caitong_experiment(
            features,
            close,
            feature_set=feature_set,
            model_kind=model_kind,
            horizon=horizon,
            seed=seed,
            **split_overrides,
        )
        for horizon in horizons
    }
