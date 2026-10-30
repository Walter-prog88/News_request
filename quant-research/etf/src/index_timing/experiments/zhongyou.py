"""Fixed-split Zhongyou LSTM-AE → GHMM → Kelly reconstruction experiments."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pandas as pd

from index_timing.features.zhongyou import (
    ZhongyouFeatureBundle,
    assemble_zhongyou_features,
)
from index_timing.models.zhongyou import (
    AutoencoderConfig,
    GHMMConfig,
    RepresentationKind,
    ZhongyouModelConfig,
    ZhongyouStateModel,
    apply_bottom_recovery_patch,
    apply_microcap_opening_patch,
    kelly_position_details,
)

PAPER_TRAIN_START_2008 = pd.Timestamp("2008-01-01")
PAPER_TRAIN_START_2016 = pd.Timestamp("2016-01-01")
PAPER_TRAIN_END = pd.Timestamp("2020-12-31")
PAPER_TEST_START = pd.Timestamp("2021-01-01")
STAR_COMPOSITE_TRAIN_END = pd.Timestamp("2022-12-31")

ZHONGYOU_PROXY_ASSUMPTIONS: tuple[str, ...] = (
    "正文逐项仅恢复17维而非其所称25+维，忠实代理不补入未披露特征。",
    "LSTM仅优化90日历史序列重建误差，不以未来收益作为标签。",
    "原文未披露六状态双Sigmoid方程；采用a=0.524、k=5的预注册软规则代理。",
    "GHMM固定5状态、full covariance和20次初始化，并以因果前向概率推断。",
    "凯利仓位只读取当前日期以前已经实现的15个同MAP状态次日超额收益。",
    "微盘满仓和底部修复属于事后仓位补丁，不进入主规格有效性结论。",
)


@dataclass(slots=True)
class ZhongyouExperimentResult:
    daily: pd.DataFrame
    causal_context: pd.DataFrame
    summary: dict[str, float | int | str | None]
    metadata: dict[str, Any]
    features: ZhongyouFeatureBundle
    model: ZhongyouStateModel


def _cash_daily(
    cash_returns: pd.Series | float | int | None, index: pd.DatetimeIndex
) -> pd.Series:
    if cash_returns is None:
        return pd.Series(0.0, index=index, name="cash_return")
    if np.isscalar(cash_returns):
        return pd.Series(float(cash_returns), index=index, name="cash_return")
    output = pd.to_numeric(cash_returns, errors="coerce").copy()
    output.index = pd.DatetimeIndex(pd.to_datetime(output.index))
    return output.reindex(index).ffill().fillna(0.0).rename("cash_return")


def _default_model_config(
    representation: RepresentationKind, seed: int
) -> ZhongyouModelConfig:
    return ZhongyouModelConfig(
        representation=representation,
        autoencoder=AutoencoderConfig(seed=seed),
        ghmm=GHMMConfig(seed=seed),
    )


def run_zhongyou_experiment(
    market: pd.DataFrame,
    *,
    representation: RepresentationKind = "lstm",
    seed: int = 0,
    model_config: ZhongyouModelConfig | None = None,
    train_start: str | pd.Timestamp = PAPER_TRAIN_START_2008,
    train_end: str | pd.Timestamp = PAPER_TRAIN_END,
    test_start: str | pd.Timestamp = PAPER_TEST_START,
    test_end: str | pd.Timestamp | None = None,
    train_index: pd.DatetimeIndex | None = None,
    validation_index: pd.DatetimeIndex | None = None,
    cash_returns: pd.Series | float | int | None = None,
    kelly_lookback: int = 15,
    microcap_patch: bool = False,
    bottom_patch: bool = False,
    bottom_trigger: pd.Series | None = None,
) -> ZhongyouExperimentResult:
    """Train on a disclosed split and produce causal states and target positions.

    ``cash_returns`` are daily decimal returns already known for their row.
    The returned close-to-close strategy return is a diagnostic; the project's
    canonical next-open, cost-aware engine consumes ``target_position`` later.
    """

    train_start_timestamp = pd.Timestamp(train_start)
    train_end_timestamp = pd.Timestamp(train_end)
    test_start_timestamp = pd.Timestamp(test_start)
    test_end_timestamp = pd.Timestamp(test_end) if test_end is not None else None
    if not train_start_timestamp <= train_end_timestamp < test_start_timestamp:
        raise ValueError("split must satisfy train_start <= train_end < test_start")
    if model_config is None:
        model_config = _default_model_config(representation, seed)
    elif model_config.representation != representation:
        raise ValueError("model_config representation must match representation")

    features = assemble_zhongyou_features(market)
    model = ZhongyouStateModel(model_config).fit(
        features.standardized,
        features.soft_states,
        train_start=train_start_timestamp,
        train_end=train_end_timestamp,
        train_index=train_index,
        validation_index=validation_index,
    )
    probabilities = model.filter_probabilities(
        features.standardized, features.soft_states
    )
    states = model.map_states(features.standardized, features.soft_states)

    close = pd.to_numeric(market["close"], errors="coerce").astype(float)
    close.index = pd.DatetimeIndex(pd.to_datetime(close.index), name="date")
    asset_return = close.pct_change(fill_method=None).rename("asset_return")
    cash_return = _cash_daily(cash_returns, close.index)
    next_day_excess = (
        asset_return.shift(-1) - cash_return.shift(-1)
    ).rename("next_day_excess_return")
    kelly = kelly_position_details(
        states,
        next_day_excess,
        lookback_same_state=kelly_lookback,
    )
    base_position = kelly["kelly_position"].rename("base_kelly_position")
    target = apply_microcap_opening_patch(
        base_position, enabled=microcap_patch
    ).rename("target_position")
    target = apply_bottom_recovery_patch(
        target,
        bottom_trigger,
        enabled=bottom_patch,
    ).rename("target_position")
    exposure = target.shift(1).rename("exposure")
    strategy_return = (
        exposure * asset_return + (1.0 - exposure) * cash_return
    ).rename("diagnostic_gross_return")

    full = pd.concat(
        [
            close.rename("close"),
            probabilities,
            states,
            kelly,
            base_position,
            target,
            exposure,
            asset_return,
            cash_return,
            next_day_excess,
            strategy_return,
        ],
        axis=1,
    )
    test_mask = full.index >= test_start_timestamp
    if test_end_timestamp is not None:
        test_mask &= full.index <= test_end_timestamp
    daily = full.loc[test_mask].copy()

    valid_return = daily["diagnostic_gross_return"].dropna()
    summary: dict[str, float | int | str | None] = {
        "representation": representation,
        "train_observations": (
            len(model.training_observations_)
            if model.training_observations_ is not None
            else 0
        ),
        "test_rows": len(daily),
        "test_state_rows": int(daily["map_state"].notna().sum()),
        "average_target_position": float(daily["target_position"].mean()),
        "gross_total_return": (
            float((1.0 + valid_return).prod() - 1.0) if len(valid_return) else np.nan
        ),
    }
    metadata: dict[str, Any] = {
        "study": "zhongyou_lstm_ghmm_proxy",
        "replication_level": "method_reconstruction",
        "is_author_original_code": False,
        "train_start": train_start_timestamp.date().isoformat(),
        "train_end": train_end_timestamp.date().isoformat(),
        "test_start": test_start_timestamp.date().isoformat(),
        "test_end": test_end_timestamp.date().isoformat() if test_end_timestamp else None,
        "walk_forward_fit_rows": (
            len(model.training_observations_) if train_index is not None else None
        ),
        "walk_forward_validation_rows": (
            len(model.validation_observations_)
            if model.validation_observations_ is not None
            else None
        ),
        "training_regime": (
            "paper_primary_2008_2020"
            if train_start_timestamp == PAPER_TRAIN_START_2008
            else "paper_conflict_2016_2020"
            if train_start_timestamp == PAPER_TRAIN_START_2016
            else "custom"
        ),
        "feature_dimensions_disclosed": 17,
        "soft_state_dimensions": 6,
        "kelly_lookback_same_state": kelly_lookback,
        "microcap_patch": {
            "enabled": microcap_patch,
            "post_hoc": True,
            "rule": "Kelly > 50% forced to 100%",
        },
        "bottom_patch": {
            "enabled": bottom_patch,
            "post_hoc": True,
            "rule": "50% + 50% × Kelly when external trigger is true",
        },
        "model": model.metadata,
        "assumptions": list((*ZHONGYOU_PROXY_ASSUMPTIONS, *features.assumptions)),
    }
    return ZhongyouExperimentResult(
        daily=daily,
        causal_context=full.loc[
            :, ["map_state", "next_day_excess_return"]
        ].copy(),
        summary=summary,
        metadata=metadata,
        features=features,
        model=model,
    )


def run_training_window_sensitivity(
    market: pd.DataFrame,
    *,
    representation: RepresentationKind = "lstm",
    seed: int = 0,
    **overrides: object,
) -> dict[str, ZhongyouExperimentResult]:
    """Run the contradictory 2008-2020 and 2016-2020 paper train windows."""

    return {
        "2008_2020": run_zhongyou_experiment(
            market,
            representation=representation,
            seed=seed,
            train_start=PAPER_TRAIN_START_2008,
            **overrides,
        ),
        "2016_2020": run_zhongyou_experiment(
            market,
            representation=representation,
            seed=seed,
            train_start=PAPER_TRAIN_START_2016,
            **overrides,
        ),
    }


def run_representation_comparison(
    market: pd.DataFrame,
    *,
    base_config: ZhongyouModelConfig | None = None,
    seed: int = 0,
    **overrides: object,
) -> dict[RepresentationKind, ZhongyouExperimentResult]:
    """Compare Raw-GHMM, PCA10-GHMM and LSTM10-GHMM under one split."""

    default = base_config or _default_model_config("lstm", seed)
    output: dict[RepresentationKind, ZhongyouExperimentResult] = {}
    for representation in ("raw", "pca", "lstm"):
        config = replace(default, representation=representation)
        output[representation] = run_zhongyou_experiment(
            market,
            representation=representation,
            seed=seed,
            model_config=config,
            **overrides,
        )
    return output


__all__ = [
    "PAPER_TEST_START",
    "PAPER_TRAIN_END",
    "PAPER_TRAIN_START_2008",
    "PAPER_TRAIN_START_2016",
    "STAR_COMPOSITE_TRAIN_END",
    "ZHONGYOU_PROXY_ASSUMPTIONS",
    "ZhongyouExperimentResult",
    "run_representation_comparison",
    "run_training_window_sensitivity",
    "run_zhongyou_experiment",
]
