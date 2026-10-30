"""Artifact-producing Zhongyou reconstruction and validation workflows."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from .artifacts import write_run_artifacts
from .backtest import cash_rate_to_daily_returns, run_cost_scenarios, run_long_cash_backtest
from .config import load_validation_protocol
from .data_service import load_cash_rate
from .experiments.zhongyou import run_zhongyou_experiment
from .models.zhongyou import (
    AutoencoderConfig,
    GHMMConfig,
    RepresentationKind,
    ZhongyouModelConfig,
    kelly_position_details,
)
from .runner import (
    _add_alpha_labels,
    _asset,
    _baseline_metrics,
    _execution_market,
    _feature_market,
    _market_input_metadata,
    _registered_experiment,
    _runtime_metadata,
    _scenario_metrics,
    _slice_scenario,
)
from .statistics import bootstrap_regression_alpha, hac_alpha, performance_metrics
from .validation import (
    ValidationEvidence,
    classify_validation,
    drawdown_improvement,
    purged_expanding_walk_forward,
    yearly_active_diagnostics,
)

_VALIDATION_PROTOCOL = load_validation_protocol()
ZHONGYOU_KELLY_NEIGHBORS: tuple[int, ...] = tuple(
    int(value)
    for value in _VALIDATION_PROTOCOL["parameter_robustness"][
        "zhongyou_kelly_lookbacks"
    ]
)


def _model_config(
    representation: RepresentationKind,
    *,
    seed: int,
    quick: bool,
    parameters: dict[str, Any] | None = None,
) -> ZhongyouModelConfig:
    parameters = parameters or {}
    return ZhongyouModelConfig(
        representation=representation,
        pca_components=int(parameters.get("latent_size", 10)),
        autoencoder=AutoencoderConfig(
            hidden_size=int(parameters.get("hidden_size", 64)),
            latent_size=int(parameters.get("latent_size", 10)),
            sequence_length=int(parameters.get("sequence_length", 90)),
            seed=seed,
            max_epochs=8 if quick else 100,
            patience=2 if quick else 10,
        ),
        ghmm=GHMMConfig(
            n_states=int(parameters.get("n_states", 5)),
            covariance_type=str(parameters.get("covariance_type", "full")),  # type: ignore[arg-type]
            seed=seed,
            restarts=2 if quick else int(parameters.get("ghmm_restarts", 20)),
            max_iterations=60 if quick else 300,
        ),
    )


def _merge_model_diagnostics(backtest: pd.DataFrame, model_daily: pd.DataFrame) -> pd.DataFrame:
    diagnostic_columns = [
        column
        for column in model_daily.columns
        if column.startswith("state_probability_")
        or column
        in {
            "map_state",
            "prior_same_state_samples",
            "win_probability",
            "win_loss_odds",
            "kelly_position",
            "base_kelly_position",
            "target_position",
        }
    ]
    return backtest.join(model_daily[diagnostic_columns], how="left")


def run_zhongyou_reconstruction(
    asset_id: str,
    *,
    representation: RepresentationKind = "lstm",
    seed: int = 0,
    train_window: str = "2008",
    quick: bool = False,
    microcap_patch: bool = False,
) -> Path:
    asset = _asset(asset_id)
    registered = _registered_experiment(
        asset, "zhongyou", train_window=train_window
    )
    market = _feature_market(asset)
    execution_market = _execution_market(asset)
    cash = load_cash_rate(asset.cash_series)
    runtime = _runtime_metadata()
    input_metadata = _market_input_metadata(asset)
    model_cash_returns = cash_rate_to_daily_returns(
        cash.values,
        market.index,
        lag=cash.lag,
        unit=cash.unit,
        day_count_basis=cash.day_count_basis,
    )
    if train_window not in {"2008", "2016"}:
        raise ValueError("train_window must be 2008 or 2016")
    requested_start = pd.Timestamp(
        registered.parameters.get("train_start", f"{train_window}-01-01")
    )
    train_start = max(requested_start, market.index.min())
    is_star_composite = asset_id == "star_composite"
    train_end = pd.Timestamp(
        "2022-12-31" if is_star_composite else registered.train_end
    )
    test_start = pd.Timestamp(
        "2023-01-01" if is_star_composite else registered.test_start
    )
    test_end = min(market.index.max(), execution_market.index.max())
    config = _model_config(
        representation,
        seed=seed,
        quick=quick,
        parameters=registered.parameters,
    )
    result = run_zhongyou_experiment(
        market,
        representation=representation,
        seed=seed,
        model_config=config,
        train_start=train_start,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end,
        cash_returns=model_cash_returns,
        kelly_lookback=int(
            registered.parameters.get("kelly_lookback_same_state", 15)
        ),
        microcap_patch=microcap_patch,
        bottom_patch=False,
    )
    scenarios = run_cost_scenarios(
        execution_market,
        result.daily["target_position"],
        one_way_cost_bps=asset.one_way_cost_bps,
        **cash.backtest_kwargs,
    )
    scenarios = {
        name: _slice_scenario(frame, test_start, test_end) for name, frame in scenarios.items()
    }
    daily = _merge_model_diagnostics(scenarios["base"], result.daily)
    metrics = performance_metrics(
        daily["net_return"],
        benchmark_returns=daily["equal_exposure_return"],
        risk_free_returns=daily["cash_return"],
        positions=daily["position"],
        costs=daily["cost"],
    )
    metrics.update(result.summary)
    _add_alpha_labels(metrics, daily)
    metrics.update(
        {
            "annual_alpha": metrics.get("alpha"),
            "annual_volatility": metrics.get("annualized_volatility"),
            "turnover": metrics.get("annualized_turnover"),
            "status": "Not independently validated",
        }
    )
    run_id = (
        f"zhongyou__{asset_id}__{representation}__train{train_window}__seed{seed}"
        + ("__microcap_posthoc" if microcap_patch else "")
    )
    metadata: dict[str, Any] = {
        **runtime,
        **input_metadata,
        **result.metadata,
        "asset_id": asset_id,
        "asset_name": asset.name,
        "market": asset.market.value,
        "calendar": asset.calendar,
        "signal_symbol": asset.signal_symbol,
        "execution_symbol": asset.execution_symbol or asset.signal_symbol,
        "proxy_for": asset.proxy_for,
        "replication_level": "conceptual_reconstruction",
        "registered_experiment_yaml": "configs/zhongyou.yml",
        "registered_experiment_spec": registered.model_dump(mode="json"),
        "execution": "close signal, next open, open-to-open P&L",
        "one_way_cost_bps": asset.one_way_cost_bps,
        "quick_mode": quick,
        **cash.metadata,
        "assumptions": [*result.metadata["assumptions"], *cash.assumptions],
    }
    baselines = _baseline_metrics(
        execution_market,
        start=test_start,
        end=test_end,
        one_way_cost_bps=asset.one_way_cost_bps,
        cash=cash,
    )
    return write_run_artifacts(
        run_id,
        daily=daily,
        metrics=metrics,
        metadata=metadata,
        tables={
            "forward_signals": result.daily[
                [
                    column
                    for column in result.daily.columns
                    if column.startswith("state_probability_")
                    or column
                    in {
                        "map_state",
                        "prior_same_state_samples",
                        "win_probability",
                        "win_loss_odds",
                        "kelly_position",
                        "base_kelly_position",
                        "target_position",
                    }
                ]
            ],
            "scenario_metrics": _scenario_metrics(scenarios),
            "baseline_metrics": baselines,
        },
    )


def _walk_forward_positions(
    market: pd.DataFrame,
    *,
    representation: RepresentationKind,
    seeds: Iterable[int],
    quick: bool,
    cash_returns: pd.Series,
    parameters: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    walk_forward = _VALIDATION_PROTOCOL["walk_forward"]
    folds = purged_expanding_walk_forward(
        market.index,
        minimum_history_years=int(walk_forward["minimum_train_years"]),
        validation_years=int(walk_forward["validation_years"]),
        purge_observations=int(walk_forward["purge_trading_days"]),
    )
    if not folds:
        raise ValueError("No eligible 8-year walk-forward folds")
    positions: dict[int, pd.Series] = {}
    robustness_positions: dict[str, pd.Series] = {}
    fold_records: list[dict[str, Any]] = []
    for seed in seeds:
        pieces: list[pd.Series] = []
        lookback_pieces: dict[int, list[pd.Series]] = {
            lookback: [] for lookback in ZHONGYOU_KELLY_NEIGHBORS
        }
        for fold in folds:
            config = _model_config(
                representation,
                seed=seed,
                quick=quick,
                parameters=parameters,
            )
            result = run_zhongyou_experiment(
                market,
                representation=representation,
                seed=seed,
                model_config=config,
                train_start=fold.train.min(),
                train_end=fold.train.max(),
                test_start=fold.test.min(),
                test_end=fold.test.max(),
                train_index=fold.train,
                validation_index=fold.validation,
                cash_returns=cash_returns,
                kelly_lookback=int(parameters.get("kelly_lookback_same_state", 15)),
            )
            target = result.daily["target_position"].reindex(fold.test)
            pieces.append(target)
            for lookback in ZHONGYOU_KELLY_NEIGHBORS:
                alternative = kelly_position_details(
                    result.causal_context["map_state"],
                    result.causal_context["next_day_excess_return"],
                    lookback_same_state=lookback,
                )["kelly_position"].reindex(fold.test)
                lookback_pieces[lookback].append(alternative)
            fold_records.append(
                {
                    "seed": seed,
                    "fold": fold.fold_id,
                    "train_start": fold.train.min(),
                    "train_end": fold.train.max(),
                    "training_history_years": fold.training_history_years,
                    "validation_start": fold.validation.min(),
                    "validation_end": fold.validation.max(),
                    "test_start": fold.test.min(),
                    "test_end": fold.test.max(),
                    "train_observations": result.summary["train_observations"],
                    "validation_observations": result.metadata[
                        "walk_forward_validation_rows"
                    ],
                    "test_state_rows": result.summary["test_state_rows"],
                    "average_target_position": result.summary["average_target_position"],
                }
            )
        positions[seed] = pd.concat(pieces).sort_index()
        for lookback, parts in lookback_pieces.items():
            robustness_positions[
                f"seed{seed}__kelly{lookback}"
            ] = pd.concat(parts).sort_index()
    return (
        pd.DataFrame(positions).sort_index(),
        pd.DataFrame(fold_records),
        pd.DataFrame(robustness_positions).sort_index(),
    )


def validate_zhongyou(
    asset_id: str,
    *,
    representation: RepresentationKind = "lstm",
    seeds: Iterable[int] = range(10),
    n_boot: int = 10_000,
    quick: bool = False,
) -> Path:
    asset = _asset(asset_id)
    registered = _registered_experiment(asset, "zhongyou")
    market = _feature_market(asset)
    execution_market = _execution_market(asset)
    cash = load_cash_rate(asset.cash_series)
    runtime = _runtime_metadata()
    input_metadata = _market_input_metadata(asset)
    model_cash_returns = cash_rate_to_daily_returns(
        cash.values,
        market.index,
        lag=cash.lag,
        unit=cash.unit,
        day_count_basis=cash.day_count_basis,
    )
    seed_list = list(seeds)
    seed_positions, folds, robustness_positions = _walk_forward_positions(
        market,
        representation=representation,
        seeds=seed_list,
        quick=quick,
        cash_returns=model_cash_returns,
        parameters=registered.parameters,
    )
    ensemble = seed_positions.mean(axis=1, skipna=True)
    scenarios = run_cost_scenarios(
        execution_market,
        ensemble,
        one_way_cost_bps=asset.one_way_cost_bps,
        **cash.backtest_kwargs,
    )
    start, end = ensemble.index.min(), ensemble.index.max()
    scenarios = {name: _slice_scenario(frame, start, end) for name, frame in scenarios.items()}
    base = scenarios["base"]
    metrics = performance_metrics(
        base["net_return"],
        benchmark_returns=base["equal_exposure_return"],
        risk_free_returns=base["cash_return"],
        positions=base["position"],
        costs=base["cost"],
    )
    benchmark_metrics = performance_metrics(
        base["equal_exposure_return"], risk_free_returns=base["cash_return"]
    )
    bootstrap = bootstrap_regression_alpha(
        base["net_return"],
        base["equal_exposure_return"],
        risk_free_returns=base["cash_return"],
        n_boot=n_boot,
    )
    seed_records: list[dict[str, Any]] = []
    for seed in seed_list:
        seed_daily = run_long_cash_backtest(
            execution_market,
            seed_positions[seed],
            one_way_cost_bps=asset.one_way_cost_bps,
            **cash.backtest_kwargs,
        )
        seed_daily = _slice_scenario(seed_daily, start, end)
        alpha = hac_alpha(
            seed_daily["net_return"],
            seed_daily["equal_exposure_return"],
            risk_free_returns=seed_daily["cash_return"],
        )["alpha"]
        seed_records.append({"seed": seed, "annual_alpha": alpha})
    seed_table = pd.DataFrame(seed_records)
    robust_seed_fraction = float((seed_table["annual_alpha"] > 0).mean())
    robustness_records: list[dict[str, Any]] = []
    for column in robustness_positions:
        scenario_daily = run_long_cash_backtest(
            execution_market,
            robustness_positions[column],
            one_way_cost_bps=asset.one_way_cost_bps,
            **cash.backtest_kwargs,
        )
        scenario_daily = _slice_scenario(scenario_daily, start, end)
        alpha = hac_alpha(
            scenario_daily["net_return"],
            scenario_daily["equal_exposure_return"],
            risk_free_returns=scenario_daily["cash_return"],
        )["alpha"]
        seed_text, lookback_text = column.split("__kelly")
        robustness_records.append(
            {
                "combination": column,
                "seed": int(seed_text.removeprefix("seed")),
                "kelly_lookback": int(lookback_text),
                "annual_alpha": alpha,
                "positive_alpha": bool(alpha > 0),
            }
        )
    robustness_table = pd.DataFrame(robustness_records)
    robust_fraction = float(robustness_table["positive_alpha"].mean())
    yearly = yearly_active_diagnostics(
        base["net_return"],
        base["equal_exposure_return"],
        base["cash_return"],
    )
    double_alpha = hac_alpha(
        scenarios["double_cost"]["net_return"],
        scenarios["double_cost"]["equal_exposure_return"],
        risk_free_returns=scenarios["double_cost"]["cash_return"],
    )["alpha"]
    delayed_alpha = hac_alpha(
        scenarios["extra_delay"]["net_return"],
        scenarios["extra_delay"]["equal_exposure_return"],
        risk_free_returns=scenarios["extra_delay"]["cash_return"],
    )["alpha"]
    sharpe_improvement = float(metrics["sharpe"] - benchmark_metrics["sharpe"])
    dd_improvement = drawdown_improvement(base["net_return"], base["equal_exposure_return"])
    minimum_training_history = float(folds["training_history_years"].min())
    expected_combinations = len(set(seed_list)) * len(ZHONGYOU_KELLY_NEIGHBORS)
    parameter_neighborhood_complete = bool(
        len(robustness_table) == expected_combinations
        and set(robustness_table["kelly_lookback"])
        == set(ZHONGYOU_KELLY_NEIGHBORS)
    )
    formal_evidence_eligible = bool(
        not quick
        and seed_list == list(_VALIDATION_PROTOCOL["walk_forward"]["seeds"])
        and n_boot == int(_VALIDATION_PROTOCOL["bootstrap"]["replications"])
        and cash.values is not None
        and parameter_neighborhood_complete
        and input_metadata["market_input_hashes_complete"]
        and runtime["code_identity_complete"]
    )
    evidence = ValidationEvidence(
        years_history=minimum_training_history,
        net_alpha=float(metrics["alpha"]),
        alpha_ci_low=bootstrap["alpha_ci_low"],
        fdr_reject=bootstrap["bootstrap_pvalue"] <= 0.10,
        sharpe_improvement=sharpe_improvement,
        max_drawdown_improvement=dd_improvement,
        positive_fold_fraction=yearly["positive_fold_fraction"],
        double_cost_alpha=double_alpha,
        delayed_alpha=delayed_alpha,
        robust_seed_parameter_fraction=robust_fraction,
        best_year_alpha_share=yearly["best_year_alpha_share"],
    )
    decision = classify_validation(evidence, formal_eligible=formal_evidence_eligible)
    _add_alpha_labels(metrics, base)
    metrics.update(
        {
            "annual_alpha": metrics.get("alpha"),
            "annual_volatility": metrics.get("annualized_volatility"),
            "turnover": metrics.get("annualized_turnover"),
            "alpha_bootstrap_ci_low": bootstrap["alpha_ci_low"],
            "alpha_bootstrap_ci_high": bootstrap["alpha_ci_high"],
            "bootstrap_pvalue": bootstrap["bootstrap_pvalue"],
            "sharpe_improvement": sharpe_improvement,
            "max_drawdown_improvement": dd_improvement,
            "positive_fold_fraction": yearly["positive_fold_fraction"],
            "robust_seed_fraction": robust_seed_fraction,
            "robust_seed_parameter_fraction": robust_fraction,
            "minimum_training_history_years": minimum_training_history,
            "oos_years": len(base) / 252.0,
            "best_year_alpha_share": yearly["best_year_alpha_share"],
            "positive_active_year_fraction": yearly[
                "positive_active_year_fraction"
            ],
            "best_year_hac_alpha_share": yearly["best_year_hac_alpha_share"],
            "double_cost_alpha": double_alpha,
            "delayed_alpha": delayed_alpha,
            "status": decision.status.value,
        }
    )
    daily = base.join(seed_positions.add_prefix("seed_position_"), how="left")
    forward_signals = seed_positions.add_prefix("seed_position_").copy()
    forward_signals["target_position"] = ensemble
    run_id = f"validation__zhongyou__{representation}__{asset_id}"
    metadata = {
        **runtime,
        **input_metadata,
        "study": "zhongyou_independent_validation",
        "asset_id": asset_id,
        "market": asset.market.value,
        "calendar": asset.calendar,
        "representation": representation,
        "registered_experiment_yaml": "configs/zhongyou.yml",
        "registered_experiment_spec": registered.model_dump(mode="json"),
        "replication_level": "independent_validation",
        "seeds": seed_list,
        "bootstrap_replications": n_boot,
        "quick_mode": quick,
        "formal_evidence_eligible": formal_evidence_eligible,
        "formal_eligibility_requirements": {
            "not_quick": not quick,
            "exact_unique_seeds_0_to_9": seed_list
            == list(_VALIDATION_PROTOCOL["walk_forward"]["seeds"]),
            "bootstrap_10000": n_boot
            == int(_VALIDATION_PROTOCOL["bootstrap"]["replications"]),
            "cash_snapshot_available": cash.values is not None,
            "parameter_neighborhood_complete": parameter_neighborhood_complete,
            "market_input_hashes_complete": input_metadata[
                "market_input_hashes_complete"
            ],
            "code_identity_complete": runtime["code_identity_complete"],
        },
        "robustness_parameter_grid": {
            "kelly_lookback": list(ZHONGYOU_KELLY_NEIGHBORS),
            "combinations": len(robustness_table),
        },
        "validation_evidence": asdict(evidence),
        "validation_checks": decision.checks,
        "validation_summary": decision.summary,
        **cash.metadata,
        "assumptions": [
            "年度扩展walk-forward；最少8年历史、2年保留期、90交易日purge。",
            "GHMM每个外层fold只使用此前观测拟合，测试状态只用前向filter。",
            "中邮事后仓位补丁在独立验证中强制关闭。",
            *cash.assumptions,
        ],
    }
    return write_run_artifacts(
        run_id,
        daily=daily,
        metrics=metrics,
        metadata=metadata,
        tables={
            "forward_signals": forward_signals,
            "folds": folds,
            "seed_metrics": seed_table,
            "seed_parameter_robustness": robustness_table,
            "scenario_metrics": _scenario_metrics(scenarios),
        },
    )
