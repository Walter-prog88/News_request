"""High-level, artifact-producing research workflows."""

from __future__ import annotations

import hashlib
import platform
import subprocess
from collections.abc import Iterable
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .artifacts import write_run_artifacts
from .backtest import run_cost_scenarios, run_long_cash_backtest, slice_backtest_result
from .baselines import all_baselines
from .config import (
    load_assets,
    load_experiments,
    load_macro_config,
    load_validation_protocol,
)
from .data_service import (
    CashRateData,
    MacroSeriesData,
    latest_snapshot,
    load_cash_rate,
    load_macro_series,
    load_market,
)
from .features.caitong import (
    CAITONG_FEATURE_GROUPS,
    CAITONG_FEATURE_SETS,
    CaitongFeatureBundle,
    PointInTimeSource,
    assemble_caitong_features,
)
from .models.caitong import CaitongModelConfig, ModelKind
from .paths import ROOT
from .statistics import bootstrap_regression_alpha, hac_alpha, performance_metrics
from .types import AssetSpec, ExperimentSpec, Market
from .validation import (
    ValidationEvidence,
    classify_validation,
    drawdown_improvement,
    purged_expanding_walk_forward,
    yearly_active_diagnostics,
)

_VALIDATION_PROTOCOL = load_validation_protocol()
CAITONG_THRESHOLD_NEIGHBORS: tuple[float, ...] = tuple(
    float(value)
    for value in _VALIDATION_PROTOCOL["parameter_robustness"][
        "caitong_signal_thresholds"
    ]
)


def _asset(asset_id: str) -> AssetSpec:
    assets = load_assets()
    if asset_id not in assets:
        raise ValueError(f"Unknown asset_id={asset_id!r}")
    return assets[asset_id]


def _registered_experiment(
    asset: AssetSpec,
    study: str,
    *,
    train_window: str | None = None,
) -> ExperimentSpec:
    """Resolve the YAML preregistration that governs an asset run."""

    path = "caitong.yml" if study == "caitong" else "zhongyou.yml"
    candidates = [
        spec
        for spec in load_experiments(path)
        if spec.study == study and asset.asset_id in spec.asset_ids
    ]
    if not candidates:
        raise ValueError(f"No registered {study} experiment for {asset.asset_id}")
    if asset.market == Market.UNITED_STATES:
        us = [spec for spec in candidates if spec.experiment_id.endswith("_us_frozen")]
        if us:
            return us[0]
    if study == "zhongyou" and train_window == "2016":
        conflict = [spec for spec in candidates if "2016" in spec.experiment_id]
        if conflict:
            return conflict[0]
    primary = [spec for spec in candidates if "2016" not in spec.experiment_id]
    return primary[0] if primary else candidates[0]


def _feature_market(asset: AssetSpec) -> pd.DataFrame:
    signal_market = load_market(asset.asset_id, role="signal")
    if not asset.execution_symbol:
        return signal_market
    # Price features and labels remain tied to the named index.  A tradable ETF
    # supplies only volume because broad price indices do not publish a useful
    # consolidated volume series.
    execution_market = load_market(asset.asset_id, role="execution")
    combined = signal_market.copy()
    combined["volume"] = execution_market["volume"].reindex(combined.index)
    return combined


def _execution_market(asset: AssetSpec) -> pd.DataFrame:
    role = "execution" if asset.execution_symbol else "signal"
    return load_market(asset.asset_id, role=role)


def _caitong_features(
    asset: AssetSpec,
    market: pd.DataFrame,
    feature_set: str,
) -> tuple[CaitongFeatureBundle, dict[str, object]]:
    """Load market-appropriate PIT sources and expose empty factor families."""

    _feature_set_suffix(feature_set)
    requested_groups = {
        "technical": {"technical"},
        "technical_plus_valuation_fund": {"technical", "valuation_fund"},
        "technical_plus_market_global": {
            "technical",
            "valuation_fund",
            "rates_commodities_overseas",
        },
        "full": set(CAITONG_FEATURE_GROUPS),
    }[feature_set]
    source_groups = {
        "rates_liquidity": "rates_commodities_overseas",
        "risk_credit": "rates_commodities_overseas",
        "dollar_commodity": "rates_commodities_overseas",
        "high_frequency_economy": "rates_commodities_overseas",
        "low_frequency_macro": "low_frequency_macro",
    }
    macro_config = (
        load_macro_config() if asset.market == Market.UNITED_STATES else None
    )
    macro_data: list[MacroSeriesData] = []
    for spec in macro_config.series if macro_config is not None else []:
        feature_group = source_groups[spec.group]
        if feature_group in requested_groups:
            macro_data.append(load_macro_series(spec))
        else:
            macro_data.append(
                MacroSeriesData(
                    spec=spec,
                    frame=None,
                    status="not_requested",
                    reason=f"excluded_by_feature_set:{feature_set}",
                )
            )
    pit_sources: list[PointInTimeSource] = []
    configured_by_group: dict[str, list[str]] = {
        group: [] for group in CAITONG_FEATURE_GROUPS
    }
    loaded_by_group: dict[str, list[str]] = {
        group: [] for group in CAITONG_FEATURE_GROUPS
    }
    for source in macro_data:
        feature_group = source_groups[source.spec.group]
        configured_by_group[feature_group].append(source.spec.series_id)
        if source.frame is None:
            continue
        loaded_by_group[feature_group].append(source.spec.series_id)
        pit_sources.append(
            PointInTimeSource(
                name=source.spec.series_id,
                group=feature_group,  # type: ignore[arg-type]
                data=source.frame,
                release_timestamp_col="available_at",
                value_columns=("value",),
                prefix=source.spec.series_id,
            )
        )
    features = assemble_caitong_features(market, pit_sources=pit_sources)
    group_status: dict[str, dict[str, object]] = {}
    missing_requested_series: list[str] = []
    for group in CAITONG_FEATURE_GROUPS:
        columns = list(features.groups[group])
        missing_series = sorted(
            set(configured_by_group[group]) - set(loaded_by_group[group])
        )
        if group not in requested_groups:
            status = "not_requested"
        elif columns and not missing_series:
            status = "available"
        elif columns:
            status = "partial_configured_sources_unavailable"
        elif group == "valuation_fund":
            status = "empty_no_free_point_in_time_source"
        elif asset.market != Market.UNITED_STATES:
            status = "empty_no_market_macro_config"
        elif configured_by_group[group]:
            status = "empty_all_configured_sources_unavailable"
        else:
            status = "empty_no_configured_sources"
        if group in requested_groups:
            missing_requested_series.extend(missing_series)
        group_status[group] = {
            "status": status,
            "column_count": len(columns),
            "columns": columns,
            "configured_series": configured_by_group[group],
            "loaded_series": loaded_by_group[group],
            "missing_series": missing_series,
        }
    incomplete_groups = [
        group
        for group in requested_groups
        if group_status[group]["status"] != "available"
    ]
    audit = {
        "macro_config": "macro_us.yml" if asset.market == Market.UNITED_STATES else None,
        "macro_availability_rule": (
            macro_config.availability_rule if macro_config is not None else None
        ),
        "macro_series_audit": [source.metadata for source in macro_data],
        "feature_group_availability": group_status,
        "requested_feature_groups": sorted(requested_groups),
        "effective_nonempty_feature_groups": [
            group for group in CAITONG_FEATURE_GROUPS if features.groups[group]
        ],
        "incomplete_requested_feature_groups": sorted(incomplete_groups),
        "missing_requested_macro_series": sorted(missing_requested_series),
        "feature_set_degraded": bool(incomplete_groups),
    }
    return features, audit


def _feature_set_suffix(feature_set: str) -> str:
    if feature_set not in CAITONG_FEATURE_SETS:
        raise ValueError(
            f"feature_set must be one of {list(CAITONG_FEATURE_SETS)}; got {feature_set!r}"
        )
    # Full retains the historical id for report compatibility.  Ablations get
    # isolated ids and therefore cannot overwrite a full-specification run.
    return "" if feature_set == "full" else f"__fs{feature_set}"


def _source_tree_sha256(root: Path = ROOT) -> tuple[str, bool]:
    """Hash only authored research inputs, never interpreter cache files."""

    tracked_inputs: list[Path] = []
    excluded_directories = {"__pycache__", ".pytest_cache", ".ruff_cache"}
    for relative in ("src", "configs"):
        directory = root / relative
        if not directory.exists():
            continue
        tracked_inputs.extend(
            path
            for path in directory.rglob("*")
            if path.is_file()
            and not excluded_directories.intersection(path.relative_to(root).parts)
            and path.suffix not in {".pyc", ".pyo"}
        )
    lock_path = root / "uv.lock"
    tracked_inputs.extend(
        path for path in (root / "pyproject.toml", lock_path) if path.exists()
    )
    digest = hashlib.sha256()
    for path in sorted(tracked_inputs):
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest(), bool(tracked_inputs and lock_path.exists())


def _runtime_metadata() -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit = "uncommitted"
    source_tree_sha, code_identity_complete = _source_tree_sha256(ROOT)
    try:
        git_status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        git_dirty: bool | None = bool(git_status.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        git_dirty = None
    lock_path = ROOT / "uv.lock"
    lock_sha = hashlib.sha256(lock_path.read_bytes()).hexdigest() if lock_path.exists() else None
    return {
        "created_at": datetime.now().astimezone().isoformat(),
        "git_commit": commit,
        "git_dirty": git_dirty,
        "source_tree_sha256": source_tree_sha,
        "uv_lock_sha256": lock_sha,
        "code_identity_complete": bool(code_identity_complete and lock_sha),
        "python": platform.python_version(),
        "platform": platform.platform(),
    }


def _market_input_metadata(asset: AssetSpec) -> dict[str, Any]:
    """Record the exact immutable market snapshots consumed by a run."""

    signal = latest_snapshot(asset.asset_id, role="signal")
    execution = latest_snapshot(
        asset.asset_id,
        role="execution" if asset.execution_symbol else "signal",
    )

    def fields(prefix: str, snapshot: Any) -> dict[str, Any]:
        return {
            f"{prefix}_snapshot_id": snapshot.dataset_id,
            f"{prefix}_snapshot_sha256": snapshot.sha256,
            f"{prefix}_snapshot_as_of": snapshot.as_of.isoformat(),
            f"{prefix}_snapshot_retrieved_at": snapshot.retrieved_at.isoformat(),
            f"{prefix}_snapshot_provider": snapshot.provider,
            f"{prefix}_snapshot_vintage": snapshot.vintage,
            f"{prefix}_snapshot_min_date": (
                snapshot.min_date.isoformat() if snapshot.min_date else None
            ),
            f"{prefix}_snapshot_max_date": (
                snapshot.max_date.isoformat() if snapshot.max_date else None
            ),
            f"{prefix}_snapshot_warnings": list(snapshot.warnings),
        }

    return {
        **fields("signal_market", signal),
        **fields("execution_market", execution),
        "market_input_hashes_complete": bool(signal.sha256 and execution.sha256),
    }


def _add_alpha_labels(metrics: dict[str, Any], daily: pd.DataFrame) -> None:
    """Keep equal-exposure alpha and market Jensen alpha explicitly separate."""

    metrics["alpha_vs_equal_exposure"] = metrics.get("alpha")
    metrics["active_return_vs_equal_exposure"] = metrics.get(
        "annualized_active_return"
    )
    jensen = hac_alpha(
        daily["net_return"],
        daily["benchmark_return"],
        risk_free_returns=daily["cash_return"],
    )
    metrics["jensen_alpha_vs_market"] = jensen["alpha"]
    metrics["jensen_beta_vs_market"] = jensen["beta"]


def _slice_scenario(
    frame: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp | None,
) -> pd.DataFrame:
    return slice_backtest_result(frame, start=start, end=end)


def _scenario_metrics(scenarios: dict[str, pd.DataFrame]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for name, daily in scenarios.items():
        metrics = performance_metrics(
            daily["net_return"],
            benchmark_returns=daily["equal_exposure_return"],
            risk_free_returns=daily["cash_return"],
            positions=daily["position"],
            costs=daily["cost"],
        )
        records.append({"scenario": name, **metrics})
    return pd.DataFrame(records).set_index("scenario")


def _baseline_metrics(
    market: pd.DataFrame,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp | None,
    one_way_cost_bps: float,
    cash: CashRateData | None = None,
) -> pd.DataFrame:
    cash = cash or CashRateData(series_id=None, values=None)
    records: list[dict[str, Any]] = []
    for name, position in all_baselines(market["close"]).items():
        daily = run_long_cash_backtest(
            market,
            position,
            one_way_cost_bps=one_way_cost_bps,
            **cash.backtest_kwargs,
        )
        daily = _slice_scenario(daily, start, end)
        metrics = performance_metrics(
            daily["net_return"],
            benchmark_returns=daily["equal_exposure_return"],
            risk_free_returns=daily["cash_return"],
            positions=daily["position"],
            costs=daily["cost"],
        )
        records.append({"baseline": name, **metrics})
    return pd.DataFrame(records).set_index("baseline")


def run_caitong_reconstruction(
    asset_id: str,
    *,
    model_kind: ModelKind = "mlp",
    seed: int = 0,
    horizon: int = 5,
    quick: bool = False,
    feature_set: str = "full",
) -> Path:
    from .experiments.caitong import run_caitong_experiment

    asset = _asset(asset_id)
    registered = _registered_experiment(asset, "caitong")
    feature_market = _feature_market(asset)
    execution_market = _execution_market(asset)
    cash = load_cash_rate(asset.cash_series)
    feature_suffix = _feature_set_suffix(feature_set)
    features, feature_audit = _caitong_features(asset, feature_market, feature_set)
    train_end = pd.Timestamp(registered.train_end)
    test_start = pd.Timestamp(registered.test_start)
    requested_end = (
        pd.Timestamp(registered.test_end) if registered.test_end is not None else None
    )
    test_end = (
        min(requested_end, execution_market.index.max())
        if requested_end is not None
        else execution_market.index.max()
    )
    parameters = registered.parameters
    config = CaitongModelConfig(
        kind=model_kind,
        seed=seed,
        hidden_sizes=tuple(parameters.get("hidden_sizes", [128, 64])),  # type: ignore[arg-type]
        learning_rate=float(parameters.get("learning_rate", 1e-3)),
        weight_decay=float(parameters.get("l2", 1e-4)),
        max_epochs=(
            12
            if quick and model_kind != "elastic_net_logistic"
            else int(parameters.get("max_epochs", 100))
        ),
        patience=(
            3
            if quick and model_kind != "elastic_net_logistic"
            else int(parameters.get("patience", 10))
        ),
    )
    result = run_caitong_experiment(
        features,
        feature_market["close"],
        feature_set=feature_set,
        model_kind=model_kind,
        horizon=horizon,
        seed=seed,
        model_config=config,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end,
    )
    scenarios = run_cost_scenarios(
        execution_market,
        result.daily["signal"],
        one_way_cost_bps=asset.one_way_cost_bps,
        **cash.backtest_kwargs,
    )
    scenarios = {
        name: _slice_scenario(frame, test_start, test_end) for name, frame in scenarios.items()
    }
    daily = scenarios["base"].join(
        result.daily[["probability", "probability_ma5", "signal", "regime"]], how="left"
    )
    metrics = performance_metrics(
        daily["net_return"],
        benchmark_returns=daily["equal_exposure_return"],
        risk_free_returns=daily["cash_return"],
        positions=daily["position"],
        costs=daily["cost"],
    )
    metrics.update(result.summary)
    _add_alpha_labels(metrics, daily)
    metrics["annual_alpha"] = metrics.get("alpha")
    metrics["annual_volatility"] = metrics.get("annualized_volatility")
    metrics["turnover"] = metrics.get("annualized_turnover")
    metrics["status"] = "Not independently validated"
    scenario_table = _scenario_metrics(scenarios)
    baselines = _baseline_metrics(
        execution_market,
        start=test_start,
        end=test_end,
        one_way_cost_bps=asset.one_way_cost_bps,
        cash=cash,
    )
    run_id = (
        f"caitong__{asset_id}__{model_kind}__h{horizon}__seed{seed}{feature_suffix}"
    )
    runtime = _runtime_metadata()
    input_metadata = _market_input_metadata(asset)
    metadata = {
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
        "execution": "close signal, next open, open-to-open P&L",
        "one_way_cost_bps": asset.one_way_cost_bps,
        "quick_mode": quick,
        "registered_test_end": (
            requested_end.date().isoformat() if requested_end is not None else None
        ),
        "effective_test_end": test_end.date().isoformat(),
        "requested_feature_set": feature_set,
        "registered_experiment_yaml": "configs/caitong.yml",
        "registered_experiment_spec": registered.model_dump(mode="json"),
        **feature_audit,
        **cash.metadata,
        "assumptions": [*result.metadata["assumptions"], *cash.assumptions],
    }
    return write_run_artifacts(
        run_id,
        daily=daily,
        metrics=metrics,
        metadata=metadata,
        tables={
            "forward_signals": result.daily[
                ["probability", "probability_ma5", "signal", "regime"]
            ],
            "intervals": result.intervals,
            "probability_calibration": result.calibration,
            "scenario_metrics": scenario_table,
            "baseline_metrics": baselines,
        },
    )


def _caitong_walk_forward_positions(
    market: pd.DataFrame,
    *,
    features: CaitongFeatureBundle,
    feature_set: str,
    seeds: Iterable[int],
    quick: bool,
    model_parameters: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    from .experiments.caitong import run_caitong_experiment

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
    probabilities: dict[int, pd.Series] = {}
    robustness_positions: dict[str, pd.Series] = {}
    fold_records: list[dict[str, Any]] = []
    for seed in seeds:
        pieces: list[pd.Series] = []
        probability_pieces: list[pd.Series] = []
        threshold_pieces: dict[float, list[pd.Series]] = {
            threshold: [] for threshold in CAITONG_THRESHOLD_NEIGHBORS
        }
        for fold in folds:
            config = CaitongModelConfig(
                kind="mlp",
                seed=seed,
                hidden_sizes=tuple(
                    model_parameters.get("hidden_sizes", [128, 64])
                ),  # type: ignore[arg-type]
                learning_rate=float(model_parameters.get("learning_rate", 1e-3)),
                weight_decay=float(model_parameters.get("l2", 1e-4)),
                max_epochs=(
                    8 if quick else int(model_parameters.get("max_epochs", 100))
                ),
                patience=(
                    2 if quick else int(model_parameters.get("patience", 10))
                ),
            )
            result = run_caitong_experiment(
                features,
                market["close"],
                feature_set=feature_set,
                model_kind="mlp",
                seed=seed,
                model_config=config,
                train_end=fold.train.max(),
                test_start=fold.test.min(),
                test_end=fold.test.max(),
                train_index=fold.train,
                validation_index=fold.validation,
            )
            probability = result.daily["probability_ma5"].reindex(fold.test)
            probability_pieces.append(
                result.daily["probability"].reindex(fold.test)
            )
            signal = pd.Series(
                np.where(
                    probability.notna(),
                    (probability > 0.50).astype(float),
                    np.nan,
                ),
                index=probability.index,
                name="signal",
            )
            pieces.append(signal)
            for threshold in CAITONG_THRESHOLD_NEIGHBORS:
                threshold_pieces[threshold].append(
                    pd.Series(
                        np.where(
                            probability.notna(),
                            (probability > threshold).astype(float),
                            np.nan,
                        ),
                        index=probability.index,
                    )
                )
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
                    "train_rows": result.summary["train_rows"],
                    "validation_rows": result.metadata[
                        "walk_forward_validation_rows"
                    ],
                    "interval_win_rate": result.summary["interval_win_rate"],
                }
            )
        positions[seed] = pd.concat(pieces).sort_index()
        probabilities[seed] = pd.concat(probability_pieces).sort_index()
        for threshold, parts in threshold_pieces.items():
            robustness_positions[
                f"seed{seed}__threshold{threshold:.2f}"
            ] = pd.concat(parts).sort_index()
    frame = pd.DataFrame(positions).sort_index()
    probability_frame = pd.DataFrame(probabilities).sort_index()
    robustness = pd.DataFrame(robustness_positions).sort_index()
    return frame, pd.DataFrame(fold_records), robustness, probability_frame


def validate_caitong(
    asset_id: str,
    *,
    seeds: Iterable[int] = range(10),
    n_boot: int = 10_000,
    quick: bool = False,
    feature_set: str = "full",
) -> Path:
    asset = _asset(asset_id)
    registered = _registered_experiment(asset, "caitong")
    market = _feature_market(asset)
    execution_market = _execution_market(asset)
    cash = load_cash_rate(asset.cash_series)
    runtime = _runtime_metadata()
    input_metadata = _market_input_metadata(asset)
    feature_suffix = _feature_set_suffix(feature_set)
    features, feature_audit = _caitong_features(asset, market, feature_set)
    seed_list = list(seeds)
    seed_positions, folds, robustness_positions, seed_probabilities = (
        _caitong_walk_forward_positions(
        market,
        features=features,
        feature_set=feature_set,
        seeds=seed_list,
        quick=quick,
        model_parameters=registered.parameters,
        )
    )
    ensemble = seed_positions.mean(axis=1, skipna=True)
    ensemble_probability = seed_probabilities.mean(axis=1, skipna=True)
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
        seed_text, threshold_text = column.split("__threshold")
        robustness_records.append(
            {
                "combination": column,
                "seed": int(seed_text.removeprefix("seed")),
                "signal_threshold": float(threshold_text),
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
    dd_improvement = drawdown_improvement(
        base["net_return"], base["equal_exposure_return"]
    )
    minimum_training_history = float(folds["training_history_years"].min())
    expected_combinations = len(set(seed_list)) * len(CAITONG_THRESHOLD_NEIGHBORS)
    parameter_neighborhood_complete = bool(
        len(robustness_table) == expected_combinations
        and set(robustness_table["signal_threshold"])
        == set(CAITONG_THRESHOLD_NEIGHBORS)
    )
    formal_evidence_eligible = bool(
        not quick
        and seed_list == list(_VALIDATION_PROTOCOL["walk_forward"]["seeds"])
        and n_boot == int(_VALIDATION_PROTOCOL["bootstrap"]["replications"])
        and cash.values is not None
        and not feature_audit["feature_set_degraded"]
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
    from .experiments.caitong import probability_calibration_table

    forward_5d = market["close"].shift(-5).div(market["close"]).sub(1.0)
    target_5d = forward_5d.gt(0).astype(float).where(forward_5d.notna())
    calibration_input = pd.DataFrame(
        {
            "probability": ensemble_probability,
            "target_up_5d": target_5d.reindex(ensemble_probability.index),
        }
    )
    calibration_table = probability_calibration_table(calibration_input)
    calibrated = calibration_input.dropna()
    brier = (
        float(
            ((calibrated["probability"] - calibrated["target_up_5d"]) ** 2).mean()
        )
        if len(calibrated)
        else float("nan")
    )
    calibration_ece = (
        float(
            (
                calibration_table["absolute_gap"]
                * calibration_table["observations"]
            ).sum()
            / calibration_table["observations"].sum()
        )
        if len(calibration_table)
        else float("nan")
    )
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
            "probability_brier_score": brier,
            "probability_calibration_ece": calibration_ece,
            "probability_calibration_rows": len(calibrated),
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
    daily["ensemble_probability"] = ensemble_probability.reindex(daily.index)
    daily["target_up_5d"] = target_5d.reindex(daily.index)
    forward_signals = seed_positions.add_prefix("seed_position_").copy()
    forward_signals["ensemble_probability"] = ensemble_probability
    forward_signals["target_position"] = ensemble
    run_id = f"validation__caitong__{asset_id}{feature_suffix}"
    metadata = {
        **runtime,
        **input_metadata,
        "study": "caitong_independent_validation",
        "asset_id": asset_id,
        "market": asset.market.value,
        "calendar": asset.calendar,
        "replication_level": "independent_validation",
        "seeds": seed_list,
        "bootstrap_replications": n_boot,
        "quick_mode": quick,
        "requested_feature_set": feature_set,
        "registered_experiment_yaml": "configs/caitong.yml",
        "registered_experiment_spec": registered.model_dump(mode="json"),
        "formal_evidence_eligible": formal_evidence_eligible,
        "formal_eligibility_requirements": {
            "not_quick": not quick,
            "exact_unique_seeds_0_to_9": seed_list
            == list(_VALIDATION_PROTOCOL["walk_forward"]["seeds"]),
            "bootstrap_10000": n_boot
            == int(_VALIDATION_PROTOCOL["bootstrap"]["replications"]),
            "cash_snapshot_available": cash.values is not None,
            "feature_set_not_degraded": not feature_audit["feature_set_degraded"],
            "parameter_neighborhood_complete": parameter_neighborhood_complete,
            "market_input_hashes_complete": input_metadata[
                "market_input_hashes_complete"
            ],
            "code_identity_complete": runtime["code_identity_complete"],
        },
        "robustness_parameter_grid": {
            "signal_threshold": list(CAITONG_THRESHOLD_NEIGHBORS),
            "combinations": len(robustness_table),
        },
        "validation_evidence": asdict(evidence),
        "validation_checks": decision.checks,
        "validation_summary": decision.summary,
        **feature_audit,
        **cash.metadata,
        "assumptions": [
            "年度扩展walk-forward；最少8年历史、2年内部验证、90交易日purge。",
            "该单资产运行先使用10%局部显著性；跨资产报告会另做BH-FDR。",
            "参数在外层测试前冻结，外层结果不参与调参。",
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
            "probability_calibration": calibration_table,
            "scenario_metrics": _scenario_metrics(scenarios),
        },
    )


def generate_demo(*, seed: int = 7) -> Path:
    """Create a deterministic, clearly synthetic end-to-end artifact."""

    from .experiments.caitong import run_caitong_experiment

    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2010-01-04", "2026-06-30", name="date")
    state = np.zeros(len(dates), dtype=int)
    for index in range(1, len(state)):
        state[index] = state[index - 1] if rng.random() < 0.97 else 1 - state[index - 1]
    drift = np.where(state == 1, 0.00065, -0.00025)
    returns = drift + rng.normal(0.0, np.where(state == 1, 0.009, 0.014), len(dates))
    close = pd.Series(100.0 * np.exp(np.cumsum(returns)), index=dates)
    overnight = rng.normal(0, 0.002, len(dates))
    open_price = close.shift(1).fillna(close.iloc[0]) * np.exp(overnight)
    high = np.maximum(open_price, close) * (1 + rng.uniform(0, 0.01, len(dates)))
    low = np.minimum(open_price, close) * (1 - rng.uniform(0, 0.01, len(dates)))
    volume = pd.Series(rng.lognormal(15, 0.35, len(dates)), index=dates)
    market = pd.DataFrame(
        {
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "adj_close": close,
            "volume": volume,
        },
        index=dates,
    )
    features = assemble_caitong_features(market)
    config = CaitongModelConfig(kind="mlp", seed=seed, max_epochs=8, patience=2)
    experiment = run_caitong_experiment(
        features,
        close,
        model_kind="mlp",
        seed=seed,
        model_config=config,
        train_end="2022-12-30",
        test_start="2023-01-02",
        test_end=dates.max(),
    )
    scenarios = run_cost_scenarios(
        market,
        experiment.daily["signal"],
        one_way_cost_bps=2.0,
    )
    scenarios = {
        name: _slice_scenario(frame, pd.Timestamp("2023-01-02"), dates.max())
        for name, frame in scenarios.items()
    }
    daily = scenarios["base"].join(
        experiment.daily[["probability", "probability_ma5", "signal"]], how="left"
    )
    metrics = performance_metrics(
        daily["net_return"],
        benchmark_returns=daily["equal_exposure_return"],
        positions=daily["position"],
        costs=daily["cost"],
    )
    metrics.update(experiment.summary)
    metrics.update(
        {
            "annual_alpha": metrics.get("alpha"),
            "annual_volatility": metrics.get("annualized_volatility"),
            "turnover": metrics.get("annualized_turnover"),
            "status": "Not independently validated",
        }
    )
    return write_run_artifacts(
        "demo__synthetic__caitong",
        daily=daily,
        metrics=metrics,
        metadata={
            **_runtime_metadata(),
            **experiment.metadata,
            "asset_id": "synthetic_regime_market",
            "study": "demo",
            "replication_level": "synthetic_smoke_test",
            "assumptions": [
                "该数据完全由固定随机种子生成，只用于检查代码链路，不是有效性证据。",
                *experiment.metadata["assumptions"],
            ],
        },
        tables={
            "forward_signals": experiment.daily[
                ["probability", "probability_ma5", "signal"]
            ],
            "intervals": experiment.intervals,
            "scenario_metrics": _scenario_metrics(scenarios),
        },
    )
