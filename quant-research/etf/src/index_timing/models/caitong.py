"""Proxy classifiers for the Caitong timing reconstruction.

The report describes neural-network tuning but not a reproducible architecture.
The registered primary proxy is therefore an explicit 128/64 MLP.  Elastic-net
logistic regression and one-layer GRU/LSTM models are kept as named alternatives,
not presented as the report author's original network.
"""

from __future__ import annotations

import copy
import inspect
import random
from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ModelKind = Literal["mlp", "elastic_net_logistic", "gru", "lstm"]


@dataclass(frozen=True, slots=True)
class CaitongModelConfig:
    """Pre-registered proxy model parameters."""

    kind: ModelKind = "mlp"
    seed: int = 0
    hidden_sizes: tuple[int, int] = (128, 64)
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 10
    max_epochs: int = 100
    batch_size: int = 64
    validation_fraction: float = 0.2
    sequence_length: int = 20
    recurrent_hidden_size: int = 64
    elasticnet_l1_ratio: float = 0.5
    logistic_c: float = 1.0

    def __post_init__(self) -> None:
        if self.hidden_sizes != (128, 64) and self.kind == "mlp":
            raise ValueError("The registered Caitong MLP architecture must be (128, 64)")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("learning_rate must be positive and weight_decay non-negative")
        if self.patience < 1 or self.max_epochs < 1 or self.batch_size < 1:
            raise ValueError("patience, max_epochs and batch_size must be positive")
        if not 0.05 <= self.validation_fraction <= 0.5:
            raise ValueError("validation_fraction must be between 0.05 and 0.5")
        if self.sequence_length < 2:
            raise ValueError("sequence_length must be >= 2")
        if not 0.0 <= self.elasticnet_l1_ratio <= 1.0:
            raise ValueError("elasticnet_l1_ratio must be in [0, 1]")


class _MLP(nn.Module):
    def __init__(self, input_size: int, hidden_sizes: tuple[int, int]) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], 1),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values).squeeze(-1)


class _RecurrentClassifier(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, kind: Literal["gru", "lstm"]):
        super().__init__()
        recurrent = nn.GRU if kind == "gru" else nn.LSTM
        self.recurrent = recurrent(input_size, hidden_size, num_layers=1, batch_first=True)
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        recurrent_output, _ = self.recurrent(values)
        return self.output(recurrent_output[:, -1, :]).squeeze(-1)


def _set_deterministic(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # pragma: no cover - CI uses CPU
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _chronological_split(n_rows: int, fraction: float) -> int:
    if n_rows < 10:
        raise ValueError("At least 10 labelled observations are required")
    validation_rows = max(1, int(np.ceil(n_rows * fraction)))
    split = n_rows - validation_rows
    if split < 2:
        raise ValueError("Not enough observations before validation split")
    return split


def _trailing_sequences(values: np.ndarray, length: int) -> tuple[np.ndarray, np.ndarray]:
    """Return trailing sequences and the corresponding end-row positions."""

    if len(values) < length:
        return np.empty((0, length, values.shape[1]), dtype=np.float32), np.array([], dtype=int)
    sequences = np.stack(
        [values[end - length + 1 : end + 1] for end in range(length - 1, len(values))]
    )
    endpoints = np.arange(length - 1, len(values), dtype=int)
    return sequences.astype(np.float32), endpoints


class CaitongClassifier:
    """A common pandas-facing wrapper for all registered Caitong proxies."""

    def __init__(self, config: CaitongModelConfig | None = None) -> None:
        self.config = config or CaitongModelConfig()
        self.feature_names_: tuple[str, ...] | None = None
        self.imputer_: SimpleImputer | None = None
        self.scaler_: StandardScaler | None = None
        self.estimator_: LogisticRegression | nn.Module | None = None
        self.scaler_fit_end_: pd.Timestamp | None = None
        self.training_end_: pd.Timestamp | None = None
        self.training_rows_: int = 0
        self.validation_start_: pd.Timestamp | None = None
        self.validation_end_: pd.Timestamp | None = None
        self.validation_rows_: int = 0
        self.explicit_validation_: bool = False
        self.best_epoch_: int | None = None
        self.best_validation_loss_: float | None = None

    def _validate_training_data(
        self, X: pd.DataFrame, y: pd.Series
    ) -> tuple[pd.DataFrame, pd.Series]:
        if not isinstance(X.index, pd.DatetimeIndex):
            raise TypeError("X must use a DatetimeIndex")
        if not X.index.is_monotonic_increasing or X.index.has_duplicates:
            raise ValueError("X index must be sorted and unique")
        if X.columns.has_duplicates:
            raise ValueError("X feature columns must be unique")
        aligned_y = pd.to_numeric(y.reindex(X.index), errors="coerce")
        labelled = aligned_y.notna()
        X = X.loc[labelled].astype(float)
        aligned_y = aligned_y.loc[labelled].astype(float)
        if not aligned_y.isin([0.0, 1.0]).all():
            raise ValueError("y must contain only binary labels 0/1")
        if aligned_y.nunique() != 2:
            raise ValueError("Training labels must contain both classes")
        if len(X) < 10:
            raise ValueError("At least 10 labelled observations are required")
        return X, aligned_y

    def _fit_preprocessor(self, X: pd.DataFrame, fit_rows: int) -> np.ndarray:
        self.imputer_ = SimpleImputer(strategy="median", keep_empty_features=True)
        self.scaler_ = StandardScaler()
        core = X.iloc[:fit_rows]
        core_imputed = self.imputer_.fit_transform(core)
        self.scaler_.fit(core_imputed)
        self.scaler_fit_end_ = pd.Timestamp(core.index[-1])
        return self.scaler_.transform(self.imputer_.transform(X)).astype(np.float32)

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        *,
        validation_X: pd.DataFrame | None = None,
        validation_y: pd.Series | None = None,
    ) -> CaitongClassifier:
        """Fit without learning preprocessing statistics from validation/test rows.

        Walk-forward callers supply the pre-registered two-year validation
        block explicitly.  Fixed-split reconstructions retain the chronological
        trailing-fraction fallback because the reports disclose no validation
        dates for their one-off fits.
        """

        X, y = self._validate_training_data(X, y)
        if (validation_X is None) != (validation_y is None):
            raise ValueError("validation_X and validation_y must be supplied together")
        explicit_validation = validation_X is not None
        validation_frame: pd.DataFrame | None = None
        validation_labels: pd.Series | None = None
        if validation_X is not None and validation_y is not None:
            if not isinstance(validation_X.index, pd.DatetimeIndex):
                raise TypeError("validation_X must use a DatetimeIndex")
            if not validation_X.index.is_monotonic_increasing or validation_X.index.has_duplicates:
                raise ValueError("validation_X index must be sorted and unique")
            if tuple(validation_X.columns) != tuple(X.columns):
                raise ValueError("validation features must exactly match training features")
            aligned_validation_y = pd.to_numeric(
                validation_y.reindex(validation_X.index), errors="coerce"
            )
            labelled = aligned_validation_y.notna()
            validation_frame = validation_X.loc[labelled].astype(float)
            validation_labels = aligned_validation_y.loc[labelled].astype(float)
            if validation_frame.empty:
                raise ValueError("explicit validation block has no labelled observations")
            if not validation_labels.isin([0.0, 1.0]).all():
                raise ValueError("validation_y must contain only binary labels 0/1")
            if validation_frame.index.min() <= X.index.max():
                raise ValueError("explicit validation must follow the fit block")

        self.feature_names_ = tuple(X.columns)
        self.training_end_ = pd.Timestamp(X.index[-1])
        self.training_rows_ = len(X)
        self.explicit_validation_ = explicit_validation
        _set_deterministic(self.config.seed)

        if self.config.kind == "elastic_net_logistic":
            scaled = self._fit_preprocessor(X, len(X))
            logistic_parameters: dict[str, object] = {
                "solver": "saga",
                "l1_ratio": self.config.elasticnet_l1_ratio,
                "C": self.config.logistic_c,
                "class_weight": "balanced",
                "max_iter": 5_000,
                "random_state": self.config.seed,
            }
            # sklearn 1.8+ infers elastic net from l1_ratio and deprecates
            # explicitly setting penalty; older supported releases require it.
            penalty_default = inspect.signature(LogisticRegression).parameters["penalty"].default
            if penalty_default != "deprecated":
                logistic_parameters["penalty"] = "elasticnet"
            estimator = LogisticRegression(**logistic_parameters)
            estimator.fit(scaled, y.to_numpy(dtype=int))
            self.estimator_ = estimator
            return self

        if explicit_validation:
            assert validation_frame is not None
            assert validation_labels is not None
            scaled = self._fit_preprocessor(X, len(X))
            assert self.imputer_ is not None
            assert self.scaler_ is not None
            scaled_validation = self.scaler_.transform(
                self.imputer_.transform(validation_frame)
            ).astype(np.float32)
            labels = y.to_numpy(dtype=np.float32)
            validation_values = validation_labels.to_numpy(dtype=np.float32)
            self.validation_start_ = pd.Timestamp(validation_frame.index[0])
            self.validation_end_ = pd.Timestamp(validation_frame.index[-1])
            self.validation_rows_ = len(validation_frame)
            if self.config.kind == "mlp":
                model = _MLP(scaled.shape[1], self.config.hidden_sizes)
                train_X, valid_X = scaled, scaled_validation
                train_y, valid_y = labels, validation_values
            else:
                train_X, train_endpoints = _trailing_sequences(
                    scaled, self.config.sequence_length
                )
                valid_X, valid_endpoints = _trailing_sequences(
                    scaled_validation, self.config.sequence_length
                )
                if not len(train_X) or not len(valid_X):
                    raise ValueError(
                        "Insufficient observations for explicit sequence validation"
                    )
                train_y = labels[train_endpoints]
                valid_y = validation_values[valid_endpoints]
                model = _RecurrentClassifier(
                    scaled.shape[1], self.config.recurrent_hidden_size, self.config.kind
                )
        else:
            split = _chronological_split(len(X), self.config.validation_fraction)
            scaled = self._fit_preprocessor(X, split)
            labels = y.to_numpy(dtype=np.float32)
            self.validation_start_ = pd.Timestamp(X.index[split])
            self.validation_end_ = pd.Timestamp(X.index[-1])
            self.validation_rows_ = len(X) - split
            if self.config.kind == "mlp":
                model = _MLP(scaled.shape[1], self.config.hidden_sizes)
                train_X, valid_X = scaled[:split], scaled[split:]
                train_y, valid_y = labels[:split], labels[split:]
            else:
                sequences, endpoints = _trailing_sequences(scaled, self.config.sequence_length)
                train_mask = endpoints < split
                valid_mask = endpoints >= split
                if not train_mask.any() or not valid_mask.any():
                    raise ValueError(
                        "Insufficient observations for chronological sequence validation"
                    )
                train_X, valid_X = sequences[train_mask], sequences[valid_mask]
                train_y, valid_y = labels[endpoints[train_mask]], labels[endpoints[valid_mask]]
                model = _RecurrentClassifier(
                    scaled.shape[1], self.config.recurrent_hidden_size, self.config.kind
                )

        self._fit_torch(model, train_X, train_y, valid_X, valid_y)
        return self

    def _fit_torch(
        self,
        model: nn.Module,
        train_X: np.ndarray,
        train_y: np.ndarray,
        valid_X: np.ndarray,
        valid_y: np.ndarray,
    ) -> None:
        device = torch.device("cpu")
        model.to(device)
        positives = float(train_y.sum())
        negatives = float(len(train_y) - positives)
        if positives == 0.0 or negatives == 0.0:
            raise ValueError("Chronological fit subset must contain both classes")
        loss_function = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(negatives / positives, dtype=torch.float32, device=device)
        )
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        generator = torch.Generator().manual_seed(self.config.seed)
        dataset = TensorDataset(
            torch.as_tensor(train_X, dtype=torch.float32),
            torch.as_tensor(train_y, dtype=torch.float32),
        )
        loader = DataLoader(
            dataset,
            batch_size=min(self.config.batch_size, len(dataset)),
            shuffle=True,
            generator=generator,
            num_workers=0,
        )
        validation_values = torch.as_tensor(valid_X, dtype=torch.float32, device=device)
        validation_labels = torch.as_tensor(valid_y, dtype=torch.float32, device=device)

        best_state: dict[str, torch.Tensor] | None = None
        best_loss = np.inf
        stale_epochs = 0
        for epoch in range(self.config.max_epochs):
            model.train()
            for batch_values, batch_labels in loader:
                optimizer.zero_grad(set_to_none=True)
                loss = loss_function(model(batch_values.to(device)), batch_labels.to(device))
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                validation_loss = float(
                    loss_function(model(validation_values), validation_labels).detach().cpu()
                )
            if validation_loss < best_loss - 1e-8:
                best_loss = validation_loss
                best_state = copy.deepcopy(model.state_dict())
                self.best_epoch_ = epoch + 1
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= self.config.patience:
                    break

        if best_state is None:  # pragma: no cover - finite inputs make this defensive
            raise RuntimeError("Neural model produced no finite validation loss")
        model.load_state_dict(best_state)
        model.eval()
        self.best_validation_loss_ = best_loss
        self.estimator_ = model

    def _check_fitted(self, X: pd.DataFrame) -> None:
        if self.feature_names_ is None or self.imputer_ is None or self.scaler_ is None:
            raise RuntimeError("CaitongClassifier is not fitted")
        missing = sorted(set(self.feature_names_) - set(X.columns))
        extra = sorted(set(X.columns) - set(self.feature_names_))
        if missing or extra:
            raise ValueError(f"Feature mismatch; missing={missing}, extra={extra}")

    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        """Return the probability of a positive forward return, index-aligned."""

        self._check_fitted(X)
        assert self.feature_names_ is not None
        assert self.imputer_ is not None
        assert self.scaler_ is not None
        assert self.estimator_ is not None
        ordered = X.loc[:, self.feature_names_].astype(float)
        scaled = self.scaler_.transform(self.imputer_.transform(ordered)).astype(np.float32)

        if self.config.kind == "elastic_net_logistic":
            assert isinstance(self.estimator_, LogisticRegression)
            probabilities = self.estimator_.predict_proba(scaled)[:, 1]
            return pd.Series(probabilities, index=X.index, name="probability")

        assert isinstance(self.estimator_, nn.Module)
        values: np.ndarray
        result_index: pd.Index
        if self.config.kind == "mlp":
            values = scaled
            result_index = X.index
        else:
            values, endpoints = _trailing_sequences(scaled, self.config.sequence_length)
            result_index = X.index[endpoints]
        if not len(values):
            return pd.Series(np.nan, index=X.index, name="probability")
        with torch.no_grad():
            logits = self.estimator_(torch.as_tensor(values, dtype=torch.float32))
            probabilities = torch.sigmoid(logits).cpu().numpy()
        output = pd.Series(np.nan, index=X.index, name="probability", dtype=float)
        output.loc[result_index] = probabilities
        return output

    @property
    def metadata(self) -> dict[str, object]:
        """Serializable audit metadata for experiment artifacts."""

        return {
            "model_kind": self.config.kind,
            "registered_proxy": self.config.kind == "mlp",
            "config": asdict(self.config),
            "feature_names": list(self.feature_names_ or ()),
            "training_rows": self.training_rows_,
            "training_end": self.training_end_.isoformat() if self.training_end_ else None,
            "validation_start": (
                self.validation_start_.isoformat() if self.validation_start_ else None
            ),
            "validation_end": (
                self.validation_end_.isoformat() if self.validation_end_ else None
            ),
            "validation_rows": self.validation_rows_,
            "explicit_walk_forward_validation": self.explicit_validation_,
            "scaler_fit_end": (
                self.scaler_fit_end_.isoformat() if self.scaler_fit_end_ is not None else None
            ),
            "best_epoch": self.best_epoch_,
            "best_validation_loss": self.best_validation_loss_,
            "assumption": (
                "MLP/GRU/LSTM均为未披露架构的代理；ElasticNet Logistic为线性基准。"
            ),
        }
