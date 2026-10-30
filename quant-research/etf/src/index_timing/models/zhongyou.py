"""Zhongyou LSTM autoencoder, causal GHMM and state-conditioned Kelly sizing."""

from __future__ import annotations

import copy
import random
from dataclasses import asdict, dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
import torch
from hmmlearn.hmm import GaussianHMM
from scipy.special import logsumexp
from sklearn.decomposition import PCA
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from index_timing.features.zhongyou import (
    DISCLOSED_FEATURE_COLUMNS,
    SOFT_STATE_COLUMNS,
)

RepresentationKind = Literal["raw", "pca", "lstm"]


def _set_deterministic(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # pragma: no cover - CI uses CPU
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _sequence_array(
    frame: pd.DataFrame, sequence_length: int
) -> tuple[np.ndarray, np.ndarray]:
    values = frame.to_numpy(dtype=np.float32)
    if len(values) < sequence_length:
        return (
            np.empty((0, sequence_length, values.shape[1]), dtype=np.float32),
            np.array([], dtype=int),
        )
    sequences: list[np.ndarray] = []
    endpoints: list[int] = []
    for end in range(sequence_length - 1, len(values)):
        section = values[end - sequence_length + 1 : end + 1]
        if np.isfinite(section).all():
            sequences.append(section)
            endpoints.append(end)
    if not sequences:
        return (
            np.empty((0, sequence_length, values.shape[1]), dtype=np.float32),
            np.array([], dtype=int),
        )
    return np.stack(sequences).astype(np.float32), np.asarray(endpoints, dtype=int)


@dataclass(frozen=True, slots=True)
class AutoencoderConfig:
    input_size: int = 17
    hidden_size: int = 64
    latent_size: int = 10
    sequence_length: int = 90
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 100
    patience: int = 10
    batch_size: int = 64
    validation_fraction: float = 0.20
    seed: int = 0

    def __post_init__(self) -> None:
        if self.input_size != len(DISCLOSED_FEATURE_COLUMNS):
            raise ValueError(
                "The faithful proxy input size must remain the disclosed 17 dimensions"
            )
        if self.hidden_size != 64 or self.latent_size != 10:
            raise ValueError("The registered LSTM architecture must use hidden=64 and latent=10")
        if self.sequence_length < 2:
            raise ValueError("sequence_length must be at least two")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("learning rate must be positive and weight decay non-negative")
        if self.max_epochs < 1 or self.patience < 1 or self.batch_size < 1:
            raise ValueError("training controls must be positive")
        if not 0.05 <= self.validation_fraction <= 0.5:
            raise ValueError("validation_fraction must be between 0.05 and 0.5")


class LSTMAutoencoderNetwork(nn.Module):
    """17 → LSTM(64) → latent(10) → LSTM(64) → 17 sequence decoder."""

    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        self.encoder_lstm = nn.LSTM(
            config.input_size, config.hidden_size, num_layers=1, batch_first=True
        )
        self.to_latent = nn.Linear(config.hidden_size, config.latent_size)
        self.decoder_lstm = nn.LSTM(
            config.latent_size, config.hidden_size, num_layers=1, batch_first=True
        )
        self.to_features = nn.Linear(config.hidden_size, config.input_size)

    def encode_tensor(self, values: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.encoder_lstm(values)
        return self.to_latent(hidden[-1])

    def forward(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encode_tensor(values)
        repeated = latent.unsqueeze(1).expand(-1, values.shape[1], -1)
        decoded, _ = self.decoder_lstm(repeated)
        return self.to_features(decoded), latent


class LSTMAutoencoder:
    """Pandas-facing deterministic autoencoder trained only on supplied rows."""

    def __init__(self, config: AutoencoderConfig | None = None) -> None:
        self.config = config or AutoencoderConfig()
        self.network_: LSTMAutoencoderNetwork | None = None
        self.feature_names_: tuple[str, ...] | None = None
        self.training_start_: pd.Timestamp | None = None
        self.training_end_: pd.Timestamp | None = None
        self.training_sequences_: int = 0
        self.validation_start_: pd.Timestamp | None = None
        self.validation_end_: pd.Timestamp | None = None
        self.validation_sequences_: int = 0
        self.explicit_validation_: bool = False
        self.best_epoch_: int | None = None
        self.best_validation_loss_: float | None = None

    def _validate_features(self, features: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(features.index, pd.DatetimeIndex):
            raise TypeError("features must use a DatetimeIndex")
        if not features.index.is_monotonic_increasing or features.index.has_duplicates:
            raise ValueError("features index must be sorted and unique")
        expected = tuple(DISCLOSED_FEATURE_COLUMNS)
        if tuple(features.columns) != expected:
            raise ValueError("autoencoder features must be the ordered disclosed 17 dimensions")
        return features.apply(pd.to_numeric, errors="coerce").astype(float)

    def fit(
        self,
        features: pd.DataFrame,
        *,
        validation_features: pd.DataFrame | None = None,
    ) -> LSTMAutoencoder:
        values = self._validate_features(features)
        sequences, sequence_positions = _sequence_array(
            values, self.config.sequence_length
        )
        if len(sequences) < 10:
            raise ValueError("At least 10 complete trailing sequences are required")
        if validation_features is not None:
            validation_values = self._validate_features(validation_features)
            if validation_values.index.min() <= values.index.max():
                raise ValueError("explicit validation must follow the fit block")
            validation_sequences, _ = _sequence_array(
                validation_values, self.config.sequence_length
            )
            if not len(validation_sequences):
                raise ValueError("explicit validation has no complete trailing sequence")
            training_sequences = sequences
            self.validation_start_ = pd.Timestamp(validation_values.index[0])
            self.validation_end_ = pd.Timestamp(validation_values.index[-1])
            self.explicit_validation_ = True
        else:
            validation_rows = max(
                1, int(np.ceil(len(sequences) * self.config.validation_fraction))
            )
            split = len(sequences) - validation_rows
            if split < 2:
                raise ValueError("Insufficient sequences before chronological validation")
            training_sequences = sequences[:split]
            validation_sequences = sequences[split:]
            self.validation_start_ = pd.Timestamp(values.index[sequence_positions[split]])
            self.validation_end_ = pd.Timestamp(values.index[sequence_positions[-1]])

        _set_deterministic(self.config.seed)
        network = LSTMAutoencoderNetwork(self.config)
        optimizer = torch.optim.Adam(
            network.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        loss_function = nn.MSELoss()
        generator = torch.Generator().manual_seed(self.config.seed)
        dataset = TensorDataset(torch.as_tensor(training_sequences, dtype=torch.float32))
        loader = DataLoader(
            dataset,
            batch_size=min(self.config.batch_size, len(dataset)),
            shuffle=True,
            generator=generator,
            num_workers=0,
        )
        validation = torch.as_tensor(validation_sequences, dtype=torch.float32)

        best_state: dict[str, torch.Tensor] | None = None
        best_loss = np.inf
        stale_epochs = 0
        for epoch in range(self.config.max_epochs):
            network.train()
            for (batch,) in loader:
                optimizer.zero_grad(set_to_none=True)
                reconstruction, _ = network(batch)
                loss = loss_function(reconstruction, batch)
                loss.backward()
                optimizer.step()
            network.eval()
            with torch.no_grad():
                reconstructed, _ = network(validation)
                validation_loss = float(loss_function(reconstructed, validation).cpu())
            if validation_loss < best_loss - 1e-8:
                best_loss = validation_loss
                best_state = copy.deepcopy(network.state_dict())
                self.best_epoch_ = epoch + 1
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= self.config.patience:
                    break
        if best_state is None:  # pragma: no cover - finite sequences are defensive
            raise RuntimeError("autoencoder produced no finite validation loss")
        network.load_state_dict(best_state)
        network.eval()
        self.network_ = network
        self.feature_names_ = tuple(values.columns)
        self.training_start_ = pd.Timestamp(values.index[0])
        self.training_end_ = pd.Timestamp(values.index[-1])
        self.training_sequences_ = len(training_sequences)
        self.validation_sequences_ = len(validation_sequences)
        self.best_validation_loss_ = best_loss
        return self

    def encode(self, features: pd.DataFrame) -> pd.DataFrame:
        values = self._validate_features(features)
        if self.network_ is None or self.feature_names_ is None:
            raise RuntimeError("LSTMAutoencoder is not fitted")
        sequences, endpoints = _sequence_array(values, self.config.sequence_length)
        columns = [f"latent_{position}" for position in range(self.config.latent_size)]
        output = pd.DataFrame(np.nan, index=values.index, columns=columns, dtype=float)
        if len(sequences):
            with torch.no_grad():
                latent = self.network_.encode_tensor(
                    torch.as_tensor(sequences, dtype=torch.float32)
                )
            output.iloc[endpoints] = latent.cpu().numpy()
        return output

    @property
    def metadata(self) -> dict[str, object]:
        return {
            "config": asdict(self.config),
            "training_start": self.training_start_.isoformat() if self.training_start_ else None,
            "training_end": self.training_end_.isoformat() if self.training_end_ else None,
            "training_sequences": self.training_sequences_,
            "validation_start": (
                self.validation_start_.isoformat() if self.validation_start_ else None
            ),
            "validation_end": (
                self.validation_end_.isoformat() if self.validation_end_ else None
            ),
            "validation_sequences": self.validation_sequences_,
            "explicit_walk_forward_validation": self.explicit_validation_,
            "best_epoch": self.best_epoch_,
            "best_validation_loss": self.best_validation_loss_,
            "objective": "causal trailing-sequence reconstruction MSE",
        }


@dataclass(frozen=True, slots=True)
class GHMMConfig:
    n_states: int = 5
    covariance_type: Literal["full"] = "full"
    restarts: int = 20
    max_iterations: int = 300
    tolerance: float = 1e-3
    min_covar: float = 1e-4
    seed: int = 0

    def __post_init__(self) -> None:
        if self.n_states < 2:
            raise ValueError("GHMM requires at least two states")
        if self.covariance_type != "full":
            raise ValueError("The registered GHMM must use full covariance")
        if self.restarts < 1 or self.max_iterations < 1:
            raise ValueError("restarts and max_iterations must be positive")
        if self.tolerance <= 0 or self.min_covar <= 0:
            raise ValueError("tolerance and min_covar must be positive")


class CausalGaussianHMM:
    """Fit with hmmlearn, infer with an explicit forward-only recursion."""

    def __init__(self, config: GHMMConfig | None = None) -> None:
        self.config = config or GHMMConfig()
        self.startprob_: np.ndarray | None = None
        self.transmat_: np.ndarray | None = None
        self.means_: np.ndarray | None = None
        self.covars_: np.ndarray | None = None
        self.best_log_likelihood_: float | None = None
        self.best_selection_score_: float | None = None
        self.best_restart_: int | None = None
        self.training_rows_: int = 0
        self.validation_rows_: int = 0
        self.selection_scope_: str = "training_log_likelihood"
        self.feature_names_: tuple[str, ...] | None = None

    @property
    def is_fitted(self) -> bool:
        return all(
            parameter is not None
            for parameter in (self.startprob_, self.transmat_, self.means_, self.covars_)
        )

    def set_parameters(
        self,
        *,
        startprob: np.ndarray,
        transmat: np.ndarray,
        means: np.ndarray,
        covars: np.ndarray,
    ) -> CausalGaussianHMM:
        """Set fixed parameters, primarily for audited synthetic tests."""

        states = self.config.n_states
        start = np.asarray(startprob, dtype=float)
        transition = np.asarray(transmat, dtype=float)
        means_array = np.asarray(means, dtype=float)
        covariances = np.asarray(covars, dtype=float)
        if start.shape != (states,) or transition.shape != (states, states):
            raise ValueError("start probability or transition matrix has the wrong shape")
        if means_array.ndim != 2 or means_array.shape[0] != states:
            raise ValueError("means must have shape (n_states, n_features)")
        dimension = means_array.shape[1]
        if covariances.shape != (states, dimension, dimension):
            raise ValueError("full covariances have the wrong shape")
        if (start < 0).any() or (transition < 0).any():
            raise ValueError("probabilities cannot be negative")
        if not np.isclose(start.sum(), 1.0) or not np.allclose(transition.sum(axis=1), 1.0):
            raise ValueError("start and transition probabilities must sum to one")
        self.startprob_ = start.copy()
        self.transmat_ = transition.copy()
        self.means_ = means_array.copy()
        self.covars_ = covariances.copy()
        return self

    def fit(
        self,
        observations: pd.DataFrame,
        *,
        validation_observations: pd.DataFrame | None = None,
    ) -> CausalGaussianHMM:
        if not isinstance(observations.index, pd.DatetimeIndex):
            raise TypeError("observations must use a DatetimeIndex")
        if observations.columns.has_duplicates:
            raise ValueError("observation feature names must be unique")
        numeric = observations.apply(pd.to_numeric, errors="coerce").astype(float)
        if not np.isfinite(numeric.to_numpy()).all():
            raise ValueError("GHMM fitting observations must all be finite")
        if len(numeric) <= self.config.n_states:
            raise ValueError("GHMM needs more rows than hidden states")
        validation_numeric: pd.DataFrame | None = None
        if validation_observations is not None:
            if tuple(validation_observations.columns) != tuple(numeric.columns):
                raise ValueError("GHMM validation columns must match fit observations")
            validation_numeric = validation_observations.apply(
                pd.to_numeric, errors="coerce"
            ).astype(float)
            if not np.isfinite(validation_numeric.to_numpy()).all():
                raise ValueError("GHMM validation observations must all be finite")
            if len(validation_numeric) <= self.config.n_states:
                raise ValueError("GHMM validation needs more rows than hidden states")

        best_model: GaussianHMM | None = None
        best_selection_score = -np.inf
        best_training_score = -np.inf
        best_restart: int | None = None
        for restart in range(self.config.restarts):
            candidate = GaussianHMM(
                n_components=self.config.n_states,
                covariance_type="full",
                n_iter=self.config.max_iterations,
                tol=self.config.tolerance,
                min_covar=self.config.min_covar,
                random_state=self.config.seed + restart,
            )
            try:
                candidate.fit(numeric.to_numpy())
                training_score = float(candidate.score(numeric.to_numpy()))
                selection_score = (
                    float(candidate.score(validation_numeric.to_numpy()))
                    if validation_numeric is not None
                    else training_score
                )
            except (ValueError, np.linalg.LinAlgError):
                continue
            if np.isfinite(selection_score) and selection_score > best_selection_score:
                best_model = candidate
                best_selection_score = selection_score
                best_training_score = training_score
                best_restart = restart
        if best_model is None:
            raise RuntimeError("all GHMM restarts failed")
        self.set_parameters(
            startprob=best_model.startprob_,
            transmat=best_model.transmat_,
            means=best_model.means_,
            covars=best_model.covars_,
        )
        self.best_log_likelihood_ = best_training_score
        self.best_selection_score_ = best_selection_score
        self.best_restart_ = best_restart
        self.training_rows_ = len(numeric)
        self.validation_rows_ = len(validation_numeric) if validation_numeric is not None else 0
        self.selection_scope_ = (
            "explicit_walk_forward_validation_log_likelihood"
            if validation_numeric is not None
            else "training_log_likelihood"
        )
        self.feature_names_ = tuple(numeric.columns)
        return self

    def _emission_log_likelihood(self, observations: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("CausalGaussianHMM is not fitted")
        assert self.means_ is not None
        assert self.covars_ is not None
        if observations.shape[1] != self.means_.shape[1]:
            raise ValueError("observation dimension does not match fitted GHMM")
        output = np.empty((len(observations), self.config.n_states), dtype=float)
        dimension = observations.shape[1]
        identity = np.eye(dimension)
        for state in range(self.config.n_states):
            covariance = (self.covars_[state] + self.covars_[state].T) / 2.0
            covariance = covariance + identity * self.config.min_covar
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            eigenvalues = np.maximum(eigenvalues, self.config.min_covar)
            inverse = (eigenvectors / eigenvalues) @ eigenvectors.T
            log_determinant = float(np.log(eigenvalues).sum())
            difference = observations - self.means_[state]
            quadratic = np.einsum("ij,jk,ik->i", difference, inverse, difference)
            output[:, state] = -0.5 * (
                dimension * np.log(2.0 * np.pi) + log_determinant + quadratic
            )
        return output

    def filter_proba(self, observations: pd.DataFrame | np.ndarray) -> pd.DataFrame:
        """Return P(S_t | o_1,...,o_t), never the smoothed full-sequence posterior."""

        if isinstance(observations, pd.DataFrame):
            index = observations.index
            values = observations.to_numpy(dtype=float)
            if (
                self.feature_names_ is not None
                and tuple(observations.columns) != self.feature_names_
            ):
                raise ValueError("observation columns differ from the fitted GHMM")
        else:
            values = np.asarray(observations, dtype=float)
            index = pd.RangeIndex(len(values))
        if values.ndim != 2:
            raise ValueError("observations must be a two-dimensional matrix")
        valid = np.isfinite(values).all(axis=1)
        probabilities = np.full((len(values), self.config.n_states), np.nan, dtype=float)
        if not valid.any():
            return pd.DataFrame(
                probabilities,
                index=index,
                columns=[f"state_probability_{state}" for state in range(self.config.n_states)],
            )
        emissions = self._emission_log_likelihood(values[valid])
        assert self.startprob_ is not None
        assert self.transmat_ is not None
        previous: np.ndarray | None = None
        emission_row = 0
        for row, is_valid in enumerate(valid):
            if not is_valid:
                continue
            prediction = self.startprob_ if previous is None else previous @ self.transmat_
            log_alpha = np.log(np.clip(prediction, 1e-300, None)) + emissions[emission_row]
            log_alpha -= logsumexp(log_alpha)
            previous = np.exp(log_alpha)
            probabilities[row] = previous
            emission_row += 1
        return pd.DataFrame(
            probabilities,
            index=index,
            columns=[f"state_probability_{state}" for state in range(self.config.n_states)],
        )

    def map_states(self, observations: pd.DataFrame | np.ndarray) -> pd.Series:
        probabilities = self.filter_proba(observations)
        valid = probabilities.notna().all(axis=1)
        states = pd.Series(pd.NA, index=probabilities.index, dtype="Int64", name="map_state")
        states.loc[valid] = probabilities.loc[valid].to_numpy().argmax(axis=1)
        return states

    @property
    def metadata(self) -> dict[str, object]:
        return {
            "config": asdict(self.config),
            "training_rows": self.training_rows_,
            "feature_names": list(self.feature_names_ or ()),
            "best_restart": self.best_restart_,
            "best_log_likelihood": self.best_log_likelihood_,
            "best_selection_score": self.best_selection_score_,
            "selection_scope": self.selection_scope_,
            "validation_rows": self.validation_rows_,
            "inference": "custom forward filter P(S_t | o_1,...,o_t); no smoother/Viterbi",
        }


@dataclass(frozen=True, slots=True)
class ZhongyouModelConfig:
    representation: RepresentationKind = "lstm"
    pca_components: int = 10
    autoencoder: AutoencoderConfig = field(default_factory=AutoencoderConfig)
    ghmm: GHMMConfig = field(default_factory=GHMMConfig)

    def __post_init__(self) -> None:
        if self.representation not in {"raw", "pca", "lstm"}:
            raise ValueError("representation must be raw, pca or lstm")
        if not 1 <= self.pca_components <= len(DISCLOSED_FEATURE_COLUMNS):
            raise ValueError("pca_components must lie between 1 and 17")


class ZhongyouStateModel:
    """Raw/PCA/LSTM representation followed by a fixed causal GHMM."""

    def __init__(self, config: ZhongyouModelConfig | None = None) -> None:
        self.config = config or ZhongyouModelConfig()
        self.autoencoder_: LSTMAutoencoder | None = None
        self.pca_: PCA | None = None
        self.ghmm_: CausalGaussianHMM | None = None
        self.training_start_: pd.Timestamp | None = None
        self.training_end_: pd.Timestamp | None = None
        self.validation_start_: pd.Timestamp | None = None
        self.validation_end_: pd.Timestamp | None = None
        self.observation_columns_: tuple[str, ...] | None = None
        self.training_observations_: pd.DataFrame | None = None
        self.validation_observations_: pd.DataFrame | None = None

    def _validate_inputs(
        self, features: pd.DataFrame, soft_states: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        if tuple(features.columns) != tuple(DISCLOSED_FEATURE_COLUMNS):
            raise ValueError("features must contain the ordered disclosed 17 dimensions")
        if tuple(soft_states.columns) != tuple(SOFT_STATE_COLUMNS):
            raise ValueError("soft_states must contain the ordered six inherited probabilities")
        if not features.index.equals(soft_states.index):
            raise ValueError("features and soft states must have identical indices")
        if not isinstance(features.index, pd.DatetimeIndex):
            raise TypeError("model inputs must use a DatetimeIndex")
        if not features.index.is_monotonic_increasing or features.index.has_duplicates:
            raise ValueError("model input index must be sorted and unique")
        return features.astype(float), soft_states.astype(float)

    def _fit_representation(
        self,
        features: pd.DataFrame,
        train_mask: np.ndarray,
        validation_mask: np.ndarray | None = None,
    ) -> pd.DataFrame:
        if self.config.representation == "raw":
            return features.add_prefix("raw__")
        if self.config.representation == "pca":
            train = features.loc[train_mask].dropna()
            if len(train) < self.config.pca_components:
                raise ValueError("not enough complete training rows for PCA")
            self.pca_ = PCA(n_components=self.config.pca_components, svd_solver="full")
            self.pca_.fit(train.to_numpy())
            valid = features.notna().all(axis=1)
            columns = [f"pca_{component}" for component in range(self.config.pca_components)]
            output = pd.DataFrame(np.nan, index=features.index, columns=columns)
            output.loc[valid] = self.pca_.transform(features.loc[valid].to_numpy())
            return output
        validation_features = (
            features.loc[validation_mask] if validation_mask is not None else None
        )
        self.autoencoder_ = LSTMAutoencoder(self.config.autoencoder).fit(
            features.loc[train_mask],
            validation_features=validation_features,
        )
        return self.autoencoder_.encode(features)

    def _transform_representation(self, features: pd.DataFrame) -> pd.DataFrame:
        if self.config.representation == "raw":
            return features.add_prefix("raw__")
        if self.config.representation == "pca":
            if self.pca_ is None:
                raise RuntimeError("PCA representation is not fitted")
            valid = features.notna().all(axis=1)
            columns = [f"pca_{component}" for component in range(self.config.pca_components)]
            output = pd.DataFrame(np.nan, index=features.index, columns=columns)
            output.loc[valid] = self.pca_.transform(features.loc[valid].to_numpy())
            return output
        if self.autoencoder_ is None:
            raise RuntimeError("LSTM representation is not fitted")
        return self.autoencoder_.encode(features)

    @staticmethod
    def _observations(representation: pd.DataFrame, soft_states: pd.DataFrame) -> pd.DataFrame:
        return pd.concat([representation, soft_states], axis=1)

    def fit(
        self,
        features: pd.DataFrame,
        soft_states: pd.DataFrame,
        *,
        train_start: str | pd.Timestamp,
        train_end: str | pd.Timestamp,
        train_index: pd.DatetimeIndex | None = None,
        validation_index: pd.DatetimeIndex | None = None,
    ) -> ZhongyouStateModel:
        features, soft_states = self._validate_inputs(features, soft_states)
        start = pd.Timestamp(train_start)
        end = pd.Timestamp(train_end)
        if start > end:
            raise ValueError("train_start must not be after train_end")
        if (train_index is None) != (validation_index is None):
            raise ValueError("train_index and validation_index must be supplied together")
        if train_index is not None:
            fit_dates = pd.DatetimeIndex(train_index)
            validation_dates = pd.DatetimeIndex(validation_index)
            train_mask = np.asarray(features.index.isin(fit_dates))
            validation_mask: np.ndarray | None = np.asarray(
                features.index.isin(validation_dates)
            )
            start = pd.Timestamp(fit_dates.min())
            end = pd.Timestamp(fit_dates.max())
            self.validation_start_ = pd.Timestamp(validation_dates.min())
            self.validation_end_ = pd.Timestamp(validation_dates.max())
            if self.validation_start_ <= end:
                raise ValueError("explicit validation must follow the fit block")
        else:
            train_mask = np.asarray((features.index >= start) & (features.index <= end))
            validation_mask = None
        if not train_mask.any():
            raise ValueError("training interval contains no observations")
        representation = self._fit_representation(
            features, train_mask, validation_mask
        )
        observations = self._observations(representation, soft_states)
        train_observations = observations.loc[train_mask].dropna()
        validation_observations = (
            observations.loc[validation_mask].dropna()
            if validation_mask is not None
            else None
        )
        self.ghmm_ = CausalGaussianHMM(self.config.ghmm).fit(
            train_observations,
            validation_observations=validation_observations,
        )
        self.training_start_ = start
        self.training_end_ = end
        self.observation_columns_ = tuple(observations.columns)
        self.training_observations_ = train_observations
        self.validation_observations_ = validation_observations
        return self

    def observation_frame(
        self, features: pd.DataFrame, soft_states: pd.DataFrame
    ) -> pd.DataFrame:
        features, soft_states = self._validate_inputs(features, soft_states)
        representation = self._transform_representation(features)
        observations = self._observations(representation, soft_states)
        if (
            self.observation_columns_ is not None
            and tuple(observations.columns) != self.observation_columns_
        ):
            raise ValueError("observation columns differ from fitted model")
        return observations

    def filter_probabilities(
        self, features: pd.DataFrame, soft_states: pd.DataFrame
    ) -> pd.DataFrame:
        if self.ghmm_ is None or self.training_start_ is None:
            raise RuntimeError("ZhongyouStateModel is not fitted")
        observations = self.observation_frame(features, soft_states)
        eligible = observations.loc[observations.index >= self.training_start_]
        probabilities = self.ghmm_.filter_proba(eligible)
        output = pd.DataFrame(np.nan, index=observations.index, columns=probabilities.columns)
        output.loc[probabilities.index] = probabilities
        return output

    def map_states(self, features: pd.DataFrame, soft_states: pd.DataFrame) -> pd.Series:
        probabilities = self.filter_probabilities(features, soft_states)
        valid = probabilities.notna().all(axis=1)
        output = pd.Series(pd.NA, index=probabilities.index, dtype="Int64", name="map_state")
        output.loc[valid] = probabilities.loc[valid].to_numpy().argmax(axis=1)
        return output

    @property
    def metadata(self) -> dict[str, object]:
        representation_metadata: dict[str, object] | None = None
        if self.autoencoder_ is not None:
            representation_metadata = self.autoencoder_.metadata
        elif self.pca_ is not None:
            representation_metadata = {
                "components": self.config.pca_components,
                "explained_variance_ratio": self.pca_.explained_variance_ratio_.tolist(),
                "fit_scope": "training rows only",
            }
        return {
            "config": asdict(self.config),
            "representation": self.config.representation,
            "representation_metadata": representation_metadata,
            "observation_dimensions": len(self.observation_columns_ or ()),
            "ghmm": self.ghmm_.metadata if self.ghmm_ is not None else None,
            "train_start": self.training_start_.isoformat() if self.training_start_ else None,
            "train_end": self.training_end_.isoformat() if self.training_end_ else None,
            "validation_start": (
                self.validation_start_.isoformat() if self.validation_start_ else None
            ),
            "validation_end": (
                self.validation_end_.isoformat() if self.validation_end_ else None
            ),
        }


def kelly_position_details(
    map_states: pd.Series,
    next_day_excess_returns: pd.Series,
    *,
    lookback_same_state: int = 15,
    minimum_samples: int | None = None,
) -> pd.DataFrame:
    """Kelly fraction from only already-realized outcomes of the current MAP state."""

    if lookback_same_state < 1:
        raise ValueError("lookback_same_state must be positive")
    required = lookback_same_state if minimum_samples is None else minimum_samples
    if not 1 <= required <= lookback_same_state:
        raise ValueError("minimum_samples must be between one and the lookback")
    states = pd.to_numeric(map_states, errors="coerce")
    outcomes = pd.to_numeric(next_day_excess_returns.reindex(states.index), errors="coerce")
    details = pd.DataFrame(
        {
            "prior_same_state_samples": 0,
            "win_probability": np.nan,
            "win_loss_odds": np.nan,
            "kelly_position": 0.0,
        },
        index=states.index,
    )
    state_values = states.to_numpy(dtype=float)
    return_values = outcomes.to_numpy(dtype=float)
    for row, current_state in enumerate(state_values):
        if not np.isfinite(current_state):
            continue
        # Outcome at row j is the j→j+1 return and becomes known at row j+1.
        # Slicing strictly before the current row therefore enforces causality.
        eligible = np.flatnonzero(
            (state_values[:row] == current_state) & np.isfinite(return_values[:row])
        )
        eligible = eligible[-lookback_same_state:]
        details.iloc[row, details.columns.get_loc("prior_same_state_samples")] = len(eligible)
        if len(eligible) < required:
            continue
        sample = return_values[eligible]
        wins = sample[sample > 0.0]
        losses = sample[sample < 0.0]
        if not len(wins) or not len(losses):
            continue
        probability = float(len(wins) / len(sample))
        odds = float(wins.mean() / abs(losses.mean()))
        fraction = probability - (1.0 - probability) / odds if odds > 0 else 0.0
        details.iloc[row, details.columns.get_loc("win_probability")] = probability
        details.iloc[row, details.columns.get_loc("win_loss_odds")] = odds
        details.iloc[row, details.columns.get_loc("kelly_position")] = np.clip(
            fraction, 0.0, 1.0
        )
    details["prior_same_state_samples"] = details["prior_same_state_samples"].astype(int)
    return details


def state_conditioned_kelly(
    map_states: pd.Series,
    next_day_excess_returns: pd.Series,
    *,
    lookback_same_state: int = 15,
    minimum_samples: int | None = None,
) -> pd.Series:
    details = kelly_position_details(
        map_states,
        next_day_excess_returns,
        lookback_same_state=lookback_same_state,
        minimum_samples=minimum_samples,
    )
    return details["kelly_position"].rename("target_position")


def apply_microcap_opening_patch(
    positions: pd.Series,
    *,
    enabled: bool = False,
    threshold: float = 0.50,
) -> pd.Series:
    """Post-hoc microcap rule: Kelly above 50% is forced to full exposure."""

    if not 0 <= threshold <= 1:
        raise ValueError("microcap patch threshold must lie in [0, 1]")
    output = pd.to_numeric(positions, errors="coerce").astype(float).copy()
    if enabled:
        output = output.mask(output > threshold, 1.0)
    return output.rename(positions.name)


def apply_bottom_recovery_patch(
    positions: pd.Series,
    trigger: pd.Series | None,
    *,
    enabled: bool = False,
) -> pd.Series:
    """Documented 50% + 50%×Kelly patch; disabled without an external trigger."""

    output = pd.to_numeric(positions, errors="coerce").astype(float).copy()
    if not enabled:
        return output.rename(positions.name)
    if trigger is None:
        raise ValueError("bottom-recovery patch requires an explicit point-in-time trigger")
    active = trigger.reindex(output.index).fillna(False).astype(bool)
    output.loc[active] = 0.5 + 0.5 * output.loc[active]
    return output.clip(0.0, 1.0).rename(positions.name)


__all__ = [
    "AutoencoderConfig",
    "CausalGaussianHMM",
    "GHMMConfig",
    "LSTMAutoencoder",
    "LSTMAutoencoderNetwork",
    "RepresentationKind",
    "ZhongyouModelConfig",
    "ZhongyouStateModel",
    "apply_bottom_recovery_patch",
    "apply_microcap_opening_patch",
    "kelly_position_details",
    "state_conditioned_kelly",
]
