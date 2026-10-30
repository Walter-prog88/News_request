from __future__ import annotations

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest
from sklearn.metrics import adjusted_rand_score

from index_timing.features.zhongyou import (
    DISCLOSED_FEATURE_COLUMNS,
    SOFT_STATE_COLUMNS,
)
from index_timing.models.zhongyou import (
    AutoencoderConfig,
    CausalGaussianHMM,
    GHMMConfig,
    LSTMAutoencoder,
    LSTMAutoencoderNetwork,
    ZhongyouModelConfig,
    ZhongyouStateModel,
    apply_bottom_recovery_patch,
    apply_microcap_opening_patch,
    kelly_position_details,
)


def _feature_frame(rows: int = 150) -> pd.DataFrame:
    rng = np.random.default_rng(8)
    return pd.DataFrame(
        rng.normal(size=(rows, 17)),
        index=pd.bdate_range("2019-01-02", periods=rows),
        columns=DISCLOSED_FEATURE_COLUMNS,
    )


def _soft_states(index: pd.DatetimeIndex) -> pd.DataFrame:
    rng = np.random.default_rng(9)
    values = rng.dirichlet(np.ones(6), size=len(index))
    return pd.DataFrame(values, index=index, columns=SOFT_STATE_COLUMNS)


def test_registered_architecture_and_ghmm_defaults() -> None:
    autoencoder = AutoencoderConfig()
    ghmm = GHMMConfig()
    assert autoencoder.input_size == 17
    assert autoencoder.hidden_size == 64
    assert autoencoder.latent_size == 10
    assert autoencoder.sequence_length == 90
    assert ghmm.n_states == 5
    assert ghmm.covariance_type == "full"
    assert ghmm.restarts == 20
    network = LSTMAutoencoderNetwork(autoencoder)
    assert network.encoder_lstm.input_size == 17
    assert network.encoder_lstm.hidden_size == 64
    assert network.to_latent.out_features == 10
    assert network.to_features.out_features == 17


def test_lstm_encoder_is_deterministic_and_prefix_invariant_after_fit() -> None:
    features = _feature_frame()
    config = AutoencoderConfig(
        sequence_length=10,
        max_epochs=2,
        patience=2,
        batch_size=32,
        seed=12,
    )
    first = LSTMAutoencoder(config).fit(features.iloc[:110])
    encoded_full = first.encode(features)
    encoded_prefix = first.encode(features.iloc[:125])
    pdt.assert_frame_equal(encoded_full.iloc[:125], encoded_prefix, atol=1e-7, rtol=1e-7)

    second = LSTMAutoencoder(config).fit(features.iloc[:110])
    pdt.assert_frame_equal(
        encoded_full,
        second.encode(features),
        check_exact=False,
        atol=1e-7,
        rtol=1e-7,
    )


def test_lstm_encoder_accepts_a_separate_walk_forward_validation_block() -> None:
    features = _feature_frame(220)
    config = AutoencoderConfig(
        sequence_length=10,
        max_epochs=2,
        patience=1,
        batch_size=32,
        seed=6,
    )
    model = LSTMAutoencoder(config).fit(
        features.iloc[:120], validation_features=features.iloc[140:]
    )

    assert model.metadata["explicit_walk_forward_validation"] is True
    assert model.metadata["validation_start"] == features.index[140].isoformat()
    assert model.metadata["training_sequences"] == 111
    assert model.metadata["validation_sequences"] == 71


def _manual_hmm() -> CausalGaussianHMM:
    return CausalGaussianHMM(GHMMConfig(n_states=2, restarts=1)).set_parameters(
        startprob=np.array([0.5, 0.5]),
        transmat=np.array([[0.95, 0.05], [0.05, 0.95]]),
        means=np.array([[-1.0], [1.0]]),
        covars=np.array([[[0.10]], [[0.10]]]),
    )


def test_custom_ghmm_filter_is_forward_only_and_prefix_invariant() -> None:
    index = pd.bdate_range("2024-01-02", periods=12)
    observations = pd.DataFrame(
        {"x": [-1.0] * 6 + [1.0] * 6}, index=index
    )
    model = _manual_hmm()
    model.feature_names_ = ("x",)
    full = model.filter_proba(observations)
    prefix = model.filter_proba(observations.iloc[:8])
    pdt.assert_frame_equal(full.iloc[:8], prefix)
    np.testing.assert_allclose(full.sum(axis=1), 1.0, atol=1e-12)
    assert full.iloc[2, 0] > 0.99
    assert full.iloc[-1, 1] > 0.99

    changed_future = observations.copy()
    changed_future.iloc[8:] = -100.0
    changed = model.filter_proba(changed_future)
    pdt.assert_frame_equal(full.iloc[:8], changed.iloc[:8])


def test_ghmm_recovers_synthetic_regimes_up_to_label_permutation() -> None:
    rng = np.random.default_rng(17)
    true_state = np.repeat([0, 1, 0, 1], 100)
    observations = pd.DataFrame(
        {
            "x": rng.normal(np.where(true_state == 0, -2.0, 2.0), 0.25),
            "vol": rng.normal(np.where(true_state == 0, 0.4, 1.4), 0.10),
        },
        index=pd.bdate_range("2020-01-02", periods=len(true_state)),
    )
    config = GHMMConfig(n_states=2, restarts=4, max_iterations=100, seed=5)
    fitted = CausalGaussianHMM(config).fit(observations)
    inferred = fitted.filter_proba(observations).to_numpy().argmax(axis=1)

    # Adjusted Rand is invariant to the arbitrary numeric names assigned to
    # hidden states, which is the correct criterion for HMM label switching.
    assert adjusted_rand_score(true_state, inferred) > 0.90

    permutation = np.array([1, 0])
    permuted = CausalGaussianHMM(config).set_parameters(
        startprob=fitted.startprob_[permutation],
        transmat=fitted.transmat_[np.ix_(permutation, permutation)],
        means=fitted.means_[permutation],
        covars=fitted.covars_[permutation],
    )
    permuted.feature_names_ = tuple(observations.columns)
    permuted_state = permuted.filter_proba(observations).to_numpy().argmax(axis=1)
    assert adjusted_rand_score(inferred, permuted_state) == pytest.approx(1.0)


def test_kelly_reads_only_15_already_realized_same_state_outcomes() -> None:
    index = pd.bdate_range("2024-01-02", periods=21)
    states = pd.Series(0, index=index, dtype="Int64")
    outcomes = pd.Series(
        [0.02] * 9 + [-0.01] * 6 + [100.0] * 6,
        index=index,
    )
    details = kelly_position_details(states, outcomes, lookback_same_state=15)
    assert details.iloc[14]["kelly_position"] == 0.0
    # 9/15 wins, average gain/loss odds=2: f=.6-.4/2=.4.
    assert details.iloc[15]["kelly_position"] == pytest.approx(0.4)
    assert details.iloc[15]["prior_same_state_samples"] == 15

    changed = outcomes.copy()
    changed.iloc[15:] = -1_000_000.0
    changed_details = kelly_position_details(states, changed, lookback_same_state=15)
    pdt.assert_series_equal(
        details["kelly_position"].iloc[:16],
        changed_details["kelly_position"].iloc[:16],
    )


def test_post_hoc_position_patches_are_explicit_and_bounded() -> None:
    index = pd.bdate_range("2024-01-02", periods=4)
    base = pd.Series([0.0, 0.5, 0.5001, 0.8], index=index, name="position")
    disabled = apply_microcap_opening_patch(base)
    pdt.assert_series_equal(disabled, base)
    patched = apply_microcap_opening_patch(base, enabled=True)
    np.testing.assert_allclose(patched, [0.0, 0.5, 1.0, 1.0])

    trigger = pd.Series([False, True, False, True], index=index)
    bottom = apply_bottom_recovery_patch(base, trigger, enabled=True)
    np.testing.assert_allclose(bottom, [0.0, 0.75, 0.5001, 0.9])
    with pytest.raises(ValueError, match="explicit point-in-time trigger"):
        apply_bottom_recovery_patch(base, None, enabled=True)


@pytest.mark.parametrize("representation,expected_dimensions", [("raw", 23), ("pca", 16)])
def test_raw_and_pca_baselines_join_six_soft_states(
    representation: str, expected_dimensions: int
) -> None:
    features = _feature_frame(180)
    soft = _soft_states(features.index)
    config = ZhongyouModelConfig(
        representation=representation,  # type: ignore[arg-type]
        ghmm=GHMMConfig(
            n_states=2,
            restarts=1,
            max_iterations=20,
            seed=4,
        ),
    )
    model = ZhongyouStateModel(config).fit(
        features,
        soft,
        train_start=features.index[0],
        train_end=features.index[139],
    )
    observations = model.observation_frame(features, soft)
    assert observations.shape[1] == expected_dimensions
    probabilities = model.filter_probabilities(features, soft)
    np.testing.assert_allclose(probabilities.dropna().sum(axis=1), 1.0, atol=1e-10)


def test_state_model_selects_ghmm_restart_on_explicit_validation() -> None:
    features = _feature_frame(190)
    soft = _soft_states(features.index)
    train_index = features.index[:100]
    validation_index = features.index[120:170]
    model = ZhongyouStateModel(
        ZhongyouModelConfig(
            representation="raw",
            ghmm=GHMMConfig(n_states=2, restarts=2, max_iterations=20, seed=3),
        )
    ).fit(
        features,
        soft,
        train_start=train_index.min(),
        train_end=train_index.max(),
        train_index=train_index,
        validation_index=validation_index,
    )

    assert model.validation_observations_ is not None
    assert len(model.validation_observations_) == len(validation_index)
    assert model.metadata["ghmm"]["selection_scope"] == (
        "explicit_walk_forward_validation_log_likelihood"
    )
