import copy

import numpy as np
import pytest

from histvae import (
    HistVAE,
    aggregate_latent_views,
    empirical_reference_percentile,
    euclidean_reference_knn,
    reference_knn_from_distances,
)


def make_config(dimension):
    bins = 4
    return {
        "device": "cpu",
        "num_points": 4,
        "in_channels": 1,
        "in_dims": dimension,
        "min_vals": [0.0] * dimension,
        "max_vals": [1.0] * dimension,
        "bins": bins,
        "histogram_mode": "probability_mass",
        "out_of_range_policy": "clip",
        "value_transform": "none",
        "group_coordinate_mode": "none",
        "train_sampling_mode": "random",
        "eval_sampling_mode": "full",
        "train_target_sampling_mode": "full",
        "eval_target_sampling_mode": "full",
        "num_workers": 0,
        "pin_memory": False,
        "latent_dim": 2,
        "hidden_dims": [2],
        "dropout_conv": 0.0,
        "decoder_output_mode": "simplex_softmax",
        "reconstruction_loss": "forward_kl",
        "ot_loss": "none",
        "ot_weight": 0.0,
        "condition_mode": "none",
        "condition_dim": 0,
        "beta": 0.0,
        "latent_kl_schedule": "constant",
        "latent_kl_warmup_epochs": 0,
        "num_classes": 2,
        "num_layers": 1,
        "hidden_head": 2,
        "dropout_head": 0.0,
        "frozen": False,
        "use_pretrain_loss": False,
        "optimizer": "radam",
        "batch_size": 2,
        "epochs": 1,
        "lr": 0.001,
        "weight_decay": 0.0,
        "transform": False,
        "accum_grad": 1,
        "clip_grad": 1.0,
        "save_model_every": 0,
        "patience": 0,
        "early_stop_mode": "min",
        "pretrain_monitor": "test_recon",
        "active_latent_threshold": 0.01,
        "log_every": 10,
    }


def make_grouped_data(dimension):
    first_axis = np.array(
        [0.05, 0.12, 0.20, 0.28, 0.36, 0.44, 0.52, 0.60, 0.68, 0.76],
        dtype=np.float32,
    )
    second_axis = np.array(
        [0.18, 0.25, 0.32, 0.39, 0.46, 0.53, 0.60, 0.67, 0.74, 0.81],
        dtype=np.float32,
    )
    if dimension == 1:
        group_a = first_axis[:, None]
        group_b = np.clip(first_axis[:, None] + 0.12, 0.0, 1.0)
    else:
        group_a = np.stack([first_axis, second_axis], axis=1)
        group_b = np.clip(group_a + np.array([0.12, -0.08]), 0.0, 1.0)
    data = np.concatenate([group_a, group_b]).astype(np.float32)
    groups = np.array(["a"] * len(group_a) + ["b"] * len(group_b))
    return data, groups


@pytest.mark.smoke
@pytest.mark.parametrize("dimension", [1, 2])
def test_multiview_latent_is_deterministic_and_dimension_agnostic(
        tmp_path, dimension
        ):
    config = make_config(dimension)
    data, groups = make_grouped_data(dimension)
    histvae = HistVAE(
        config=copy.deepcopy(config),
        outdir=str(tmp_path),
        exp_name=f"multiview-{dimension}d",
        seed=11,
    )
    histvae.prep_data(
        train_data=data,
        train_group=groups,
        test_data=data,
        test_group=groups,
    )
    histvae.prep_model("pretrain")

    first = histvae.get_multiview_latent(
        histvae.test_dataset,
        num_views=5,
        random_seed=123,
        return_histograms=True,
    )
    second = histvae.get_multiview_latent(
        histvae.test_dataset,
        num_views=5,
        random_seed=123,
        return_histograms=True,
    )

    assert first["mu"].shape == (2, 5, 2)
    assert first["logvar"].shape == (2, 5, 2)
    assert first["posterior_sd"].shape == (2, 5, 2)
    assert first["mu_mean"].shape == (2, 2)
    assert first["mu_sd"].shape == (2, 2)
    assert first["mu_rms_distance_to_mean"].shape == (2,)
    assert first["histogram"].shape == (2, 5, 1, *([4] * dimension))

    for key in (
        "mu",
        "logvar",
        "posterior_sd",
        "mu_mean",
        "mu_sd",
        "mu_rms_distance_to_mean",
        "histogram",
    ):
        np.testing.assert_allclose(first[key], second[key], rtol=0, atol=0)

    summary = aggregate_latent_views(first["mu"])
    np.testing.assert_allclose(first["mu_mean"], summary["mean"])
    np.testing.assert_allclose(first["mu_sd"], summary["sd"])
    np.testing.assert_allclose(
        first["mu_rms_distance_to_mean"],
        summary["rms_distance_to_mean"],
    )
    np.testing.assert_allclose(
        first["histogram"].reshape(2, 5, -1).sum(axis=2),
        1.0,
        rtol=1e-6,
        atol=1e-6,
    )


@pytest.mark.smoke
def test_multiview_child_seeds_do_not_depend_on_indices_order(tmp_path):
    config = make_config(1)
    data, groups = make_grouped_data(1)
    histvae = HistVAE(
        config=config,
        outdir=str(tmp_path),
        exp_name="multiview-order",
        seed=12,
    )
    histvae.prep_data(
        train_data=data,
        train_group=groups,
        test_data=data,
        test_group=groups,
    )
    histvae.prep_model("pretrain")

    forward = histvae.get_multiview_latent(
        histvae.test_dataset,
        indices=[0, 1],
        num_views=4,
        random_seed=456,
        return_histograms=True,
    )
    reverse = histvae.get_multiview_latent(
        histvae.test_dataset,
        indices=[1, 0],
        num_views=4,
        random_seed=456,
        return_histograms=True,
    )

    np.testing.assert_allclose(forward["mu"], reverse["mu"][::-1])
    np.testing.assert_allclose(
        forward["histogram"], reverse["histogram"][::-1]
    )


@pytest.mark.smoke
def test_reference_knn_supports_self_exclusion_and_deterministic_ties():
    distance_matrix = np.array([
        [0.0, 1.0, 1.0],
        [2.0, 0.0, 3.0],
    ])

    result = reference_knn_from_distances(
        distance_matrix,
        k=2,
        exclude_reference_indices=np.array([0, 1]),
    )

    np.testing.assert_array_equal(
        result["neighbor_indices"],
        np.array([[1, 2], [0, 2]]),
    )
    np.testing.assert_allclose(result["score"], np.array([1.0, 2.5]))


@pytest.mark.smoke
def test_euclidean_knn_and_empirical_percentile_are_generic():
    values = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [3.0, 0.0],
        [8.0, 0.0],
    ])
    reference = values[:3]
    exclusions = np.array([0, 1, 2, -1])

    result = euclidean_reference_knn(
        values,
        reference,
        k=1,
        exclude_reference_indices=exclusions,
    )
    np.testing.assert_allclose(result["score"], np.array([1.0, 1.0, 2.0, 5.0]))

    percentile = empirical_reference_percentile(
        result["score"],
        reference_indices=np.array([0, 1, 2]),
        leave_one_out=True,
    )
    assert percentile.shape == (4,)
    assert percentile[-1] == pytest.approx(1.0)
    assert percentile[2] > percentile[0]


@pytest.mark.smoke
def test_multiview_and_reference_validation_is_strict(tmp_path):
    with pytest.raises(ValueError, match="mu_views must have shape"):
        aggregate_latent_views(np.zeros((2, 3)))

    with pytest.raises(ValueError, match="fewer than k"):
        reference_knn_from_distances(
            np.array([[0.0, 1.0]]),
            k=2,
            exclude_reference_indices=np.array([0]),
        )

    config = make_config(1)
    data, groups = make_grouped_data(1)
    histvae = HistVAE(
        config=config,
        outdir=str(tmp_path),
        exp_name="multiview-validation",
    )
    histvae.prep_data(
        train_data=data,
        train_group=groups,
        test_data=data,
        test_group=groups,
    )
    histvae.prep_model("pretrain")

    with pytest.raises(ValueError, match="num_views must be a positive integer"):
        histvae.get_multiview_latent(num_views=0)
    with pytest.raises(ValueError, match="random_seed must be a non-negative"):
        histvae.get_multiview_latent(random_seed=-1)
