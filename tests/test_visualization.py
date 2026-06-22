from pathlib import Path

import numpy as np
import pytest
import torch

from histvae import HistVAE
from histvae.data_handler import Histogram
from histvae.visualization import (
    histogram_bin_edges,
    plot_hist,
    prepare_histogram_for_plot,
)


def make_visualization_config():
    return {
        "device": "cpu",
        "num_points": 3,
        "in_channels": 1,
        "in_dims": 1,
        "max_vals": [99.0],
        "bins": 4,
        "histogram_mode": "probability_mass",
        "out_of_range_policy": "clip",
        "value_transform": "log1p",
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
        "lr": 1e-3,
        "weight_decay": 0.0,
        "transform": False,
        "accum_grad": 1,
        "clip_grad": 1.0,
        "save_model_every": 0,
        "patience": 0,
        "early_stop_mode": "min",
        "pretrain_monitor": "test_recon",
        "finetune_monitor": "test_loss",
        "active_latent_threshold": 0.01,
        "log_every": 1,
    }


def make_visualization_data():
    data = np.array(
        [
            [0.0], [1.0], [4.0], [9.0], [30.0], [90.0],
            [0.5], [2.0], [8.0], [20.0], [50.0], [99.0],
        ],
        dtype=np.float32,
    )
    group = np.array(["a"] * 6 + ["b"] * 6)
    return data, group


@pytest.mark.smoke
def test_log1p_histogram_edges_are_inverted_to_raw_space():
    expected = np.array([0.0, 9.0, 99.0])

    direct = histogram_bin_edges(
        max_vals=[99.0],
        bins=2,
        value_transform="log1p",
        coordinate_space="raw",
    )
    histogram = Histogram(
        dimension=1,
        max_vals=[99.0],
        bins_per_dim=2,
        histogram_mode="probability_mass",
        value_transform="log1p",
    )

    np.testing.assert_allclose(direct[0], expected, rtol=0, atol=1e-12)
    np.testing.assert_allclose(
        histogram.get_bin_edges("raw")[0], expected, rtol=0, atol=1e-12
    )
    np.testing.assert_allclose(
        histogram.get_bin_edges("transformed")[0],
        np.linspace(0.0, np.log1p(99.0), 3),
        rtol=0,
        atol=1e-12,
    )


@pytest.mark.smoke
def test_raw_density_integrates_back_to_probability_mass():
    mass = np.array([0.25, 0.75], dtype=np.float64)
    raw_edges = [np.array([0.0, 9.0, 99.0])]

    density, normalized_edges = prepare_histogram_for_plot(
        mass,
        bin_edges=raw_edges,
        value_mode="density",
    )

    recovered_mass = np.sum(
        density * np.diff(normalized_edges[0])
    )
    assert recovered_mass == pytest.approx(1.0)
    np.testing.assert_allclose(
        density,
        np.array([0.25 / 9.0, 0.75 / 90.0]),
    )


@pytest.mark.smoke
def test_two_dimensional_raw_density_preserves_total_mass():
    mass = np.array(
        [[0.1, 0.2], [0.3, 0.4]],
        dtype=np.float64,
    )
    raw_edges = histogram_bin_edges(
        max_vals=[99.0, 8.0],
        bins=[2, 2],
        value_transform="log1p",
        coordinate_space="raw",
    )

    density, normalized_edges = prepare_histogram_for_plot(
        mass,
        bin_edges=raw_edges,
        value_mode="density",
    )

    raw_area = np.multiply.outer(
        np.diff(normalized_edges[0]),
        np.diff(normalized_edges[1]),
    )
    assert np.sum(density * raw_area) == pytest.approx(1.0)
    np.testing.assert_allclose(raw_edges[0], np.array([0.0, 9.0, 99.0]))
    np.testing.assert_allclose(raw_edges[1], np.array([0.0, 2.0, 8.0]))


@pytest.mark.smoke
def test_plot_hist_uses_explicit_raw_edges(tmp_path):
    output = tmp_path / "raw_histogram.png"
    raw_edges = histogram_bin_edges(
        max_vals=[99.0],
        bins=2,
        value_transform="log1p",
        coordinate_space="raw",
    )

    figure, axes = plot_hist(
        [np.array([0.25, 0.75])],
        output=str(output),
        bin_edges=raw_edges,
        value_mode="density",
        xlabel="FITC_Sum",
        ylabel="probability density per raw unit",
        close=False,
    )

    stairs = axes[0].patches[0].get_data()
    np.testing.assert_allclose(stairs.edges, raw_edges[0])
    assert axes[0].get_xlim() == pytest.approx((0.0, 99.0))
    assert axes[0].get_xscale() == "linear"
    assert axes[0].get_xlabel() == "FITC_Sum"
    assert output.exists()
    figure.clf()


@pytest.mark.smoke
def test_histvae_reconstruction_plot_is_raw_space_and_deterministic(tmp_path):
    config = make_visualization_config()
    data, group = make_visualization_data()
    histvae = HistVAE(
        config=config,
        outdir=str(tmp_path),
        exp_name="raw-reconstruction",
        seed=7,
    )
    histvae.prep_data(
        train_data=data,
        train_group=group,
    )
    histvae.prep_model("pretrain")

    output = tmp_path / "reconstruction.png"
    result0, figure, axes = histvae.plot_reconstruction(
        dataset=histvae.train_dataset,
        indices=[0, 1],
        input_mode="full",
        coordinate_space="raw",
        output=str(output),
        close=False,
        axis_labels=["FITC_Sum"],
    )
    result1 = histvae.get_reconstruction(
        dataset=histvae.train_dataset,
        indices=[0, 1],
        input_mode="full",
    )

    np.testing.assert_allclose(result0["bin_edges"][0][-1], 99.0)
    assert result0["plot_value_mode"] == "density"
    np.testing.assert_allclose(
        result0["reconstruction"], result1["reconstruction"], rtol=0, atol=0
    )
    np.testing.assert_allclose(
        result0["reconstruction"].sum(axis=(-1, -2)),
        np.ones(2),
        rtol=1e-6,
        atol=1e-6,
    )
    assert np.all(np.isfinite(result0["metric_values"]))
    assert output.exists()
    for axis in axes:
        assert axis.get_xlim() == pytest.approx((0.0, 99.0))
    figure.clf()


@pytest.mark.smoke
def test_sampled_reconstruction_view_is_reproducible(tmp_path):
    config = make_visualization_config()
    data, group = make_visualization_data()
    histvae = HistVAE(
        config=config,
        outdir=str(tmp_path),
        exp_name="sampled-reconstruction",
        seed=7,
    )
    histvae.prep_data(
        train_data=data,
        train_group=group,
    )
    histvae.prep_model("pretrain")

    first = histvae.get_reconstruction(
        dataset=histvae.train_dataset,
        indices=[0, 1],
        input_mode="sampled",
        random_seed=123,
    )
    second = histvae.get_reconstruction(
        dataset=histvae.train_dataset,
        indices=[0, 1],
        input_mode="sampled",
        random_seed=123,
    )

    np.testing.assert_allclose(first["input"], second["input"], rtol=0, atol=0)
    np.testing.assert_allclose(
        first["reconstruction"], second["reconstruction"], rtol=0, atol=0
    )
    np.testing.assert_allclose(
        first["target"].sum(axis=(-1, -2)),
        np.ones(2),
        rtol=1e-6,
        atol=1e-6,
    )


@pytest.mark.smoke
def test_plot_hist_supports_two_dimensional_raw_coordinates(tmp_path):
    output = tmp_path / "raw_histogram_2d.png"
    mass = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64)
    raw_edges = histogram_bin_edges(
        max_vals=[99.0, 8.0],
        bins=[2, 2],
        value_transform="log1p",
        coordinate_space="raw",
    )

    figure, axes = plot_hist(
        [mass],
        output=str(output),
        bin_edges=raw_edges,
        value_mode="density",
        xlabel="raw x",
        ylabel="raw y",
        close=False,
    )

    assert output.exists()
    assert axes[0].get_xlim() == pytest.approx((0.0, 99.0))
    assert axes[0].get_ylim() == pytest.approx((0.0, 8.0))
    figure.clf()


@pytest.mark.smoke
def test_conditioned_reconstruction_uses_group_condition(tmp_path):
    config = make_visualization_config()
    config.update({
        "condition_mode": "decoder",
        "condition_dim": 2,
    })
    data, group = make_visualization_data()
    condition = np.array(
        [[1.0, 0.0]] * 6 + [[0.0, 1.0]] * 6,
        dtype=np.float32,
    )
    histvae = HistVAE(
        config=config,
        outdir=str(tmp_path),
        exp_name="conditioned-reconstruction",
        seed=7,
    )
    histvae.prep_data(
        train_data=data,
        train_group=group,
        train_condition=condition,
    )
    histvae.prep_model("pretrain")

    result = histvae.get_reconstruction(
        dataset=histvae.train_dataset,
        indices=[0, 1],
        input_mode="full",
    )

    assert result["reconstruction"].shape == (2, 1, 4)
    np.testing.assert_allclose(
        result["reconstruction"].sum(axis=(-1, -2)),
        np.ones(2),
        rtol=1e-6,
        atol=1e-6,
    )
