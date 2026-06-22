import numpy as np
import pytest
import torch

from histvae import HistVAE
from histvae.core import calc_hist
from histvae.data_handler import Histogram, PointHistDataset


@pytest.mark.smoke
def test_histogram_count_mode_remains_the_default():
    data = np.array([[0.1], [0.2], [0.6], [0.9]], dtype=np.float32)

    hist = Histogram(dimension=1, max_vals=[1.0], bins_per_dim=2).compute(data)

    np.testing.assert_array_equal(hist, np.array([2, 2]))


@pytest.mark.smoke
def test_density_mode_removes_replication_intensity():
    data = np.array([[0.1], [0.2], [0.6], [0.9]], dtype=np.float32)
    repeated = np.repeat(data, repeats=5, axis=0)

    count_hist = Histogram(
        dimension=1,
        max_vals=[1.0],
        bins_per_dim=2,
        histogram_mode="count",
    )
    density_hist = Histogram(
        dimension=1,
        max_vals=[1.0],
        bins_per_dim=2,
        histogram_mode="density",
    )

    np.testing.assert_array_equal(
        count_hist.compute(repeated), count_hist.compute(data) * 5
    )
    np.testing.assert_allclose(
        density_hist.compute(repeated), density_hist.compute(data)
    )
    np.testing.assert_allclose(density_hist.compute(data).sum() * 0.5, 1.0)


@pytest.mark.smoke
def test_probability_mass_is_bounded_and_replication_invariant():
    data = np.array([[0.1], [0.2], [0.6], [0.9]], dtype=np.float32)
    repeated = np.repeat(data, repeats=5, axis=0)
    histogram = Histogram(
        dimension=1,
        max_vals=[1.0],
        bins_per_dim=2,
        histogram_mode="probability_mass",
    )

    mass = histogram.compute(data)

    np.testing.assert_allclose(mass, [0.5, 0.5])
    np.testing.assert_allclose(histogram.compute(repeated), mass)
    assert mass.min() >= 0
    assert mass.max() <= 1
    np.testing.assert_allclose(mass.sum(), 1.0)


@pytest.mark.smoke
def test_density_mode_removes_group_size_from_dataset_normalization():
    base = np.array([[0.1], [0.2], [0.6], [0.9]], dtype=np.float32)
    data = np.vstack([base, np.tile(base, (5, 1))])
    group = np.array(["small"] * len(base) + ["large"] * (len(base) * 5))

    count_dataset = PointHistDataset(
        data=data,
        group=group,
        max_vals=[1.0],
        num_points=4,
        bins=2,
        histogram_mode="count",
    )
    density_dataset = PointHistDataset(
        data=data,
        group=group,
        max_vals=[1.0],
        num_points=4,
        bins=2,
        histogram_mode="density",
    )

    assert count_dataset.log1p_max["small"] != count_dataset.log1p_max["large"]
    np.testing.assert_allclose(
        density_dataset.log1p_max["small"],
        density_dataset.log1p_max["large"],
    )


@pytest.mark.smoke
def test_probability_mass_dataset_views_stay_in_decoder_range():
    data = np.array(
        [[0.05], [0.1], [0.2], [0.3], [0.6], [0.7], [0.8], [0.9]],
        dtype=np.float32,
    )
    group = np.array(["sample"] * len(data))
    dataset = PointHistDataset(
        data=data,
        group=group,
        max_vals=[1.0],
        num_points=4,
        bins=4,
        histogram_mode="probability_mass",
    )

    for _ in range(10):
        (hist0, hist1), _ = dataset[0]
        for hist in (hist0, hist1):
            assert float(hist.min()) >= 0
            assert float(hist.max()) <= 1
            torch.testing.assert_close(hist.sum(), torch.tensor(1.0))


@pytest.mark.smoke
def test_invalid_histogram_mode_is_rejected():
    with pytest.raises(ValueError, match="Unsupported histogram_mode"):
        Histogram(
            dimension=1,
            max_vals=[1.0],
            bins_per_dim=4,
            histogram_mode="probability",
        )


@pytest.mark.smoke
def test_density_mode_rejects_a_range_with_no_observations():
    hist = Histogram(
        dimension=1,
        max_vals=[1.0],
        bins_per_dim=4,
        histogram_mode="density",
    )

    with pytest.raises(ValueError, match="configured range"):
        hist.compute(np.array([[2.0], [3.0]], dtype=np.float32))


@pytest.mark.smoke
def test_out_of_range_policy_is_explicit():
    data = np.array([[-1.0], [0.25], [2.0]], dtype=np.float32)
    dropped = Histogram(
        dimension=1,
        max_vals=[1.0],
        bins_per_dim=2,
        histogram_mode="probability_mass",
        out_of_range_policy="drop",
    ).compute(data)
    clipped = Histogram(
        dimension=1,
        max_vals=[1.0],
        bins_per_dim=2,
        histogram_mode="probability_mass",
        out_of_range_policy="clip",
    ).compute(data)

    np.testing.assert_allclose(dropped, [1.0, 0.0])
    np.testing.assert_allclose(clipped, [2 / 3, 1 / 3])
    with pytest.raises(ValueError, match="outside"):
        Histogram(
            dimension=1,
            max_vals=[1.0],
            bins_per_dim=2,
            out_of_range_policy="error",
        ).compute(data)


@pytest.mark.smoke
def test_log1p_value_transform_changes_bin_geometry():
    data = np.array([[0.0], [1.0], [10.0], [100.0]], dtype=np.float32)
    linear = Histogram(
        dimension=1,
        max_vals=[100.0],
        bins_per_dim=2,
        histogram_mode="count",
        value_transform="none",
    ).compute(data)
    logspaced = Histogram(
        dimension=1,
        max_vals=[100.0],
        bins_per_dim=2,
        histogram_mode="count",
        value_transform="log1p",
    ).compute(data)

    np.testing.assert_array_equal(linear, [3, 1])
    np.testing.assert_array_equal(logspaced, [2, 2])


@pytest.mark.smoke
def test_legacy_calc_hist_accepts_the_same_strict_mode():
    data = np.array([[0.1], [0.2], [0.6], [0.9]], dtype=np.float32)

    count = calc_hist(data, bins=2)
    density = calc_hist(data, bins=2, histogram_mode="density")

    np.testing.assert_array_equal(count, np.array([2, 2]))
    np.testing.assert_allclose(density * np.diff([0.1, 0.5, 0.9]), [0.5, 0.5])


@pytest.mark.smoke
def test_histvae_1d_density_runtime_override_reaches_model(tmp_path):
    config = {
        "device": "cpu",
        "num_points": 4,
        "in_channels": 1,
        "in_dims": 1,
        "max_vals": [1.0],
        "bins": 4,
        "histogram_mode": "count",
        "num_workers": 0,
        "pin_memory": False,
        "latent_dim": 2,
        "hidden_dims": [1],
        "dropout_conv": 0.0,
        "beta": 1.0,
        "num_classes": 2,
        "num_layers": 1,
        "hidden_head": 2,
        "dropout_head": 0.0,
        "frozen": False,
        "use_pretrain_loss": False,
        "batch_size": 2,
        "epochs": 1,
        "lr": 0.001,
        "optimizer": "radam",
        "weight_decay": 0.0,
        "transform": False,
        "accum_grad": 1,
        "clip_grad": 1.0,
        "save_model_every": 0,
        "patience": 0,
        "log_every": 1,
    }
    data = np.array(
        [[0.1], [0.2], [0.3], [0.4], [0.6], [0.7], [0.8], [0.9]],
        dtype=np.float32,
    )
    group = np.array(["sample_a"] * 4 + ["sample_b"] * 4)

    histvae = HistVAE(config=config, outdir=str(tmp_path), exp_name="density-1d")
    histvae.prep_data(
        train_data=data,
        train_group=group,
        histogram_mode="density",
        out_of_range_policy="clip",
        value_transform="log1p",
    )
    histvae.prep_model("pretrain")

    assert histvae.config["histogram_mode"] == "density"
    assert histvae.config["out_of_range_policy"] == "clip"
    assert histvae.config["value_transform"] == "log1p"
    assert histvae.train_dataset.histogram_mode == "density"
    assert histvae.train_dataset.hist.out_of_range_policy == "clip"
    assert histvae.train_dataset.hist.value_transform == "log1p"
    (hist0, hist1), _ = next(iter(histvae.train_loader))
    assert hist0.shape == (2, 1, 4)
    assert hist1.shape == (2, 1, 4)
    assert torch.isfinite(hist0).all()
    recon, _, _ = histvae.model(hist0)
    assert recon.shape == hist0.shape


@pytest.mark.smoke
def test_histvae_records_count_for_legacy_configs_without_the_new_key(tmp_path):
    config = {
        "device": "cpu",
        "in_channels": 1,
        "in_dims": 1,
        "bins": 4,
    }

    HistVAE(config=config, outdir=str(tmp_path), exp_name="legacy-config")

    assert config["histogram_mode"] == "count"
