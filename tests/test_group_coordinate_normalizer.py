import numpy as np
import pytest
import torch
import yaml

from histvae import (
    AxisPreprocessingSpec,
    GroupCoordinateNormalizer,
    HistogramPreprocessor,
    HistVAE,
)
from histvae.data_handler import PointHistDataset


def make_histvae_config(group_coordinate_mode="none"):
    return {
        "device": "cpu",
        "num_points": 2,
        "in_channels": 1,
        "in_dims": 1,
        "min_vals": [-1.0],
        "max_vals": [1.0],
        "bins": 8,
        "histogram_mode": "probability_mass",
        "out_of_range_policy": "clip",
        "value_transform": "none",
        "group_coordinate_mode": group_coordinate_mode,
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


def fixed_preprocessor(lower, upper, bins=8, transform="none"):
    return HistogramPreprocessor(
        [
            AxisPreprocessingSpec(
                name="signal",
                transform=transform,
                lower_mode="fixed",
                lower_value=float(lower),
                upper_mode="fixed",
                upper_value=float(upper),
            )
        ],
        bins=bins,
        histogram_mode="probability_mass",
        tail_policy="clip",
    )


@pytest.mark.smoke
def test_raw_median_center_is_additive_invariant_and_preserves_width():
    data = np.array(
        [[10.0], [12.0], [14.0], [20.0], [100.0], [104.0], [108.0], [112.0]]
    )
    group = np.array(["a"] * 4 + ["b"] * 4)
    normalizer = GroupCoordinateNormalizer(
        mode="raw_median_center", dimension=1
    )

    centered, statistics = normalizer.transform(
        data, group, return_statistics=True
    )
    shifted = normalizer.transform(data + 37.0, group)

    np.testing.assert_array_equal(centered, shifted)
    assert np.any(centered < 0)
    for group_value in ("a", "b"):
        selected = group == group_value
        assert np.ptp(centered[selected, 0]) == pytest.approx(
            np.ptp(data[selected, 0])
        )
        assert np.median(centered[selected, 0]) == pytest.approx(0.0)

    assert statistics["event_count"].tolist() == [4, 4]
    assert statistics["raw_median_0"].tolist() == pytest.approx([13.0, 106.0])
    assert statistics["mode"].tolist() == ["raw_median_center"] * 2


@pytest.mark.smoke
def test_raw_median_ratio_is_multiplicative_invariant_and_strict():
    data = np.array(
        [[2.0], [4.0], [4.0], [8.0], [16.0], [32.0], [32.0], [64.0]]
    )
    group = np.array(["a"] * 4 + ["b"] * 4)
    normalizer = GroupCoordinateNormalizer(
        mode="raw_median_ratio", dimension=1
    )

    ratio = normalizer.transform(data, group)
    scaled = normalizer.transform(data * 8.0, group)
    np.testing.assert_array_equal(ratio, scaled)
    assert np.median(ratio[group == "a", 0]) == pytest.approx(0.0)
    assert np.median(ratio[group == "b", 0]) == pytest.approx(0.0)

    invalid = np.array([[-1.0], [0.0], [1.0]])
    with pytest.raises(ValueError, match="strictly positive full-group median"):
        normalizer.transform(invalid, np.array(["bad"] * 3))


@pytest.mark.smoke
def test_log_median_center_matches_explicit_full_group_calculation():
    data = np.array([[0.0], [3.0], [8.0], [15.0]])
    group = np.array(["a"] * 4)
    normalizer = GroupCoordinateNormalizer(
        mode="log_median_center", dimension=1
    )

    transformed = normalizer.transform(data, group)
    expected = np.log1p(data) - np.median(np.log1p(data), axis=0)
    np.testing.assert_allclose(transformed, expected, rtol=0, atol=0)

    with pytest.raises(ValueError, match="requires non-negative raw values"):
        normalizer.transform(np.array([[-1.0], [1.0]]), np.array(["b", "b"]))


@pytest.mark.smoke
def test_group_coordinate_state_round_trip_is_safe_and_exact(tmp_path):
    normalizer = GroupCoordinateNormalizer(
        mode="raw_median_ratio", dimension=2
    )
    output = normalizer.save(tmp_path / "group_coordinate.yaml")

    with output.open("r", encoding="utf-8") as handle:
        saved = yaml.safe_load(handle)
    assert saved["schema_version"] == 1
    assert saved["mode"] == "raw_median_ratio"

    restored = GroupCoordinateNormalizer.load(output)
    assert restored.state_dict() == normalizer.state_dict()
    assert restored.state_sha256 == normalizer.state_sha256

    tampered = restored.state_dict()
    tampered["mode"] = "raw_median_center"
    with pytest.raises(ValueError, match="state_sha256"):
        GroupCoordinateNormalizer.from_state_dict(tampered)

    with pytest.raises(ValueError, match="Unsupported group_coordinate_mode"):
        GroupCoordinateNormalizer(mode="median", dimension=1)


@pytest.mark.smoke
def test_random_input_and_full_target_share_the_full_group_median(monkeypatch):
    raw_data = np.array([[0.0], [0.0], [10.0], [100.0]])
    group = np.array(["sample"] * 4)
    normalizer = GroupCoordinateNormalizer(
        mode="raw_median_center", dimension=1
    )
    normalized = normalizer.transform(raw_data, group)
    preprocessor = fixed_preprocessor(-10.0, 100.0, bins=11).fit(
        normalized, group
    )
    dataset = PointHistDataset(
        data=raw_data,
        group=group,
        num_points=2,
        sampling_mode="random",
        target_sampling_mode="full",
        transform=False,
        histogram_preprocessor=preprocessor,
        group_coordinate_normalizer=normalizer,
    )

    monkeypatch.setattr(
        np.random,
        "choice",
        lambda population, size, replace: np.array([0, 1]),
    )
    (full_target, sampled_input), _ = dataset[0]

    expected_full = torch.tensor(
        preprocessor.compute(normalized), dtype=torch.float32
    ).unsqueeze(0)
    expected_sampled = torch.tensor(
        preprocessor.compute(np.array([[-5.0], [-5.0]])),
        dtype=torch.float32,
    ).unsqueeze(0)
    local_median_sampled = torch.tensor(
        preprocessor.compute(np.array([[0.0], [0.0]])),
        dtype=torch.float32,
    ).unsqueeze(0)

    torch.testing.assert_close(full_target, expected_full, rtol=0, atol=0)
    torch.testing.assert_close(sampled_input, expected_sampled, rtol=0, atol=0)
    assert not torch.equal(sampled_input, local_median_sampled)
    assert dataset.group_coordinate_statistics.loc[0, "raw_median_0"] == 5.0
    np.testing.assert_allclose(dataset.data[:, 0], [-5.0, -5.0, 5.0, 95.0])


@pytest.mark.smoke
def test_histvae_replays_coordinate_and_histogram_states_exactly(tmp_path):
    train_data = np.array(
        [[10.0], [12.0], [20.0], [22.0], [100.0], [104.0], [112.0], [116.0]],
        dtype=np.float32,
    )
    train_group = np.array(["a"] * 4 + ["b"] * 4)
    test_data = np.array([[50.0], [54.0], [60.0], [66.0]], dtype=np.float32)
    test_group = np.array(["c"] * 4)

    normalizer = GroupCoordinateNormalizer(
        mode="raw_median_center", dimension=1
    )
    normalized_train = normalizer.transform(train_data, train_group)
    preprocessor = fixed_preprocessor(-10.0, 10.0).fit(
        normalized_train, train_group
    )

    model = HistVAE(
        config=make_histvae_config("raw_median_center"),
        outdir=str(tmp_path),
        exp_name="coordinate-replay",
        histogram_preprocessor=preprocessor,
        group_coordinate_normalizer=normalizer,
    )
    model.prep_data(
        train_data=train_data,
        train_group=train_group,
        test_data=test_data,
        test_group=test_group,
    )

    assert model.config["group_coordinate_mode"] == "raw_median_center"
    assert model.config["group_coordinate_normalizer_state"]["state_sha256"] == (
        normalizer.state_sha256
    )
    assert model.train_dataset.group_coordinate_normalizer is normalizer
    np.testing.assert_allclose(model.train_dataset.data, normalized_train)
    assert model.group_coordinate_statistics["test"].loc[0, "raw_median_0"] == 57.0
    yaml.safe_load(yaml.safe_dump(model.config))

    restored = HistVAE(
        config=dict(model.config),
        outdir=str(tmp_path),
        exp_name="coordinate-restored",
    )
    restored.prep_data(
        train_data=train_data,
        train_group=train_group,
        test_data=test_data,
        test_group=test_group,
    )

    assert restored.group_coordinate_normalizer.state_sha256 == (
        normalizer.state_sha256
    )
    assert restored.histogram_preprocessor.state_sha256 == (
        preprocessor.state_sha256
    )
    for index in range(len(model.train_dataset)):
        torch.testing.assert_close(
            restored.train_dataset.get_full_histogram(index),
            model.train_dataset.get_full_histogram(index),
            rtol=0,
            atol=0,
        )


@pytest.mark.smoke
def test_strict_coordinate_pipeline_rejects_wrong_geometry_fit(tmp_path):
    train_data = np.array(
        [[10.0], [12.0], [20.0], [22.0]], dtype=np.float32
    )
    train_group = np.array(["a"] * 4)
    normalizer = GroupCoordinateNormalizer(
        mode="raw_median_center", dimension=1
    )

    model_without_geometry = HistVAE(
        config=make_histvae_config("raw_median_center"),
        outdir=str(tmp_path),
        exp_name="missing-geometry",
    )
    with pytest.raises(ValueError, match="requires a fitted HistogramPreprocessor"):
        model_without_geometry.prep_data(
            train_data=train_data,
            train_group=train_group,
        )

    wrong_preprocessor = fixed_preprocessor(0.0, 30.0).fit(
        train_data, train_group
    )
    wrong_model = HistVAE(
        config=make_histvae_config("raw_median_center"),
        outdir=str(tmp_path),
        exp_name="wrong-fit-data",
        histogram_preprocessor=wrong_preprocessor,
        group_coordinate_normalizer=normalizer,
    )
    with pytest.raises(ValueError, match="fitted on different training data"):
        wrong_model.prep_data(
            train_data=train_data,
            train_group=train_group,
        )

    log_preprocessor = fixed_preprocessor(0.0, 30.0, transform="log1p").fit(
        train_data, train_group
    )
    with pytest.raises(ValueError, match="axis transforms to be 'none'"):
        HistVAE(
            config=make_histvae_config("raw_median_center"),
            outdir=str(tmp_path),
            exp_name="double-transform",
            histogram_preprocessor=log_preprocessor,
            group_coordinate_normalizer=normalizer,
        )

    with pytest.raises(ValueError, match="conflicts with the explicit"):
        HistVAE(
            config=make_histvae_config("raw_median_ratio"),
            outdir=str(tmp_path),
            exp_name="coordinate-conflict",
            group_coordinate_normalizer=normalizer,
        )
