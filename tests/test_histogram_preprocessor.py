from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from histvae import AxisPreprocessingSpec, HistogramPreprocessor, HistVAE
from histvae.data_handler import PointHistDataset
from histvae.utils import load_config


def quantile_axis(
        weighting="event", transform="none", upper_quantile=0.75
        ):
    return AxisPreprocessingSpec(
        name="signal",
        transform=transform,
        lower_mode="fixed",
        lower_value=0.0,
        upper_mode="quantile",
        upper_value=None,
        upper_quantile=upper_quantile,
        quantile_weighting=weighting,
    )


@pytest.mark.smoke
def test_group_equal_quantile_does_not_let_large_groups_dominate():
    data = np.concatenate([
        np.ones((100, 1), dtype=np.float64),
        np.array([[100.0]], dtype=np.float64),
    ])
    group = np.array(["large"] * 100 + ["small"])

    event = HistogramPreprocessor(
        [quantile_axis("event")], bins=4
    ).fit(data, group)
    group_equal = HistogramPreprocessor(
        [quantile_axis("group_equal")], bins=4
    ).fit(data, group)

    assert event.max_vals == pytest.approx([1.0])
    assert group_equal.max_vals == pytest.approx([100.0])
    assert group_equal.fit_summary["n_groups"] == 2


@pytest.mark.smoke
def test_group_equal_quantile_requires_group_labels():
    data = np.array([[0.0], [1.0], [2.0]], dtype=np.float64)
    preprocessor = HistogramPreprocessor(
        [quantile_axis("group_equal", upper_quantile=0.9)]
    )

    with pytest.raises(ValueError, match="group is required"):
        preprocessor.fit(data)


@pytest.mark.smoke
def test_log1p_with_nonzero_raw_bounds_round_trips_and_exports_edges():
    spec = AxisPreprocessingSpec(
        name="signal",
        transform="log1p",
        lower_mode="fixed",
        lower_value=1.0,
        upper_mode="fixed",
        upper_value=99.0,
    )
    preprocessor = HistogramPreprocessor([spec], bins=2).fit(
        np.array([[1.0], [9.0], [99.0]])
    )

    transformed = preprocessor.transform(
        np.array([[0.0], [9.0], [200.0]])
    )
    raw = preprocessor.inverse_transform(transformed)
    np.testing.assert_allclose(raw[:, 0], [1.0, 9.0, 99.0])

    transformed_edges = preprocessor.get_bin_edges("transformed")[0]
    raw_edges = preprocessor.get_bin_edges("raw")[0]
    np.testing.assert_allclose(
        transformed_edges,
        np.linspace(np.log1p(1.0), np.log1p(99.0), 3),
    )
    np.testing.assert_allclose(raw_edges[[0, -1]], [1.0, 99.0])
    assert raw_edges[1] == pytest.approx(np.sqrt(200.0) - 1.0)


@pytest.mark.smoke
def test_error_tail_policy_rejects_out_of_range_application_data():
    spec = AxisPreprocessingSpec(
        name="signal",
        lower_mode="fixed",
        lower_value=0.0,
        upper_mode="fixed",
        upper_value=10.0,
    )
    preprocessor = HistogramPreprocessor(
        [spec], bins=4, tail_policy="error"
    ).fit(np.array([[0.0], [5.0], [10.0]]))

    with pytest.raises(ValueError, match="outside the fitted histogram range"):
        preprocessor.transform(np.array([[11.0]]))

    with pytest.raises(ValueError, match="outside the finite configured"):
        preprocessor.compute(np.array([[11.0]]))

    quantile_preprocessor = HistogramPreprocessor(
        [quantile_axis("event", upper_quantile=0.5)],
        bins=4,
        tail_policy="error",
    )
    with pytest.raises(ValueError, match="excludes training observations"):
        quantile_preprocessor.fit(np.array([[0.0], [1.0], [2.0]]))


@pytest.mark.smoke
def test_preprocessor_state_round_trip_is_safe_and_exact(tmp_path):
    data = np.array([[0.0], [1.0], [10.0], [100.0]], dtype=np.float64)
    group = np.array(["a", "a", "b", "b"])
    preprocessor = HistogramPreprocessor(
        [quantile_axis("event", transform="log1p", upper_quantile=0.9)],
        bins=4,
        histogram_mode="probability_mass",
        tail_policy="clip",
    ).fit(data, group)

    output = preprocessor.save(tmp_path / "preprocessing.yaml")
    with output.open("r", encoding="utf-8") as handle:
        assert yaml.safe_load(handle)["schema_version"] == 1

    restored = HistogramPreprocessor.load(output)
    assert restored.state_sha256 == preprocessor.state_sha256
    assert restored.state_dict() == preprocessor.state_dict()
    np.testing.assert_allclose(
        restored.compute(data), preprocessor.compute(data), rtol=0, atol=0
    )

    tampered = restored.state_dict()
    tampered["resolved_max_vals"][0] += 1.0
    with pytest.raises(ValueError, match="state_sha256"):
        HistogramPreprocessor.from_state_dict(tampered)

    missing_hash = restored.state_dict()
    missing_hash.pop("state_sha256")
    with pytest.raises(ValueError, match="missing keys"):
        HistogramPreprocessor.from_state_dict(missing_hash)


@pytest.mark.smoke
def test_axis_specific_transforms_work_in_two_dimensions():
    specs = [
        AxisPreprocessingSpec(
            name="signed",
            transform="none",
            lower_mode="fixed",
            lower_value=-1.0,
            upper_mode="fixed",
            upper_value=1.0,
        ),
        AxisPreprocessingSpec(
            name="positive",
            transform="log1p",
            lower_mode="fixed",
            lower_value=0.0,
            upper_mode="fixed",
            upper_value=99.0,
        ),
    ]
    preprocessor = HistogramPreprocessor(specs, bins=[2, 2]).fit(
        np.array([[-1.0, 0.0], [0.0, 9.0], [1.0, 99.0]])
    )

    raw_edges = preprocessor.get_bin_edges("raw")
    transformed_edges = preprocessor.get_bin_edges("transformed")
    np.testing.assert_allclose(raw_edges[0], [-1.0, 0.0, 1.0])
    np.testing.assert_allclose(raw_edges[1], [0.0, 9.0, 99.0])
    np.testing.assert_allclose(transformed_edges[0], [-1.0, 0.0, 1.0])
    np.testing.assert_allclose(
        transformed_edges[1], np.linspace(0.0, np.log1p(99.0), 3)
    )
    assert preprocessor.compute(
        np.array([[-1.0, 0.0], [1.0, 99.0]])
    ).shape == (2, 2)


@pytest.mark.smoke
def test_point_hist_dataset_uses_fitted_preprocessor_as_source_of_truth():
    data = np.array(
        [[0.0], [1.0], [9.0], [100.0], [0.0], [2.0], [8.0], [200.0]],
        dtype=np.float32,
    )
    group = np.array(["a"] * 4 + ["b"] * 4)
    spec = AxisPreprocessingSpec(
        name="signal",
        transform="log1p",
        lower_mode="fixed",
        lower_value=0.0,
        upper_mode="fixed",
        upper_value=99.0,
    )
    preprocessor = HistogramPreprocessor(
        [spec], bins=4, tail_policy="clip"
    ).fit(data, group)

    dataset = PointHistDataset(
        data=data,
        group=group,
        histogram_preprocessor=preprocessor,
        sampling_mode="full",
        transform=False,
    )

    assert dataset.histogram_preprocessor is preprocessor
    assert dataset.max_vals.tolist() == pytest.approx([99.0])
    assert dataset.hist.value_transforms == ("log1p",)
    for index in range(len(dataset)):
        torch.testing.assert_close(
            dataset.get_full_histogram(index).sum(), torch.tensor(1.0)
        )

    with pytest.raises(ValueError, match="bins conflicts"):
        PointHistDataset(
            data=data,
            group=group,
            bins=8,
            histogram_preprocessor=preprocessor,
        )


def make_histvae_config():
    return {
        "device": "cpu",
        "num_points": 2,
        "in_channels": 1,
        "in_dims": 1,
        "min_vals": [0.0],
        "max_vals": [1.0],
        "bins": 4,
        "histogram_mode": "probability_mass",
        "out_of_range_policy": "clip",
        "value_transform": "none",
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


@pytest.mark.smoke
def test_histvae_records_and_reuses_one_train_fitted_preprocessor(tmp_path):
    train_data = np.array([[0.0], [1.0], [5.0], [10.0]], dtype=np.float32)
    train_group = np.array(["a", "a", "b", "b"])
    test_data = np.array([[0.0], [20.0]], dtype=np.float32)
    test_group = np.array(["c", "c"])
    spec = AxisPreprocessingSpec(
        name="signal",
        transform="log1p",
        lower_mode="fixed",
        lower_value=0.0,
        upper_mode="fixed",
        upper_value=10.0,
    )
    preprocessor = HistogramPreprocessor(
        [spec], bins=4, tail_policy="clip"
    ).fit(train_data, train_group)

    model = HistVAE(
        config=make_histvae_config(),
        outdir=str(tmp_path),
        exp_name="preprocessor",
    )
    model.prep_data(
        train_data=train_data,
        train_group=train_group,
        test_data=test_data,
        test_group=test_group,
        histogram_preprocessor=preprocessor,
    )

    assert model.histogram_preprocessor.state_sha256 == preprocessor.state_sha256
    assert model.config["max_vals"] == pytest.approx([10.0])
    assert model.config["value_transform"] == "log1p"
    assert model.config["histogram_preprocessor_state"]["state_sha256"] == (
        preprocessor.state_sha256
    )
    assert model.config["histogram_preprocessor_diagnostics"]["test"][
        "fraction_any_above"
    ] == pytest.approx(0.5)
    assert model.train_dataset.histogram_preprocessor is preprocessor
    assert model.test_dataset.histogram_preprocessor is preprocessor
    yaml.safe_load(yaml.safe_dump(model.config))

    model.prep_model("pretrain")
    batch, _ = next(iter(model.test_loader))
    reconstruction, _, _ = model.model(batch[1], sample_latent=False)
    torch.testing.assert_close(
        reconstruction.sum(dim=(-1, -2)), torch.ones(reconstruction.shape[0])
    )

    restored = HistVAE(
        config=dict(model.config),
        outdir=str(tmp_path),
        exp_name="restored-preprocessor",
    )
    restored.prep_data(
        train_data=train_data,
        train_group=train_group,
        test_data=test_data,
        test_group=test_group,
    )
    assert restored.histogram_preprocessor.state_sha256 == (
        preprocessor.state_sha256
    )
    torch.testing.assert_close(
        restored.test_dataset.get_full_histogram(0),
        model.test_dataset.get_full_histogram(0),
        rtol=0,
        atol=0,
    )


@pytest.mark.smoke
def test_histvae_rejects_ambiguous_runtime_histogram_override(tmp_path):
    data = np.array([[0.0], [1.0], [5.0], [10.0]], dtype=np.float32)
    group = np.array(["a", "a", "b", "b"])
    spec = AxisPreprocessingSpec(
        name="signal",
        lower_mode="fixed",
        lower_value=0.0,
        upper_mode="fixed",
        upper_value=10.0,
    )
    preprocessor = HistogramPreprocessor([spec], bins=4).fit(data, group)
    model = HistVAE(
        config=make_histvae_config(),
        outdir=str(tmp_path),
        exp_name="strict",
    )

    with pytest.raises(ValueError, match="runtime overrides must be omitted"):
        model.prep_data(
            train_data=data,
            train_group=group,
            histogram_preprocessor=preprocessor,
            value_transform="none",
        )


@pytest.mark.smoke
def test_default_geometry_can_be_replaced_by_one_dimensional_preprocessor(tmp_path):
    data = np.array([[0.0], [1.0], [5.0], [10.0]], dtype=np.float32)
    group = np.array(["a", "a", "b", "b"])
    preprocessor = HistogramPreprocessor(
        [
            AxisPreprocessingSpec(
                name="signal",
                lower_mode="fixed",
                lower_value=0.0,
                upper_mode="fixed",
                upper_value=10.0,
            )
        ],
        bins=4,
    ).fit(data, group)

    config, _ = load_config(
        overrides={
            "device": "cpu",
            "in_dims": 1,
            "bins": 4,
            "hidden_dims": [2],
            "latent_dim": 2,
            "optimizer": "radam",
            "num_workers": 0,
            "pin_memory": False,
            "transform": False,
        }
    )
    model = HistVAE(
        config=config,
        outdir=str(tmp_path),
        exp_name="default-geometry-preprocessor",
        histogram_preprocessor=preprocessor,
    )
    model.prep_data(
        train_data=data,
        train_group=group,
    )
    assert model.config["min_vals"] == pytest.approx([0.0])
    assert model.config["max_vals"] == pytest.approx([10.0])

    unmatched_config, _ = load_config(
        overrides={
            "device": "cpu",
            "in_dims": 1,
            "bins": 4,
            "hidden_dims": [2],
            "latent_dim": 2,
            "optimizer": "radam",
            "num_workers": 0,
            "pin_memory": False,
            "transform": False,
        }
    )
    unmatched = HistVAE(
        config=unmatched_config,
        outdir=str(tmp_path),
        exp_name="default-geometry-unmatched",
    )
    with pytest.raises(ValueError, match="Supply matching bounds"):
        unmatched.prep_data(
            train_data=data,
            train_group=group,
        )
