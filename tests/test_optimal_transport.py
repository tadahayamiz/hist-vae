import importlib.util
import json
import math

import numpy as np
import pytest
import torch

from histvae import (
    HistVAE,
    JointSinkhornDivergence,
    build_joint_bin_support,
    validate_ot_config,
)
from histvae.models import ConvVAE


torch.set_num_threads(1)


@pytest.mark.smoke
@pytest.mark.parametrize(
    "bins_per_axis",
    [(4,), (3, 4), (2, 3, 4)],
)
def test_joint_bin_support_uses_the_full_joint_grid(bins_per_axis):
    edges = [
        np.linspace(-2.0 * (axis + 1), 3.0 * (axis + 1), bins + 1)
        for axis, bins in enumerate(bins_per_axis)
    ]

    support = build_joint_bin_support(edges)

    assert support.shape == (int(np.prod(bins_per_axis)), len(bins_per_axis))
    assert np.isfinite(support).all()
    pairwise = support[:, None] - support[None]
    assert np.linalg.norm(pairwise, axis=-1).max() <= 1.0 + 1e-6
    for axis, bins in enumerate(bins_per_axis):
        assert len(np.unique(support[:, axis])) == bins


@pytest.mark.smoke
def test_joint_sinkhorn_identity_symmetry_and_distance_order():
    support = build_joint_bin_support([np.linspace(0.0, 1.0, 9)])
    metric = JointSinkhornDivergence(
        support=support,
        p=1,
        blur=0.02,
        scaling=0.8,
        backend="tensorized",
    )

    def point_mass(index):
        value = torch.zeros(1, 1, 8, dtype=torch.float32)
        value[0, 0, index] = 1.0
        return value

    anchor = point_mass(3)
    near = point_mass(4)
    far = point_mass(7)

    identity = metric(anchor, anchor)
    near_value = metric(anchor, near)
    reverse_value = metric(near, anchor)
    far_value = metric(anchor, far)

    assert float(identity) < 1e-6
    torch.testing.assert_close(near_value, reverse_value, rtol=1e-5, atol=1e-6)
    assert 0 < float(near_value) < float(far_value)


@pytest.mark.smoke
@pytest.mark.parametrize("spatial_shape", [(5,), (3, 4)])
def test_joint_sinkhorn_pairwise_matches_aligned_forward(spatial_shape):
    edges = [np.linspace(0.0, 1.0, bins + 1) for bins in spatial_shape]
    metric = JointSinkhornDivergence(
        support=build_joint_bin_support(edges),
        p=1,
        blur=0.05,
        scaling=0.8,
        backend="tensorized",
    )

    query = torch.rand(3, 1, *spatial_shape)
    query = query / query.flatten(start_dim=1).sum(dim=1).view(
        3, *([1] * (len(spatial_shape) + 1))
    )
    reference = torch.rand(2, 1, *spatial_shape)
    reference = reference / reference.flatten(start_dim=1).sum(dim=1).view(
        2, *([1] * (len(spatial_shape) + 1))
    )

    pairwise = metric.pairwise(query, reference, pair_batch_size=2)

    assert pairwise.shape == (3, 2)
    for query_index in range(3):
        for reference_index in range(2):
            aligned = metric(
                query[query_index:query_index + 1],
                reference[reference_index:reference_index + 1],
                reduction="none",
            )
            torch.testing.assert_close(
                pairwise[query_index, reference_index],
                aligned[0],
                rtol=1e-5,
                atol=1e-6,
            )

    with pytest.raises(ValueError, match="pair_batch_size"):
        metric.pairwise(query, reference, pair_batch_size=0)


@pytest.mark.smoke
def test_joint_sinkhorn_is_differentiable_with_respect_to_prediction_mass():
    support = build_joint_bin_support([np.linspace(0.0, 1.0, 9)])
    metric = JointSinkhornDivergence(
        support=support,
        p=1,
        blur=0.05,
        scaling=0.8,
        backend="tensorized",
    )
    target = torch.zeros(2, 1, 8, dtype=torch.float32)
    target[0, 0, 1] = 1.0
    target[1, 0, 6] = 1.0
    logits = torch.zeros(2, 1, 8, dtype=torch.float32, requires_grad=True)
    prediction = torch.softmax(logits.flatten(start_dim=1), dim=1).view_as(logits)

    loss = metric(target, prediction)
    loss.backward()

    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert float(logits.grad.abs().sum()) > 0


@pytest.mark.smoke
def test_ot_config_is_strict_about_auxiliary_loss_contract():
    base = {
        "histogram_mode": "probability_mass",
        "decoder_output_mode": "simplex_softmax",
        "reconstruction_loss": "forward_kl",
    }

    with pytest.raises(ValueError, match="ot_weight must be 0"):
        validate_ot_config({**base, "ot_loss": "none", "ot_weight": 0.1})

    with pytest.raises(ValueError, match="requires ot_weight > 0"):
        validate_ot_config({**base, "ot_loss": "sinkhorn", "ot_weight": 0.0})

    with pytest.raises(ValueError, match="histogram_mode='probability_mass'"):
        validate_ot_config({
            **base,
            "histogram_mode": "count",
            "ot_loss": "sinkhorn",
            "ot_weight": 0.1,
        })

    with pytest.raises(ValueError, match="ot_p must be 1 or 2"):
        validate_ot_config({
            **base,
            "ot_loss": "sinkhorn",
            "ot_weight": 0.1,
            "ot_p": 3,
        })

    resolved = validate_ot_config({
        **base,
        "ot_loss": "sinkhorn",
        "ot_weight": 0.1,
        "ot_p": 1,
        "ot_blur": 0.05,
        "ot_scaling": 0.8,
        "ot_backend": "tensorized",
        "ot_mass_epsilon": 0.0,
    })
    assert resolved["ot_loss"] == "sinkhorn"
    assert resolved["ot_weight"] == 0.1


@pytest.mark.smoke
@pytest.mark.parametrize("spatial_shape", [(4,), (4, 4), (4, 4, 4)])
def test_convvae_combines_forward_kl_and_joint_sinkhorn(spatial_shape):
    edges = [np.linspace(0.0, 1.0, bins + 1) for bins in spatial_shape]
    support = build_joint_bin_support(edges)
    model = ConvVAE(
        input_shape=(1, *spatial_shape),
        latent_dim=2,
        hidden_dims=[2],
        dropout_conv=0.0,
        decoder_output_mode="simplex_softmax",
        reconstruction_loss="forward_kl",
        ot_loss="sinkhorn",
        ot_weight=0.3,
        ot_p=1,
        ot_blur=0.05,
        ot_scaling=0.8,
        ot_backend="tensorized",
        ot_support=support,
    )
    model.eval()
    target = torch.rand(2, 1, *spatial_shape)
    target = target / target.flatten(start_dim=1).sum(dim=1).view(
        2, *([1] * (len(spatial_shape) + 1))
    )

    reconstruction, mu, logvar = model(target, sample_latent=False)
    total, observation, latent_kl, components = model.vae_loss(
        reconstruction,
        target,
        mu,
        logvar,
        beta=0.2,
        return_components=True,
    )

    assert torch.isfinite(total)
    assert torch.isfinite(observation)
    assert torch.isfinite(latent_kl)
    assert torch.isfinite(components["sinkhorn"])
    torch.testing.assert_close(
        observation,
        components["base_reconstruction"]
        + 0.3 * components["sinkhorn"],
    )
    torch.testing.assert_close(total, observation + 0.2 * latent_kl)


def _small_ot_config():
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
        "ot_loss": "sinkhorn",
        "ot_weight": 0.2,
        "ot_p": 1,
        "ot_blur": 0.05,
        "ot_scaling": 0.8,
        "ot_backend": "tensorized",
        "ot_mass_epsilon": 0.0,
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


@pytest.mark.smoke
def test_histvae_ot_api_logs_components_and_checkpoint_contract(tmp_path):
    config = _small_ot_config()
    data = np.array(
        [[0.05], [0.20], [0.35], [0.45], [0.55], [0.70], [0.85], [0.95]],
        dtype=np.float32,
    )
    group = np.array(["a"] * 4 + ["b"] * 4)
    histvae = HistVAE(
        config=config,
        outdir=str(tmp_path),
        exp_name="ot-smoke",
        seed=7,
    )
    histvae.prep_data(
        train_data=data,
        train_group=group,
        test_data=data,
        test_group=group,
    )
    histvae.prep_model("pretrain")

    assert histvae.ot_support.shape == (4, 1)
    assert histvae.config["ot_support_metadata"]["support_size"] == 4
    histvae.train(verbose=False)

    run_dir = tmp_path / "ot-smoke"
    history = json.loads((run_dir / "history.json").read_text())
    for key in (
        "train_base_recon",
        "test_base_recon",
        "train_sinkhorn",
        "test_sinkhorn",
        "train_weighted_sinkhorn",
        "test_weighted_sinkhorn",
    ):
        assert key in history
        assert len(history[key]) == 1
        assert math.isfinite(history[key][0])

    checkpoint = torch.load(run_dir / "model_best.pt", map_location="cpu")
    assert checkpoint["ot_loss"] == "sinkhorn"
    assert checkpoint["ot_weight"] == 0.2
    assert "ot_divergence.support" in checkpoint["model"]

    reconstruction = histvae.get_reconstruction(histvae.test_dataset)
    np.testing.assert_allclose(
        reconstruction["observation_loss"],
        reconstruction["base_reconstruction_loss"]
        + reconstruction["weighted_sinkhorn_divergence"],
        rtol=1e-6,
        atol=1e-7,
    )
    assert np.all(reconstruction["sinkhorn_divergence"] >= 0)


@pytest.mark.smoke
def test_scalable_backend_requires_explicit_pykeops_dependency():
    if importlib.util.find_spec("pykeops") is not None:
        pytest.skip("PyKeOps is installed in this environment.")
    support = build_joint_bin_support([np.linspace(0.0, 1.0, 5)])
    with pytest.raises(ImportError, match="ot-scalable"):
        JointSinkhornDivergence(support=support, backend="online")
