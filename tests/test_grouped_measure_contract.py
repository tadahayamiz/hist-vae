from pathlib import Path

import numpy as np
import pytest
import torch

from histvae import HistVAE
from histvae.data_handler import Histogram, PointHistDataLoader, PointHistDataset
from histvae.models import ConvVAE


torch.set_num_threads(1)


@pytest.mark.smoke
def test_two_dimensional_histogram_preserves_axis_order():
    data = np.array(
        [
            [0.25, 17.5],
            [1.75, 7.5],
        ],
        dtype=np.float32,
    )
    histogram = Histogram(
        dimension=2,
        max_vals=[2.0, 20.0],
        bins_per_dim=[2, 4],
        histogram_mode="count",
        out_of_range_policy="error",
    )

    result = histogram.compute(data)

    expected = np.zeros((2, 4), dtype=np.int64)
    expected[0, 3] = 1
    expected[1, 1] = 1
    np.testing.assert_array_equal(result, expected)


@pytest.mark.smoke
def test_random_input_can_use_a_fixed_full_group_target():
    data = np.array(
        [[0.05], [0.15], [0.25], [0.35], [0.65], [0.75], [0.85], [0.95]],
        dtype=np.float32,
    )
    group = np.array(["sample"] * len(data))
    dataset = PointHistDataset(
        data=data,
        group=group,
        max_vals=[1.0],
        num_points=3,
        bins=4,
        histogram_mode="probability_mass",
        sampling_mode="random",
        target_sampling_mode="full",
        transform=False,
    )
    full_target = dataset.get_full_histogram(0)

    for _ in range(5):
        (target_hist, input_hist), _ = dataset[0]
        torch.testing.assert_close(target_hist, full_target, rtol=0, atol=0)
        torch.testing.assert_close(input_hist.sum(), torch.tensor(1.0))


@pytest.mark.smoke
def test_probability_mass_rejects_legacy_histogram_value_augmentation():
    data = np.array([[0.1], [0.2], [0.8], [0.9]], dtype=np.float32)
    group = np.array(["sample"] * len(data))

    with pytest.raises(ValueError, match="probability simplex"):
        PointHistDataset(
            data=data,
            group=group,
            max_vals=[1.0],
            bins=4,
            histogram_mode="probability_mass",
            transform=True,
        )


@pytest.mark.smoke
@pytest.mark.parametrize("spatial_shape", [(4,), (4, 4), (4, 4, 4)])
def test_simplex_decoder_is_dimension_agnostic(spatial_shape):
    model = ConvVAE(
        input_shape=(1, *spatial_shape),
        latent_dim=2,
        hidden_dims=[2],
        dropout_conv=0.0,
        decoder_output_mode="simplex_softmax",
        reconstruction_loss="forward_kl",
    )
    model.eval()
    target = torch.rand(2, 1, *spatial_shape)
    target = target / target.flatten(start_dim=1).sum(dim=1).view(
        2, *([1] * (len(spatial_shape) + 1))
    )

    with torch.no_grad():
        reconstruction, mu, logvar = model(target, sample_latent=False)
        total, recon_loss, latent_kl = model.vae_loss(
            reconstruction, target, mu, logvar, beta=1.0
        )

    assert reconstruction.shape == target.shape
    torch.testing.assert_close(
        reconstruction.flatten(start_dim=1).sum(dim=1),
        torch.ones(2),
        rtol=1e-6,
        atol=1e-6,
    )
    assert torch.isfinite(total)
    assert torch.isfinite(recon_loss)
    assert torch.isfinite(latent_kl)


@pytest.mark.smoke
def test_forward_kl_is_zero_for_an_exact_distribution_match():
    model = ConvVAE(
        input_shape=(1, 4),
        latent_dim=2,
        hidden_dims=[2],
        dropout_conv=0.0,
        decoder_output_mode="simplex_softmax",
        reconstruction_loss="forward_kl",
    )
    target = torch.tensor([[[0.2, 0.3, 0.1, 0.4]]], dtype=torch.float32)
    shifted = torch.tensor([[[0.4, 0.1, 0.3, 0.2]]], dtype=torch.float32)
    mu = torch.zeros(1, 2)
    logvar = torch.zeros(1, 2)

    _, exact_kl, _ = model.vae_loss(target, target, mu, logvar, beta=0.0)
    _, shifted_kl, _ = model.vae_loss(shifted, target, mu, logvar, beta=0.0)

    assert abs(float(exact_kl)) < 1e-6
    assert float(shifted_kl) > 0


@pytest.mark.smoke
def test_full_group_histogram_aggregates_rows_across_technical_partitions():
    interleaved_data = np.array(
        [[0.1], [0.8], [0.2], [0.9]],
        dtype=np.float32,
    )
    contiguous_data = np.array(
        [[0.1], [0.2], [0.8], [0.9]],
        dtype=np.float32,
    )
    group = np.array(["sample"] * 4)

    interleaved_dataset = PointHistDataset(
        data=interleaved_data,
        group=group,
        max_vals=[1.0],
        bins=2,
        histogram_mode="probability_mass",
        sampling_mode="full",
    )
    contiguous_dataset = PointHistDataset(
        data=contiguous_data,
        group=group,
        max_vals=[1.0],
        bins=2,
        histogram_mode="probability_mass",
        sampling_mode="full",
    )

    torch.testing.assert_close(
        interleaved_dataset.get_full_histogram(0),
        contiguous_dataset.get_full_histogram(0),
    )
    torch.testing.assert_close(
        interleaved_dataset.get_full_histogram(0),
        torch.tensor([[0.5, 0.5]], dtype=torch.float32),
    )


@pytest.mark.smoke
def test_group_constant_condition_is_collated_with_histograms():
    data = np.array(
        [[0.1], [0.2], [0.3], [0.4], [0.6], [0.7], [0.8], [0.9]],
        dtype=np.float32,
    )
    group = np.array(["a"] * 4 + ["b"] * 4)
    condition = np.array(
        [[1.0, 0.0]] * 4 + [[0.0, 1.0]] * 4,
        dtype=np.float32,
    )
    dataset = PointHistDataset(
        data=data,
        group=group,
        condition=condition,
        max_vals=[1.0],
        bins=4,
        num_points=2,
        histogram_mode="probability_mass",
        sampling_mode="random",
        target_sampling_mode="full",
    )
    loader = PointHistDataLoader(
        dataset=dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    data_batch, labels = next(iter(loader))
    hist0, hist1, condition_batch = data_batch

    assert hist0.shape == hist1.shape == (2, 1, 4)
    torch.testing.assert_close(
        condition_batch,
        torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
    )
    torch.testing.assert_close(labels, torch.tensor([-1, -1]))


@pytest.mark.smoke
def test_condition_must_be_constant_within_each_group():
    data = np.array([[0.1], [0.2], [0.8], [0.9]], dtype=np.float32)
    group = np.array(["a", "a", "b", "b"])
    condition = np.array([[0.0], [1.0], [0.0], [0.0]], dtype=np.float32)

    with pytest.raises(ValueError, match="constant within each group"):
        PointHistDataset(
            data=data,
            group=group,
            condition=condition,
            max_vals=[1.0],
            bins=2,
        )


def make_conditioned_config(condition_mode="decoder", condition_dim=2):
    return {
        "device": "cpu",
        "num_points": 2,
        "in_channels": 1,
        "in_dims": 1,
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
        "condition_mode": condition_mode,
        "condition_dim": condition_dim,
        "beta": 0.0,
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
        "log_every": 1,
    }


@pytest.mark.smoke
def test_condition_mode_none_rejects_silent_condition_ignoring(tmp_path):
    config = make_conditioned_config(condition_mode="none", condition_dim=0)
    histvae = HistVAE(config=config, outdir=str(tmp_path), exp_name="none")
    data = np.array([[0.1], [0.2], [0.8], [0.9]], dtype=np.float32)
    group = np.array(["a", "a", "b", "b"])
    condition = np.zeros((4, 1), dtype=np.float32)

    with pytest.raises(ValueError, match="condition_mode='none'"):
        histvae.prep_data(
            train_data=data,
            train_group=group,
            train_condition=condition,
        )


@pytest.mark.smoke
def test_condition_array_width_and_finiteness_are_strict(tmp_path):
    config = make_conditioned_config(condition_dim=2)
    data = np.array([[0.1], [0.2], [0.8], [0.9]], dtype=np.float32)
    group = np.array(["a", "a", "b", "b"])
    histvae = HistVAE(config=config, outdir=str(tmp_path), exp_name="strict")

    with pytest.raises(ValueError, match="width must equal condition_dim"):
        histvae.prep_data(
            train_data=data,
            train_group=group,
            train_condition=np.zeros((4, 1), dtype=np.float32),
        )

    with pytest.raises(ValueError, match="finite"):
        histvae.prep_data(
            train_data=data,
            train_group=group,
            train_condition=np.array(
                [[1.0, 0.0], [1.0, np.nan], [0.0, 1.0], [0.0, 1.0]],
                dtype=np.float32,
            ),
        )


@pytest.mark.smoke
def test_decoder_conditioning_does_not_enter_the_encoder():
    model = ConvVAE(
        input_shape=(1, 4),
        latent_dim=2,
        hidden_dims=[2],
        dropout_conv=0.0,
        decoder_output_mode="simplex_softmax",
        reconstruction_loss="forward_kl",
        condition_mode="decoder",
        condition_dim=2,
    )
    model.eval()
    target = torch.tensor(
        [
            [[0.1, 0.2, 0.3, 0.4]],
            [[0.1, 0.2, 0.3, 0.4]],
        ],
        dtype=torch.float32,
    )
    condition = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    with torch.no_grad():
        mu, logvar = model.encode(target)
        reconstruction, forward_mu, forward_logvar = model(
            target,
            sample_latent=False,
            condition=condition,
        )

    torch.testing.assert_close(forward_mu, mu)
    torch.testing.assert_close(forward_logvar, logvar)
    torch.testing.assert_close(
        reconstruction.flatten(start_dim=1).sum(dim=1),
        torch.ones(2),
    )
    with pytest.raises(ValueError, match="condition is required"):
        model(target, sample_latent=False)


@pytest.mark.smoke
def test_conditioned_pretraining_executes_and_saves_best_checkpoint(tmp_path):
    config = make_conditioned_config()
    data = np.array(
        [[0.1], [0.2], [0.3], [0.4], [0.6], [0.7], [0.8], [0.9]],
        dtype=np.float32,
    )
    train_group = np.array(["a"] * 4 + ["b"] * 4)
    test_group = np.array(["c"] * 4 + ["d"] * 4)
    condition = np.array(
        [[1.0, 0.0]] * 4 + [[0.0, 1.0]] * 4,
        dtype=np.float32,
    )

    histvae = HistVAE(
        config=config,
        outdir=str(tmp_path),
        exp_name="conditioned-measure",
    )
    histvae.prep_data(
        train_data=data,
        train_group=train_group,
        train_condition=condition,
        test_data=data,
        test_group=test_group,
        test_condition=condition,
    )
    histvae.prep_model("pretrain")
    train_metrics = histvae.trainer.train_epoch(histvae.train_loader)
    eval_metrics = histvae.trainer.evaluate(histvae.test_loader)
    histvae.train(verbose=False)

    assert all(np.isfinite(value) for value in train_metrics)
    assert all(np.isfinite(value) for value in eval_metrics)
    assert Path(tmp_path, "conditioned-measure", "model_best.pt").exists()


@pytest.mark.smoke
def test_conditioned_finetune_prediction_path(tmp_path):
    config = make_conditioned_config(condition_dim=1)
    data = np.array(
        [[0.1], [0.2], [0.3], [0.4], [0.6], [0.7], [0.8], [0.9]],
        dtype=np.float32,
    )
    group = np.array(["a"] * 4 + ["b"] * 4)
    label = np.array([0] * 4 + [1] * 4, dtype=np.int64)
    condition = np.array([0.0] * 4 + [1.0] * 4, dtype=np.float32)

    pretrain = HistVAE(
        config=config.copy(),
        outdir=str(tmp_path),
        exp_name="conditioned-pretrain-state",
    )
    pretrain.prep_model("pretrain")
    checkpoint = tmp_path / "conditioned_state.pt"
    torch.save(pretrain.model.state_dict(), checkpoint)

    finetune = HistVAE(
        config=config.copy(),
        outdir=str(tmp_path),
        exp_name="conditioned-finetune",
    )
    finetune.prep_data(
        train_data=data,
        train_group=group,
        train_label=label,
        train_condition=condition,
        test_data=data,
        test_group=group,
        test_label=label,
        test_condition=condition,
    )
    finetune.prep_model("finetune", model_path=str(checkpoint))

    predictions, logits, labels = finetune.predict(finetune.test_loader)

    assert predictions.shape == (2,)
    assert logits.shape == (2, 2)
    assert labels.shape == (2,)


@pytest.mark.smoke
def test_condition_contract_is_strict():
    with pytest.raises(ValueError, match="condition_dim must be 0"):
        ConvVAE(
            input_shape=(1, 4),
            latent_dim=2,
            hidden_dims=[2],
            condition_mode="none",
            condition_dim=1,
        )

    with pytest.raises(ValueError, match="condition_dim must be positive"):
        ConvVAE(
            input_shape=(1, 4),
            latent_dim=2,
            hidden_dims=[2],
            condition_mode="decoder",
            condition_dim=0,
        )


@pytest.mark.smoke
def test_unconditioned_forward_keeps_positional_sample_latent_api():
    model = ConvVAE(
        input_shape=(1, 4),
        latent_dim=2,
        hidden_dims=[2],
        dropout_conv=0.0,
        decoder_output_mode="simplex_softmax",
        reconstruction_loss="forward_kl",
    )
    model.eval()
    target = torch.tensor(
        [
            [[0.1, 0.2, 0.3, 0.4]],
            [[0.4, 0.3, 0.2, 0.1]],
        ],
        dtype=torch.float32,
    )

    with torch.no_grad():
        positional = model(target, False)
        keyword = model(target, sample_latent=False)

    for positional_tensor, keyword_tensor in zip(positional, keyword):
        torch.testing.assert_close(positional_tensor, keyword_tensor)
