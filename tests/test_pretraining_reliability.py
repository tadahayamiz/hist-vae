import types
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
import yaml

from histvae import HistVAE
from histvae.core import make_optimizer
from histvae.models import ConvVAE
from histvae.trainer import PreTrainer


def make_1d_config():
    return {
        "device": "cpu",
        "num_points": 4,
        "in_channels": 1,
        "in_dims": 1,
        "max_vals": [1.0],
        "bins": 4,
        "histogram_mode": "probability_mass",
        "out_of_range_policy": "clip",
        "value_transform": "none",
        "train_sampling_mode": "random",
        "eval_sampling_mode": "full",
        "num_workers": 0,
        "pin_memory": False,
        "latent_dim": 2,
        "hidden_dims": [2],
        "dropout_conv": 0.0,
        "beta": 1.0,
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
        "finetune_monitor": "test_loss",
        "active_latent_threshold": 0.01,
        "log_every": 1,
    }


def make_1d_data():
    train_data = np.array(
        [
            [0.05], [0.1], [0.2], [0.3], [0.4], [0.45],
            [0.55], [0.6], [0.7], [0.8], [0.9], [0.95],
        ],
        dtype=np.float32,
    )
    train_group = np.array(["a"] * 6 + ["b"] * 6)
    test_data = np.array(
        [
            [0.08], [0.12], [0.22], [0.32], [0.42], [0.48],
            [0.52], [0.58], [0.68], [0.78], [0.88], [0.92],
        ],
        dtype=np.float32,
    )
    test_group = np.array(["c"] * 6 + ["d"] * 6)
    return train_data, train_group, test_data, test_group


@pytest.mark.smoke
def test_full_group_validation_and_latent_extraction_are_deterministic(tmp_path):
    config = make_1d_config()
    train_data, train_group, test_data, test_group = make_1d_data()
    histvae = HistVAE(config=config, outdir=str(tmp_path), exp_name="deterministic")
    histvae.prep_data(
        train_data=train_data,
        train_group=train_group,
        test_data=test_data,
        test_group=test_group,
    )
    histvae.prep_model("pretrain")

    assert histvae.train_dataset.sampling_mode == "random"
    assert histvae.test_dataset.sampling_mode == "full"

    first = histvae.test_dataset[0][0]
    second = histvae.test_dataset[0][0]
    torch.testing.assert_close(first[0], second[0], rtol=0, atol=0)
    torch.testing.assert_close(first[1], second[1], rtol=0, atol=0)

    metrics0 = histvae.trainer.evaluate(histvae.test_loader)
    metrics1 = histvae.trainer.evaluate(histvae.test_loader)
    np.testing.assert_allclose(metrics0, metrics1, rtol=0, atol=0)

    latent0 = histvae.get_latent(histvae.train_dataset)
    latent1 = histvae.get_latent(histvae.train_dataset)
    np.testing.assert_allclose(latent0, latent1, rtol=0, atol=0)


@pytest.mark.smoke
def test_deterministic_forward_uses_latent_mean():
    model = ConvVAE(
        input_shape=[1, 4],
        latent_dim=2,
        hidden_dims=[2],
        dropout_conv=0.0,
    ).eval()
    x = torch.tensor([[[0.1, 0.2, 0.3, 0.4]]], dtype=torch.float32)

    with torch.no_grad():
        recon0, mu0, _ = model(x, sample_latent=False)
        recon1, mu1, _ = model(x, sample_latent=False)
        decoded_mu = model.decode(mu0)

    torch.testing.assert_close(recon0, recon1, rtol=0, atol=0)
    torch.testing.assert_close(recon0, decoded_mu, rtol=0, atol=0)
    torch.testing.assert_close(mu0, mu1, rtol=0, atol=0)


@pytest.mark.smoke
def test_dropout_conv_config_reaches_all_convolutional_blocks():
    no_dropout = ConvVAE(
        input_shape=[1, 8],
        latent_dim=2,
        hidden_dims=[2, 4],
        dropout_conv=0.0,
    )
    with_dropout = ConvVAE(
        input_shape=[1, 8],
        latent_dim=2,
        hidden_dims=[2, 4],
        dropout_conv=0.17,
    )

    assert not any(isinstance(module, nn.Dropout) for module in no_dropout.modules())
    dropouts = [
        module for module in with_dropout.modules()
        if isinstance(module, nn.Dropout)
    ]
    assert len(dropouts) == 4
    assert all(module.p == pytest.approx(0.17) for module in dropouts)


@pytest.mark.smoke
def test_optimizer_selection_is_explicit():
    parameter = nn.Parameter(torch.tensor(1.0))
    config = {
        "optimizer": "radam",
        "lr": 1e-3,
        "weight_decay": 0.0,
    }

    optimizer = make_optimizer([parameter], config)

    assert type(optimizer) is torch.optim.RAdam
    with pytest.raises(ValueError, match="Unsupported optimizer"):
        make_optimizer([parameter], {**config, "optimizer": "automatic"})


@pytest.mark.smoke
def test_trainer_restores_and_saves_monitored_best_epoch(tmp_path):
    model = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.zero_()
    optimizer = torch.optim.RAdam(model.parameters(), lr=1e-3)
    config = {
        "device": "cpu",
        "exp_name": "best-checkpoint",
        "save_model_every": 0,
        "log_every": 10,
        "epochs": 3,
        "patience": 0,
        "early_stop_mode": "min",
        "pretrain_monitor": "test_recon",
        "active_latent_threshold": 0.01,
        "input_shape": (1, 4),
    }
    trainer = PreTrainer(
        config=config,
        model=model,
        optimizer=optimizer,
        outdir=str(tmp_path),
    )
    monitored_recon = iter([3.0, 1.0, 2.0])

    def fake_train_epoch(self, _loader):
        with torch.no_grad():
            self.model.weight.add_(1.0)
        return 1.0, 1.0, 0.0

    def fake_evaluate(self, _loader):
        recon = next(monitored_recon)
        return recon, recon, 0.0, 0.0, 0

    trainer.train_epoch = types.MethodType(fake_train_epoch, trainer)
    trainer.evaluate = types.MethodType(fake_evaluate, trainer)
    trainer.train(None, None)

    run_dir = Path(tmp_path) / "best-checkpoint"
    best = torch.load(run_dir / "model_best.pt", weights_only=False)
    last = torch.load(run_dir / "model_last.pt", weights_only=False)
    with open(run_dir / "config.yaml", "r") as handle:
        saved_config = yaml.safe_load(handle)

    assert float(trainer.model.weight.item()) == pytest.approx(2.0)
    assert float(best["model"]["weight"].item()) == pytest.approx(2.0)
    assert float(last["model"]["weight"].item()) == pytest.approx(3.0)
    assert best["epoch"] == 2
    assert best["score"] == pytest.approx(1.0)
    assert best["monitor_metric"] == "test_recon"
    assert saved_config["input_shape"] == [1, 4]
