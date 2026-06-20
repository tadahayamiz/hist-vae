import json
import types
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from histvae.trainer import (
    PreTrainer,
    latent_kl_monitor_start_epoch,
    latent_kl_weight,
    validate_latent_kl_schedule,
)


def make_schedule_config(**overrides):
    config = {
        "device": "cpu",
        "exp_name": "latent-kl-schedule",
        "save_model_every": 0,
        "log_every": 10,
        "epochs": 4,
        "patience": 0,
        "early_stop_mode": "min",
        "pretrain_monitor": "test_recon",
        "active_latent_threshold": 0.01,
        "input_shape": (1, 4),
        "beta": 0.4,
        "latent_kl_schedule": "linear_warmup",
        "latent_kl_warmup_epochs": 3,
    }
    config.update(overrides)
    return config


@pytest.mark.smoke
def test_linear_latent_kl_schedule_has_explicit_epoch_contract():
    config = make_schedule_config()

    validate_latent_kl_schedule(config)

    assert [latent_kl_weight(config, epoch) for epoch in range(1, 6)] == pytest.approx(
        [0.0, 0.2, 0.4, 0.4, 0.4]
    )
    assert latent_kl_monitor_start_epoch(config) == 3


@pytest.mark.smoke
@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"latent_kl_schedule": "cyclic"}, "Unsupported latent_kl_schedule"),
        ({"beta": -0.1}, "beta must be finite and non-negative"),
        (
            {
                "latent_kl_schedule": "constant",
                "latent_kl_warmup_epochs": 2,
            },
            "must be 0",
        ),
        (
            {
                "latent_kl_schedule": "linear_warmup",
                "beta": 0.0,
            },
            "requires beta > 0",
        ),
        (
            {"latent_kl_warmup_epochs": 1},
            "requires latent_kl_warmup_epochs >= 2",
        ),
        (
            {"latent_kl_warmup_epochs": 5},
            "must not exceed epochs",
        ),
        (
            {"latent_kl_warmup_epochs": 2.0},
            "must be an integer",
        ),
    ],
)
def test_latent_kl_schedule_validation_is_strict(overrides, message):
    with pytest.raises(ValueError, match=message):
        validate_latent_kl_schedule(make_schedule_config(**overrides))


@pytest.mark.smoke
def test_linear_warmup_defers_best_checkpoint_monitoring(tmp_path):
    model = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.zero_()
    optimizer = torch.optim.RAdam(model.parameters(), lr=1e-3)
    config = make_schedule_config()
    trainer = PreTrainer(
        config=config,
        model=model,
        optimizer=optimizer,
        outdir=str(tmp_path),
    )
    monitored_recon = iter([0.1, 0.2, 1.0, 2.0])

    def fake_train_epoch(self, _loader):
        with torch.no_grad():
            self.model.weight.add_(1.0)
        return 1.0 + self.current_beta, 1.0, 1.0

    def fake_evaluate(self, _loader):
        recon = next(monitored_recon)
        return recon + self.current_beta, recon, 1.0, 0.0, 0

    trainer.train_epoch = types.MethodType(fake_train_epoch, trainer)
    trainer.evaluate = types.MethodType(fake_evaluate, trainer)
    trainer.train(None, None)

    run_dir = Path(tmp_path) / config["exp_name"]
    best = torch.load(run_dir / "model_best.pt", weights_only=False)
    last = torch.load(run_dir / "model_last.pt", weights_only=False)
    history = json.loads((run_dir / "history.json").read_text())

    assert history["beta"] == pytest.approx([0.0, 0.2, 0.4, 0.4])
    assert history["monitor_eligible"] == [False, False, True, True]
    assert history["monitor_start_epoch"] == 3
    assert history["best_epoch"] == 3
    assert history["best_beta"] == pytest.approx(0.4)
    assert history["last_beta"] == pytest.approx(0.4)
    assert float(trainer.model.weight.item()) == pytest.approx(3.0)
    assert float(best["model"]["weight"].item()) == pytest.approx(3.0)
    assert float(last["model"]["weight"].item()) == pytest.approx(4.0)
    assert best["epoch"] == 3
    assert best["beta"] == pytest.approx(0.4)
    assert last["epoch"] == 4
    assert last["beta"] == pytest.approx(0.4)


@pytest.mark.smoke
def test_scheduled_total_loss_cannot_be_used_for_checkpoint_selection(tmp_path):
    model = nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.RAdam(model.parameters(), lr=1e-3)

    with pytest.raises(ValueError, match="pretrain_monitor='test_recon'"):
        PreTrainer(
            config=make_schedule_config(pretrain_monitor="test_loss"),
            model=model,
            optimizer=optimizer,
            outdir=str(tmp_path),
        )
