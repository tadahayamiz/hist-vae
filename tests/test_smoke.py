from pathlib import Path
import copy

import numpy as np
import torch

from histvae import HistVAE


def make_config(tmp_path):
    return {
        "device": "cpu",
        "num_points": 4,
        "in_channels": 1,
        "in_dims": 2,
        "max_vals": [1.0, 1.0],
        "bins": 4,
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
        "weight_decay": 0.0,
        "transform": False,
        "accum_grad": 1,
        "clip_grad": 1.0,
        "save_model_every": 0,
        "patience": 0,
        "log_every": 1,
    }


def make_toy_data(with_labels=False):
    train_data = np.array([
        [0.1, 0.2], [0.15, 0.25], [0.2, 0.3],
        [0.7, 0.8], [0.75, 0.85], [0.8, 0.9],
    ], dtype=np.float32)
    train_group = np.array([0, 0, 0, 1, 1, 1])
    test_data = np.array([
        [0.12, 0.18], [0.18, 0.28], [0.22, 0.32],
        [0.68, 0.82], [0.74, 0.88], [0.78, 0.92],
    ], dtype=np.float32)
    test_group = np.array([10, 10, 10, 11, 11, 11])

    if not with_labels:
        return train_data, train_group, test_data, test_group, None, None

    train_label = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    test_label = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    return train_data, train_group, test_data, test_group, train_label, test_label


def test_histvae_prep_data_without_labels(tmp_path):
    config = make_config(tmp_path)
    train_data, train_group, test_data, test_group, _, _ = make_toy_data(with_labels=False)

    model = HistVAE(config=config, outdir=str(tmp_path), exp_name="toy")
    model.prep_data(
        train_data=train_data,
        train_group=train_group,
        test_data=test_data,
        test_group=test_group,
    )

    assert len(model.train_dataset) == 2
    assert len(model.test_dataset) == 2
    (hist0, hist1), label = model.train_dataset[0]
    assert hist0.shape == (1, 4, 4)
    assert hist1.shape == (1, 4, 4)
    assert label is None


def test_pretrain_train_epoch_and_evaluate_smoke(tmp_path):
    config = make_config(tmp_path)
    train_data, train_group, test_data, test_group, train_label, test_label = make_toy_data(with_labels=True)

    model = HistVAE(config=config, outdir=str(tmp_path), exp_name="toy-pretrain")
    model.prep_data(
        train_data=train_data,
        train_group=train_group,
        train_label=train_label,
        test_data=test_data,
        test_group=test_group,
        test_label=test_label,
    )
    model.prep_model("pretrain")

    train_loss, train_recon, train_kl = model.trainer.train_epoch(model.train_loader)
    test_loss, test_recon, test_kl = model.trainer.evaluate(model.test_loader)

    for value in [train_loss, train_recon, train_kl, test_loss, test_recon, test_kl]:
        assert np.isfinite(value)
        assert value >= 0



def test_finetune_forward_and_loss_smoke(tmp_path):
    config = make_config(tmp_path)
    train_data, train_group, test_data, test_group, train_label, test_label = make_toy_data(with_labels=True)

    pretrain = HistVAE(config=copy.deepcopy(config), outdir=str(tmp_path), exp_name="toy-pretrain-ckpt")
    pretrain.prep_data(
        train_data=train_data,
        train_group=train_group,
        train_label=train_label,
        test_data=test_data,
        test_group=test_group,
        test_label=test_label,
    )
    pretrain.prep_model("pretrain")
    ckpt_path = Path(tmp_path) / "pretrained_state.pt"
    torch.save(pretrain.model.state_dict(), ckpt_path)

    finetune = HistVAE(config=copy.deepcopy(config), outdir=str(tmp_path), exp_name="toy-finetune")
    finetune.prep_data(
        train_data=train_data,
        train_group=train_group,
        train_label=train_label,
        test_data=test_data,
        test_group=test_group,
        test_label=test_label,
    )
    finetune.prep_model("finetune", model_path=str(ckpt_path))

    (hist0, _), label = next(iter(finetune.train_loader))
    logits, recon, mu, logvar = finetune.model(hist0)
    loss = finetune.loss_fn(logits, label)

    assert logits.shape[0] == label.shape[0]
    assert recon.shape == hist0.shape
    assert mu.shape[0] == hist0.shape[0]
    assert logvar.shape == mu.shape
    assert np.isfinite(loss.detach().item())



def test_installed_distribution_metadata():
    import importlib.metadata as metadata

    assert metadata.version("histvae") == "0.0.1"



def test_src_layout_is_active_package():
    import histvae

    root = Path(__file__).resolve().parents[1]
    src_pkg = root / "src" / "histvae"
    root_pkg = root / "histvae"

    assert src_pkg.exists()
    assert not root_pkg.exists()
    assert Path(histvae.__file__).resolve() == (src_pkg / "__init__.py").resolve()



def test_obsolete_packaging_files_removed():
    root = Path(__file__).resolve().parents[1]

    assert not (root / "setup.py").exists()
    assert not (root / "MANIFEST.in").exists()
    assert not (root / "requirements.txt").exists()
