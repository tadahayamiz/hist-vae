import numpy as np

from histvae import HistVAE


def make_config(tmp_path):
    return {
        "device": "cpu",
        "num_points": 16,
        "in_channels": 1,
        "in_dims": 2,
        "max_vals": [1.0, 1.0],
        "bins": 16,
        "num_workers": 0,
        "pin_memory": False,
        "latent_dim": 4,
        "hidden_dims": [8, 16],
        "dropout_conv": 0.0,
        "beta": 1.0,
        "num_classes": 2,
        "num_layers": 2,
        "hidden_head": 8,
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


def make_toy_data():
    train_data = np.array([
        [0.1, 0.2], [0.15, 0.25], [0.2, 0.3],
        [0.7, 0.8], [0.75, 0.85], [0.8, 0.9],
        [0.25, 0.2], [0.3, 0.15], [0.35, 0.1],
        [0.65, 0.7], [0.7, 0.75], [0.72, 0.78],
    ], dtype=np.float32)
    train_group = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])
    test_data = np.array([
        [0.12, 0.18], [0.18, 0.28], [0.22, 0.32],
        [0.68, 0.82], [0.74, 0.88], [0.78, 0.92],
    ], dtype=np.float32)
    test_group = np.array([10, 10, 10, 11, 11, 11])
    return train_data, train_group, test_data, test_group


def test_histvae_prep_data_without_labels(tmp_path):
    config = make_config(tmp_path)
    train_data, train_group, test_data, test_group = make_toy_data()

    model = HistVAE(config=config, outdir=str(tmp_path), exp_name="toy")
    model.prep_data(
        train_data=train_data,
        train_group=train_group,
        test_data=test_data,
        test_group=test_group,
    )

    assert len(model.train_dataset) == 4
    assert len(model.test_dataset) == 2
    (hist0, hist1), label = model.train_dataset[0]
    assert hist0.shape == (1, 16, 16)
    assert hist1.shape == (1, 16, 16)
    assert label is None


def test_histvae_pretrain_setup_smoke(tmp_path):
    config = make_config(tmp_path)
    train_data, train_group, test_data, test_group = make_toy_data()

    model = HistVAE(config=config, outdir=str(tmp_path), exp_name="toy")
    model.prep_data(
        train_data=train_data,
        train_group=train_group,
        test_data=test_data,
        test_group=test_group,
    )
    model.prep_model(mode="pretrain")

    assert model.model is not None
    assert model.optimizer is not None
    assert model.trainer is not None

    (hist0, hist1), labels = next(iter(model.train_loader))
    assert hist0.shape == (2, 1, 16, 16)
    assert hist1.shape == (2, 1, 16, 16)
    assert labels.shape == (2,)
    assert (labels == -1).all()
