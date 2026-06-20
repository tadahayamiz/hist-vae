# HistVAE

HistVAE is a Python package for learning latent representations from point-cloud or histogram-style data using a variational autoencoder (VAE).

The package provides utilities for:

- preparing histogram / point-cloud data
- VAE-based representation learning
- pretraining and fine-tuning workflows
- command-line execution for experiments

This repository contains the reference implementation used in our research.

## Installation

Clone the repository and install it in editable mode:

```bash
git clone https://github.com/mizuno-group/hist-vae.git
cd histvae
pip install -e .
```

## Dependencies

HistVAE depends primarily on:

- Python >= 3.10
- PyTorch
- NumPy
- pandas
- PyYAML
- matplotlib
- tqdm
- schedulefree

PyTorch is intentionally not installed directly by HistVAE because the appropriate build depends on your CPU/CUDA environment. Install a suitable PyTorch build first, then install HistVAE. The packaged configuration explicitly uses `schedulefree.RAdamScheduleFree`; set `optimizer: radam` to use `torch.optim.RAdam` instead. HistVAE does not silently switch optimizers.

Example:

```bash
pip install torch torchvision
```

## Quick Start

The primary Python API is the `HistVAE` class. The CLI is a thin convenience wrapper around the same training workflow.

Example usage from Python with the packaged default config:

```python
from histvae import HistVAE
from histvae.utils import load_config

config, _ = load_config()
model = HistVAE(config=config, exp_name="example")
```

Example usage from Python with a user config that overrides the packaged default config:

```python
from histvae import HistVAE
from histvae.utils import load_config

config, _ = load_config("configs/pretrain.yaml")
model = HistVAE(config=config, exp_name="example")
```

Configuration priority is:

```text
packaged default config < user config file < CLI/runtime overrides
```

This means `src/histvae/config.yaml` provides the baseline defaults, an optional external YAML overrides those defaults, and runtime values such as `exp_name` and `device` are applied last.

### Histogram representation

Histogram construction can preserve the original bin-count representation or
remove group-size intensity with either a density or bounded probability mass:

```yaml
histogram_mode: count  # original behavior; default
# histogram_mode: density  # unit-integral density + legacy log/max scaling
# histogram_mode: probability_mass  # bin probabilities in [0, 1], sum to 1
```

The mode can also be overridden when preparing data. For one-dimensional
`FITC_Sum` values grouped by `sample_name`, the input arrays have shapes
`(n_rows, 1)` and `(n_rows,)` respectively:

```python
data = df[["FITC_Sum"]].to_numpy(dtype="float32")
group = df["sample_name"].to_numpy()

config["in_dims"] = 1
config["max_vals"] = [350000]

config["value_transform"] = "log1p"
config["out_of_range_policy"] = "clip"

model = HistVAE(config=config, exp_name="fitc-probability")
model.prep_data(
    train_data=data,
    train_group=group,
    histogram_mode="probability_mass",
)
```

`count` and `density` retain their legacy log/max normalization. The recommended
bounded representation for a sigmoid decoder is `probability_mass`, which
normalizes in-range bin counts to sum to one without the legacy scaling.
`value_transform: log1p` creates log-spaced bins while keeping `max_vals` in the
original data units. `out_of_range_policy: clip` collects underflow and overflow
in the edge bins instead of silently dropping them; `drop` preserves the prior
behavior and `error` rejects them.

By default, training uses two random point subsets per group, while validation
uses the full group and `z = mu`, making repeated validation and latent
extraction deterministic. Pretraining checkpoints are selected by
`pretrain_monitor` (`test_recon` in the packaged config). `model_best.pt`
contains the monitored best epoch and `model_last.pt` preserves the final epoch.

Command line usage with the packaged default config:

```bash
histvae --exp_name example --input_path path/to/input.csv
```

Command line usage with a user config override:

```bash
histvae --config_path configs/pretrain.yaml --exp_name example --input_path path/to/input.csv
```

## Project Structure

```text
repo
│
├─ src/
│  └─ histvae/
│     ├─ __init__.py
│     ├─ cli.py
│     ├─ core.py
│     ├─ config.yaml
│     ├─ data_handler.py
│     ├─ models.py
│     ├─ trainer.py
│     └─ utils.py
│
├─ tests/
├─ pyproject.toml
├─ README.md
└─ LICENSE
```

## Testing

Run the lightweight smoke tests used in CI with:

```bash
pytest -m smoke
```

End-to-end toy pretrain and finetune tests are also included, but they are marked as slow because they execute actual training loops. Run them explicitly with:

```bash
RUN_SLOW_HISTVAE_TESTS=1 pytest -m slow
```

## License

This project is released under the MIT License.

## Author

[Tadahaya Mizuno](https://github.com/tadahayamiz)

## Contact

For questions or collaboration inquiries:

`tadahaya[at]gmail.com`
