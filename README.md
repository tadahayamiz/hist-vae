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

PyTorch is intentionally not installed automatically by `pip install histvae` because the appropriate build depends on your CPU/CUDA environment. Install a suitable PyTorch build first, then install HistVAE.

The optional `schedulefree` package is not required. If it is installed, HistVAE uses `schedulefree.RAdamScheduleFree`. Otherwise it falls back to `torch.optim.RAdam` with a small compatibility wrapper.

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

### Count and density histograms

Histogram construction can preserve the original bin-count representation or
remove group-size intensity by using a probability density:

```yaml
histogram_mode: count    # original behavior; default
# histogram_mode: density  # unit-integral histogram
```

The mode can also be overridden when preparing data. For one-dimensional
`FITC_Sum` values grouped by `sample_name`, the input arrays have shapes
`(n_rows, 1)` and `(n_rows,)` respectively:

```python
data = df[["FITC_Sum"]].to_numpy(dtype="float32")
group = df["sample_name"].to_numpy()

config["in_dims"] = 1
config["max_vals"] = [350000]

model = HistVAE(config=config, exp_name="fitc-density")
model.prep_data(
    train_data=data,
    train_group=group,
    histogram_mode="density",
)
```

`density` normalizes the histogram to unit integral before the existing
`log1p` and per-sample max scaling. `count` remains the default so existing
experiments retain their previous representation. Values outside the fixed
range defined by `max_vals` are excluded, so the range should be determined
from the training domain and recorded with the experiment.

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
