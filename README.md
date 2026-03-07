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

Install PyTorch first if it is not already available in your environment.

Example:

```bash
pip install torch torchvision
```

## Quick Start

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

Run tests with:

```bash
pytest
```

## License

This project is released under the MIT License.

## Author

[Tadahaya Mizuno](https://github.com/tadahayamiz)

## Contact

For questions or collaboration inquiries:

`tadahaya[at]gmail.com`
