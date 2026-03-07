# HistVAE

HistVAE is a Python package for learning latent representations from point-cloud or histogram-style data using a Variational Autoencoder (VAE).

The package provides utilities for:

- histogram / point-cloud data preparation
- VAE-based representation learning
- pretraining and fine‑tuning workflows
- command‑line execution for experiments

This repository contains the reference implementation used in our research.

---

## Installation

Clone the repository and install it in editable mode:

```bash
git clone https://github.com/<your-repo>/histvae.git
cd histvae
pip install -e .
```

---

## Dependencies

HistVAE depends primarily on:

- Python ≥ 3.8
- PyTorch
- NumPy
- pandas
- PyYAML

Install PyTorch first if it is not already available in your environment.

Example:

```bash
pip install torch torchvision
```

---

## Quick Start

Example usage from Python:

```python
from histvae import HistVAE

model = HistVAE()
```

Command line usage:

```bash
histvae --config config.yaml
```

---

## Project Structure

```
repo
│
├─ src/
│   └─ histvae/
│
├─ tests/
│
├─ sample.ipynb
│
├─ pyproject.toml
├─ README.md
└─ LICENSE
```

---

## Testing

Run tests with:

```bash
pytest
```

---

## License

This project is released under the MIT License.

---

## Author

Tadahaya Mizuno  
University of Tokyo

---

## Contact

For questions or collaboration inquiries:

tadahaya[at]gmail.com
