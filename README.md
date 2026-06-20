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

### Histogram and grouped-measure representations

The packaged defaults preserve the legacy count path:

```yaml
histogram_mode: count
decoder_output_mode: legacy_sigmoid
reconstruction_loss: mse
train_target_sampling_mode: paired
condition_mode: none
```

For distribution-shape learning from grouped low-dimensional points, use the
canonical grouped-measure path:

```yaml
histogram_mode: probability_mass
value_transform: log1p
out_of_range_policy: clip

decoder_output_mode: simplex_softmax
reconstruction_loss: forward_kl

train_sampling_mode: random
train_target_sampling_mode: full
eval_sampling_mode: full
eval_target_sampling_mode: full
transform: false

condition_mode: none
```

`probability_mass` converts every group to non-negative bin masses that sum to
one. `simplex_softmax` applies softmax over all spatial bins, so the decoder
returns the same type of object in 1D, 2D, and 3D. `forward_kl` is the
reconstruction divergence between the target and decoded distributions; the
usual VAE latent KL remains the separate term weighted by `beta`.

The grouped-measure denoising target can be fixed to the complete group while
the model input is generated from a random point subset. This estimates a
stable sample-level distribution rather than reconstructing one noisy subset
from another. The legacy independent random target remains available through
`train_target_sampling_mode: paired`.

Acquisition partitions such as image slices or tiles do not require a separate
model level when they only split one specimen's measured area. Give all points
the same group identifier and they are pooled before histogram construction.
Partition columns can remain in the source table for QC.

For one-dimensional `FITC_Sum` values grouped by `sample_name`:

```python
data = df[["FITC_Sum"]].to_numpy(dtype="float32")
group = df["sample_name"].to_numpy()

config["in_dims"] = 1
config["max_vals"] = [100000]
config["histogram_mode"] = "probability_mass"
config["value_transform"] = "log1p"
config["out_of_range_policy"] = "clip"
config["decoder_output_mode"] = "simplex_softmax"
config["reconstruction_loss"] = "forward_kl"
config["train_target_sampling_mode"] = "full"
config["transform"] = False

model = HistVAE(config=config, exp_name="fitc-measure")
model.prep_data(
    train_data=data,
    train_group=group,
)
```

### Optional decoder-side technical conditioning

Sample-level technical covariates can be excluded or supplied only to the
decoder through a generic numeric condition vector:

```yaml
condition_mode: decoder
condition_dim: 8
```

Each condition vector must be finite and constant within a group. Categorical
batch values should be converted outside HistVAE to one-hot or another explicit
numeric encoding whose mapping is fitted on the training split. Continuous
technical covariates can use the same API after train-fitted scaling:

```python
model.prep_data(
    train_data=train_data,
    train_group=train_group,
    train_condition=train_condition,
    test_data=test_data,
    test_group=test_group,
    test_condition=test_condition,
)
```

The encoder and `get_latent()` do not receive the condition vector. Decoder
conditioning therefore provides an optional nuisance-covariate path without
concatenating it to the returned latent, but it does not guarantee nuisance
invariance or resolve biological-label confounding. Runs with
`condition_mode: none` and `condition_mode: decoder` should be compared, and
technical-condition predictability from the latent should be reported.

### Latent-KL scheduling

Pretraining can use the configured latent-KL weight at every epoch or introduce
it gradually after reconstruction starts to form:

```yaml
beta: 0.0001
latent_kl_schedule: linear_warmup
latent_kl_warmup_epochs: 25
pretrain_monitor: test_recon
```

`constant` requires `latent_kl_warmup_epochs: 0`. `linear_warmup` uses zero
latent-KL weight at epoch 1, reaches `beta` at the requested warmup epoch, and
then keeps that value. Warmup requires reconstruction-based checkpoint
selection because total VAE loss changes as beta changes. Epochs before the
final beta is reached are logged but are not eligible for early stopping or the
best checkpoint, so `epochs` must be at least `latent_kl_warmup_epochs`. The
effective beta and monitoring eligibility are written to `history.json`, and
best/last checkpoint beta values are stored in their checkpoint files. The
schedule is applied by the pretrainer; fine-tuning with pretraining loss uses
the configured final `beta`.

Probability-mass normalization intentionally removes group size. Event count or
exposure-normalized event rate is a separate feature/modeling path and is not
implicitly contained in the current shape latent.

By default, full-group validation and `z = mu` make repeated validation and
latent extraction deterministic. Pretraining checkpoints are selected by
`pretrain_monitor`; `model_best.pt` contains the monitored best epoch and
`model_last.pt` preserves the final epoch.

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
