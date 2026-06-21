# Current State

Updated: 2026-06-21

## Repository identity

HistVAE learns sample-level latent representations from grouped,
low-dimensional point data by converting each biological sample into a
histogram or empirical measure and applying a dimension-aware convolutional
VAE.

## Current objective

The current shape-only mainline is complete for the attached single-molecule
enzyme-activity dataset. It learns a stable latent representation of each
sample's empirical distribution while excluding total event abundance and
technical batch information by default.

The attached assay is the validation dataset, not the definition of the model.
The reusable contract remains applicable to grouped low-dimensional events,
cells, particles, or image-derived spots.

## Sample contract

- `group` is the biological sample whose latent representation is required.
- Acquisition-only partitions such as image `slice` values are merged when
  they tile a larger measured region.
- For the attached data, all rows with the same `sample_name` form one sample.
- `slice` remains QC or future block-resampling metadata; it is not a separate
  latent level in the selected model.

## Selected mainline

```yaml
histogram_mode: probability_mass
value_transform: log1p
out_of_range_policy: clip
max_vals: [100000.0]
bins: 64

train_sampling_mode: random
train_target_sampling_mode: full
eval_sampling_mode: full
eval_target_sampling_mode: full
num_points: 1024
transform: false

latent_dim: 4
hidden_dims: [8, 16]
dropout_conv: 0.0

decoder_output_mode: simplex_softmax
reconstruction_loss: forward_kl

beta: 0.0001
latent_kl_schedule: linear_warmup
latent_kl_warmup_epochs: 25
pretrain_monitor: test_recon

epochs: 300
patience: 20
optimizer: radam

condition_mode: none
condition_dim: 0
```

The epoch count is a ceiling. The selected three seeds stopped between epochs
110 and 186 after patience-based early stopping.

Legacy `count` and `density` paths remain available with sigmoid/MSE. Optional
decoder-only numeric conditioning remains implemented but was not selected for
the current dataset.

## Final evidence status

The beta and condition mode were selected using train/validation data only.
The holdout was then evaluated once with all three fixed seeds and finalized.

Across the 20 holdout groups:

```text
mean full-group forward KL:           0.006251 +/- 0.001777
train-mean forward-KL baseline:       0.051049
mean reconstruction improvement:      87.75% +/- 3.48%
groups beating the baseline:          20 / 20 for every seed
active latent dimensions:             4 / 4 for every seed
between/within random-view ratio:      5.146 +/- 0.502
random-view retrieval accuracy:       0.829 +/- 0.027
input-W1 / latent-distance Spearman:  0.882 +/- 0.053
```

The selected model therefore passes the intended sample-representation gate:
it reconstructs unseen sample distributions, remains stable under finite-event
subsampling, and preserves much of the input-distribution geometry.

The fixed disease-label probe produced mean ROC AUC `0.734 +/- 0.054`; the
secondary three-seed probability average produced AUC `0.774`. This is
exploratory evidence of weak disease-related information, not diagnostic
validation. The overall C and PC distributions are visually similar, and only
a subset of PC samples may contain a specific component.

## Model interpretation

The model is non-collapsed and suitable for deterministic sample
representation via full-group posterior mean `mu`. Because the selected beta
is small, posterior standard deviations are narrow, and latent KL remains
substantial, it is best described as a weakly VAE-regularized denoising
distributional autoencoder.

The current evidence does not establish calibrated posterior uncertainty,
realistic unconditional generation from the standard-normal prior, causal
batch correction, or clinical diagnostic performance.

## Active references

- `R-260619-00`: histogram representation mode
- `R-260620-01`: grouped-measure representation contract
- `R-260620-02`: latent-KL schedule contract
- `R-260621-00`: selected mainline and interpretation
- `R-260621-01`: raw-space histogram and reconstruction visualization
- `R-260621-02`: train-fitted histogram preprocessing

## Active evidence

- `E-260619-00`: attached FITC density smoke validation
- `E-260620-00`: bounded-input and deterministic-evaluation smoke
- `E-260620-01`: legacy sigmoid probability-mass failure
- `E-260620-02`: grouped-measure implementation smoke
- `E-260620-03`: reconstruction gate and latent-KL warmup smoke
- `E-260621-00`: multi-seed beta convergence and selection
- `E-260621-01`: decoder-conditioning ablation
- `E-260621-02`: finalized holdout evaluation

## Next action

Freeze the current shape-only model and artifact chain. Raw-space histogram
and reconstruction visualization is implemented, including inverse log1p bin
edges and raw-coordinate density for probability-mass plots.

For future datasets and new development cycles, use the train-fitted
`HistogramPreprocessor` contract rather than fitting ranges in notebook code.
Pass it to the `HistVAE` constructor so its state becomes authoritative before
model-contract validation. It makes log scaling optional per axis, supports
fixed or percentile bounds, can weight percentiles equally by biological
group, records tail fractions, and reuses the exact fitted state on validation
and holdout data. This new API does not reopen or alter the finalized FITC
model.

The next reporting work should generate reproducible figures and fixed-seed
latent exports from the finalized artifacts without further use of the
holdout for selection.

Any abundance/rate branch, tail-sensitive objective, OT auxiliary loss,
supervised disease head, or one-class model is a new development theme and
must use nested development evaluation or independent data.

## Deferred or out of scope

- Optional total mass or exposure-normalized event-rate branch
- Tail-sensitive or cancer-specific rare-event representation
- OT reconstruction auxiliary loss
- Point-coordinate measurement-noise augmentation
- Adversarial batch removal
- Prior-calibrated unconditional generation
- Clinical diagnostic claims
- CLI cleanup
