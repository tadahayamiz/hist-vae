# Current State

Updated: 2026-06-22

## Repository identity

HistVAE learns sample-level latent representations from grouped,
low-dimensional point data by converting each biological sample into a
histogram or empirical measure and applying a dimension-aware convolutional
VAE.

## Current objective

The finalized FITC model remains a frozen absolute-coordinate `log1p`
probability-mass benchmark. It is stable under finite-event subsampling but is
not a device-invariant shape-only model.

The active development mainline is now `raw_median_ratio`, selected in the
holdout-free coordinate ablation E-260622-00 under the prespecified assumption
that instrument intensity variation is primarily a positive multiplicative
gain. The coordinate is

```text
u_si = x_si / median_group(x_s) - 1
```

and therefore retains relative width, skewness, multimodality, and relative
tails while removing group-level gain. `log_median_center` remains the
sensitivity reference. Total event abundance and the removed raw median remain
separate sample metadata rather than implicit components of the shape latent.

The next empirical question is whether adding joint observation-space
Sinkhorn divergence to forward KL improves geometric reconstruction or latent
quality. The strict reusable OT config/API is implemented; its benefit has not
yet been established by the ablation.

## Sample contract

- `group` is the biological sample whose latent representation is required.
- Acquisition-only partitions such as image `slice` values are merged when
  they tile a larger measured region.
- For the attached data, all rows with the same `sample_name` form one sample.
- `slice` remains QC or future block-resampling metadata; it is not a separate
  latent level in the selected model.

## Frozen absolute-coordinate benchmark

```yaml
histogram_mode: probability_mass
value_transform: log1p
group_coordinate_mode: none
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

## Active development direction

The KL-only coordinate ablation is complete:

```text
main:                    raw_median_ratio
sensitivity reference:   log_median_center
additive-shift reference: raw_median_center
IQR scaling:              sensitivity analysis only
```

`raw_median_ratio` was exactly invariant to all prespecified 0.5x, 0.75x,
1.5x, and 2.0x synthetic gains after each validation group was renormalized:
input W1 and latent shift were zero and self-retrieval remained one for every
seed. This is a mathematical contract check, not proof that all real device
variation is purely multiplicative.

The strict reusable `GroupCoordinateNormalizer` remains upstream of the
train-fitted `HistogramPreprocessor`. Non-`none` coordinate modes require
pointwise histogram transform `none`, and global histogram geometry is fitted
on normalized training groups only.

The repository now exposes a strict joint-Sinkhorn observation-loss API:

```yaml
reconstruction_loss: forward_kl
ot_loss: sinkhorn
ot_weight: <positive lambda>
ot_p: 1
ot_blur: 0.05
ot_scaling: 0.8
ot_backend: tensorized  # online/multiscale require the ot-scalable extra
ot_mass_epsilon: 0.0
```

The implemented observation term is
`forward KL + ot_weight * joint Sinkhorn divergence`. Joint bin centers are
constructed from the active train-fitted histogram geometry, each axis is
scaled globally for the ground metric, and the complete 1D/2D/3D joint support
is transported. Axis-wise marginal OT is not used. Training history,
checkpoints, and reconstruction exports report base KL, unweighted Sinkhorn,
weighted Sinkhorn, and their combined observation loss separately.

## Final evidence status

The beta and condition mode of the frozen absolute-coordinate benchmark were
selected using train/validation data only. Its holdout was then evaluated once
with all three fixed seeds and finalized. These results apply to that coordinate
contract and do not establish invariance to device intensity shifts.

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

The frozen benchmark therefore passes the intended sample-representation gate:
it reconstructs unseen sample distributions, remains stable under finite-event
subsampling, and preserves much of the input-distribution geometry.

The fixed disease-label probe produced mean ROC AUC `0.734 +/- 0.054`; the
secondary three-seed probability average produced AUC `0.774`. This is
exploratory evidence of weak disease-related information, not diagnostic
validation. The overall C and PC distributions are visually similar, and only
a subset of PC samples may contain a specific component.

## Model interpretation

The frozen benchmark is non-collapsed and suitable for deterministic sample
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
- `R-260621-00`: frozen absolute-coordinate grouped-measure benchmark
- `R-260621-01`: raw-space histogram and reconstruction visualization
- `R-260621-02`: train-fitted histogram preprocessing
- `R-260621-03`: shape-oriented raw-coordinate and joint-OT development contract

## Active evidence

- `E-260619-00`: attached FITC density smoke validation
- `E-260620-00`: bounded-input and deterministic-evaluation smoke
- `E-260620-01`: legacy sigmoid probability-mass failure
- `E-260620-02`: grouped-measure implementation smoke
- `E-260620-03`: reconstruction gate and latent-KL warmup smoke
- `E-260621-00`: multi-seed beta convergence and selection
- `E-260621-01`: decoder-conditioning ablation
- `E-260621-02`: finalized absolute-coordinate holdout evaluation
- `E-260621-03`: log-domain shape-normalization pilot; holdout untouched
- `E-260622-00`: KL-only raw shape-coordinate ablation; holdout untouched

## Next action

Keep the finalized absolute-coordinate checkpoints and their reporting artifact
chain frozen. The next one-theme task is the development-only OT ablation on
`raw_median_ratio`:

```text
forward KL
versus
forward KL + joint Sinkhorn divergence
```

Hold split, architecture, beta, seed matrix, random/full sampling contract, and
all downstream evaluation rules fixed. Select `ot_blur` and `ot_weight` on
development data only; do not use the finalized holdout. Compare base forward
KL, Sinkhorn, exact 1D W1, quantile/tail fidelity, random-view retrieval,
between/within separation, and input-distance/latent-distance rank
correlation. A new independent holdout is evaluated once after all settings are
frozen.

## Fresh-VM execution policy

Notebook execution must assume that every Colab/VM session starts empty. Each
future experiment notebook therefore begins with one self-contained setup cell
that mounts storage, clones and pins the exact Git commit, installs the repo and
all dependencies required by the whole planned run, copies or validates input
artifacts, and records provenance. No later cell may depend on packages or
variables from an earlier VM session. When linear probing, UMAP, reporting, or
a scalable OT backend is part of the planned run, their dependencies and cells
are included from the outset. A linear probe is report-only unless its model-
selection role is prespecified before execution.

## Deferred or out of scope

- Optional total mass or exposure-normalized event-rate branch
- Tail-sensitive or cancer-specific rare-event representation
- Median-plus-IQR normalization as a mainline rather than sensitivity analysis
- Point-coordinate measurement-noise augmentation
- Adversarial batch removal
- Prior-calibrated unconditional generation
- Clinical diagnostic claims
- CLI cleanup
