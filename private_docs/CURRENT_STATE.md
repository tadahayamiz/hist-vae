# Current State

Updated: 2026-06-22

## Repository identity

HistVAE learns sample-level latent representations from grouped,
low-dimensional point data by converting each biological sample into a
histogram or empirical measure and applying a dimension-aware convolutional
VAE.

## Current objective

The finalized FITC model is a frozen absolute-coordinate `log1p`
probability-mass benchmark. It learns a stable representation of each sample's
empirical distribution and excludes total event abundance, but it retains
absolute distribution location. It is therefore not a device-invariant
shape-only model.

Because instrument-specific intensity shifts are expected, the active future
development direction is a shape-oriented raw-domain coordinate with
per-biological-sample robust normalization before histogram construction. The
attached assay is an application dataset, not the definition of the model; the
reusable grouped-measure contract remains applicable to low-dimensional events,
cells, particles, or image-derived spots.

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

The next shape-oriented coordinate study does not use plain raw absolute
`FITC_Sum`. It prespecifies:

```text
primary raw candidate:   x - group_median(x)
required raw comparator: x / group_median(x) - 1
completed log reference: log_median_center
IQR scaling:             sensitivity analysis only
```

Median centering removes additive device offsets while retaining width. Median
ratio removes multiplicative gain while retaining relative width. Technical
controls and synthetic shift tests, not reconstruction alone, determine which
invariance is appropriate.

The strict reusable `GroupCoordinateNormalizer` is now implemented upstream of
`HistogramPreprocessor`. It supports the four explicit modes in
R-260621-03, computes one full-group statistic before random-view sampling,
exports raw median/IQR/event-count summaries, persists a deterministic state,
and requires the global histogram geometry to replay the exact normalized
training rows. Non-`none` modes require `HistogramPreprocessor` axis transforms
to remain `none`.

After the coordinate mode is fixed, forward KL will be ablated against forward
KL plus a joint Sinkhorn divergence. The same joint OT definition will be used
in 1D, 2D, and 3D; exact 1D W1 is an evaluation check rather than a separate
training method.

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

## Next action

Keep the finalized absolute-coordinate checkpoints and their reporting artifact
chain frozen. Descriptive figures and three-seed latent exports may be generated
without model or seed selection.

The next one-theme task is a KL-only coordinate ablation on new development
data, comparing `raw_median_center` and `raw_median_ratio` with the completed
`log_median_center` reference. Do not use the finalized holdout. The ablation
must include additive/multiplicative synthetic shifts, reconstruction against
each model-specific mean baseline, random-view stability/retrieval, technical-
batch predictability, and shape-summary fidelity.

Only after one coordinate mode and any required beta retuning are fixed should
a separate one-theme change add joint Sinkhorn divergence as an auxiliary
observation loss across 1D/2D/3D.

## Deferred or out of scope

- Optional total mass or exposure-normalized event-rate branch
- Tail-sensitive or cancer-specific rare-event representation
- Median-plus-IQR normalization as a mainline rather than sensitivity analysis
- Point-coordinate measurement-noise augmentation
- Adversarial batch removal
- Prior-calibrated unconditional generation
- Clinical diagnostic claims
- CLI cleanup
