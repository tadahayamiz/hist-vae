# Implementation Schedule

Updated: 2026-06-22

## Done

- Added strict `count`, `density`, and `probability_mass` histogram selection.
- Added explicit range handling and optional log1p bin geometry.
- Made validation and latent extraction deterministic and full-group based.
- Made best and last checkpoints explicit and saved config safe-loadable.
- Made optimizer and convolutional dropout configuration strict.
- Recorded that acquisition-only `slice` partitions are merged into the
  biological `group` rather than modeled as independent replicates.
- Added a dimension-aware `simplex_softmax` decoder for 1D, 2D, and 3D.
- Added `forward_kl` as a distribution reconstruction loss distinct from the
  latent VAE KL.
- Added `train_target_sampling_mode: full` for random-input/full-target
  denoising while retaining `paired` as the legacy option.
- Rejected legacy histogram-value augmentation for probability-mass inputs.
- Added optional decoder-only generic numeric conditioning with strict
  group-level constant vectors and no encoder access to conditions.
- Corrected two-dimensional histogram axis ordering.
- Added strict constant and linear-warmup latent-KL schedules, per-epoch beta
  logging, and beta provenance in best/last checkpoints.
- Cleared the reconstruction-first gate on the attached data.
- Completed the 3-seed beta convergence study with a 300-epoch ceiling and
  selected `beta=1e-4` by the predefined gate.
- Completed decoder-conditioning ablation and selected
  `condition_mode=none`.
- Performed the one-time fixed-seed holdout evaluation and wrote a finalized
  artifact chain.
- Confirmed on holdout that all four latent dimensions remain active, every
  sample beats the train-mean distribution baseline, random views retain
  sample identity, and latent distances preserve input W1 geometry.
- Added a single coordinate-aware visualization implementation for 1D and 2D
  histograms and reconstructions.
- Added raw and transformed bin-edge export, inverse-log1p raw coordinates,
  raw-coordinate density conversion for probability masses, and reproducible
  full/sampled reconstruction diagnostics.
- Updated `check_data()` and the new `plot_reconstruction()` high-level API to
  use raw coordinates by default.
- Added `AxisPreprocessingSpec` and a train-fitted `HistogramPreprocessor`.
- Added explicit per-axis `none`/`log1p`, independent fixed/quantile lower and
  upper bounds, pooled-event or group-equal quantile weighting, and strict
  `clip`/`error` tail policies.
- Added nonzero raw lower bounds and per-axis transforms to the reusable
  histogram and visualization geometry.
- Added safe-YAML preprocessing state, state hashes, train/test tail
  diagnostics, and strict reuse through `HistVAE.prep_data()`.
- Kept legacy drop behavior outside the fitted preprocessor for backward
  compatibility while preventing drop-and-renormalize in the new contract.
- Corrected the research terminology: the finalized `log1p` model is a frozen
  absolute-coordinate benchmark, not a device-invariant shape-only model.
- Recorded the holdout-free log-normalization pilot and retained
  `log_median_center` as the reference over median-plus-IQR scaling.
- Defined the future raw shape-coordinate candidates and a dimension-independent
  joint Sinkhorn-divergence ablation in R-260621-03.
- Added strict `GroupCoordinateNormalizer` modes: `none`,
  `raw_median_center`, `raw_median_ratio`, and `log_median_center`.
- Applied the group-coordinate transform to complete groups before random-view
  sampling, so random inputs and full targets share the same group statistic.
- Added strict positive-median checks for ratio normalization, negative
  centered-coordinate support, raw median/IQR/event-count summaries, and safe
  deterministic state serialization.
- Required non-`none` group-coordinate modes to use a fitted
  `HistogramPreprocessor` with pointwise transform `none`.
- Added exact normalized-training-data replay validation for fitted histogram
  geometry and restored both preprocessing states from saved configs.
- Completed the three-seed KL-only coordinate ablation without touching the
  finalized holdout and selected `raw_median_ratio` under the prespecified
  multiplicative-gain invariance contract.
- Added a strict, dimension-independent joint Sinkhorn-divergence config/API for
  probability-mass/simplex/forward-KL models.
- Added joint 1D/2D/3D bin-support construction from active histogram geometry,
  global metric scaling, strict backend validation, and a tensorized memory
  guard.
- Added base forward-KL, unweighted Sinkhorn, weighted Sinkhorn, and combined
  observation-loss logging to histories, checkpoints, and reconstruction
  exports.
- Added the pinned `geomloss==0.3.1` dependency and the explicit
  `ot-scalable` PyKeOps extra for online or multiscale backends.
- Added focused OT tests covering config strictness, joint support, identity,
  symmetry, distance ordering, autograd, 1D/2D/3D loss composition, artifact
  logging, and strict scalable-backend dependency handling.

## Verification

Implementation verification after the raw-space visualization update:

```text
raw-space visualization focused tests: 8 passed
full pytest including slow tests: 65 passed
actual-data raw-axis smoke: 0 to 100,000 FITC_Sum; density integral ~1
latent-KL schedule focused tests: 10 passed
simplex output verified in 1D, 2D, and 3D
```

Train-fitted preprocessing verification:

```text
focused HistogramPreprocessor tests: 10 passed
regular full pytest:                  70 passed, 5 skipped
full pytest including slow tests:    75 passed
actual-data smoke:                    519,118 rows; 94 train / 20 validation groups
fitted group-equal q0.999 upper:      97,626 FITC_Sum
train / validation upper-tail rate:  0.1065% / 0.0660%
train, validation, reconstruction:    finite simplex tensors of shape (*, 1, 64)
```

Group-coordinate normalization verification:

```text
focused normalizer tests:             7 passed
related preprocessing/measure tests: 39 passed
regular full pytest:                 77 passed, 5 skipped
full pytest including slow tests:    82 passed
actual-data smoke:                   519,118 rows; 134 groups
raw_median_center q0.001/q0.999:     -19,279 / 53,707
raw_median_ratio q0.001/q0.999:      -0.41854 / 1.34531
max abs normalized group median:     0.0 / 5.6e-17
```

Coordinate-ablation result:

```text
selected development coordinate: raw_median_ratio
validation groups / seeds:        20 / 3
multiplicative factors:           0.5, 0.75, 1.5, 2.0
shifted input W1:                 0 for every factor and seed
shifted latent distance:          0 for every factor and seed
self-retrieval:                   1.0 for every factor and seed
active dimensions:                4 / 4 for every seed
finalized holdout used:           no
```

Joint-Sinkhorn API verification:

```text
focused OT tests:                 11 passed
related measure/preprocessor:     34 passed
regular full pytest:              88 passed, 5 skipped
full pytest including slow:       93 passed
wheel build (`pip wheel --no-deps .`): passed
```

Final selected validation result:

```text
beta: 1e-4
condition_mode: none
mean validation forward KL: 0.006772 +/- 0.001440
active dimensions: 4 / 4 for every seed
all three seeds early-stopped before the 300-epoch ceiling
```

Final holdout result:

```text
mean full-group forward KL:          0.006251 +/- 0.001777
train-mean baseline:                0.051049
mean reconstruction improvement:    87.75%
random-view retrieval:              0.829 +/- 0.027
input-W1 / latent Spearman:         0.882 +/- 0.053
fixed linear-probe ROC AUC:         0.734 +/- 0.054
secondary probability-ensemble AUC: 0.774
```

## Next

- Keep the finalized absolute-coordinate checkpoints frozen and finish only
  descriptive figures, tables, and three-seed `mu` exports from those artifacts.
- Next one theme: run the development-only OT ablation on fixed
  `raw_median_ratio`, comparing forward KL with forward KL plus the implemented
  joint Sinkhorn divergence under the same split, architecture, beta, and
  seeds.
- Use a self-contained fresh-VM notebook that clones a pinned Git commit and
  installs every dependency needed by the complete run at the start, including
  scalable OT, linear-probe, UMAP, and reporting dependencies when those
  analyses are planned.
- Freeze all settings before one-time evaluation on a new independent holdout.

## Deferred

- Optional event-mass or exposure-normalized-rate input/output branch
- Tail-mass summaries or rare-event-aware loss
- Median-plus-IQR normalization as a primary representation
- Point-coordinate jitter calibrated from technical controls
- Weak supervised disease head, one-class model, or contrastive
  disease-specific latent
- Prior-calibration and unconditional-generation evaluation
- CLI repair or redesign

## Temporary shims

None.
