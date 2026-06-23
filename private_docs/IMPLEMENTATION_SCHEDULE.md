# Implementation Schedule

Updated: 2026-06-23

## Done

### Core grouped-measure model

- Added strict `count`, `density`, and `probability_mass` histogram selection.
- Added explicit range handling and optional `log1p` bin geometry.
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

### Frozen absolute-coordinate benchmark

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
- Reclassified the absolute `log1p` model as a frozen benchmark rather than a
  device-invariant shape-only model.

### Visualization and train-fitted preprocessing

- Added coordinate-aware visualization for 1D and 2D histograms and
  reconstructions.
- Added raw and transformed bin-edge export, inverse-`log1p` coordinates,
  raw-coordinate density conversion for probability masses, and reproducible
  full/sampled reconstruction diagnostics.
- Added `AxisPreprocessingSpec` and a train-fitted
  `HistogramPreprocessor`.
- Added per-axis `none`/`log1p`, independent fixed/quantile lower and upper
  bounds, pooled-event or group-equal quantile weighting, and strict
  `clip`/`error` tail policies.
- Added safe-YAML preprocessing state, state hashes, train/test tail
  diagnostics, and strict reuse through `HistVAE.prep_data()`.

### Shape-coordinate development

- Recorded the holdout-free log-normalization pilot and retained
  `log_median_center` over median-plus-IQR scaling.
- Added strict `GroupCoordinateNormalizer` modes: `none`,
  `raw_median_center`, `raw_median_ratio`, and `log_median_center`.
- Applied the group-coordinate transform to complete groups before random-view
  sampling so random inputs and full targets share the same group statistic.
- Added strict positive-median checks for ratio normalization, negative
  centered-coordinate support, raw median/IQR/event-count summaries, and safe
  deterministic state serialization.
- Required non-`none` group-coordinate modes to use a fitted
  `HistogramPreprocessor` with pointwise transform `none`.
- Added exact normalized-training-data replay validation and restored both
  preprocessing states from saved configs.
- Completed the three-seed KL-only coordinate ablation without using the
  legacy holdout and selected `raw_median_ratio` under the prespecified
  multiplicative-gain invariance contract.

### Joint Sinkhorn implementation

- Added a strict, dimension-independent joint Sinkhorn-divergence config/API
  for probability-mass/simplex/forward-KL models.
- Added joint 1D/2D/3D bin-support construction from active histogram geometry,
  global metric scaling, strict backend validation, and a tensorized memory
  guard.
- Added base forward-KL, unweighted Sinkhorn, weighted Sinkhorn, and combined
  observation-loss logging to histories, checkpoints, and reconstruction
  exports.
- Added pinned `geomloss==0.3.1` and the explicit `ot-scalable` PyKeOps extra
  for online or multiscale backends.
- Added focused OT tests covering config strictness, joint support, identity,
  symmetry, distance ordering, autograd, 1D/2D/3D loss composition, artifact
  logging, and strict scalable-backend dependency handling.

### Joint Sinkhorn ablation and mainline selection

- Retrained KL-only references and OT candidates on the same branch snapshot,
  avoiding cross-environment replay as the comparison basis.
- Calibrated
  `lambda_equal = median_seed(median_train_KL / median_train_Sinkhorn)`.
- Obtained `lambda_equal=67.57868479947342` and tested OT factors
  `0.1`, `0.3`, and `1.0`.
- Recorded the complete three-seed OT ablation as E-260623-00.
- Selected `ot_factor=0.1`, corresponding to
  `ot_weight=6.7578684799473425`.
- Rejected factors `0.3` and `1.0` because they exceeded the predefined 10%
  forward-KL degradation gate.
- Fixed the development mainline as four-dimensional `raw_median_ratio` plus
  forward KL plus weak joint Sinkhorn.
- Fixed `raw_median_ratio_kl_sinkhorn_f0p10_seed73` as the single-checkpoint
  descriptive/export artifact while retaining three-seed summaries as the
  scientific result.
- Retained all four posterior-mean dimensions for formal export and downstream
  analysis; active-unit filtering remains diagnostic only.

### Multi-view reference-scoring support

- Recorded the completed collaborator follow-up and fixed Healthy-reference
  anomaly pilot as E-260623-01.
- Added inference-only `HistVAE.get_multiview_latent()` with deterministic
  child seeds, repeated sampled histograms, posterior parameters, mean
  posterior `mu`, view SD, and RMS view displacement.
- Kept the implementation independent of diagnosis, run IDs, filesystem
  paths, and histogram dimensionality.
- Added generic reference mean-kNN scoring, leave-one-out exclusions, and
  empirical reference percentiles.
- Added batched query-reference pairwise evaluation to the existing joint
  Sinkhorn module for 1D, 2D, and 3D probability histograms.
- Fixed R-260623-01 as the next analysis contract: compare aggregated latent
  anomaly with direct joint-Sinkhorn shape anomaly before considering any new
  training objective.

## Verification

### Implementation tests

```text
raw-space visualization focused tests: 8 passed
full pytest including slow tests after visualization: 65 passed
latent-KL schedule focused tests: 10 passed
simplex output verified in 1D, 2D, and 3D

focused HistogramPreprocessor tests: 10 passed
regular full pytest:                  70 passed, 5 skipped
full pytest including slow tests:    75 passed

focused normalizer tests:             7 passed
related preprocessing/measure tests: 39 passed
regular full pytest:                 77 passed, 5 skipped
full pytest including slow tests:    82 passed

focused OT tests:                    11 passed
related measure/preprocessor tests: 34 passed
regular full pytest:                 88 passed, 5 skipped
full pytest including slow tests:   93 passed
wheel build (`pip wheel --no-deps .`): passed

multiview/reference + OT focused:    19 passed
regular full pytest after install:   96 passed, 5 skipped
```

### Coordinate-ablation result

```text
selected coordinate:             raw_median_ratio
train / validation groups:       94 / 20
seeds:                           17, 42, 73
multiplicative factors:          0.5, 0.75, 1.5, 2.0
shifted input W1:                0 for every factor and seed
shifted latent distance:         0 for every factor and seed
self-retrieval:                  1.0 for every factor and seed
active dimensions:               4 / 4 for every seed
legacy holdout used:             no
```

### OT-weight calibration

```text
seed 17 lambda_equal: 66.221982
seed 42 lambda_equal: 73.841289
seed 73 lambda_equal: 67.578685
median lambda_equal:  67.57868479947342

factor 0.1 -> weight  6.7578684799473425
factor 0.3 -> weight 20.273605439842026
factor 1.0 -> weight 67.57868479947342
```

### Three-seed OT ablation

| Loss | OT factor | Forward KL | Sinkhorn | Exact W1 | Between/within | Retrieval | W1-latent | Gate |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| KL only | 0.0 | 0.013313 | 0.000358 | 0.004534 | 4.179 | 0.766 | 0.521 | pass |
| KL + Sinkhorn | **0.1** | **0.013428** | **0.000302** | **0.004274** | **4.189** | **0.778** | **0.521** | **pass** |
| KL + Sinkhorn | 0.3 | 0.015623 | 0.000312 | 0.004298 | 3.803 | 0.751 | 0.561 | fail |
| KL + Sinkhorn | 1.0 | 0.017357 | 0.000289 | 0.004236 | 3.530 | 0.731 | 0.649 | fail |

Selected factor `0.1` versus KL only:

```text
forward KL:       +0.87%
Sinkhorn:         -15.8%
exact 1D W1:      -5.7%
retrieval:        +1.25 percentage points
between/within:   maintained
W1-latent:        maintained
active dimensions: 4 / 4 in every seed
linear probe:      report-only; not used for selection
legacy holdout:    not used
```

### Frozen absolute-coordinate benchmark evidence

```text
mean full-group holdout forward KL:  0.006251 +/- 0.001777
train-mean baseline:                 0.051049
mean reconstruction improvement:     87.75%
random-view retrieval:               0.829 +/- 0.027
input-W1 / latent Spearman:          0.882 +/- 0.053
fixed linear-probe ROC AUC:          0.734 +/- 0.054
secondary probability-ensemble AUC:  0.774
```

This remains a separate frozen benchmark and is not validation of the selected
shape-plus-OT model.

## Next

- Run the frozen-model multi-view latent/direct-shape concordance artifact:
  - 30 deterministic point-subsample views per sample,
  - sample latent as mean posterior `mu`,
  - view SD and RMS displacement as sampling-instability diagnostics,
  - train-Healthy Euclidean mean-kNN score in standardized latent space,
  - train-Healthy joint-Sinkhorn mean-kNN score on mean sampled histograms,
  - train-fitted PCA/UMAP of the aggregated latent,
  - candidate overlap, score concordance, and view-threshold repeat rate.
- Keep the model, loss, coordinate, OT weight, latent width, checkpoint, and
  thresholds frozen.
- Use labels only to define the Healthy reference and descriptive colors/rates.
- If the latent does not add stability or organization over direct Sinkhorn
  distance, retain the direct metric and do not add metric-learning or
  view-consistency losses.
- Treat the legacy holdout as opened/descriptive and require a new independent
  cohort for confirmation.

## Deferred

- Optional event-mass or exposure-normalized-rate input/output branch
- Tail-mass summaries or rare-event-aware loss
- Median-plus-IQR normalization as a primary representation
- Eight-dimensional latent sensitivity analysis
- Point-coordinate jitter calibrated from technical controls
- Weak supervised disease head, one-class model, or contrastive
  disease-specific latent
- Prior-calibration and unconditional-generation evaluation
- CLI repair or redesign

## Temporary shims

None.
