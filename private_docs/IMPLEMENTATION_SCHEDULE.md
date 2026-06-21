# Implementation Schedule

Updated: 2026-06-21

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

- Freeze and document the selected configuration and exact checkpoint hashes.
- Generate raw-space figures and tables from the finalized artifacts without
  selecting a seed or changing the model.
- Export the three fixed-seed `mu` representations with explicit seed and
  split provenance.
- Use a new development protocol or independent cohort before testing any new
  disease-specific objective or model branch.

## Deferred

- Optional event-mass or exposure-normalized-rate input/output branch
- Tail-mass summaries or rare-event-aware loss
- Exact 1D W1 and multidimensional Sinkhorn auxiliary losses
- Point-coordinate jitter calibrated from technical controls
- Weak supervised disease head, one-class model, or contrastive
  disease-specific latent
- Prior-calibration and unconditional-generation evaluation
- CLI repair or redesign

## Temporary shims

None.
