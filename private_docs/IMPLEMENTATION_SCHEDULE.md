# Implementation Schedule

Updated: 2026-06-20

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
- Assigned categorical encoding, unknown-category checks, and scaling to the
  experiment layer so the reusable model remains assay-independent.
- Corrected two-dimensional histogram axis ordering.
- Added focused grouped-measure and conditioning tests.
- Recorded the failed legacy sigmoid probability-mass pilot and the new
  implementation smoke.
- Cleared the reconstruction-first gate on the attached data: validation
  forward KL improved from the train-mean baseline `0.06056` to `0.02285` with
  `beta=0`.
- Added strict constant and linear-warmup latent-KL schedules, per-epoch beta
  logging, and beta provenance in best/last checkpoints.
- Verified a 50-epoch `beta=1e-4` warmup smoke with all four latent dimensions
  active and validation forward KL `0.02128`.

## Verification

- Full pytest including slow tests: 57 passed
- Latent-KL schedule focused tests: 10 passed
- Simplex output verified in 1D, 2D, and 3D
- Synthetic decoder-conditioned pretraining completed and wrote the canonical
  best checkpoint
- Attached-data one-epoch smoke:
  - 94 train / 20 validation groups
  - random 1,024-point input / full target
  - simplex and forward KL
  - target, input, and reconstruction mass-sum error <= approximately `1.2e-7`
  - finite training and validation metrics

## Next

- Run an unconditioned beta-target grid with at least three seeds.
- Keep `pretrain_monitor: test_recon` and compare against the fixed train-mean
  forward-KL baseline.
- Evaluate same-sample random-view latent stability and between-sample
  separation for each candidate checkpoint.
- Run technical-conditioning ablation only after reporting label-by-condition
  overlap.

## Deferred

- Optional event-mass or exposure-normalized-rate input/output branch
- Exact 1D W1 and multidimensional Sinkhorn auxiliary losses
- Point-coordinate jitter calibrated from technical controls
- Weak supervised disease head or contrastive disease-specific latent
- CLI repair or redesign

## Temporary shims

None.
