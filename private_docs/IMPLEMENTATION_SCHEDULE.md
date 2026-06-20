# Implementation Schedule

Updated: 2026-06-20

## Done

- Added strict `count` / `density` histogram selection while preserving the
  original count default.
- Added bounded `probability_mass` histograms for sigmoid-decoder pretraining.
- Added explicit `drop` / `clip` / `error` range handling and optional log1p bin
  geometry.
- Made validation use full-group histograms and deterministic latent means.
- Made `get_latent()` deterministic and independent of sampled dataset views.
- Added configurable pretraining checkpoint monitoring and active-latent
  diagnostics.
- Made `model_best.pt` canonical and added `model_last.pt`.
- Made saved YAML safe-loadable and checkpoint metadata explicit.
- Wired `dropout_conv` through encoder and decoder blocks.
- Replaced optimizer auto-fallback with strict optimizer selection.
- Added focused reliability tests and attached-data smoke validation.

## Verification

- Related tests: 17 passed
- Smoke tests: 25 passed, 5 deselected
- Full pytest with slow tests enabled: 30 passed
- Attached CSV reliability smoke: 519,118 rows; 94 train and 20 validation
  groups; probability-mass inputs finite, bounded, and sum to one; repeated
  validation metrics exactly equal; one-epoch training and safe artifact reload
  completed.

## Next

- Run a three-seed pilot using smaller latent/hidden dimensions.
- Compare linear and log1p bins plus robust training-only range choices.
- Compare beta values and inspect reconstruction, KL, active latent dimensions,
  and downstream sample-level separation.

## Deferred

- CLI repair or redesign
- Broad preprocessing refactor
- Full production pretraining before pilot selection

## Temporary shims

None.
