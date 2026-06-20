# Current State

Updated: 2026-06-20

## Repository identity

HistVAE converts grouped point data into histograms and learns latent
representations with a dimension-aware convolutional VAE.

## Current objective

Prepare reliable one-dimensional pretraining for `FITC_Sum` grouped by
`sample_name`, without forcing group-size/count intensity into the
representation and without breaking the original count-based path.

## Active mainline

- `histogram_mode` has three strict values: `count`, `density`, and
  `probability_mass`.
- `count` and `density` preserve the previous log1p/per-group-max path.
- `probability_mass` produces bounded bin probabilities in `[0, 1]` whose sum
  is 1 and is the recommended path for the current sigmoid decoder.
- `value_transform="log1p"` provides log-spaced bins while `max_vals` remains in
  the original data units.
- `out_of_range_policy` explicitly selects `drop`, `clip`, or `error`.
- Training defaults to random point subsets; evaluation defaults to full-group
  histograms and deterministic `z = mu` reconstruction.
- `model_best.pt` is restored and saved from the configured monitored epoch;
  `model_last.pt` preserves the final epoch.
- Optimizer choice is explicit (`radam_schedule_free` or `radam`), and
  `dropout_conv` now reaches every convolutional block.

## Active references

- `R-260619-00`: histogram representation mode
- `R-260620-00`: reliable pretraining representation and evaluation contract

## Active evidence

- `E-260619-00`: attached `FITC_Sum` density smoke validation
- `E-260620-00`: attached `FITC_Sum` probability-mass reliability smoke

## Next action

Run a controlled pilot over small model capacity, bin/range choices, beta, and
at least three seeds. Compare validation reconstruction, KL, active latent
coordinates, and downstream usefulness before starting full pretraining.

## Deferred or out of scope

- Full biological/statistical interpretation of the attached dataset
- Final hyperparameter selection or a claim that one representation is best
- Cleanup of the pre-existing CLI path
