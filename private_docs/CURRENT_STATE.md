# Current State

Updated: 2026-06-19

## Repository identity

HistVAE converts grouped point data into histograms and learns latent
representations with a dimension-aware convolutional VAE.

## Current objective

Support one-dimensional `FITC_Sum` data grouped by `sample_name` without
forcing group-size/count intensity into the representation, while preserving
the original count-based behavior for existing experiments.

## Active mainline

- `histogram_mode` has two strict values: `count` and `density`.
- `count` is the packaged default and preserves prior behavior.
- `density` uses NumPy unit-integral histogram density before the existing
  `log1p` and per-group max scaling.
- `HistVAE.prep_data(..., histogram_mode=...)` provides a runtime override and
  records the selected mode in the experiment config.
- The 1D data-to-model path is verified with shape `(batch, 1, bins)`.

## Active references

- `R-260619-00`: histogram representation mode

## Active evidence

- `E-260619-00`: attached `FITC_Sum` density smoke validation

## Next action

Define the analysis split and a reproducible experiment config for the attached
dataset. Fit histogram range/bin choices on training data only before model
comparison.

## Deferred or out of scope

- Full biological/statistical interpretation of the attached dataset
- Density-versus-count ablation and model-quality comparison
- Cleanup of the pre-existing CLI path
