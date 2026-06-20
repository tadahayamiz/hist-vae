# Current State

Updated: 2026-06-20

## Repository identity

HistVAE learns sample-level latent representations from grouped,
low-dimensional point data by converting each biological sample into a
histogram or empirical measure and applying a dimension-aware convolutional
VAE.

## Current objective

Build a generic representation-learning path for biological samples whose
observations are sets of events, cells, particles, or image-derived spots. The
latent should describe stable distribution shape and location, while total
event mass/rate and technical covariates remain explicit optional components.

The attached single-molecule enzyme assay is the current validation dataset,
not the definition of the model.

## Sample contract

- `group` is the biological sample whose latent representation is required.
- Acquisition-only partitions such as image `slice` values are merged when
  they only tile a larger sample area.
- For the attached data, all rows with the same `sample_name` form one sample.
- Partition metadata may be retained for QC but is not a hierarchical encoder
  input in the mainline.

## Active mainline

- Legacy `count` and `density` paths remain available with sigmoid/MSE.
- The grouped empirical-measure path uses:

  ```yaml
  histogram_mode: probability_mass
  decoder_output_mode: simplex_softmax
  reconstruction_loss: forward_kl
  train_sampling_mode: random
  train_target_sampling_mode: full
  eval_sampling_mode: full
  eval_target_sampling_mode: full
  transform: false
  ```

- The simplex decoder supports one-, two-, and three-dimensional histograms and
  guarantees non-negative reconstructions whose total mass is one.
- The observation-space forward KL is separate from the usual latent VAE KL.
- Training can reconstruct a deterministic full-sample target from a random
  finite point subset.
- Technical conditioning is optional and decoder-only:
  `condition_mode: none | decoder`. The condition is a generic numeric
  sample-level vector; categorical batches are encoded outside the model.
- The encoder and exported latent mean never receive condition vectors.
- Decoder conditioning is an ablation tool, not a guarantee of deconfounding.
- Total event mass or exposure-normalized event rate is not yet part of the
  encoder; this remains a separate future branch.

## Active references

- `R-260619-00`: histogram representation mode
- `R-260620-01`: grouped-measure representation contract

## Active evidence

- `E-260619-00`: attached FITC density smoke validation
- `E-260620-00`: bounded-input and deterministic-evaluation smoke
- `E-260620-01`: FITC probability-mass pilot motivating a simplex decoder
- `E-260620-02`: grouped-measure implementation and attached-data smoke

## Next action

Run a reconstruction-first attached-data pilot with technical conditioning off,
small capacity, random input/full target, simplex softmax, forward KL, and
`beta=0`. Compare against a train-mean distribution baseline before adding
latent regularization.

After that base path clears the baseline, compare `condition_mode: none`
against `decoder` only where the technical categories overlap across biological labels.

## Deferred or out of scope

- Optional total mass/exposure-rate branch
- OT reconstruction auxiliary loss
- Point-coordinate measurement-noise augmentation
- Adversarial batch removal
- Causal claims under label-batch confounding
- CLI cleanup
