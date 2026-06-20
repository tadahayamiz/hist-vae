# E-260620-02: Grouped-measure implementation smoke

## 1. Purpose

Verify the first generic grouped empirical-measure implementation on unit tests
and the attached one-dimensional data, including optional decoder-side
technical conditioning.

## 2. Implemented contract

- Dimension-aware simplex-softmax decoder for 1D, 2D, and 3D
- Forward-KL reconstruction
- Random model input with deterministic full-group target
- Pooling of acquisition-only partitions through the user-defined sample group
- Probability-simplex protection against legacy histogram augmentation
- Optional decoder-only generic numeric conditioning
- Correct two-dimensional histogram axis ordering

## 3. Automated verification

- Full test suite including slow tests: 47 passed
- Simplex output tests covered 1D, 2D, and 3D
- Interleaved technical-partition rows produced the same pooled histogram as
  contiguous rows from the same group
- Synthetic decoder-conditioned pretraining and fine-tuning paths completed
- Strict invalid condition modes, widths, non-finite values, and within-group
  inconsistencies were rejected
- The existing positional `sample_latent` forward API remained valid

## 4. Attached-data smoke without technical conditioning

Conditions:

- 94 train and 20 validation groups
- `FITC_Sum`, 64 log1p-spaced bins, maximum 100,000, overflow clipped
- random 1,024-point training input and full-group target
- latent dimension 4, hidden dimensions `[4, 8]`
- simplex softmax, forward KL, `beta=0`, `condition_mode=none`
- one CPU epoch with RAdam

Results:

- Target, input, and reconstruction were finite probability masses.
- Reconstruction mass ranged from approximately `0.99999994` to `1.0`.
- Train loss after one epoch: approximately `2.6956`.
- Validation loss/reconstruction after one epoch: approximately `2.4324`.
- Active latent dimensions after one epoch: `0 / 4` at threshold `0.01`.

## 5. Attached-data conditioned execution smoke

A candidate technical batch was derived from the prefix of `data_24`. The eight
training categories were encoded as train-fitted one-hot vectors and supplied
through `condition_mode=decoder`; this derivation is an experiment-layer choice,
not a package-level assumption.

Results after one CPU epoch:

- All eight validation categories were represented in training.
- Full-target equality for a sampled training item was exact.
- Encoder `mu` and `logvar` were unchanged by decoder conditions.
- Reconstruction mass ranged from approximately `0.99999988` to `1.00000012`.
- Validation loss/reconstruction: approximately `2.4285`.

The one-epoch conditioned and unconditioned values are technical smoke results,
not evidence that conditioning improves deconfounding or representation quality.

## 6. Interpretation

The new observation-space contract executes correctly on the full attached
dataset. Acquisition `slice` values need no latent hierarchy because they tile
one sample region; pooling by `sample_name` is sufficient for the mainline.
The generic condition path can be enabled or disabled without changing the
encoder contract.

One epoch with `beta=0` does not establish representation quality or rule out
later activation of latent dimensions.

## 7. Next action

Run a longer reconstruction-first sanity pilot with technical conditioning off,
then compare optional decoder conditioning only after the base model clears the
train-mean distribution baseline and the label-by-condition contingency is
reported.

## 8. Related reference

- R-260620-01
