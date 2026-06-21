# R-260620-01: Grouped-measure representation contract

Status: active
Updated: 2026-06-21

## 1. Scope

HistVAE is intended to learn one latent representation per biological or
experimental sample from a low-dimensional collection of detected points. A
point may be a molecule, cell, particle, or event, and each coordinate may be a
measured intensity or feature. The contract is generic and is not tied to the
current single-molecule enzyme assay.

The atomic observation is the user-provided `group`. In the attached assay,
`sample_name` is the group and `slice` is only an acquisition partition of a
larger imaged region. All slices belonging to one sample are pooled by default.
A slice is not treated as a biological replicate, an independent sample, or a
separate latent level.

Partition identifiers may later be used for QC or explicit block-resampling,
but they are not part of the main model input. Random point sampling is
performed from the pooled group.

## 2. Canonical shape representation

For the distribution-shape path, grouped point observations are binned directly
as a probability mass `p` satisfying:

```text
p >= 0
sum(p) = 1
```

An equivalent count histogram would be converted by dividing by its total
count; an equivalent density histogram would be converted by multiplying by
bin volume and normalizing. The current public data path constructs the mass
from grouped points rather than accepting precomputed histograms.

This contract is the same for one-, two-, and three-dimensional histograms.
The convolution type changes with dimension, but the output constraint and
reconstruction loss do not.

The decoder mode for this path is `simplex_softmax`. Its final layer has no
BatchNorm, residual addition, dropout, or sigmoid. Softmax is applied over all
spatial bins, so every reconstructed sample is a valid probability mass.

The first reconstruction loss is forward KL:

```text
KL(target_mass || reconstructed_mass)
```

This is distinct from the VAE latent KL between the encoder posterior and the
latent prior. OT is a deferred geometry-aware auxiliary loss; it is not needed
to establish the canonical measure contract.

Legacy `count` and `density` behavior remains available through
`legacy_sigmoid + mse`.

## 3. Denoising contract

The recommended grouped-measure training views are:

```text
input  = histogram from a random point subset of the pooled group
target = deterministic histogram from all points in the pooled group
```

This is configured with:

```yaml
train_sampling_mode: random
train_target_sampling_mode: full
eval_sampling_mode: full
eval_target_sampling_mode: full
```

The model therefore learns to recover a stable group-level distribution from
finite event sampling. The legacy independent random-target behavior remains
available as `train_target_sampling_mode: paired`.

The legacy post-histogram augmentation does not preserve the probability
simplex and is rejected for `probability_mass`. Measure-path denoising should
come from point subsampling or a separately specified point-coordinate
corruption.

## 4. Acquisition partitions

Acquisition partitions such as image tiles, fields, slices, or chunks are
pooled when they cover one sample and differ only because the measured region
was split for acquisition or storage.

For the attached data:

```text
group = sample_name
partition/QC column = slice
```

The shape histogram is built from all detected points across the 12 slices.
If event abundance is later modeled, numerator and exposure denominator must be
aggregated across slices before constructing the sample-level rate. Per-slice
metadata repeated on every point row must first be deduplicated at the
`sample_name + slice` level.

A partition-aware hierarchy is deferred unless there is evidence that the
partition represents a real repeated measurement or biologically meaningful
spatial unit. Partition identifiers such as the current `slice` must not be
passed through the sample-level condition interface because they vary within a
group; they remain available for QC or block-resampling outside the encoder.

## 5. Optional decoder-side technical conditioning

Known sample-level technical covariates are controlled by a generic numeric
condition interface:

```yaml
condition_mode: none
condition_dim: 0
```

or:

```yaml
condition_mode: decoder
condition_dim: K
```

`none` excludes condition information. `decoder` concatenates a finite numeric
condition vector to the latent only before decoding. The encoder and returned
latent mean never receive the condition vector. This gives the decoder a
separate path for known acquisition shifts without defining an assay-specific
batch architecture.

Condition vectors must be constant within each group. Categorical technical
batches must be encoded outside HistVAE, for example as train-fitted one-hot
vectors. Continuous technical variables may use the same API after scaling
parameters are fitted on the training split. The experiment layer is
responsible for applying the same category mapping/scaling to validation and
holdout data and for rejecting unknown categories rather than silently changing
condition width.

Decoder conditioning is optional and does not mathematically guarantee a
condition-invariant latent representation. Both `none` and `decoder` must be
compared, and technical-condition predictability from the latent must be
reported. When technical condition and the biological label are strongly
confounded, neither setting supports a claim that technical and biological
effects have been identified separately. The unconditioned run remains the
primary reference whenever such overlap is insufficient.

## 6. Shape versus abundance/intensity

The event coordinate itself, such as `FITC_Sum`, is part of the shape
distribution and is retained through its location on the histogram grid.

The number or rate of detected events is a separate quantity. Probability-mass
normalization intentionally removes it. A later abundance path should use an
explicit sample-level scalar, preferably a rate with a known exposure
denominator, rather than asking the shape encoder to infer group size.

For the attached assay, the candidate rate is based on aggregated
`Activity_Spot_Number / All_well_count`. This path is not implemented in the
current phase because its exposure and likelihood contract must be fixed
separately. Until then, the canonical latent represents distribution shape,
not event abundance.

## 7. Current implementation boundary

Implemented in this phase:

```text
- dimension-generic simplex decoder for 1D/2D/3D
- forward-KL reconstruction for probability mass
- random-input / full-target denoising
- strict rejection of legacy histogram augmentation on probability mass
- optional decoder-only numeric technical conditioning
- corrected 2D histogram axis order
```

Deferred:

```text
- event-rate or total-mass encoder/decoder head
- OT reconstruction auxiliary loss
- free-bits or alternative latent regularization beyond R-260620-02
- partition-aware resampling
- adversarial batch removal
- disease-label supervision
```

## 8. Evaluation contract

The main latent artifact is the deterministic posterior mean from the pooled
full-group histogram. Evaluation must include:

```text
- validation forward KL and a train-mean distribution baseline
- repeated-subsample latent stability within each group
- active latent coordinates and latent posterior statistics
- relation between input-distribution distances and latent distances
- batch predictability from latent, when batch metadata exists
- biological-label linear probing without using labels in pretraining
```

Holdout groups remain excluded from model fitting and model selection.

## 9. Empirical selection outcome

The current attached-data selection is recorded in R-260621-00 and
E-260621-00 through E-260621-02. The selected shape-only configuration uses
`condition_mode: none`. Decoder-only conditioning remains available for future
experiments but was rejected for this dataset because it did not consistently
improve reconstruction or latent geometry and did not reduce measured batch
signal.

The grouped-measure contract itself remains generic; this empirical selection
does not remove the optional condition API or imply that conditioning will
never be useful on a better-balanced experimental design.
