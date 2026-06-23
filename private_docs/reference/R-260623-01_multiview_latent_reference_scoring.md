# R-260623-01: Multi-view latent aggregation and reference scoring

Status: active

## 1. Purpose

This contract evaluates whether the existing HistVAE latent is a useful,
sampling-stable shape representation without changing the model, loss, or
training cohort.

The target pattern is not global Healthy-versus-PDAC separation. The intended
representation keeps Healthy and Healthy-like PDAC on the same main manifold
while allowing a minority of reproducibly atypical shapes to appear far from a
Healthy reference.

## 2. Model boundary

The selected HistVAE checkpoint remains frozen. This analysis adds no:

```text
view-consistency loss
metric-learning loss
diagnosis loss
posterior sampling
architecture change
checkpoint selection
```

Repeated event views are an inference-time measurement of finite-event
sampling stability. They do not change whether the model is a VAE.

## 3. Multi-view sample representation

For sample `i`, draw `R` deterministic point-subsample views under the fitted
histogram contract and encode each view without latent sampling:

```text
x_i^(r) -> q_phi(z | x_i^(r)) -> mu_i^(r), logvar_i^(r)
```

The sample position is:

```text
mu_bar_i = mean_r mu_i^(r)
```

Sampling instability is reported separately as the per-dimension SD and the
RMS Euclidean distance of `mu_i^(r)` from `mu_bar_i`.

A large, sample-consistent shape difference should remain displaced across
views. A displacement caused only by point-sampling noise should have a large
view spread or a low threshold-exceedance rate.

## 4. Healthy-reference latent score

The latent is standardized by a train-fitted transform fixed before reference
scoring. The anomaly score is the mean Euclidean distance to the `k` nearest
reference samples. Reference queries exclude themselves.

The implementation is label-agnostic. The experiment chooses train Healthy as
the normal reference, but the reusable functions accept arbitrary reference
indices.

## 5. Direct shape baseline

A latent is not required when direct histogram distance already answers the
question. Therefore every latent anomaly result must be compared with a direct
shape baseline.

For dimension-general evaluation, use the repository's joint Sinkhorn
divergence on the complete probability histogram support. The same contract
applies to 1D, 2D, and 3D; axis-wise marginal distances are not substituted.
The direct sample histogram for the multi-view comparison is the arithmetic
mean of the sampled probability-mass histograms.

The latent is useful only if it adds at least one of:

```text
better event-view stability
compact visualization
nonlinear organization of several normal modes
stable grouping of similar atypical shapes
useful information beyond direct joint-Sinkhorn distance
```

If these benefits are not observed, direct joint-Sinkhorn distance remains the
primary anomaly score and no metric-learning objective is introduced.

## 6. Fixed exploratory rule

The current exploratory comparison carries forward:

```text
reference k: 5
shape percentile threshold: 0.95
latent percentile threshold: 0.90
required latent-view exceedance rate: 0.80
```

These percentiles are recalibrated from the reference scores. Raw distance
thresholds are not transferred across models, dimensions, folds, or histogram
geometries.

## 7. Evidence boundary

Diagnosis labels may define the Healthy reference and color descriptive plots,
but they do not fit the encoder, PCA, UMAP, or distance metric. The opened
legacy holdout remains descriptive and must not drive threshold, checkpoint,
coordinate, architecture, or loss changes.

This analysis tests representation stability and direct-shape concordance. It
is not diagnostic validation and does not establish calibrated posterior
uncertainty.
