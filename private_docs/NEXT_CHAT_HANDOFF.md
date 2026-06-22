# Next Chat Handoff

Updated: 2026-06-22

## Current state

The existing probability-mass/simplex/forward-KL model and its 20-group holdout
are finalized. Its coordinate is absolute `log1p(FITC_Sum)` with a frozen
0-to-100,000 range. It is a stable absolute-coordinate distribution benchmark,
not a device-invariant shape-only model. Do not reopen its beta, architecture,
condition mode, checkpoint, seed, probe, or holdout.

A separate holdout-free pilot compared `log_median_center` with
`log_median_iqr`. Median centering was retained as the reference because it
preserved width and had stronger improvement over the mean-distribution
baseline, sample/view separation, and retrieval. IQR scaling remains
sensitivity analysis.

## New research decision

Instrument-specific intensity shifts are expected. Plain raw absolute
`FITC_Sum` is therefore not the future mainline. The next development cycle is
shape-oriented and prespecifies:

```text
raw_median_center = x - median_group(x)       # additive-shift invariant
raw_median_ratio  = x / median_group(x) - 1   # multiplicative-gain invariant
log_median_center                              # completed reference
```

Width, skewness, multimodality, and relative tails remain part of shape. Event
count/rate is stored separately because probability-mass normalization removes
it regardless of the coordinate.

`GroupCoordinateNormalizer` is now implemented as the strict upstream step.
It computes one full-group median before random sampling, supports negative
centered coordinates, rejects nonpositive ratio denominators, exports raw
median/IQR/event-count summaries, and persists a deterministic state. A
non-`none` coordinate mode requires a fitted `HistogramPreprocessor` with
pointwise transform `none`, and HistVAE verifies that its geometry was fitted
on the exact normalized training rows.

## Completed in the latest change

Phase 1 is complete. Focused tests cover additive and multiplicative
invariance, width preservation, negative centered coordinates, strict ratio
denominators, full-group statistic sharing, state replay, and wrong-geometry
rejection. Full pytest including slow tests passes.

## Next one theme

Run the KL-only coordinate ablation on new development data:

```text
raw_median_center
raw_median_ratio
log_median_center  # completed reference
```

Do not use the finalized holdout. Keep architecture, observation loss, split,
and seed matrix fixed. Selection must include synthetic additive and
multiplicative shift invariance, reconstruction versus each model-specific mean
baseline, random-view stability/retrieval, technical-batch predictability, and
shape-summary fidelity. All posterior-mean dimensions remain the formal latent
artifact.

## Following theme, not now

After fixing the coordinate and any required beta retuning, add an
observation-space joint Sinkhorn divergence to forward KL. Use the same joint OT
contract in 1D/2D/3D; do not average independent axis-wise distances. Exact 1D
CDF-W1 is a validation metric for the implementation, not a separate training
method.
