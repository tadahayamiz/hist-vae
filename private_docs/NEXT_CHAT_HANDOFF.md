# Next Chat Handoff

Updated: 2026-06-22

## Current state

The frozen absolute-coordinate `log1p(FITC_Sum)` benchmark and its one-time
20-group holdout remain finalized. Do not reopen its beta, architecture,
condition mode, checkpoint, seed, probe, or holdout.

A separate development cycle has selected the shape coordinate without using
that holdout. The three-seed KL-only ablation compared:

```text
raw_median_center
raw_median_ratio
log_median_center
```

Under the prespecified positive multiplicative-gain assumption,
`raw_median_ratio = x / median_group(x) - 1` is the development mainline. It
was exactly invariant to 0.5x, 0.75x, 1.5x, and 2.0x synthetic gains after
per-group renormalization. `log_median_center` remains the sensitivity
reference. See E-260622-00.

## Completed in the latest code change

A strict observation-space OT config/API is now part of the reusable repo; no
notebook monkey-patch is required.

```yaml
ot_loss: none              # none or sinkhorn
ot_weight: 0.0
ot_p: 1                    # 1 or 2
ot_blur: 0.05
ot_scaling: 0.8
ot_backend: tensorized     # tensorized, online, multiscale
ot_mass_epsilon: 0.0
```

For `ot_loss: sinkhorn`, strict validation requires:

```text
histogram_mode: probability_mass
decoder_output_mode: simplex_softmax
reconstruction_loss: forward_kl
ot_weight > 0
```

HistVAE constructs the complete joint bin support from the active fitted
histogram geometry in 1D, 2D, or 3D. Each axis is globally mapped to `[0, 1]`
for the OT ground metric and the joint coordinate is divided by `sqrt(d)`.
Axis-wise marginal OT is never substituted. The observation loss is:

```text
base forward KL + ot_weight * debiased joint Sinkhorn divergence
```

History, checkpoints, and reconstruction exports now retain base KL,
unweighted Sinkhorn, weighted Sinkhorn, and combined observation loss
separately. `geomloss==0.3.1` is a core dependency. Online and multiscale
backends require the explicit `ot-scalable` PyKeOps extra; tensorized mode has a
strict pairwise-memory guard.

## Verification

```text
focused OT tests:             11 passed
related contract tests:       34 passed
regular full pytest:          88 passed, 5 skipped
full pytest including slow:   93 passed
wheel build:                  passed
```

No temporary shim or runtime loss monkey-patch remains.

## Next one theme

Run the development-only OT ablation with the coordinate and representation
settings fixed:

```text
coordinate: raw_median_ratio
loss A:     forward KL
loss B:     forward KL + joint Sinkhorn
split:      same train/validation development split
seeds:      17, 42, 73
holdout:    untouched
```

Calibrate a prespecified `ot_weight` grid from the fixed KL-only training-loss
scales, and compare base validation KL, Sinkhorn, exact 1D W1, quantile/tail
fidelity, random-view retrieval, between/within separation, and input-distance
versus latent-distance rank correlation. Do not select by combined loss across
different weights because its numeric scale changes with lambda.

## Fresh-VM notebook rule

Assume every Colab/VM starts empty. The next notebook must be self-contained:

```text
mount Drive/storage
-> clone repository
-> checkout one exact commit
-> install PyTorch-compatible repo dependencies
-> install the complete planned analysis stack at the start
-> validate/copy data and split artifacts
-> run training, evaluation, LP, UMAP, and export without relying on a prior VM
```

When scalable OT may be needed, install `.[ot-scalable]` in the first setup
cell. When linear probing, UMAP, or reporting is planned, install and include
those steps from the outset. Linear probing remains report-only unless a model-
selection role is prespecified. If a future cell does not need a change, state
explicitly that no change is required rather than reproducing it.
