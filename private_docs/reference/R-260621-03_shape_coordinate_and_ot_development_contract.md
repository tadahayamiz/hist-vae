# R-260621-03: Shape-oriented raw-coordinate and OT development contract

Status: superseded
Updated: 2026-06-23

> Superseded by R-260623-00 after completion of the OT ablation. This record is
> retained as the development contract and implementation rationale.

## 1. Decision

The frozen FITC benchmark uses an absolute `log1p` coordinate. Probability-mass
normalization removes total event abundance, but it does not remove a shift of
the event-intensity coordinate. Because instrument-specific intensity shifts
are expected in this assay, that benchmark must not be called a
measurement-invariant shape-only model.

The next development cycle will therefore use a per-biological-sample robust
coordinate normalization before histogram construction. Plain raw absolute
`FITC_Sum` is not the future mainline. The scientific target is a distribution
shape representation that retains width, skewness, multimodality, and relative
tails while reducing device-dependent location shifts.

Total event count or exposure-normalized event rate remains a separate feature.
It is not recovered by switching from log to raw coordinates because the
histogram is still normalized to probability mass.

## 2. Coordinate decision after E-260622-00

For sample `s`, let `m_s` be the median of its complete raw event set `x_si`.
The KL-only development ablation fixed the coordinate under the stated assay
assumption that instrument intensity variation is primarily a positive
multiplicative gain.

### 2.1 Selected mainline: raw ratio to the median

```text
u_si = x_si / m_s - 1
```

This is exactly invariant to a positive multiplicative gain:

```text
x'_si = a * x_si, a > 0  =>  u'_si = u_si
```

It is dimensionless and preserves relative width, skewness, multimodality, and
relative tail position. It requires a finite strictly positive group median;
violating that condition is a hard error rather than a silent offset or
fallback.

In E-260622-00, all prespecified 0.5x, 0.75x, 1.5x, and 2.0x validation shifts
produced zero normalized input W1, zero standardized-latent shift, and perfect
self-retrieval for every seed. The finalized holdout was untouched.

### 2.2 Sensitivity reference: log median centering

`log_median_center` remains the sensitivity reference. It is approximately
multiplicative-invariant at the observed large positive intensities and showed
lower native validation forward KL, but it defines log-domain rather than
linear-ratio bin geometry.

### 2.3 Additive-offset reference

```text
u_si = x_si - m_s
```

`raw_median_center` is exactly invariant to additive offsets and preserves
width in raw measurement units. In the ablation it was not invariant to
multiplicative gains, so it is not the development mainline under the current
scientific contract.

`median + IQR` normalization remains sensitivity analysis only because dividing
by IQR removes sample-specific width. Plain raw absolute intensity is not a
shape-only selection candidate.

The coordinate decision is conditional on the stated technical mechanism.
Physical technical replicates should still test whether real device variation
contains a material additive component.

## 3. Processing and leakage boundary

The canonical order for a future sample is:

```text
raw grouped events
-> strict per-group coordinate normalization
-> train-fitted global histogram bounds
-> clipping diagnostics
-> joint probability-mass histogram
-> random-input / full-target HistVAE
```

Rules:

1. The group statistic is calculated from the complete biological sample and
   applied to all events in that sample.
2. The random training view and the deterministic full target use the same
   full-group statistic, matching the intended full-sample deployment artifact.
3. Validation and future holdout samples calculate their own group median, but
   never refit global histogram bounds, scaling, model parameters, or OT
   hyperparameters.
4. Global histogram bounds are fitted only on normalized training groups,
   preferably with `group_equal` quantile weighting and explicit `clip` tail
   diagnostics.
5. Labels and technical-batch identifiers do not enter the normalizer.
6. The exported artifact records the raw group median, raw IQR, event count,
   exposure-normalized rate when available, and the selected coordinate mode.

If a future deployment has access only to a small partial event set rather than
a full sample, view-local median estimation must be evaluated separately. The
current random-view stability metric does not test uncertainty in a median that
was estimated from the full group.

## 4. Current implementation boundary

`HistogramPreprocessor` provides pointwise `none`/`log1p` transforms,
train-fitted lower and upper bounds, group-equal quantiles, clipping diagnostics,
and persisted geometry. Group-dependent normalization is implemented separately
by `GroupCoordinateNormalizer` upstream of it. The configuration surface is
explicit:

```text
group_coordinate_mode:
  none
  raw_median_center
  raw_median_ratio
  log_median_center
```

No aliases, automatic guessing, or fallback conversion are provided. The
normalizer computes one statistic from every complete group before dataset
sampling. For a non-`none` mode, HistVAE requires a fitted
`HistogramPreprocessor` with pointwise transform `none` and validates that its
fit-data hash matches the normalized training rows exactly. Saved configs
restore and hash-check both states.

Implemented focused tests cover:

```text
- exact additive invariance of raw_median_center
- exact positive multiplicative invariance of raw_median_ratio
- preservation of width under median centering
- strict failure for a nonpositive ratio denominator
- support for negative centered coordinates
- one full-group statistic shared by random input and full target
- training-only fitting and exact replay of histogram geometry
- deterministic serialization of the selected coordinate contract
```

## 5. Coordinate-selection experiment: complete

E-260622-00 compared `raw_median_center`, `raw_median_ratio`, and
`log_median_center` with forward KL only, fixed architecture/beta/split, three
seeds, and no holdout access. Selection used reconstruction, random-view
stability, distribution/latent geometry, technical-batch diagnostics, and
prespecified synthetic shifts rather than native forward KL alone.

Three-seed means were:

| Coordinate | Val KL | Recon W1 | Between/within | Retrieval | W1-latent | Multiplicative latent/between | Multiplicative retrieval |
|---|---:|---:|---:|---:|---:|---:|---:|
| `log_median_center` | 0.012763 | 0.004571 | 3.970 | 0.745 | 0.657 | 0.000 | 1.000 |
| `raw_median_center` | 0.014440 | 0.004684 | 5.506 | 0.781 | 0.723 | 1.604 | 0.054 |
| `raw_median_ratio` | 0.013312 | 0.004534 | 4.178 | 0.765 | 0.521 | 0.000 | 1.000 |

All four latent dimensions were active for every seed. `raw_median_ratio` was
selected because its exact multiplicative-invariance contract matches the
stated technical assumption. Raw median and event count/rate remain separate
sample metadata.

## 6. Dimension-independent OT contract and implemented API

OT is a second ablation after fixing `raw_median_ratio`, architecture, and the
latent-KL schedule. The observation-space loss is:

```text
L_observation = KL(p || p_hat) + lambda_ot * S_epsilon(p, p_hat)
L_total = L_observation + beta(t) * KL(q(z | input) || N(0, I))
```

where the debiased entropic Sinkhorn divergence is:

```text
S_epsilon(P, Q)
  = OT_epsilon(P, Q)
  - 0.5 * OT_epsilon(P, P)
  - 0.5 * OT_epsilon(Q, Q)
```

The same mathematical contract is used for 1D, 2D, and 3D. The transport is
computed on the flattened joint histogram, never by averaging axis-wise
marginal distances. Diagonal, quadrant, and other cross-dimensional
dependencies therefore remain represented.

The ground support is constructed from the active histogram bin centers. Each
axis is mapped with its train-fitted global edge range into `[0, 1]`, then the
joint coordinate is divided by `sqrt(d)` so the enclosing Euclidean diameter is
at most one. This is a metric-only transformation; it does not change the
sample histograms or remove sample-specific width.

The strict repository configuration is now:

```yaml
ot_loss: none              # none or sinkhorn
ot_weight: 0.0
ot_p: 1                    # 1: Euclidean; 2: half squared Euclidean
ot_blur: 0.05
ot_scaling: 0.8
ot_backend: tensorized     # tensorized, online, multiscale
ot_mass_epsilon: 0.0
```

`ot_loss: sinkhorn` requires all of:

```text
histogram_mode: probability_mass
decoder_output_mode: simplex_softmax
reconstruction_loss: forward_kl
ot_weight > 0
```

`JointSinkhornDivergence` is differentiable and stores the normalized joint
support as a model buffer. `HistVAE` derives that support from the fitted
`HistogramPreprocessor` when present, otherwise from explicit fixed config
geometry. Non-`none` group-coordinate modes therefore use exactly the
normalized training geometry already recorded by the preprocessor.

`geomloss==0.3.1` is a core dependency. `tensorized` is the default and has a
strict pairwise-memory guard. `online` and `multiscale` preserve the same loss
contract but require the explicit `ot-scalable` PyKeOps extra; missing PyKeOps
is a hard error, not a fallback. Exact CDF-based 1D W1 remains an evaluation and
numerical-validation metric rather than a separate training objective.

Training history, checkpoint metadata, and deterministic reconstruction exports
record separately:

```text
base_reconstruction     # forward KL
sinkhorn                # unweighted divergence
weighted_sinkhorn       # lambda_ot * sinkhorn
observation             # base + weighted term
```

`pretrain_monitor: test_recon` continues to monitor the complete observation
loss for OT-enabled runs. Different lambda values must not be ranked by the
combined observation value alone because changing lambda changes its numeric
scale. Selection uses prespecified gates on base forward KL and compares
unweighted geometric/fidelity metrics across candidates.

`ot_blur` and `ot_weight` remain development hyperparameters. A practical
weight grid is scaled from fixed KL-only training medians so numeric units do
not determine the result. This is observation-space OT and is not WAE-style
latent-prior matching.

## 7. Ordered roadmap

```text
Phase 1: strict group-coordinate normalization                         [complete]
Phase 2: KL-only coordinate ablation                                   [complete]
Phase 3: fix raw_median_ratio under multiplicative-gain contract       [complete]
Phase 4a: strict joint Sinkhorn config/API and tests                    [complete]
Phase 4b: development-only KL versus KL + Sinkhorn ablation            [complete]
Phase 5: freeze all choices and evaluate a new independent holdout once [next]
```

Future execution notebooks must assume a fresh VM: clone the named
`dev-2026` branch, record its resolved HEAD, install the complete planned
dependency stack in the setup cell, and include reporting analyses such as
linear probing and UMAP from the outset when they are part of the run. No cell
may rely on state from a previous VM session.

## 8. Claim boundary

The completed absolute-coordinate benchmark remains valid evidence for its own
frozen data contract and is not evidence of device-shift invariance.
E-260622-00 verifies that `raw_median_ratio` exactly satisfies the prespecified
synthetic multiplicative-gain contract and supports its use as the development
mainline under that assumption. It does not prove that real device variation is
purely multiplicative or that this coordinate is universally superior.

The repository verifies that joint Sinkhorn is implemented, strict,
differentiable, and artifact-traceable in 1D/2D/3D. E-260623-00 completed
Phase 4b and selected weak OT (`ot_factor=0.1`,
`ot_weight=6.7578684799473425`) because it improved Sinkhorn and exact W1 with
minimal forward-KL cost. R-260623-00 is the current mainline reference.
