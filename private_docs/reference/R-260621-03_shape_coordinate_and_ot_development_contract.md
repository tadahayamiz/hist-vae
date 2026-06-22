# R-260621-03: Shape-oriented raw-coordinate and OT development contract

Status: active
Updated: 2026-06-22

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

## 2. Prespecified coordinate candidates

For sample `s`, let `m_s` be the median of its full set of raw event values
`x_si`.

### 2.1 Primary raw candidate: median centering

```text
u_si = x_si - m_s
```

This is exactly invariant to an additive device offset:

```text
x'_si = x_si + c  =>  u'_si = u_si
```

It preserves distribution width in the original measurement unit as well as
skewness, multiple modes, and relative tails. Negative normalized coordinates
are expected and valid.

### 2.2 Required raw comparator: ratio to the median

```text
u_si = x_si / m_s - 1
```

This is exactly invariant to a multiplicative gain:

```text
x'_si = a * x_si, a > 0  =>  u'_si = u_si
```

It is dimensionless and preserves relative width and relative tail position. It
requires a finite strictly positive group median; violating that condition is a
hard error rather than a silent offset or fallback.

### 2.3 Existing log reference

`log_median_center` remains the completed development reference. It is not the
raw candidate, but it provides a useful benchmark because a multiplicative gain
is approximately a translation after `log1p` for sufficiently large positive
values.

`median + IQR` normalization is not a main candidate because dividing by IQR
removes sample-specific width. It remains a sensitivity analysis only when
technical controls show that width itself is dominated by instrumentation.

The additive-versus-multiplicative mechanism must be determined empirically
from technical controls and prespecified synthetic-shift tests. Reconstruction
alone is not sufficient to choose between median centering and median ratio.

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

## 5. Coordinate-selection experiment

The first new experiment changes only the coordinate contract and keeps the
observation loss as forward KL. It compares:

```text
raw_median_center
raw_median_ratio
log_median_center  # completed reference
```

`median + IQR` is report-only sensitivity analysis. Plain raw absolute is a
technical diagnostic, not a selection candidate.

Use a new development protocol or independent cohort. The finalized holdout
from E-260621-02 is unavailable for this choice.

Selection must consider all of the following rather than native forward KL
alone, because different coordinate geometries allocate bins differently:

```text
- improvement over each model's train-mean distribution baseline
- full-group reconstruction and quantile/tail fidelity
- random-view within/between distance ratio and retrieval
- synthetic additive- and multiplicative-shift invariance
- technical-batch predictability from latent
- distribution-distance versus latent-distance rank correlation
- all latent dimensions retained for downstream analysis
- active-dimension count used only as a collapse diagnostic
```

Distance-based downstream analyses use all posterior-mean dimensions, with
standardization fitted on the training split. Near-zero-variance removal is a
numerical preprocessing step only, not the definition of the representation.

## 6. Dimension-independent OT contract

OT is a second ablation after the coordinate mode, architecture, and latent-KL
schedule are fixed. The observation-space loss is:

```text
L_recon = KL(p || p_hat) + lambda_ot * S_epsilon(p, p_hat)
L_total = L_recon + beta(t) * KL(q(z | input) || N(0, I))
```

where the debiased entropic Sinkhorn divergence is:

```text
S_epsilon(P, Q)
  = OT_epsilon(P, Q)
  - 0.5 * OT_epsilon(P, P)
  - 0.5 * OT_epsilon(Q, Q)
```

The same mathematical contract is used for 1D, 2D, and 3D. The transport plan
is computed on the joint histogram, not by averaging axis-wise marginal OT.
Therefore diagonal, quadrant, and other cross-dimensional dependencies remain
visible.

The ground cost uses Euclidean distance between joint bin centers after the
selected group-coordinate normalization. Each axis is scaled with its
train-fitted global range to a comparable `[0, 1]` metric range for the OT cost
only; this does not remove sample-specific width or alter the model histogram.

An exact CDF-based 1D W1 may be retained as a numerical validation metric for
the Sinkhorn implementation, but it is not a different 1D training objective.
Higher-dimensional implementations may use dense, sparse, separable, or
tensorized backends while preserving the same joint Sinkhorn-divergence
contract.

`epsilon` and `lambda_ot` are selected on development data only. A practical
lambda grid should be scaled from the median forward-KL and Sinkhorn magnitudes
of the fixed KL-only model so that numeric units do not determine the result.
This is observation-space OT and must not be confused with WAE-style latent
prior matching.

## 7. Ordered roadmap

```text
Phase 1: implement and test strict group-coordinate normalization [complete]
Phase 2: run the KL-only coordinate ablation on new development data
Phase 3: fix the raw shape coordinate and, if necessary, retune beta
Phase 4: ablate forward KL versus forward KL + joint Sinkhorn divergence
Phase 5: freeze all choices and evaluate a new independent holdout once
```

Do not implement the group-coordinate change and OT loss in one code request.
The next one-theme task is Phase 1.

## 8. Claim boundary

The completed absolute-coordinate benchmark remains valid evidence for its own
frozen data contract. It is not evidence that the latent is invariant to device
intensity shifts. The completed log-normalization pilot supports median
centering over IQR scaling within the tested log family, but it does not yet
establish that a raw-domain candidate is superior or that Sinkhorn improves the
representation.
