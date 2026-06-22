# R-260621-02: Train-fitted histogram preprocessing contract

Status: active
Updated: 2026-06-21

## 1. Purpose

Histogram transforms and numeric ranges are part of the learned representation.
They must therefore be fitted on training data only, persisted as an artifact,
and reused unchanged for validation, holdout, plotting, and checkpoint replay.

The library must not silently choose log scaling or an outlier percentile from
all available data. The experiment layer selects an explicit contract and
records the resolved result.

## 2. Public abstraction

The reusable entry points are:

```python
from histvae import AxisPreprocessingSpec, HistogramPreprocessor
```

One axis specification defines:

```text
- scientific axis name
- coordinate transform: none or log1p
- lower bound: fixed or train-fitted quantile
- upper bound: fixed or train-fitted quantile
- quantile weighting: event or group_equal
```

The preprocessor defines the shared histogram contract:

```text
- bin count per axis
- count, density, or probability_mass representation
- tail policy: clip or error
```

Lower and upper bounds are independent. A fixed physical lower bound and a
train-fitted upper percentile are therefore represented directly rather than
through an ambiguous automatic range mode.

## 3. Quantile weighting

`event` gives every detected event equal weight and matches an ordinary pooled
quantile.

`group_equal` assigns every event in group `s` weight proportional to
`1 / n_s`. Every biological sample therefore contributes the same total weight
to the fitted empirical CDF, even when group event counts differ.

`group_equal` requires explicit training group identifiers. Validation and
holdout groups are never used to fit a bound.

## 4. Transform contract

`none` preserves raw differences and permits negative raw coordinates when the
fixed or fitted bounds contain them.

`log1p` allocates more bins to the low-valued region and requires non-negative
resolved bounds. It is an explicit axis choice, not a global mandatory default.
Different axes may use different transforms in multidimensional data.

Raw and transformed bin edges are both available. Scientific figures continue
to default to raw coordinates under R-260621-01.

This class currently applies only pointwise axis transforms. It does not compute
per-group medians, centering, ratios, or IQR scaling. The future strict
group-coordinate stage is upstream of this preprocessor and is defined in
R-260621-03; after group normalization, this preprocessor remains responsible
for train-fitted global histogram geometry.

## 5. Tail contract

`clip` maps values below or above the fitted range to the corresponding edge
bin. It preserves event count and total probability mass while recording the
underflow and overflow fractions.

`error` rejects application data outside the fitted range. A quantile-fitted
`error` contract is rejected during fit when it already excludes training
observations.

Drop-and-renormalize is intentionally not part of `HistogramPreprocessor`,
because it can silently remove a rare tail and change the conditional
probability distribution. The legacy `Histogram` path retains `drop` for
backward compatibility.

Explicit overflow bins, `asinh`, and unbalanced-mass modeling remain separate
future themes.

## 6. Persisted state

A fitted state records:

```text
- schema version
- axis specifications
- resolved raw lower and upper bounds
- bins and histogram mode
- tail policy
- training underflow/overflow diagnostics
- row and group counts
- canonical training-array hash
- canonical state hash
```

The state is safe-YAML and JSON compatible. `save()` and `load()` preserve the
same histogram geometry. A state-hash mismatch is a hard error.

## 7. HistVAE integration

The fitted object is passed explicitly at model construction so it becomes the
configuration source of truth before histogram/model contract validation:

```python
model = HistVAE(
    config=config,
    histogram_preprocessor=preprocessor,
)
model.prep_data(
    train_data=train_data,
    train_group=train_group,
    test_data=validation_data,
    test_group=validation_group,
)
```

The same object is used for train and validation datasets. Its resolved values
are serialized under `histogram_preprocessor_state`. Application diagnostics
are stored separately for train and test data. `prep_data()` also accepts the
object for an already compatible base config, but constructor injection is the
canonical path because it removes duplicated geometry settings.

Passing runtime histogram overrides together with a fitted preprocessor is an
error. Loading a saved experiment config restores the same fitted state without
refitting.

## 8. Current finalized model

The finalized FITC model remains frozen with its previously selected contract:

```text
log1p; lower 0; upper 100,000; clip; 64 bins
```

The new preprocessor does not retroactively reopen that holdout or alter the
selected checkpoints. It is the canonical implementation for future datasets
and new development cycles.

## 9. Verification

The focused tests cover:

```text
- pooled-event versus group-equal percentiles
- strict group requirement for group-equal fitting
- nonzero raw bounds and inverse log1p edges
- clip and error tail behavior
- safe-YAML save/load, required state hashes, and tamper detection
- axis-specific transforms in 2D
- direct PointHistDataset integration
- constructor-level source-of-truth injection
- HistVAE train/test reuse and serialized-state replay
- rejection of ambiguous runtime overrides
```

Verification on 2026-06-21:

```text
focused tests:                         10 passed
regular full suite:                    70 passed, 5 skipped
full suite with slow tests:            75 passed
attached-data rows:                    519,118
train / validation biological groups: 94 / 20
q0.999 group-equal upper bound:        97,626 FITC_Sum
train / validation fraction clipped:  0.0010648 / 0.0006605
histogram and reconstruction shape:    (batch, 1, 64)
```

The attached-data run is a compatibility smoke for the new preprocessing API,
not a reopening of the finalized 100,000-bound model-selection artifact.
