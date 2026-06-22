# E-260621-03: Log-domain shape-normalization pilot

## 1. Purpose

Compare two per-sample log-domain normalizations while keeping the previously
selected architecture, beta, condition mode, split, and three seeds fixed. The
experiment asks whether distribution width should remain part of the sample
shape representation.

The source artifact is the external notebook `260620-0v7.ipynb`. The existing
holdout rows were removed before preprocessing and were not evaluated.

## 2. Compared coordinates

```text
log_median_center:
  log1p(x_si) - median_i(log1p(x_si))

log_median_iqr:
  [log1p(x_si) - median_i(log1p(x_si))] / IQR_i(log1p(x_si))
```

Both use probability-mass histograms, a simplex decoder, forward-KL
reconstruction, random 1,024-event inputs, full-group targets, latent dimension
4, hidden dimensions `[8, 16]`, `beta=1e-4`, 25-epoch warmup, and seeds
17/42/73.

Artifact provenance:

```text
source notebook: 260620-0v7.ipynb
repo ref:        dev-2026
repo commit:     a3f348ed2b9876f15c309629d37bd24aca8acd39
input file:      Final_Master_Combined_Data.csv
split manifest:  split_manifest.csv
train groups:    94
validation:      20
holdout:         20, removed before preprocessing and untouched
```

## 3. Aggregate validation results

| Metric, mean across three seeds | Median center | Median + IQR |
|---|---:|---:|
| Validation forward KL | 0.01276 | 0.01232 |
| Improvement over own train-mean baseline | 61.7% | 53.2% |
| Between-sample / within-view distance ratio | 4.02 | 3.75 |
| Random-view retrieval | 73.5% | 65.1% |
| Shape-W1 / latent-distance Spearman | 0.657 | 0.779 |
| Log-IQR-distance / latent-distance Spearman | 0.339 | 0.083 |
| Active latent dimensions | 4/4 | 4/4 |

## 4. Interpretation

`log_median_iqr` had slightly lower native validation forward KL and a stronger
shape-W1/latent rank correlation. However, it removed distribution width and
had weaker improvement over its own mean-distribution baseline, lower
sample/view separation, and lower retrieval.

The selected reference is therefore:

```text
log_median_center
```

This removes sample location while retaining width, skewness, multimodality,
and relative tails. `log_median_iqr` remains a sensitivity analysis for a
future setting in which technical controls demonstrate that width is primarily
instrumental.

The correlation between latent distance and the original sample median is not,
by itself, evidence of intensity leakage. Shape features can be statistically
associated with the original median even after the coordinate is exactly
centered. Technical robustness must instead be tested by controlled coordinate
shifts and repeat measurements.

## 5. Limitations

```text
- Both candidates are log-domain; no raw-domain coordinate was tested.
- Synthetic additive and multiplicative shift invariance was not yet measured.
- The random-view metric uses coordinates normalized with full-group statistics.
- Native forward KL values are tied to each candidate's bin geometry.
- This experiment does not evaluate an OT reconstruction term.
- The existing holdout remains unavailable for subsequent selection.
```

## 6. Decision and next step

Use `log_median_center` only as the completed shape-normalization reference.
The next development experiment compares raw median centering and raw median
ratio under the strict contract in R-260621-03. OT is evaluated only after one
coordinate mode is fixed.

## 7. Related artifacts and references

- External notebook: `260620-0v7.ipynb`
- R-260621-03: shape-oriented raw-coordinate and OT development contract
