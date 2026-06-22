# E-260622-00: KL-only shape-coordinate ablation

Status: complete
Date: 2026-06-22

## 1. Purpose

Select the group-coordinate contract before introducing observation-space OT.
The experiment compares additive-shift and multiplicative-gain invariance while
holding the histogram, architecture, beta, data split, and seed matrix fixed.

## 2. Conditions and method

```text
data: Final_Master_Combined_Data.csv
coordinate candidates:
  raw_median_center
  raw_median_ratio
  log_median_center
histogram: 1D, 64-bin probability mass
range: train-fitted group-equal q0.001 to q0.999, overflow clip
input / target: random 1,024-event view / full-group histogram
observation loss: forward KL only
latent / hidden: 4 / [8, 16]
beta: 1e-4 with 25-epoch linear warmup
optimizer: RAdam
maximum epochs / patience: 300 / 20
seeds: 17, 42, 73
train groups: 94 (68 C, 26 PC)
validation groups: 20 (14 C, 6 PC)
finalized holdout groups: 20, untouched
labels: report-only; not used for representation training or selection
```

Synthetic shifts were applied to validation events and each shifted group was
normalized again before histogram construction:

```text
additive: +/-5% and +/-10% of the median training-group raw median
multiplicative: 0.5x, 0.75x, 1.5x, 2.0x
```

All distance-based latent diagnostics used all four posterior-mean dimensions
standardized by training-set mean and standard deviation. Active dimensions
were retained only as a collapse diagnostic.

## 3. Results

Three-seed means:

| Coordinate | Val forward KL | Recon W1 | Between/within | Retrieval | Input-W1 / latent Spearman | Additive latent/between | Multiplicative latent/between | Multiplicative retrieval | Active |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `log_median_center` | 0.012763 | 0.004571 | 3.970 | 0.745 | 0.657 | 0.462 | 0.000 | 1.000 | 4/4 |
| `raw_median_center` | 0.014440 | 0.004684 | 5.506 | 0.781 | 0.723 | 0.000 | 1.604 | 0.054 | 4/4 |
| `raw_median_ratio` | 0.013312 | 0.004534 | 4.178 | 0.765 | 0.521 | 0.489 | 0.000 | 1.000 | 4/4 |

`raw_median_center` was exactly invariant to additive shifts but strongly
changed under multiplicative gains. `raw_median_ratio` was exactly invariant to
all prespecified positive multiplicative gains: shifted input W1 and latent
shift were zero and self-retrieval was one for every tested factor and seed.
`log_median_center` was also approximately multiplicative-invariant for these
large positive intensities, but it retains log-domain rather than linear-ratio
bin geometry.

## 4. Interpretation and decision

Under the stated assay assumption that device variation is primarily a
positive multiplicative intensity gain, select:

```text
main development coordinate: raw_median_ratio
sensitivity reference:       log_median_center
additive-offset comparator:  raw_median_center
```

This decision is based on the scientific invariance contract, not on native
forward KL alone. Native KL values are not a complete cross-coordinate ranking
because each coordinate allocates finite bins differently.

`raw_median_ratio` remains a shape-oriented probability-mass representation. It
removes the group median scale and total event abundance. Raw median and event
count/exposure-normalized rate must therefore remain separate sample metadata
or downstream features.

## 5. Limitations

- Synthetic shifts validate mathematical behavior, not the actual physical
  decomposition of instrument variation into gain and offset.
- `raw_median_ratio` showed greater seed variability in validation KL than the
  log reference.
- The finalized holdout was not used and does not evaluate this new coordinate.
- No OT term was used in this experiment.

## 6. Next action

Fix `raw_median_ratio`, architecture, beta, split, and seeds. Compare forward KL
against forward KL plus the newly implemented joint Sinkhorn divergence using
development data only. A new independent holdout is evaluated once only after
all settings are frozen.

## 7. Related reference

- R-260621-03

## 8. Related artifact

```text
fitc_group_coordinate_ablation_kl_only_v1/
  coordinate_ablation_manifest.json
  run_metrics.csv
  aggregate_metrics.csv
  coordinate_comparison_table.csv
  synthetic_shift_metrics_all_runs.csv
```
