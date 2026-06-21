# E-260621-01: Decoder-conditioning ablation

## 1. Purpose

Determine whether optional decoder-only technical conditioning should be part
of the selected shape-only mainline after fixing `beta=1e-4`.

## 2. Condition definition

A candidate technical batch was derived from the first two underscore-delimited
parts of `data_24`, for example:

```text
260216_52_0000 -> 260216_52
```

Eight train-observed categories were encoded as train-fitted one-hot vectors.
The condition was supplied only to the decoder. The encoder and exported `mu`
received no condition vector.

The development contingency table showed partial label-batch overlap, but some
validation batches contained only C samples. This limits interpretation of
batch classification metrics and rules out a causal deconfounding claim.

## 3. Compared settings

```text
condition_mode: none
condition_mode: decoder
```

The selected beta, architecture, split, three seeds, and all other training
settings were fixed. Holdout was untouched.

## 4. Aggregate results

| metric | none mean +/- SD | decoder mean +/- SD |
|---|---:|---:|
| validation forward KL | 0.006772 +/- 0.001440 | 0.007161 +/- 0.002221 |
| random-view retrieval | 0.8042 +/- 0.0422 | 0.7865 +/- 0.0443 |
| W1-latent Spearman | 0.8795 +/- 0.0636 | 0.8760 +/- 0.0867 |
| batch explained variance | 0.0884 +/- 0.0070 | 0.1010 +/- 0.0030 |
| label ROC AUC, report only | 0.6984 +/- 0.0069 | 0.6270 +/- 0.0698 |

Decoder conditioning did use the condition: permuting condition labels
increased validation forward KL in all three seeds. However, the effect did not
translate into a consistent representation benefit.

Seed-paired reconstruction ratios (`decoder / none`) were approximately:

```text
seed 17: 1.149
seed 42: 0.619
seed 73: 1.578
```

The decoder setting therefore improved one seed but substantially worsened two.
It also failed the W1-geometry gate and did not reduce batch explained
variance.

## 5. Decision

```text
recommended condition_mode: none
selection ready: true
```

The decoder path remains implemented as an optional ablation interface, but it
is not part of the selected current-data mainline.

The scikit-learn warning that predictions contained classes absent from the
validation truth reflects incomplete validation-batch coverage, not a training
failure. It further weakens balanced-accuracy as a deconfounding measure in
this split; the selection decision does not depend on that metric alone.

## 6. Artifacts

Run directory name:

```text
fitc_decoder_condition_ablation_v1
```

Key artifacts:

```text
pilot_manifest.json
condition_mapping.json
run_metrics.csv
paired_seed_deltas.csv
aggregate_metrics.csv
condition_selection_summary.json
```

## 7. Related references

- R-260620-01
- R-260621-00
