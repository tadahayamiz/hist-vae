# E-260623-00: Joint-Sinkhorn weight ablation on fixed raw-median-ratio geometry

Status: complete
Date: 2026-06-23

## 1. Purpose

Determine whether a weak observation-space joint Sinkhorn auxiliary improves
geometric reconstruction without materially degrading forward-KL fidelity or
sample-level latent stability.

The coordinate, histogram geometry, architecture, latent regularization, split,
and seed matrix were fixed before this experiment. The legacy 20-group holdout
was not used.

## 2. Conditions and method

```text
data: Final_Master_Combined_Data.csv
coordinate: raw_median_ratio = x / median_group(x) - 1
histogram: 1D, 64-bin probability mass
range: fixed train-fitted geometry from E-260622-00
input / target: random 1,024-event view / full-group histogram
base reconstruction: forward KL
OT: debiased joint Sinkhorn divergence
OT p / blur / scaling: 1 / 0.05 / 0.8
backend: tensorized
latent / hidden: 4 / [8, 16]
beta: 1e-4 with 25-epoch linear warmup
optimizer: RAdam
maximum epochs / patience: 300 / 20
seeds: 17, 42, 73
train groups: 94
validation groups: 20
legacy holdout: not used
labels: report-only; not used for representation training or OT selection
```

KL-only references were retrained on the same branch snapshot as all OT runs.
This avoided using exact numerical replay of checkpoints produced under a
previous runtime as the comparison basis.

## 3. OT-weight calibration

For each KL-only seed:

```text
lambda_equal_seed
  = median_train_forward_KL / median_train_Sinkhorn
```

The values were:

| Seed | Median train KL | Median train Sinkhorn | lambda_equal_seed |
|---:|---:|---:|---:|
| 17 | 0.010517 | 0.0001588 | 66.221982 |
| 42 | 0.012964 | 0.0001756 | 73.841289 |
| 73 | 0.009553 | 0.0001414 | 67.578685 |

The calibration reference was the median across seeds:

```text
lambda_equal = 67.57868479947342
```

The tested relative factors and actual config weights were:

| OT factor | Actual `ot_weight` |
|---:|---:|
| 0.1 | 6.7578684799473425 |
| 0.3 | 20.273605439842026 |
| 1.0 | 67.57868479947342 |

`ot_factor` is experiment metadata. The repository config receives
`ot_weight`. Therefore `ot_weight=0.1` is not the selected setting.

## 4. Prespecified selection gate

A candidate was eligible only if:

```text
mean validation forward KL <= 110% of KL-only
all four latent dimensions active for every seed
mean random-view retrieval no worse than 5 percentage points vs KL-only
```

After passing the gate, preference was given to lower unweighted Sinkhorn and
exact 1D W1 while preserving between/within separation and input-W1/latent
rank correlation. Combined observation losses were not compared across
weights because changing lambda changes their numeric scale.

The regularized linear probe was report-only and excluded from the gate.

## 5. Results

Three-seed development means:

| Loss | OT factor | OT weight | Forward KL | Sinkhorn | Exact W1 | Between/within | Retrieval | W1-latent | Active | Gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| forward KL | 0.0 | 0.000000 | 0.013313 | 0.000358 | 0.004534 | 4.179 | 0.765625 | 0.520846 | 4/4 | pass |
| KL + Sinkhorn | **0.1** | **6.757868** | **0.013428** | **0.000302** | **0.004274** | **4.189** | **0.778125** | **0.521112** | **4/4** | **pass** |
| KL + Sinkhorn | 0.3 | 20.273605 | 0.015623 | 0.000312 | 0.004298 | 3.803 | 0.751042 | 0.560760 | 4/4 | fail |
| KL + Sinkhorn | 1.0 | 67.578685 | 0.017357 | 0.000289 | 0.004236 | 3.530 | 0.731250 | 0.648898 | 4/4 | fail |

Relative to KL-only, factor `0.1` produced:

```text
forward KL ratio:       1.008695  (+0.87%)
Sinkhorn ratio:         0.841847  (-15.8%)
exact W1 ratio:         0.942826  (-5.7%)
retrieval delta:        +0.012500
between/within delta:   +0.009741
W1-latent delta:        +0.000266
active dimensions:      4/4 for every seed
```

Factors `0.3` and `1.0` increased W1-latent correlation but degraded base
forward KL by `17.4%` and `30.4%`, respectively. They also reduced
between/within separation and retrieval, so they were rejected.

The report-only linear probe remained near chance across settings. It did not
support a disease-classification claim and was not used to select the model.

## 6. Decision

Select:

```text
coordinate: raw_median_ratio
base loss: forward KL
OT: joint Sinkhorn
OT factor: 0.1
actual ot_weight: 6.7578684799473425
latent_dim: 4
```

This is a weak auxiliary geometry term rather than an OT-dominated objective.
The selected setting improves geometric reconstruction with a small cost in
forward-KL fidelity and no detected loss of view stability or active latent
capacity.

The scientific result remains the three-seed aggregate. For a single
collaborator-facing deterministic export, select:

```text
raw_median_ratio_kl_sinkhorn_f0p10_seed73
```

because it had the lowest validation combined observation loss among the three
runs at the already selected factor `0.1`.

## 7. Limitations

- Only the existing 1D FITC assay and one development split were tested.
- The legacy holdout was not used, but it is also not a new independent cohort
  because it was previously used for the frozen absolute-coordinate benchmark.
- Synthetic multiplicative invariance does not prove that all real device
  variation is multiplicative.
- OT acts on target versus reconstruction. It does not directly constrain
  latent Euclidean distance to equal Wasserstein distance.
- The validation set contains only 20 groups, so seed-level variation remains
  important.
- The linear probe is too small and unstable for a diagnostic claim.

## 8. Next action

Create the deterministic collaborator artifact from the selected seed-73
checkpoint: all requested posterior means, posterior SDs, train-fit PCA/UMAP,
and input-versus-reconstruction figures. If the legacy holdout is included,
mark it as opened/descriptive and prohibit further selection from it.

Confirmatory evaluation requires a new independent cohort.

## 9. Related references

- R-260621-03
- R-260623-00

## 10. Related artifact

```text
fitc_raw_median_ratio_joint_sinkhorn_ablation_v4/
  experiment_contract.json
  fixed_inputs.npz
  group_coordinate_normalizer.yaml
  preprocessing.yaml
  ot_weight_calibration.csv
  ot_weight_calibration.json
  run_metrics.csv
  ot_comparison_table.csv
  selection_note.json
  raw_median_ratio_kl_only_seed*/
  raw_median_ratio_kl_sinkhorn_f0p10_seed*/
  raw_median_ratio_kl_sinkhorn_f0p30_seed*/
  raw_median_ratio_kl_sinkhorn_f1p00_seed*/
```
