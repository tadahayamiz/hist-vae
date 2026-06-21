# E-260621-00: Beta convergence and selection

## 1. Purpose

Select the latent-KL target after the initial 50-epoch screen showed that most
candidate runs were still improving near the epoch ceiling.

## 2. Data and fixed contract

- Input: `Final_Master_Combined_Data.csv`
- Coordinate: `FITC_Sum`
- Group: `sample_name`
- Fixed split: 94 train, 20 validation, 20 untouched holdout groups
- Holdout was not encoded or used for model selection
- 64 log1p-spaced bins, train-fitted upper bound 100,000, overflow clipped
- Random 1,024-point train input and deterministic full-group target
- Full-group validation with `z = mu`
- `latent_dim=4`, `hidden_dims=[8, 16]`, `dropout_conv=0`
- Simplex decoder, forward-KL reconstruction, no technical condition
- RAdam, learning rate 0.001, warmup 25 epochs, patience 20
- Seeds: 17, 42, 73
- Epoch ceiling: 300

The train-mean validation forward-KL baseline was `0.0605591`.

## 3. Candidate settings

```text
beta = 1e-4
beta = 3e-4
beta = 1e-3
```

All candidates used zero-to-target linear warmup and reconstruction-based
checkpoint selection.

## 4. Results

| beta | mean validation forward KL | SD | mean latent KL | mean between/within ratio | mean view retrieval | mean W1-latent Spearman |
|---:|---:|---:|---:|---:|---:|---:|
| 1e-4 | 0.006772 | 0.001440 | 18.2346 | 5.3363 | 0.8042 | 0.8795 |
| 3e-4 | 0.007752 | 0.002078 | 13.4330 | 5.2716 | 0.8188 | 0.9005 |
| 1e-3 | 0.009046 | 0.003058 | 10.3659 | 4.9172 | 0.7854 | 0.9081 |

All nine runs:

```text
- beat the train-mean distribution baseline
- retained 4 / 4 active train and validation dimensions
- stopped by patience before the 300-epoch ceiling
```

For `beta=1e-4`, the three best epochs were 135, 90, and 166. This confirms
that the preceding 120-epoch ceiling was insufficient for two of the three
seeds.

## 5. Selection rule and decision

The preregistered gate required:

```text
1. every seed beats the train-mean baseline
2. all four latent dimensions are active in train and validation
3. mean validation reconstruction is within 10% of the best candidate
4. every seed converges by patience-based early stopping
5. geometry and random-view metrics are used only among eligible settings
```

Only `beta=1e-4` passed the 10% reconstruction gate. It was selected as the
mainline target. `beta=3e-4` remains a regularization-sensitivity result but is
not a co-primary model.

## 6. Interpretation

The selected setting provides a reproducible non-collapsed representation and
strongly improves on the global train-mean distribution. Larger beta values
reduce latent KL as expected, but the reconstruction cost exceeds the fixed
eligibility margin.

No holdout information contributed to this decision.

## 7. Artifacts

Run directory name:

```text
fitc_grouped_measure_beta_convergence_v2
```

Key artifacts:

```text
pilot_manifest.json
selection_summary.json
run_metrics.csv
gate_summary.csv
beta_1em04_seed17/model_best.pt
beta_1em04_seed42/model_best.pt
beta_1em04_seed73/model_best.pt
```

## 8. Related references

- R-260620-01
- R-260620-02
- R-260621-00
