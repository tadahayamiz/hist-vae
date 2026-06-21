# E-260621-02: Finalized holdout evaluation

## 1. Purpose

Perform the one-time final evaluation of the already selected unconditioned
`beta=1e-4` model without further architecture, hyperparameter, condition,
checkpoint, threshold, or seed selection.

## 2. Final contract

- Three fixed seeds: 17, 42, 73
- Fixed best checkpoints selected on validation reconstruction
- `condition_mode=none`
- Full-group deterministic evaluation with `z = mu`
- 16 fixed random 1,024-event views per holdout group
- Holdout composition: 14 C and 6 PC groups
- Linear probe trained on the fixed train plus validation latents
- Linear-probe hyperparameters were not tuned on holdout
- No best seed was selected

The train-mean holdout forward-KL baseline was `0.0510488`.

## 3. Per-seed results

| seed | holdout forward KL | random-view KL | improvement vs baseline | active dims | separation ratio | retrieval | W1-latent Spearman | probe AUC | balanced accuracy |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 17 | 0.004919 | 0.005676 | 90.36% | 4 | 5.710 | 0.859 | 0.928 | 0.738 | 0.774 |
| 42 | 0.008268 | 0.008886 | 83.80% | 4 | 4.981 | 0.819 | 0.895 | 0.786 | 0.774 |
| 73 | 0.005567 | 0.006219 | 89.09% | 4 | 4.748 | 0.809 | 0.824 | 0.679 | 0.607 |

Every holdout group had lower forward KL under the model than under the global
train-mean distribution.

## 4. Aggregate representation results

```text
holdout full forward KL:             0.006251 +/- 0.001777
holdout random-view forward KL:      0.006927 +/- 0.001718
reconstruction improvement:          0.877541 +/- 0.034804
fraction beating baseline:           1.000 for every seed
train active dimensions:             4 / 4 for every seed
holdout active dimensions:           4 / 4 for every seed
holdout latent KL:                   18.1923 +/- 1.4364
holdout posterior SD mean:           0.03382 +/- 0.00741
between/within separation ratio:     5.1462 +/- 0.5017
random-view retrieval accuracy:      0.8292 +/- 0.0266
chance retrieval accuracy:           0.05
input-W1 / latent-distance Spearman: 0.8820 +/- 0.0532
```

Repeated deterministic reconstruction produced a maximum difference of zero.

## 5. Disease-label probe

The fixed per-seed probe results were:

```text
ROC AUC:            0.7341 +/- 0.0537
balanced accuracy:  0.7183 +/- 0.0962
```

A secondary arithmetic mean of the three fixed-seed PC probabilities gave:

```text
ROC AUC:            0.7738
balanced accuracy:  0.6905
```

The ensemble is descriptive and does not replace the three co-primary seed
results.

The assay context is that C and PC distributions are visually similar overall
and some PC samples may contain a cancer-specific component absent from C.
Accordingly, a moderate ranking signal without a clear two-cluster geometry is
scientifically plausible. Because only six PC samples are in holdout, the probe
is exploratory and is not evidence of clinical diagnostic performance.

## 6. Interpretation

The final results support the primary representation claim:

```text
The selected model derives a low-dimensional sample representation that
reconstructs unseen empirical distributions, is robust to finite-event
subsampling, retains sample identity, and preserves much of the original 1D
Wasserstein geometry.
```

They do not establish unconditional generative validity. The small selected
beta, narrow posterior standard deviations, and non-negligible latent KL imply
that the model is best interpreted as a weakly VAE-regularized denoising
distributional autoencoder whose principal output is deterministic `mu`.

## 7. Finalization

Run directory name:

```text
fitc_final_holdout_v1
```

The directory contains `FINALIZED.txt`. The holdout must not be reused for
model selection. Future tail-sensitive, abundance-aware, supervised, OT, or
one-class extensions require a new development protocol and preferably an
independent cohort.

Key artifacts:

```text
evaluation_contract.json
final_holdout_summary.json
holdout_run_metrics.csv
holdout_aggregate_metrics.csv
holdout_probe_predictions.csv
latent_all_seed17.csv
latent_all_seed42.csv
latent_all_seed73.csv
holdout_diagnostics_seed17.npz
holdout_diagnostics_seed42.npz
holdout_diagnostics_seed73.npz
FINALIZED.txt
```

## 8. Related references

- R-260620-01
- R-260620-02
- R-260621-00
