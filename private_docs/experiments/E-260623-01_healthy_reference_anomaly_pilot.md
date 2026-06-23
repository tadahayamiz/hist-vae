# E-260623-01: Healthy-reference anomaly pilot on the frozen seed-73 model

Status: complete, exploratory/descriptive

## 1. Purpose

Test whether a minority of samples are repeatedly far from a train-Healthy
shape and latent reference under fixed 1,024-event resampling, without
retraining or selecting the HistVAE model.

## 2. Conditions

```text
checkpoint: raw_median_ratio_kl_sinkhorn_f0p10_seed73
normal reference: train Healthy
reference k: 5
input events per repeat: 1,024
repeats: 30
shape score: exact 1D W1 kNN
latent score: Euclidean kNN in train-standardized 4D mu
reconstruction score: W1 from full target to sampled-input reconstruction
holdout: opened/descriptive
```

Robust novelty required full-group shape percentile >= 0.95, full-group latent
percentile >= 0.90, and sampled shape/latent percentiles >= 0.90 in at least
80% of repeats.

## 3. Results

All exported samples:

```text
Healthy robust shape novelty: 4 / 96 = 4.2%
PDAC robust shape novelty:    5 / 38 = 13.2%
Healthy stable reconstruction misfit: 1.0%
PDAC stable reconstruction misfit:    7.9%
```

Validation plus opened holdout:

```text
Healthy robust shape novelty: 2 / 28 = 7.1%
PDAC robust shape novelty:    2 / 12 = 16.7%
Healthy stable reconstruction misfit: 3.6%
PDAC stable reconstruction misfit:   16.7%
```

Coherent shape-novelty candidates without stable high reconstruction misfit:

```text
A106 PDAC train
A006 Healthy validation
A119 Healthy train
A020 Healthy train
A088 PDAC train
```

Shape novelty plus model misfit:

```text
A061 PDAC train
A083 Healthy opened holdout
A108 PDAC validation
A093 PDAC opened holdout
```

No sample met the fixed rule for reconstruction misfit without corresponding
strong shape and latent novelty.

## 4. Interpretation

The pilot supports a minority-anomaly hypothesis rather than global
Healthy-versus-PDAC separation. PDAC had a higher descriptive anomaly fraction,
but the sample counts are small and the opened holdout is not confirmatory.

The strongest coherent PDAC candidates were train samples. The out-of-sample
PDAC candidates also had high reconstruction misfit, so technical effects and
finite-event noise remain plausible. Healthy anomalies show that the normal
shape distribution may be multimodal or affected by batch/sample quality.

## 5. Limitations

```text
VAE trained on all diagnoses
reference and calibration not cross-fitted
1D exact W1 used for the pilot shape score
reconstruction target remained the finite full-group empirical histogram
opened holdout descriptive only
single checkpoint for displayed artifact
```

## 6. Next

Keep the model frozen and test the dimension-general contract in
R-260623-01:

```text
aggregate repeated posterior means per sample
measure latent sampling instability
score aggregated latent against train Healthy
score mean sampled histogram by joint nD Sinkhorn
compare candidate overlap, stability, and geometry
```

Do not add view-consistency or metric-learning losses unless this minimal
comparison demonstrates a clear need.

## 7. Related reference

- R-260623-01

## 8. Related artifact

External collaborator export:

```text
all_samples_latent_pca_umap_reconstruction_seed73_v1_with_followup_and_anomaly_pilot.zip
```

The artifact is not embedded in this source-repository snapshot.
