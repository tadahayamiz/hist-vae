# E-260620-03: Reconstruction gate and latent-KL warmup smoke

## 1. Purpose

Test whether the grouped-measure path can beat a deterministic train-mean
probability-distribution baseline before latent regularization, then verify the
new linear latent-KL schedule on the attached FITC data.

## 2. Data and split

- Input: `Final_Master_Combined_Data.csv`
- Coordinate: `FITC_Sum`
- Group: `sample_name`
- Label-stratified fixed split with seed 42
- 94 train, 20 validation, and 20 untouched holdout groups
- Holdout was not used for fitting or model selection
- 64 log1p-spaced bins
- Train-fitted 99.9th-percentile range rounded to 100,000
- Values above the range were clipped to the edge bin

## 3. Shared model and training conditions

```text
histogram_mode: probability_mass
decoder_output_mode: simplex_softmax
reconstruction_loss: forward_kl
train input: random 1,024-point subset
train target: deterministic full-group distribution
validation: full-group distribution with z = mu
condition_mode: none
latent_dim: 4
hidden_dims: [8, 16]
dropout_conv: 0.0
optimizer: RAdam
learning rate: 0.001
epochs: 50
seed: 42
```

Python 3.13.5 and PyTorch 2.10.0 CPU were used for this local verification.

## 4. Train-mean baseline

The average train distribution was used as the same prediction for every
validation group.

```text
validation forward KL: 0.0605592
```

## 5. Reconstruction-first result

With constant `beta=0`:

```text
best epoch: 48
best validation forward KL: 0.0228470
improvement versus train mean: 62.27%
validation latent std mean: 0.4014
active latent dimensions: 4 / 4
```

The model therefore cleared the reconstruction gate and did not merely emit the
global train mean.

## 6. Linear-warmup execution smoke

With:

```yaml
beta: 0.0001
latent_kl_schedule: linear_warmup
latent_kl_warmup_epochs: 25
```

results were:

```text
best epoch: 50
best epoch beta: 0.0001
best validation forward KL: 0.0212813
validation latent KL: 20.8760
validation latent std mean: 0.3850
active latent dimensions: 4 / 4
improvement versus train mean: 64.86%
```

The checkpoint and history beta values matched the configured schedule. This is
a single-seed implementation smoke, not a selected final beta.

## 7. Interpretation

The canonical grouped-measure architecture now passes the reconstruction-first
gate. A small warmup target retained all four active latent coordinates while
preserving reconstruction quality in this smoke run.

The next experiment must compare several beta targets and seeds, then evaluate
same-sample random-view stability and between-sample separation. Technical
conditioning remains off until the unconditioned latent is characterized.

## 8. Related references

- R-260620-01
- R-260620-02
