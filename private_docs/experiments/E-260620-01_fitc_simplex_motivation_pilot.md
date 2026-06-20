# E-260620-01: FITC probability-mass pilot motivating a simplex decoder

## 1. Purpose

Evaluate the corrected probability-mass input and deterministic validation path
on the attached `FITC_Sum` data before changing the decoder contract.

## 2. Conditions and method

- 519,118 point rows and 134 `sample_name` groups
- 94 train, 20 validation, and 20 untouched holdout groups
- Label-stratified group split: train 68 C / 26 PC; validation 14 C / 6 PC;
  holdout 14 C / 6 PC
- `probability_mass`, `log1p` coordinates, 64 bins
- Training-only 99.9th percentile range rounded to 100,000
- Overflow policy: `clip`
- Random 1,024-point training views; full-group validation
- Legacy sigmoid decoder, MSE reconstruction, latent dimension 128, hidden
  dimensions `[32, 64, 128, 256]`, beta 1, three epochs

## 3. Results

- Training fraction clipped: 0.0007329
- Validation fraction clipped: 0.0004112
- Holdout fraction clipped: 0.0005433
- Inputs were finite, bounded in `[0, 1]`, and summed to one.
- Repeated validation was exactly deterministic.
- Train-mean validation MSE baseline: 0.0151811
- Best monitored epoch: 2
- Best validation reconstruction: 18.77995
- Best validation latent standard-deviation mean: 0.0015937
- Active latent dimensions at the best epoch: 0 / 128

## 4. Interpretation

The input and evaluation contracts worked, but the legacy sigmoid decoder did
not produce a probability mass and was drastically worse than the train-mean
distribution baseline. The best checkpoint also showed posterior collapse at
the configured activity threshold.

This result motivates a simplex-constrained decoder and distribution-aware
reconstruction loss. It does not establish that probability mass is inferior,
nor does it support biological conclusions.

## 5. Limitations

The model was intentionally overlarge for 94 training groups, beta was fixed at
1, and only three epochs were run. These limitations are not sufficient to
explain away the invalid decoder-output contract, but they remain relevant for
the next pilot.

## 6. Next action

Run a small-model pilot using `simplex_softmax`, `forward_kl`, random pooled
input, full pooled target, and beta 0 as the first reconstruction sanity check.
Compare `condition_mode: none` and `decoder` only after the unconditioned
shape path reconstructs better than its baseline.

## 7. Related reference

- R-260620-01

## 8. Related artifact

The run artifacts remain in the user-managed Google Drive experiment directory.
