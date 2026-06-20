# E-260620-00: Attached FITC probability-mass reliability smoke

## 1. Purpose

Verify the revised bounded representation, deterministic evaluation, training,
and artifact chain on the attached `FITC_Sum` data.

## 2. Conditions and method

- Input: 519,118 rows grouped into 134 `sample_name` values
- Split seed: 42, stratified by `label`
- Groups: 94 train, 20 validation, 20 held out
- Representation: `probability_mass`
- Value transform: `log1p`
- Range: training 99.9th percentile rounded to 100,000
- Out-of-range policy: `clip`
- Bins: 64
- Points per sampled training view: 1,024
- Smoke model: latent 8, hidden `[8, 16]`, one epoch, CPU RAdam

## 3. Results

- Training fraction above 100,000 before clipping: approximately 0.000733
- Validation fraction above 100,000 before clipping: approximately 0.000411
- Sampled train and full validation inputs were finite, in `[0, 1]`, and summed
  to one per group.
- Repeated validation metrics were exactly equal before training.
- One training epoch completed with finite metrics.
- `config.yaml` was reloadable with `yaml.safe_load`.
- `model_best.pt` and `model_last.pt` were both written.

## 4. Interpretation

The revised path is technically suitable for controlled pilot experiments. The
smoke model had zero active latent coordinates at the 0.01 threshold before
training, so this record does not support a representation-quality claim.

## 5. Limitations

One epoch and one small model are insufficient for hyperparameter or biological
conclusions. The robust range and log1p bins are candidate settings only.

## 6. Next action

Run a multi-seed pilot over capacity, beta, and bin/range choices.

## 7. Related reference

- R-260620-00

## 8. Related artifact

Local smoke output was used for verification only and is not included in the
repository archive.
