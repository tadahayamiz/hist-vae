# E-260619-00: Attached FITC density smoke validation

## 1. Purpose

Verify that the attached CSV can be represented as 1D density histograms grouped
by `sample_name` and passed through the existing VAE without non-finite values or
shape errors.

## 2. Conditions and method

- Input: `Final_Master_Combined_Data.csv` (external to the returned repo zip)
- Value: `FITC_Sum`
- Group: `sample_name`
- Rows: 519,118
- Groups: 134
- Observed `FITC_Sum` maximum: 334,262
- `in_dims=1`
- `bins=64`
- `max_vals=[350000.0]`
- `num_points=2048`
- `histogram_mode="density"`
- CPU forward pass only; no training

## 3. Result

- Histogram batch shape: `(8, 1, 64)`
- Reconstruction shape: `(8, 1, 64)`
- Latent mean shape: `(8, 128)`
- Histogram and reconstruction tensors were finite.

## 4. Interpretation

The updated repository is technically compatible with the requested grouped 1D
density representation.

## 5. Limitations

This is not a performance evaluation. The range and bin count were selected for
smoke validation and must be justified using the training split in the actual
analysis.

## 6. Next action

Define the analysis split and compare `density` with `count` under identical
training conditions.

## 7. Related reference

- R-260619-00

## 8. Related artifact

- External input CSV supplied with the task
- `tests/test_histogram_modes.py`
