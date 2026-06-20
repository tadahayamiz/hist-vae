# Claim to Evidence Map

Updated: 2026-06-19

| Claim | Evidence | Status | Notes |
|---|---|---|---|
| The repo can construct and pass 1D density histograms for the attached `FITC_Sum` data through the VAE. | E-260619-00 | verified | Technical smoke validation only; no training-quality claim. |
| Replicating identical observations changes count histograms but not density histograms. | `tests/test_histogram_modes.py` | verified | Unit test of the intended intensity-removal behavior. |
| Density is better for the downstream research objective. | None | open | Requires controlled experiments. |
