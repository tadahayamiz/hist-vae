# Claim to Evidence Map

Updated: 2026-06-20

| Claim | Evidence | Status | Notes |
|---|---|---|---|
| The repo can construct and pass 1D density histograms for the attached `FITC_Sum` data through the VAE. | E-260619-00 | verified | Technical smoke validation only. |
| Replicating identical observations changes count histograms but not density or probability-mass histograms. | `tests/test_histogram_modes.py` | verified | Unit tests of group-size invariance. |
| Probability-mass model inputs remain in `[0, 1]` and sum to one. | `tests/test_histogram_modes.py`, E-260620-00 | verified | Holds for sampled and full-group attached-data views. |
| Repeated validation and latent extraction are deterministic under the default evaluation path. | `tests/test_pretraining_reliability.py`, E-260620-00 | verified | Full-group histograms and `z = mu`. |
| `model_best.pt` corresponds to the configured monitored best epoch. | `tests/test_pretraining_reliability.py` | verified | `model_last.pt` separately preserves the final epoch. |
| Probability mass is better for the downstream research objective. | None | open | Requires controlled multi-seed experiments and downstream evaluation. |
