# Claim to Evidence Map

Updated: 2026-06-20

| Claim | Evidence | Status | Notes |
|---|---|---|---|
| The repo can construct and pass 1D density histograms for the attached `FITC_Sum` data through the VAE. | E-260619-00 | verified | Technical smoke validation only. |
| Replicating identical observations changes count histograms but not density or probability-mass histograms. | `tests/test_histogram_modes.py` | verified | Unit tests of group-size invariance. |
| Probability-mass inputs are non-negative and sum to one. | `tests/test_histogram_modes.py`, E-260620-00 | verified | Sampled and full-group views. |
| Legacy sigmoid/MSE is unsuitable as the main probability-mass output contract for the attached pilot. | E-260620-01 | verified | Model reconstruction was far worse than the train-mean MSE baseline and latent activity collapsed. |
| The simplex decoder returns a valid probability mass in 1D, 2D, and 3D. | `tests/test_grouped_measure_contract.py`, E-260620-02 | verified | Non-negative output with total mass one. |
| Random finite inputs can reconstruct a deterministic full-sample target. | `tests/test_grouped_measure_contract.py`, E-260620-02 | verified | New denoising contract. |
| Decoder-only technical conditioning leaves the encoder API and exported `mu` condition-blind. | `tests/test_grouped_measure_contract.py` | verified | This does not prove empirical deconfounding. |
| The small unconditioned grouped-measure model can outperform a train-mean distribution baseline on the attached validation split. | E-260620-03 | verified | Single split/seed reconstruction gate; not yet evidence of biological utility. |
| Linear latent-KL warmup is applied, logged, and checkpointed with strict reconstruction-based model selection. | `tests/test_latent_kl_schedule.py`, E-260620-03 | verified | Free bits and alternative schedules remain deferred. |
| Repeated validation and latent extraction are deterministic under the default evaluation path. | `tests/test_pretraining_reliability.py`, E-260620-00 | verified | Full-group histograms and `z = mu`. |
| `model_best.pt` corresponds to the configured monitored best epoch. | `tests/test_pretraining_reliability.py` | verified | `model_last.pt` separately preserves the final epoch. |
| The current model yields a biologically useful disease-related latent representation. | None | open | Requires reconstruction, stability, batch, and downstream evaluation. |
