# Claim to Evidence Map

Updated: 2026-06-22

| Claim | Evidence | Status | Notes |
|---|---|---|---|
| The repo can construct and pass 1D density histograms for the attached `FITC_Sum` data through the VAE. | E-260619-00 | verified | Technical smoke validation only. |
| Replicating identical observations changes count histograms but not density or probability-mass histograms. | `tests/test_histogram_modes.py` | verified | Unit tests of group-size invariance. |
| Probability-mass inputs are non-negative and sum to one. | `tests/test_histogram_modes.py`, E-260620-00 | verified | Sampled and full-group views. |
| Legacy sigmoid/MSE is unsuitable as the main probability-mass output contract for the attached pilot. | E-260620-01 | verified | Reconstruction was far worse than the train-mean baseline and latent activity collapsed. |
| The simplex decoder returns a valid probability mass in 1D, 2D, and 3D. | `tests/test_grouped_measure_contract.py`, E-260620-02 | verified | Non-negative output with total mass one. |
| Random finite inputs can reconstruct a deterministic full-sample target. | `tests/test_grouped_measure_contract.py`, E-260620-02 | verified | Selected denoising contract. |
| Decoder-only technical conditioning leaves the encoder API and exported `mu` condition-blind. | `tests/test_grouped_measure_contract.py` | verified | This does not imply empirical deconfounding. |
| Linear latent-KL warmup is applied, logged, and checkpointed with strict reconstruction-based model selection. | `tests/test_latent_kl_schedule.py`, E-260620-03 | verified | Effective beta is recorded in history and checkpoints. |
| Repeated validation and latent extraction are deterministic under the default evaluation path. | `tests/test_pretraining_reliability.py`, E-260620-00, E-260621-02 | verified | Full-group histograms and `z = mu`; final repeated reconstruction difference was zero. |
| `model_best.pt` corresponds to the configured monitored best epoch. | `tests/test_pretraining_reliability.py` | verified | `model_last.pt` separately preserves the final epoch. |
| `beta=1e-4` is the selected latent-KL target for the frozen absolute-coordinate benchmark under the predefined multi-seed validation gate. | E-260621-00 | verified | Three seeds, all converged; only candidate within the 10% reconstruction gate. |
| Decoder-only technical conditioning should not be used in the frozen absolute-coordinate benchmark. | E-260621-01 | verified | It used the condition but did not consistently improve reconstruction or geometry and did not reduce batch signal. |
| The frozen absolute-coordinate benchmark generalizes beyond the train-mean distribution to its finalized holdout. | E-260621-02 | verified | Mean holdout forward KL `0.006251` versus baseline `0.051049`; 20/20 groups beat baseline for every seed. |
| The frozen absolute-coordinate latent is stable to finite-event subsampling. | E-260621-00, E-260621-02 | verified | Holdout retrieval `0.829 +/- 0.027` versus chance `0.05`; between/within ratio `5.146 +/- 0.502`. |
| The frozen absolute-coordinate latent preserves much of its original one-dimensional log1p distribution geometry. | E-260621-00, E-260621-02 | verified | Holdout W1-latent Spearman `0.882 +/- 0.053`; this is not evidence of device-shift invariance. |
| The frozen absolute-coordinate benchmark avoids posterior collapse for the current data. | E-260621-00, E-260621-02 | verified | All four train and holdout dimensions active for every seed. |
| The frozen absolute-coordinate latent contains exploratory disease-related information. | E-260621-02 | tentative | Fixed holdout probe AUC `0.734 +/- 0.054`; probability ensemble AUC `0.774`; only six PC holdout samples. |
| The model provides a clinically validated cancer classifier. | None | rejected | The probe is exploratory, the sample is small, and no diagnostic validation was performed. |
| The frozen benchmark posterior is calibrated and supports realistic generation from `N(0, I)`. | None | open | Small beta, narrow posterior SD, and substantial latent KL; generation was not evaluated. |
| Decoder conditioning causally removes technical batch effects. | E-260621-01 | rejected | Conditioning was not selected and label-batch overlap is incomplete. |
| Probability-mass normalization alone produces a device-invariant shape representation. | R-260621-03 | rejected | It removes event abundance but retains coordinate location and width. |
| Within the tested log-domain shape family, median centering is the preferred reference over median-plus-IQR scaling. | E-260621-03 | verified | Median centering retained width and had stronger baseline improvement, view separation, and retrieval; holdout was untouched. |
| Under the prespecified multiplicative-gain invariance contract, `raw_median_ratio` is the selected development coordinate. | E-260622-00 | verified | Exact invariance to 0.5x, 0.75x, 1.5x, and 2.0x synthetic gains; holdout untouched. This does not prove all real device variation is multiplicative. |
| The repo exposes a strict, differentiable joint Sinkhorn auxiliary-loss API for 1D, 2D, and 3D probability histograms. | `tests/test_optimal_transport.py`, R-260621-03 | verified | Complete joint support, strict config, autograd, artifact logging, and backend dependency checks. |
| Joint Sinkhorn divergence improves reconstruction or latent geometry beyond forward KL on this assay. | None | open | The API is implemented; the fixed-coordinate development ablation has not yet been run. |
