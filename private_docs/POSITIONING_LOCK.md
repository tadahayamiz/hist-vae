# Positioning Lock

Updated: 2026-06-21

HistVAE is a grouped empirical-measure representation model for biological
samples observed as low-dimensional point sets. It is not specific to one
single-molecule enzyme assay, one column name, or one acquisition partitioning
scheme.

- The biological sample is the latent unit.
- Image tiles, fields, or slices are merged when they are acquisition
  partitions of one measured sample region.
- Fixed histogram coordinates retain absolute distribution location and width.
- Probability-mass normalization removes total event count, not point-value
  intensity or distribution shifts.
- Total event mass/rate is a distinct optional feature and must be modeled
  explicitly when scientifically meaningful.
- The measure-aware mainline uses simplex output and distribution
  reconstruction; legacy count/density behavior remains available.
- Technical metadata is excluded by default. Decoder-only technical
  conditioning exists as an optional ablation but was not selected for the
  current dataset.
- Decoder conditioning is not equivalent to causal batch correction and cannot
  identify disease effects under label-batch confounding.
- Disease labels are not used in unsupervised pretraining or model selection.
- The selected output for downstream use is deterministic full-group posterior
  mean `mu`, reported across three fixed seeds.
- The selected model is a weakly VAE-regularized denoising distributional
  autoencoder. Representation quality is supported; prior-calibrated generation
  and posterior-uncertainty calibration are not established.
- Final holdout reconstruction, random-view stability, and geometry preservation
  support a sample-representation claim.
- The disease linear probe is exploratory. It does not establish clinical
  diagnosis, two-cluster separation, or causality.
- The finalized holdout must not be reused for further model or threshold
  selection.
- Technical execution and reconstruction quality do not by themselves establish
  biological mechanism.
