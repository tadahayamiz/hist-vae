# Positioning Lock

Updated: 2026-06-21

HistVAE is a grouped empirical-measure representation model for biological
samples observed as low-dimensional point sets. It is not specific to one
single-molecule enzyme assay, one column name, or one acquisition partitioning
scheme.

- The biological sample is the latent unit.
- Image tiles, fields, or slices are merged when they are acquisition
  partitions of one measured sample region.
- Probability-mass normalization removes total event count, not point-value
  intensity or distribution shifts. Probability mass alone is therefore not a
  shape-only transformation.
- The frozen `log1p` benchmark retains absolute distribution location and width.
  Its finalized evidence does not establish invariance to device intensity
  shifts.
- The future shape mainline uses a strict per-sample robust coordinate
  normalization before histogram construction; plain raw absolute intensity is
  not the main candidate.
- Distribution width is retained as biological shape by default. IQR scaling is
  a sensitivity analysis unless technical controls show that width is
  predominantly instrumental.
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
- For the frozen benchmark, the downstream output is deterministic full-group
  posterior mean `mu`, reported across three fixed seeds.
- The frozen benchmark is a weakly VAE-regularized denoising distributional
  autoencoder. Representation quality is supported; prior-calibrated generation
  and posterior-uncertainty calibration are not established.
- Its final holdout reconstruction, random-view stability, and geometry
  preservation support an absolute-coordinate sample-representation claim.
- The disease linear probe is exploratory. It does not establish clinical
  diagnosis, two-cluster separation, or causality.
- The finalized absolute-coordinate holdout must not be reused for raw-shape,
  OT, or any other model or threshold selection.
- Technical execution and reconstruction quality do not by themselves establish
  biological mechanism.
