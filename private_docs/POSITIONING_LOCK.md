# Positioning Lock

Updated: 2026-06-23

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
- The selected shape-oriented development coordinate is
  `raw_median_ratio = x / median_group(x) - 1`, conditional on the assay
  assumption that dominant technical variation is a positive multiplicative
  gain.
- `raw_median_ratio` retains relative width, skewness, multimodality, and
  relative tails. It removes group-level scale and must be accompanied by raw
  median and event count/rate metadata when those quantities matter.
- Distribution width is retained by default. IQR scaling remains sensitivity
  analysis unless technical controls show that width is predominantly
  instrumental.
- The measure-aware mainline uses simplex output, forward KL, and a weak joint
  Sinkhorn auxiliary loss. Legacy count/density behavior remains available.
- The selected OT setting is `ot_factor=0.1`, implemented as the calibrated
  config value `ot_weight=6.7578684799473425` for the current geometry and loss
  scales.
- Joint Sinkhorn is computed on the complete multidimensional histogram support,
  not as an average of axis-wise marginals. It is an observation-space
  geometry term, not WAE-style prior matching.
- Stronger OT factors `0.3` and `1.0` were rejected because they degraded base
  forward-KL fidelity beyond the predefined gate.
- Technical metadata is excluded by default. Decoder-only technical
  conditioning exists as an optional ablation but was not selected for the
  current dataset.
- Decoder conditioning is not equivalent to causal batch correction and cannot
  identify disease effects under label-batch confounding.
- Disease labels are not used in unsupervised pretraining or model selection.
- The formal downstream output is the deterministic full-group posterior mean
  `mu` using all four latent dimensions. Active-unit counting is a collapse
  diagnostic, not a default feature-selection rule.
- PCA and UMAP are descriptive transforms fitted on training latent values and
  applied out of sample. Healthy/Cancer labels may be used for color only.
- The current model is a weakly VAE-regularized denoising distributional
  autoencoder with a geometry-aware observation auxiliary. Representation
  quality is supported; prior-calibrated generation and posterior-uncertainty
  calibration are not established.
- The disease linear probe is exploratory. It does not establish clinical
  diagnosis, two-cluster separation, causality, or biological mechanism.
- The legacy absolute-coordinate holdout was not used to select the new
  coordinate or OT weight, but it was already used for the older frozen model
  and is not a new independent confirmatory cohort.
- If that legacy holdout is included in an all-sample collaborator export, it
  becomes descriptive/opened for the new model and cannot be used for any
  further selection.
- A new independent cohort is required for confirmatory evaluation of the
  selected shape-plus-OT mainline.
- Technical execution and reconstruction quality do not by themselves establish
  biological mechanism.
