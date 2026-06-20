# Positioning Lock

Updated: 2026-06-20

HistVAE is a grouped empirical-measure representation model for biological
samples observed as low-dimensional point sets. It is not specific to one
single-molecule enzyme assay, one column name, or one acquisition partitioning
scheme.

- The biological sample is the latent unit.
- Image tiles, fields, or slices are merged when they are only acquisition
  partitions of that sample.
- Fixed histogram coordinates retain absolute distribution location and width.
- Probability-mass normalization removes total event count, not point-value
  intensity or distribution shifts.
- Total event mass/rate is a distinct optional feature and must be modeled
  explicitly when scientifically meaningful.
- The measure-aware mainline uses simplex output and distribution
  reconstruction; legacy count/density behavior remains available.
- Technical metadata is excluded by default and may be supplied as a numeric
  decoder-only condition. Batch identity is one possible condition, not a
  model-specific field.
- Decoder conditioning is not equivalent to causal batch correction and cannot
  identify disease effects under perfect label-batch confounding.
- Disease labels are not used in unsupervised pretraining unless a later,
  explicitly supervised extension is evaluated.
- Technical execution and reconstruction quality do not by themselves establish
  biological validity.
