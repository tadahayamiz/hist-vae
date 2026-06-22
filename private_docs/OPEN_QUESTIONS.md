# Open Questions

Updated: 2026-06-21

## Resolved for the frozen absolute-coordinate benchmark

- Small architecture: `latent_dim=4`, `hidden_dims=[8, 16]`.
- Latent regularization: `beta=1e-4`, 25-epoch linear warmup.
- Training ceiling and stopping: 300 epochs with patience 20.
- Technical conditioning: `condition_mode=none` for the finalized benchmark.
- Primary exported artifact: deterministic full-group posterior mean `mu`.
- Finalized holdout: unavailable for additional selection.
- Probability mass removes event abundance but does not remove coordinate
  location; the finalized model is not device-shift invariant.
- Within the completed log-domain pilot, median centering is the reference and
  median-plus-IQR is sensitivity analysis only.

## Active coordinate questions

- Are device differences primarily additive offsets, multiplicative gains, or a
  mixture? This determines whether `raw_median_center` or
  `raw_median_ratio` is the stronger invariance contract.
- Do technical replicates and synthetic shifts confirm invariance without
  removing biologically meaningful width, skewness, multimodality, or tails?
- Should the full-group median remain the canonical statistic at deployment, or
  must view-local median uncertainty also be modeled for low-event samples?
- What common reporting metrics best compare coordinate systems whose native
  bin geometries and forward-KL scales differ?

## Active OT questions

- What entropic regularization `epsilon` and auxiliary weight `lambda_ot` retain
  forward-KL fidelity while improving geometric reconstruction?
- Which sparse, separable, or tensorized backend is needed to compute the same
  joint Sinkhorn-divergence contract efficiently in 2D and 3D?
- Does observation-space joint Sinkhorn improve reconstruction, view stability,
  or latent geometry after the coordinate mode is fixed?

## Other open research questions

- Should total event abundance be represented as total mass,
  exposure-normalized rate, or both when the exposure denominator is reliable?
- Do prespecified high-intensity tail mass or one-class novelty measures capture
  a hypothesized disease-specific minority component better than a global
  linear probe?
- Are posterior standard deviations calibrated, and can prior samples or latent
  interpolation be made biologically meaningful without weakening the selected
  representation?
- Should acquisition partitions be used for block-bootstrap uncertainty or QC
  even though they are not separate latent units?
- Which external cohort or repeat-measurement design should provide the new
  independent validation set?
