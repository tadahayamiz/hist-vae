# Open Questions

Updated: 2026-06-23

## Resolved for the frozen absolute-coordinate benchmark

- Small architecture: `latent_dim=4`, `hidden_dims=[8, 16]`.
- Latent regularization: `beta=1e-4`, 25-epoch linear warmup.
- Training ceiling and stopping: 300 epochs with patience 20.
- Technical conditioning: `condition_mode=none`.
- Primary exported artifact: deterministic full-group posterior mean `mu`.
- Probability mass removes event abundance but does not remove coordinate
  location; the frozen model is not device-shift invariant.
- The finalized absolute-coordinate holdout is unavailable for additional
  selection.

## Resolved for the shape-plus-OT development mainline

- Main coordinate: `raw_median_ratio = x / median_group(x) - 1` under the
  prespecified positive multiplicative-gain assumption.
- Sensitivity coordinate: `log_median_center`.
- Median-plus-IQR scaling: sensitivity analysis only.
- Observation loss: forward KL plus weak joint Sinkhorn.
- Selected relative OT strength: `ot_factor=0.1`.
- Calibrated config coefficient: `ot_weight=6.7578684799473425`.
- OT geometry: joint support, `p=1`, `blur=0.05`, `scaling=0.8`.
- Latent width: retain `4` as the mainline; all dimensions are exported.
- Single deterministic collaborator artifact: factor-0.1 seed-73 checkpoint.
- Linear probing: report-only and excluded from unsupervised selection.
- Coordinate and OT ablations did not use the legacy holdout.

## Remaining coordinate and measurement questions

- Do physical technical replicates confirm that real device variation is
  predominantly multiplicative rather than an additive-plus-multiplicative
  mixture?
- Should the full-group median remain the canonical statistic at deployment, or
  must view-local median uncertainty be modeled for low-event samples?
- Which calibrated raw-median metadata can be interpreted biologically across
  instruments after the shape coordinate removes group-level gain?
- Should acquisition partitions be used for block-bootstrap uncertainty or QC
  even though they are not separate latent units?

## Remaining OT and dimensionality questions

- At what joint bin count must the tensorized backend be replaced by the same
  joint-Sinkhorn contract using the online or multiscale PyKeOps backend?
- Does `ot_p=1` remain the fixed method definition for future multidimensional
  assays, or is a separate prespecified `p=2` sensitivity analysis warranted?
- Does the selected weak OT term remain beneficial in genuinely independent
  data and in 2D/3D assays?
- An eight-dimensional latent is optional sensitivity analysis only. It should
  be tested only if a concrete residual-capacity question arises and adopted
  only after clear multi-seed improvement without loss of stability.

## Artifact and validation questions

- Will the collaborator export include the legacy holdout? If yes, the manifest
  must mark it as opened/descriptive and it cannot support further model
  selection.
- Which external cohort or repeat-measurement design will provide the new
  independent confirmatory set for the shape-plus-OT model?
- Should PCA/UMAP be shown for all splits together, or should the main figure
  emphasize train/validation and place opened descriptive samples in a
  supplement?

## Other open research questions

- Should total event abundance be represented as total mass,
  exposure-normalized rate, or both when the exposure denominator is reliable?
- Do prespecified high-intensity tail mass or one-class novelty measures capture
  a hypothesized disease-specific minority component better than a global
  linear probe?
- Are posterior standard deviations calibrated, and can prior samples or latent
  interpolation be made biologically meaningful without weakening the selected
  representation?
