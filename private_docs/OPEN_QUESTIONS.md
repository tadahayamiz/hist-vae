# Open Questions

Updated: 2026-06-22

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
- Under the prespecified multiplicative-gain contract, `raw_median_ratio` is the
  selected development coordinate; `log_median_center` remains sensitivity
  analysis and the finalized holdout was untouched.
- Synthetic 0.5x, 0.75x, 1.5x, and 2.0x gains produce exactly zero input and
  latent shift after `raw_median_ratio` normalization.

## Remaining coordinate questions

- Do physical technical replicates confirm that real device variation is
  predominantly multiplicative rather than an additive-plus-multiplicative
  mixture?
- Should the full-group median remain the canonical statistic at deployment, or
  must view-local median uncertainty also be modeled for low-event samples?
- Which calibrated raw-median metadata can be interpreted biologically across
  instruments after the shape coordinate removes group-level gain?

## Active OT questions

- What `ot_blur` and `ot_weight` retain forward-KL fidelity while improving
  geometric reconstruction on fixed `raw_median_ratio` data?
- Does observation-space joint Sinkhorn improve exact 1D W1, quantile/tail
  fidelity, view stability, or latent geometry across three seeds?
- At what joint bin count must the tensorized backend be replaced by the same
  mathematical contract using the online or multiscale PyKeOps backend?
- Does `ot_p=1` remain preferable to `ot_p=2` after the primary weight ablation,
  or should `p` be held fixed as part of the method definition?

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
