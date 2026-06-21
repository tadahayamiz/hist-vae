# Open Questions

Updated: 2026-06-21

## Resolved for the current shape-only mainline

- Small architecture: `latent_dim=4`, `hidden_dims=[8, 16]`.
- Latent regularization: `beta=1e-4`, 25-epoch linear warmup.
- Training ceiling and stopping: 300 epochs with patience 20.
- Technical conditioning: `condition_mode=none` for the current dataset.
- Primary representation: deterministic full-group posterior mean `mu`.
- Current holdout: finalized and unavailable for additional selection.

## Open research questions

- Should total event abundance be represented as total mass,
  exposure-normalized rate, or both when the exposure denominator is reliable?
- Do pre-specified high-intensity tail mass or one-class novelty measures capture
  the hypothesized cancer-specific minority component better than a global
  linear probe?
- Can such a rare-event or abundance-aware extension improve on a new cohort or
  nested development protocol without sacrificing sampling stability?
- Does exact 1D Wasserstein reconstruction add information beyond the strong
  forward-KL and W1-geometry results already observed?
- Are posterior standard deviations calibrated, and can prior samples or latent
  interpolation be made biologically meaningful without weakening the selected
  representation?
- How stable are latent coordinates across cohorts, instruments, and acquisition
  dates after alignment of independently trained seeds or models?
- Should acquisition partitions be used for block-bootstrap uncertainty or QC
  even though they are not separate latent units?
- Which external cohort or repeat-measurement design should be used for
  independent biological validation?
