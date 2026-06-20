# Open Questions

Updated: 2026-06-20

- How many epochs and what small capacity are required for simplex/forward-KL
  reconstruction to beat the train-mean distribution baseline?
- Which latent-KL warmup or free-bits setting preserves active latent
  coordinates after the reconstruction-first phase?
- Should total event intensity be represented as raw mass, exposure-normalized
  rate, or both when reliable exposure metadata are available?
- Which dataset columns define a valid technical batch, and how strongly is
  that batch associated with disease label?
- Does decoder-only technical conditioning improve cross-condition reconstruction
  and reduce latent batch predictability without removing disease-related signal?
- What same-sample random-view stability threshold should be required before a
  latent is used downstream?
- Does exact 1D Wasserstein reconstruction add value beyond forward KL?
- Should coordinate jitter be introduced only after its scale is estimated from
  technical controls or repeat measurements?
- Which downstream sample-level task should be used for final model selection?
