# Next Chat Handoff

Updated: 2026-06-20

The biological sample is now the explicit grouping unit. Acquisition `slice`
values are treated as spatial partitions of the same sample and are merged by
`sample_name`; no slice hierarchy is part of the mainline.

The previous probability-mass pilot proved that bounded inputs and
deterministic validation worked, but legacy sigmoid/MSE reconstruction was
approximately 1,237 times worse than the train-mean MSE baseline and selected a
collapsed latent. The repo now includes a generic simplex-softmax decoder,
forward-KL reconstruction, random-input/full-target denoising, and optional
decoder-only generic technical conditioning.

The first implementation smoke passed all 47 tests and completed one epoch on
the attached data with probability mass conserved by target, input, and
reconstruction.

Next one theme: run a longer reconstruction-first pilot with
`condition_mode: none`, small capacity, and `beta=0`. Do not add OT, a mass
head, supervised labels, or adversarial batch removal in the same run.

If the base model clears the train-mean distribution baseline, the following
separate theme is a controlled `none` versus `decoder` technical-conditioning
ablation. Report label-by-batch contingency first; conditioning does not solve
perfect confounding.
