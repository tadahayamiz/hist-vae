# Next Chat Handoff

Updated: 2026-06-20

The biological sample remains the explicit grouping unit; acquisition-only
`slice` values are pooled by `sample_name`. The canonical path is probability
mass, simplex softmax, forward KL, random input, and deterministic full-group
target. Optional technical conditioning remains decoder-only and is off in the
primary run.

The reconstruction-first gate has now passed on the attached FITC data. With a
small `[8, 16]` encoder/decoder, latent dimension 4, and `beta=0`, validation
forward KL reached `0.02285` versus the train-mean baseline `0.06056`; all four
latent coordinates were active.

The repo now implements `latent_kl_schedule: constant | linear_warmup`. Linear
warmup starts at zero, reaches `beta` at the configured warmup epoch, requires
`pretrain_monitor: test_recon`, and records the effective beta in history and
best/last checkpoints. A single-seed `beta=1e-4`, 25-epoch warmup smoke reached
validation forward KL `0.02128` with 4/4 active coordinates.

Next one theme: run a small unconditioned beta grid over at least three seeds
and evaluate reconstruction, same-sample random-view stability, and
between-sample separation. Do not add OT, a mass head, supervised labels, or
adversarial batch removal in the same phase. Decoder conditioning is the
following separate ablation and requires label-by-condition overlap reporting.
