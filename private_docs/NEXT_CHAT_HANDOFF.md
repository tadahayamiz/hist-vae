# Next Chat Handoff

Updated: 2026-06-20

The first attached-data density run completed, but sampled density inputs could
exceed the sigmoid decoder range and total validation loss selected epoch 1 as
KL increased. The repository now has a bounded `probability_mass` path,
log-spaced-bin support, explicit overflow handling, deterministic full-group
validation, active-latent logging, and canonical best checkpoint restoration.

For the attached `FITC_Sum` data, the reliability smoke used
`histogram_mode="probability_mass"`, `value_transform="log1p"`,
`out_of_range_policy="clip"`, 64 bins, and a training-only 99.9th-percentile
range rounded to 100,000. This is technical evidence, not a final recipe.

Next theme: run a small controlled pilot across model capacity, beta, bin/range
choice, and three seeds. Do not start the full pretraining run or mix this with
CLI cleanup before the pilot decision is recorded.
