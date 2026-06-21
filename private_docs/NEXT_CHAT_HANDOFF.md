# Next Chat Handoff

Updated: 2026-06-21

The shape-only grouped-measure mainline is selected and the holdout is
finalized. Use pooled `sample_name` groups; acquisition `slice` values remain
QC metadata only.

Frozen configuration:

```text
probability mass + log1p bins + clipping at 100,000
64 bins; random 1,024-event train input; full-group target/evaluation
latent dimension 4; hidden dimensions [8, 16]
simplex-softmax decoder; forward-KL reconstruction
beta 1e-4 with 25-epoch linear warmup
300-epoch ceiling; patience 20; RAdam
condition_mode none
seeds 17, 42, 73
```

The beta convergence experiment selected `1e-4`; the decoder-conditioning
ablation selected `none`. The final 20-group holdout was evaluated once with
all three fixed seeds. Mean holdout forward KL was `0.006251` versus the
train-mean baseline `0.051049`; all samples beat the baseline, all four latent
dimensions were active, random-view retrieval was `0.829`, and input-W1 versus
latent-distance Spearman was `0.882`.

The fixed disease-label probe gave mean AUC `0.734`; the secondary probability
average gave AUC `0.774`. Treat this as exploratory because the holdout has only
six PC samples and the expected disease signal may be sparse rather than a
clean two-cluster shift.

Do not use the finalized holdout to alter beta, architecture, condition mode,
checkpoint, seed, probe hyperparameters, threshold, or a new loss. Do not pick
the best seed. The current main artifact is deterministic full-group posterior
mean `mu` from all three fixed seeds.

Raw-space visualization is now implemented. `Histogram.get_bin_edges("raw")`
inverts log1p bin geometry; `HistVAE.check_data()` and
`HistVAE.plot_reconstruction()` default to raw coordinates. For probability
mass on unequal raw-width bins, `plot_value_mode="auto"` displays raw-space
density while forward KL remains computed in model mass space. The sampled
reconstruction mode uses an explicit seed and a full-group target. Focused
visualization tests passed 8/8, and the full suite including slow tests passed
65/65.

Train-fitted preprocessing is now a reusable API for future datasets:

```text
AxisPreprocessingSpec
HistogramPreprocessor.fit(train_data, train_group)
HistVAE(..., histogram_preprocessor=preprocessor)
HistVAE.prep_data(...)
```

Transforms are explicit per axis (`none` or `log1p`). Lower and upper bounds
can independently be fixed or fitted as training quantiles. Percentiles can
weight events equally or give each biological group equal total weight. The
new path permits only `clip` or `error`, persists safe-YAML state and hashes,
and reuses the same fitted object on validation/holdout. Constructor injection
is the canonical path because the fitted state replaces stale default geometry
before model-contract validation. Do not refit it on validation or holdout. The
actual-data smoke fitted a group-equal q0.999 upper bound of 97,626 and produced
finite 64-bin simplex tensors. The finalized FITC checkpoints retain their
original 0-to-100,000 log1p/clip contract and are not reopened by this
implementation.

Next one theme: generate the final descriptive reconstruction figures and
three-seed latent exports from the frozen artifacts. Do not use visual quality
on the finalized holdout to select a seed or alter the model. Any
abundance/rate, tail-sensitive, OT, supervised, one-class, or prior-generation
study starts a new development cycle with new validation data.
