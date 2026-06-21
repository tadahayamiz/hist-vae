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

Next one theme: prepare reproducible figures/tables and latent exports from the
existing finalized artifacts. Any abundance/rate, tail-sensitive, OT,
supervised, one-class, or prior-generation study starts a new development
cycle with new validation data.
