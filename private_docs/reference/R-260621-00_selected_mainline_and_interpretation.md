# R-260621-00: Selected grouped-measure mainline and interpretation

Status: active
Updated: 2026-06-21

## 1. Selected use case

The selected mainline learns one deterministic sample-level representation from
pooled, low-dimensional point measurements. For the current assay, all
`FITC_Sum` observations sharing a `sample_name` form one biological sample;
acquisition `slice` values tile the measured region and are pooled rather than
encoded as independent replicates.

The primary artifact is the posterior mean `mu` obtained from the deterministic
full-group probability-mass histogram. The model is intended to preserve stable
sample-to-sample differences in distribution location and shape while reducing
finite-event sampling noise.

## 2. Frozen shape-only configuration

The selected configuration is:

```yaml
histogram_mode: probability_mass
value_transform: log1p
out_of_range_policy: clip
max_vals: [100000.0]
bins: 64

train_sampling_mode: random
train_target_sampling_mode: full
eval_sampling_mode: full
eval_target_sampling_mode: full
num_points: 1024
transform: false

latent_dim: 4
hidden_dims: [8, 16]
dropout_conv: 0.0

decoder_output_mode: simplex_softmax
reconstruction_loss: forward_kl

beta: 0.0001
latent_kl_schedule: linear_warmup
latent_kl_warmup_epochs: 25
pretrain_monitor: test_recon

epochs: 300
patience: 20
optimizer: radam

condition_mode: none
condition_dim: 0
```

`epochs: 300` is a ceiling rather than an expected run length. The selected
three runs stopped between epochs 110 and 186 after the reconstruction-best
epoch plus 20 patience epochs.

## 3. Loss interpretation

The fitted objective is:

```text
loss = recon_kl + beta(epoch) * latent_kl
```

where:

```text
recon_kl  = KL(full-group target mass || decoded mass)
latent_kl = KL(q(z | random input mass) || N(0, I))
```

The first term is an observation-space distribution reconstruction loss. The
second is the ordinary VAE latent regularizer. They have different roles and
must remain separately logged and interpreted.

The decoder applies a spatial softmax over all bins, so every reconstruction is
a non-negative probability mass summing to one. The input is a random 1,024-
event view during training, while the target is the deterministic distribution
from all events in the sample. Evaluation and exported latents use the full
sample and `z = mu`.

## 4. Empirical model selection

The beta target was selected before holdout evaluation from three seeds and a
fixed validation split. `beta=1e-4` was the only candidate satisfying all of:

```text
- every seed beat the train-mean distribution baseline
- all four train and validation latent dimensions were active
- mean validation forward KL was within 10% of the best candidate
- every seed converged by patience-based early stopping
```

Decoder-only technical conditioning was then rejected as the primary setting.
Although permuting the condition worsened reconstruction in all three runs,
conditioning did not consistently improve reconstruction or latent geometry
and did not reduce measured batch signal. The unconditioned model is therefore
the selected mainline.

## 5. Final holdout interpretation

The final 20-group holdout was evaluated once with all three fixed seeds and no
seed selection. Across seeds:

```text
holdout full forward KL:              0.006251 +/- 0.001777
train-mean holdout forward KL:        0.051049
reconstruction improvement:           87.75% +/- 3.48%
holdout groups beating baseline:      20 / 20
active latent dimensions:             4 / 4 for every seed
between/within random-view ratio:      5.146 +/- 0.502
random-view retrieval accuracy:       0.829 +/- 0.027
chance retrieval accuracy:            0.05
input-W1 / latent-distance Spearman:  0.882 +/- 0.053
```

These results support the claim that the model learned a stable sample-level
representation of the measured one-dimensional distributions. The latent
retains the geometry of the input distributions and is substantially more
stable within repeated finite-event views than it is similar across samples.

## 6. Disease-label interpretation

Disease labels were not used for pretraining, beta selection, condition-mode
selection, checkpoint selection, or seed selection. A fixed linear probe
trained on train plus validation latents produced the following holdout results:

```text
per-seed ROC AUC:              0.734 +/- 0.054
per-seed balanced accuracy:    0.718 +/- 0.096
three-seed probability AUC:    0.774  (secondary descriptive result)
```

The holdout contains only six PC and fourteen C samples. The scientific prior
is that the overall distributions are visually similar and that only a subset
of cancer samples may contain a cancer-specific component. Therefore, the
probe is interpreted as exploratory evidence of weak disease-related
information, not as diagnostic validation or evidence of two well-separated
clusters.

The negative or near-zero label silhouette observed during development is not
inconsistent with a moderate AUC: a sparse or one-directional disease signal
can support ranking without forming two compact clusters.

## 7. VAE claim boundary

The model did not collapse: all four latent dimensions remained active and
sample identity was robust to random finite-event views. However, the selected
beta is small, the holdout posterior standard deviation averaged approximately
`0.034`, and the latent KL remained approximately `18.2`. The selected model is
therefore best described as a weakly VAE-regularized denoising distributional
autoencoder.

The current evidence supports use of deterministic `mu` as a representation.
It does not establish:

```text
- calibrated uncertainty from posterior standard deviations
- realistic unconditional generation from z ~ N(0, I)
- biologically valid interpolation everywhere in latent space
- causal removal of batch effects
- clinical diagnostic performance
```

## 8. Finalization rule

The current holdout is finalized. Its results must not be used to change beta,
architecture, condition mode, checkpoint, seed, probe hyperparameters, or a
decision threshold. The three fixed seeds remain co-primary evaluation runs;
no best seed is selected. The probability ensemble is secondary and
descriptive.

Any new tail-sensitive loss, abundance/rate branch, OT term, supervised head,
or one-class disease model begins a new development cycle and requires nested
development evaluation or an independent cohort.

## 9. Related evidence

- E-260621-00: beta convergence and selection
- E-260621-01: decoder-conditioning ablation
- E-260621-02: finalized holdout evaluation
