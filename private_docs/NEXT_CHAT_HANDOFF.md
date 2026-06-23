# Next Chat Handoff

Updated: 2026-06-23

## Current state

Two distinct model lines exist and must not be conflated.

### Frozen benchmark

The absolute-coordinate `log1p(FITC_Sum)` benchmark and its one-time 20-group
holdout are finalized. Do not reopen its beta, architecture, condition mode,
checkpoint, seed, probe, or holdout. Its evidence applies only to that absolute
coordinate contract.

### Selected shape-plus-OT development mainline

The holdout-free coordinate ablation selected:

```text
raw_median_ratio = x / median_group(x) - 1
```

under the prespecified assumption that device variation is primarily a positive
multiplicative gain. It was exactly invariant to 0.5x, 0.75x, 1.5x, and 2.0x
synthetic gains for every seed. See E-260622-00.

The subsequent same-branch OT ablation is complete. The selected model contract
is:

```yaml
group_coordinate_mode: raw_median_ratio
histogram_mode: probability_mass
decoder_output_mode: simplex_softmax
reconstruction_loss: forward_kl

ot_loss: sinkhorn
ot_weight: 6.7578684799473425
ot_p: 1
ot_blur: 0.05
ot_scaling: 0.8
ot_backend: tensorized
ot_mass_epsilon: 0.0

latent_dim: 4
hidden_dims: [8, 16]
beta: 0.0001
latent_kl_schedule: linear_warmup
latent_kl_warmup_epochs: 25
```

## Why the OT weight is 6.757868 rather than 0.1

The selected scientific setting is `ot_factor=0.1`. The actual config accepts
an absolute coefficient, so the experiment first calibrated:

```text
lambda_equal_seed
  = median_train_forward_KL / median_train_Sinkhorn
```

across same-branch KL-only reference models. The three values were:

```text
66.221982
73.841289
67.578685
```

Their median was:

```text
lambda_equal = 67.57868479947342
```

Therefore:

```text
ot_weight = ot_factor * lambda_equal
          = 0.1 * 67.57868479947342
          = 6.7578684799473425
```

Do not set `ot_weight: 0.1`; that would be about 67.6 times weaker than the
selected run.

## OT-ablation result

Three-seed development means:

| Loss | Factor | Forward KL | Sinkhorn | Exact W1 | Between/within | Retrieval | W1-latent | Active |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| KL only | 0.0 | 0.013313 | 0.000358 | 0.004534 | 4.179 | 0.766 | 0.521 | 4/4 |
| KL + Sinkhorn | **0.1** | **0.013428** | **0.000302** | **0.004274** | **4.189** | **0.778** | **0.521** | **4/4** |
| KL + Sinkhorn | 0.3 | 0.015623 | 0.000312 | 0.004298 | 3.803 | 0.751 | 0.561 | 4/4 |
| KL + Sinkhorn | 1.0 | 0.017357 | 0.000289 | 0.004236 | 3.530 | 0.731 | 0.649 | 4/4 |

Factor `0.1` passed the predefined unsupervised gate. It increased forward KL
by only 0.87%, improved Sinkhorn by 15.8%, improved exact W1 by 5.7%, and
increased retrieval by 1.25 percentage points. Factors `0.3` and `1.0` failed
the forward-KL gate. See E-260623-00.

The report-only linear probe remained near chance and played no role in the OT
selection.

## Single-checkpoint artifact choice

Scientific claims use the three-seed aggregate. For one deterministic
collaborator export, use:

```text
raw_median_ratio_kl_sinkhorn_f0p10_seed73
```

It had the lowest validation combined observation loss within the already
selected factor-0.1 setting.

The formal latent artifact is the complete four-dimensional posterior mean
`mu` from the full-group histogram. Do not apply active-unit filtering to the
exported matrix. Standardization is an analysis transform fitted on train, not
a replacement for the raw `mu` artifact.

## Completed collaborator/anomaly follow-up

The deterministic all-sample export, collaborator figures, frozen-feature
readout, and fixed 30-view Healthy-reference anomaly pilot are complete for
`raw_median_ratio_kl_sinkhorn_f0p10_seed73`. The legacy holdout is opened and
descriptive.

The main exploratory observation is a minority-anomaly pattern rather than
global diagnosis separation:

```text
all samples robust novelty: Healthy 4/96, PDAC 5/38
val + opened holdout:       Healthy 2/28, PDAC 2/12
```

The out-of-sample PDAC candidates also had high reconstruction misfit. Do not
interpret this as confirmed disease-specific biology. See E-260623-01.

## New reusable support

The repository now provides:

```text
HistVAE.get_multiview_latent()
aggregate_latent_views()
reference_knn_from_distances()
euclidean_reference_knn()
empirical_reference_percentile()
JointSinkhornDivergence.pairwise()
```

These APIs contain no diagnosis, path, run-name, or current-dataset logic. The
multiview method is inference-only and works with the existing dimension-aware
histogram model. No consistency loss or metric-learning loss was added.

## Next one theme

Run the frozen-model multi-view latent/direct-shape concordance experiment from
R-260623-01.

```text
30 deterministic event views
-> aggregate posterior means per sample
-> report view SD / RMS displacement
-> train-Healthy latent mean-kNN score
-> train-Healthy joint-Sinkhorn mean-kNN score on mean sampled histogram
-> compare correlation, candidate overlap, and view-repeat stability
-> train-fitted PCA/UMAP of aggregated latent for display
```

Use the existing train-fitted latent standardization. The direct shape baseline
must use the complete joint Sinkhorn support so the same analysis applies to
1D, 2D, and 3D. Carry forward the fixed exploratory thresholds:

```text
shape percentile >= 0.95
latent percentile >= 0.90
latent threshold exceeded in >= 0.80 of views
reference k = 5
```

Do not change the model. If aggregated latent does not add stability or useful
organization over direct Sinkhorn distance, retain direct distance and do not
introduce a new training loss.

## Holdout warning

The legacy 20-group holdout was not used in the coordinate or OT ablations, but
it was already used to finalize the older absolute-coordinate benchmark. It is
not a genuinely new independent holdout for the shape-plus-OT model.

If the all-sample export includes it, the manifest must record it as opened for
this model. The resulting plots and metrics are descriptive only and must not
be used to change the coordinate, OT weight, latent width, checkpoint rule,
threshold, or downstream model. Confirmatory evaluation requires a new
independent cohort.

## Latent width

Keep `latent_dim=4` as the mainline. All four dimensions are active in every
seed. An eight-dimensional run is optional sensitivity analysis only and has no
current priority unless a clear residual-capacity question emerges.

## Fresh-VM rule

Assume every Colab/VM starts empty. The first cell must:

```text
mount Drive/storage
-> clone the named dev-2026 branch
-> record the resolved branch HEAD
-> install the repo and complete planned analysis stack
-> validate/copy data, split, checkpoint, normalizer, and preprocessor
-> record provenance
```

The user should not need to type a commit hash. No later cell may rely on a
previous VM session. Install scalable OT, linear-probe, UMAP, plotting, and
reporting dependencies in the first setup cell whenever they are part of the
planned run.

## Do not do next

- Do not retune the coordinate or OT factor from PCA/UMAP appearance.
- Do not select by combined observation loss across different OT weights.
- Do not treat Healthy/Cancer color separation as diagnostic validation.
- Do not drop low-variance latent dimensions from the formal artifact.
- Do not reopen the frozen absolute-coordinate benchmark.
- Do not claim that synthetic gain invariance proves all real device variation
  is multiplicative.
