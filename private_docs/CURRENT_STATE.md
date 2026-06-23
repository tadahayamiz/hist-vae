# Current State

Updated: 2026-06-23

## Repository identity

HistVAE learns sample-level latent representations from grouped,
low-dimensional point data by converting each biological sample into a
histogram or empirical measure and applying a dimension-aware convolutional
VAE.

## Current objective

The finalized FITC model remains a frozen absolute-coordinate `log1p`
probability-mass benchmark. It is stable under finite-event subsampling but is
not a device-invariant shape-only model.

A separate holdout-free development cycle has now fixed the shape-oriented
mainline as:

```text
raw_median_ratio
+ probability-mass histogram
+ forward KL
+ weak joint Sinkhorn auxiliary loss
```

For group `s`, the coordinate is

```text
u_si = x_si / median_group(x_s) - 1
```

which is exactly invariant to positive multiplicative gain. It retains relative
width, skewness, multimodality, and relative tails. Total event abundance and
the removed raw median remain separate sample metadata rather than implicit
components of the shape latent.

The OT ablation is complete. The selected relative OT strength is
`ot_factor=0.1`, corresponding to the actual config value
`ot_weight=6.7578684799473425` after scale calibration. This weak auxiliary
term improved validation Sinkhorn divergence and exact 1D W1 while preserving
forward-KL fidelity, random-view retrieval, between/within separation, and all
four active latent dimensions.

## Sample contract

- `group` is the biological sample whose latent representation is required.
- Acquisition-only partitions such as image `slice` values are merged when
  they tile a larger measured region.
- For the attached data, all rows with the same `sample_name` form one sample.
- `slice` remains QC or future block-resampling metadata; it is not a separate
  latent level in the selected model.

## Selected shape-plus-OT development mainline

```yaml
group_coordinate_mode: raw_median_ratio

histogram_mode: probability_mass
value_transform: none
out_of_range_policy: clip
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

ot_loss: sinkhorn
ot_weight: 6.7578684799473425
ot_p: 1
ot_blur: 0.05
ot_scaling: 0.8
ot_backend: tensorized
ot_mass_epsilon: 0.0

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

The scientific relative-strength setting is `ot_factor=0.1`. The repository
accepts the calibrated loss coefficient as `ot_weight`; therefore
`ot_weight=0.1` is not equivalent to the selected experiment.

The formal sample representation is the full four-dimensional posterior mean
`mu` from the deterministic full-group histogram. Active-dimension counting is
a collapse diagnostic only; dimensions are not dropped from the exported
matrix or standard downstream analyses.

## OT-weight calibration and ablation result

The KL-only references were retrained on the same branch snapshot as the OT
models. For each seed,

```text
lambda_equal_seed
  = median_train_forward_KL / median_train_Sinkhorn
```

was computed. The three values were `66.221982`, `73.841289`, and `67.578685`;
the median was:

```text
lambda_equal = 67.57868479947342
```

The tested weights were `ot_factor * lambda_equal`:

```text
0.1 ->  6.7578684799473425
0.3 -> 20.273605439842026
1.0 -> 67.57868479947342
```

Three-seed development means were:

| Loss | OT factor | Forward KL | Sinkhorn | Exact W1 | Between/within | Retrieval | W1-latent | Active |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| KL only | 0.0 | 0.013313 | 0.000358 | 0.004534 | 4.179 | 0.766 | 0.521 | 4/4 |
| KL + Sinkhorn | **0.1** | **0.013428** | **0.000302** | **0.004274** | **4.189** | **0.778** | **0.521** | **4/4** |
| KL + Sinkhorn | 0.3 | 0.015623 | 0.000312 | 0.004298 | 3.803 | 0.751 | 0.561 | 4/4 |
| KL + Sinkhorn | 1.0 | 0.017357 | 0.000289 | 0.004236 | 3.530 | 0.731 | 0.649 | 4/4 |

`ot_factor=0.1` increased forward KL by only `0.87%`, reduced Sinkhorn by
`15.8%`, reduced exact W1 by `5.7%`, and increased retrieval by `1.25`
percentage points. Factors `0.3` and `1.0` failed the prespecified forward-KL
gate and were rejected. The linear probe remained report-only and near chance;
it was not used to select the OT weight.

## Single-checkpoint reporting choice

Scientific conclusions remain based on the three-seed summary. For a single
collaborator-facing deterministic export, use:

```text
raw_median_ratio_kl_sinkhorn_f0p10_seed73
```

because it had the lowest validation combined observation loss within the
already selected `ot_factor=0.1` setting. This is an artifact-selection rule,
not evidence that seed 73 is a distinct scientific model class.

## Frozen absolute-coordinate benchmark

The earlier absolute-coordinate `log1p(FITC_Sum)` benchmark remains frozen with
its own finalized evidence chain. Across its 20 holdout groups:

```text
mean full-group forward KL:           0.006251 +/- 0.001777
train-mean forward-KL baseline:       0.051049
mean reconstruction improvement:      87.75% +/- 3.48%
groups beating the baseline:          20 / 20 for every seed
active latent dimensions:             4 / 4 for every seed
between/within random-view ratio:      5.146 +/- 0.502
random-view retrieval accuracy:       0.829 +/- 0.027
input-W1 / latent-distance Spearman:  0.882 +/- 0.053
```

Those results apply only to the frozen absolute-coordinate contract and do not
establish device-shift invariance or validate the new shape-plus-OT mainline.
The disease-label probe from that benchmark remains exploratory, not clinical
validation.

## Evidence and holdout boundary

The coordinate and OT ablations used `94` training and `20` validation groups
with seeds `17`, `42`, and `73`. The legacy 20-group holdout was not accessed
for either ablation. However, that group set was already used to finalize the
older absolute-coordinate benchmark and therefore is not a new independent
holdout for the shape-plus-OT development cycle.

It may be exported descriptively together with all samples only if the artifact
manifest explicitly records that it has been opened for this model. It must not
then be used for further hyperparameter, threshold, architecture, or checkpoint
selection. A genuinely new independent cohort is required for confirmatory
evaluation of the selected shape-plus-OT model.

## Model interpretation

The selected model is a weakly VAE-regularized denoising distributional
autoencoder with a geometry-aware observation auxiliary. Forward KL preserves
probability-mass fidelity; joint Sinkhorn adds a distance-aware penalty on the
complete 1D/2D/3D joint histogram support. This is observation-space OT, not a
Wasserstein autoencoder prior-matching objective.

The current evidence supports deterministic sample representation and improved
geometric reconstruction under the tested development split. It does not
establish calibrated posterior uncertainty, realistic unconditional generation,
causal batch correction, biological mechanism, or clinical diagnostic
performance.

## Exploratory Healthy-reference anomaly evidence

The collaborator export and fixed follow-up analysis are complete for the
selected seed-73 checkpoint. Healthy and PDAC mean shapes overlap strongly, and
the four-dimensional latent does not provide stable global Healthy-versus-PDAC
separation. The exploratory 30-view anomaly pilot instead supports a
minority-anomaly hypothesis:

```text
all samples:                 Healthy 4/96, PDAC 5/38 robust novelty
validation + opened holdout: Healthy 2/28, PDAC 2/12 robust novelty
```

This enrichment is descriptive. The current encoder was trained on all
diagnoses, the normal reference was not cross-fitted, and the legacy holdout is
opened. Out-of-sample PDAC candidates also had high reconstruction misfit, so
technical and finite-event effects remain unresolved. See E-260623-01.

The immediate methodological question is whether the existing latent adds
sampling stability or organization beyond direct histogram distance. The
minimal next contract therefore leaves the model frozen, aggregates repeated
posterior means per sample, and compares Healthy-reference latent kNN distance
with joint nD Sinkhorn kNN distance. See R-260623-01.

## Active references

- `R-260619-00`: histogram representation mode
- `R-260620-01`: grouped-measure representation contract
- `R-260620-02`: latent-KL schedule contract
- `R-260621-00`: frozen absolute-coordinate grouped-measure benchmark
- `R-260621-01`: raw-space histogram and reconstruction visualization
- `R-260621-02`: train-fitted histogram preprocessing
- `R-260623-00`: selected raw-median-ratio plus weak-joint-Sinkhorn mainline
- `R-260623-01`: multi-view latent aggregation and reference scoring

`R-260621-03` is retained as the superseded development contract that led to
the selected mainline.

## Active evidence

- `E-260619-00`: attached FITC density smoke validation
- `E-260620-00`: bounded-input and deterministic-evaluation smoke
- `E-260620-01`: legacy sigmoid probability-mass failure
- `E-260620-02`: grouped-measure implementation smoke
- `E-260620-03`: reconstruction gate and latent-KL warmup smoke
- `E-260621-00`: multi-seed beta convergence and selection
- `E-260621-01`: decoder-conditioning ablation
- `E-260621-02`: finalized absolute-coordinate holdout evaluation
- `E-260621-03`: log-domain shape-normalization pilot
- `E-260622-00`: KL-only raw shape-coordinate ablation
- `E-260623-00`: joint-Sinkhorn weight ablation on fixed `raw_median_ratio`
- `E-260623-01`: Healthy-reference anomaly pilot on the frozen seed-73 model

## Next action

The next one-theme task is a frozen-model, dimension-general comparison:

```text
30 deterministic point-subsample views per sample
-> posterior mean for every view
-> sample position = mean posterior mu across views
-> sampling instability = view SD / RMS displacement
-> train-Healthy latent mean-kNN score
-> train-Healthy joint-Sinkhorn mean-kNN score on the mean sampled histogram
-> compare score concordance, candidate overlap, and view stability
```

No training loss, architecture, coordinate, OT weight, checkpoint, or diagnosis
head is changed. The same direct-shape path uses the existing full joint support
and applies to 1D, 2D, and 3D probability histograms.

The fixed exploratory thresholds remain shape percentile 0.95, latent
percentile 0.90, and latent-view exceedance rate 0.80. They must not be retuned
from PDAC identities or the opened holdout. If the latent does not improve
stability or organization over direct Sinkhorn distance, retain the direct
metric and do not introduce metric-learning or view-consistency losses.

## Fresh-VM execution policy

Assume every Colab/VM session starts empty. Each experiment or export notebook
begins with one self-contained setup cell that mounts storage, clones the named
`dev-2026` branch, records its resolved HEAD, installs the repo and the complete
planned dependency stack, validates input artifacts, and records provenance.
The user does not need to specify a commit manually. Later cells must not depend
on state from an earlier VM session.

## Deferred or out of scope

- Optional total mass or exposure-normalized event-rate branch
- Tail-sensitive or cancer-specific rare-event representation
- Median-plus-IQR normalization as a mainline rather than sensitivity analysis
- Eight-dimensional latent sensitivity analysis
- Point-coordinate measurement-noise augmentation
- Adversarial batch removal
- Prior-calibrated unconditional generation
- Clinical diagnostic claims
- CLI cleanup
