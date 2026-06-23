# R-260623-00: Selected raw-median-ratio plus weak-joint-Sinkhorn mainline

Status: active
Updated: 2026-06-23

## 1. Decision

The current shape-oriented HistVAE development mainline is:

```text
complete biological group
-> raw median-ratio coordinate
-> train-fitted probability-mass histogram
-> random-input / full-target denoising VAE
-> forward KL + weak joint Sinkhorn observation loss
-> four-dimensional posterior mean artifact
```

The selected coordinate is:

```text
u_si = x_si / median_group(x_s) - 1
```

The selected observation objective is:

```text
L_observation
  = KL(p || p_hat)
  + 6.7578684799473425 * Sinkhorn(p, p_hat)

L_total
  = L_observation
  + beta(t) * KL(q(z | input) || N(0, I))
```

with `beta=1e-4` after a 25-epoch linear warmup.

## 2. Canonical configuration

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

optimizer: radam
lr: 0.001
batch_size: 16
epochs: 300
patience: 20

condition_mode: none
condition_dim: 0
```

This configuration is an assay-specific selected operating point, not a change
to the package's legacy-compatible defaults.

## 3. Meaning of `ot_factor` and `ot_weight`

The ablation used a dimensionless relative factor to make weight choices
interpretable across the different numerical scales of KL and Sinkhorn.
For each KL-only seed:

```text
lambda_equal_seed
  = median_train_forward_KL / median_train_Sinkhorn
```

The median across seeds was:

```text
lambda_equal = 67.57868479947342
```

The selected factor was:

```text
ot_factor = 0.1
```

and the actual repository coefficient is:

```text
ot_weight = ot_factor * lambda_equal
          = 6.7578684799473425
```

`ot_factor` belongs in experiment provenance. `ot_weight` belongs in the
runtime config. Setting `ot_weight=0.1` would not reproduce the selected model.

## 4. Why weak OT was selected

Across three development seeds, factor `0.1` versus KL-only:

```text
forward KL:      +0.87%
Sinkhorn:        -15.8%
exact 1D W1:     -5.7%
retrieval:       +1.25 percentage points
between/within:  maintained
W1-latent:       maintained
active dims:     4/4 for every seed
```

This supports a narrow claim: weak observation-space OT improves geometric
reconstruction on the tested development split at little cost to base
probability fidelity or view stability.

Factors `0.3` and `1.0` were rejected. Although they increased some geometry
metrics, they worsened forward KL beyond the predefined gate and reduced
sample separation or retrieval. The method is therefore not "more OT is
better"; OT is a weak auxiliary.

## 5. Multidimensional contract

The same loss definition applies in 1D, 2D, and 3D. The complete joint
histogram is flattened over the joint bin support, and transport uses the joint
bin-center geometry. Axis-wise marginal Wasserstein distances are not averaged.
Consequently, cross-dimensional patterns such as diagonal or quadrant-specific
mass remain represented.

Each axis is globally scaled from its train-fitted histogram range for the OT
ground metric, then the joint coordinate is divided by `sqrt(d)`. This is a
metric transformation only; it does not normalize away sample-specific width
or alter the histogram mass.

## 6. Representation artifact

The canonical sample artifact is:

```text
mu_s = posterior mean from the full-group histogram
```

All four dimensions are retained. Store alongside:

```text
posterior SD per dimension
raw group median
raw group IQR
event count
exposure-normalized rate when available
split and technical metadata
reconstruction forward KL
reconstruction Sinkhorn divergence
```

Active-unit filtering is a collapse diagnostic only. It is not the formal
feature-selection rule. For PCA, UMAP, and regularized linear models, fit
standardization on training samples only and transform other samples with the
stored train statistics.

## 7. Reporting checkpoint

The scientific result is the three-seed aggregate. When one checkpoint is
needed for deterministic figures and CSV export, use:

```text
raw_median_ratio_kl_sinkhorn_f0p10_seed73
```

This was the lowest validation combined observation loss within the already
selected factor-0.1 family. The exported latent should use full-group input and
`sample_latent=False`, so `z=mu`.

## 8. PCA, UMAP, and reconstruction display contract

For collaborator-facing reporting:

```text
latent standardization: fit on train only
PCA: fit on train only; transform validation/opened descriptive samples
UMAP: fit on train only; transform validation/opened descriptive samples
Healthy/Cancer labels: color only; never used to fit the embedding
formal latent CSV: raw four-dimensional mu retained
```

Input-versus-reconstruction examples must use a deterministic rule rather than
best-case visual selection. The current rule is the sample nearest to the 50th
and 90th forward-KL percentile within each diagnosis, plus an all-sample
multipage PDF for auditability.

## 9. Holdout and validation boundary

The coordinate and OT ablations did not access the legacy 20-group holdout.
That group set was already used to finalize the older absolute-coordinate
benchmark and is not a new independent holdout for this mainline.

If included in an all-sample export, it must be labeled opened/descriptive and
must not influence any later choice. Confirmatory claims require a new
independent cohort.

## 10. Latent width decision

Keep `latent_dim=4` as the mainline. All four dimensions were active for every
seed, and the current model already balances reconstruction and view stability.
An eight-dimensional model is optional sensitivity analysis only. It should not
be adopted without a clear multi-seed improvement of roughly practical size
while preserving retrieval, separation, seed stability, and generalization.

## 11. Claim boundary

Supported:

- exact positive multiplicative-gain invariance of the coordinate contract;
- valid joint 1D/2D/3D OT implementation;
- weak-OT geometric improvement on the tested development split;
- deterministic four-dimensional sample representation.

Not supported:

- universal superiority of raw ratio over all coordinate systems;
- proof that real device variation is purely multiplicative;
- equality between latent Euclidean distance and Wasserstein distance;
- diagnostic Healthy/Cancer separation;
- calibrated posterior uncertainty or unconditional generation;
- confirmatory generalization of the selected shape-plus-OT model.

## 12. Related evidence

- E-260622-00: KL-only coordinate ablation
- E-260623-00: joint-Sinkhorn weight ablation
