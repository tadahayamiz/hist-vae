# R-260620-02: Latent-KL schedule contract

Status: active
Updated: 2026-06-20

## 1. Purpose

The grouped-measure decoder can now reconstruct the attached validation data
better than a train-mean distribution baseline with `beta=0`. The next strict
step is to introduce the ordinary VAE latent KL gradually without changing the
observation-space representation, decoder, reconstruction loss, or technical
conditioning contract.

The observation-space forward KL and the latent-space KL remain distinct:

```text
recon_kl  = KL(target probability mass || decoded probability mass)
latent_kl = KL(q(z | input) || N(0, I))
loss      = recon_kl + beta(epoch) * latent_kl
```

## 2. Configuration

The pretrainer accepts:

```yaml
beta: 0.0001
latent_kl_schedule: linear_warmup
latent_kl_warmup_epochs: 25
pretrain_monitor: test_recon
```

Supported schedules are:

```text
constant
linear_warmup
```

For `constant`, `beta` is used at every epoch and
`latent_kl_warmup_epochs` must be zero.

For `linear_warmup`, epoch 1 uses zero latent-KL weight, the configured `beta`
is reached exactly at `latent_kl_warmup_epochs`, and that value is retained for
later epochs. Linear warmup requires positive `beta` and at least two warmup
epochs.

## 3. Model-selection rule

A changing beta makes total VAE loss values non-comparable across warmup
epochs. Therefore, `linear_warmup` strictly requires:

```yaml
pretrain_monitor: test_recon
```

Warmup epochs before the final beta is reached are logged but are not eligible
for best-checkpoint selection or early stopping. The first eligible epoch is
`latent_kl_warmup_epochs`; this prevents a low-beta transient from becoming the
canonical checkpoint. The configured total number of epochs must therefore be
at least the warmup length.

The validation reconstruction metric remains deterministic and is evaluated
from the full-group histogram using `z = mu`.

Every epoch records the effective `beta`. `model_best.pt` and
`model_last.pt` record the beta belonging to their saved epoch, and
`history.json` records `best_beta` and `last_beta`.

## 4. Interpretation

Warmup is not evidence that the latent is biologically useful. It is a control
for the rate-distortion trade-off after reconstruction has been established.
Candidate settings must be compared using at least:

```text
- validation forward KL versus the train-mean distribution baseline
- active latent dimensions
- latent posterior statistics
- same-sample random-view stability
- between-sample separation
- biological-label and technical-condition linear probes
```

The exported representation remains the deterministic full-group posterior
mean `mu`.

## 5. Current boundary

Implemented:

```text
- constant latent-KL weight
- zero-to-beta linear warmup
- strict schedule validation
- reconstruction-based model selection beginning at the final warmup epoch
- epoch, history, and checkpoint beta provenance
```

Deferred to separate themes:

```text
- free bits or capacity schedules
- cyclical beta schedules
- WAE/MMD or latent-space OT regularization
- same-sample consistency as a training loss
```

## 6. Related evidence

- E-260620-03
