# R-260620-00: Reliable pretraining representation and evaluation contract

Status: superseded  
Updated: 2026-06-20

## Decision

Preserve legacy `count` and `density` modes and add
`histogram_mode="probability_mass"` for bounded distribution-shape input.
Probability mass is computed as in-range bin count divided by total retained
count and bypasses legacy log1p/per-group-maximum normalization.

## Range contract

- `value_transform="none"` uses linear bins.
- `value_transform="log1p"` uses log1p-spaced bins while callers specify
  `max_vals` in raw units.
- `out_of_range_policy="drop"` preserves legacy exclusion.
- `clip` puts underflow/overflow into edge bins.
- `error` rejects out-of-range values.

## Evaluation contract

Validation uses one deterministic full-group histogram per sample and decodes
from `mu` rather than a sampled latent. `get_latent()` follows the same
full-group input contract.

## Checkpoint contract

Pretraining monitoring is configurable. The packaged default monitors
validation reconstruction. The trainer always tracks and restores the best
model and optimizer state, writes it to `model_best.pt`, and separately writes
the terminal state to `model_last.pt`.

## Diagnostics

Validation history records reconstruction, KL, mean latent-coordinate standard
deviation, and the number of coordinates above `active_latent_threshold`.
These are diagnostics; no single metric alone establishes representation
quality.

## Supersession

The deterministic evaluation and checkpoint requirements remain valid, but the
probability-mass decoder contract is replaced by R-260620-01.