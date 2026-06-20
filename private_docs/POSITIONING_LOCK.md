# Positioning Lock

Updated: 2026-06-20

HistVAE supports selectable histogram representations; it does not assume one
representation is universally superior.

- `count` is the backward-compatible representation when count intensity is
  meaningful.
- `density` remains available for unit-integral density followed by the legacy
  log/max normalization.
- `probability_mass` is a bounded group-size-invariant representation aligned
  with the current sigmoid decoder.
- Deterministic validation and canonical best-checkpoint handling are
  reliability requirements, not evidence of biological validity.
- Current attached-data runs establish technical execution only. Model quality
  and biological claims require controlled pilot and downstream evaluation.
