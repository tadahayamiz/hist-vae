# Positioning Lock

Updated: 2026-06-19

HistVAE supports alternative histogram representations; it does not assume that
one representation is universally superior.

- `count` is the backward-compatible representation and may be appropriate when
  group-size/count intensity is meaningful.
- `density` is appropriate when the distributional shape should be emphasized
  independently of group size.
- The current change establishes selectable mechanics and technical validity.
- No model-performance or biological claim is made by this implementation-only
  update.
