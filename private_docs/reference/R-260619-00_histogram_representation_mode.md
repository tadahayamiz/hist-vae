# R-260619-00: Histogram representation mode

## Decision

Expose a strict `histogram_mode` with exactly two values:

- `count`: NumPy histogram counts; existing behavior and default
- `density`: NumPy probability density; unit integral over configured edges

The mode is accepted in configuration and as a runtime override to
`HistVAE.prep_data`.

## Rationale

The dataset samples a fixed number of points for each training view but derives
its normalization reference from each full group. In count mode, that reference
can retain group-size intensity. Density mode removes the multiplicative effect
of duplicating the same empirical distribution before the existing nonlinear
normalization.

## Compatibility

Existing callers that do not specify `histogram_mode` continue to use `count`.
This is an intentional supported mode, not a temporary compatibility shim.
Invalid mode names fail explicitly.

## Constraints

Histogram edges remain fixed from zero to each configured `max_vals` entry.
Values outside those edges are excluded by NumPy; callers must choose ranges
that cover the intended training domain.

## Non-goals

- Automatic range estimation
- Automatic bin selection
- CLI redesign
- Claims about downstream accuracy
