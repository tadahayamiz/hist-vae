# R-260621-01: Raw-space histogram and reconstruction visualization contract

Status: active
Updated: 2026-06-21

## 1. Purpose

HistVAE may construct bins in a transformed coordinate such as
`u = log1p(x)` for numerical resolution, but scientific figures must be able to
show the original measurement coordinate `x`. Plotting bin index or transformed
coordinate as though it were the raw measurement is misleading, especially
when transformed bins have strongly unequal widths in raw space.

The canonical visualization path therefore separates:

```text
histogram construction coordinate
from
figure display coordinate
```

Raw display coordinates are the default for sample-distribution and decoder
reconstruction figures. Axes are linear in raw values by default; `log1p` is
used only to define model bin boundaries, not as the displayed coordinate.

## 2. Edge transformation

For each axis, histogram construction uses equally spaced edges between the
configured transformed lower and upper bounds:

```text
u_i = T(min_value) + i * (T(max_value) - T(min_value)) / bins
```

where `T(x)` is either the identity or `log1p(x)`. Raw-coordinate edges are:

```text
x_i = T^{-1}(u_i)
```

For `log1p`, the inverse is `expm1`. Nonzero lower bounds and per-axis
transforms follow R-260621-02. The implementation exposes both through:

```python
Histogram.get_bin_edges("raw")
Histogram.get_bin_edges("transformed")
```

The last raw edge is pinned to the configured raw maximum to avoid numerical
round-off in the inverse transform.

## 3. Probability mass versus raw-coordinate density

The simplex decoder returns probability mass `p_i` per model bin. When bins are
equal in transformed coordinate but unequal in raw coordinate, plotting `p_i`
as a raw-space curve does not represent density per unit raw measurement.

For a one-dimensional raw-space figure, the canonical ordinate is:

```text
f_i = p_i / (x_{i+1} - x_i)
```

so that:

```text
sum_i f_i * (x_{i+1} - x_i) = 1
```

For two dimensions:

```text
f_ij = p_ij / (delta_x_i * delta_y_j)
```

This conversion changes only the visualization. Training, checkpoint selection,
and forward-KL evaluation continue to use probability mass in model bins.

The plotting API uses:

```text
plot_value_mode = auto
```

which resolves to raw-coordinate density for probability-mass histograms shown
in raw coordinates. Use `plot_value_mode="bin_value"` only to inspect exact
stored model-bin values.

## 4. Reconstruction visualization

`HistVAE.get_reconstruction()` returns the numeric diagnostic artifact:

```text
indices
groups
full-group target
model input
reconstruction
posterior mean
posterior log variance
```

`HistVAE.plot_reconstruction()` renders that artifact with raw coordinates by
default. It supports:

```text
input_mode = full
    deterministic full-group input and z = mu

input_mode = sampled
    one reproducible unaugmented num_points input,
    full-group target, and z = mu
```

The sampled mode uses an explicit random seed and does not apply the legacy
histogram-value augmentation. Decoder-only condition vectors are passed when
the loaded model requires them.

For the probability-mass/forward-KL path, the figure title includes per-sample
forward KL computed in model mass space even when the displayed ordinate is
raw-coordinate density.

## 5. Supported dimensions

The reusable plotting implementation supports:

```text
1D: overlaid input, target, and reconstruction stairs
2D: aligned input, target, reconstruction, and optional difference maps
```

Three-dimensional model training remains supported, but direct 3D histogram
rendering is not part of this plotting contract. A future 3D visualization
should use explicit slices, projections, or interactive volume rendering rather
than silently flattening a dimension.

## 6. Finalized-holdout rule

Raw-space reconstruction figures from the finalized holdout are descriptive
reporting artifacts. They must not be used to select a seed, checkpoint,
architecture, beta, condition mode, display threshold, or new objective.

## 7. Public API

Canonical entry points:

```python
from histvae import (
    histogram_bin_edges,
    plot_hist,
    plot_reconstruction,
    prepare_histogram_for_plot,
)

model.check_data(..., coordinate_space="raw", plot_value_mode="auto")
model.plot_reconstruction(..., coordinate_space="raw", plot_value_mode="auto")
```

The old `histvae.data_handler.plot_hist` and `histvae.core.plot_hist` import
surfaces remain available, but both delegate to the single implementation in
`histvae.visualization`.

## 8. Verification

The implementation is covered by focused tests for:

```text
- inverse log1p edge recovery in raw coordinates
- one-dimensional mass-to-raw-density integration
- two-dimensional mass-to-raw-density integration
- explicit one- and two-dimensional raw-axis rendering
- deterministic full-group reconstruction plotting
- seeded sampled-view reconstruction reproducibility
- decoder-conditioned reconstruction diagnostics
```

Verification result:

```text
focused visualization tests: 8 passed
full pytest including slow tests: 65 passed
```
