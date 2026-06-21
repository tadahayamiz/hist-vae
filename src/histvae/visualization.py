# -*- coding: utf-8 -*-
"""Histogram and reconstruction visualization utilities.

The model may construct histograms in a transformed coordinate system such as
``log1p(x)``.  Plotting is intentionally decoupled from that construction
space: callers can supply raw-coordinate bin edges and, when plotting
probability masses, convert them to a density with respect to the displayed
raw coordinates.
"""
from pathlib import Path
import math

import matplotlib.pyplot as plt
import numpy as np

from .preprocessing import (
    normalize_bins,
    normalize_range_values,
    normalize_value_transforms,
)


PLOT_VALUE_MODES = ("bin_value", "density")

COORDINATE_SPACES = ("raw", "transformed")


def validate_coordinate_space(coordinate_space):
    if coordinate_space not in COORDINATE_SPACES:
        raise ValueError(
            f"Unsupported coordinate_space: {coordinate_space!r}. "
            "Use 'raw' or 'transformed'."
        )
    return coordinate_space


def histogram_bin_edges(
        max_vals, bins, value_transform="none", coordinate_space="raw",
        min_vals=None
        ):
    """Construct histogram bin edges in raw or transformed coordinates.

    This mirrors the histogram construction contract without requiring a
    dataset instance.  ``coordinate_space="raw"`` inverts ``log1p`` binning
    with ``expm1`` so plots use the original measurement scale.
    """
    coordinate_space = validate_coordinate_space(coordinate_space)
    raw_max_vals = np.asarray(max_vals, dtype=np.float64)
    if raw_max_vals.ndim != 1 or len(raw_max_vals) == 0:
        raise ValueError("max_vals must be a non-empty one-dimensional sequence.")
    dimension = len(raw_max_vals)
    raw_min_vals = normalize_range_values(
        min_vals, dimension, "min_vals", default=0.0
    )
    raw_max_vals = normalize_range_values(
        raw_max_vals, dimension, "max_vals"
    )
    if np.any(raw_max_vals <= raw_min_vals):
        raise ValueError("max_vals must exceed min_vals on every axis.")
    bins_per_dim = normalize_bins(bins, dimension)
    transforms = normalize_value_transforms(value_transform, dimension)

    transformed_min_vals = raw_min_vals.copy()
    transformed_max_vals = raw_max_vals.copy()
    for axis, transform in enumerate(transforms):
        if transform == "log1p":
            if raw_min_vals[axis] < 0:
                raise ValueError(
                    "value_transform='log1p' requires non-negative bounds."
                )
            transformed_min_vals[axis] = np.log1p(raw_min_vals[axis])
            transformed_max_vals[axis] = np.log1p(raw_max_vals[axis])
    transformed_edges = [
        np.linspace(
            transformed_min_vals[axis], transformed_max_vals[axis], count + 1
        )
        for axis, count in enumerate(bins_per_dim)
    ]
    if coordinate_space == "transformed":
        return [edge.copy() for edge in transformed_edges]

    raw_edges = []
    for axis, edge in enumerate(transformed_edges):
        if transforms[axis] == "log1p":
            raw_edge = np.expm1(edge)
        else:
            raw_edge = edge.copy()
        raw_edge[0] = raw_min_vals[axis]
        raw_edge[-1] = raw_max_vals[axis]
        raw_edges.append(raw_edge)
    return raw_edges


def validate_plot_value_mode(value_mode):
    if value_mode not in PLOT_VALUE_MODES:
        raise ValueError(
            f"Unsupported plot value mode: {value_mode!r}. "
            "Use 'bin_value' or 'density'."
        )
    return value_mode


def _normalize_bin_edges(bin_edges, hist_shape):
    """Return validated bin edges for one histogram shape."""
    dimension = len(hist_shape)
    if bin_edges is None:
        return [
            np.arange(size + 1, dtype=np.float64)
            for size in hist_shape
        ]

    if dimension == 1:
        candidate = np.asarray(bin_edges)
        if candidate.ndim == 1:
            edge_list = [candidate]
        else:
            edge_list = list(bin_edges)
    else:
        edge_list = list(bin_edges)

    if len(edge_list) != dimension:
        raise ValueError(
            f"bin_edges must contain {dimension} edge arrays; "
            f"got {len(edge_list)}."
        )

    normalized = []
    for axis, (edges, size) in enumerate(zip(edge_list, hist_shape)):
        values = np.asarray(edges, dtype=np.float64)
        if values.ndim != 1:
            raise ValueError(f"bin_edges[{axis}] must be one-dimensional.")
        if len(values) != size + 1:
            raise ValueError(
                f"bin_edges[{axis}] must have length {size + 1}; "
                f"got {len(values)}."
            )
        if not np.all(np.isfinite(values)):
            raise ValueError(f"bin_edges[{axis}] must contain finite values.")
        if np.any(np.diff(values) <= 0):
            raise ValueError(
                f"bin_edges[{axis}] must be strictly increasing."
            )
        normalized.append(values.copy())
    return normalized


def prepare_histogram_for_plot(
        histogram, bin_edges=None, value_mode="bin_value"
        ):
    """Prepare histogram values and edges for coordinate-aware plotting.

    Parameters
    ----------
    histogram : np.ndarray
        One histogram without batch or channel dimensions.
    bin_edges : sequence of np.ndarray, optional
        One edge array per histogram dimension.  A single one-dimensional
        array is accepted for a one-dimensional histogram.
    value_mode : {"bin_value", "density"}
        ``"bin_value"`` displays the stored tensor values directly.
        ``"density"`` divides each bin by its volume in the displayed
        coordinates.  For a probability-mass histogram this preserves unit
        integral in those coordinates, including raw coordinates recovered
        from transformed binning.

    Returns
    -------
    values : np.ndarray
        Values to display.
    edges : list of np.ndarray
        Validated edge arrays.
    """
    value_mode = validate_plot_value_mode(value_mode)
    values = np.asarray(histogram, dtype=np.float64)
    if values.ndim < 1:
        raise ValueError("histogram must have at least one dimension.")
    if not np.all(np.isfinite(values)):
        raise ValueError("histogram must contain only finite values.")

    edges = _normalize_bin_edges(bin_edges, values.shape)
    if value_mode == "bin_value":
        return values.copy(), edges

    volume = np.ones(values.shape, dtype=np.float64)
    for axis, axis_edges in enumerate(edges):
        widths = np.diff(axis_edges)
        reshape = [1] * values.ndim
        reshape[axis] = len(widths)
        volume *= widths.reshape(reshape)
    return values / volume, edges


def _resolve_grid(num_plots, nrow, ncol):
    if num_plots <= 0:
        raise ValueError("At least one histogram is required.")
    if nrow is not None and (not isinstance(nrow, int) or nrow <= 0):
        raise ValueError("nrow must be a positive integer or None.")
    if ncol is not None and (not isinstance(ncol, int) or ncol <= 0):
        raise ValueError("ncol must be a positive integer or None.")

    if nrow is None and ncol is None:
        ncol = min(3, num_plots)
        nrow = math.ceil(num_plots / ncol)
    elif nrow is None:
        nrow = math.ceil(num_plots / ncol)
    elif ncol is None:
        ncol = math.ceil(num_plots / nrow)

    if nrow * ncol < num_plots:
        raise ValueError(
            f"The subplot grid ({nrow} x {ncol}) cannot hold "
            f"{num_plots} plots."
        )
    return nrow, ncol


def _save_figure(fig, output, dpi):
    if not output:
        return
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")


def plot_hist(
        hist_list, output="", show=False, bin_edges=None,
        value_mode="bin_value", close=True, **plot_params
        ):
    """Plot one- or two-dimensional histograms with explicit coordinates.

    ``bin_edges`` controls the displayed coordinates.  Passing raw-coordinate
    edges therefore shows the histogram in raw data space even when the model
    was trained with transformed binning.  For probability masses on unequal
    raw-width bins, use ``value_mode="density"`` so plotted area corresponds to
    probability mass.
    """
    histograms = [np.asarray(hist) for hist in hist_list]
    if not histograms:
        raise ValueError("hist_list must contain at least one histogram.")

    dimensions = {hist.ndim for hist in histograms}
    if not dimensions.issubset({1, 2}):
        raise NotImplementedError(
            "Only 1D and 2D histograms are supported."
        )
    if len(dimensions) != 1:
        raise ValueError("All histograms must have the same dimensionality.")

    default_params = {
        "nrow": None,
        "ncol": None,
        "figsize": None,
        "xlabel": "x",
        "ylabel": "value",
        "colorbar_label": None,
        "title_list": None,
        "cmap": "viridis",
        "aspect": "auto",
        "color": "royalblue",
        "alpha": 0.7,
        "linewidth": 1.5,
        "xscale": "linear",
        "yscale": "linear",
        "dpi": 150,
    }
    params = {**default_params, **plot_params}
    nrow, ncol = _resolve_grid(
        len(histograms), params["nrow"], params["ncol"]
    )
    figsize = params["figsize"] or (5 * ncol, 4 * nrow)
    fig, axes = plt.subplots(nrow, ncol, figsize=figsize, squeeze=False)
    flat_axes = axes.flatten()

    titles = params["title_list"]
    if titles is not None and len(titles) != len(histograms):
        raise ValueError("title_list length must match hist_list length.")

    for index, histogram in enumerate(histograms):
        ax = flat_axes[index]
        values, edges = prepare_histogram_for_plot(
            histogram,
            bin_edges=bin_edges,
            value_mode=value_mode,
        )
        title = titles[index] if titles is not None else (
            f"{histogram.ndim}D Histogram {index + 1}"
        )

        if histogram.ndim == 1:
            ax.stairs(
                values,
                edges[0],
                baseline=0,
                fill=True,
                color=params["color"],
                alpha=params["alpha"],
                linewidth=params["linewidth"],
            )
            ax.set_xlim(edges[0][0], edges[0][-1])
            ax.set_xlabel(params["xlabel"])
            ax.set_ylabel(params["ylabel"])
            ax.set_xscale(params["xscale"])
            ax.set_yscale(params["yscale"])
        else:
            image = ax.pcolormesh(
                edges[0],
                edges[1],
                values.T,
                shading="auto",
                cmap=params["cmap"],
            )
            ax.set_xlim(edges[0][0], edges[0][-1])
            ax.set_ylim(edges[1][0], edges[1][-1])
            ax.set_aspect(params["aspect"])
            ax.set_xlabel(params["xlabel"])
            ax.set_ylabel(params["ylabel"])
            ax.set_xscale(params["xscale"])
            ax.set_yscale(params["yscale"])
            fig.colorbar(
                image,
                ax=ax,
                label=params["colorbar_label"] or "value",
            )
        ax.set_title(title)

    for index in range(len(histograms), len(flat_axes)):
        fig.delaxes(flat_axes[index])

    fig.tight_layout()
    _save_figure(fig, output, params["dpi"])
    if show:
        plt.show()
    active_axes = flat_axes[:len(histograms)]
    if close:
        plt.close(fig)
    return fig, active_axes


def _normalize_reconstruction_batch(values, name):
    array = np.asarray(values, dtype=np.float64)
    if array.ndim in {3, 4} and array.shape[1] == 1:
        array = array[:, 0]
    if array.ndim not in {2, 3}:
        raise ValueError(
            f"{name} must have shape (samples, bins), (samples, x, y), "
            "or the corresponding single-channel shape."
        )
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def plot_reconstruction(
        target, reconstruction, input_hist=None, bin_edges=None,
        value_mode="bin_value", group_labels=None, metric_values=None,
        metric_name=None, output="", show=False, close=True, **plot_params
        ):
    """Plot target, model input, and deterministic reconstruction.

    One-dimensional distributions are overlaid in one raw-coordinate panel per
    sample.  Two-dimensional distributions are shown as aligned input, target,
    reconstruction, and optional absolute-difference panels.
    """
    target = _normalize_reconstruction_batch(target, "target")
    reconstruction = _normalize_reconstruction_batch(
        reconstruction, "reconstruction"
    )
    if target.shape != reconstruction.shape:
        raise ValueError("target and reconstruction shapes must match.")

    input_values = None
    if input_hist is not None:
        input_values = _normalize_reconstruction_batch(input_hist, "input_hist")
        if input_values.shape != target.shape:
            raise ValueError("input_hist shape must match target shape.")

    n_samples = target.shape[0]
    spatial_dim = target.ndim - 1
    labels = (
        [f"sample {index}" for index in range(n_samples)]
        if group_labels is None
        else [str(label) for label in group_labels]
    )
    if len(labels) != n_samples:
        raise ValueError("group_labels length must match the sample count.")

    if metric_values is not None:
        metric_values = np.asarray(metric_values, dtype=np.float64)
        if metric_values.shape != (n_samples,):
            raise ValueError("metric_values must have shape (n_samples,).")
        if not np.all(np.isfinite(metric_values)):
            raise ValueError("metric_values must contain only finite values.")

    default_params = {
        "axis_labels": None,
        "value_label": "value",
        "target_label": "target",
        "input_label": "model input",
        "reconstruction_label": "reconstruction",
        "target_color": "black",
        "input_color": "tab:blue",
        "reconstruction_color": "tab:orange",
        "target_linewidth": 2.0,
        "input_linewidth": 1.3,
        "reconstruction_linewidth": 1.6,
        "xscale": "linear",
        "yscale": "linear",
        "figsize": None,
        "cmap": "viridis",
        "difference_cmap": "magma",
        "show_difference": True,
        "dpi": 150,
        "legend": True,
    }
    params = {**default_params, **plot_params}
    axis_labels = params["axis_labels"] or [
        "x" if axis == 0 else "y"
        for axis in range(spatial_dim)
    ]
    if len(axis_labels) != spatial_dim:
        raise ValueError(
            f"axis_labels must contain {spatial_dim} labels."
        )

    if spatial_dim == 1:
        ncol = min(2, n_samples)
        nrow = math.ceil(n_samples / ncol)
        figsize = params["figsize"] or (7 * ncol, 3.8 * nrow)
        fig, axes = plt.subplots(nrow, ncol, figsize=figsize, squeeze=False)
        flat_axes = axes.flatten()
        for sample_index in range(n_samples):
            ax = flat_axes[sample_index]
            target_values, edges = prepare_histogram_for_plot(
                target[sample_index], bin_edges, value_mode
            )
            reconstruction_values, _ = prepare_histogram_for_plot(
                reconstruction[sample_index], edges, value_mode
            )
            if input_values is not None:
                displayed_input, _ = prepare_histogram_for_plot(
                    input_values[sample_index], edges, value_mode
                )
                ax.stairs(
                    displayed_input,
                    edges[0],
                    label=params["input_label"],
                    color=params["input_color"],
                    linewidth=params["input_linewidth"],
                    linestyle=":",
                )
            ax.stairs(
                target_values,
                edges[0],
                label=params["target_label"],
                color=params["target_color"],
                linewidth=params["target_linewidth"],
            )
            ax.stairs(
                reconstruction_values,
                edges[0],
                label=params["reconstruction_label"],
                color=params["reconstruction_color"],
                linewidth=params["reconstruction_linewidth"],
                linestyle="--",
            )
            title = labels[sample_index]
            if metric_values is not None:
                name = metric_name or "metric"
                title += f" | {name}={metric_values[sample_index]:.4g}"
            ax.set_title(title)
            ax.set_xlim(edges[0][0], edges[0][-1])
            ax.set_xlabel(axis_labels[0])
            ax.set_ylabel(params["value_label"])
            ax.set_xscale(params["xscale"])
            ax.set_yscale(params["yscale"])
            if params["legend"]:
                ax.legend()
        for index in range(n_samples, len(flat_axes)):
            fig.delaxes(flat_axes[index])
        active_axes = flat_axes[:n_samples]
    else:
        panel_names = []
        if input_values is not None:
            panel_names.append("input")
        panel_names.extend(["target", "reconstruction"])
        if params["show_difference"]:
            panel_names.append("absolute difference")
        ncol = len(panel_names)
        figsize = params["figsize"] or (4.2 * ncol, 4.0 * n_samples)
        fig, axes = plt.subplots(
            n_samples, ncol, figsize=figsize, squeeze=False
        )
        for sample_index in range(n_samples):
            target_values, edges = prepare_histogram_for_plot(
                target[sample_index], bin_edges, value_mode
            )
            reconstruction_values, _ = prepare_histogram_for_plot(
                reconstruction[sample_index], edges, value_mode
            )
            panel_values = []
            if input_values is not None:
                displayed_input, _ = prepare_histogram_for_plot(
                    input_values[sample_index], edges, value_mode
                )
                panel_values.append(displayed_input)
            panel_values.extend([target_values, reconstruction_values])
            if params["show_difference"]:
                panel_values.append(
                    np.abs(target_values - reconstruction_values)
                )

            shared_values = panel_values[:3] if input_values is not None else panel_values[:2]
            shared_min = min(float(value.min()) for value in shared_values)
            shared_max = max(float(value.max()) for value in shared_values)
            for panel_index, (name, values) in enumerate(
                    zip(panel_names, panel_values)
                    ):
                ax = axes[sample_index, panel_index]
                is_difference = name == "absolute difference"
                image = ax.pcolormesh(
                    edges[0],
                    edges[1],
                    values.T,
                    shading="auto",
                    cmap=(
                        params["difference_cmap"]
                        if is_difference else params["cmap"]
                    ),
                    vmin=None if is_difference else shared_min,
                    vmax=None if is_difference else shared_max,
                )
                ax.set_xlim(edges[0][0], edges[0][-1])
                ax.set_ylim(edges[1][0], edges[1][-1])
                ax.set_xlabel(axis_labels[0])
                ax.set_ylabel(axis_labels[1])
                ax.set_xscale(params["xscale"])
                ax.set_yscale(params["yscale"])
                title = name
                if panel_index == 0:
                    title = f"{labels[sample_index]} | {name}"
                ax.set_title(title)
                fig.colorbar(image, ax=ax, label=params["value_label"])
        active_axes = axes

    fig.tight_layout()
    _save_figure(fig, output, params["dpi"])
    if show:
        plt.show()
    if close:
        plt.close(fig)
    return fig, active_axes


def plot_scatter(points_list, output="", show=False, close=True, **plot_params):
    """Plot a list of two-dimensional point arrays."""
    default_params = {
        "nrow": None,
        "ncol": None,
        "figsize": None,
        "xlabel": "x",
        "ylabel": "y",
        "title_list": None,
        "color": "royalblue",
        "alpha": 0.7,
        "s": 10,
        "dpi": 150,
    }
    params = {**default_params, **plot_params}
    if not isinstance(points_list, (list, tuple)) or len(points_list) == 0:
        raise ValueError("points_list must contain at least one point array.")
    nrow, ncol = _resolve_grid(
        len(points_list), params["nrow"], params["ncol"]
    )
    figsize = params["figsize"] or (5 * ncol, 4 * nrow)
    titles = params["title_list"]
    if titles is not None and len(titles) != len(points_list):
        raise ValueError("title_list length must match points_list length.")

    fig, axes = plt.subplots(nrow, ncol, figsize=figsize, squeeze=False)
    flat_axes = axes.flatten()
    for index, points in enumerate(points_list):
        values = np.asarray(points)
        if values.ndim != 2 or values.shape[1] != 2:
            raise ValueError(f"Expected shape (N, 2), got {values.shape}.")
        if not np.all(np.isfinite(values)):
            raise ValueError("Point arrays must contain only finite values.")
        ax = flat_axes[index]
        ax.scatter(
            values[:, 0], values[:, 1],
            color=params["color"], alpha=params["alpha"], s=params["s"],
        )
        ax.set_xlabel(params["xlabel"])
        ax.set_ylabel(params["ylabel"])
        ax.set_title(
            titles[index] if titles is not None else f"Scatter {index + 1}"
        )
        ax.grid(True)

    for index in range(len(points_list), len(flat_axes)):
        fig.delaxes(flat_axes[index])

    fig.tight_layout()
    _save_figure(fig, output, params["dpi"])
    if show:
        plt.show()
    active_axes = flat_axes[:len(points_list)]
    if close:
        plt.close(fig)
    return fig, active_axes
