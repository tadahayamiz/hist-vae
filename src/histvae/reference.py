"""Generic reference-based scoring for latent or metric representations.

The helpers in this module are label-agnostic.  Callers choose the reference
samples and may use any distance matrix, including Euclidean latent distances
or joint Sinkhorn distances between one-, two-, or three-dimensional
probability histograms.
"""
from __future__ import annotations

from numbers import Integral

import numpy as np


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a positive integer.")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return result


def aggregate_latent_views(mu_views):
    """Aggregate repeated posterior means for each sample.

    Parameters
    ----------
    mu_views : array-like, shape (n_samples, n_views, latent_dim)
        Posterior means obtained from repeated input views.  This function does
        not sample from the VAE posterior and does not modify the model.

    Returns
    -------
    dict
        ``mean`` and per-dimension ``sd`` across views, plus the scalar
        root-mean-square Euclidean distance of the views from their sample mean
        (``rms_distance_to_mean``).
    """
    values = np.asarray(mu_views, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError(
            "mu_views must have shape (n_samples, n_views, latent_dim)."
        )
    if any(size <= 0 for size in values.shape):
        raise ValueError("mu_views dimensions must be non-empty.")
    if not np.all(np.isfinite(values)):
        raise ValueError("mu_views must contain only finite values.")

    mean = values.mean(axis=1)
    centered = values - mean[:, None, :]
    sd = values.std(axis=1, ddof=0)
    rms_distance = np.sqrt(np.mean(np.sum(centered ** 2, axis=2), axis=1))

    return {
        "mean": mean,
        "sd": sd,
        "rms_distance_to_mean": rms_distance,
    }


def reference_knn_from_distances(
        distance_matrix,
        k=5,
        exclude_reference_indices=None,
        ):
    """Compute deterministic mean-kNN scores from query-reference distances.

    Parameters
    ----------
    distance_matrix : array-like, shape (n_query, n_reference)
        Non-negative distances.  Infinite values are allowed and are ignored.
    k : int
        Number of nearest reference samples to average.
    exclude_reference_indices : array-like of int, optional
        One reference-column index per query row.  Use ``-1`` when no reference
        must be excluded.  This supports leave-one-out scoring when a query is
        itself part of the reference set.

    Returns
    -------
    dict
        ``score`` has shape ``(n_query,)``. ``neighbor_indices`` and
        ``neighbor_distances`` have shape ``(n_query, k)`` and are sorted first
        by distance and then by reference index for deterministic ties.
    """
    distances = np.asarray(distance_matrix, dtype=np.float64)
    if distances.ndim != 2:
        raise ValueError(
            "distance_matrix must have shape (n_query, n_reference)."
        )
    if distances.shape[0] == 0 or distances.shape[1] == 0:
        raise ValueError("distance_matrix must be non-empty.")
    if np.any(np.isnan(distances)):
        raise ValueError("distance_matrix must not contain NaN values.")
    if np.any(distances < -1e-12):
        raise ValueError("distance_matrix must contain non-negative distances.")
    distances = np.maximum(distances, 0.0).copy()

    k = _positive_integer(k, "k")
    n_query, n_reference = distances.shape
    if k > n_reference:
        raise ValueError("k must not exceed the number of reference samples.")

    if exclude_reference_indices is None:
        exclusions = np.full(n_query, -1, dtype=np.int64)
    else:
        exclusions = np.asarray(exclude_reference_indices)
        if exclusions.shape != (n_query,):
            raise ValueError(
                "exclude_reference_indices must have shape (n_query,)."
            )
        if not np.issubdtype(exclusions.dtype, np.integer):
            raise ValueError("exclude_reference_indices must contain integers.")
        exclusions = exclusions.astype(np.int64, copy=False)
        if np.any((exclusions < -1) | (exclusions >= n_reference)):
            raise ValueError(
                "exclude_reference_indices entries must be -1 or valid "
                "reference-column indices."
            )

    rows_with_exclusion = np.flatnonzero(exclusions >= 0)
    distances[rows_with_exclusion, exclusions[rows_with_exclusion]] = np.inf

    neighbor_indices = np.empty((n_query, k), dtype=np.int64)
    neighbor_distances = np.empty((n_query, k), dtype=np.float64)

    for row_index in range(n_query):
        finite_indices = np.flatnonzero(np.isfinite(distances[row_index]))
        if len(finite_indices) < k:
            raise ValueError(
                f"Query row {row_index} has fewer than k finite reference "
                "distances after exclusions."
            )
        order = np.lexsort((
            finite_indices,
            distances[row_index, finite_indices],
        ))
        selected = finite_indices[order[:k]]
        neighbor_indices[row_index] = selected
        neighbor_distances[row_index] = distances[row_index, selected]

    return {
        "score": neighbor_distances.mean(axis=1),
        "neighbor_indices": neighbor_indices,
        "neighbor_distances": neighbor_distances,
    }


def euclidean_reference_knn(
        query,
        reference,
        k=5,
        exclude_reference_indices=None,
        return_distance_matrix=False,
        ):
    """Score query vectors against reference vectors by mean Euclidean kNN."""
    query_values = np.asarray(query, dtype=np.float64)
    reference_values = np.asarray(reference, dtype=np.float64)

    if query_values.ndim != 2 or reference_values.ndim != 2:
        raise ValueError("query and reference must both be two-dimensional.")
    if query_values.shape[0] == 0 or reference_values.shape[0] == 0:
        raise ValueError("query and reference must be non-empty.")
    if query_values.shape[1] != reference_values.shape[1]:
        raise ValueError("query and reference feature dimensions must match.")
    if not np.all(np.isfinite(query_values)):
        raise ValueError("query must contain only finite values.")
    if not np.all(np.isfinite(reference_values)):
        raise ValueError("reference must contain only finite values.")

    distances = np.linalg.norm(
        query_values[:, None, :] - reference_values[None, :, :],
        axis=2,
    )
    result = reference_knn_from_distances(
        distances,
        k=k,
        exclude_reference_indices=exclude_reference_indices,
    )
    if return_distance_matrix:
        result["distance_matrix"] = distances
    return result


def empirical_reference_percentile(
        values,
        reference_indices,
        leave_one_out=True,
        ):
    """Rank scores against selected reference rows with optional self removal.

    Larger input values always receive larger empirical percentiles.  A
    ``+1`` convention is used so non-reference queries never receive exactly
    zero.  When ``leave_one_out`` is true, each reference row is ranked against
    the other reference rows.
    """
    scores = np.asarray(values, dtype=np.float64)
    if scores.ndim != 1 or len(scores) == 0:
        raise ValueError("values must be a non-empty one-dimensional array.")
    if not np.all(np.isfinite(scores)):
        raise ValueError("values must contain only finite entries.")

    indices = np.asarray(reference_indices)
    if indices.ndim != 1 or len(indices) == 0:
        raise ValueError(
            "reference_indices must be a non-empty one-dimensional array."
        )
    if not np.issubdtype(indices.dtype, np.integer):
        raise ValueError("reference_indices must contain integers.")
    indices = indices.astype(np.int64, copy=False)
    if np.any((indices < 0) | (indices >= len(scores))):
        raise IndexError("reference_indices contains an out-of-range index.")
    if len(np.unique(indices)) != len(indices):
        raise ValueError("reference_indices must not contain duplicates.")
    if leave_one_out and len(indices) < 2:
        raise ValueError(
            "At least two reference rows are required for leave-one-out ranks."
        )

    reference_scores = scores[indices]
    sorted_reference = np.sort(reference_scores)
    percentiles = (
        1.0 + np.searchsorted(sorted_reference, scores, side="right")
    ) / (len(reference_scores) + 1.0)

    if leave_one_out:
        for reference_position, sample_index in enumerate(indices):
            other_scores = np.delete(reference_scores, reference_position)
            percentiles[sample_index] = (
                1.0
                + np.searchsorted(
                    np.sort(other_scores),
                    scores[sample_index],
                    side="right",
                )
            ) / (len(other_scores) + 1.0)

    return np.clip(percentiles, 0.0, 1.0)
