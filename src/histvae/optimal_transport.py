"""Joint observation-space optimal transport for probability histograms.

The public contract uses one debiased entropic Sinkhorn divergence on the
flattened *joint* histogram in one, two, or three dimensions. Axis-wise
marginal distances are intentionally not used because they discard
cross-dimensional dependence.
"""
from __future__ import annotations

import importlib.util
import math
from numbers import Integral, Real
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
from geomloss import SamplesLoss


OT_LOSSES = ("none", "sinkhorn")
OT_BACKENDS = ("tensorized", "online", "multiscale")
OT_P_VALUES = (1, 2)
_TENSORIZED_MAX_PAIRWISE_BYTES = 256 * 1024 * 1024


def _finite_float(value, name, *, lower=None, upper=None, lower_open=False,
                  upper_open=False):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite numeric value.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite numeric value.")
    if lower is not None:
        invalid = result <= lower if lower_open else result < lower
        if invalid:
            relation = ">" if lower_open else ">="
            raise ValueError(f"{name} must be {relation} {lower}.")
    if upper is not None:
        invalid = result >= upper if upper_open else result > upper
        if invalid:
            relation = "<" if upper_open else "<="
            raise ValueError(f"{name} must be {relation} {upper}.")
    return result


def validate_ot_config(config):
    """Normalize and strictly validate observation-space OT settings in-place."""
    ot_loss = config.get("ot_loss", "none")
    if ot_loss not in OT_LOSSES:
        raise ValueError(
            f"Unsupported ot_loss: {ot_loss!r}. Use 'none' or 'sinkhorn'."
        )

    ot_weight = _finite_float(
        config.get("ot_weight", 0.0), "ot_weight", lower=0.0
    )
    p_value = config.get("ot_p", 1)
    if isinstance(p_value, bool) or not isinstance(p_value, Integral):
        raise ValueError("ot_p must be the integer 1 or 2.")
    ot_p = int(p_value)
    if ot_p not in OT_P_VALUES:
        raise ValueError("ot_p must be 1 or 2.")

    ot_blur = _finite_float(
        config.get("ot_blur", 0.05), "ot_blur", lower=0.0,
        lower_open=True,
    )
    ot_scaling = _finite_float(
        config.get("ot_scaling", 0.8), "ot_scaling", lower=0.0,
        upper=1.0, lower_open=True, upper_open=True,
    )
    ot_backend = config.get("ot_backend", "tensorized")
    if ot_backend not in OT_BACKENDS:
        raise ValueError(
            f"Unsupported ot_backend: {ot_backend!r}. "
            "Use 'tensorized', 'online', or 'multiscale'."
        )
    ot_mass_epsilon = _finite_float(
        config.get("ot_mass_epsilon", 0.0),
        "ot_mass_epsilon",
        lower=0.0,
    )

    if ot_loss == "none":
        if ot_weight != 0.0:
            raise ValueError("ot_weight must be 0 when ot_loss='none'.")
    else:
        if ot_weight <= 0.0:
            raise ValueError("ot_loss='sinkhorn' requires ot_weight > 0.")
        if config.get("histogram_mode") != "probability_mass":
            raise ValueError(
                "ot_loss='sinkhorn' requires histogram_mode='probability_mass'."
            )
        if config.get("decoder_output_mode") != "simplex_softmax":
            raise ValueError(
                "ot_loss='sinkhorn' requires "
                "decoder_output_mode='simplex_softmax'."
            )
        if config.get("reconstruction_loss") != "forward_kl":
            raise ValueError(
                "ot_loss='sinkhorn' requires "
                "reconstruction_loss='forward_kl'."
            )

    config["ot_loss"] = ot_loss
    config["ot_weight"] = ot_weight
    config["ot_p"] = ot_p
    config["ot_blur"] = ot_blur
    config["ot_scaling"] = ot_scaling
    config["ot_backend"] = ot_backend
    config["ot_mass_epsilon"] = ot_mass_epsilon
    return config


def build_joint_bin_support(bin_edges: Sequence[np.ndarray]) -> np.ndarray:
    """Return normalized joint bin centers for one, two, or three dimensions.

    Each axis is globally scaled from its first to last fitted histogram edge
    into ``[0, 1]``. The full coordinate vector is then divided by
    ``sqrt(d)`` so the enclosing joint-space diameter is at most one. This is
    a metric-only transformation and does not modify histogram values.
    """
    if not isinstance(bin_edges, Sequence) or isinstance(bin_edges, (str, bytes)):
        raise TypeError("bin_edges must be a sequence of one to three arrays.")
    dimension = len(bin_edges)
    if dimension not in (1, 2, 3):
        raise ValueError("Joint Sinkhorn support is implemented for 1D, 2D, or 3D.")

    normalized_centers = []
    for axis, values in enumerate(bin_edges):
        edges = np.asarray(values, dtype=np.float64)
        if edges.ndim != 1 or edges.size < 2:
            raise ValueError(
                f"bin_edges[{axis}] must be a one-dimensional array with "
                "at least two values."
            )
        if not np.all(np.isfinite(edges)):
            raise ValueError(f"bin_edges[{axis}] must contain finite values.")
        differences = np.diff(edges)
        if np.any(differences <= 0):
            raise ValueError(f"bin_edges[{axis}] must be strictly increasing.")
        span = float(edges[-1] - edges[0])
        centers = 0.5 * (edges[:-1] + edges[1:])
        normalized_centers.append((centers - edges[0]) / span)

    grids = np.meshgrid(*normalized_centers, indexing="ij")
    support = np.stack([grid.reshape(-1) for grid in grids], axis=1)
    support /= math.sqrt(dimension)
    return support.astype(np.float32, copy=False)


class JointSinkhornDivergence(nn.Module):
    """Debiased entropic Sinkhorn divergence for joint probability histograms."""

    def __init__(
            self,
            support,
            p=1,
            blur=0.05,
            scaling=0.8,
            backend="tensorized",
            mass_epsilon=0.0,
            ):
        super().__init__()
        config = {
            "ot_loss": "sinkhorn",
            "ot_weight": 1.0,
            "ot_p": p,
            "ot_blur": blur,
            "ot_scaling": scaling,
            "ot_backend": backend,
            "ot_mass_epsilon": mass_epsilon,
            "histogram_mode": "probability_mass",
            "decoder_output_mode": "simplex_softmax",
            "reconstruction_loss": "forward_kl",
        }
        validate_ot_config(config)

        support_tensor = torch.as_tensor(support, dtype=torch.float32)
        if support_tensor.ndim != 2:
            raise ValueError("support must have shape (n_joint_bins, dimension).")
        if support_tensor.shape[0] < 2:
            raise ValueError("support must contain at least two joint bins.")
        if support_tensor.shape[1] not in (1, 2, 3):
            raise ValueError("support dimension must be 1, 2, or 3.")
        if not torch.isfinite(support_tensor).all():
            raise ValueError("support must contain only finite values.")

        if config["ot_backend"] in ("online", "multiscale"):
            if importlib.util.find_spec("pykeops") is None:
                raise ImportError(
                    f"ot_backend={config['ot_backend']!r} requires PyKeOps. "
                    "Install HistVAE with the 'ot-scalable' extra."
                )

        self.p = config["ot_p"]
        self.blur = config["ot_blur"]
        self.scaling = config["ot_scaling"]
        self.backend = config["ot_backend"]
        self.mass_epsilon = config["ot_mass_epsilon"]
        self.register_buffer("support", support_tensor.contiguous())
        self._loss = SamplesLoss(
            loss="sinkhorn",
            p=self.p,
            blur=self.blur,
            reach=None,
            diameter=1.0,
            scaling=self.scaling,
            debias=True,
            backend=self.backend,
        )

    @property
    def support_size(self):
        return int(self.support.shape[0])

    @property
    def dimension(self):
        return int(self.support.shape[1])

    def _validate_and_prepare_weights(self, histogram, name):
        if not torch.is_tensor(histogram):
            raise TypeError(f"{name} must be a torch.Tensor.")
        if histogram.ndim < 2:
            raise ValueError(f"{name} must include batch and histogram dimensions.")
        if not histogram.is_floating_point():
            raise ValueError(f"{name} must use a floating-point dtype.")
        weights = histogram.flatten(start_dim=1)
        if weights.shape[1] != self.support_size:
            raise ValueError(
                f"{name} has {weights.shape[1]} flattened bins, but the OT "
                f"support contains {self.support_size}."
            )
        if not torch.isfinite(weights).all():
            raise ValueError(f"{name} must contain only finite values.")
        tolerance = 1e-5
        if torch.any(weights < -tolerance):
            raise ValueError(f"{name} must be non-negative probability mass.")
        weights = weights.clamp_min(0)
        mass = weights.sum(dim=1, keepdim=True)
        if not torch.allclose(
                mass,
                torch.ones_like(mass),
                rtol=tolerance,
                atol=tolerance,
                ):
            raise ValueError(f"Each {name} histogram must sum to one.")
        if self.mass_epsilon > 0:
            weights = weights + self.mass_epsilon
            weights = weights / weights.sum(dim=1, keepdim=True)
        return weights

    def _check_tensorized_memory(self, batch_size, dtype):
        element_size = torch.empty((), dtype=dtype).element_size()
        estimated = batch_size * self.support_size ** 2 * element_size
        if estimated > _TENSORIZED_MAX_PAIRWISE_BYTES:
            gib = estimated / 1024 ** 3
            raise RuntimeError(
                "ot_backend='tensorized' would materialize at least "
                f"{gib:.2f} GiB for one pairwise tensor. Select "
                "ot_backend='online' with the 'ot-scalable' extra, reduce "
                "the batch size, or reduce the joint bin count."
            )

    def _per_sample(self, target_weights, prediction_weights):
        batch_size = target_weights.shape[0]
        support = self.support.to(
            device=target_weights.device,
            dtype=target_weights.dtype,
        )

        if self.backend == "tensorized":
            self._check_tensorized_memory(batch_size, target_weights.dtype)

        if self.backend == "multiscale":
            values = []
            for index in range(batch_size):
                values.append(
                    self._loss(
                        target_weights[index],
                        support,
                        prediction_weights[index],
                        support,
                    )
                )
            result = torch.stack(values)
        else:
            batched_support = support.unsqueeze(0).expand(
                batch_size, -1, -1
            ).contiguous()
            result = self._loss(
                target_weights,
                batched_support,
                prediction_weights,
                batched_support,
            )
            if result.ndim == 0:
                result = result.unsqueeze(0)

        if not torch.isfinite(result).all():
            raise FloatingPointError("Sinkhorn divergence produced non-finite values.")
        if torch.any(result < -1e-6):
            minimum = float(result.min().detach().cpu())
            raise FloatingPointError(
                "Sinkhorn divergence produced a materially negative value: "
                f"{minimum}."
            )
        return result.clamp_min(0)

    def pairwise(self, query, reference, pair_batch_size=64):
        """Return all query-reference Sinkhorn divergences.

        Parameters
        ----------
        query, reference : torch.Tensor
            Probability-mass histograms with leading sample dimensions. The
            remaining histogram dimensions may describe one-, two-, or
            three-dimensional joint histograms, but their flattened bin count
            must match this instance's support.
        pair_batch_size : int
            Maximum number of query-reference pairs evaluated in one call to
            the underlying Sinkhorn loss. This limits temporary memory without
            changing the returned distance matrix.

        Returns
        -------
        torch.Tensor
            Matrix with shape ``(n_query, n_reference)``.
        """
        if isinstance(pair_batch_size, bool) or not isinstance(
                pair_batch_size, Integral
                ):
            raise ValueError("pair_batch_size must be a positive integer.")
        pair_batch_size = int(pair_batch_size)
        if pair_batch_size <= 0:
            raise ValueError("pair_batch_size must be a positive integer.")

        query_weights = self._validate_and_prepare_weights(query, "query")
        reference_weights = self._validate_and_prepare_weights(
            reference, "reference"
        )
        if query_weights.device != reference_weights.device:
            raise ValueError("query and reference must be on the same device.")
        if query_weights.dtype != reference_weights.dtype:
            raise ValueError("query and reference must use the same dtype.")

        n_query = int(query_weights.shape[0])
        n_reference = int(reference_weights.shape[0])
        if n_query == 0 or n_reference == 0:
            raise ValueError("query and reference batches must be non-empty.")

        flat_result = torch.empty(
            n_query * n_reference,
            dtype=query_weights.dtype,
            device=query_weights.device,
        )
        for start in range(0, flat_result.numel(), pair_batch_size):
            stop = min(start + pair_batch_size, flat_result.numel())
            flat_indices = torch.arange(
                start, stop, device=query_weights.device
            )
            query_indices = torch.div(
                flat_indices, n_reference, rounding_mode="floor"
            )
            reference_indices = torch.remainder(flat_indices, n_reference)
            flat_result[start:stop] = self._per_sample(
                query_weights[query_indices],
                reference_weights[reference_indices],
            )

        return flat_result.reshape(n_query, n_reference)

    def forward(self, target, prediction, reduction="mean"):
        """Compute joint Sinkhorn divergence for one batch of histograms."""
        if target.shape != prediction.shape:
            raise ValueError("target and prediction histogram shapes must match.")
        target_weights = self._validate_and_prepare_weights(target, "target")
        prediction_weights = self._validate_and_prepare_weights(
            prediction, "prediction"
        )
        values = self._per_sample(target_weights, prediction_weights)
        if reduction == "none":
            return values
        if reduction == "mean":
            return values.mean()
        if reduction == "sum":
            return values.sum()
        raise ValueError("reduction must be 'none', 'mean', or 'sum'.")
