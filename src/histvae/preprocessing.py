# -*- coding: utf-8 -*-
"""Train-fitted histogram preprocessing for grouped point data.

The preprocessor separates data-dependent range fitting from dataset and model
construction.  It is fitted on training events only, stores the resolved raw
bounds and coordinate transforms, and can then be reused unchanged for
validation, holdout, plotting, and checkpoint provenance.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml


HISTOGRAM_MODES = ("count", "density", "probability_mass")
VALUE_TRANSFORMS = ("none", "log1p")
BOUND_MODES = ("fixed", "quantile")
QUANTILE_WEIGHTINGS = ("event", "group_equal")
PREPROCESSOR_TAIL_POLICIES = ("clip", "error")
PREPROCESSOR_SCHEMA_VERSION = 1


def validate_histogram_mode(histogram_mode: str) -> str:
    if histogram_mode not in HISTOGRAM_MODES:
        raise ValueError(
            f"Unsupported histogram_mode: {histogram_mode!r}. "
            "Use 'count', 'density', or 'probability_mass'."
        )
    return histogram_mode


def validate_value_transform(value_transform: str) -> str:
    if value_transform not in VALUE_TRANSFORMS:
        raise ValueError(
            f"Unsupported value_transform: {value_transform!r}. "
            "Use 'none' or 'log1p'."
        )
    return value_transform


def normalize_value_transforms(
        value_transform: str | Sequence[str], dimension: int
        ) -> tuple[str, ...]:
    """Return one validated transform name per axis."""
    if isinstance(value_transform, str):
        transform = validate_value_transform(value_transform)
        return (transform,) * dimension

    try:
        transforms = tuple(value_transform)
    except TypeError as exc:
        raise ValueError(
            "value_transform must be a string or one transform per dimension."
        ) from exc
    if len(transforms) != dimension:
        raise ValueError(
            f"value_transform must contain {dimension} values; "
            f"got {len(transforms)}."
        )
    return tuple(validate_value_transform(value) for value in transforms)


def compress_axis_setting(values: Sequence[Any]) -> Any:
    """Return a scalar for a shared setting, otherwise a plain list."""
    values = list(values)
    if values and all(value == values[0] for value in values[1:]):
        return values[0]
    return values


def normalize_bins(bins: int | Sequence[int], dimension: int) -> tuple[int, ...]:
    """Return one positive integer bin count per axis."""
    if isinstance(bins, (int, np.integer)):
        if int(bins) <= 0:
            raise ValueError("bins must be positive.")
        return (int(bins),) * dimension

    try:
        values = tuple(bins)
    except TypeError as exc:
        raise ValueError("bins must be an integer or one integer per dimension.") from exc
    if len(values) != dimension:
        raise ValueError(
            f"bins must contain {dimension} values; got {len(values)}."
        )
    if any(
            not isinstance(value, (int, np.integer)) or int(value) <= 0
            for value in values
            ):
        raise ValueError("bins must contain positive integers.")
    return tuple(int(value) for value in values)


def normalize_range_values(
        values: Sequence[float] | np.ndarray | None,
        dimension: int,
        name: str,
        default: float | None = None,
        ) -> np.ndarray:
    """Return one finite raw-coordinate bound per axis."""
    if values is None:
        if default is None:
            raise ValueError(f"{name} must be provided.")
        result = np.full(dimension, float(default), dtype=np.float64)
    else:
        result = np.asarray(values, dtype=np.float64)
        if result.shape != (dimension,):
            raise ValueError(f"{name} must contain one value per dimension.")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain finite values.")
    return result


def _validate_bound_mode(mode: str, name: str) -> str:
    if mode not in BOUND_MODES:
        raise ValueError(
            f"Unsupported {name}: {mode!r}. Use 'fixed' or 'quantile'."
        )
    return mode


def _validate_quantile_weighting(weighting: str) -> str:
    if weighting not in QUANTILE_WEIGHTINGS:
        raise ValueError(
            f"Unsupported quantile_weighting: {weighting!r}. "
            "Use 'event' or 'group_equal'."
        )
    return weighting


def _validate_tail_policy(tail_policy: str) -> str:
    if tail_policy not in PREPROCESSOR_TAIL_POLICIES:
        raise ValueError(
            f"Unsupported tail_policy: {tail_policy!r}. "
            "Use 'clip' or 'error'."
        )
    return tail_policy


@dataclass(frozen=True)
class AxisPreprocessingSpec:
    """Preprocessing contract for one raw measurement axis.

    Bounds can be fixed independently or fitted as empirical quantiles.  This
    permits the common contract of a fixed physical lower bound and a
    train-fitted upper percentile without introducing a silent automatic rule.
    """

    name: str
    transform: str = "none"
    lower_mode: str = "fixed"
    lower_value: float | None = 0.0
    lower_quantile: float | None = None
    upper_mode: str = "fixed"
    upper_value: float | None = None
    upper_quantile: float | None = None
    quantile_weighting: str = "event"

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Axis name must be a non-empty string.")
        validate_value_transform(self.transform)
        _validate_bound_mode(self.lower_mode, "lower_mode")
        _validate_bound_mode(self.upper_mode, "upper_mode")
        _validate_quantile_weighting(self.quantile_weighting)
        self._validate_bound(
            mode=self.lower_mode,
            value=self.lower_value,
            quantile=self.lower_quantile,
            name="lower",
        )
        self._validate_bound(
            mode=self.upper_mode,
            value=self.upper_value,
            quantile=self.upper_quantile,
            name="upper",
        )
        if (
                self.lower_mode == "quantile"
                and self.upper_mode == "quantile"
                and self.lower_quantile >= self.upper_quantile
                ):
            raise ValueError("lower_quantile must be smaller than upper_quantile.")
        if self.transform == "log1p":
            for mode, value, name in (
                    (self.lower_mode, self.lower_value, "lower_value"),
                    (self.upper_mode, self.upper_value, "upper_value"),
                    ):
                if mode == "fixed" and value < 0:
                    raise ValueError(
                        f"{name} must be non-negative for transform='log1p'."
                    )

    @staticmethod
    def _validate_bound(mode, value, quantile, name) -> None:
        if mode == "fixed":
            if value is None or not np.isfinite(value):
                raise ValueError(
                    f"{name}_value must be finite when {name}_mode='fixed'."
                )
            if quantile is not None:
                raise ValueError(
                    f"{name}_quantile must be omitted when {name}_mode='fixed'."
                )
            return
        if value is not None:
            raise ValueError(
                f"{name}_value must be omitted when {name}_mode='quantile'."
            )
        if quantile is None or not np.isfinite(quantile):
            raise ValueError(
                f"{name}_quantile must be finite when {name}_mode='quantile'."
            )
        if not 0.0 <= float(quantile) <= 1.0:
            raise ValueError(f"{name}_quantile must be in [0, 1].")

    @classmethod
    def from_mapping(
            cls, mapping: Mapping[str, Any], axis_index: int | None = None
            ) -> "AxisPreprocessingSpec":
        if not isinstance(mapping, Mapping):
            raise TypeError("Each axis specification must be a mapping or dataclass.")
        allowed = set(cls.__dataclass_fields__)
        unknown = set(mapping).difference(allowed)
        if unknown:
            raise ValueError(
                f"Unsupported axis specification keys: {sorted(unknown)}."
            )
        values = dict(mapping)
        if "name" not in values and axis_index is not None:
            values["name"] = f"axis_{axis_index}"
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class HistogramPreprocessor:
    """Fit and persist histogram geometry on training data only.

    Parameters
    ----------
    axis_specs:
        One :class:`AxisPreprocessingSpec` or mapping per point dimension.
    bins:
        One common bin count or one count per axis.
    histogram_mode:
        Histogram representation passed to :class:`histvae.data_handler.Histogram`.
    tail_policy:
        ``"clip"`` preserves tail mass in edge bins. ``"error"`` rejects any
        application data outside the fitted range. Dropping and silently
        renormalizing outliers is intentionally not part of this preprocessor.
    """

    def __init__(
            self,
            axis_specs: Sequence[AxisPreprocessingSpec | Mapping[str, Any]],
            bins: int | Sequence[int] = 64,
            histogram_mode: str = "probability_mass",
            tail_policy: str = "clip",
            ):
        if not isinstance(axis_specs, Sequence) or isinstance(axis_specs, (str, bytes)):
            raise TypeError("axis_specs must be a non-empty sequence.")
        if len(axis_specs) == 0:
            raise ValueError("axis_specs must contain at least one axis.")
        normalized_specs = []
        for index, spec in enumerate(axis_specs):
            if isinstance(spec, AxisPreprocessingSpec):
                normalized_specs.append(spec)
            else:
                normalized_specs.append(
                    AxisPreprocessingSpec.from_mapping(spec, axis_index=index)
                )
        names = [spec.name for spec in normalized_specs]
        if len(set(names)) != len(names):
            raise ValueError("Axis names must be unique.")

        self.axis_specs = tuple(normalized_specs)
        self.dimension = len(self.axis_specs)
        self.bins_per_dim = normalize_bins(bins, self.dimension)
        self.histogram_mode = validate_histogram_mode(histogram_mode)
        self.tail_policy = _validate_tail_policy(tail_policy)

        self.raw_min_vals: np.ndarray | None = None
        self.raw_max_vals: np.ndarray | None = None
        self.fit_summary: dict[str, Any] | None = None
        self.fit_data_sha256: str | None = None

    @property
    def is_fitted(self) -> bool:
        return self.raw_min_vals is not None and self.raw_max_vals is not None

    @property
    def axis_names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self.axis_specs)

    @property
    def value_transforms(self) -> tuple[str, ...]:
        return tuple(spec.transform for spec in self.axis_specs)

    @property
    def bins(self) -> int | list[int]:
        return compress_axis_setting(self.bins_per_dim)

    @property
    def value_transform(self) -> str | list[str]:
        return compress_axis_setting(self.value_transforms)

    @property
    def min_vals(self) -> list[float]:
        self._require_fitted()
        return self.raw_min_vals.tolist()

    @property
    def max_vals(self) -> list[float]:
        self._require_fitted()
        return self.raw_max_vals.tolist()

    @property
    def state_sha256(self) -> str:
        state = self.state_dict(include_state_hash=False)
        encoded = json.dumps(
            state, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def fit(
            self,
            data: np.ndarray,
            group: np.ndarray | Sequence[Any] | None = None,
            ) -> "HistogramPreprocessor":
        """Fit raw bounds from training events and freeze the contract."""
        self.raw_min_vals = None
        self.raw_max_vals = None
        self.fit_summary = None
        self.fit_data_sha256 = None
        values = self._validate_data(data, name="data")
        requires_group = any(
            spec.quantile_weighting == "group_equal"
            and (spec.lower_mode == "quantile" or spec.upper_mode == "quantile")
            for spec in self.axis_specs
        )
        group_values = self._validate_group(
            group, n_rows=len(values), required=requires_group
        )

        min_vals = []
        max_vals = []
        for axis, spec in enumerate(self.axis_specs):
            axis_values = values[:, axis]
            weights = None
            if (
                    spec.quantile_weighting == "group_equal"
                    and (spec.lower_mode == "quantile" or spec.upper_mode == "quantile")
                    ):
                weights = self._group_equal_weights(group_values)

            lower = self._resolve_bound(
                axis_values,
                mode=spec.lower_mode,
                value=spec.lower_value,
                quantile=spec.lower_quantile,
                weighting=spec.quantile_weighting,
                weights=weights,
            )
            upper = self._resolve_bound(
                axis_values,
                mode=spec.upper_mode,
                value=spec.upper_value,
                quantile=spec.upper_quantile,
                weighting=spec.quantile_weighting,
                weights=weights,
            )
            if not lower < upper:
                raise ValueError(
                    f"Resolved bounds for axis {spec.name!r} are invalid: "
                    f"lower={lower}, upper={upper}."
                )
            if spec.transform == "log1p" and lower < 0:
                raise ValueError(
                    f"Resolved lower bound for axis {spec.name!r} is negative, "
                    "but transform='log1p' requires non-negative values."
                )
            min_vals.append(lower)
            max_vals.append(upper)

        self.raw_min_vals = np.asarray(min_vals, dtype=np.float64)
        self.raw_max_vals = np.asarray(max_vals, dtype=np.float64)
        self.fit_data_sha256 = self._hash_fit_data(values, group_values)
        diagnostics = self.diagnose(values)
        if (
                self.tail_policy == "error"
                and diagnostics["fraction_any_outside"] > 0
                ):
            self.raw_min_vals = None
            self.raw_max_vals = None
            self.fit_data_sha256 = None
            raise ValueError(
                "tail_policy='error' produced a fitted range that excludes "
                "training observations. Use wider bounds or tail_policy='clip'."
            )
        diagnostics["n_groups"] = (
            None if group_values is None else int(len(pd.unique(group_values)))
        )
        diagnostics["fit_data_sha256"] = self.fit_data_sha256
        self.fit_summary = diagnostics
        return self

    def require_fitted(self) -> "HistogramPreprocessor":
        """Raise unless fitted and return ``self`` for explicit API checks."""
        self._require_fitted()
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Apply fitted tail handling and coordinate transforms."""
        self._require_fitted()
        values = self._validate_data(data, name="data").copy()
        outside = np.any(
            (values < self.raw_min_vals) | (values > self.raw_max_vals), axis=1
        )
        if self.tail_policy == "error" and np.any(outside):
            raise ValueError(
                f"Found {int(outside.sum())} observations outside the fitted "
                "histogram range."
            )
        if self.tail_policy == "clip":
            values = np.clip(values, self.raw_min_vals, self.raw_max_vals)

        for axis, transform in enumerate(self.value_transforms):
            if transform == "log1p":
                values[:, axis] = np.log1p(values[:, axis])
        return values

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Invert only the coordinate transform; no tail information is restored."""
        self._require_fitted()
        values = self._validate_data(data, name="data").copy()
        for axis, transform in enumerate(self.value_transforms):
            if transform == "log1p":
                values[:, axis] = np.expm1(values[:, axis])
        return values

    def diagnose(self, data: np.ndarray) -> dict[str, Any]:
        """Report raw-coordinate underflow and overflow without modifying data."""
        self._require_fitted()
        values = self._validate_data(data, name="data")
        below = values < self.raw_min_vals
        above = values > self.raw_max_vals
        outside = below | above
        axes = []
        for axis, spec in enumerate(self.axis_specs):
            axes.append({
                "name": spec.name,
                "raw_min": float(self.raw_min_vals[axis]),
                "raw_max": float(self.raw_max_vals[axis]),
                "below_count": int(below[:, axis].sum()),
                "above_count": int(above[:, axis].sum()),
                "below_fraction": float(below[:, axis].mean()),
                "above_fraction": float(above[:, axis].mean()),
            })
        return {
            "n_rows": int(len(values)),
            "fraction_any_below": float(np.any(below, axis=1).mean()),
            "fraction_any_above": float(np.any(above, axis=1).mean()),
            "fraction_any_outside": float(np.any(outside, axis=1).mean()),
            "tail_policy": self.tail_policy,
            "axes": axes,
        }

    def get_bin_edges(self, coordinate_space: str = "raw") -> list[np.ndarray]:
        """Return fitted bin edges in raw or transformed coordinates."""
        self._require_fitted()
        from .visualization import histogram_bin_edges

        return histogram_bin_edges(
            min_vals=self.raw_min_vals,
            max_vals=self.raw_max_vals,
            bins=self.bins_per_dim,
            value_transform=self.value_transforms,
            coordinate_space=coordinate_space,
        )

    def make_histogram(self):
        """Construct a :class:`Histogram` using the fitted contract."""
        self._require_fitted()
        from .data_handler import Histogram

        return Histogram(
            dimension=self.dimension,
            min_vals=self.raw_min_vals,
            max_vals=self.raw_max_vals,
            bins_per_dim=self.bins_per_dim,
            histogram_mode=self.histogram_mode,
            out_of_range_policy=self.tail_policy,
            value_transform=self.value_transforms,
        )

    def compute(self, data: np.ndarray) -> np.ndarray:
        """Compute one histogram from raw points using the fitted contract."""
        return self.make_histogram().compute(data)

    def config_overrides(self) -> dict[str, Any]:
        """Return resolved HistVAE config values plus serialized provenance."""
        self._require_fitted()
        return {
            "min_vals": self.min_vals,
            "max_vals": self.max_vals,
            "bins": self.bins,
            "histogram_mode": self.histogram_mode,
            "out_of_range_policy": self.tail_policy,
            "value_transform": self.value_transform,
            "histogram_preprocessor_state": self.state_dict(),
        }

    def state_dict(self, include_state_hash: bool = True) -> dict[str, Any]:
        """Return a YAML/JSON-safe fitted-state dictionary."""
        self._require_fitted()
        state = {
            "schema_version": PREPROCESSOR_SCHEMA_VERSION,
            "type": type(self).__name__,
            "axis_specs": [spec.to_dict() for spec in self.axis_specs],
            "bins": list(self.bins_per_dim),
            "histogram_mode": self.histogram_mode,
            "tail_policy": self.tail_policy,
            "resolved_min_vals": self.min_vals,
            "resolved_max_vals": self.max_vals,
            "fit_data_sha256": self.fit_data_sha256,
            "fit_summary": self.fit_summary,
        }
        if include_state_hash:
            state["state_sha256"] = self.state_sha256
        return state

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "HistogramPreprocessor":
        """Restore and strictly validate a fitted state dictionary."""
        if not isinstance(state, Mapping):
            raise TypeError("Preprocessor state must be a mapping.")
        required = {
            "schema_version",
            "type",
            "axis_specs",
            "bins",
            "histogram_mode",
            "tail_policy",
            "resolved_min_vals",
            "resolved_max_vals",
            "fit_data_sha256",
            "fit_summary",
            "state_sha256",
        }
        missing = required.difference(state)
        unknown = set(state).difference(required)
        if missing:
            raise ValueError(f"Preprocessor state is missing keys: {sorted(missing)}.")
        if unknown:
            raise ValueError(
                f"Preprocessor state has unsupported keys: {sorted(unknown)}."
            )
        if int(state["schema_version"]) != PREPROCESSOR_SCHEMA_VERSION:
            raise ValueError(
                "Unsupported HistogramPreprocessor schema_version: "
                f"{state['schema_version']!r}."
            )
        if state["type"] != cls.__name__:
            raise ValueError(f"Unsupported preprocessor type: {state['type']!r}.")

        instance = cls(
            axis_specs=state["axis_specs"],
            bins=state["bins"],
            histogram_mode=state["histogram_mode"],
            tail_policy=state["tail_policy"],
        )
        instance.raw_min_vals = normalize_range_values(
            state["resolved_min_vals"], instance.dimension, "resolved_min_vals"
        )
        instance.raw_max_vals = normalize_range_values(
            state["resolved_max_vals"], instance.dimension, "resolved_max_vals"
        )
        if np.any(instance.raw_min_vals >= instance.raw_max_vals):
            raise ValueError("Resolved preprocessor bounds must be strictly increasing.")
        for axis, transform in enumerate(instance.value_transforms):
            if transform == "log1p" and instance.raw_min_vals[axis] < 0:
                raise ValueError("log1p preprocessor bounds must be non-negative.")
        fit_data_sha256 = state["fit_data_sha256"]
        if not cls._is_sha256(fit_data_sha256):
            raise ValueError("fit_data_sha256 must be a 64-character SHA-256 hex string.")
        if not isinstance(state["fit_summary"], Mapping):
            raise TypeError("fit_summary must be a mapping.")
        instance.fit_data_sha256 = fit_data_sha256
        instance.fit_summary = dict(state["fit_summary"])

        expected_hash = state["state_sha256"]
        if not cls._is_sha256(expected_hash):
            raise ValueError("state_sha256 must be a 64-character SHA-256 hex string.")
        if expected_hash != instance.state_sha256:
            raise ValueError("HistogramPreprocessor state_sha256 does not match state.")
        return instance

    def save(self, path: str | Path) -> Path:
        """Save the fitted state as safe YAML."""
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(
                self.state_dict(),
                handle,
                sort_keys=False,
                allow_unicode=True,
            )
        return output

    @classmethod
    def load(cls, path: str | Path) -> "HistogramPreprocessor":
        """Load a fitted preprocessor from safe YAML."""
        with Path(path).open("r", encoding="utf-8") as handle:
            state = yaml.safe_load(handle)
        return cls.from_state_dict(state)

    def _require_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError("HistogramPreprocessor.fit() must be called first.")

    def _validate_data(self, data, name: str) -> np.ndarray:
        values = np.asarray(data, dtype=np.float64)
        if self.dimension == 1 and values.ndim == 1:
            values = values.reshape(-1, 1)
        if values.ndim != 2 or values.shape[1] != self.dimension:
            raise ValueError(
                f"{name} must have shape (n_rows, {self.dimension}); "
                f"got {values.shape}."
            )
        if len(values) == 0:
            raise ValueError(f"{name} must contain at least one row.")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} must contain only finite values.")
        return values

    @staticmethod
    def _validate_group(group, n_rows: int, required: bool):
        if group is None:
            if required:
                raise ValueError(
                    "group is required for quantile_weighting='group_equal'."
                )
            return None
        values = np.asarray(group)
        if values.ndim != 1 or len(values) != n_rows:
            raise ValueError(f"group must have shape ({n_rows},).")
        if pd.isna(values).any():
            raise ValueError("group must not contain missing values.")
        return values

    @staticmethod
    def _group_equal_weights(group: np.ndarray) -> np.ndarray:
        codes, uniques = pd.factorize(group, sort=False)
        if np.any(codes < 0):
            raise ValueError("group must not contain missing values.")
        counts = np.bincount(codes, minlength=len(uniques))
        return 1.0 / counts[codes].astype(np.float64)

    @staticmethod
    def _weighted_quantile(values, quantile, weights) -> float:
        order = np.argsort(values, kind="mergesort")
        sorted_values = values[order]
        sorted_weights = np.asarray(weights, dtype=np.float64)[order]
        cumulative = np.cumsum(sorted_weights)
        threshold = float(quantile) * cumulative[-1]
        index = int(np.searchsorted(cumulative, threshold, side="left"))
        index = min(index, len(sorted_values) - 1)
        return float(sorted_values[index])

    @classmethod
    def _resolve_bound(
            cls, values, mode, value, quantile, weighting, weights
            ) -> float:
        if mode == "fixed":
            return float(value)
        if weighting == "event":
            return float(np.quantile(values, float(quantile), method="linear"))
        return cls._weighted_quantile(values, float(quantile), weights)

    @staticmethod
    def _hash_fit_data(values: np.ndarray, group: np.ndarray | None) -> str:
        digest = hashlib.sha256()
        canonical = np.ascontiguousarray(values, dtype=np.float64)
        digest.update(str(canonical.shape).encode("utf-8"))
        digest.update(canonical.tobytes())
        if group is None:
            digest.update(b"group:none")
        else:
            digest.update(b"group:")
            for value in group:
                type_name = (
                    f"{type(value).__module__}.{type(value).__qualname__}"
                ).encode("utf-8")
                encoded = repr(value).encode("utf-8")
                digest.update(len(type_name).to_bytes(8, "little"))
                digest.update(type_name)
                digest.update(len(encoded).to_bytes(8, "little"))
                digest.update(encoded)
        return digest.hexdigest()

    @staticmethod
    def _is_sha256(value: Any) -> bool:
        if not isinstance(value, str) or len(value) != 64:
            return False
        try:
            int(value, 16)
        except ValueError:
            return False
        return True
