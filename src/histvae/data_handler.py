# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 12:09:08 2019

data handler

@author: tadahaya
"""
import numpy as np
import pandas as pd
import random
import inspect

import torch
from torch.utils.data import Dataset, DataLoader

from .preprocessing import (
    HISTOGRAM_MODES,
    VALUE_TRANSFORMS,
    HistogramPreprocessor,
    compress_axis_setting,
    normalize_bins,
    normalize_range_values,
    normalize_value_transforms,
    validate_histogram_mode,
    validate_value_transform,
)
from .visualization import histogram_bin_edges, plot_hist

# functions
OUT_OF_RANGE_POLICIES = ("drop", "clip", "error")
SAMPLING_MODES = ("random", "full")
TARGET_SAMPLING_MODES = ("paired", "full")


def validate_out_of_range_policy(out_of_range_policy):
    if out_of_range_policy not in OUT_OF_RANGE_POLICIES:
        raise ValueError(
            f"Unsupported out_of_range_policy: {out_of_range_policy!r}. "
            "Use 'drop', 'clip', or 'error'."
        )
    return out_of_range_policy

def validate_sampling_mode(sampling_mode):
    if sampling_mode not in SAMPLING_MODES:
        raise ValueError(
            f"Unsupported sampling_mode: {sampling_mode!r}. "
            "Use 'random' or 'full'."
        )
    return sampling_mode


def validate_target_sampling_mode(target_sampling_mode):
    if target_sampling_mode not in TARGET_SAMPLING_MODES:
        raise ValueError(
            f"Unsupported target_sampling_mode: {target_sampling_mode!r}. "
            "Use 'paired' or 'full'."
        )
    return target_sampling_mode


def validate_condition_array(condition, n_observations=None, name="condition"):
    values = np.asarray(condition)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    if values.ndim != 2:
        raise ValueError(
            f"{name} must have shape (n_observations, condition_dim)."
        )
    if n_observations is not None and values.shape[0] != n_observations:
        raise ValueError(
            f"{name} must have {n_observations} rows; got {values.shape[0]}."
        )
    try:
        values = values.astype(np.float32, copy=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric.") from exc
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain only finite values.")
    return values


class Histogram:
    """Compute histograms with fixed raw bounds and axis transforms."""

    def __init__(
            self, dimension, max_vals, bins_per_dim, histogram_mode="count",
            out_of_range_policy="drop", value_transform="none", min_vals=None
            ):
        """Initialize one histogram contract.

        Parameters
        ----------
        dimension : int
            Number of measurement dimensions.
        min_vals, max_vals : sequence of float
            Raw-space histogram bounds. ``min_vals`` defaults to zero on every
            axis for compatibility with the legacy API.
        bins_per_dim : int or sequence of int
            Number of bins on each axis.
        histogram_mode : {"count", "density", "probability_mass"}
            Histogram value contract.
        out_of_range_policy : {"drop", "clip", "error"}
            Handling of rows outside the configured raw-space bounds.
        value_transform : str or sequence of str
            ``"none"`` or ``"log1p"`` globally, or one transform per axis.
        """
        self.dimension = int(dimension)
        self.raw_min_vals = normalize_range_values(
            min_vals, self.dimension, "min_vals", default=0.0
        )
        self.raw_max_vals = normalize_range_values(
            max_vals, self.dimension, "max_vals"
        )
        if np.any(self.raw_max_vals <= self.raw_min_vals):
            raise ValueError(
                "max_vals must exceed min_vals on every histogram axis."
            )

        self.histogram_mode = validate_histogram_mode(histogram_mode)
        self.density = self.histogram_mode == "density"
        self.probability_mass = self.histogram_mode == "probability_mass"
        self.out_of_range_policy = validate_out_of_range_policy(
            out_of_range_policy
        )
        self.value_transforms = normalize_value_transforms(
            value_transform, self.dimension
        )
        self.value_transform = compress_axis_setting(self.value_transforms)
        self.bins_per_dim = list(normalize_bins(bins_per_dim, self.dimension))

        self.min_vals = self.raw_min_vals.copy()
        self.max_vals = self.raw_max_vals.copy()
        for axis, transform in enumerate(self.value_transforms):
            if transform == "log1p":
                if self.raw_min_vals[axis] < 0:
                    raise ValueError(
                        "value_transform='log1p' requires non-negative "
                        f"bounds on axis {axis}."
                    )
                self.min_vals[axis] = np.log1p(self.raw_min_vals[axis])
                self.max_vals[axis] = np.log1p(self.raw_max_vals[axis])

        self.edges = [
            np.linspace(
                self.min_vals[axis],
                self.max_vals[axis],
                self.bins_per_dim[axis] + 1,
            )
            for axis in range(self.dimension)
        ]

        if self.dimension == 1:
            self.hist_func = self._hist_1d
        elif self.dimension == 2:
            self.hist_func = self._hist_2d
        else:
            self.hist_func = self._hist_nd

    def get_bin_edges(self, coordinate_space="raw"):
        """Return copies of the bin edges in raw or transformed coordinates."""
        return histogram_bin_edges(
            min_vals=self.raw_min_vals,
            max_vals=self.raw_max_vals,
            bins=self.bins_per_dim,
            value_transform=self.value_transforms,
            coordinate_space=coordinate_space,
        )

    def _hist_1d(self, data):
        hist, _ = np.histogram(data, bins=self.edges[0], density=self.density)
        return hist

    def _hist_2d(self, data):
        hist, _, _ = np.histogram2d(
            data[:, 0], data[:, 1],
            bins=[self.edges[0], self.edges[1]],
            density=self.density,
        )
        return hist

    def _hist_nd(self, data):
        hist, _ = np.histogramdd(data, bins=self.edges, density=self.density)
        return hist

    def _prepare_data(self, data):
        values = np.asarray(data, dtype=np.float64)
        if self.dimension == 1:
            values = values.reshape(-1, 1)
        elif values.ndim != 2 or values.shape[1] != self.dimension:
            raise ValueError(
                f"Expected data with shape (n_samples, {self.dimension}), "
                f"got {values.shape}."
            )

        finite_rows = np.all(np.isfinite(values), axis=1)
        in_range_rows = np.all(
            (values >= self.raw_min_vals) & (values <= self.raw_max_vals),
            axis=1,
        )
        valid_rows = finite_rows & in_range_rows

        if self.out_of_range_policy == "error" and not np.all(valid_rows):
            invalid_count = int((~valid_rows).sum())
            raise ValueError(
                f"Found {invalid_count} observations outside the finite "
                "configured histogram range."
            )
        if self.out_of_range_policy == "clip":
            if not np.all(finite_rows):
                raise ValueError("Cannot clip NaN or infinite histogram values.")
            values = np.clip(values, self.raw_min_vals, self.raw_max_vals)
        elif self.out_of_range_policy == "drop":
            values = values[valid_rows]

        for axis, transform in enumerate(self.value_transforms):
            if transform == "log1p":
                values[:, axis] = np.log1p(values[:, axis])

        if self.dimension == 1:
            return values[:, 0]
        return values

    def compute(self, data):
        """Compute one histogram under the fixed contract."""
        prepared_data = self._prepare_data(data)
        with np.errstate(divide="ignore", invalid="ignore"):
            hist = self.hist_func(prepared_data)
        if self.probability_mass:
            total = hist.sum()
            if total <= 0:
                raise ValueError(
                    "Probability-mass histogram is undefined because no "
                    "observations remain in the configured range."
                )
            hist = hist.astype(np.float64, copy=False) / total
        if self.density and not np.all(np.isfinite(hist)):
            raise ValueError(
                "Density histogram is undefined for the configured range. "
                "Ensure the range includes at least one observation in every group."
            )
        if not np.all(np.isfinite(hist)):
            raise ValueError("Histogram contains non-finite values.")
        return hist



class PointHistDataset(Dataset):
    def __init__(
            self, data, group, label=None, max_vals=None, min_vals=None,
            transform=False, num_points=768, bins=None, transform_params=None,
            histogram_mode=None, out_of_range_policy=None,
            value_transform=None, sampling_mode="random",
            target_sampling_mode="paired", condition=None,
            histogram_preprocessor=None
            ):
        """
        Parameters
        ----------
        data: np.ndarray
            the data to be used for training

        group: np.ndarray
            the group to which the data belongs
        
        label: np.ndarray
            the label of the data
        
        max_vals: tuple
            the maximum values for each dimension of the data

        transform: bool
            whether to apply transform to the data

        bins: int
            the number of bins for the histogram

        num_points: int
            the number of points to be sampled

        histogram_mode: str
            ``"count"`` preserves the original count-based representation.
            ``"density"`` removes group-size intensity by normalizing each
            histogram to unit integral before the existing log/max scaling.
            ``"probability_mass"`` returns bounded bin probabilities and does
            not apply the legacy log/max scaling.

        sampling_mode: str
            ``"random"`` draws the model-input point subset. ``"full"``
            deterministically uses all points in the group.

        target_sampling_mode: str
            ``"paired"`` preserves the legacy independent random target when
            ``sampling_mode="random"``. ``"full"`` uses the deterministic
            full-group histogram as the denoising target.

        condition: np.ndarray, optional
            Numeric sample-level condition vector repeated for every point in
            a group. A one-dimensional array is treated as one feature. Values
            must be finite and constant within each group. Categorical
            technical batches should be encoded outside the model, for example
            as train-fitted one-hot vectors.

        histogram_preprocessor: HistogramPreprocessor, optional
            Fitted preprocessing state. When supplied, its bounds, transforms,
            tail policy, bin counts, and histogram mode are authoritative.
            Explicit histogram arguments must either be omitted or match the
            fitted state exactly.
        
        """
        super().__init__()
        data = np.asarray(data)
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        group = np.asarray(group)
        # check the input
        assert data.shape[0] == group.shape[0], "!! data and group must have the same number of samples !!"
        if label is not None:
            assert data.shape[0] == label.shape[0], "!! data, group, and label must have the same number of samples !!"
        if condition is not None:
            condition = validate_condition_array(
                condition, n_observations=data.shape[0]
            )
        self.ndim = data.shape[1]
        if histogram_preprocessor is not None:
            if not isinstance(histogram_preprocessor, HistogramPreprocessor):
                raise TypeError(
                    "histogram_preprocessor must be a HistogramPreprocessor."
                )
            histogram_preprocessor.require_fitted()
            if histogram_preprocessor.dimension != self.ndim:
                raise ValueError(
                    "histogram_preprocessor dimensions must match data."
                )
            expected = histogram_preprocessor.config_overrides()
            provided = {
                "min_vals": min_vals,
                "max_vals": max_vals,
                "bins": bins,
                "histogram_mode": histogram_mode,
                "out_of_range_policy": out_of_range_policy,
                "value_transform": value_transform,
            }
            for name, value in provided.items():
                if value is None:
                    continue
                expected_value = expected[name]
                if name in {"min_vals", "max_vals"}:
                    matches = np.array_equal(
                        np.asarray(value, dtype=np.float64),
                        np.asarray(expected_value, dtype=np.float64),
                    )
                elif name == "bins":
                    matches = normalize_bins(value, self.ndim) == tuple(
                        histogram_preprocessor.bins_per_dim
                    )
                elif name == "value_transform":
                    matches = normalize_value_transforms(
                        value, self.ndim
                    ) == histogram_preprocessor.value_transforms
                else:
                    matches = value == expected_value
                if not matches:
                    raise ValueError(
                        f"{name} conflicts with histogram_preprocessor state."
                    )
            min_vals = histogram_preprocessor.min_vals
            max_vals = histogram_preprocessor.max_vals
            bins = histogram_preprocessor.bins
            histogram_mode = histogram_preprocessor.histogram_mode
            out_of_range_policy = histogram_preprocessor.tail_policy
            value_transform = histogram_preprocessor.value_transform
        else:
            if max_vals is None:
                raise ValueError("max_vals must be provided without a preprocessor.")
            if bins is None:
                bins = 64
            if histogram_mode is None:
                histogram_mode = "count"
            if out_of_range_policy is None:
                out_of_range_policy = "drop"
            if value_transform is None:
                value_transform = "none"

        self.data = data
        self.group = group
        self.label = label
        self.bins = compress_axis_setting(normalize_bins(bins, self.ndim))
        self.num_points = num_points
        self.min_vals = normalize_range_values(
            min_vals, self.ndim, "min_vals", default=0.0
        )
        self.max_vals = normalize_range_values(
            max_vals, self.ndim, "max_vals"
        )
        if np.any(self.max_vals <= self.min_vals):
            raise ValueError("max_vals must exceed min_vals on every axis.")
        self.histogram_preprocessor = histogram_preprocessor
        self.histogram_mode = validate_histogram_mode(histogram_mode)
        self.sampling_mode = validate_sampling_mode(sampling_mode)
        self.target_sampling_mode = validate_target_sampling_mode(
            target_sampling_mode
        )
        if self.sampling_mode == "full" and transform:
            raise ValueError("sampling_mode='full' requires transform=False.")
        if self.histogram_mode == "probability_mass" and transform:
            raise ValueError(
                "histogram_mode='probability_mass' requires transform=False; "
                "the legacy histogram-value augmentation does not preserve "
                "the probability simplex."
            )
        # tie the group to the data
        self.unique_groups = np.unique(group)
        self.idx2group = {i: j for i, j in enumerate(self.unique_groups)} # map index in the dataset to the group
        self.group2idx = {v: k for k, v in self.idx2group.items()} # map group to index in the dataset
        self.num_data = len(self.unique_groups)
        self.condition = condition
        self.group_condition = None
        self.condition_dim = 0
        if condition is not None:
            self.condition_dim = condition.shape[1]
            group_condition = {}
            for group_idx in self.unique_groups:
                selected_indices = np.where(self.group == group_idx)[0]
                group_values = condition[selected_indices]
                reference = group_values[0]
                if not np.allclose(
                    group_values, reference, rtol=0.0, atol=1e-7
                ):
                    raise ValueError(
                        "condition must be constant within each group; "
                        f"found multiple values for group {group_idx!r}."
                    )
                group_condition[group_idx] = reference.copy()
            self.group_condition = group_condition
        self.transform = transform
        if transform:
            if transform_params is not None:
                trans = PCAugmentation(**transform_params)
            else:
                trans = PCAugmentation()
            self._transform_fxn = trans
        else:
            self._transform_fxn = lambda x: x
        # prepare histogram
        self.hist = Histogram(
            self.ndim, self.max_vals, self.bins,
            histogram_mode=self.histogram_mode,
            out_of_range_policy=out_of_range_policy,
            value_transform=value_transform,
            min_vals=self.min_vals,
        )
        # store normalization parameters
        # note: Dataset cannnot modify the data, so we need to store the normalization parameters
        self.log1p_max = dict()
        if self.histogram_mode != "probability_mass":
            for i in range(self.num_data):
                group_idx = self.idx2group[i]
                selected_indices = np.where(self.group == group_idx)[0]
                pointcloud = self.data[selected_indices]
                hist = self._calc_hist(pointcloud)
                hist = np.log1p(hist)
                tmp = np.max(hist) # store the max value for normalization
                self.log1p_max[group_idx] = tmp


    def __len__(self):
        return self.num_data


    def __getitem__(self, idx):
        # get the indicated data
        group_idx = self.idx2group[idx]
        selected_indices = np.where(self.group == group_idx)[0]
        pointcloud = self.data[selected_indices]
        if self.sampling_mode == "full":
            hist0 = self._normalize_hist(self._calc_hist(pointcloud), group_idx)
            hist1 = hist0.clone()
        else:
            # limit the number of points if necessary (random sampling)
            replace = pointcloud.shape[0] <= self.num_points
            if self.target_sampling_mode == "full":
                pointcloud0 = pointcloud
            else:
                idxs0 = np.random.choice(
                    pointcloud.shape[0], self.num_points, replace=replace
                )
                pointcloud0 = pointcloud[idxs0, :]
            idxs1 = np.random.choice(
                pointcloud.shape[0], self.num_points, replace=replace
            )
            pointcloud1 = pointcloud[idxs1, :]
            # prepare and normalize histograms
            hist0 = self._normalize_hist(self._calc_hist(pointcloud0), group_idx)
            hist1 = self._normalize_hist(self._calc_hist(pointcloud1), group_idx)
            # transform
            hist1 = self._transform_fxn(hist1) # hist1 only like translation
        # add channel dimension
        hist0 = hist0.unsqueeze(0)
        hist1 = hist1.unsqueeze(0)
        # prepare label
        if self.label is not None:
            label = self.label[selected_indices][0]
            label = torch.tensor(label, dtype=torch.int64)
        else:
            label = None
        data = (hist0, hist1)
        if self.group_condition is not None:
            condition = torch.tensor(
                self.group_condition[group_idx], dtype=torch.float32
            )
            data = data + (condition,)
        # return the data
        return data, label
        # hist0, original; hist1, noisy


    def get_full_histogram(self, idx):
        """Return a deterministic, unaugmented histogram for one group."""
        group_idx = self.idx2group[idx]
        selected_indices = np.where(self.group == group_idx)[0]
        hist = self._calc_hist(self.data[selected_indices])
        return self._normalize_hist(hist, group_idx).unsqueeze(0)

    def get_sampled_histogram(self, idx, rng=None):
        """Return one unaugmented ``num_points`` histogram for a group.

        Parameters
        ----------
        idx : int
            Dataset index.
        rng : numpy.random.Generator, optional
            Explicit random generator for reproducible visualization.
        """
        group_idx = self.idx2group[idx]
        selected_indices = np.where(self.group == group_idx)[0]
        pointcloud = self.data[selected_indices]
        replace = pointcloud.shape[0] <= self.num_points
        chooser = np.random if rng is None else rng
        sampled_indices = chooser.choice(
            pointcloud.shape[0], self.num_points, replace=replace
        )
        hist = self._calc_hist(pointcloud[sampled_indices])
        return self._normalize_hist(hist, group_idx).unsqueeze(0)


    def get_group_condition(self, idx):
        """Return one group's condition vector, or ``None`` when disabled."""
        if self.group_condition is None:
            return None
        group_idx = self.idx2group[idx]
        return torch.tensor(
            self.group_condition[group_idx], dtype=torch.float32
        )


    def get_bin_edges(self, coordinate_space="raw"):
        """Return histogram bin edges for visualization."""
        return self.hist.get_bin_edges(coordinate_space=coordinate_space)


    def _calc_hist(self, data):
        """
        Calculate the histogram of the data.

        Parameters
        ----------
        data: np.ndarray
            the data to be used for training

        Returns
        -------
        hist: np.ndarray
            the histogram of the data

        """
        hist = self.hist.compute(data)
        return hist


    def _normalize_hist(self, hist, group_idx):
        """
        Normalize the histogram.
        
        """
        if self.histogram_mode == "probability_mass":
            return torch.tensor(hist, dtype=torch.float32)
        hist = np.log1p(hist)
        max_val = self.log1p_max[group_idx]
        hist = hist / (max_val + 1e-6)
        return torch.tensor(hist, dtype=torch.float32)


    def transform_on(self, **transform_params):
        """
        transform on

        """
        if self.sampling_mode == "full":
            raise ValueError("sampling_mode='full' requires transform=False.")
        if self.histogram_mode == "probability_mass":
            raise ValueError(
                "histogram_mode='probability_mass' does not support the "
                "legacy histogram-value augmentation."
            )
        self.transform = True
        trans = PCAugmentation(**transform_params)
        self._transform_fxn = trans


    def transform_off(self):
        """
        transform off

        """
        self.transform = False
        self._transform_fxn = lambda x: x

# ToDo test this
class PCAugmentation:
    def __init__(self,
                 jitter_sigma=0.01,   # Standard deviation of jitter noise (adjustable)
                 jitter_clip=0.03,    # Maximum jitter noise magnitude
                 scale_range=(0.98, 1.02),  # Scaling range (restricted within ±2%)
                 translate_range=0.05,      # Translation range (small absolute range)
                 ):
        self.jitter_sigma = jitter_sigma
        self.jitter_clip = jitter_clip
        self.scale_range = scale_range
        self.translate_range = translate_range

    def jitter(self, points):
        jitter_noise = torch.clamp(
            self.jitter_sigma * torch.randn_like(points),
            -self.jitter_clip,
            self.jitter_clip
        )
        points_jittered = points + jitter_noise
        # Clip negative values to zero since coordinates must be positive
        return torch.clamp(points_jittered, min=0)

    def scale(self, points):
        scale = torch.empty(1).uniform_(*self.scale_range).to(points.device)
        return points * scale

    def translate(self, points):
        translation = torch.empty(points.size(-1)).uniform_(
            -self.translate_range,
            self.translate_range
        ).to(points.device)
        points_translated = points + translation
        # Clip negative values to zero
        return torch.clamp(points_translated, min=0)

    def __call__(self, points):
        points = self.jitter(points)
        points = self.scale(points)
        points = self.translate(points)
        return points


class PointHistDataLoader(DataLoader):
    @staticmethod
    def _collate_fn(batch):
        data = [item[0] for item in batch]
        labels = [item[1] for item in batch]
        hist0 = torch.stack([item[0] for item in data], dim=0)
        hist1 = torch.stack([item[1] for item in data], dim=0)
        tuple_lengths = {len(item) for item in data}
        if len(tuple_lengths) != 1:
            raise ValueError("All dataset items must have the same tuple length.")
        tuple_length = tuple_lengths.pop()
        if tuple_length not in {2, 3}:
            raise ValueError(
                "Dataset tuples must contain (hist0, hist1) or "
                "(hist0, hist1, condition)."
            )
        data_batch = (hist0, hist1)
        if tuple_length == 3:
            condition = torch.stack([item[2] for item in data], dim=0)
            data_batch = data_batch + (condition,)
        if labels[0] is None:
            label_batch = torch.full((len(labels),), -1, dtype=torch.int64)
        else:
            label_batch = torch.stack(labels, dim=0)
        return data_batch, label_batch

    def __init__(
            self, dataset, batch_size, shuffle=False, num_workers=2,
            pin_memory=True, generator=None, worker_init_fn=None
            ):
        """
        My DataLoader to support the custom dataset
        
        """
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            generator=generator,
            worker_init_fn=worker_init_fn,
            collate_fn=self._collate_fn,
            )


class DataHandler:
    def __init__(self, config:dict):
        assert isinstance(config, dict), "!! config must be a dictionary !!"
        self.config = config


    def make_dataset(
            self, data, group, label=None, condition=None, transform=False,
            sampling_mode="random", target_sampling_mode="paired",
            histogram_preprocessor=None
            ):
        """
        make dataset for training and testing

        """
        # check the input
        assert data.shape[0] == group.shape[0], "!! data, group, and label must have the same number of samples !!"
        if label is not None:
            assert data.shape[0] == label.shape[0], "!! data, group, and label must have the same number of samples !!"
        if condition is not None:
            condition = validate_condition_array(
                condition, n_observations=data.shape[0]
            )
        # prepare params
        ds_params = inspect.signature(PointHistDataset.__init__).parameters # diff
        ds_args = {k: self.config[k] for k in ds_params if k in self.config}
        # integrate arguments
        ds_args.update({
            "data": data,
            "group": group,
        })
        if label is not None:
            ds_args["label"] = label
        if condition is not None:
            ds_args["condition"] = condition
        if histogram_preprocessor is not None:
            ds_args["histogram_preprocessor"] = histogram_preprocessor
        if transform is not None:
            ds_args["transform"] = transform
        ds_args["sampling_mode"] = sampling_mode
        ds_args["target_sampling_mode"] = target_sampling_mode
        # create dataset
        dataset = PointHistDataset(**ds_args)
        return dataset


    def make_dataloader(
            self, dataset, mode="train", generator=None,
            worker_init_fn=None
            ):
        """
        prepare train and test loader
        
        Parameters
        ----------
        dataset: torch.utils.data.Dataset
            prepared Dataset instance

        mode: str
            "train" or "test"
                
        """
        # check the input
        assert isinstance(dataset, Dataset), "!! dataset must be a torch.utils.data.Dataset !!"
        assert mode in ["train", "test"], "!! mode must be 'train' or 'test' !!"
        # create dataloader
        dl_params = inspect.signature(PointHistDataLoader.__init__).parameters
        # note: only child class PointHistDataLoader arguments are extracted
        dl_args = {k: self.config[k] for k in dl_params if k in self.config and k not in ["dataset", "shuffle"]}
        shuffle = True if mode == "train" else False
        # integrate arguments
        dl_args["dataset"] = dataset
        dl_args["shuffle"] = shuffle
        dl_args["generator"] = generator
        dl_args["worker_init_fn"] = worker_init_fn
        loader = PointHistDataLoader(**dl_args)
        return loader
    

    def make_lut(self, dataset):
        """
        make lookup table for the dataset

        Parameters
        ----------
        dataset: torch.utils.data.Dataset
            prepared Dataset instance

        Returns
        -------
        lut: dict
            lookup table for the dataset
        
        """
        lut = {}
        for i in range(len(dataset)):
            group = dataset.idx2group[i]
            lut[group] = i
        lut = pd.DataFrame({"group": list(lut.keys()), "idx": list(lut.values())})
        return lut