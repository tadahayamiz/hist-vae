# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 12:09:08 2019

data handler

@author: tadahaya
"""
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import inspect

import torch
from torch.utils.data import Dataset, DataLoader

# functions
HISTOGRAM_MODES = ("count", "density", "probability_mass")
OUT_OF_RANGE_POLICIES = ("drop", "clip", "error")
VALUE_TRANSFORMS = ("none", "log1p")
SAMPLING_MODES = ("random", "full")


def validate_histogram_mode(histogram_mode):
    if histogram_mode not in HISTOGRAM_MODES:
        raise ValueError(
            f"Unsupported histogram_mode: {histogram_mode!r}. "
            "Use 'count', 'density', or 'probability_mass'."
        )
    return histogram_mode


def validate_out_of_range_policy(out_of_range_policy):
    if out_of_range_policy not in OUT_OF_RANGE_POLICIES:
        raise ValueError(
            f"Unsupported out_of_range_policy: {out_of_range_policy!r}. "
            "Use 'drop', 'clip', or 'error'."
        )
    return out_of_range_policy


def validate_value_transform(value_transform):
    if value_transform not in VALUE_TRANSFORMS:
        raise ValueError(
            f"Unsupported value_transform: {value_transform!r}. "
            "Use 'none' or 'log1p'."
        )
    return value_transform


def validate_sampling_mode(sampling_mode):
    if sampling_mode not in SAMPLING_MODES:
        raise ValueError(
            f"Unsupported sampling_mode: {sampling_mode!r}. "
            "Use 'random' or 'full'."
        )
    return sampling_mode


class Histogram:
    """
    A class optimized to compute histograms efficiently without repeated dimension checks.
    """

    def __init__(
            self, dimension, max_vals, bins_per_dim, histogram_mode="count",
            out_of_range_policy="drop", value_transform="none"
            ):
        """
        Initialize the Histogram object with the optimal histogram function based on dimension.

        Parameters:
        -----------
        dimension : int
            The dimension of the input data (1, 2, or >=3).

        max_vals : list or np.ndarray
            Maximum values for each dimension.

        bins_per_dim : int or list
            Number of bins for each dimension.

        histogram_mode : str
            ``"count"`` keeps raw bin counts. ``"density"`` returns a
            probability density whose integral over the histogram range is 1.
            ``"probability_mass"`` returns non-negative bin probabilities
            whose sum is 1.

        out_of_range_policy : str
            ``"drop"`` preserves the legacy behavior. ``"clip"`` maps
            underflow and overflow values to the edge bins. ``"error"``
            rejects data outside the configured range.

        value_transform : str
            ``"none"`` uses linear-width bins. ``"log1p"`` applies log1p to
            both values and configured maxima before histogram construction.
        """
        self.dimension = dimension
        self.raw_max_vals = np.asarray(max_vals, dtype=np.float64)
        if self.raw_max_vals.shape != (dimension,):
            raise ValueError("max_vals must contain one value per dimension.")
        if not np.all(np.isfinite(self.raw_max_vals)) or np.any(self.raw_max_vals <= 0):
            raise ValueError("max_vals must contain finite positive values.")
        self.histogram_mode = validate_histogram_mode(histogram_mode)
        self.density = self.histogram_mode == "density"
        self.probability_mass = self.histogram_mode == "probability_mass"
        self.out_of_range_policy = validate_out_of_range_policy(out_of_range_policy)
        self.value_transform = validate_value_transform(value_transform)

        if self.value_transform == "log1p":
            self.max_vals = np.log1p(self.raw_max_vals)
        else:
            self.max_vals = self.raw_max_vals.copy()

        if isinstance(bins_per_dim, int):
            bins_per_dim = [bins_per_dim] * dimension

        self.bins_per_dim = bins_per_dim

        # Precompute bin edges based on dimensions
        self.edges = [
            np.linspace(0, self.max_vals[dim], self.bins_per_dim[dim] + 1)
            for dim in range(dimension)
        ]

        # Pre-select histogram function based on dimension
        if self.dimension == 1:
            self.hist_func = self._hist_1d
        elif self.dimension == 2:
            self.hist_func = self._hist_2d
        else:
            self.hist_func = self._hist_nd

    def _hist_1d(self, data):
        hist, _ = np.histogram(data, bins=self.edges[0], density=self.density)
        return hist

    def _hist_2d(self, data):
        hist, _, _ = np.histogram2d(
            data[:, 0], data[:, 1],
            bins=[self.edges[1], self.edges[0]],
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
            (values >= 0) & (values <= self.raw_max_vals), axis=1
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
            values = np.clip(values, 0, self.raw_max_vals)
        elif self.out_of_range_policy == "drop":
            values = values[valid_rows]

        if self.value_transform == "log1p":
            values = np.log1p(values)

        if self.dimension == 1:
            return values[:, 0]
        return values

    def compute(self, data):
        """
        Compute histogram using the pre-selected histogram function.

        Parameters:
        -----------
        data : np.ndarray
            Data array of shape (n_samples, dimension).

        Returns:
        --------
        hist : np.ndarray
            Computed histogram.
        """
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
                "Ensure max_vals includes at least one observation in every group."
            )
        if not np.all(np.isfinite(hist)):
            raise ValueError("Histogram contains non-finite values.")
        return hist


def plot_hist(hist_list, output="", show: bool=False, **plot_params):
    """
    Plot histograms (1D, 2D).

    Parameters:
    ----------
    hist_list : list of np.ndarray
        List of histograms to plot.

    output : str, optional
        File path to save the plot (default: "", meaning no save).

    **plot_params : dict, optional
        Dictionary containing plot customization options:
            - xlabel (str): Label for x-axis
            - ylabel (str): Label for y-axis
            - title_list (list of str): Titles for each subplot
            - cmap (str): Colormap for 2D histograms
            - aspect (str): Aspect ratio for 2D histograms (default: 'equal')
            - color (str): Bar color for 1D histograms (default: 'royalblue')
            - alpha (float): Transparency for 1D histograms (default: 0.7)
    """
    # Default plot parameters
    default_params = {
        "nrow": 1,
        "ncol": 3,
        "xlabel": "x",
        "ylabel": "y",
        "title_list": None,
        "cmap": "viridis",
        "aspect": "equal",
        "color": "royalblue",
        "alpha": 0.7
    }
    # merge default and custom params
    params = {**default_params, **plot_params}
    num_plots = len(hist_list)
    nrow, ncol = params["nrow"], params["ncol"]
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 5 * nrow))
    axes = np.atleast_1d(axes).flatten()  # Flatten for easy iteration
    for i, hist in enumerate(hist_list):
        ax = axes[i]
        dim = hist.ndim  # Detect dimensionality
        if dim == 1:
            ax.bar(range(len(hist)), hist, width=0.8, color=params["color"], alpha=params["alpha"])
            ax.set_xlabel(params["xlabel"])
            ax.set_ylabel(params["ylabel"])
            ax.set_title(params["title_list"][i] if params["title_list"] else f'1D Histogram {i+1}')
        elif dim == 2:
            im = ax.imshow(hist.T, origin='lower', cmap=params["cmap"], aspect=params["aspect"])
            fig.colorbar(im, ax=ax, label=params["ylabel"])
            ax.set_xlabel(params["xlabel"])
            ax.set_ylabel(params["ylabel"])
            ax.set_title(params["title_list"][i] if params["title_list"] else f'2D Histogram {i+1}')
        else:
            raise NotImplementedError("Only 1D and 2D histograms are supported.")
    # Remove unused subplots
    for j in range(num_plots, len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()
    if output:
        plt.savefig(output)
    if show:
        plt.show()
    plt.close()


class PointHistDataset(Dataset):
    def __init__(
            self, data, group, label=None, max_vals=(), transform=False,
            num_points=768, bins=64, transform_params=None,
            histogram_mode="count", out_of_range_policy="drop",
            value_transform="none", sampling_mode="random"
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
            ``"random"`` draws two point subsets for training. ``"full"``
            deterministically uses all points in the group for both views.
        
        """
        super().__init__()
        # check the input
        assert data.shape[0] == group.shape[0], "!! data and group must have the same number of samples !!"
        if label is not None:
            assert data.shape[0] == label.shape[0], "!! data, group, and label must have the same number of samples !!"
        assert len(max_vals) == data.shape[1], "!! max_vals must have the same number of dimensions as data !!"
        self.data = data
        self.group = group
        self.label = label
        self.bins = bins
        self.num_points = num_points
        self.ndim = data.shape[1]
        self.max_vals = max_vals
        self.histogram_mode = validate_histogram_mode(histogram_mode)
        self.sampling_mode = validate_sampling_mode(sampling_mode)
        if self.sampling_mode == "full" and transform:
            raise ValueError("sampling_mode='full' requires transform=False.")
        # tie the group to the data
        self.unique_groups = np.unique(group)
        self.idx2group = {i: j for i, j in enumerate(self.unique_groups)} # map index in the dataset to the group
        self.group2idx = {v: k for k, v in self.idx2group.items()} # map group to index in the dataset
        self.num_data = len(self.unique_groups)
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
            self.ndim, max_vals, bins,
            histogram_mode=self.histogram_mode,
            out_of_range_policy=out_of_range_policy,
            value_transform=value_transform,
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
        # return the data
        return (hist0, hist1), label
        # hist0, original; hist1, noisy


    def get_full_histogram(self, idx):
        """Return a deterministic, unaugmented histogram for one group."""
        group_idx = self.idx2group[idx]
        selected_indices = np.where(self.group == group_idx)[0]
        hist = self._calc_hist(self.data[selected_indices])
        return self._normalize_hist(hist, group_idx).unsqueeze(0)


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
        if labels[0] is None:
            label_batch = torch.full((len(labels),), -1, dtype=torch.int64)
        else:
            label_batch = torch.stack(labels, dim=0)
        return (hist0, hist1), label_batch

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
            self, data, group, label=None, transform=False,
            sampling_mode="random"
            ):
        """
        make dataset for training and testing

        """
        # check the input
        assert data.shape[0] == group.shape[0], "!! data, group, and label must have the same number of samples !!"
        if label is not None:
            assert data.shape[0] == label.shape[0], "!! data, group, and label must have the same number of samples !!"
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
        if transform is not None:
            ds_args["transform"] = transform
        ds_args["sampling_mode"] = sampling_mode
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