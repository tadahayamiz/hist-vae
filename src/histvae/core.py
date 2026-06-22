# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 12:09:08 2019

core module
a class specific to the model

@author: tadahaya
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os, yaml
from datetime import datetime

from .models import (
    ModelHandler,
    validate_condition_mode,
    validate_decoder_output_mode,
    validate_reconstruction_loss,
)
from .trainer import FineTuner, PreTrainer, validate_latent_kl_schedule
from .data_handler import (
    DataHandler,
    validate_histogram_mode,
    validate_out_of_range_policy,
    validate_sampling_mode,
    validate_target_sampling_mode,
    validate_condition_array,
    validate_value_transform,
)
from .preprocessing import (
    GroupCoordinateNormalizer,
    HistogramPreprocessor,
    compress_axis_setting,
    normalize_bins,
    normalize_range_values,
    normalize_value_transforms,
    validate_group_coordinate_mode,
)
from .utils import fix_seed
from .visualization import (
    plot_hist as plot_histogram_grid,
    plot_reconstruction as plot_reconstruction_grid,
    plot_scatter as plot_scatter_grid,
)


OPTIMIZERS = ("radam_schedule_free", "radam")
RECONSTRUCTION_INPUT_MODES = ("full", "sampled")
PLOT_VALUE_MODES = ("auto", "bin_value", "density")


def validate_reconstruction_input_mode(input_mode):
    if input_mode not in RECONSTRUCTION_INPUT_MODES:
        raise ValueError(
            f"Unsupported input_mode: {input_mode!r}. "
            "Use 'full' or 'sampled'."
        )
    return input_mode


def resolve_plot_value_mode(dataset, coordinate_space, plot_value_mode):
    if plot_value_mode not in PLOT_VALUE_MODES:
        raise ValueError(
            f"Unsupported plot_value_mode: {plot_value_mode!r}. "
            "Use 'auto', 'bin_value', or 'density'."
        )
    if plot_value_mode != "auto":
        return plot_value_mode
    if (
            coordinate_space == "raw"
            and dataset.histogram_mode == "probability_mass"
            ):
        return "density"
    return "bin_value"


def _forward_kl_per_sample(target, reconstruction):
    target_flat = np.asarray(target, dtype=np.float64).reshape(len(target), -1)
    reconstruction_flat = np.asarray(
        reconstruction, dtype=np.float64
    ).reshape(len(reconstruction), -1)
    if target_flat.shape != reconstruction_flat.shape:
        raise ValueError("target and reconstruction shapes must match.")
    positive = target_flat > 0
    terms = np.zeros_like(target_flat)
    eps = np.finfo(np.float64).eps
    terms[positive] = target_flat[positive] * (
        np.log(target_flat[positive])
        - np.log(np.clip(reconstruction_flat[positive], eps, None))
    )
    return terms.sum(axis=1)


def validate_optimizer(optimizer_name):
    if optimizer_name not in OPTIMIZERS:
        raise ValueError(
            f"Unsupported optimizer: {optimizer_name!r}. "
            "Use 'radam_schedule_free' or 'radam'."
        )
    return optimizer_name


def validate_grouped_measure_contract(config):
    """Validate strict combinations for the grouped-measure model path."""
    histogram_mode = config["histogram_mode"]
    decoder_output_mode = config["decoder_output_mode"]
    reconstruction_loss = config["reconstruction_loss"]
    condition_mode = config["condition_mode"]

    if decoder_output_mode == "simplex_softmax" and histogram_mode != "probability_mass":
        raise ValueError(
            "decoder_output_mode='simplex_softmax' requires "
            "histogram_mode='probability_mass'."
        )
    if reconstruction_loss == "forward_kl":
        if histogram_mode != "probability_mass":
            raise ValueError(
                "reconstruction_loss='forward_kl' requires "
                "histogram_mode='probability_mass'."
            )
        if decoder_output_mode != "simplex_softmax":
            raise ValueError(
                "reconstruction_loss='forward_kl' requires "
                "decoder_output_mode='simplex_softmax'."
            )

    condition_dim = int(config.get("condition_dim", 0))
    if condition_mode == "none" and condition_dim != 0:
        raise ValueError(
            "condition_dim must be 0 when condition_mode='none'."
        )
    if condition_mode == "decoder" and condition_dim <= 0:
        raise ValueError(
            "condition_dim must be positive when condition_mode='decoder'."
        )


def make_optimizer(parameters, config):
    optimizer_name = validate_optimizer(
        config.get("optimizer", "radam_schedule_free")
    )
    kwargs = {
        "lr": float(config["lr"]),
        "betas": (0.9, 0.999),
        "weight_decay": float(config["weight_decay"]),
    }
    if optimizer_name == "radam":
        return optim.RAdam(parameters, **kwargs)
    try:
        from schedulefree import RAdamScheduleFree
    except ImportError as exc:
        raise ImportError(
            "optimizer='radam_schedule_free' requires the 'schedulefree' "
            "package. Install it or set optimizer='radam'."
        ) from exc
    return RAdamScheduleFree(parameters, **kwargs)

class HistVAE:
    def __init__(
            self, config: dict=None, outdir: str=None, exp_name: str=None,
            seed: int=42, histogram_preprocessor=None,
            group_coordinate_normalizer=None
            ):
        # arguments
        assert config is not None, "!! config must be given !!"
        self.config = config
        self.config["in_dims"] = int(self.config["in_dims"])
        dimension = self.config["in_dims"]

        configured_coordinate_mode = validate_group_coordinate_mode(
            self.config.get("group_coordinate_mode", "none")
        )
        serialized_coordinate_state = self.config.get(
            "group_coordinate_normalizer_state"
        )
        serialized_coordinate_normalizer = (
            None
            if serialized_coordinate_state is None
            else GroupCoordinateNormalizer.from_state_dict(
                serialized_coordinate_state
            )
        )
        if serialized_coordinate_normalizer is not None:
            if serialized_coordinate_normalizer.dimension != dimension:
                raise ValueError(
                    "group_coordinate_normalizer dimensions must equal "
                    "config in_dims."
                )
            if configured_coordinate_mode != (
                    serialized_coordinate_normalizer.mode
                    ):
                raise ValueError(
                    "group_coordinate_mode differs from the serialized "
                    "group_coordinate_normalizer state."
                )
        if group_coordinate_normalizer is not None:
            if not isinstance(
                    group_coordinate_normalizer, GroupCoordinateNormalizer
                    ):
                raise TypeError(
                    "group_coordinate_normalizer must be a "
                    "GroupCoordinateNormalizer."
                )
            if group_coordinate_normalizer.dimension != dimension:
                raise ValueError(
                    "group_coordinate_normalizer dimensions must equal "
                    "config in_dims."
                )
            if (
                    serialized_coordinate_normalizer is None
                    and configured_coordinate_mode != "none"
                    and configured_coordinate_mode
                    != group_coordinate_normalizer.mode
                    ):
                raise ValueError(
                    "The constructor group_coordinate_normalizer conflicts "
                    "with the explicit non-'none' group_coordinate_mode in "
                    "config."
                )
            if (
                    serialized_coordinate_normalizer is not None
                    and serialized_coordinate_normalizer.state_sha256
                    != group_coordinate_normalizer.state_sha256
                    ):
                raise ValueError(
                    "The constructor group_coordinate_normalizer differs "
                    "from the serialized config state."
                )
            resolved_coordinate_normalizer = group_coordinate_normalizer
        elif serialized_coordinate_normalizer is not None:
            resolved_coordinate_normalizer = serialized_coordinate_normalizer
        else:
            resolved_coordinate_normalizer = GroupCoordinateNormalizer(
                mode=configured_coordinate_mode,
                dimension=dimension,
            )
        self.config.update(resolved_coordinate_normalizer.config_overrides())

        serialized_state = self.config.get("histogram_preprocessor_state")
        serialized_preprocessor = (
            None
            if serialized_state is None
            else HistogramPreprocessor.from_state_dict(serialized_state)
        )
        if histogram_preprocessor is not None:
            if not isinstance(histogram_preprocessor, HistogramPreprocessor):
                raise TypeError(
                    "histogram_preprocessor must be a HistogramPreprocessor."
                )
            histogram_preprocessor.require_fitted()
            if (
                    serialized_preprocessor is not None
                    and serialized_preprocessor.state_sha256
                    != histogram_preprocessor.state_sha256
                    ):
                raise ValueError(
                    "The constructor histogram_preprocessor differs from the "
                    "serialized config state."
                )
            resolved_preprocessor = histogram_preprocessor
        else:
            resolved_preprocessor = serialized_preprocessor
        if resolved_preprocessor is not None:
            if resolved_preprocessor.dimension != dimension:
                raise ValueError(
                    "histogram_preprocessor dimensions must equal config in_dims."
                )
            resolved_coordinate_normalizer.validate_histogram_preprocessor(
                resolved_preprocessor
            )
            self.config.update(resolved_preprocessor.config_overrides())

        bins_per_dim = normalize_bins(self.config["bins"], dimension)
        self.config["bins"] = compress_axis_setting(bins_per_dim)
        self._pending_histogram_geometry = {}
        self.config["min_vals"] = self._normalize_initial_range(
            self.config.get("min_vals"), dimension, "min_vals", default=0.0
        )
        self.config["max_vals"] = self._normalize_initial_range(
            self.config.get("max_vals"), dimension, "max_vals"
        )
        if not self._pending_histogram_geometry:
            if self.config["max_vals"] is not None and np.any(
                    np.asarray(self.config["max_vals"])
                    <= np.asarray(self.config["min_vals"])
                    ):
                raise ValueError("max_vals must exceed min_vals on every axis.")
        self.config["histogram_mode"] = validate_histogram_mode(
            self.config.get("histogram_mode", "count")
            )
        self.config["out_of_range_policy"] = validate_out_of_range_policy(
            self.config.get("out_of_range_policy", "drop")
            )
        self.config["value_transform"] = compress_axis_setting(
            normalize_value_transforms(
                self.config.get("value_transform", "none"), dimension
            )
        )
        self.config["train_sampling_mode"] = validate_sampling_mode(
            self.config.get("train_sampling_mode", "random")
            )
        self.config["eval_sampling_mode"] = validate_sampling_mode(
            self.config.get("eval_sampling_mode", "full")
            )
        self.config["train_target_sampling_mode"] = validate_target_sampling_mode(
            self.config.get("train_target_sampling_mode", "paired")
            )
        self.config["eval_target_sampling_mode"] = validate_target_sampling_mode(
            self.config.get("eval_target_sampling_mode", "full")
            )
        self.config["decoder_output_mode"] = validate_decoder_output_mode(
            self.config.get("decoder_output_mode", "legacy_sigmoid")
            )
        self.config["reconstruction_loss"] = validate_reconstruction_loss(
            self.config.get("reconstruction_loss", "mse")
            )
        self.config["condition_mode"] = validate_condition_mode(
            self.config.get("condition_mode", "none")
            )
        self.config["condition_dim"] = int(
            self.config.get("condition_dim", 0)
            )
        validate_grouped_measure_contract(self.config)
        validate_latent_kl_schedule(self.config)
        self.config["optimizer"] = validate_optimizer(
            self.config.get("optimizer", "radam_schedule_free")
            )
        self.outdir = outdir
        self.exp_name = exp_name
        self.seed = seed
        # delegate
        self.data_handler = DataHandler(config)
        self.model_handler = ModelHandler(config)
        # initialize
        self.train_dataset = None
        self.test_dataset = None
        self.train_loader = None
        self.test_loader = None
        self.train_lut = None
        self.test_lut = None
        self.group_coordinate_normalizer = resolved_coordinate_normalizer
        self.group_coordinate_statistics = None
        self.histogram_preprocessor = resolved_preprocessor
        self.model = None
        self.trainer = None
        self.optimizer = None
        self.loss_fn = None
        # fix seed
        g, seed_worker = fix_seed(seed, fix_cuda=True)
        self._seed = {"seed": seed, "generator": g, "worker_init_fn": seed_worker}
        # loading
        self.device = self.config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        if exp_name is None:
            exp_name = f"exp-{datetime.today().strftime('%y%m%d')}"
        self.config["exp_name"] = exp_name
        bins_per_dim = normalize_bins(self.config["bins"], self.config["in_dims"])
        tmp = [self.config["in_channels"], *bins_per_dim]
        self.config["input_shape"] = tmp # hard coded for ConvVAE


    def _normalize_initial_range(self, values, dimension, name, default=None):
        """Normalize compatible bounds while allowing a later preprocessor.

        Packaged defaults describe the default dimensionality.  A caller may
        change ``in_dims`` and then provide a fitted HistogramPreprocessor in
        :meth:`prep_data`; stale default bounds must not prevent that explicit
        source of truth from being applied.  Without a preprocessor, the same
        mismatch is rejected before dataset construction.
        """
        if values is None:
            if default is None:
                return None
            return [float(default)] * dimension
        candidate = np.asarray(values, dtype=np.float64)
        if candidate.shape != (dimension,):
            self._pending_histogram_geometry[name] = {
                "expected_dimension": dimension,
                "observed_shape": list(candidate.shape),
            }
            return candidate.tolist()
        if not np.all(np.isfinite(candidate)):
            raise ValueError(f"{name} must contain finite values.")
        return candidate.tolist()


    def _apply_histogram_preprocessor(self, preprocessor):
        """Use one fitted preprocessor as the histogram source of truth."""
        if not isinstance(preprocessor, HistogramPreprocessor):
            raise TypeError(
                "histogram_preprocessor must be a HistogramPreprocessor."
            )
        preprocessor.require_fitted()
        if preprocessor.dimension != self.config["in_dims"]:
            raise ValueError(
                "histogram_preprocessor dimensions must equal config in_dims."
            )
        self.group_coordinate_normalizer.validate_histogram_preprocessor(
            preprocessor
        )
        self.config.update(preprocessor.config_overrides())
        self._pending_histogram_geometry = {}
        validate_grouped_measure_contract(self.config)
        bins_per_dim = normalize_bins(
            self.config["bins"], self.config["in_dims"]
        )
        self.config["input_shape"] = [
            self.config["in_channels"], *bins_per_dim
        ]
        self.histogram_preprocessor = preprocessor


    def prep_data(
            self, train_data=None, train_group=None, train_label=None,
            train_condition=None, train_transform=None,
            test_data=None, test_group=None, test_label=None,
            test_condition=None, test_transform=None,
            histogram_mode=None, out_of_range_policy=None,
            value_transform=None, train_sampling_mode=None,
            test_sampling_mode=None, train_target_sampling_mode=None,
            test_target_sampling_mode=None, condition_mode=None,
            histogram_preprocessor=None
            ):
        """Prepare grouped point data as histograms.

        Parameters
        ----------
        histogram_mode: str, optional
            Runtime override for the configured histogram representation.
            Use ``"count"`` for the original count/intensity behavior or
            ``"density"`` to normalize each histogram to unit integral, or
            ``"probability_mass"`` for bounded bin probabilities.

        out_of_range_policy: str, optional
            Runtime override for ``"drop"``, ``"clip"``, or ``"error"``.

        value_transform: str, optional
            Runtime override for linear (``"none"``) or log1p-spaced bins.

        train_condition, test_condition: np.ndarray, optional
            Numeric row-level condition vectors repeated within each group.
            They are accepted only when ``condition_mode="decoder"``, must be
            finite, must have width ``condition_dim``, and must be constant
            within every group. The encoder never receives these values.
            Categorical technical batches should be encoded by the caller using
            a mapping fitted on the training split.

        condition_mode: str, optional
            Runtime override for ``"none"`` or decoder-only conditioning.

        train_target_sampling_mode, test_target_sampling_mode: str, optional
            ``"paired"`` preserves the independent sampled target used by the
            legacy denoising path. ``"full"`` uses the deterministic full-group
            histogram as the reconstruction target.

        histogram_preprocessor: HistogramPreprocessor, optional
            Fitted training-data preprocessing contract. Bounds, transforms,
            tail handling, bin counts, and histogram mode are applied unchanged
            to train and test data and serialized into the experiment config.
            Runtime histogram overrides must be omitted when this is supplied.
            Passing it to the HistVAE constructor is preferred because the
            fitted state then replaces default geometry before model-contract
            validation.

        Notes
        -----
        ``group_coordinate_mode`` is resolved in the constructor from the
        config, serialized state, or explicit ``GroupCoordinateNormalizer``.
        Raw train/test rows are normalized with one full-group statistic before
        dataset sampling. Non-``"none"`` modes require a fitted histogram
        preprocessor whose geometry exactly replays the normalized training
        rows.
        """
        histogram_overrides = {
            "histogram_mode": histogram_mode,
            "out_of_range_policy": out_of_range_policy,
            "value_transform": value_transform,
        }
        if histogram_preprocessor is not None:
            if any(value is not None for value in histogram_overrides.values()):
                raise ValueError(
                    "Histogram runtime overrides must be omitted when "
                    "histogram_preprocessor is supplied."
                )
            if self.histogram_preprocessor is not None and (
                    self.histogram_preprocessor.state_sha256
                    != histogram_preprocessor.state_sha256
                    ):
                raise ValueError(
                    "The supplied histogram_preprocessor differs from the "
                    "state already recorded in this HistVAE instance."
                )
            self._apply_histogram_preprocessor(histogram_preprocessor)
        elif self.histogram_preprocessor is not None and any(
                value is not None for value in histogram_overrides.values()
                ):
            raise ValueError(
                "Histogram runtime overrides cannot replace the serialized "
                "histogram_preprocessor state."
            )

        if histogram_mode is not None:
            self.config["histogram_mode"] = validate_histogram_mode(histogram_mode)
        if out_of_range_policy is not None:
            self.config["out_of_range_policy"] = validate_out_of_range_policy(
                out_of_range_policy
            )
        if value_transform is not None:
            self.config["value_transform"] = compress_axis_setting(
                normalize_value_transforms(
                    value_transform, self.config["in_dims"]
                )
            )
        if condition_mode is not None:
            self.config["condition_mode"] = validate_condition_mode(
                condition_mode
            )
        validate_grouped_measure_contract(self.config)
        if train_transform is None:
            train_transform = self.config.get("transform", True)
        if test_transform is None:
            test_transform = False
        if train_sampling_mode is None:
            train_sampling_mode = self.config["train_sampling_mode"]
        else:
            train_sampling_mode = validate_sampling_mode(train_sampling_mode)
            self.config["train_sampling_mode"] = train_sampling_mode
        if test_sampling_mode is None:
            test_sampling_mode = self.config["eval_sampling_mode"]
        else:
            test_sampling_mode = validate_sampling_mode(test_sampling_mode)
            self.config["eval_sampling_mode"] = test_sampling_mode
        if train_target_sampling_mode is None:
            train_target_sampling_mode = self.config["train_target_sampling_mode"]
        else:
            train_target_sampling_mode = validate_target_sampling_mode(
                train_target_sampling_mode
            )
            self.config["train_target_sampling_mode"] = train_target_sampling_mode
        if test_target_sampling_mode is None:
            test_target_sampling_mode = self.config["eval_target_sampling_mode"]
        else:
            test_target_sampling_mode = validate_target_sampling_mode(
                test_target_sampling_mode
            )
            self.config["eval_target_sampling_mode"] = test_target_sampling_mode

        resolved_condition_mode = self.config["condition_mode"]
        if resolved_condition_mode == "none":
            if train_condition is not None or test_condition is not None:
                raise ValueError(
                    "Condition arrays were provided while condition_mode='none'."
                )
        else:
            if train_condition is None:
                raise ValueError(
                    "train_condition is required when condition_mode='decoder'."
                )
            if test_data is not None and test_condition is None:
                raise ValueError(
                    "test_condition is required for test_data when "
                    "condition_mode='decoder'."
                )
            for name, condition, data in (
                ("train_condition", train_condition, train_data),
                ("test_condition", test_condition, test_data),
            ):
                if condition is None:
                    continue
                condition_array = validate_condition_array(
                    condition,
                    n_observations=data.shape[0],
                    name=name,
                )
                if condition_array.shape[1] != self.config["condition_dim"]:
                    raise ValueError(
                        f"{name} width must equal condition_dim="
                        f"{self.config['condition_dim']}; got "
                        f"{condition_array.shape[1]}."
                    )
                if name == "train_condition":
                    train_condition = condition_array
                else:
                    test_condition = condition_array

        if train_data is None or train_group is None:
            raise ValueError("train_data and train_group are required.")
        train_data = np.asarray(train_data)
        train_group = np.asarray(train_group)
        if train_data.ndim == 1:
            train_data = train_data.reshape(-1, 1)
        if train_data.ndim != 2 or train_data.shape[1] != self.config["in_dims"]:
            raise ValueError(
                "train_data dimensions must match config in_dims."
            )
        if test_data is not None:
            if test_group is None:
                raise ValueError("test_group is required when test_data is supplied.")
            test_data = np.asarray(test_data)
            test_group = np.asarray(test_group)
            if test_data.ndim == 1:
                test_data = test_data.reshape(-1, 1)
            if test_data.ndim != 2 or test_data.shape[1] != self.config["in_dims"]:
                raise ValueError(
                    "test_data dimensions must match config in_dims."
                )

        self.group_coordinate_normalizer.validate_histogram_preprocessor(
            self.histogram_preprocessor
        )
        train_coordinate_data, train_coordinate_statistics = (
            self.group_coordinate_normalizer.transform(
                train_data,
                train_group,
                return_statistics=True,
            )
        )
        test_coordinate_data = None
        test_coordinate_statistics = None
        if test_data is not None:
            test_coordinate_data, test_coordinate_statistics = (
                self.group_coordinate_normalizer.transform(
                    test_data,
                    test_group,
                    return_statistics=True,
                )
            )
        self.group_coordinate_statistics = {
            "train": train_coordinate_statistics,
        }
        if test_coordinate_statistics is not None:
            self.group_coordinate_statistics["test"] = (
                test_coordinate_statistics
            )

        if self.histogram_preprocessor is None and self._pending_histogram_geometry:
            details = ", ".join(
                f"{name}: expected {item['expected_dimension']} value(s), "
                f"got shape {tuple(item['observed_shape'])}"
                for name, item in self._pending_histogram_geometry.items()
            )
            raise ValueError(
                "Histogram bounds do not match config in_dims ("
                f"{details}). Supply matching bounds or a fitted "
                "histogram_preprocessor."
            )
        if self.config.get("max_vals") is None:
            raise ValueError(
                "max_vals must be configured or supplied by a fitted "
                "histogram_preprocessor."
            )

        if self.histogram_preprocessor is not None:
            if self.group_coordinate_normalizer.mode != "none":
                fit_used_group = (
                    self.histogram_preprocessor.fit_summary.get("n_groups")
                    is not None
                )
                self.histogram_preprocessor.validate_fit_data(
                    train_coordinate_data,
                    train_group if fit_used_group else None,
                )
            diagnostics = {
                "train": self.histogram_preprocessor.diagnose(
                    train_coordinate_data
                ),
            }
            if test_coordinate_data is not None:
                diagnostics["test"] = self.histogram_preprocessor.diagnose(
                    test_coordinate_data
                )
            self.config["histogram_preprocessor_diagnostics"] = diagnostics
        # dataset
        self.train_dataset = self.data_handler.make_dataset(
            data=train_data, group=train_group, label=train_label,
            condition=train_condition, transform=train_transform,
            sampling_mode=train_sampling_mode,
            target_sampling_mode=train_target_sampling_mode,
            histogram_preprocessor=self.histogram_preprocessor,
            group_coordinate_normalizer=self.group_coordinate_normalizer,
            )
        if test_data is not None:
            self.test_dataset = self.data_handler.make_dataset(
                data=test_data, group=test_group, label=test_label,
                condition=test_condition, transform=test_transform,
                sampling_mode=test_sampling_mode,
                target_sampling_mode=test_target_sampling_mode,
                histogram_preprocessor=self.histogram_preprocessor,
                group_coordinate_normalizer=self.group_coordinate_normalizer,
                )
        # dataloader
        self.train_loader = self.data_handler.make_dataloader(
            dataset=self.train_dataset, mode="train",
            generator=self._seed["generator"],
            worker_init_fn=self._seed["worker_init_fn"],
            )
        if self.test_dataset is not None:
            self.test_loader = self.data_handler.make_dataloader(
                dataset=self.test_dataset, mode="test",
                generator=self._seed["generator"],
                worker_init_fn=self._seed["worker_init_fn"],
                )
        # lookup table
        self.train_lut = self.data_handler.make_lut(dataset=self.train_dataset)
        self.test_lut = None
        if self.test_dataset is not None:
            self.test_lut = self.data_handler.make_lut(dataset=self.test_dataset)


    def prep_model(self, mode="pretrain", model_path:str=None):
        """
        prepare model
        hard coded parameters

        Parameters
        ----------
        mode: str
            "pretrain", "cpt", or "finetune"

        model_path: str
            path to the pretrained model
            only used in "cpt" and "finetune"
        
        """
        # check the mode
        assert mode in ["pretrain", "cpt", "finetune"], "!! mode must be pretrain, cpt, or finetune !!"
        if mode == "pretrain":
            # prepare pretraining model
            self.model = self.model_handler.make_pretrain()
            self.optimizer = make_optimizer(self.model.parameters(), self.config)
            self.trainer = PreTrainer(
                self.config, self.model, self.optimizer, outdir=self.outdir
                )
        elif mode == "cpt":
            # prepare continuous pretraining model
            assert model_path is not None, "!! model_path must be given in cpt mode!!"
            self.model = self.model_handler.make_cpt(model_path=model_path)
            self.optimizer = make_optimizer(self.model.parameters(), self.config)
            self.trainer = PreTrainer(
                self.config, self.model, self.optimizer, outdir=self.outdir
                )
        elif mode == "finetune":
            # prepare finetuning model
            assert model_path is not None, "!! model_path must be given in finetune mode!!"
            self.model = self.model_handler.make_finetune(model_path=model_path)
            self.optimizer = make_optimizer(self.model.parameters(), self.config)
            self.loss_fn = nn.CrossEntropyLoss()
            self.trainer = FineTuner(
                self.config, self.model, self.optimizer, self.loss_fn, outdir=self.outdir
                )


    def train(self, callbacks:list=None, verbose:bool=True):
        """ training """
        if callbacks is not None:
            self.trainer.set_callbacks(callbacks)
        self.trainer.train(self.train_loader, self.test_loader)
        if verbose:
            print(">> Training is done.")


    def predict(self, data_loader=None):
        """Run classifier inference with a fine-tuned model.

        Parameters
        ----------
        data_loader: torch.utils.data.DataLoader
            DataLoader that yields ``((hist0, hist1), label)``.

        Returns
        -------
        preds, probs, labels: np.ndarray
            Predicted class indices, classifier logits, and labels.
        """
        if data_loader is None:
            raise ValueError("!! Give data_loader !!")
        if self.model is None:
            raise ValueError("!! fit or load_model first !!")
        if self.loss_fn is None:
            raise RuntimeError("!! predict is only available after prep_model('finetune') !!")
        self.model.eval()
        preds = []
        probs = []
        labels = []
        with torch.no_grad():
            for data, label in data_loader:
                hist0, hist1 = (x.to(self.device) for x in data[:2])
                condition = data[2].to(self.device) if len(data) == 3 else None
                label = label.to(self.device)
                logits, recon, mu, logvar = self.model(
                    hist1, sample_latent=False, condition=condition
                    ) # use deterministic model-input histogram
                preds.append(logits.argmax(dim=1).cpu().numpy())
                probs.append(logits.cpu().numpy())
                labels.append(label.cpu().numpy())
        return np.concatenate(preds), np.concatenate(probs), np.concatenate(labels)


    def get_latent(self, dataset=None, indices=None):
        """
        get latent representation
        note: pretrained model weight is changed after finetuning.
        
        """
        if dataset is None:
            dataset = self.test_dataset
        if self.model is None:
            raise ValueError("!! fit or load_model first !!")
        self.model.eval()
        num_data = len(dataset)
        if indices is None or len(indices) == 0:
            indices = list(range(num_data))
        reps = []
        with torch.no_grad():
            for i in indices:
                hist0 = dataset.get_full_histogram(i).to(self.device).unsqueeze(0)
                mu, logvar = self.model.encode(hist0) # use original hist
                # note both ConvVAE and LinearHead have encode method
                reps.append(mu.cpu().numpy().reshape(1, -1))  # del batch dimension
        return np.vstack(reps)


    def get_reconstruction(
            self, dataset=None, indices=None, input_mode="full", random_seed=0
            ):
        """Return deterministic decoder reconstructions for grouped histograms.

        Parameters
        ----------
        dataset: PointHistDataset, optional
            Dataset to evaluate. The test dataset is used by default, falling
            back to the train dataset.
        indices: sequence of int, optional
            Dataset indices. All groups are used when omitted.
        input_mode: {"full", "sampled"}
            ``"full"`` uses the deterministic full-group histogram.
            ``"sampled"`` draws one unaugmented ``num_points`` histogram and
            retains the full-group histogram as the visualization target.
        random_seed: int
            Seed for ``input_mode="sampled"``.

        Returns
        -------
        dict
            Indices, group identifiers, target, model input, reconstruction,
            posterior mean, and posterior log variance.
        """
        if self.model is None:
            raise ValueError("!! fit or load_model first !!")
        if dataset is None:
            dataset = (
                self.test_dataset
                if self.test_dataset is not None
                else self.train_dataset
            )
        if dataset is None:
            raise ValueError("A prepared dataset is required.")
        input_mode = validate_reconstruction_input_mode(input_mode)

        if indices is None:
            indices = list(range(len(dataset)))
        else:
            indices = [int(index) for index in indices]
        if not indices:
            raise ValueError("indices must contain at least one dataset index.")
        if any(index < 0 or index >= len(dataset) for index in indices):
            raise IndexError("indices contains an out-of-range dataset index.")

        rng = np.random.default_rng(random_seed)
        targets = []
        inputs = []
        reconstructions = []
        means = []
        logvars = []
        groups = []
        model_device = next(self.model.parameters()).device

        self.model.eval()
        with torch.inference_mode():
            for index in indices:
                target = dataset.get_full_histogram(index)
                if input_mode == "full":
                    model_input = target.clone()
                else:
                    model_input = dataset.get_sampled_histogram(index, rng=rng)

                condition = dataset.get_group_condition(index)
                if condition is not None:
                    condition = condition.to(model_device).unsqueeze(0)

                outputs = self.model(
                    model_input.to(model_device).unsqueeze(0),
                    sample_latent=False,
                    condition=condition,
                )
                if not isinstance(outputs, tuple):
                    raise RuntimeError("Model output must be a tuple.")
                if len(outputs) == 3:
                    reconstruction, mean, logvar = outputs
                elif len(outputs) == 4:
                    _, reconstruction, mean, logvar = outputs
                else:
                    raise RuntimeError(
                        "Expected pretraining or fine-tuning model output."
                    )

                targets.append(target.cpu().numpy())
                inputs.append(model_input.cpu().numpy())
                reconstructions.append(reconstruction.squeeze(0).cpu().numpy())
                means.append(mean.squeeze(0).cpu().numpy())
                logvars.append(logvar.squeeze(0).cpu().numpy())
                groups.append(dataset.idx2group[index])

        return {
            "indices": np.asarray(indices, dtype=np.int64),
            "groups": np.asarray(groups),
            "target": np.stack(targets),
            "input": np.stack(inputs),
            "reconstruction": np.stack(reconstructions),
            "mu": np.stack(means),
            "logvar": np.stack(logvars),
            "input_mode": input_mode,
        }


    def plot_reconstruction(
            self, dataset=None, indices=None, input_mode="full", random_seed=0,
            coordinate_space="raw", plot_value_mode="auto", output="",
            show=False, close=True, **plot_params
            ):
        """Plot target and deterministic reconstruction in explicit coordinates.

        Raw coordinate space is the default. For probability-mass histograms
        built with log1p-spaced bins, ``plot_value_mode="auto"`` converts bin
        mass to density per raw-space width or area. This avoids displaying
        transformed coordinates or treating unequal raw-width bins as equal.

        Returns
        -------
        result, figure, axes
            Reconstruction arrays and the Matplotlib objects.
        """
        if dataset is None:
            dataset = (
                self.test_dataset
                if self.test_dataset is not None
                else self.train_dataset
            )
        if dataset is None:
            raise ValueError("A prepared dataset is required.")

        result = self.get_reconstruction(
            dataset=dataset,
            indices=indices,
            input_mode=input_mode,
            random_seed=random_seed,
        )
        value_mode = resolve_plot_value_mode(
            dataset, coordinate_space, plot_value_mode
        )
        bin_edges = dataset.get_bin_edges(coordinate_space=coordinate_space)

        if "axis_labels" not in plot_params:
            plot_params["axis_labels"] = [
                f"raw dimension {axis + 1}"
                if coordinate_space == "raw"
                else f"transformed dimension {axis + 1}"
                for axis in range(dataset.ndim)
            ]
        if "value_label" not in plot_params:
            if value_mode == "density":
                plot_params["value_label"] = (
                    "probability density per raw unit"
                    if coordinate_space == "raw"
                    else "probability density per transformed unit"
                )
            elif dataset.histogram_mode == "probability_mass":
                plot_params["value_label"] = "probability mass per bin"
            else:
                plot_params["value_label"] = "stored histogram value"

        metric_values = None
        metric_name = None
        if (
                dataset.histogram_mode == "probability_mass"
                and self.config.get("reconstruction_loss") == "forward_kl"
                ):
            metric_values = _forward_kl_per_sample(
                result["target"], result["reconstruction"]
            )
            metric_name = "forward KL"

        figure, axes = plot_reconstruction_grid(
            target=result["target"],
            reconstruction=result["reconstruction"],
            input_hist=(result["input"] if input_mode == "sampled" else None),
            bin_edges=bin_edges,
            value_mode=value_mode,
            group_labels=result["groups"],
            metric_values=metric_values,
            metric_name=metric_name,
            output=output,
            show=show,
            close=close,
            **plot_params,
        )
        result["coordinate_space"] = coordinate_space
        result["plot_value_mode"] = value_mode
        result["bin_edges"] = bin_edges
        result["metric_name"] = metric_name
        result["metric_values"] = metric_values
        return result, figure, axes


    def check_data(
            self, dataset, indices=None, output="", coordinate_space="raw",
            plot_value_mode="auto", **plot_params
            ):
        """Plot deterministic full-group histograms in explicit coordinates.

        Raw coordinates are used by default. Probability-mass histograms with
        unequal raw-width bins are displayed as raw-coordinate density when
        ``plot_value_mode="auto"``.
        """
        if indices is None or len(indices) == 0:
            indices = list(range(len(dataset)))
        else:
            indices = [int(index) for index in indices]
        if any(index < 0 or index >= len(dataset) for index in indices):
            raise IndexError("indices contains an out-of-range dataset index.")

        hist_list = [
            dataset.get_full_histogram(index).numpy()[0]
            for index in indices
        ]
        value_mode = resolve_plot_value_mode(
            dataset, coordinate_space, plot_value_mode
        )
        bin_edges = dataset.get_bin_edges(coordinate_space=coordinate_space)
        coordinate_prefix = (
            "raw" if coordinate_space == "raw" else "transformed"
        )
        plot_params.setdefault(
            "title_list", [str(dataset.idx2group[index]) for index in indices]
        )
        plot_params.setdefault("xlabel", f"{coordinate_prefix} dimension 1")
        if dataset.ndim == 1:
            if value_mode == "density":
                plot_params.setdefault(
                    "ylabel",
                    (
                        "probability density per raw unit"
                        if coordinate_space == "raw"
                        else "probability density per transformed unit"
                    ),
                )
            elif dataset.histogram_mode == "probability_mass":
                plot_params.setdefault("ylabel", "probability mass per bin")
            else:
                plot_params.setdefault("ylabel", "stored histogram value")
        else:
            plot_params.setdefault("ylabel", f"{coordinate_prefix} dimension 2")
            plot_params.setdefault(
                "colorbar_label",
                (
                    (
                        "probability density per raw area"
                        if coordinate_space == "raw"
                        else "probability density per transformed area"
                    )
                    if value_mode == "density"
                    else "histogram value"
                ),
            )
        return plot_histogram_grid(
            hist_list,
            output=output,
            bin_edges=bin_edges,
            value_mode=value_mode,
            **plot_params,
        )


    def qual_eval(self, dataset, query_indices, outdir:str=""):
        """
        qualitative evaluation
        
        Parameters
        ----------
        dataset: torch.utils.data.Dataset
            the PHTwins dataset

        indices: list
            the list of indices to be checked
        
        """
        # get representations
        reps = self.get_latent(dataset) # default: train dataset
        # query data
        query_reps = reps[query_indices]
        # calculate cosine similarity
        norm_query = np.linalg.norm(query_reps, axis=1, keepdims=True)
        norm_reps = np.linalg.norm(reps, axis=1)
        norm_query[norm_query == 0] = 1e-10
        norm_reps[norm_reps == 0] = 1e-10
        sim_matrix = np.dot(query_reps, reps.T) / (norm_query * norm_reps)
        # plot query, most similar, and least similar
        for i, idx in enumerate(query_indices):
            output = os.path.join(outdir, f"qual_eval_{i}.tif")
            indices = np.argsort(sim_matrix[i])[::-1]
            plot_indices = [idx] + [indices[0]] + [indices[-1]]
            plot_params = {
                "title_list": ["query", "most similar", "least similar"],
                "nrow": 1,
                "ncol": 3,
                }
            self.check_data(dataset, plot_indices, output, **plot_params)
            # nrows, ncols = 1, 3 (query / most similar / least similar)
        return sim_matrix


class Preprocess:
    def __init__(
            self, key_data, key_group, key_label=None
            ):
        """
        Parameters
        ----------
        key_data: list
            the keys for the data

        key_group: str
            the key to identify the data

        key_label: int
            the key for the label
            note that the label should be integer
        
        """
        self.key_data = key_data
        self.key_group = key_group
        self.key_label = key_label
        self.lut = None
        self.num_data = None
        self.data = None
        self.group = None
        self.label = None


    def fit_transform(self, df):
        """
        preprocess the data

        Parameters
        ----------
        df: pd.DataFrame
            the data to be preprocessed

        Returns
        -------
        data: np.ndarray
            preprocessed data
        
        group: np.ndarray
            preprocessed group

        label: np.ndarray
            preprocessed label

        """
        # prepare meta data and converter
        # group and label are assumed to be 1DS
        # label is converted to integer
        if self.key_label is None:
            group = df[self.key_group].values
            unique_group = np.unique(group)
            self.lut = pd.DataFrame(
                {"raw_index":list(range(len(unique_group))), "group": unique_group}
                )
            self.lut.loc[:, "label0"] = None
            self.lut.loc[:, "label"] = None
            converted_label = None
        else:
            # convert label to integer
            label0 = list(df[self.key_label].unique())
            label1 = list(range(len(label0)))
            label_encoder = dict(zip(label0, label1))
            converted_label = np.array([label_encoder[k] for k in list(df[self.key_label])])
            # store lookup table
            group2label = dict(zip(list(df[self.key_group]), list(df[self.key_label])))
            self.lut = pd.DataFrame(
                {
                    "raw_index":list(range(len(group2label))),
                    "group": list(group2label.keys()),
                    "label0": list(group2label.values())}
                )
            self.lut["label"] = self.lut["label0"].map(label_encoder)
        dic_group = dict(zip(list(self.lut["group"]), list(self.lut["raw_index"])))
        converted_group = np.array([dic_group[k] for k in list(df[self.key_group])])
        self.num_data = self.lut.shape[0] # number of data
        # data
        data = df[self.key_data].values
        data = data.astype(np.float32)
        # register
        self.data = data
        self.group = converted_group
        self.label = converted_label
        return data, converted_group, converted_label


    def get_lut(self) -> pd.DataFrame:
        """
        get lookup table
                
        """
        assert self.lut is not None, "!! fit_transform first !!"
        return self.lut
    

    def check_transform(
            self, raw_data, group, indices:list=[], num_points:int=4096, bins:int=64,
            histogram_mode="count", value_transform="none", **plot_params
            ):
        """
        check transform

        Parameters
        ----------
        raw_data: np.ndarray
            the data to be checked

        indices: list
            the list of indices to be checked

        """
        list_hist = []
        list_raw = []
        list_title = []
        for idx in indices:
            # raw data
            mask = np.where(group == idx)[0]
            raw = raw_data[mask]
            # converted data
            hist = self.to_hist(
                raw_data, group, idx, num_points=num_points, bins=bins,
                histogram_mode=histogram_mode,
                value_transform=value_transform,
                )
            # summary
            list_raw.append(raw)
            list_hist.append(hist)
            list_title.append(f"Group_{idx}")
        # plot
        plot_scatter(points_list=list_raw, title_list=list_title, **plot_params)
        plot_hist(hist_list=list_hist, title_list=list_title, **plot_params)


    def to_hist(
            self, raw_data, group, idx:int, num_points:int=4096, bins=64,
            histogram_mode="count", value_transform="none"
            ):
        """
        convert to histogram

        """
        selected_indices = np.where(group == idx)[0]
        pointcloud = raw_data[selected_indices]
        if pointcloud.shape[0] > num_points:
            idxs0 = np.random.choice(pointcloud.shape[0], num_points, replace=False)
            pointcloud0 = pointcloud[idxs0, :]
        else:
            idxs0 = np.random.choice(pointcloud.shape[0], num_points, replace=True)
            pointcloud0 = pointcloud[idxs0, :]
        # prepare histogram
        hist0 = calc_hist(
            pointcloud0, bins=bins, histogram_mode=histogram_mode,
            value_transform=value_transform,
            )
        if histogram_mode == "probability_mass":
            return hist0
        # normalize the histogram
        hist0 = np.log1p(hist0) # log1p for numerical stability
        tmp = np.max(hist0) # store the max value for normalization
        hist0 = hist0 / tmp # normalize
        return hist0


def calc_hist(X, bins=16, histogram_mode="count", value_transform="none"):
    histogram_mode = validate_histogram_mode(histogram_mode)
    value_transform = validate_value_transform(value_transform)
    values = np.asarray(X)
    if value_transform == "log1p":
        if np.any(~np.isfinite(values)) or np.any(values < 0):
            raise ValueError("value_transform='log1p' requires finite non-negative data.")
        values = np.log1p(values)
    density = histogram_mode == "density"
    with np.errstate(divide="ignore", invalid="ignore"):
        try:
            s = values.shape[1]
        except IndexError:
            s = 1
        if s == 1:
            hist, _ = np.histogram(values, bins=bins, density=density)
        elif s == 2:
            hist, _, _ = np.histogram2d(
                values[:, 0], values[:, 1], bins=bins, density=density
                )
        elif s == 3:
            hist, _ = np.histogramdd(values, bins=bins, density=density)
        else:
            raise ValueError("!! Input array must be 1D, 2D, or 3D. !!")
    if histogram_mode == "probability_mass":
        total = hist.sum()
        if total <= 0:
            raise ValueError("Probability-mass histogram is undefined for empty input data.")
        hist = hist.astype(np.float64, copy=False) / total
    if density and not np.all(np.isfinite(hist)):
        raise ValueError("Density histogram is undefined for empty input data.")
    return hist


def plot_hist(hist_list, output="", show=False, **plot_params):
    """Compatibility wrapper for :func:`histvae.visualization.plot_hist`."""
    return plot_histogram_grid(
        hist_list, output=output, show=show, **plot_params
    )


def plot_scatter(points_list, output="", show=False, **plot_params):
    """Compatibility wrapper for :func:`histvae.visualization.plot_scatter`."""
    return plot_scatter_grid(
        points_list, output=output, show=show, **plot_params
    )
