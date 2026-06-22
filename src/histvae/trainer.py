# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 12:09:08 2019

trainer

@author: tadahaya
"""
import copy
import math
import os, time
from numbers import Integral
from typing import List, Union, Any
import torch
from torch.nn.utils import clip_grad_norm_

from .utils import save_experiment, save_checkpoint, calc_elapsed_time


LATENT_KL_SCHEDULES = ("constant", "linear_warmup")


def validate_latent_kl_schedule(config):
    """Normalize and validate the pretraining latent-KL schedule in-place."""
    schedule = config.get("latent_kl_schedule", "constant")
    if schedule not in LATENT_KL_SCHEDULES:
        raise ValueError(
            f"Unsupported latent_kl_schedule: {schedule!r}. "
            "Use 'constant' or 'linear_warmup'."
        )

    beta_value = config.get("beta", 1.0)
    if isinstance(beta_value, bool):
        raise ValueError("beta must be finite and non-negative.")
    try:
        beta = float(beta_value)
    except (TypeError, ValueError) as exc:
        raise ValueError("beta must be finite and non-negative.") from exc
    if not math.isfinite(beta) or beta < 0:
        raise ValueError("beta must be finite and non-negative.")

    warmup_epochs = config.get("latent_kl_warmup_epochs", 0)
    if (
            isinstance(warmup_epochs, bool)
            or not isinstance(warmup_epochs, Integral)
            ):
        raise ValueError("latent_kl_warmup_epochs must be an integer.")
    warmup_epochs = int(warmup_epochs)

    if schedule == "constant":
        if warmup_epochs != 0:
            raise ValueError(
                "latent_kl_warmup_epochs must be 0 when "
                "latent_kl_schedule='constant'."
            )
    else:
        epochs = config.get("epochs")
        if (
                isinstance(epochs, bool)
                or not isinstance(epochs, Integral)
                or epochs < 1
                ):
            raise ValueError(
                "epochs must be a positive integer when "
                "latent_kl_schedule='linear_warmup'."
            )
        if beta <= 0:
            raise ValueError(
                "latent_kl_schedule='linear_warmup' requires beta > 0."
            )
        if warmup_epochs < 2:
            raise ValueError(
                "latent_kl_schedule='linear_warmup' requires "
                "latent_kl_warmup_epochs >= 2."
            )
        if warmup_epochs > epochs:
            raise ValueError(
                "latent_kl_warmup_epochs must not exceed epochs."
            )

    config["beta"] = beta
    config["latent_kl_schedule"] = schedule
    config["latent_kl_warmup_epochs"] = warmup_epochs
    return config


def latent_kl_weight(config, epoch):
    """Return the latent-KL weight for a one-indexed training epoch."""
    validate_latent_kl_schedule(config)
    if isinstance(epoch, bool) or not isinstance(epoch, Integral) or epoch < 1:
        raise ValueError("epoch must be a positive integer.")
    epoch = int(epoch)

    beta = config["beta"]
    if config["latent_kl_schedule"] == "constant":
        return beta

    warmup_epochs = config["latent_kl_warmup_epochs"]
    progress = min((epoch - 1) / (warmup_epochs - 1), 1.0)
    return beta * progress


def latent_kl_monitor_start_epoch(config):
    """Return the first epoch eligible for best-checkpoint monitoring."""
    validate_latent_kl_schedule(config)
    if config["latent_kl_schedule"] == "constant":
        return 1
    return config["latent_kl_warmup_epochs"]


def _clone_to_cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _clone_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_to_cpu(item) for item in value)
    return copy.deepcopy(value)


def _set_optimizer_mode(optimizer, training):
    method = getattr(optimizer, "train" if training else "eval", None)
    if callable(method):
        method()


def _prepare_histogram_batch(data, device, condition_mode):
    if condition_mode == "none":
        if len(data) != 2:
            raise ValueError(
                "condition_mode='none' expects (hist0, hist1) batches."
            )
        hist0, hist1 = (value.to(device) for value in data)
        return hist0, hist1, None
    if condition_mode == "decoder":
        if len(data) != 3:
            raise ValueError(
                "condition_mode='decoder' expects "
                "(hist0, hist1, condition) batches."
            )
        hist0, hist1, condition = (value.to(device) for value in data)
        return hist0, hist1, condition
    raise ValueError(f"Unsupported condition_mode: {condition_mode!r}.")

class BaseTrainer:
    def __init__(self):
        self.callbacks: List[Any] = []

    def train(self):
        """ train the model """
        raise NotImplementedError
    
    def train_epoch(self):
        """ train the model for one epoch """
        raise NotImplementedError

    def evaluate(self):
        """ evaluate the model """
        raise NotImplementedError

    def set_callbacks(self, callbacks: Union[List[Any], Any]):
        """
        Parameters
        ----------
        callbacks: list, instance
            list of callback instances or a single callback instance

        """
        if not isinstance(callbacks, list):
            callbacks = [callbacks]
        for callback in callbacks:
            if not callable(callback):
                raise ValueError("!! Callbacks must be callable instances. !!")
            else:
                self.callbacks.append(callback)

    def run_callbacks(self, **kwargs):
        """
        Parameters
        ----------
        kwargs: dict
            keyword arguments to pass to the callbacks

        """        
        for callback in self.callbacks:
            callback(**kwargs)


class BaseLogger:
    def __init__(self):
        self.items = []

    def __call__(self, **kwargs):
        """ Args: kwargs (dict): keyword arguments """
        self.items.append(kwargs)

    def get_items(self):
        """ get items as a dict """
        items = {}
        for item in self.items:
            for k, v in item.items():
                if k not in items:
                    items[k] = []
                items[k].append(v)
        return items


class EarlyStopping:
    def __init__(self, patience=10, mode="min", restore_best_model=True, verbose=True):
        """
        Parameters
        ----------
        patience: int
            number of epochs with no improvement after which training will be stopped

        mode: str
            one of {min, max}.

        restore_best_model: bool
            whether to restore model weights from the epoch with the best score

        """
        if patience is not None and patience < 0:
            raise ValueError("patience must be non-negative or None.")
        if mode not in {"min", "max"}:
            raise ValueError("mode must be 'min' or 'max'.")
        self.patience = patience
        self.restore_best_model = restore_best_model
        self.verbose = verbose
        self.best_score = None
        self.best_epoch = None
        self.counter = 0
        self.early_stop = False
        self.best_model_state = None
        self.best_optimizer_state = None
        self._monitor_fxn = {
            "min": lambda a, b: a < b,
            "max": lambda a, b: a > b
        }[mode]


    def __call__(self, model, score, epoch, optimizer=None):
        """
        Parameters
        ----------
        model: torch.nn.Module
            current model

        score: float
            current score (loss or accuracy)

        epoch: int
            current epoch

        """
        score = float(score)
        if not math.isfinite(score):
            raise ValueError(f"Monitored score must be finite, got {score}.")
        if self.best_score is None or self._monitor_fxn(score, self.best_score):
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            if self.restore_best_model:
                self.best_model_state = _clone_to_cpu(model.state_dict())
                if optimizer is not None:
                    self.best_optimizer_state = _clone_to_cpu(
                        optimizer.state_dict()
                    )
                # store the best model state on CPU
        else:
            self.counter += 1
            if self.patience and self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(">> EarlyStopping triggered")


    def restore(self, model, optimizer=None):
        if not self.restore_best_model or self.best_model_state is None:
            return
        model.load_state_dict(self.best_model_state)
        if optimizer is not None and self.best_optimizer_state is not None:
            optimizer.load_state_dict(self.best_optimizer_state)


class PreTrainer(BaseTrainer):
    def __init__(
            self, config, model, optimizer=None, callbacks=None, outdir:str=""
            ):
        super().__init__()
        # arguments
        self.config = config
        validate_latent_kl_schedule(self.config)
        self.latent_kl_schedule = self.config["latent_kl_schedule"]
        self.current_beta = latent_kl_weight(self.config, epoch=1)
        self.monitor_start_epoch = latent_kl_monitor_start_epoch(self.config)
        self.device = config.get(
            "device", "cuda" if torch.cuda.is_available() else "cpu"
            )
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.logger = BaseLogger()
        self.callbacks = list(callbacks or [])
        self.callbacks.append(self.logger)
        self.outdir = outdir
        # config contents
        self.exp_name = config["exp_name"]
        self.save_model_every = config["save_model_every"]
        self.log_every = config["log_every"]
        self.condition_mode = config.get("condition_mode", "none")
        # I/O
        self.resdir = os.path.join(self.outdir, self.exp_name)
        os.makedirs(self.resdir, exist_ok=True)
        # early stopping
        self.monitor_metric = config.get("pretrain_monitor", "test_loss")
        if self.monitor_metric not in {"test_loss", "test_recon", "test_kl"}:
            raise ValueError(
                "pretrain_monitor must be 'test_loss', 'test_recon', or 'test_kl'."
            )
        if (
                self.latent_kl_schedule != "constant"
                and self.monitor_metric != "test_recon"
                ):
            raise ValueError(
                "pretrain_monitor='test_recon' is required when "
                "latent_kl_schedule is not constant because scheduled total "
                "loss values are not directly comparable across warmup epochs."
            )
        self.active_latent_threshold = float(
            config.get("active_latent_threshold", 0.01)
        )
        if self.active_latent_threshold < 0:
            raise ValueError("active_latent_threshold must be non-negative.")
        self.early_stopping = EarlyStopping(
            patience=config.get("patience", 0),
            mode=config.get("early_stop_mode", "min"),
        )
        # loggings
        self.history = {
            "best_score": None,
            "early_stop_epoch": None,
            "elapsed_time": None
        }


    def train(self, trainloader, testloader):
        """
        train the model for the specified number of epochs.
        
        """
        start_time = time.time()
        last_epoch = 0
        # training
        for i in range(self.config["epochs"]):
            epoch = i + 1
            self.current_beta = latent_kl_weight(self.config, epoch=epoch)
            train_loss, train_recon, train_kl = self.train_epoch(trainloader)
            (
                test_loss,
                test_recon,
                test_kl,
                test_latent_std_mean,
                test_active_latent_dims,
            ) = self.evaluate(testloader)
            last_epoch = epoch
            monitor_eligible = epoch >= self.monitor_start_epoch
            # logging
            self.run_callbacks(
                epoch=epoch,
                train_loss=train_loss,
                test_loss=test_loss,
                train_recon=train_recon,
                train_kl=train_kl,
                test_recon=test_recon,
                test_kl=test_kl,
                test_latent_std_mean=test_latent_std_mean,
                test_active_latent_dims=test_active_latent_dims,
                beta=self.current_beta,
                monitor_eligible=monitor_eligible,
                )
            if epoch % self.log_every == 0:
                print(
                    f"Epoch: {epoch}, Train loss: {train_loss:.4f}, "
                    f"Test loss: {test_loss:.4f}, Beta: {self.current_beta:.6g}"
                    )
            # early stopping
            monitor_score = {
                "test_loss": test_loss,
                "test_recon": test_recon,
                "test_kl": test_kl,
            }[self.monitor_metric]
            if monitor_eligible:
                self.early_stopping(
                    self.model, monitor_score, epoch, optimizer=self.optimizer
                )
                if self.early_stopping.early_stop:
                    self.history["early_stop_epoch"] = epoch # record the epoch
                    break
            # save the model
            if self.save_model_every > 0 and epoch % self.save_model_every == 0:
                save_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    name=f"epoch_{epoch}",
                    outdir=self.resdir,
                    metadata={"epoch": epoch, "beta": self.current_beta},
                )
        if self.early_stopping.best_epoch is None:
            raise RuntimeError(
                "No epoch was eligible for best-checkpoint monitoring."
            )
        # save the experiment
        save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            name="last",
            outdir=self.resdir,
            metadata={
                "epoch": last_epoch,
                "beta": latent_kl_weight(self.config, epoch=last_epoch),
            },
        )
        self.early_stopping.restore(self.model, self.optimizer)
        elapsed_time = calc_elapsed_time(start_time)
        self.history["elapsed_time"] = elapsed_time
        self.history["monitor_metric"] = self.monitor_metric
        self.history["monitor_start_epoch"] = self.monitor_start_epoch
        self.history["best_score"] = self.early_stopping.best_score
        self.history["best_epoch"] = self.early_stopping.best_epoch
        self.history["best_beta"] = latent_kl_weight(
            self.config, epoch=self.early_stopping.best_epoch
        )
        self.history["last_beta"] = latent_kl_weight(
            self.config, epoch=last_epoch
        )
        self.history.update(self.logger.get_items())
        save_experiment(
            config=self.config,
            model=self.model,
            optimizer=self.optimizer,
            history=self.history,
            outdir=self.resdir,
            checkpoint_metadata={
                "epoch": self.early_stopping.best_epoch,
                "score": self.early_stopping.best_score,
                "monitor_metric": self.monitor_metric,
                "beta": latent_kl_weight(
                    self.config, epoch=self.early_stopping.best_epoch
                ),
            },
            )


    def train_epoch(self, trainloader):
        """ train the model for one epoch """
        self.model.train()
        _set_optimizer_mode(self.optimizer, training=True)
        total_loss = 0.0
        total_recon_loss = 0.0
        total_kl_loss = 0.0
        total_samples = 0 # for averaging the loss
        # initialize the gradients
        self.optimizer.zero_grad()
        for i, (data, label) in enumerate(trainloader):
            # data = (original hist, noisy hist)
            hist0, hist1, condition = _prepare_histogram_batch(
                data, self.device, self.condition_mode
            )
            label = label.to(self.device)
            # forward
            recon, mu, logvar = self.model(
                hist1, condition=condition
                ) # output, mu, logvar
            # loss calculation
            loss, recon_loss, kl_loss = self.model.vae_loss(
                recon, hist0, mu, logvar, beta=self.current_beta
                )
            # note: loss is averaged over the batch
            # backpropagation
            loss.backward()
            # clip the gradients
            if self.config["clip_grad"] > 0:
                clip_grad_norm_(self.model.parameters(), self.config["clip_grad"])
            # update the parameters
            if (i + 1) % self.config["accum_grad"] == 0 or (i + 1) == len(trainloader):
                self.optimizer.step()  # Perform the parameter update
                self.optimizer.zero_grad()  # Reset gradients for the next accumulation
            batch_size = hist0.shape[0]
            total_loss += loss.detach().item() * batch_size
            total_recon_loss += recon_loss.detach().item() * batch_size
            total_kl_loss += kl_loss.detach().item() * batch_size
            total_samples += batch_size
        return total_loss / total_samples, total_recon_loss / total_samples, total_kl_loss / total_samples


    def evaluate(self, testloader):
        self.model.eval()
        _set_optimizer_mode(self.optimizer, training=False)
        total_loss = 0.0
        total_recon_loss = 0.0
        total_kl_loss = 0.0
        total_samples = 0 # for averaging the loss
        latent_means = []
        with torch.no_grad():
            for data, label in testloader:
                hist0, hist1, condition = _prepare_histogram_batch(
                    data, self.device, self.condition_mode
                )
                label = label.to(self.device)
                # forward
                recon, mu, logvar = self.model(
                    hist1, sample_latent=False, condition=condition
                    ) # deterministic validation path
                # loss calculation
                loss, recon_loss, kl_loss = self.model.vae_loss(
                    recon, hist0, mu, logvar, beta=self.current_beta
                    )
                # Loss accumulation
                batch_size = hist0.shape[0]
                total_loss += loss.item() * batch_size # detach() is not necessary
                total_recon_loss += recon_loss.item() * batch_size
                total_kl_loss += kl_loss.item() * batch_size
                total_samples += batch_size
                latent_means.append(mu.detach().cpu())
        latent_means = torch.cat(latent_means, dim=0)
        latent_std = latent_means.std(dim=0, unbiased=False)
        latent_std_mean = float(latent_std.mean())
        active_latent_dims = int(
            (latent_std >= self.active_latent_threshold).sum().item()
        )
        return (
            total_loss / total_samples,
            total_recon_loss / total_samples,
            total_kl_loss / total_samples,
            latent_std_mean,
            active_latent_dims,
        )


class FineTuner(BaseTrainer):
    def __init__(self, config, model, optimizer=None, loss_fn=None, callbacks=None, outdir=""):
        super().__init__()
        # arguments
        self.config = config
        self.device = config.get(
            "device", "cuda" if torch.cuda.is_available() else "cpu"
            )
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.logger = BaseLogger()
        self.callbacks = list(callbacks or [])
        self.callbacks.append(self.logger)
        self.outdir = outdir
        # config contents
        self.exp_name = config["exp_name"]
        self.save_model_every = config["save_model_every"]
        self.log_every = config["log_every"]
        self.condition_mode = config.get("condition_mode", "none")
        if config["frozen"]:
            self.use_pretrain_loss = False # if the model is frozen, pretrain loss is never used
        else:
            self.use_pretrain_loss = config["use_pretrain_loss"]
        # I/O
        self.resdir = os.path.join(self.outdir, self.exp_name)
        os.makedirs(self.resdir, exist_ok=True)
        # early stopping
        self.monitor_metric = config.get("finetune_monitor", "test_loss")
        if self.monitor_metric not in {"test_loss", "test_accuracy"}:
            raise ValueError(
                "finetune_monitor must be 'test_loss' or 'test_accuracy'."
            )
        monitor_mode = "max" if self.monitor_metric == "test_accuracy" else "min"
        self.early_stopping = EarlyStopping(
            patience=config.get("patience", 0),
            mode=monitor_mode,
        )
        # loggings
        self.history = {
            "best_score": None,
            "early_stop_epoch": None,
            "elapsed_time": None
        }


    def train(self, trainloader, testloader):
        """
        train the model for the specified number of epochs.
        
        """
        start_time = time.time()
        last_epoch = 0
        # training
        for i in range(self.config["epochs"]):
            train_loss, train_recon, train_kl, train_acc = self.train_epoch(trainloader)
            test_loss, test_recon, test_kl, test_acc = self.evaluate(testloader)
            last_epoch = i + 1
            # logging
            self.run_callbacks(
                epoch=i + 1,
                train_loss=train_loss,
                test_loss=test_loss,
                train_recon=train_recon,
                train_kl=train_kl,
                test_recon=test_recon,
                test_kl=test_kl,
                train_accuracy=train_acc,
                test_accuracy=test_acc,
                )
            if (i + 1) % self.log_every == 0:
                print(f"Epoch: {i + 1}")
                print(f"  Train loss: {train_loss:.4f}, Test loss: {test_loss:.4f}")
                print(f"  Train accuracy: {train_acc:.4f}, Test accuracy: {test_acc:.4f}")
            # early stopping
            monitor_score = {
                "test_loss": test_loss,
                "test_accuracy": test_acc,
            }[self.monitor_metric]
            self.early_stopping(
                self.model, monitor_score, i + 1, optimizer=self.optimizer
            )
            if self.early_stopping.early_stop:
                self.history["early_stop_epoch"] = i + 1 # record the epoch
                break
            # save the model
            if self.save_model_every > 0 and (i + 1) % self.save_model_every == 0:
                save_checkpoint(model=self.model, optimizer=self.optimizer, name=f"epoch_{i + 1}", outdir=self.resdir)
        # save the experiment
        save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            name="last",
            outdir=self.resdir,
            metadata={"epoch": last_epoch},
        )
        self.early_stopping.restore(self.model, self.optimizer)
        elapsed_time = calc_elapsed_time(start_time)
        self.history["elapsed_time"] = elapsed_time
        self.history["monitor_metric"] = self.monitor_metric
        self.history["best_score"] = self.early_stopping.best_score
        self.history["best_epoch"] = self.early_stopping.best_epoch
        self.history.update(self.logger.get_items())
        save_experiment(
            config=self.config,
            model=self.model,
            optimizer=self.optimizer,
            history=self.history,
            outdir=self.resdir,
            checkpoint_metadata={
                "epoch": self.early_stopping.best_epoch,
                "score": self.early_stopping.best_score,
                "monitor_metric": self.monitor_metric,
            },
            )


    def train_epoch(self, trainloader):
        """ train the model for one epoch """
        self.model.train()
        _set_optimizer_mode(self.optimizer, training=True)
        total_loss = 0.0
        total_pt_loss = 0.0
        total_ft_loss = 0.0
        total_samples = 0 # for averaging the loss
        correct = 0
        # initialize the gradients
        self.optimizer.zero_grad()
        for i, (data, label) in enumerate(trainloader):
            # data = (original hist, noisy hist)
            hist0, hist1, condition = _prepare_histogram_batch(
                data, self.device, self.condition_mode
            )
            label = label.to(self.device)
            # forward/loss calculation
            if self.use_pretrain_loss:
                logits, recon, mu, logvar = self.model(
                    hist1, condition=condition
                    ) # use noisy hist for pretraining
                pt_loss, _, _ = self.model.vae_loss(
                    recon, hist0, mu, logvar, beta=self.config["beta"]
                    )
                # note: ignore the reconstruction loss and kl_loss in logging
                ft_loss = self.loss_fn(logits, label)
                loss = pt_loss + ft_loss
            else:
                logits, recon, mu, logvar = self.model(
                    hist0, condition=condition
                    ) # use original hist
                ft_loss = self.loss_fn(logits, label)
                pt_loss = 0
                loss = ft_loss
            # backpropagation
            loss.backward()
            # clip the gradients
            if self.config["clip_grad"] > 0:
                clip_grad_norm_(self.model.parameters(), self.config["clip_grad"])
            # update the parameters
            if (i + 1) % self.config["accum_grad"] == 0 or (i + 1) == len(trainloader):
                self.optimizer.step()  # Perform the parameter update
                self.optimizer.zero_grad()  # Reset gradients for the next accumulation
            # Loss accumulation
            batch_size = hist0.shape[0]
            total_loss += loss.detach().item() * batch_size
            total_pt_loss += float(pt_loss.detach().item() if torch.is_tensor(pt_loss) else pt_loss) * batch_size
            total_ft_loss += float(ft_loss.detach().item()) * batch_size
            total_samples += batch_size
            # Accuracy calculation (disable gradients for efficiency)
            with torch.no_grad():
                predictions = torch.argmax(logits, dim=1)
                correct += (predictions == label).sum().item()
        return total_loss / total_samples, total_pt_loss / total_samples, total_ft_loss / total_samples, correct / total_samples
            

    def evaluate(self, testloader):
        """Evaluate the model on the test set"""
        self.model.eval()
        _set_optimizer_mode(self.optimizer, training=False)
        total_loss = 0.0
        total_pt_loss = 0.0
        total_ft_loss = 0.0
        total_samples = 0
        correct = 0
        with torch.no_grad():
            for data, label in testloader:
                # Move data to device
                hist0, hist1, condition = _prepare_histogram_batch(
                    data, self.device, self.condition_mode
                )
                label = label.to(self.device)
                # forward/loss calculation
                if self.use_pretrain_loss:
                    logits, recon, mu, logvar = self.model(
                        hist1, sample_latent=False, condition=condition
                        ) # deterministic validation path
                    pt_loss, _, _ = self.model.vae_loss(
                        recon, hist0, mu, logvar, beta=self.config["beta"]
                        )
                    # note: ignore the reconstruction loss and kl_loss in logging
                    ft_loss = self.loss_fn(logits, label)
                    loss = pt_loss + ft_loss
                else:
                    logits, recon, mu, logvar = self.model(
                        hist0, sample_latent=False, condition=condition
                        ) # use original histogram
                    ft_loss = self.loss_fn(logits, label)
                    pt_loss = 0
                    loss = ft_loss
                # Loss accumulation
                batch_size = hist0.shape[0]
                total_loss += loss.item() * batch_size # detach() is not necessary
                total_pt_loss += float(pt_loss.item() if torch.is_tensor(pt_loss) else pt_loss) * batch_size
                total_ft_loss += float(ft_loss.item()) * batch_size
                total_samples += batch_size
                # Accuracy calculation
                predictions = torch.argmax(logits, dim=1)
                correct += int((predictions == label).sum())
        return total_loss / total_samples, total_pt_loss / total_samples, total_ft_loss / total_samples, correct / total_samples
