# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 12:09:08 2019

utils

@author: tadahaya
"""
import json
import os
import random
import time
from collections.abc import Mapping
from functools import partial
from importlib.resources import files
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml


def _seed_worker(worker_id, base_seed):
    worker_seed = base_seed + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def fix_seed(seed: int=42, fix_cuda: bool=False):
    """
    fix the seed for reproducibility

    Parameters:
    ----------
    seed : int
        the seed number

    """
    # general seed
    random.seed(seed)  # Python random
    np.random.seed(seed)  # NumPy random
    torch.manual_seed(seed)  # PyTorch CPU seed
    torch.cuda.manual_seed(seed)  # PyTorch GPU seed
    torch.cuda.manual_seed_all(seed)  # PyTorch all GPU seed
    # cudnn seed
    if fix_cuda:
        torch.backends.cudnn.deterministic = True  # for fixing calculation order etc.
        torch.backends.cudnn.benchmark = False  # do not use the optimized algorithm
    # prepare worker seed for DataLoader
    g = torch.Generator()
    g.manual_seed(seed)
    return g, partial(_seed_worker, base_seed=seed)  # for DataLoader workers



def get_default_config_path():
    """Return the packaged default config.yaml path."""
    return str(files("histvae").joinpath("config.yaml"))



def load_default_config():
    """Load the packaged default config.yaml."""
    config_path = get_default_config_path()
    with open(config_path, "r") as f:
        config = yaml.safe_load(f) or {}
    return config, config_path



def merge_config(base: dict, override: dict | None):
    """Recursively merge config dictionaries without mutating the inputs."""
    merged = dict(base)
    if not override:
        return merged
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = merge_config(merged[key], value)
        else:
            merged[key] = value
    return merged



def load_config(config_path: str=None, overrides: dict | None=None):
    """
    Load config with layered overrides.

    Order:
    1. packaged default config
    2. optional user-provided YAML config
    3. optional runtime overrides
    """
    config, default_config_path = load_default_config()
    resolved_user_config = None

    if config_path is not None:
        with open(config_path, "r") as f:
            user_config = yaml.safe_load(f) or {}
        config = merge_config(config, user_config)
        resolved_user_config = str(config_path)

    if overrides:
        config = merge_config(config, overrides)

    meta = {
        "default_config_path": default_config_path,
        "user_config_path": resolved_user_config,
    }
    return config, meta



def load_yaml_config(config_path: str=None):
    """Backward-compatible wrapper around load_config()."""
    config, meta = load_config(config_path=config_path)
    resolved_path = meta["user_config_path"] or meta["default_config_path"]
    return config, resolved_path



def _to_builtin(value):
    if isinstance(value, Mapping):
        return {key: _to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (Path, torch.device)):
        return str(value)
    return value


def save_experiment(
        config, model, optimizer, history, outdir, plot_progress=True,
        checkpoint_metadata=None
        ):
    """
    save the experiment: config, model, metrics, and progress plot
    
    outdir
    ├── experiment_name (resdir)
        ├── config.yaml
        ├── history.json
        ├── progress_loss.tif
        ├── model_best.pt
        ├── model_last.pt
        ├── model_1.pt
        ├── model_2.pt
        ├── ...
    
    """
    os.makedirs(outdir, exist_ok=True)
    # save config
    configfile = os.path.join(outdir, 'config.yaml')
    with open(configfile, 'w') as f:
        yaml.safe_dump(
            _to_builtin(config), f, default_flow_style=False, sort_keys=False
        )
    # save history
    historyfile = os.path.join(outdir, 'history.json')
    with open(historyfile, 'w') as f:
        json.dump(_to_builtin(history), f, sort_keys=True, indent=4)
    # save the model
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        name="best",
        outdir=outdir,
        metadata=checkpoint_metadata,
    )
    # plot progress
    if plot_progress:
        progress_plot(
            outdir=outdir,
            train_values=history["train_loss"],
            test_values=history["test_loss"]
        )



def save_checkpoint(model, optimizer, name, outdir, metadata=None):
    """
    save the model checkpoint
    
    """
    cpfile = os.path.join(outdir, f"model_{name}.pt")
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    if metadata:
        payload.update(_to_builtin(metadata))
    torch.save(payload, cpfile)



def load_experiments(model, optimizer, resdir, checkpoint_name="model_best.pt"):
    """
    load the experiment

    Parameters
    ----------
    model: nn.Module
        initialized model

    optimizer: torch.optim
        initialized optimizer

    resdir: str
        the result directory
    
    checkpoint_name: str
        the checkpoint file name, such as model_best.pt
    
    """
    # load config
    configfile = os.path.join(resdir, "config.yaml")
    with open(configfile, 'r') as f:
        config = yaml.safe_load(f)
    # load history
    historyfile = os.path.join(resdir, 'history.json')
    with open(historyfile, 'r') as f:
        history = json.load(f)
    # load model
    pkg = torch.load(os.path.join(resdir, checkpoint_name))
    model.load_state_dict(pkg["model"])
    optimizer.load_state_dict(pkg["optimizer"])
    return model, optimizer, config, history



def progress_plot(
        outdir:str, train_values:list, test_values:list=[],
        xlabel="epoch", ylabel="loss", fontsize=14
        ):
    """ plot learning progress """
    fileout = os.path.join(outdir, f"progress_{ylabel}.tif")
    x = list(range(1, len(train_values) + 1, 1))
    fig, ax = plt.subplots()
    plt.rcParams['font.size'] = fontsize
    ax.plot(x, train_values, c='navy', label='train')
    if len(test_values) > 0:
        ax.plot(x, test_values, c='darkgoldenrod', label='test')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid()
    ax.legend()
    plt.tight_layout()
    plt.savefig(fileout, dpi=300, bbox_inches='tight')
    plt.close(fig)



def calc_elapsed_time(start_time):
    """ calculate elapsed time """
    elapsed_time = time.time() - start_time
    h = int(elapsed_time // 3600)
    m = int((elapsed_time % 3600) // 60)
    s = elapsed_time % 60
    return f"{h}h {m}m {s}s"
