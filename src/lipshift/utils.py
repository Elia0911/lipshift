"""Seeding, logging and config helpers."""

from __future__ import annotations

import logging
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def get_logger(name: str = "lipshift", level: int = logging.INFO) -> logging.Logger:
    """Return a logger with a single stream handler."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def seed_everything(seed: int, deterministic_torch: bool = False) -> None:
    """Seed Python, NumPy and torch.

    Args:
        seed: Seed value.
        deterministic_torch: Also request deterministic torch kernels. Slower, and some
            CUDA ops have no deterministic version, so results can still differ across
            GPUs and library versions.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic_torch:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file into a dict."""
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config {path} must be a mapping, got {type(cfg).__name__}")
    return cfg
