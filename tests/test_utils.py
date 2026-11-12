"""Tests for seeding and config loading."""

import random
from pathlib import Path

import numpy as np
import pytest
import torch

from lipshift.utils import load_config, seed_everything


def test_seed_everything_reproducible() -> None:
    seed_everything(7)
    a = (random.random(), np.random.rand(), torch.rand(1).item())
    seed_everything(7)
    b = (random.random(), np.random.rand(), torch.rand(1).item())
    assert a == b


def test_load_config(tmp_path: Path) -> None:
    p = tmp_path / "c.yaml"
    p.write_text("seed: 3\nper_cell: 10\n")
    assert load_config(p) == {"seed": 3, "per_cell": 10}
    p.write_text("- just\n- a list\n")
    with pytest.raises(ValueError):
        load_config(p)


def test_repo_configs_parse() -> None:
    root = Path(__file__).resolve().parents[1] / "configs"
    for cfg in root.glob("*.yaml"):
        assert isinstance(load_config(cfg), dict)
