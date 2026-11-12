"""Tests for import isolation between upstream repositories."""

import importlib
import sys
from pathlib import Path

import pytest

from lipshift.extract.repos import switch_repo


def _fake_repo(root: Path, value: str) -> Path:
    pkg = root / "espnet"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text(f"VALUE = {value!r}\n")
    return root


@pytest.fixture
def clean_sys_path():
    saved_path = list(sys.path)
    yield
    sys.path[:] = saved_path
    for name in list(sys.modules):
        if name.split(".")[0] == "espnet":
            del sys.modules[name]


def test_switch_repo_swaps_same_named_packages(tmp_path: Path, clean_sys_path) -> None:
    a = _fake_repo(tmp_path / "a", "A")
    b = _fake_repo(tmp_path / "b", "B")
    switch_repo(a)
    assert importlib.import_module("espnet").VALUE == "A"
    switch_repo(b)
    assert importlib.import_module("espnet").VALUE == "B"
    switch_repo(a)
    assert importlib.import_module("espnet").VALUE == "A"


def test_switch_repo_does_not_duplicate_path_entries(tmp_path: Path, clean_sys_path) -> None:
    a = _fake_repo(tmp_path / "a", "A")
    switch_repo(a)
    switch_repo(a)
    assert sys.path.count(str(a.resolve())) == 1
