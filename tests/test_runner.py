"""Tests for resumable extraction."""

import json
from pathlib import Path

import numpy as np
import pytest

from lipshift.extract.runner import feature_filename, run_extraction, video_path_from_filename


def _fake_feature_fn(path: str):
    if "bad" in path:
        raise RuntimeError("no face")
    return np.ones((5, 4), np.float16), {"sec_crop": 0.1}


def _read_log(out: Path) -> list[dict]:
    return [json.loads(line) for line in (out / "log.jsonl").read_text().splitlines()]


@pytest.mark.parametrize(
    "vp",
    [
        "arabic/inswapper/02690.png_Po82BhllEjA_340_1.mp4.mp4",
        "english/real/a__b.mp4",
        "english__real/a.mp4",
        "x/100%2F.mp4",
    ],
)
def test_feature_filename_is_flat_and_reversible(vp: str) -> None:
    name = feature_filename(vp)
    assert "/" not in name and name.endswith(".npy")
    assert video_path_from_filename(name) == vp


def test_feature_filename_has_no_collisions() -> None:
    paths = ["english/real/a.mp4", "english__real/a.mp4", "english%2Freal/a.mp4"]
    assert len({feature_filename(p) for p in paths}) == len(paths)


def test_extraction_writes_skips_and_logs(tmp_path: Path) -> None:
    ids = ["en/real/1.mp4", "en/fake/bad.mp4", "es/real/2.mp4"]
    first = run_extraction(ids, resolve=lambda x: x, feature_fn=_fake_feature_fn, out_dir=tmp_path)
    assert first == {"done": 2, "skipped": 0, "failed": 1}
    arr = np.load(tmp_path / feature_filename(ids[0]))
    assert arr.shape == (5, 4) and arr.dtype == np.float16
    assert not list(tmp_path.glob("*.tmp"))

    second = run_extraction(ids, resolve=lambda x: x, feature_fn=_fake_feature_fn, out_dir=tmp_path)
    assert second == {"done": 0, "skipped": 2, "failed": 1}  # failed clip is retried

    log = _read_log(tmp_path)
    assert [r["status"] for r in log] == ["ok", "error", "ok", "error"]
    assert "no face" in log[1]["error"] and log[0]["shape"] == [5, 4]


def test_max_errors_stops_early(tmp_path: Path) -> None:
    ids = ["a/bad1.mp4", "a/bad2.mp4", "a/ok.mp4"]
    counts = run_extraction(ids, lambda x: x, _fake_feature_fn, tmp_path, max_errors=2)
    assert counts == {"done": 0, "skipped": 0, "failed": 2}
