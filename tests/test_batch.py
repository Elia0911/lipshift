"""Tests for multi-encoder batch extraction (no video, no upstream repos)."""

import json
import threading
from pathlib import Path

import numpy as np
import pytest
import torch

from lipshift.extract.batch import EncoderSpec, run_batch, summarize_progress
from lipshift.extract.runner import feature_filename


class FakeEncoder(torch.nn.Module):
    """Mean over pixels, repeated to ``dim`` channels."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.calls = 0

    def preprocess(self, mouths: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(mouths).float()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return x.mean(dim=(1, 2, 3)).unsqueeze(1).repeat(1, self.dim)


def fake_prepare(path: str):
    if "noface" in path:
        raise AssertionError("Cannot detect any frames in the video")
    return np.ones((6, 96, 96, 3), np.uint8), {"sec_crop": 0.0}


ROWS = [
    {"video_path": "english/real/a.mp4", "language": "english"},
    {"video_path": "spanish/inswapper/b.mp4", "language": "spanish"},
    {"video_path": "arabic/real/c.mp4", "language": "arabic"},
    {"video_path": "english/echomimic/noface.mp4", "language": "english"},
]


@pytest.fixture
def encoders():
    return [
        EncoderSpec("core", FakeEncoder(8)),
        EncoderSpec("ml_en", FakeEncoder(4), frozenset({"english", "spanish"})),
        EncoderSpec("ml_es", FakeEncoder(4), frozenset({"english", "spanish"})),
    ]


def _log(root: Path) -> list[dict]:
    return [json.loads(x) for x in (root / "log.jsonl").read_text().splitlines()]


def test_runs_applicable_encoders_and_resumes(tmp_path: Path, encoders) -> None:
    resolved, cleaned = [], []
    counts = run_batch(
        ROWS,
        resolve=lambda vp: resolved.append(vp) or vp,
        prepare=fake_prepare,
        encoders=encoders,
        out_root=tmp_path,
        cleanup=cleaned.append,
    )
    assert counts == {"done": 3, "skipped": 0, "failed": 1, "stopped": 0}
    a = feature_filename("english/real/a.mp4")
    c = feature_filename("arabic/real/c.mp4")
    assert np.load(tmp_path / "core" / a).shape == (6, 8)
    assert np.load(tmp_path / "ml_es" / a).shape == (6, 4)
    assert (tmp_path / "core" / c).exists()
    assert not (tmp_path / "ml_en" / c).exists()  # language filter
    assert sorted(cleaned) == sorted(resolved)  # every downloaded file is cleaned up
    assert not list(tmp_path.rglob("*.tmp"))

    # second run: only the failed clip is retried
    counts2 = run_batch(ROWS, lambda vp: vp, fake_prepare, encoders, tmp_path)
    assert counts2 == {"done": 0, "skipped": 3, "failed": 1, "stopped": 0}

    progress = summarize_progress(ROWS, encoders, tmp_path)
    assert progress["core"] == {"applicable": 4, "done": 3}
    assert progress["ml_en"] == {"applicable": 3, "done": 2}


def test_only_missing_encoders_run(tmp_path: Path, encoders) -> None:
    run_batch(ROWS[:1], lambda vp: vp, fake_prepare, encoders[:1], tmp_path)
    for e in encoders:
        e.encoder.calls = 0
    run_batch(ROWS[:1], lambda vp: vp, fake_prepare, encoders, tmp_path)
    assert [e.encoder.calls for e in encoders] == [0, 1, 1]
    assert _log(tmp_path)[-1]["encoders"] == ["ml_en", "ml_es"]


def test_time_budget_stops_before_work(tmp_path: Path, encoders) -> None:
    counts = run_batch(ROWS, lambda vp: vp, fake_prepare, encoders, tmp_path, max_minutes=0)
    assert counts["done"] == 0 and counts["stopped"] == 4


def test_max_errors(tmp_path: Path, encoders) -> None:
    rows = [{"video_path": f"english/x/noface{i}.mp4", "language": "english"} for i in range(5)]
    counts = run_batch(rows, lambda vp: vp, fake_prepare, encoders, tmp_path, max_errors=2)
    assert counts["failed"] == 2 and counts["stopped"] == 3


def test_resolve_errors_are_logged_not_raised(tmp_path: Path, encoders) -> None:
    def bad_resolve(vp: str) -> str:
        raise OSError("download failed")

    counts = run_batch(ROWS[:2], bad_resolve, fake_prepare, encoders, tmp_path)
    assert counts["failed"] == 2
    assert all("download failed" in r["error"] for r in _log(tmp_path))


def test_prefetch_runs_downloads_in_background(tmp_path: Path, encoders) -> None:
    threads = set()

    def resolve(vp: str) -> str:
        threads.add(threading.current_thread().name)
        return vp

    run_batch(ROWS[:3], resolve, fake_prepare, encoders, tmp_path, prefetch=2)
    assert threads and threading.main_thread().name not in threads
