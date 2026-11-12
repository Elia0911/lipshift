"""Tests for video decoding and frame-rate normalization."""

from pathlib import Path

import numpy as np
import pytest

from lipshift.extract.video import load_video_rgb, resample_to_fps

cv2 = pytest.importorskip("cv2")


def _write_clip(path: Path, n: int, fps: float, rgb: tuple[int, int, int]) -> None:
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (64, 48))
    frame_bgr = np.zeros((48, 64, 3), np.uint8)
    frame_bgr[:] = rgb[::-1]
    for _ in range(n):
        w.write(frame_bgr)
    w.release()


def test_load_video_returns_rgb_frames(tmp_path: Path) -> None:
    clip = tmp_path / "red.mp4"
    _write_clip(clip, n=10, fps=25.0, rgb=(220, 20, 20))
    frames, fps = load_video_rgb(clip)
    assert frames.shape == (10, 48, 64, 3)
    assert frames.dtype == np.uint8
    assert fps == pytest.approx(25.0, abs=0.01)
    mean = frames.reshape(-1, 3).mean(axis=0)
    assert mean[0] > 150 and mean[2] < 80  # channel 0 is red, so order is RGB


def test_load_video_max_frames(tmp_path: Path) -> None:
    clip = tmp_path / "c.mp4"
    _write_clip(clip, n=20, fps=25.0, rgb=(0, 0, 0))
    frames, _ = load_video_rgb(clip, max_frames=7)
    assert frames.shape[0] == 7


def test_load_video_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(OSError):
        load_video_rgb(tmp_path / "nope.mp4")


@pytest.mark.parametrize(
    ("n", "fps", "expected"),
    [(50, 25.0, 50), (60, 30.0, 50), (60, 29.97, 50), (168, 23.976, 175)],
)
def test_resample_to_fps_lengths(n: int, fps: float, expected: int) -> None:
    out = resample_to_fps(np.arange(n), fps, 25.0)
    assert len(out) == expected


def test_resample_keeps_first_and_last_frame_and_order() -> None:
    out = resample_to_fps(np.arange(60), 30.0, 25.0)
    assert out[0] == 0 and out[-1] == 59
    assert np.all(np.diff(out) >= 0)


def test_resample_identity_returns_same_object() -> None:
    frames = np.arange(10)
    assert resample_to_fps(frames, 25.0) is frames


def test_resample_rejects_bad_fps() -> None:
    with pytest.raises(ValueError):
        resample_to_fps(np.arange(10), 0.0)
