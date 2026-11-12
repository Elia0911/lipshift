"""Video decoding and frame-rate normalization."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def load_video_rgb(path: str | Path, max_frames: int | None = None) -> tuple[np.ndarray, float]:
    """Read a video into an RGB uint8 array of shape (T, H, W, 3).

    OpenCV is used because ``torchvision.io.read_video`` is not present in torchvision 0.29.

    Args:
        path: Video file path.
        max_frames: Stop after this many frames (None reads all).

    Returns:
        Frames and the frame rate reported by the container (25.0 if unreported).
    """
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise OSError(f"Cannot open video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
    frames: list[np.ndarray] = []
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            if max_frames is not None and len(frames) >= max_frames:
                break
    finally:
        cap.release()
    if not frames:
        raise ValueError(f"No frames decoded from: {path}")
    return np.stack(frames), fps


def resample_to_fps(frames: np.ndarray, fps: float, target_fps: float = 25.0) -> np.ndarray:
    """Select frames on an even grid so the clip plays at ``target_fps``.

    Same rule as the multilingual VSR repository: ``speed_rate = fps / target_fps`` and
    ``torch.linspace`` index selection. Lower input rates duplicate frames; higher rates
    drop frames.
    """
    if fps <= 0 or target_fps <= 0:
        raise ValueError("fps and target_fps must be positive")
    speed_rate = fps / target_fps
    if abs(speed_rate - 1.0) < 1e-3:
        return frames
    n_out = int(frames.shape[0] / speed_rate)
    if n_out < 1:
        raise ValueError(f"Clip too short to resample: {frames.shape[0]} frames at {fps} fps")
    idx = torch.linspace(0, frames.shape[0] - 1, n_out, dtype=torch.int64).numpy()
    return frames[idx]
