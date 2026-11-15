"""One clip: video -> 25 fps -> mouth crops -> frozen VSR features."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lipshift.extract.crop import crop_mouths
from lipshift.extract.video import load_video_rgb, resample_to_fps


def prepare_clip(
    video_path: str | Path,
    auto_avsr_dir: str | Path,
    face_model_path: str | Path,
    max_frames: int | None = 300,
    target_fps: float = 25.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Decode, resample to ``target_fps`` and crop mouths: (T, 96, 96, 3) uint8 and timings."""
    t0 = time.perf_counter()
    frames, fps = load_video_rgb(video_path, max_frames=max_frames)
    frames = resample_to_fps(frames, fps, target_fps)
    t1 = time.perf_counter()
    mouths = crop_mouths(frames, auto_avsr_dir, face_model_path)
    t2 = time.perf_counter()
    info = {
        "fps_in": round(fps, 3),
        "n_frames_25fps": int(frames.shape[0]),
        "sec_decode": round(t1 - t0, 3),
        "sec_crop": round(t2 - t1, 3),
    }
    return mouths, info


def encode_mouths(encoder: torch.nn.Module, mouths: np.ndarray, device: str = "cpu") -> np.ndarray:
    """Run one encoder on mouth crops; returns (T, D) float16.

    ``encoder`` must provide ``preprocess(mouths_rgb)`` and ``forward(x)``, as both classes
    in :mod:`lipshift.extract.encoders` do.
    """
    x = encoder.preprocess(mouths).to(device)
    return encoder.to(device).eval()(x).float().cpu().numpy().astype(np.float16)


def extract_clip(
    video_path: str | Path,
    encoder: torch.nn.Module,
    auto_avsr_dir: str | Path,
    face_model_path: str | Path,
    device: str = "cpu",
    max_frames: int | None = 300,
    target_fps: float = 25.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return (T, D) float16 features and timing info for one clip and one encoder."""
    mouths, info = prepare_clip(video_path, auto_avsr_dir, face_model_path, max_frames, target_fps)
    t0 = time.perf_counter()
    feats = encode_mouths(encoder, mouths, device)
    info["sec_encode"] = round(time.perf_counter() - t0, 3)
    return feats, info
