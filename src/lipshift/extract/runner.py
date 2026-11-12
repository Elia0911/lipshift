"""Resumable batch feature extraction.

Each clip is written to its own ``.npy`` file through a temporary file and an atomic
rename, so an interrupted Colab session never leaves a truncated feature file. Clips
whose output already exists are skipped. Every attempt is appended to ``log.jsonl``.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import numpy as np

from lipshift.utils import get_logger

logger = get_logger(__name__)

FeatureFn = Callable[[str], tuple[np.ndarray, dict[str, Any]]]


def feature_filename(video_path: str) -> str:
    """Map a dataset-relative video path to a flat ``.npy`` file name, reversibly.

    ``%`` is escaped first, then folder separators become ``%2F``, so two different
    paths can never map to the same name (YouTube ids may contain ``__``).
    """
    norm = video_path.replace("\\", "/")
    return norm.replace("%", "%25").replace("/", "%2F") + ".npy"


def video_path_from_filename(name: str) -> str:
    """Inverse of :func:`feature_filename`."""
    stem = name[: -len(".npy")] if name.endswith(".npy") else name
    return stem.replace("%2F", "/").replace("%25", "%")


def save_atomic(path: Path, array: np.ndarray) -> None:
    """Write ``array`` to ``path`` via a temporary file and ``os.replace``."""
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.save(f, array)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def run_extraction(
    video_paths: Iterable[str],
    resolve: Callable[[str], str],
    feature_fn: FeatureFn,
    out_dir: str | Path,
    max_errors: int | None = None,
) -> dict[str, int]:
    """Extract features for every clip not yet on disk.

    Args:
        video_paths: Dataset-relative ids (MAVOS-DD ``video_path``).
        resolve: Maps an id to a local file, e.g. by downloading it.
        feature_fn: Maps a local file to (features, info dict).
        out_dir: Output folder for ``.npy`` files and ``log.jsonl``.
        max_errors: Stop after this many failures (None never stops).

    Returns:
        Counts of done, skipped and failed clips.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    counts = {"done": 0, "skipped": 0, "failed": 0}
    with open(out / "log.jsonl", "a", encoding="utf-8") as log:
        for vp in video_paths:
            target = out / feature_filename(vp)
            if target.exists():
                counts["skipped"] += 1
                continue
            t0 = time.perf_counter()
            record: dict[str, Any] = {"video_path": vp}
            try:
                feats, info = feature_fn(resolve(vp))
                save_atomic(target, feats)
                record.update(status="ok", shape=list(feats.shape), **info)
                counts["done"] += 1
            except Exception as exc:  # noqa: BLE001  (one bad clip must not stop a long run)
                record.update(status="error", error=f"{type(exc).__name__}: {exc}")
                counts["failed"] += 1
                logger.warning("failed %s: %s", vp, record["error"])
            record["sec_total"] = round(time.perf_counter() - t0, 3)
            log.write(json.dumps(record, default=str) + "\n")
            log.flush()
            if max_errors is not None and counts["failed"] >= max_errors:
                logger.error("stopping after %d errors", counts["failed"])
                break
    logger.info("extraction finished: %s", counts)
    return counts
