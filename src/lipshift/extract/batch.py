"""Multi-encoder, resumable, time-budgeted batch extraction for Colab.

For each clip in a manifest:

1. Skip it if every applicable encoder already has a feature file.
2. Otherwise get a local copy (``resolve``, e.g. a Hugging Face download). Downloads for
   the next clips run in background threads while the current clip is processed.
3. Decode and crop once (``prepare``), run each missing encoder, save atomically.
4. Delete the local video (``cleanup``) and append a record to ``log.jsonl``.

The run stops before starting a new clip once ``max_minutes`` have passed, so a Colab
session can be ended cleanly and resumed later with the same call.
"""

from __future__ import annotations

import json
import time
from collections import deque
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lipshift.extract.pipeline import encode_mouths
from lipshift.extract.runner import feature_filename, save_atomic
from lipshift.utils import get_logger

logger = get_logger(__name__)

PrepareFn = Callable[[str], tuple[np.ndarray, dict[str, Any]]]


@dataclass
class EncoderSpec:
    """An encoder, its output folder name, and the languages it should run on."""

    name: str
    encoder: torch.nn.Module
    languages: frozenset[str] | None = None  # None: every language

    def applies_to(self, language: str) -> bool:
        return self.languages is None or language in self.languages


def pending_encoders(
    row: dict[str, Any], encoders: Sequence[EncoderSpec], out_root: Path
) -> list[EncoderSpec]:
    """Encoders that apply to this clip and have no feature file yet."""
    name = feature_filename(row["video_path"])
    return [
        e
        for e in encoders
        if e.applies_to(row["language"]) and not (out_root / e.name / name).exists()
    ]


def run_batch(
    rows: Iterable[dict[str, Any]],
    resolve: Callable[[str], str],
    prepare: PrepareFn,
    encoders: Sequence[EncoderSpec],
    out_root: str | Path,
    device: str = "cpu",
    cleanup: Callable[[str], None] | None = None,
    max_minutes: float | None = None,
    max_errors: int | None = None,
    prefetch: int = 4,
) -> dict[str, int]:
    """Process every clip that still needs features. Returns counts.

    Args:
        rows: Manifest rows with at least ``video_path`` and ``language``.
        resolve: Dataset-relative ``video_path`` -> local file path.
        prepare: Local file -> (mouth crops, info dict).
        encoders: Encoders to run; each writes to ``out_root / name``.
        out_root: Root folder for features and ``log.jsonl``.
        device: Torch device for encoders.
        cleanup: Called with the local path after each clip (e.g. delete the video).
        max_minutes: Stop before starting a clip after this much wall time.
        max_errors: Stop after this many failed clips.
        prefetch: Number of clips resolved ahead in background threads (0 disables).
    """
    out = Path(out_root)
    for e in encoders:
        (out / e.name).mkdir(parents=True, exist_ok=True)
    todo = [(r, pending_encoders(r, encoders, out)) for r in rows]
    counts = {"done": 0, "skipped": sum(1 for _, p in todo if not p), "failed": 0, "stopped": 0}
    todo = [(r, p) for r, p in todo if p]
    logger.info("%d clips to process, %d already complete", len(todo), counts["skipped"])

    start = time.perf_counter()
    pool = ThreadPoolExecutor(max_workers=max(prefetch, 1))
    queue: deque[tuple[dict[str, Any], list[EncoderSpec], Future]] = deque()
    it = iter(todo)

    def fill() -> None:
        while len(queue) < max(prefetch, 1):
            try:
                r, p = next(it)
            except StopIteration:
                return
            queue.append((r, p, pool.submit(resolve, r["video_path"])))

    try:
        with open(out / "log.jsonl", "a", encoding="utf-8") as log:
            fill()
            while queue:
                if max_minutes is not None and (time.perf_counter() - start) / 60 >= max_minutes:
                    counts["stopped"] = len(queue) + sum(1 for _ in it)
                    logger.info("time budget reached, %d clips left", counts["stopped"])
                    break
                row, needed, fut = queue.popleft()
                fill()
                record: dict[str, Any] = {"video_path": row["video_path"]}
                t0 = time.perf_counter()
                local: str | None = None
                try:
                    local = fut.result()
                    mouths, info = prepare(local)
                    record.update(info)
                    name = feature_filename(row["video_path"])
                    for e in needed:
                        t1 = time.perf_counter()
                        feats = encode_mouths(e.encoder, mouths, device)
                        save_atomic(out / e.name / name, feats)
                        record[f"sec_encode_{e.name}"] = round(time.perf_counter() - t1, 3)
                    record.update(status="ok", encoders=[e.name for e in needed])
                    counts["done"] += 1
                except Exception as exc:  # noqa: BLE001  (one bad clip must not stop the run)
                    record.update(status="error", error=f"{type(exc).__name__}: {exc}")
                    counts["failed"] += 1
                    logger.warning("failed %s: %s", row["video_path"], record["error"])
                finally:
                    if cleanup is not None and local is not None:
                        try:
                            cleanup(local)
                        except OSError as exc:
                            logger.warning("cleanup failed for %s: %s", local, exc)
                record["sec_total"] = round(time.perf_counter() - t0, 3)
                log.write(json.dumps(record, default=str) + "\n")
                log.flush()
                if max_errors is not None and counts["failed"] >= max_errors:
                    counts["stopped"] = len(queue) + sum(1 for _ in it)
                    logger.error("stopping after %d errors", counts["failed"])
                    break
    finally:
        for _, _, fut in queue:
            fut.cancel()
        pool.shutdown(wait=True, cancel_futures=True)
    logger.info("batch finished: %s", counts)
    return counts


def summarize_progress(
    rows: Iterable[dict[str, Any]], encoders: Sequence[EncoderSpec], out_root: str | Path
) -> dict[str, dict[str, int]]:
    """Per encoder: how many applicable clips exist and how many are done."""
    out = Path(out_root)
    rows = list(rows)
    summary: dict[str, dict[str, int]] = {}
    for e in encoders:
        applicable = [r for r in rows if e.applies_to(r["language"])]
        done = sum((out / e.name / feature_filename(r["video_path"])).exists() for r in applicable)
        summary[e.name] = {"applicable": len(applicable), "done": int(done)}
    return summary
