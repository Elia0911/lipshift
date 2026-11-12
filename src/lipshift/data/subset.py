"""Stratified, reproducible subset selection over MAVOS-DD metadata.

MAVOS-DD metadata rows have the fields ``video_path``, ``label``, ``split``,
``open_set_model``, ``open_set_language``, ``language`` and ``generative_method``
(as listed in the dataset card). A "cell" is one combination of the stratification
columns, for example (english, inswapper, fake, train). Each cell is capped at
``per_cell`` clips so no language or generator dominates the subset.

CLI:
    python -m lipshift.data.subset --meta /path/to/mavos_meta --config configs/subset.yaml \
        --out data/manifest.csv
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from lipshift.utils import get_logger, load_config

logger = get_logger(__name__)

REQUIRED_COLUMNS = ("video_path", "label", "split", "language", "generative_method")
DEFAULT_CELL_COLUMNS = ("language", "generative_method", "label", "split")


def validate_metadata(df: pd.DataFrame) -> None:
    """Raise if required columns are missing or ``video_path`` is not unique."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(f"Metadata is missing columns: {missing}")
    dup = df["video_path"].duplicated()
    if dup.any():
        raise ValueError(f"{int(dup.sum())} duplicated video_path values in metadata")


def filter_metadata(
    df: pd.DataFrame,
    languages: Sequence[str] | None = None,
    splits: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Keep only the requested languages and splits (None keeps all)."""
    out = df
    if languages:
        out = out[out["language"].isin(languages)]
    if splits:
        out = out[out["split"].isin(splits)]
    return out


def cell_counts(
    df: pd.DataFrame, cell_columns: Sequence[str] = DEFAULT_CELL_COLUMNS
) -> pd.DataFrame:
    """Number of clips per cell, missing values kept as their own group."""
    return (
        df.groupby(list(cell_columns), dropna=False)
        .size()
        .rename("n")
        .reset_index()
        .sort_values(list(cell_columns))
        .reset_index(drop=True)
    )


def select_subset(
    df: pd.DataFrame,
    per_cell: int,
    cell_columns: Sequence[str] = DEFAULT_CELL_COLUMNS,
    seed: int = 0,
) -> pd.DataFrame:
    """Sample at most ``per_cell`` clips from every cell, reproducibly.

    Cells smaller than ``per_cell`` are kept whole. Output is sorted by ``video_path``
    so the manifest is identical across runs and pandas versions' groupby order.
    """
    if per_cell < 1:
        raise ValueError("per_cell must be >= 1")
    validate_metadata(df)
    # Sort first: DataFrame.sample picks rows by position, so without a canonical order
    # the same seed would select different clips when metadata rows arrive reordered.
    df = df.sort_values("video_path", kind="stable").reset_index(drop=True)
    parts = [
        group.sample(n=min(per_cell, len(group)), random_state=seed)
        for _, group in df.groupby(list(cell_columns), dropna=False, sort=True)
    ]
    if not parts:
        return df.iloc[0:0].copy()
    out = pd.concat(parts).sort_values("video_path").reset_index(drop=True)
    return out


def load_mavos_metadata(meta_dir: str | Path) -> pd.DataFrame:
    """Load MAVOS-DD metadata saved with ``datasets.save_to_disk`` (needs ``datasets``)."""
    from datasets import load_from_disk

    meta_dir = Path(meta_dir)
    candidates = sorted({p.parent for p in meta_dir.rglob("state.json")}) or [meta_dir]
    return load_from_disk(str(candidates[0])).to_pandas()


def main(argv: Sequence[str] | None = None) -> None:
    """Write a stratified manifest CSV and its per-cell counts."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--meta", required=True, help="MAVOS-DD metadata dir, or a CSV export")
    parser.add_argument("--config", default="configs/subset.yaml")
    parser.add_argument("--out", default="data/manifest.csv")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    meta = Path(args.meta)
    df = pd.read_csv(meta) if meta.suffix == ".csv" else load_mavos_metadata(meta)
    validate_metadata(df)
    df = filter_metadata(df, cfg.get("languages"), cfg.get("splits"))
    sub = select_subset(
        df, cfg["per_cell"], cfg.get("cell_columns", DEFAULT_CELL_COLUMNS), cfg["seed"]
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out, index=False)
    counts_path = out.with_name(out.stem + "_counts.csv")
    cell_counts(sub, cfg.get("cell_columns", DEFAULT_CELL_COLUMNS)).to_csv(counts_path, index=False)
    logger.info("wrote %d clips to %s (counts: %s)", len(sub), out, counts_path)


if __name__ == "__main__":
    main()
