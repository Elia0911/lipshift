"""Tests for stratified subset selection."""

import numpy as np
import pandas as pd
import pytest

from lipshift.data.subset import cell_counts, filter_metadata, select_subset, validate_metadata


@pytest.fixture
def meta() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for lang in ["english", "spanish", "arabic"]:
        for gen, label, n in [
            ("real", "real", 40),
            ("inswapper", "fake", 30),
            ("echomimic", "fake", 5),
        ]:
            for i in range(n):
                rows.append(
                    {
                        "video_path": f"{lang}/{gen}/{i:04d}.mp4",
                        "label": label,
                        "split": rng.choice(["train", "test"]),
                        "language": lang,
                        "generative_method": gen,
                        "open_set_model": gen == "echomimic",
                        "open_set_language": lang == "arabic",
                    }
                )
    return pd.DataFrame(rows)


def test_cells_are_capped_and_small_cells_kept_whole(meta: pd.DataFrame) -> None:
    sub = select_subset(meta, per_cell=10, seed=0)
    counts = cell_counts(sub)
    full = cell_counts(meta).set_index(["language", "generative_method", "label", "split"])["n"]
    for _, row in counts.iterrows():
        key = (row["language"], row["generative_method"], row["label"], row["split"])
        assert row["n"] == min(10, full[key])
    assert len(counts) == len(full)  # no cell lost


def test_selection_is_deterministic_and_seed_dependent(meta: pd.DataFrame) -> None:
    a = select_subset(meta, per_cell=10, seed=0)
    b = select_subset(meta.sample(frac=1.0, random_state=123), per_cell=10, seed=0)
    c = select_subset(meta, per_cell=10, seed=1)
    pd.testing.assert_frame_equal(a, b)  # independent of input row order
    assert not a["video_path"].equals(c["video_path"])


def test_no_duplicates_and_sorted(meta: pd.DataFrame) -> None:
    sub = select_subset(meta, per_cell=10)
    assert sub["video_path"].is_unique
    assert sub["video_path"].is_monotonic_increasing


def test_filter_metadata(meta: pd.DataFrame) -> None:
    out = filter_metadata(meta, languages=["english"], splits=["test"])
    assert set(out["language"]) == {"english"} and set(out["split"]) == {"test"}
    assert len(filter_metadata(meta)) == len(meta)


def test_validation_errors(meta: pd.DataFrame) -> None:
    with pytest.raises(KeyError):
        validate_metadata(meta.drop(columns=["language"]))
    with pytest.raises(ValueError):
        validate_metadata(pd.concat([meta, meta.iloc[:1]]))
    with pytest.raises(ValueError):
        select_subset(meta, per_cell=0)
