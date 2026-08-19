# tests/test_dice_dataset_dir.py
"""Regression tests for the dicee dataset-directory routing guard."""

from __future__ import annotations

from pathlib import Path
import tempfile

import pytest

from src.models.dice import (
    _assert_dicee_safe_dataset_dir,
    stage_dicee_dataset,
)


@pytest.mark.parametrize(
    "tainted",
    [
        "train",
        "trained_models",
        "pretraining",
        "valid",
        "validation",
        "test",
        "testbed",
        "latest",  # contains "test"
    ],
)
def test_reserved_tokens_in_any_ancestor_are_rejected(
    tmp_path: Path, tainted: str
) -> None:
    # Two levels up, not the immediate parent: the substring match is
    # against the whole path, so depth must not matter.
    directory = tmp_path / tainted / "runs" / "kg"
    directory.mkdir(parents=True)
    with pytest.raises(ValueError, match="ancestor path segments"):
        _assert_dicee_safe_dataset_dir(directory)


def test_clean_paths_are_accepted() -> None:
    directory = Path(tempfile.mkdtemp()) / "benchmark1" / "embeddings" / "kg"
    directory.mkdir(parents=True)
    _assert_dicee_safe_dataset_dir(directory)  # must not raise


def test_staging_copies_all_three_splits(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    for name in ("train", "valid", "test"):
        (source / f"{name}.txt").write_text(f"a\tb\t{name}\n")

    staged = stage_dicee_dataset(source, tmp_path / "staging")

    for name in ("train", "valid", "test"):
        assert (staged / f"{name}.txt").read_text() == f"a\tb\t{name}\n"