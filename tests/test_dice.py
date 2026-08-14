# tests/test_dicee.py
from __future__ import annotations

from pathlib import Path

import pytest

from src.data.ontology import Triple, local_name, parse_triples
from src.models.dice import (
    _selection_score,
    generate_random_embeddings,
    write_dicee_dataset,
)


def test_local_name_handles_hash_and_slash() -> None:
    assert local_name("http://e.com/father#stefan") == "stefan"
    assert local_name("http://e.com/father/stefan") == "stefan"
    assert local_name("stefan") == "stefan"


def test_parse_triples_skips_literals(kb_path: Path) -> None:
    triples = parse_triples(kb_path)
    assert triples
    assert all(t.subject.startswith("http") for t in triples)


def test_dataset_split_never_leaves_empty_files(tmp_path: Path) -> None:
    triples = [Triple(f"s{i}", "p", f"o{i}") for i in range(3)]
    counts = write_dicee_dataset(triples, tmp_path, seed=1)

    for name in ("train", "valid", "test"):
        path = tmp_path / f"{name}.txt"
        assert path.is_file()
        # DICE crashes on empty valid/test files when eval spans them.
        assert path.read_text(encoding="utf-8").strip()
        assert counts[name] >= 1


def test_dataset_split_is_tab_separated(tmp_path: Path) -> None:
    write_dicee_dataset([Triple("s", "p", "o")], tmp_path, seed=1)
    line = (tmp_path / "train.txt").read_text(encoding="utf-8").splitlines()[0]
    assert line.split("\t") == ["s", "p", "o"]


def test_empty_triples_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="zero triples"):
        write_dicee_dataset([], tmp_path, seed=1)


def test_selection_score_prefers_validation_mrr() -> None:
    report = {"Train": {"MRR": 0.9}, "Val": {"MRR": 0.5}, "Test": {"MRR": 0.7}}
    score, error = _selection_score(report)
    assert score == 0.5
    assert error is None


def test_selection_score_falls_back_to_test_mrr() -> None:
    score, error = _selection_score({"Test": {"MRR": 0.7}})
    assert score == 0.7
    assert "test MRR" in error


def test_selection_score_reports_missing_metric() -> None:
    score, error = _selection_score({})
    assert score is None
    assert "No MRR" in error


def test_random_baseline_is_deterministic(tmp_path: Path) -> None:
    import pandas as pd

    names = ["http://e.com/f#stefan", "http://e.com/f#anna"]
    first = generate_random_embeddings(
        names, tmp_path / "a.csv", embedding_dim=8, seed=42
    )
    second = generate_random_embeddings(
        names, tmp_path / "b.csv", embedding_dim=8, seed=42
    )

    left = pd.read_csv(first, index_col=0)
    right = pd.read_csv(second, index_col=0)
    assert left.equals(right)
    assert list(left.index) == ["stefan", "anna"]
    assert left.shape == (2, 8)


def test_random_baseline_deduplicates_local_names(tmp_path: Path) -> None:
    import pandas as pd

    path = generate_random_embeddings(
        ["http://a#x", "http://b#x", "http://a#y"],
        tmp_path / "e.csv",
        embedding_dim=4,
        seed=1,
    )
    frame = pd.read_csv(path, index_col=0)
    assert list(frame.index) == ["x", "y"]
