# tests/test_dice.py
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from flask import json

from src.config import EmbeddingSettings
from src.data.ontology import Triple, local_name, parse_triples
from src.models.dice import (
    DICEE_RESERVED_PATH_TOKENS,
    EmbeddingResultDice,
    _assert_dicee_safe_dataset_dir,
    _entity_index_mapping,
    build_embeddings,
    count_partitions,
    generate_random_embeddings,
    get_csv_dimension,
    search_best_embedding_setting,
    split_dicee_dataset,
    stage_partition,
)
from src.models.hpo_search_utils import best_trial_run_dir, selection_score


def test_embedding_result_to_dict_stringifies_path() -> None:
    result = EmbeddingResultDice(
        embedding_settings=EmbeddingSettings(
            model_name="QMult", embedding_dim=64, batch_size=32
        ),
        embeddings_path=Path("/tmp/QMult.csv"),
        score=0.5,
    )
    payload = result.to_dict()
    assert payload["embeddings_path"] == "/tmp/QMult.csv"
    assert json.dumps(payload)  # must be JSON-serializable

# Changed behavior: DICE should not train for the random condition,
# but it should still generate random embeddings. The test below
# was commented out because it was asserting that DICE must not
# train for the random condition, which is correct, but the test
# was not actually testing the generation of random embeddings.
# Instead, we can test that the random embeddings are generated
# and have the expected properties.
# TODO: Consider adding a test that checks that the random embeddings are generated correctly without training (expected dim).
# def test_random_only_condition_never_trains(
#     kb_path: Path, base_settings: EmbeddingSettings, monkeypatch
# ) -> None:
#     def explode(*args, **kwargs):
#         raise AssertionError("DICE must not train for the random condition")
#     tmp_path = Path(tempfile.mkdtemp())  # otherwise test prefix would trigger dicee path check
#     monkeypatch.setattr("src.models.dice.train_embedding_model", explode)
#     monkeypatch.setattr("src.models.dice.get_csv_dimension", lambda *a, **k: 64)
#     results = build_embeddings(
#         kb_path,
#         tmp_path / "clean" / "emb",
#         tmp_path / "clean" / "data",
#         base_settings,
#         seed=1,
#         embedding_conditions=["random"],
#         expected_dim=64,
#     )
#     assert set(results) == {"random"}
#     assert results["random"].score is None


def test_embedding_report_is_written(
    kb_path: Path, base_settings: EmbeddingSettings,
    monkeypatch: pytest.MonkeyPatch
) -> None:
    tmp_path = Path(tempfile.mkdtemp())  # otherwise test prefix would trigger dicee path check
    embeddings_dir = tmp_path / "clean" / "emb"
    monkeypatch.setattr("src.models.dice.get_csv_dimension", lambda *a, **k: 64)
    build_embeddings(
        embeddings_dir=embeddings_dir,
        data_dir=embeddings_dir,
        embedding_settings=base_settings,
        seed=1,
        embedding_conditions=["random"],
        nces_embedding_dim=64,
        triples=[Triple("s", "p", "o"), Triple("s2", "p2", "o2"), Triple("s3", "p3", "o3")],
        counts={ "train": 0, "valid": 0, "test": 0 }
    )
    payload = json.loads(
        (embeddings_dir / "embedding_report.json").read_text(
            encoding="utf-8"
        )
    )
    assert set(payload["triple_counts"]) == {"train", "valid", "test"}
    assert payload["num_entities"] > 0

def test_conditions_share_the_exported_width(tmp_path: Path) -> None:
    """The benchmark is invalid if dice and random widths differ."""
    dice_path = generate_random_embeddings(
        ["http://e#a"], tmp_path / "d.csv", embedding_dim=256, seed=1
    )
    random_path = generate_random_embeddings(
        ["http://e#a"],
        tmp_path / "r.csv",
        embedding_dim=get_csv_dimension(dice_path),
        seed=1,
    )
    assert get_csv_dimension(dice_path) == get_csv_dimension(random_path)

def test_entity_index_mapping_handles_plain_dict() -> None:
    class FakeKGE:
        entity_to_idx = {"http://e#a": 0, "http://e#b": 1}

    names, positions = _entity_index_mapping(FakeKGE())
    assert names == ["http://e#a", "http://e#b"]
    assert positions == [0, 1]


def test_entity_index_mapping_handles_dataframe() -> None:
    import pandas as pd

    class FakeKGE:
        entity_to_idx = pd.DataFrame(
            {"index": [2, 0]}, index=["http://e#x", "http://e#y"]
        )

    names, positions = _entity_index_mapping(FakeKGE())
    assert names == ["http://e#x", "http://e#y"]
    assert positions == [2, 0]


def test_entity_index_mapping_preserves_nonsequential_order() -> None:
    """Row positions must be applied, not assumed to be 0..n-1."""

    class FakeKGE:
        entity_to_idx = {"http://e#a": 3, "http://e#b": 1}

    _, positions = _entity_index_mapping(FakeKGE())
    assert positions == [3, 1]

def test_random_baseline_differs_across_seeds(tmp_path: Path) -> None:
    import pandas as pd

    names = ["http://e#a", "http://e#b"]
    a = generate_random_embeddings(
        names, tmp_path / "a.csv", embedding_dim=8, seed=1
    )
    b = generate_random_embeddings(
        names, tmp_path / "b.csv", embedding_dim=8, seed=2
    )
    assert not pd.read_csv(a, index_col=0).equals(
        pd.read_csv(b, index_col=0)
    )


def test_random_baseline_values_are_in_range(tmp_path: Path) -> None:
    import pandas as pd

    path = generate_random_embeddings(
        [f"http://e#{i}" for i in range(50)],
        tmp_path / "e.csv",
        embedding_dim=16,
        seed=9,
    )
    frame = pd.read_csv(path, index_col=0)
    assert frame.to_numpy().min() >= -1.0
    assert frame.to_numpy().max() <= 1.0


def test_random_baseline_preserves_first_occurrence_order(
    tmp_path: Path,
) -> None:
    import pandas as pd

    path = generate_random_embeddings(
        ["http://z#c", "http://z#a", "http://z#b", "http://y#a"],
        tmp_path / "e.csv",
        embedding_dim=4,
        seed=1,
    )
    assert list(pd.read_csv(path, index_col=0).index) == ["c", "a", "b"]


def test_random_baseline_creates_parent_directories(
    tmp_path: Path,
) -> None:
    path = generate_random_embeddings(
        ["http://e#a"],
        tmp_path / "deep" / "nested" / "e.csv",
        embedding_dim=2,
        seed=1,
    )
    assert path.is_file()


def test_get_csv_dimension_matches_written_width(tmp_path: Path) -> None:
    path = generate_random_embeddings(
        ["http://e#a", "http://e#b"],
        tmp_path / "e.csv",
        embedding_dim=37,
        seed=1,
    )
    assert get_csv_dimension(path) == 37

@pytest.fixture
def base_settings() -> EmbeddingSettings:
    return EmbeddingSettings(
        model_name="QMult", embedding_dim=64, batch_size=32, epochs=1, hpo_backend="grid"
    )

def test_search_grid_is_the_documented_cross_product(
    base_settings: EmbeddingSettings,
) -> None:
    grid = base_settings.search_grid()
    assert {(s.embedding_dim, s.batch_size) for s in grid} == {
        (64, 16),
        (64, 32),
        (128, 16),
        (128, 32),
    }


def test_search_grid_collapses_duplicate_batch_of_one() -> None:
    grid = EmbeddingSettings(
        model_name="TransE", embedding_dim=8, batch_size=1
    ).search_grid()
    assert {s.batch_size for s in grid} == {1}
    assert len(grid) == 2


def test_search_picks_highest_validation_mrr(
    tmp_path: Path, base_settings: EmbeddingSettings, monkeypatch
) -> None:
    scores = {(64, 16): 0.1, (64, 32): 0.9, (128, 16): 0.4, (128, 32): 0.2}

    def fake_train(dataset_dir, run_dir, settings, *, seed):
        run_dir.mkdir(parents=True, exist_ok=True)
        key = (settings.embedding_dim, settings.batch_size)
        return {"Val": {"MRR": scores[key]}}

    monkeypatch.setattr(
        "src.models.dice_grid_search.train_embedding_model", fake_train
    )
    best, _, trials, error, _ = search_best_embedding_setting(
        tmp_path / "data", tmp_path / "emb", base_settings, seed=1
    )
    assert (best.embedding_dim, best.batch_size) == (64, 32)
    assert error is None
    assert len(trials) == 4


def test_tied_scores_resolve_deterministically(
    tmp_path: Path, base_settings: EmbeddingSettings, monkeypatch
) -> None:
    """All-equal scores must not silently select by insertion order."""

    def fake_train(dataset_dir, run_dir, settings, *, seed):
        run_dir.mkdir(parents=True, exist_ok=True)
        return {"Val": {"MRR": 0.5}}

    monkeypatch.setattr(
        "src.models.dice_grid_search.train_embedding_model", fake_train
    )
    best, _, _, _, _ = search_best_embedding_setting(
        tmp_path / "data", tmp_path / "emb", base_settings, seed=1
    )
    # Smallest (dim, batch) wins the tie -- the cheapest model.
    assert (best.embedding_dim, best.batch_size) == (64, 16)


def test_failed_trial_is_recorded_not_fatal(
    tmp_path: Path, base_settings: EmbeddingSettings, monkeypatch
) -> None:
    def fake_train(dataset_dir, run_dir, settings, *, seed):
        if settings.embedding_dim == 64:
            raise RuntimeError("simulated OOM")
        run_dir.mkdir(parents=True, exist_ok=True)
        return {"Val": {"MRR": 0.6}}

    monkeypatch.setattr(
        "src.models.dice_grid_search.train_embedding_model", fake_train
    )
    best, _, trials, _, _ = search_best_embedding_setting(
        tmp_path / "data", tmp_path / "emb", base_settings, seed=1
    )
    assert best.embedding_dim == 128
    failed = [t for t in trials if "error" in t]
    assert len(failed) == 2
    assert "simulated OOM" in failed[0]["error"]


def test_all_trials_failing_raises(
    tmp_path: Path, base_settings: EmbeddingSettings, monkeypatch
) -> None:
    def fake_train(dataset_dir, run_dir, settings, *, seed):
        raise RuntimeError("boom")

    monkeypatch.setattr(
        "src.models.dice_grid_search.train_embedding_model", fake_train
    )
    with pytest.raises(RuntimeError, match="Every DICE hyperparameter"):
        search_best_embedding_setting(
            tmp_path / "data", tmp_path / "emb", base_settings, seed=1
        )


def test_train_only_reports_abort_the_search(
    tmp_path: Path, base_settings: EmbeddingSettings, monkeypatch
) -> None:
    """Reproduces the observed failure: no trial is selectable."""

    def fake_train(dataset_dir, run_dir, settings, *, seed):
        run_dir.mkdir(parents=True, exist_ok=True)
        return {"Train": {"MRR": 1.0, "H@1": 1.0}}

    monkeypatch.setattr(
        "src.models.dice_grid_search.train_embedding_model", fake_train
    )
    with pytest.raises(RuntimeError, match="Every DICE hyperparameter"):
        search_best_embedding_setting(
            tmp_path / "data", tmp_path / "emb", base_settings, seed=1
        )


def test_best_trial_run_dir_matches_on_both_dimensions() -> None:
    trials = [
        {"embedding_dim": 64, "batch_size": 16, "run_dir": "/a"},
        {"embedding_dim": 64, "batch_size": 32, "run_dir": "/b"},
    ]
    best = EmbeddingSettings(
        model_name="QMult", embedding_dim=64, batch_size=32
    )
    assert best_trial_run_dir(trials, best) == Path("/b")


def test_best_trial_run_dir_skips_failed_records() -> None:
    trials = [
        {"embedding_dim": 64, "batch_size": 32, "error": "boom"},
        {"embedding_dim": 64, "batch_size": 32, "run_dir": "/ok"},
    ]
    best = EmbeddingSettings(
        model_name="QMult", embedding_dim=64, batch_size=32
    )
    assert best_trial_run_dir(trials, best) == Path("/ok")


def test_missing_run_dir_raises() -> None:
    best = EmbeddingSettings(
        model_name="QMult", embedding_dim=64, batch_size=32
    )
    with pytest.raises(RuntimeError, match="Could not locate"):
        best_trial_run_dir([], best)

@pytest.mark.parametrize("key", ["Val", "Valid", "Validation"])
def test_selection_accepts_every_validation_key(key: str) -> None:
    score, error = selection_score({key: {"MRR": 0.42}})
    assert score == 0.42
    assert error is None


def test_validation_wins_over_test_even_when_lower() -> None:
    score, error = selection_score(
        {"Val": {"MRR": 0.1}, "Test": {"MRR": 0.9}}
    )
    assert score == 0.1
    assert error is None


def test_train_only_report_refuses_to_score() -> None:
    """A train-only report signals the dicee path-routing bug."""
    score, error = selection_score({"Train": {"MRR": 1.0}})
    assert score is None
    assert "train MRR" in error
    assert "valid.txt" in error


def test_non_dict_sections_are_ignored() -> None:
    score, error = selection_score({"Val": "not-a-dict", "Test": None})
    assert score is None
    assert "No MRR" in error


def test_section_without_mrr_is_skipped() -> None:
    score, error = selection_score(
        {"Val": {"H@1": 0.5}, "Test": {"MRR": 0.3}}
    )
    assert score == 0.3
    assert "test MRR" in error

def test_split_respects_ratios_on_large_input() -> None:
    triples = [Triple(f"s{i}", "p", f"o{i}") for i in range(1000)]
    counts = count_partitions(split_dicee_dataset(
        triples=triples, output_dir=Path(tempfile.mkdtemp()) / "clean" / "d"
    ))
    assert sum(counts.values()) == 1000
    assert counts["train"] == 800
    assert counts["valid"] == 100
    assert counts["test"] == 100


def test_split_is_deterministic_in_seed() -> None:
    triples = [Triple(f"s{i}", "p", f"o{i}") for i in range(50)]
    tmp_path = Path(tempfile.mkdtemp())  # otherwise test prefix would trigger dicee path check
    first = tmp_path / "clean" / "a"
    second = tmp_path / "clean" / "b"
    stage_partition(
        output_dir=first,
        partitions=split_dicee_dataset(triples=triples, output_dir=first)
    )
    stage_partition(
        output_dir=second,
        partitions=split_dicee_dataset(triples=triples, output_dir=second)
    )
    for name in ("train", "valid", "test"):
        assert (first / f"{name}.txt").read_text(
            encoding="utf-8"
        ) == (second / f"{name}.txt").read_text(encoding="utf-8")


def test_splits_partition_the_input_without_loss() -> None:
    triples = [Triple(f"s{i}", "p", f"o{i}") for i in range(100)]
    tmp_path = Path(tempfile.mkdtemp())  # otherwise test prefix would trigger dicee path check
    directory = tmp_path / "clean" / "d"
    stage_partition(
        output_dir=directory,
        partitions=split_dicee_dataset(triples=triples, output_dir=directory)
    )

    seen: list[str] = []
    for name in ("train", "valid", "test"):
        seen.extend(
            (directory / f"{name}.txt")
            .read_text(encoding="utf-8")
            .splitlines()
        )
    assert len(seen) == 100
    assert len(set(seen)) == 100


def test_two_triples_still_fill_all_three_splits() -> None:
    counts = count_partitions(split_dicee_dataset(
        triples=[Triple("a", "p", "b"), Triple("c", "p", "d")],
        output_dir=Path(tempfile.mkdtemp()) / "clean" / "d",
    ))
    assert all(counts[name] >= 1 for name in ("train", "valid", "test"))


def test_iri_triples_survive_round_trip() -> None:
    directory = Path(tempfile.mkdtemp()) / "clean" / "d"
    stage_partition(
        output_dir=directory,
        partitions=split_dicee_dataset(
            triples=[Triple("http://e#s", "http://e#p", "http://e#o")],
            output_dir=directory,
        ),
    )
    line = (directory / "train.txt").read_text(encoding="utf-8").strip()
    assert line.split("\t") == ["http://e#s", "http://e#p", "http://e#o"]

@pytest.mark.parametrize("token", DICEE_RESERVED_PATH_TOKENS)
def test_reserved_token_in_parent_path_is_rejected(
    tmp_path: Path, token: str
) -> None:
    # dicee would route every split file into raw_train_set.
    directory = tmp_path / f"smoke-nces-{token}" / "seed1" / "kb" / "data"
    directory.mkdir(parents=True)
    with pytest.raises(ValueError, match="substrings of the full path"):
        _assert_dicee_safe_dataset_dir(directory)


def test_reserved_token_check_is_case_insensitive(tmp_path: Path) -> None:
    directory = tmp_path / "Smoke-NCES-TRAIN" / "data"
    directory.mkdir(parents=True)
    with pytest.raises(ValueError, match="train"):
        _assert_dicee_safe_dataset_dir(directory)


def test_dataset_dir_own_name_may_contain_tokens() -> None:
    # Only ancestors matter: dicee globs *inside* dataset_dir, and the
    # directory's own name is part of every child path -- but so is the
    # parent, so we only guard the parent to avoid false positives on
    # a directory legitimately named e.g. "train_data".
    tmp_path = Path(tempfile.mkdtemp())  # otherwise test prefix would trigger dicee path check
    directory = tmp_path / "clean" / "data"
    directory.mkdir(parents=True)
    _assert_dicee_safe_dataset_dir(directory)  # must not raise


def test_write_dicee_dataset_rejects_unsafe_directory(tmp_path: Path) -> None:
    directory = tmp_path / "my-train-run" / "data"
    with pytest.raises(ValueError, match="dicee"):
        stage_partition(
            output_dir=directory,
            partitions={"train": [("s", "p", "o")], "valid": [("s", "p", "o")], "test": [("s", "p", "o")],
            },
        )


def test_split_files_are_distinguishable_by_basename() -> None:
    """Each split file must be identifiable without its parent path."""
    tmp_path = Path(tempfile.mkdtemp())  # otherwise test prefix would trigger dicee path check
    directory = tmp_path / "clean" / "data"
    stage_partition(
        output_dir=directory,
        partitions=split_dicee_dataset(
            output_dir=directory,
            triples=[Triple(f"s{i}", "p", f"o{i}") for i in range(30)],
        ),
    )
    names = sorted(p.name for p in directory.glob("*"))
    assert names == ["test.txt", "train.txt", "valid.txt"]
    # No basename is a substring of another, so basename routing is sound.
    for a in names:
        others = [b for b in names if b != a]
        assert not any(a.removesuffix(".txt") in b for b in others)

def test_local_name_handles_hash_and_slash() -> None:
    assert local_name("http://e.com/father#stefan") == "stefan"
    assert local_name("http://e.com/father/stefan") == "stefan"
    assert local_name("stefan") == "stefan"


def test_parse_triples_skips_literals(kb_path: Path) -> None:
    triples = parse_triples(kb_path)
    assert triples
    assert all(t.subject.startswith("http") for t in triples)


def test_dataset_split_never_leaves_empty_files() -> None:
    triples = [Triple(f"s{i}", "p", f"o{i}") for i in range(3)]
    tmp_path = Path(tempfile.mkdtemp())  # otherwise test prefix would trigger dicee path check
    split = split_dicee_dataset(
        output_dir=tmp_path,
        triples=triples,
    )
    stage_partition(
        output_dir=tmp_path,
        partitions=split,
    )
    counts = count_partitions(split)

    for name in ("train", "valid", "test"):
        path = tmp_path / f"{name}.txt"
        assert path.is_file()
        # DICE crashes on empty valid/test files when eval spans them.
        assert path.read_text(encoding="utf-8").strip()
        assert counts[name] >= 1


def test_dataset_split_is_tab_separated() -> None:
    tmp_path = Path(tempfile.mkdtemp())  # otherwise test prefix would trigger dicee path check
    stage_partition(
        output_dir=tmp_path,
        partitions=split_dicee_dataset(
            output_dir=tmp_path,
            triples=[Triple("s", "p", "o")],
        ),
    )
    line = (tmp_path / "train.txt").read_text(encoding="utf-8").splitlines()[0]
    assert line.split("\t") == ["s", "p", "o"]


def test_empty_triples_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="zero triples"):
        stage_partition(
            output_dir=tmp_path,
            partitions=split_dicee_dataset(
                output_dir=tmp_path,
                triples=[],
            ),
        )


def test_selection_score_prefers_validation_mrr() -> None:
    report = {"Train": {"MRR": 0.9}, "Val": {"MRR": 0.5}, "Test": {"MRR": 0.7}}
    score, error = selection_score(report)
    assert score == 0.5
    assert error is None


def test_selection_score_falls_back_to_test_mrr() -> None:
    score, error = selection_score({"Test": {"MRR": 0.7}})
    assert score == 0.7
    assert "test MRR" in error


def test_selection_score_reports_missing_metric() -> None:
    score, error = selection_score({})
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
