"""Serialisation must be lossless and order-preserving."""

from __future__ import annotations

import json

import pytest
from lp.determinism.conftest import make_hardness, make_problem

from src.data.lp import (
    LearningProblem,
    load_learning_problems,
    save_learning_problems,
    save_split,
    split_learning_problems,
)


class TestRoundTrip:
    def test_single_problem_round_trip(self):
        problem = make_problem("A ⊓ B")
        restored = LearningProblem.from_dict(problem.to_dict())
        assert restored == problem

    def test_hardness_survives_round_trip(self):
        problem = make_problem("A", hardness=make_hardness(atomic_baseline_f1=0.75))
        restored = LearningProblem.from_dict(problem.to_dict())
        assert restored.complexity.hardness == problem.complexity.hardness
        assert restored.complexity.is_annotated

    def test_num_pos_neg_are_recomputed(self):
        problem = make_problem("A", n_pos=4, n_neg=6)
        assert problem.num_pos == 4
        assert problem.num_neg == 6
        assert LearningProblem.from_dict(problem.to_dict()).num_pos == 4

    def test_file_round_trip_preserves_ids(self, population, tmp_path):
        path = tmp_path / "problems.json"
        save_learning_problems(population, path)
        restored = load_learning_problems(path)
        assert {p.id for p in restored} == {p.id for p in population}

    def test_load_order_is_canonical(self, population, tmp_path):
        path = tmp_path / "problems.json"
        save_learning_problems(population, path)
        restored = load_learning_problems(path)
        assert [p.id for p in restored] == sorted(p.id for p in restored)

    def test_split_survives_round_trip(self, population, tmp_path):
        path = tmp_path / "problems.json"
        save_learning_problems(population, path)
        restored = load_learning_problems(path)

        before = split_learning_problems(population, seed=7)
        after = split_learning_problems(restored, seed=7)
        assert {p.id for p in before["test"]} == {p.id for p in after["test"]}

    def test_save_split_writes_both_files(self, population, tmp_path):
        split = split_learning_problems(population, seed=7)
        save_split(split, tmp_path)
        for name in ("train", "test"):
            payload = json.loads((tmp_path / f"{name}_problems.json").read_text())
            assert len(payload) == len(split[name])

    def test_saved_split_is_byte_identical_across_runs(self, population, tmp_path):
        a, b = tmp_path / "a", tmp_path / "b"
        save_split(split_learning_problems(population, seed=7), a)
        save_split(split_learning_problems(population, seed=7), b)
        for name in ("train", "test"):
            assert (a / f"{name}_problems.json").read_bytes() == (
                b / f"{name}_problems.json"
            ).read_bytes()


class TestValidation:
    def test_missing_positives_rejected(self):
        from lp.determinism.conftest import make_complexity

        with pytest.raises(ValueError, match="no positive examples"):
            LearningProblem(
                id="lp_x",
                target_concept="A",
                complexity=make_complexity(),
                pos_example=[],
                neg_example=["y"],
            )

    def test_missing_negatives_rejected(self):
        from lp.determinism.conftest import make_complexity

        with pytest.raises(ValueError, match="no negative examples"):
            LearningProblem(
                id="lp_x",
                target_concept="A",
                complexity=make_complexity(),
                pos_example=["x"],
                neg_example=[],
            )


class TestNCESDatapoint:
    def test_iris_reduced_to_local_names(self):
        problem = make_problem("A")
        name, examples = problem.as_nces_datapoint()
        assert name == "A"
        assert all("#" not in e for e in examples["positive examples"])
        assert all("/" not in e for e in examples["positive examples"])

    def test_example_counts_preserved(self):
        problem = make_problem("A", n_pos=3, n_neg=5)
        _, examples = problem.as_nces_datapoint()
        assert len(examples["positive examples"]) == 3
        assert len(examples["negative examples"]) == 5