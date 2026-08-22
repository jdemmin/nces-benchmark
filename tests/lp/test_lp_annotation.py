# tests/lp/test_lp_annotation.py
"""Complexity and hardness annotation."""

from __future__ import annotations

import pytest
from test_lp_schema import blank_complexity, make_problem

from src.data.complexity import Complexity, Hardness
from src.data.lp import LearningProblem


def hardness(*, ratio=0.25, baseline=0.5, redundant=False) -> Hardness:
    return Hardness(
        extension_size=4,
        extension_ratio=ratio,
        atomic_baseline_f1=baseline,
        redundant=redundant,
    )


def test_annotate_complexity_returns_a_new_problem() -> None:
    problem = make_problem()
    annotated = problem.annotate_complexity(blank_complexity(dl_length=9))
    assert annotated is not problem
    assert problem.complexity.dl_length == 1
    assert annotated.complexity.dl_length == 9


def test_annotate_complexity_preserves_identity_fields() -> None:
    problem = make_problem(3)
    annotated = problem.annotate_complexity(blank_complexity(dl_length=9))
    assert (annotated.id, annotated.target_concept) == (
        problem.id,
        problem.target_concept,
    )
    assert annotated.pos_example == problem.pos_example
    assert annotated.neg_example == problem.neg_example
    assert (annotated.num_pos, annotated.num_neg) == (
        problem.num_pos,
        problem.num_neg,
    )


def test_annotate_complexity_copies_example_lists() -> None:
    """The copy must not alias the original's lists."""
    problem = make_problem()
    annotated = problem.annotate_complexity(blank_complexity())
    assert annotated.pos_example is not problem.pos_example


def test_annotate_hardness_populates_hardness_fields() -> None:
    problem = make_problem(complexity=blank_complexity(dl_length=4, depth=1))
    annotated = problem.annotate_hardness(problem.complexity, hardness())
    assert annotated.complexity.hardness.atomic_baseline_f1 == 0.5
    assert annotated.complexity.hardness.extension_ratio == 0.25


def test_annotate_hardness_keeps_structural_fields() -> None:
    structural = blank_complexity(dl_length=4, depth=1)
    problem = make_problem(complexity=structural)
    annotated = problem.annotate_hardness(structural, hardness())
    assert annotated.complexity.dl_length == 4
    assert annotated.complexity.depth == 1


def test_annotate_hardness_does_not_mutate_the_receiver() -> None:
    problem = make_problem()
    problem.annotate_hardness(problem.complexity, hardness())
    assert problem.complexity.hardness.atomic_baseline_f1 is None


def test_annotate_hardness_uses_the_passed_complexity_not_self() -> None:
    """Documents the current contract: the complexity argument replaces
    self.complexity outright. If the intent was to annotate self's own
    complexity, this test is the one that should change."""
    problem = make_problem(complexity=blank_complexity(dl_length=1))
    annotated = problem.annotate_hardness(
        blank_complexity(dl_length=12, depth=3), hardness()
    )
    assert annotated.complexity.dl_length == 12
    assert annotated.complexity.depth == 3


def test_annotate_hardness_is_idempotent_in_the_last_write() -> None:
    problem = make_problem()
    once = problem.annotate_hardness(problem.complexity, hardness(baseline=0.5))
    twice = once.annotate_hardness(once.complexity, hardness(baseline=0.9))
    assert twice.complexity.hardness.atomic_baseline_f1 == 0.9


def test_annotated_problem_survives_serialization() -> None:
    problem = make_problem().annotate_hardness(
        blank_complexity(dl_length=4), hardness(redundant=True)
    )
    restored = LearningProblem.from_dict(problem.to_dict())
    assert restored.complexity.hardness.redundant is True
    assert restored == problem