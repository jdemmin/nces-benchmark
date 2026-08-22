# tests/lp/test_lp_split_sizing.py
"""Split sizing arithmetic and edge cases."""

from __future__ import annotations

import pytest
from test_lp_schema import blank_complexity, make_problem

from src.data.lp import split_learning_problems


def population(n: int, *, depth: int = 0):
    return [
        make_problem(i, complexity=blank_complexity(dl_length=1, depth=depth))
        for i in range(n)
    ]


def test_default_ratio_is_eighty_twenty_within_a_stratum() -> None:
    split = split_learning_problems(population(100), seed=1)
    assert len(split["train"]) == 80
    assert len(split["test"]) == 20


def test_every_problem_lands_in_exactly_one_split() -> None:
    problems = population(37)
    split = split_learning_problems(problems, seed=1)
    ids = [p.id for group in split.values() for p in group]
    assert sorted(ids) == sorted(p.id for p in problems)


def test_two_problem_stratum_splits_one_and_one() -> None:
    split = split_learning_problems(population(2), seed=1)
    assert len(split["train"]) == 1
    assert len(split["test"]) == 1


def test_three_problem_stratum_reserves_a_test_item() -> None:
    """int(3 * 0.8) == 2 and int(3 * 0.2) == 0, so the correction branch
    must still leave one problem for test."""
    split = split_learning_problems(population(3), seed=1)
    assert len(split["test"]) == 1

# Change behaviour to raise an error if there are fewer than
# 2 problems, since a split is not possible with only one
# problem. Test is kept as a reminder of the previous behaviour.
# def test_single_problem_goes_to_test_via_the_guard() -> None:
#     split = split_learning_problems(population(1), seed=1)
#     assert not split["train"]
#     assert len(split["test"]) == 1


def test_custom_ratios_are_honoured() -> None:
    split = split_learning_problems(population(100), seed=1, ratios=(0.5, 0.5))
    assert len(split["train"]) == 50
    assert len(split["test"]) == 50


def test_train_heavy_ratio_still_yields_a_test_split() -> None:
    split = split_learning_problems(population(10), seed=1, ratios=(0.99, 0.01))
    assert split["test"]


def test_stratification_preserves_per_stratum_proportions() -> None:
    problems = population(50, depth=0) + [
        make_problem(100 + i, complexity=blank_complexity(dl_length=1, depth=1))
        for i in range(10)
    ]
    split = split_learning_problems(problems, seed=1, stratify_by="depth")
    test_depths = [p.complexity.depth for p in split["test"]]
    assert test_depths.count(0) == 10
    assert test_depths.count(1) == 2


def test_thin_stratum_does_not_starve_a_rich_one() -> None:
    """One depth-1 problem must not consume the whole test split."""
    problems = population(20, depth=0) + [
        make_problem(100, complexity=blank_complexity(dl_length=1, depth=1))
    ]
    split = split_learning_problems(problems, seed=1, stratify_by="depth")
    assert len([p for p in split["train"] if p.complexity.depth == 0]) == 16


def test_unstratified_input_forms_a_single_bucket() -> None:
    """Without stratification every problem shares the dl_length fallback
    key, so sizing is computed once over the whole population."""
    problems = population(10, depth=0) + population(10, depth=1)
    split = split_learning_problems(problems, seed=1, stratify_by=None)
    assert len(split["train"]) == 16
    assert len(split["test"]) == 4


def test_split_does_not_mutate_the_input_order() -> None:
    problems = population(20)
    before = [p.id for p in problems]
    split_learning_problems(problems, seed=1)
    assert [p.id for p in problems] == before


def test_seed_is_combined_with_the_stratum_key() -> None:
    """Strata are shuffled with f"{seed}:{key}", so two strata under the
    same seed must not receive the same permutation."""
    problems = population(20, depth=0) + [
        make_problem(100 + i, complexity=blank_complexity(dl_length=1, depth=1))
        for i in range(20)
    ]
    split = split_learning_problems(problems, seed=1, stratify_by="depth")
    flat = [int(p.id.split("_")[1]) for p in split["test"] if p.complexity.depth == 0]
    deep = [
        int(p.id.split("_")[1]) - 100
        for p in split["test"]
        if p.complexity.depth == 1
    ]
    assert flat != deep


def test_split_is_reproducible_across_input_orderings() -> None:
    """Determinism must come from the seed, not from input order."""
    problems = population(20)
    forward = split_learning_problems(problems, seed=5)
    backward = split_learning_problems(list(reversed(problems)), seed=5)
    assert {p.id for p in forward["test"]} == {p.id for p in backward["test"]}