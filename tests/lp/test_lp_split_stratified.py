# tests/test_lp_split_stratified.py

from __future__ import annotations

from src.data.complexity import structural_complexity
from src.data.lp import LearningProblem, split_learning_problems


def make_problem(index: int, expression: str) -> LearningProblem:
    return LearningProblem(
        id=f"lp_{index:04d}",
        target_concept=expression,
        pos_example=[f"http://x#p{index}"],
        neg_example=[f"http://x#n{index}"],
        complexity=structural_complexity(expression),
    )


def make_population() -> list[LearningProblem]:
    """20 depth-0 and 20 depth-1 problems."""
    flat = [make_problem(i, "male ⊓ person") for i in range(20)]
    deep = [make_problem(20 + i, "∃ hasChild.person") for i in range(20)]
    return flat + deep


def test_splits_are_disjoint():
    split = split_learning_problems(make_population(), seed=1)
    ids = [{p.id for p in problems} for problems in split.values()]
    assert not (ids[0] & ids[1])


def test_split_is_exhaustive():
    problems = make_population()
    split = split_learning_problems(problems, seed=1)
    recovered = sum(len(items) for items in split.values())
    assert recovered == len(problems)


def test_both_strata_reach_the_test_split():
    """The point of stratifying: test is not accidentally all-easy."""
    split = split_learning_problems(make_population(), seed=1, stratify_by="depth")
    depths = {p.complexity.depth for p in split["test"]}
    assert depths == {0, 1}


def test_unstratified_path_still_works():
    split = split_learning_problems(make_population(), seed=1, stratify_by=None)
    assert sum(len(items) for items in split.values()) == 40
    assert split["test"]


def test_deterministic_in_seed():
    first = split_learning_problems(make_population(), seed=7)
    second = split_learning_problems(make_population(), seed=7)
    assert [p.id for p in first["test"]] == [p.id for p in second["test"]]


def test_different_seeds_differ():
    first = split_learning_problems(make_population(), seed=1)
    second = split_learning_problems(make_population(), seed=2)
    assert [p.id for p in first["test"]] != [p.id for p in second["test"]]

# Behaviour changed to raise an error if there are fewer than 2 problems,
# since a split is not possible with only one problem. Test is kept as a
# reminder of the previous behaviour.
# def test_test_split_never_empty_on_thin_input():
#     split = split_learning_problems([make_problem(0, "male")], seed=1)
#     assert len(split["test"]) == 1
#     assert not split["train"]


# def test_empty_input():
#     split = split_learning_problems([], seed=1)
#     assert split == {"train": [], "test": []}


def test_strata_seeded_independently():
    """Each stratum gets its own permutation, not a shared one."""
    split = split_learning_problems(make_population(), seed=1, stratify_by="depth")
    flat = [p.id for p in split["train"] if p.complexity.depth == 0]
    deep = [p.id for p in split["train"] if p.complexity.depth == 1]
    offsets_flat = [int(i.split("_")[1]) for i in flat]
    offsets_deep = [int(i.split("_")[1]) - 20 for i in deep]
    assert offsets_flat != offsets_deep