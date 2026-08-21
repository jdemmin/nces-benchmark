"""Normalisation of ontolearn's LPs.json payload."""

from __future__ import annotations

import pytest

from src.data.lp import _expand, _normalise

NS = "http://example.com/father#"


def payload(**problems):
    return {
        name: {"positive examples": pos, "negative examples": neg}
        for name, (pos, neg) in problems.items()
    }


# --- _expand ---------------------------------------------------------


def test_expand_prefixes_local_names() -> None:
    assert _expand(["stefan", "anna"], NS) == [f"{NS}stefan", f"{NS}anna"]


def test_expand_leaves_absolute_iris_untouched() -> None:
    assert _expand(["https://other.org/x#y"], NS) == ["https://other.org/x#y"]


def test_expand_handles_mixed_input() -> None:
    assert _expand(["stefan", "http://o#z"], NS) == [f"{NS}stefan", "http://o#z"]


def test_expand_without_namespace_is_identity() -> None:
    assert _expand(["stefan"], "") == ["stefan"]


def test_expand_is_a_no_op_on_empty_input() -> None:
    assert _expand([], NS) == []


# --- _normalise ------------------------------------------------------


def test_normalise_assigns_sequential_zero_padded_ids() -> None:
    raw = payload(
        male=(["stefan"], ["anna"]),
        female=(["anna"], ["stefan"]),
    )
    problems = _normalise(raw, namespace=NS)
    assert [p.id for p in problems] == ["lp_0000", "lp_0001"]


def test_normalise_sorts_examples() -> None:
    raw = payload(male=(["stefan", "anna", "markus"], ["michelle"]))
    (problem,) = _normalise(raw, namespace=NS)
    assert problem.pos_example == sorted(problem.pos_example)
    assert problem.pos_example[0].endswith("#anna")


def test_normalise_expands_to_full_iris() -> None:
    (problem,) = _normalise(payload(male=(["stefan"], ["anna"])), namespace=NS)
    assert problem.pos_example == [f"{NS}stefan"]


def test_normalise_drops_degenerate_problems() -> None:
    raw = payload(
        good=(["stefan"], ["anna"]),
        no_negatives=(["stefan"], []),
        no_positives=([], ["anna"]),
    )
    problems = _normalise(raw, namespace=NS)
    assert [p.target_concept for p in problems] == ["good"]


def test_normalise_ids_are_positional_not_compacted() -> None:
    """A dropped problem still consumes its index, so ids are stable
    against the raw payload rather than against the retained subset.
    This is intentional; changing it would silently renumber artifacts."""
    raw = payload(
        skipped=(["stefan"], []),
        kept=(["stefan"], ["anna"]),
    )
    (problem,) = _normalise(raw, namespace=NS)
    assert problem.id == "lp_0001"


def test_normalise_accepts_a_list_of_pairs() -> None:
    raw = [("male", {"positive examples": ["stefan"], "negative examples": ["anna"]})]
    (problem,) = _normalise(raw, namespace=NS)
    assert problem.target_concept == "male"


def test_normalise_tolerates_missing_example_keys() -> None:
    assert _normalise({"male": {}}, namespace=NS) == []


def test_normalise_computes_structural_complexity() -> None:
    raw = payload(**{"male ⊓ ∃ hasChild.person": (["stefan"], ["anna"])})
    (problem,) = _normalise(raw, namespace=NS)
    assert problem.complexity.dl_length > 1
    assert problem.complexity.depth == 1


def test_normalise_leaves_hardness_unpopulated() -> None:
    """Hardness is the annotation stage's job, not generation's."""
    (problem,) = _normalise(payload(male=(["stefan"], ["anna"])), namespace=NS)
    assert problem.complexity.hardness.atomic_baseline_f1 is None