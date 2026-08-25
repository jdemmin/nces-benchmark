# tests/lp/test_lp_schema.py
"""Learning-problem schema: construction, invariants, serialization."""

from __future__ import annotations

import json

import pytest

from src.data.complexity import Complexity, Hardness
from src.data.lp import LearningProblem


def blank_complexity(dl_length: int = 1, depth: int = 0) -> Complexity:
    return Complexity(
        dl_length=dl_length,
        num_atomic_classes=1,
        num_roles=0,
        expressivity="EL",
        hardness=Hardness.get_blank_hardness(),
        depth=depth,
        constructors={},
    )


def make_problem(
    index: int = 0,
    *,
    complexity: Complexity | None = None,
    positives: list[str] | None = None,
    negatives: list[str] | None = None,
) -> LearningProblem:
    return LearningProblem(
        id=f"lp_{index:04d}",
        target_concept=f"C{index}",
        pos_example=positives or [f"http://x#p{index}"],
        neg_example=negatives or [f"http://x#n{index}"],
        complexity=complexity or blank_complexity(),
    )


# --- construction invariants -----------------------------------------


def test_missing_positives_is_rejected() -> None:
    with pytest.raises(ValueError, match="no positive examples"):
        LearningProblem(
            id="x",
            target_concept="male",
            pos_example=[],
            neg_example=["a"],
            complexity=blank_complexity(),
        )


def test_example_counts_are_derived_not_trusted() -> None:
    """num_pos/num_neg are overwritten in __post_init__, so a caller
    cannot smuggle in inconsistent counts."""
    problem = LearningProblem(
        id="x",
        target_concept="male",
        pos_example=["a", "b", "c"],
        neg_example=["d"],
        complexity=blank_complexity(),
        num_pos=99,
        num_neg=99,
    )
    assert (problem.num_pos, problem.num_neg) == (3, 1)


def test_duplicate_examples_are_counted_as_given() -> None:
    """The schema does not deduplicate; document that explicitly so a
    future dedup change has to break a test."""
    problem = make_problem(positives=["a", "a"], negatives=["b"])
    assert problem.num_pos == 2


# --- NCES datapoint conversion ---------------------------------------


def test_nces_datapoint_reduces_both_separator_styles() -> None:
    problem = make_problem(
        positives=["http://ex.org/kb#stefan", "http://ex.org/kb/markus"],
        negatives=["http://ex.org/kb#anna"],
    )
    _, examples = problem.as_nces_datapoint()
    assert examples["positive examples"] == ["stefan", "markus"]
    assert examples["negative examples"] == ["anna"]


def test_nces_datapoint_preserves_order() -> None:
    """NCES is configured with sorted_examples=True upstream, but the
    datapoint itself must not reorder — ordering is the caller's
    contract."""
    positives = ["http://x#c", "http://x#a", "http://x#b"]
    _, examples = make_problem(positives=positives).as_nces_datapoint()
    assert examples["positive examples"] == ["c", "a", "b"]


def test_nces_datapoint_target_is_the_dl_string() -> None:
    problem = LearningProblem(
        id="lp_0000",
        target_concept="male ⊓ ∃ hasChild.person",
        pos_example=["http://x#p"],
        neg_example=["http://x#n"],
        complexity=blank_complexity(dl_length=4),
    )
    name, _ = problem.as_nces_datapoint()
    assert name == "male ⊓ ∃ hasChild.person"


# --- to_dict / from_dict round trip ----------------------------------


def test_to_dict_nests_complexity_as_object() -> None:
    payload = make_problem().to_dict()
    assert isinstance(payload["complexity"], dict)
    assert payload["complexity"]["dl_length"] == 1


def test_to_dict_is_json_serializable() -> None:
    payload = make_problem().to_dict()
    assert json.loads(json.dumps(payload, ensure_ascii=False)) == payload


def test_from_dict_round_trip_preserves_complexity() -> None:
    original = make_problem(complexity=blank_complexity(dl_length=7, depth=2))
    restored = LearningProblem.from_dict(original.to_dict())
    assert restored == original


def test_from_dict_coerces_id_to_string() -> None:
    restored = LearningProblem.from_dict(
        {
            "id": 12,
            "target_concept": "male",
            "pos_example": ["a"],
            "neg_example": ["b"],
            "complexity": blank_complexity(dl_length=1).to_dict(),
        }
    )
    assert restored.id == "12"


def test_from_dict_copies_example_lists() -> None:
    """A mutation of the source payload must not reach the problem."""
    positives = ["http://x#p"]
    payload = {
        "id": "lp_0000",
        "target_concept": "male",
        "pos_example": positives,
        "neg_example": ["http://x#n"],
        "complexity": blank_complexity(dl_length=1).to_dict(),
    }
    problem = LearningProblem.from_dict(payload)
    positives.append("http://x#q")
    assert problem.pos_example == ["http://x#p"]


def test_from_dict_rejects_degenerate_payload() -> None:
    with pytest.raises(ValueError):
        LearningProblem.from_dict(
            {
                "id": "lp_0000",
                "target_concept": "male",
                "pos_example": ["a"],
                "neg_example": [],
                "complexity": blank_complexity(dl_length=1).to_dict(),
            }
        )