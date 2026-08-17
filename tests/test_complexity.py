# tests/test_complexity.py
"""Structural complexity, hardness annotation, and the atomic baseline."""

from __future__ import annotations

import pytest

from src.data.complexity import (
    Complexity,
    Hardness,
    annotate_hardness,
    atomic_baseline_f1,
    structural_complexity,
)


class TestStructuralComplexity:
    def test_atomic_concept(self):
        complexity = structural_complexity("male")
        assert complexity.dl_length == 1
        assert complexity.depth == 0
        assert complexity.constructors == {}
        assert complexity.num_atomic_classes == 1
        assert complexity.num_roles == 0
        assert complexity.expressivity == "EL"

    def test_flat_conjunction_has_zero_depth(self):
        complexity = structural_complexity("male ⊓ person")
        assert complexity.depth == 0
        assert complexity.constructors == {"⊓": 1}
        assert complexity.num_atomic_classes == 2

    def test_role_is_not_counted_as_atomic_class(self):
        complexity = structural_complexity("∃ hasChild.person")
        assert complexity.num_roles == 1
        assert complexity.num_atomic_classes == 1
        assert complexity.depth == 1

    def test_nested_restriction_depth(self):
        complexity = structural_complexity("∃ hasChild.(male ⊓ ∃ hasChild.person)")
        assert complexity.depth == 2

    def test_sibling_restrictions_do_not_accumulate_depth(self):
        """Two restrictions in a conjunction are both at depth 1, not 2."""
        complexity = structural_complexity(
            "(∃ hasChild.male) ⊓ (∃ hasSibling.female)"
        )
        assert complexity.depth == 1
        assert complexity.num_roles == 2

    @pytest.mark.parametrize(
        "expression,expected",
        [
            ("male ⊓ person", "EL"),
            ("∃ hasChild.person", "EL"),
            ("male ⊔ female", "ALC"),
            ("¬male", "ALC"),
            ("∀ hasChild.person", "ALC"),
            ("≥ 2 hasChild.person", "ALCHIQD"),
        ],
    )
    def test_expressivity_class(self, expression, expected):
        assert structural_complexity(expression).expressivity == expected

    def test_cardinality_numeral_is_not_an_atomic_class(self):
        """Regression: the old token counter scored the bound as an atom."""
        complexity = structural_complexity("≥ 2 hasChild.person")
        assert complexity.num_atomic_classes == 1
        assert complexity.num_roles == 1
        assert complexity.constructors == {"≥": 1}

    def test_repeated_class_counted_once(self):
        complexity = structural_complexity("person ⊓ ∃ hasChild.person")
        assert complexity.num_atomic_classes == 1


class TestComplexitySerialization:
    def test_round_trip(self):
        original = structural_complexity("male ⊓ ∃ hasChild.person")
        assert Complexity.from_dict(original.to_dict()) == original

    def test_v1_integer_is_read_as_dl_length(self):
        complexity = Complexity.from_dict(4)
        assert complexity.dl_length == 4
        assert complexity.expressivity == "unknown"
        assert complexity.is_annotated is False

    def test_structural_complexity_is_not_annotated(self):
        assert structural_complexity("male").is_annotated is False

    def test_with_hardness_marks_annotated(self):
        annotated = structural_complexity("male").with_hardness(
            hardness=Hardness(
                extension_size=3,
                extension_ratio=0.5,
                atomic_baseline_f1=0.8,
                redundant=False,
            )
        )
        assert annotated.is_annotated is True
        assert annotated.dl_length == 1  # structural fields survive


class TestAtomicBaselineF1:
    def test_redundant_target_scores_one(self):
        score, winner = atomic_baseline_f1(
            {"a", "b"}, {"male": {"a", "b"}, "female": {"c"}}
        )
        assert score == pytest.approx(1.0)
        assert winner == "male"

    def test_disjoint_target_scores_zero(self):
        score, winner = atomic_baseline_f1({"x"}, {"male": {"a", "b"}})
        assert score == 0.0
        assert winner is None

    def test_empty_target_scores_zero(self):
        assert atomic_baseline_f1(set(), {"male": {"a"}}) == (0.0, None)

    def test_no_atomic_classes_scores_zero(self):
        assert atomic_baseline_f1({"a"}, {}) == (0.0, None)

    def test_picks_best_of_several(self):
        score, winner = atomic_baseline_f1(
            {"a", "b", "c"},
            {"narrow": {"a"}, "close": {"a", "b", "c", "d"}, "wrong": {"z"}},
        )
        # close: p=3/4, r=1.0 -> 0.857; narrow: p=1.0, r=1/3 -> 0.5
        assert winner == "close"
        assert score == pytest.approx(6 / 7)

    def test_accepts_list_and_frozenset_inputs(self):
        """Callers pass whatever the reasoner returned; all must work."""
        score, _ = atomic_baseline_f1(["a", "b"], {"male": frozenset({"a", "b"})})
        assert score == pytest.approx(1.0)


class TestAnnotateHardness:
    @pytest.fixture
    def structural(self):
        return structural_complexity("male ⊓ ∃ hasChild.person")

    def test_populates_all_hardness_fields(self, structural):
        annotated = annotate_hardness(
            structural,
            target_extension={"a", "b"},
            all_individuals={"a", "b", "c", "d"},
            atomic_extensions={"male": {"a", "b", "c"}},
        )
        assert annotated.hardness.extension_size == 2
        assert annotated.hardness.extension_ratio == pytest.approx(0.5)
        assert annotated.hardness.atomic_baseline_f1 == pytest.approx(0.8)
        assert annotated.hardness.redundant is False
        assert annotated.is_annotated is True

    def test_empty_target_does_not_raise(self, structural):
        annotated = annotate_hardness(
            structural,
            target_extension=set(),
            all_individuals={"a", "b"},
            atomic_extensions={"male": {"a"}},
        )
        assert annotated.hardness.extension_size == 0
        assert annotated.hardness.extension_ratio == 0.0
        assert annotated.hardness.atomic_baseline_f1 == 0.0
        assert annotated.hardness.redundant is False

    def test_empty_universe_does_not_divide_by_zero(self, structural):
        annotated = annotate_hardness(
            structural,
            target_extension={"a"},
            all_individuals=set(),
            atomic_extensions={},
        )
        assert annotated.hardness.extension_ratio == 0.0

    def test_redundancy_detected_on_exact_match(self, structural):
        annotated = annotate_hardness(
            structural,
            target_extension={"a", "b"},
            all_individuals={"a", "b", "c"},
            atomic_extensions={"male": {"a", "b"}},
        )
        assert annotated.hardness.redundant is True
        assert annotated.hardness.atomic_baseline_f1 == pytest.approx(1.0)

    def test_structural_fields_are_preserved(self, structural):
        annotated = annotate_hardness(
            structural,
            target_extension={"a"},
            all_individuals={"a", "b"},
            atomic_extensions={},
        )
        assert annotated.dl_length == structural.dl_length
        assert annotated.depth == structural.depth
        assert annotated.constructors == structural.constructors
        assert annotated.expressivity == structural.expressivity

    def test_owl_objects_would_not_silently_match(self, structural):
        """Guards the IRI-string convention.

        Extensions are sets of IRI strings. If a caller passes OWL objects
        instead, nothing compares equal and hardness reads as zero -- so this
        documents the failure mode rather than tolerating it.
        """

        class FakeIndividual:
            def __init__(self, iri): self.iri = iri

        annotated = annotate_hardness(
            structural,
            target_extension={FakeIndividual("a")}, # type: ignore
            all_individuals={"a", "b"},
            atomic_extensions={"male": {"a"}},
        )
        assert annotated.hardness.atomic_baseline_f1 == 0.0