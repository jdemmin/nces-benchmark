"""_stable_id must be a pure function of learning-problem content."""

from __future__ import annotations

from src.data.lp import _stable_id


class TestStableId:
    def test_deterministic(self):
        assert _stable_id("A ⊓ B", ["x"], ["y"]) == _stable_id("A ⊓ B", ["x"], ["y"])

    def test_invariant_to_example_order(self):
        a = _stable_id("A", ["x", "y", "z"], ["p", "q"])
        b = _stable_id("A", ["z", "x", "y"], ["q", "p"])
        assert a == b

    def test_sensitive_to_concept(self):
        assert _stable_id("A", ["x"], ["y"]) != _stable_id("B", ["x"], ["y"])

    def test_sensitive_to_positives(self):
        assert _stable_id("A", ["x"], ["y"]) != _stable_id("A", ["x2"], ["y"])

    def test_sensitive_to_negatives(self):
        assert _stable_id("A", ["x"], ["y"]) != _stable_id("A", ["x"], ["y2"])

    def test_pos_neg_are_not_interchangeable(self):
        """The \\x00 separator must prevent field confusion."""
        assert _stable_id("A", ["x"], ["y"]) != _stable_id("A", ["y"], ["x"])

    def test_no_collision_from_concatenation(self):
        """['ab'] and ['a','b'] must not hash alike."""
        assert _stable_id("C", ["ab"], ["z"]) != _stable_id("C", ["a", "b"], ["z"])

    def test_unicode_dl_syntax_is_handled(self):
        assert _stable_id("∃ r.(A ⊓ ¬B)", ["x"], ["y"]).startswith("lp_")

    def test_format(self):
        problem_id = _stable_id("A", ["x"], ["y"])
        assert problem_id.startswith("lp_")
        assert len(problem_id) == len("lp_") + 12

    def test_collision_free_on_realistic_population(self):
        seen = {
            _stable_id(f"C{i}", [f"p{i}_{j}" for j in range(5)], [f"n{i}"])
            for i in range(20_000)
        }
        assert len(seen) == 20_000