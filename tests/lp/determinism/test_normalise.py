"""_normalise must produce a content-addressed, order-independent list."""

from __future__ import annotations

import pytest

from src.data.lp import _expand, _normalise

NS = "http://example.org/kb#"


def raw_payload() -> dict:
    return {
        "A ⊓ B": {
            "positive examples": ["i1", "i2"],
            "negative examples": ["i3"],
        },
        "∃ r.C": {
            "positive examples": ["i4"],
            "negative examples": ["i5", "i6"],
        },
        "D": {
            "positive examples": ["i7"],
            "negative examples": ["i8"],
        },
    }


class TestNormalise:
    def test_ids_are_independent_of_input_order(self):
        payload = raw_payload()
        forward = _normalise(payload, namespace=NS, seed=1)
        reversed_payload = dict(reversed(list(payload.items())))
        backward = _normalise(reversed_payload, namespace=NS, seed=1)
        assert {p.id for p in forward} == {p.id for p in backward}

    def test_ids_are_independent_of_seed(self):
        payload = raw_payload()
        a = _normalise(payload, namespace=NS, seed=1)
        b = _normalise(payload, namespace=NS, seed=999)
        assert {p.id: p.target_concept for p in a} == {
            p.id: p.target_concept for p in b
        }

    def test_concept_to_id_mapping_survives_added_problems(self):
        """The core cross-run guarantee: ids do not renumber."""
        payload = raw_payload()
        before = {p.target_concept: p.id for p in _normalise(payload, namespace=NS, seed=1)}

        payload["E ⊔ F"] = {
            "positive examples": ["i9"],
            "negative examples": ["i10"],
        }
        after = {p.target_concept: p.id for p in _normalise(payload, namespace=NS, seed=1)}

        for concept, problem_id in before.items():
            assert after[concept] == problem_id

    def test_concept_to_id_mapping_survives_dropped_problems(self):
        payload = raw_payload()
        before = {p.target_concept: p.id for p in _normalise(payload, namespace=NS, seed=1)}
        del payload["D"]
        after = {p.target_concept: p.id for p in _normalise(payload, namespace=NS, seed=1)}
        assert all(after[c] == i for c, i in before.items() if c != "D")

    def test_degenerate_problems_are_skipped(self):
        payload = raw_payload()
        payload["Empty"] = {"positive examples": [], "negative examples": ["i9"]}
        payload["AlsoEmpty"] = {"positive examples": ["i9"], "negative examples": []}
        problems = _normalise(payload, namespace=NS, seed=1)
        concepts = {p.target_concept for p in problems}
        assert "Empty" not in concepts
        assert "AlsoEmpty" not in concepts
        assert len(problems) == 3

    def test_dropping_degenerate_does_not_shift_other_ids(self):
        """The enumerate-skip bug regression test."""
        clean = raw_payload()
        before = {p.target_concept: p.id for p in _normalise(clean, namespace=NS, seed=1)}

        with_degenerate = raw_payload()
        with_degenerate["Zzz"] = {
            "positive examples": [],
            "negative examples": ["i9"],
        }
        after = {
            p.target_concept: p.id
            for p in _normalise(with_degenerate, namespace=NS, seed=1)
        }
        assert before == after

    def test_examples_are_sorted(self):
        problems = _normalise(raw_payload(), namespace=NS, seed=1)
        for problem in problems:
            assert problem.pos_example == sorted(problem.pos_example)
            assert problem.neg_example == sorted(problem.neg_example)

    def test_output_sorted_by_id(self):
        problems = _normalise(raw_payload(), namespace=NS, seed=1)
        assert [p.id for p in problems] == sorted(p.id for p in problems)

    def test_accepts_list_payload(self):
        payload = list(raw_payload().items())
        problems = _normalise(payload, namespace=NS, seed=1)
        assert len(problems) == 3

    def test_list_and_dict_payloads_agree(self):
        as_dict = _normalise(raw_payload(), namespace=NS, seed=1)
        as_list = _normalise(list(raw_payload().items()), namespace=NS, seed=1)
        assert [p.id for p in as_dict] == [p.id for p in as_list]

    def test_namespace_is_applied(self):
        problems = _normalise(raw_payload(), namespace=NS, seed=1)
        for problem in problems:
            assert all(iri.startswith(NS) for iri in problem.pos_example)

    def test_namespace_change_changes_ids(self):
        """Ids are content-addressed, so a namespace bug is visible."""
        a = {p.id for p in _normalise(raw_payload(), namespace=NS, seed=1)}
        b = {
            p.id
            for p in _normalise(raw_payload(), namespace="http://other#", seed=1)
        }
        assert a != b


class TestExpand:
    def test_local_names_get_namespace(self):
        assert _expand(["x"], NS) == [f"{NS}x"]

    @pytest.mark.parametrize("iri", ["http://a#b", "https://a#b"])
    def test_full_iris_pass_through(self, iri):
        assert _expand([iri], NS) == [iri]

    def test_empty_namespace_leaves_names_bare(self):
        assert _expand(["x"], "") == ["x"]

    def test_order_is_preserved(self):
        assert _expand(["c", "a", "b"], NS) == [f"{NS}c", f"{NS}a", f"{NS}b"]