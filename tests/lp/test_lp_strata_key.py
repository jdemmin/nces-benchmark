# tests/lp/test_lp_strata_key.py
"""Strata-key resolution for stratified splitting."""

from __future__ import annotations

import pytest
from test_lp_annotation import hardness
from test_lp_schema import blank_complexity, make_problem

from src.data.complexity import Hardness
from src.data.lp import _get_strata_key


def test_none_problem_raises() -> None:
    with pytest.raises(AttributeError, match="Problem is None"):
        _get_strata_key("depth", None)


def test_structural_field_is_used_directly() -> None:
    problem = make_problem(complexity=blank_complexity(dl_length=4, depth=2))
    assert _get_strata_key("depth", problem) == "2"


def test_unknown_field_falls_back_to_dl_length() -> None:
    problem = make_problem(complexity=blank_complexity(dl_length=4, depth=2))
    assert _get_strata_key("not_a_field", problem) == "4"


def test_none_stratify_by_falls_back_to_dl_length() -> None:
    problem = make_problem(complexity=blank_complexity(dl_length=4, depth=2))
    assert _get_strata_key(None, problem) == "4"


def test_hardness_field_falls_back_when_unannotated() -> None:
    """Hardness is null before the annotation stage, so a hardness-keyed
    split must degrade to dl_length rather than produce 'None' buckets."""
    problem = make_problem(complexity=blank_complexity(dl_length=4))
    assert _get_strata_key("extension_ratio", problem) == "4"


def test_hardness_field_is_used_once_annotated() -> None:
    problem = make_problem(complexity=blank_complexity(dl_length=4)).annotate_hardness(
        blank_complexity(dl_length=4), hardness(ratio=0.25)
    )
    assert _get_strata_key("extension_ratio", problem) == "0.25"


def test_hardness_gate_keys_off_atomic_baseline_f1() -> None:
    """The validity check inspects atomic_baseline_f1 specifically, so a
    hardness object with that field null falls back even if the requested
    field is populated."""
    problem = make_problem(complexity=blank_complexity(dl_length=4)).annotate_hardness(
        blank_complexity(dl_length=4),
        Hardness(
            extension_size=4,
            extension_ratio=0.25,
            atomic_baseline_f1=None,
            redundant=False,
        ),
    )
    assert _get_strata_key("extension_ratio", problem) == "4"


def test_key_is_always_a_string() -> None:
    problem = make_problem(complexity=blank_complexity(dl_length=4, depth=0))
    assert isinstance(_get_strata_key("depth", problem), str)


@pytest.mark.parametrize("field_name", ["dl_length", "depth", "expressivity"])
def test_all_structural_fields_are_addressable(field_name: str) -> None:
    problem = make_problem(complexity=blank_complexity(dl_length=4, depth=1))
    assert _get_strata_key(field_name, problem)