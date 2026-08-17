# tests/test_atomic_extensions.py
"""compute_atomic_class_extensions against a real knowledge base."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("ontolearn")

from src.data.ontology import (  # noqa: E402
    compute_atomic_class_extensions,
    individual_iris,
    load_knowledge_base,
)

KB_PATH = Path("datasets/semantic_bible.owl")
pytestmark = pytest.mark.skipif(
    not KB_PATH.is_file(), reason="datasets/semantic_bible.owl not present"
)


@pytest.fixture(scope="module")
def knowledge_base():
    return load_knowledge_base(KB_PATH)


def test_returns_one_entry_per_named_class(knowledge_base):
    extensions = compute_atomic_class_extensions(knowledge_base)
    assert extensions
    assert all(isinstance(key, str) for key in extensions)


def test_excludes_thing_and_nothing(knowledge_base):
    extensions = compute_atomic_class_extensions(knowledge_base)
    assert "⊤" not in extensions
    assert "⊥" not in extensions


def test_extensions_are_iri_strings(knowledge_base):
    extensions = compute_atomic_class_extensions(knowledge_base)
    for extension in extensions.values():
        assert all(iri.startswith("http") for iri in extension)


def test_extensions_are_subsets_of_the_universe(knowledge_base):
    universe = set(individual_iris(knowledge_base))
    for extension in compute_atomic_class_extensions(knowledge_base).values():
        assert extension <= universe


def test_deterministic_across_calls(knowledge_base):
    assert compute_atomic_class_extensions(
        knowledge_base
    ) == compute_atomic_class_extensions(knowledge_base)