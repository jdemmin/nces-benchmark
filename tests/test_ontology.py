# tests/test_ontology.py
from __future__ import annotations

from pathlib import Path

import pytest

from src.data.ontology import (
    concept_extension,
    individual_iris,
    iter_atomic_concepts,
    load_knowledge_base,
)

pytest.importorskip("ontolearn")


def test_individuals_are_discovered(kb_path: Path) -> None:
    kb = load_knowledge_base(kb_path)
    names = {iri.rsplit("#", 1)[-1] for iri in individual_iris(kb)}
    assert {"stefan", "markus", "anna", "heinz", "michelle"} <= names


def test_atomic_concepts_include_tbox_classes(kb_path: Path) -> None:
    kb = load_knowledge_base(kb_path)
    assert {"male", "female"} <= set(iter_atomic_concepts(kb))


def test_extension_of_atomic_concept(kb_path: Path) -> None:
    kb = load_knowledge_base(kb_path)
    extension = {n.rsplit("#", 1)[-1] for n in concept_extension(kb, "male")}
    assert extension == {"stefan", "markus", "heinz"}


def test_unparsable_expression_yields_empty_extension(kb_path: Path) -> None:
    kb = load_knowledge_base(kb_path)
    assert concept_extension(kb, "((( not dl ⊓") == frozenset()
