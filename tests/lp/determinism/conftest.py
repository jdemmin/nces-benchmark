"""Shared fixtures for learning-problem determinism tests."""

from __future__ import annotations

import random
from typing import Any

import pytest

from src.data.complexity import Complexity, Hardness

NS = "http://example.org/kb#"

FATHER_OWL = """<?xml version="1.0"?>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
         xmlns:owl="http://www.w3.org/2002/07/owl#"
         xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#"
         xml:base="http://example.com/father#"
         xmlns="http://example.com/father#">
  <owl:Ontology rdf:about="http://example.com/father"/>
  <owl:Class rdf:about="#person"/>
  <owl:Class rdf:about="#male"><rdfs:subClassOf rdf:resource="#person"/></owl:Class>
  <owl:Class rdf:about="#female"><rdfs:subClassOf rdf:resource="#person"/></owl:Class>
  <owl:ObjectProperty rdf:about="#hasChild"/>
  <owl:NamedIndividual rdf:about="#stefan"><rdf:type rdf:resource="#male"/>
    <hasChild rdf:resource="#markus"/></owl:NamedIndividual>
  <owl:NamedIndividual rdf:about="#markus"><rdf:type rdf:resource="#male"/>
  </owl:NamedIndividual>
  <owl:NamedIndividual rdf:about="#anna"><rdf:type rdf:resource="#female"/>
    <hasChild rdf:resource="#heinz"/></owl:NamedIndividual>
  <owl:NamedIndividual rdf:about="#heinz"><rdf:type rdf:resource="#male"/>
  </owl:NamedIndividual>
  <owl:NamedIndividual rdf:about="#michelle"><rdf:type rdf:resource="#female"/>
  </owl:NamedIndividual>
</rdf:RDF>
"""

def make_complexity(
    *,
    dl_length: int = 3,
    depth: int = 1,
    constructors: dict[str, int] | None = None,
    num_atomic_classes: int = 2,
    num_roles: int = 1,
    expressivity: str = "EL",
    hardness: Hardness | None = None,
) -> Complexity:
    return Complexity(
        dl_length=dl_length,
        depth=depth,
        constructors=dict(constructors or {"⊓": 1}),
        num_atomic_classes=num_atomic_classes,
        num_roles=num_roles,
        expressivity=expressivity,
        hardness=hardness or Hardness.get_blank_hardness(),
    )


def make_hardness(
    *,
    extension_size: int = 10,
    extension_ratio: float = 0.25,
    atomic_baseline_f1: float = 0.5,
    redundant: bool = False,
) -> Hardness:
    return Hardness(
        extension_size=extension_size,
        extension_ratio=extension_ratio,
        atomic_baseline_f1=atomic_baseline_f1,
        redundant=redundant,
    )


def make_problem(
    concept: str,
    *,
    dl_length: int = 3,
    depth: int = 1,
    n_pos: int = 3,
    n_neg: int = 3,
    hardness: Hardness | None = None,
):
    """Build a LearningProblem with a content-derived id."""
    from src.data.lp import LearningProblem, _stable_id

    positives = [f"{NS}pos_{concept}_{i}" for i in range(n_pos)]
    negatives = [f"{NS}neg_{concept}_{i}" for i in range(n_neg)]
    return LearningProblem(
        id=_stable_id(concept, positives, negatives),
        target_concept=concept,
        pos_example=sorted(positives),
        neg_example=sorted(negatives),
        complexity=make_complexity(
            dl_length=dl_length, depth=depth, hardness=hardness
        ),
    )


def make_problems(
    count: int, *, lengths: tuple[int, ...] = (1, 2, 3, 5, 10), annotate: bool = False
) -> list[Any]:
    """A deterministic population spread over several dl_length strata."""
    rng = random.Random(0xC0FFEE)
    problems = []
    for i in range(count):
        length = lengths[i % len(lengths)]
        hardness = (
            make_hardness(
                extension_size=i + 1,
                extension_ratio=rng.random(),
                atomic_baseline_f1=rng.random(),
            )
            if annotate
            else None
        )
        problems.append(
            make_problem(
                f"C{i:03d}",
                dl_length=length,
                depth=1 + (i % 3),
                hardness=hardness,
            )
        )
    return problems


@pytest.fixture
def population():
    return make_problems(60)


@pytest.fixture
def annotated_population():
    return make_problems(60, annotate=True)


# def pytest_configure(config):
#     """Re-exec under a fixed hash seed so set iteration order is stable.

#     Ontolearn's LPGen samples from unordered sets of IRI strings, so
#     PYTHONHASHSEED leaks into the generated learning problems.
#     """
#     if os.environ.get("PYTHONHASHSEED") != "0":
#         os.environ["PYTHONHASHSEED"] = "0"
#         os.execv(sys.executable, [sys.executable, "-m", "pytest", *sys.argv[1:]])