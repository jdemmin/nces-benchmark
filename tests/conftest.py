# tests/conftest.py
"""Shared fixtures. A tiny synthetic knowledge base keeps tests fast."""

from __future__ import annotations

from pathlib import Path

import pytest

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


@pytest.fixture
def kb_path(tmp_path: Path) -> Path:
    path = tmp_path / "father.owl"
    path.write_text(FATHER_OWL, encoding="utf-8")
    return path


@pytest.fixture
def problems():
    from src.data.lp import LearningProblem

    namespace = "http://example.com/father#"
    return [
        LearningProblem(
            id="lp_0000",
            target_concept="male",
            pos_example=[f"{namespace}stefan", f"{namespace}markus"],
            neg_example=[f"{namespace}anna", f"{namespace}michelle"],
            complexity=1,
        ),
        LearningProblem(
            id="lp_0001",
            target_concept="∃ hasChild.male",
            pos_example=[f"{namespace}stefan", f"{namespace}anna"],
            neg_example=[f"{namespace}markus"],
            complexity=3,
        ),
    ]
