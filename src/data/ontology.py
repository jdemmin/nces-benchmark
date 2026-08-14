# src/data/ontology.py
"""OWL parsing, RDF-triple extraction, and knowledge-base introspection."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

logger = logging.getLogger(__name__)

#: Predicates that carry no learnable ABox signal for embedding training.
_SKIPPED_PREDICATES = frozenset(
    {
        "http://www.w3.org/2002/07/owl#topObjectProperty",
        "http://www.w3.org/2002/07/owl#bottomObjectProperty",
    }
)


@dataclass(frozen=True)
class Triple:
    """One ``(subject, predicate, object)`` RDF triple."""

    subject: str
    predicate: str
    object: str

    def as_tuple(self) -> tuple[str, str, str]:
        return (self.subject, self.predicate, self.object)


def local_name(iri: str) -> str:
    """Return the local name of an IRI.

    NCES vocabularies are keyed on local names, so every IRI crossing into
    NCES must pass through this function.
    """
    for separator in ("#", "/"):
        if separator in iri:
            candidate = iri.rsplit(separator, 1)[-1]
            if candidate:
                return candidate
    return iri


def load_knowledge_base(kb_path: Path):
    """Load an OWL file into an ``ontolearn`` knowledge base (lazy import)."""
    from ontolearn.knowledge_base import KnowledgeBase

    logger.info("Loading knowledge base from %s", kb_path)
    return KnowledgeBase(path=str(kb_path))


def parse_triples(kb_path: Path) -> list[Triple]:
    """Parse an OWL/RDF-XML knowledge base into RDF triples.

    Only IRI-to-IRI statements are retained: literals cannot be embedded as
    entities by DICE and would pollute the entity index mapping.
    """
    from rdflib import Graph, URIRef

    graph = Graph()
    graph.parse(str(kb_path))

    triples: list[Triple] = []
    for subject, predicate, obj in graph:
        if not isinstance(subject, URIRef) or not isinstance(obj, URIRef):
            continue
        if str(predicate) in _SKIPPED_PREDICATES:
            continue
        triples.append(Triple(str(subject), str(predicate), str(obj)))

    logger.info("Parsed %d RDF triples from %s", len(triples), kb_path.name)
    return triples


def individual_iris(knowledge_base) -> list[str]:
    """Return every named-individual IRI in the knowledge base, sorted."""
    return sorted(individual.str for individual in knowledge_base.individuals())


def iter_atomic_concepts(knowledge_base) -> Iterator[str]:
    """Yield the DL rendering of every atomic concept in the TBox."""
    from owlapy.render import DLSyntaxObjectRenderer

    renderer = DLSyntaxObjectRenderer()
    for owl_class in knowledge_base.ontology.classes_in_signature():
        yield renderer.render(owl_class)


def concept_extension(knowledge_base, dl_expression: str) -> frozenset[str]:
    """Return the extension of a DL expression as a set of individual IRIs.

    The expression is parsed back into an OWL class expression so that the
    reasoner, not string matching, determines membership. Returns an empty
    set when the expression cannot be parsed.
    """
    from owlapy.parser import DLSyntaxParser

    namespace = _guess_namespace(knowledge_base)
    parser = DLSyntaxParser(namespace_or_prefix_map=namespace)
    try:
        expression = parser.parse_expression(dl_expression)
    except Exception:  # noqa: BLE001 - upstream raises bare exceptions
        logger.debug("Could not parse DL expression %r", dl_expression)
        return frozenset()

    return frozenset(
        individual.str for individual in knowledge_base.individuals(expression)
    )


def _guess_namespace(knowledge_base) -> str:
    """Infer the default namespace from the first individual's IRI."""
    for individual in knowledge_base.individuals():
        iri = individual.str
        for separator in ("#", "/"):
            if separator in iri:
                return iri.rsplit(separator, 1)[0] + separator
    return "http://example.com/father#"
