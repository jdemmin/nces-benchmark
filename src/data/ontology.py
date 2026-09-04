# src/data/ontology.py
"""OWL parsing, RDF-triple extraction, and knowledge-base introspection."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

from ontolearn.knowledge_base import KnowledgeBase
from owlapy.owl_reasoner import SyncReasoner

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


def load_knowledge_base(kb_path: Path, uses_structural_reasoner: bool = True):
    """Load an OWL file into an ``ontolearn`` knowledge base (lazy import)."""
    from ontolearn.knowledge_base import KnowledgeBase

    logger.info("Loading knowledge base from %s", kb_path)
    if uses_structural_reasoner:
        return KnowledgeBase(path=str(kb_path))
    return KnowledgeBase(path=str(kb_path), reasoner=SyncReasoner(ontology=str(kb_path), reasoner="HermiT"))


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
    triples.sort(key=lambda triple: (triple.subject, triple.predicate, triple.object))
    return triples


def individual_iris(knowledge_base: KnowledgeBase) -> list[str]:
    """Return every named-individual IRI in the knowledge base, sorted."""
    return sorted(individual.str for individual in knowledge_base.individuals())


def local_name_collisions(iris: Iterable[str]) -> dict[str, list[str]]:
    """Group distinct IRIs that share a local name.

    Per methodology sec:meth:kbs, colliding IRIs are dropped rather than
    silently merged, and the collision count is reported per knowledge base.
    Only names shared by more than one distinct IRI are returned.
    """
    groups: dict[str, set[str]] = {}
    for iri in iris:
        groups.setdefault(local_name(iri), set()).add(iri)
    return {
        name: sorted(members)
        for name, members in groups.items()
        if len(members) > 1
    }


def iter_atomic_concepts(knowledge_base: KnowledgeBase) -> Iterator[str]:
    """Yield the DL rendering of every atomic concept in the TBox."""
    from owlapy.render import DLSyntaxObjectRenderer

    renderer = DLSyntaxObjectRenderer()
    for owl_class in knowledge_base.ontology.classes_in_signature():
        yield renderer.render(owl_class)


def concept_extension(knowledge_base: KnowledgeBase, dl_expression: str) -> frozenset[str]:
    """Return the extension of a DL expression as a set of individual IRIs.

    The expression is parsed back into an OWL class expression so that the
    reasoner, not string matching, determines membership. Returns an empty
    set when the expression cannot be parsed.
    """
    from owlapy.parser import DLSyntaxParser

    namespace = _guess_namespace(knowledge_base)
    parser = DLSyntaxParser(namespace=namespace)
    try:
        expression = parser.parse_expression(dl_expression)
    except Exception:  # noqa: BLE001 - upstream raises bare exceptions
        logger.debug("Could not parse DL expression %r", dl_expression)
        return frozenset()

    return frozenset(
        individual.str for individual in knowledge_base.individuals(expression)
    )

def compute_atomic_class_extensions(
    knowledge_base: KnowledgeBase,
) -> dict[str, frozenset[str]]:
    """Return the extension of every named class in the TBox.

    Keyed by the class's DL rendering, so keys line up with the vocabulary
    used in target concepts and in ``iter_atomic_concepts``. Unlike
    ``concept_extension`` this does not round-trip through DL syntax: the OWL
    class objects are handed to the reasoner directly.

    ``owl:Thing`` and ``owl:Nothing`` are excluded. Thing's extension is the
    whole universe, which would make the atomic baseline trivially high for
    any large target concept, and Nothing's is empty and never overlaps.

    One reasoner query per class. Compute once per knowledge base and reuse:
    this is the added cost of the hardness-annotation stage.
    """
    from owlapy.class_expression import OWLClass
    from owlapy.render import DLSyntaxObjectRenderer

    renderer = DLSyntaxObjectRenderer()
    extensions: dict[str, frozenset[str]] = {}

    for owl_class in knowledge_base.ontology.classes_in_signature():
        if not isinstance(owl_class, OWLClass):
            continue
        if owl_class.is_owl_thing() or owl_class.is_owl_nothing():
            continue
        try:
            extension = frozenset(
                individual.str
                for individual in sorted(knowledge_base.individuals(owl_class))
            )
        except Exception as error:  # noqa: BLE001 - upstream raises bare exceptions
            logger.warning(
                "Could not compute extension of %s: %s: %s",
                owl_class,
                type(error).__name__,
                error,
            )
            continue
        extensions[renderer.render(owl_class)] = extension

    logger.info("Computed extensions for %d atomic classes", len(extensions))
    return extensions

def _guess_namespace(knowledge_base: KnowledgeBase) -> str:
    """Infer the default namespace from the first individual's IRI."""
    
    for individual in knowledge_base.individuals():
        iri = individual.str
        for separator in ("#", "/"):
            if separator in iri:
                return iri.rsplit(separator, 1)[0] + separator
    return "http://example.com/SOMETHING_WENT_WRONG#"
