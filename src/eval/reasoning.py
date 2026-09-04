# src/eval/reasoning.py
"""Reasoner-backed extension computation with a persistent cache.

Reasoning uses OWLAPY's ``SyncReasoner`` (HermiT), which is sound and complete
for SROIQ and therefore complete for every expressivity class the generator
can produce. Completeness is load-bearing: an incomplete reasoner would
systematically deflate both ``semantic_equivalence`` and
``atomic_baseline_f1``.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Self

logger = logging.getLogger(__name__)


class UnparseableExpression(RuntimeError):
    """Raised when a hypothesis string cannot be parsed into a class expression."""


class ExtensionOracle:
    """Compute and cache class-expression extensions for one knowledge base.

    Extensions are keyed by expression string and shared between target
    concepts and hypotheses, so the same expression is never reasoned over
    twice within a run.
    """

    def __init__(
        self,
        *,
        ontology_path: Path,
        cache_path: Path | None = None,
    ) -> None:
        self._ontology_path = ontology_path
        self._cache_path = cache_path
        self._cache: dict[str, frozenset[str]] = {}
        self._reasoner: Any | None = None
        self._parser: Any | None = None
        self._universe: frozenset[str] | None = None
        self._atomic: dict[str, frozenset[str]] | None = None
        if cache_path is not None and cache_path.is_file():
            self._load_cache(cache_path)

    # -- lifecycle ---------------------------------------------------------

    def _ensure_reasoner(self) -> None:
        if self._reasoner is not None:
            return
        from owlapy.class_expression import OWLClass
        from owlapy.iri import IRI
        from owlapy.owl_ontology import Ontology
        from owlapy.owl_reasoner import SyncReasoner
        from owlapy.parser import DLSyntaxParser

        ontology = Ontology(str(self._ontology_path), load=True)
        self._reasoner = SyncReasoner(str(self._ontology_path))
        namespace = _default_namespace(ontology)
        self._parser = DLSyntaxParser(namespace)
        self._ontology = ontology
        self._owl_class = OWLClass
        self._iri = IRI

    def close(self) -> None:
        """Persist the cache. Reasoner handles are dropped."""
        if self._cache_path is not None:
            self._write_cache(self._cache_path)
        self._reasoner = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    # -- extensions --------------------------------------------------------

    @property
    def universe(self) -> frozenset[str]:
        """All named individuals; the universe ``U`` of every metric."""
        if self._universe is None:
            self._ensure_reasoner()
            self._universe = frozenset(
                str(individual.str)
                for individual in self._ontology.individuals_in_signature()
            )
        return self._universe

    @property
    def atomic_extensions(self) -> dict[str, frozenset[str]]:
        """Extensions of all named classes, excluding Thing and Nothing."""
        if self._atomic is None:
            self._ensure_reasoner()
            skip = {
                "http://www.w3.org/2002/07/owl#Thing",
                "http://www.w3.org/2002/07/owl#Nothing",
            }
            extensions: dict[str, frozenset[str]] = {}
            for owl_class in self._ontology.classes_in_signature():
                iri = str(owl_class.str)
                if iri in skip:
                    continue
                extensions[iri] = frozenset(
                    str(i.str)
                    for i in self._reasoner.instances(owl_class, direct=False)
                )
            self._atomic = extensions
        return self._atomic

    def extension(self, expression: str) -> frozenset[str]:
        """Return the extension of a rendered class expression.

        Raises :class:`UnparseableExpression` when the string is not a valid
        class expression, which is the expected outcome for a malformed
        synthesised hypothesis.
        """
        if expression in self._cache:
            return self._cache[expression]

        self._ensure_reasoner()
        try:
            parsed = self._parser.parse_expression(expression)
        except Exception as exc:  # owlapy raises bare exceptions here
            raise UnparseableExpression(expression) from exc

        result = frozenset(
            str(i.str) for i in self._reasoner.instances(parsed, direct=False)
        )
        self._cache[expression] = result
        return result

    def extension_or_none(self, expression: str) -> frozenset[str] | None:
        """Non-raising variant, for hypotheses that may be malformed."""
        if not expression or not expression.strip():
            return frozenset()
        try:
            return self.extension(expression)
        except UnparseableExpression:
            logger.debug("Unparseable expression: %r", expression)
            return None

    def prime(self, expressions: Iterator[str]) -> None:
        """Warm the cache; used by the hardness-annotation stage."""
        for expression in expressions:
            self.extension_or_none(expression)

    # -- cache i/o ---------------------------------------------------------

    def _load_cache(self, path: Path) -> None:
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        self._cache = {k: frozenset(v) for k, v in payload.items()}
        logger.info("Loaded %d cached extensions from %s", len(self._cache), path)

    def _write_cache(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {k: sorted(v) for k, v in sorted(self._cache.items())}
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False)
        logger.info("Wrote %d cached extensions to %s", len(payload), path)


def _default_namespace(ontology: Any) -> str:
    """Best-effort default namespace for the DL-syntax parser."""
    iri = str(ontology.get_ontology_id().get_ontology_iri().str)
    return iri if iri.endswith(("#", "/")) else iri + "#"