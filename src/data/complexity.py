# src/data/complexity.py
"""Multi-dimensional complexity characterisation of target concepts.

Complexity has two halves. *Structural* fields are computed from the concept
expression alone and are available at generation time. *Hardness* fields
require the reasoner and are populated later, by the hardness-annotation
stage.

Hardness must never depend on any embedding-derived quantity: it describes the
learning problem, which is held constant across embedding conditions.
"""

from __future__ import annotations

import logging
from collections.abc import Collection, Mapping
from dataclasses import asdict, dataclass
from typing import Any

logger = logging.getLogger(__name__)

#: DL constructor tokens, as rendered by owlapy's DL renderer.
CONJUNCTION = "⊓"
DISJUNCTION = "⊔"
NEGATION = "¬"
EXISTENTIAL = "∃"
UNIVERSAL = "∀"
MIN_CARDINALITY = "≥"
MAX_CARDINALITY = "≤"
EXACT_CARDINALITY = "="

CONSTRUCTORS = frozenset({
    CONJUNCTION, DISJUNCTION, NEGATION, EXISTENTIAL,
    UNIVERSAL, MIN_CARDINALITY, MAX_CARDINALITY, EXACT_CARDINALITY,
})

#: Constructors outside EL.
_BEYOND_EL = frozenset({
    DISJUNCTION, NEGATION, UNIVERSAL,
    MIN_CARDINALITY, MAX_CARDINALITY, EXACT_CARDINALITY,
})

#: Constructors outside ALC.
_BEYOND_ALC = frozenset({
    MIN_CARDINALITY, MAX_CARDINALITY, EXACT_CARDINALITY,
})

@dataclass(frozen=True)
class Hardness:
    """Reasoner-derived characterisation of a target concept's extension."""

    extension_size: int | None
    extension_ratio: float | None
    atomic_baseline_f1: float | None
    redundant: bool | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Hardness:
        """Deserialize a hardness object."""
        return cls(
            extension_size=payload.get("extension_size"),
            extension_ratio=payload.get("extension_ratio"),
            atomic_baseline_f1=payload.get("atomic_baseline_f1"),
            redundant=payload.get("redundant"),
        )

    @staticmethod
    def get_blank_hardness() -> Hardness:
        """Return a copy with the hardness fields blanked out."""
        return Hardness(
            extension_size=None,
            extension_ratio=None,
            atomic_baseline_f1=None,
            redundant=None,
        )

@dataclass(frozen=True)
class Complexity:
    """Structural and semantic characterisation of a target concept."""

    # Structural — from the expression alone.
    dl_length: int
    depth: int
    constructors: dict[str, int]
    num_atomic_classes: int
    num_roles: int
    expressivity: str
    hardness: Hardness

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def with_hardness(
        self,
        *,
        hardness: Hardness,
    ) -> Complexity:
        """Return a copy with the hardness fields populated."""
        return Complexity(
            dl_length=self.dl_length,
            depth=self.depth,
            constructors=dict(self.constructors),
            num_atomic_classes=self.num_atomic_classes,
            num_roles=self.num_roles,
            expressivity=self.expressivity,
            hardness=hardness,
        )

    @property
    def is_annotated(self) -> bool:
        """Whether the hardness fields have been populated."""
        # The hardness fields are all None when unpopulated, so we can check any one of them.
        return self.hardness.atomic_baseline_f1 is not None

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | int) -> Complexity:
        """Deserialize, accepting both schema v1 and v2.

        A bare integer is a v1 ``complexity`` and is read as a DL length with
        every other field unknown.
        """
        if isinstance(payload, int):
            return cls(
                dl_length=payload,
                depth=0,
                constructors={},
                num_atomic_classes=0,
                num_roles=0,
                expressivity="unknown",
                hardness=Hardness.get_blank_hardness(),
            )
        return cls(
            dl_length=int(payload["dl_length"]),
            depth=int(payload.get("depth", 0)),
            constructors=dict(payload.get("constructors", {})),
            num_atomic_classes=int(payload.get("num_atomic_classes", 0)),
            num_roles=int(payload.get("num_roles", 0)),
            expressivity=str(payload.get("expressivity", "unknown")),
            hardness=Hardness.from_dict(payload.get("hardness", Hardness.get_blank_hardness().to_dict())),
        )

def structural_complexity(dl_expression: str) -> Complexity:
    """Compute structural complexity from a rendered DL expression.

    Used when only the rendered string is available — for example when
    normalising ``LPs.json``, whose keys are strings. Prefer
    :func:`structural_complexity_from_owl` where the parsed expression exists.
    """
    tokens = _tokenize(dl_expression)

    constructors: dict[str, int] = {}
    atoms: set[str] = set()
    roles: set[str] = set()

    for index, token in enumerate(tokens):
        if token in CONSTRUCTORS:
            constructors[token] = constructors.get(token, 0) + 1
            continue
        if token in {"(", ")", "."}:
            continue
        if _is_numeral(token):
            # Cardinality bound; part of the restriction, not an atom.
            continue
        if _follows_quantifier(tokens, index):
            roles.add(token)
        else:
            atoms.add(token)

    atomic_classes = set(atoms)
    present = frozenset(constructors)

    return Complexity(
        dl_length=len(atomic_classes | roles) + sum(constructors.values()),
        depth=_nesting_depth(tokens),
        constructors=dict(sorted(constructors.items())),
        num_atomic_classes=len(atomic_classes),
        num_roles=len(roles),
        expressivity=_expressivity(present),
        hardness=Hardness.get_blank_hardness(),
    )


def _expressivity(present: frozenset[str]) -> str:
    if present & _BEYOND_ALC:
        return "ALCHIQD"
    if present & _BEYOND_EL:
        return "ALC"
    return "EL"


def _tokenize(dl_expression: str) -> list[str]:
    """Split a DL expression, keeping brackets and dots as separate tokens."""
    spaced = dl_expression
    for symbol in set(CONSTRUCTORS) | {"(", ")", "."}:
        spaced = spaced.replace(symbol, f" {symbol} ")
    return spaced.split()


def _is_numeral(token: str) -> bool:
    return token.isdigit()


def _follows_quantifier(tokens: list[str], index: int) -> bool:
    """Whether the token at ``index`` is the role of a restriction.

    A role is the token after a quantifier, or after a quantifier and its
    cardinality bound.
    """
    cursor = index - 1
    if cursor >= 0 and _is_numeral(tokens[cursor]):
        cursor -= 1
    return cursor >= 0 and tokens[cursor] in {
        EXISTENTIAL, UNIVERSAL, MIN_CARDINALITY,
        MAX_CARDINALITY, EXACT_CARDINALITY,
    }


def _nesting_depth(tokens: list[str]) -> int:
    """Maximum quantifier nesting depth.

    Depth increases on entering a restriction's filler and is tracked through
    bracketing, so ``∃ r.(A ⊓ ∃ s.B)`` yields 2 rather than 1.
    """
    max_depth = 0
    current = 0
    # Stack of bracket depths at which a quantifier scope was opened.
    scopes: list[int] = []
    brackets = 0

    for index, token in enumerate(tokens):
        if token == "(":
            brackets += 1
        elif token == ")":
            brackets -= 1
            while scopes and scopes[-1] > brackets:
                scopes.pop()
                current -= 1
        elif token in {
            EXISTENTIAL, UNIVERSAL, MIN_CARDINALITY,
            MAX_CARDINALITY, EXACT_CARDINALITY,
        }:
            current += 1
            scopes.append(brackets)
            max_depth = max(max_depth, current)
    return max_depth

def structural_complexity_from_owl(expression) -> Complexity:
    """Compute structural complexity by walking an OWL class expression.

    Preferred over the string-based path: the tree is unambiguous, whereas
    rendered DL syntax can omit brackets that disambiguate filler scope.
    """
    from owlapy.class_expression import (
        OWLObjectAllValuesFrom,
        OWLObjectCardinalityRestriction,
        OWLObjectComplementOf,
        OWLObjectIntersectionOf,
        OWLObjectSomeValuesFrom,
        OWLObjectUnionOf,
    )

    constructors: dict[str, int] = {}
    classes: set[str] = set()
    roles: set[str] = set()

    def bump(symbol: str) -> None:
        constructors[symbol] = constructors.get(symbol, 0) + 1

    def walk(node, depth: int) -> int:
        if isinstance(node, OWLObjectIntersectionOf):
            bump(CONJUNCTION)
            return max(walk(operand, depth) for operand in node.operands())
        if isinstance(node, OWLObjectUnionOf):
            bump(DISJUNCTION)
            return max(walk(operand, depth) for operand in node.operands())
        if isinstance(node, OWLObjectComplementOf):
            bump(NEGATION)
            return walk(node.get_operand(), depth)
        if isinstance(node, OWLObjectSomeValuesFrom):
            bump(EXISTENTIAL)
            roles.add(str(node.get_property().get_named_property()._iri))
            return walk(node.get_filler(), depth + 1)
        if isinstance(node, OWLObjectAllValuesFrom):
            bump(UNIVERSAL)
            roles.add(str(node.get_property().get_named_property()._iri))
            return walk(node.get_filler(), depth + 1)
        if isinstance(node, OWLObjectCardinalityRestriction):
            # TODO: distinguish min/max/exact cardinality restrictions and count them
            # separately for expressivity. For now, treat all as min cardinality.
            bump(MIN_CARDINALITY)  # refine per concrete subclass
            roles.add(str(node.get_property().get_named_property()._iri))
            return walk(node.get_filler(), depth + 1)
        classes.add(str(node.get_iri()))
        return depth

    max_depth = walk(expression, 0)
    present = frozenset(constructors)

    return Complexity(
        dl_length=len(classes) + len(roles) + sum(constructors.values()),
        depth=max_depth,
        constructors=dict(sorted(constructors.items())),
        num_atomic_classes=len(classes),
        num_roles=len(roles),
        expressivity=_expressivity(present),
        hardness=Hardness.get_blank_hardness(),
    )

def atomic_baseline_f1(
    target_extension: Collection[str],
    atomic_extensions: Mapping[str, Collection[str]],
) -> tuple[float, str | None]:
    """Best F1 achievable by any single atomic class.

    Returns the score and the winning class IRI, or ``(0.0, None)`` when no
    atomic class overlaps the target at all. This is the floor a hypothesis
    must clear for its learning problem to count as non-trivially solved.
    """
    target = frozenset(target_extension)
    if not target:
        return 0.0, None

    best_score = 0.0
    best_class: str | None = None

    for iri, extension in atomic_extensions.items():
        candidate = frozenset(extension)
        intersection = len(target & candidate)
        if not intersection:
            continue
        precision = intersection / len(candidate)
        recall = intersection / len(target)
        score = 2 * precision * recall / (precision + recall)
        if score > best_score:
            best_score, best_class = score, iri

    return best_score, best_class


def annotate_hardness(
    complexity: Complexity,
    *,
    target_extension: Collection[str],
    all_individuals: Collection[str],
    atomic_extensions: Mapping[str, Collection[str]],
) -> Complexity:
    """Populate the hardness fields of a complexity object.

    Depends on the knowledge base alone. No embedding-derived quantity may
    enter here: hardness describes the learning problem, which is held
    constant across embedding conditions.
    """
    target = frozenset(target_extension)
    universe = frozenset(all_individuals)

    if not target:
        return complexity.with_hardness(hardness=Hardness(
            extension_size=0,
            extension_ratio=0.0,
            atomic_baseline_f1=0.0,
            redundant=False,
        ))

    baseline, _ = atomic_baseline_f1(
        target_extension=target,
        atomic_extensions=atomic_extensions,
    )
    redundant = any(
        frozenset(extension) == target for extension in atomic_extensions.values()
    )

    return complexity.with_hardness(hardness=Hardness(
        extension_size=len(target),
        extension_ratio=len(target) / len(universe) if universe else 0.0,
        atomic_baseline_f1=baseline,
        redundant=redundant,
    ))