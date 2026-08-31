# src/data/lp.py
"""Learning-problem generation and the canonical learning-problem schema.

A learning problem satisfies the benchmark constraint: for target concept
``T``, positive examples ``E+`` and negative examples ``E-``, the learner must
find a concept expression ``C`` such that in ``K' = K ∪ {T ≡ C}`` every
``e+ ∈ E+`` is entailed as ``C(e+)`` and no ``e- ∈ E-`` is.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rdflib import OWL, RDF

from src.config import DataGenerationSettings
from src.data.complexity import Complexity, Hardness, structural_complexity
from src.data.ontology import local_name

logger = logging.getLogger(__name__)

#: Key used by ontolearn's ``LPs.json`` for positive examples.
_POS_KEY = "positive examples"
#: Key used by ontolearn's ``LPs.json`` for negative examples.
_NEG_KEY = "negative examples"
    

@dataclass(frozen=True)
class LearningProblem:
    """One learning problem in the canonical project schema."""

    id: str
    target_concept: str
    complexity: Complexity
    pos_example: list[str] = field(default_factory=list)
    neg_example: list[str] = field(default_factory=list)
    num_pos: int = field(default=0)
    num_neg: int = field(default=0)


    def __post_init__(self) -> None:
        if not self.pos_example:
            raise ValueError(f"Learning problem {self.id} has no positive examples.")
        if not self.neg_example:
            raise ValueError(f"Learning problem {self.id} has no negative examples.")
        object.__setattr__(self, "num_pos", len(self.pos_example))
        object.__setattr__(self, "num_neg", len(self.neg_example))

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "target_concept": self.target_concept,
            "complexity": self.complexity.to_dict(),
            "pos_example": self.pos_example,
            "neg_example": self.neg_example,
            "num_pos": self.num_pos,
            "num_neg": self.num_neg,
        }

    def as_nces_datapoint(self) -> tuple[str, dict[str, list[str]]]:
        """Convert to the ``(name, examples)`` tuple that ``NCES`` consumes.

        NCES indexes its embedding matrix by local name, so IRIs are reduced
        here rather than inside the learner.
        """

        return (
            self.target_concept,
            {
                _POS_KEY: [local_name(iri) for iri in self.pos_example],
                _NEG_KEY: [local_name(iri) for iri in self.neg_example],
            },
        )

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> LearningProblem:
        return cls(
            id=str(payload["id"]),
            target_concept=str(payload["target_concept"]),
            pos_example=list(payload["pos_example"]),
            neg_example=list(payload["neg_example"]),
            complexity=Complexity.from_dict(payload["complexity"]),
        )

    def _with_complexity(self, complexity: Complexity) -> LearningProblem:
            """Return a copy carrying an updated complexity object."""
            return LearningProblem(
                id=self.id,
                target_concept=self.target_concept,
                pos_example=list(self.pos_example),
                neg_example=list(self.neg_example),
                complexity=complexity,
            )

    def annotate_complexity(self, complexity: Complexity) -> LearningProblem:
        """Return a copy carrying an updated complexity object."""
        return self._with_complexity(complexity=complexity)

    def annotate_hardness(self, complexity: Complexity, hardness: Hardness) -> LearningProblem:
        return self._with_complexity(complexity.with_hardness(hardness=hardness))

def generate_learning_problems(
    kb_path: Path,
    storage_path: Path,
    settings: DataGenerationSettings,
    seed: str,
) -> list[LearningProblem]:
    """Generate learning problems for one knowledge base.

    Wraps ``ontolearn.lp_generator.LPGen``, which writes ``LPs.json`` into
    ``storage_path``. The generated file is then normalised into the canonical
    project schema, with positive/negative examples expanded back to full IRIs.
    """
    from ontolearn.lp_generator import LPGen
    storage_path.mkdir(parents=True, exist_ok=True)
    kwargs = settings.lpgen_kwargs(kb_path, storage_path)
    logger.info(
        "Generating up to %d learning problems for %s",
        kwargs["max_num_lps"],
        kb_path.name,
    )
    LPGen(**kwargs).generate()

    raw_path = storage_path / "LPs.json"
    if not raw_path.is_file():
        raise RuntimeError(f"LPGen did not produce {raw_path}.")

    with raw_path.open(encoding="utf-8") as handle:
        raw = json.load(handle)

    namespace = _infer_namespace(kb_path)
    problems = _normalise(raw, namespace=namespace)
    logger.info("Normalised %d learning problems", len(problems))
    return problems


def _infer_namespace(kb_path: Path) -> str:
    """Read the ontology's default namespace so local names can be expanded."""
    from rdflib import Graph, URIRef

    graph = Graph()
    graph.parse(str(kb_path))

    onto_iris = sorted(
        str(o)
        for o in graph.subjects(RDF.type, OWL.Ontology)
        if isinstance(o, URIRef)
    )
    if onto_iris:
        iri = onto_iris[0]
        return iri if iri.endswith(("#", "/")) else iri + "#"

    # Fall back to the most common namespace among *named individuals*,
    # ties broken lexicographically so the result is order-independent.
    from collections import Counter

    counts: Counter[str] = Counter()
    for subject in graph.subjects(RDF.type, None):
        if not isinstance(subject, URIRef):
            continue
        iri = str(subject)
        if iri.startswith(("http://www.w3.org/", "http://purl.org/dc/")):
            continue
        for separator in ("#", "/"):
            if separator in iri:
                counts[iri.rsplit(separator, 1)[0] + separator] += 1
                break
    if counts:
        return min(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0]
    return ""





def _normalise(
    raw: Iterable[Any], *, namespace: str
) -> list[LearningProblem]:
    """Convert ontolearn's ``LPs.json`` payload into learning problems."""

    entries: list[tuple[str, dict[str, Any]]]
    if isinstance(raw, dict):
        entries = list(raw.items())
    else:
        entries = [(item[0], item[1]) for item in raw]
    problems: list[LearningProblem] = []
    for target_concept, examples in sorted(entries, key=lambda x: x[0]):
        positives = _expand(examples.get(_POS_KEY, []), namespace)
        negatives = _expand(examples.get(_NEG_KEY, []), namespace)
        if not positives or not negatives:
            logger.debug("Skipping degenerate problem %r", target_concept)
            continue
        problems.append(
            LearningProblem(
                id=_stable_id(target_concept, positives, negatives),
                target_concept=target_concept,
                pos_example=sorted(positives),
                neg_example=sorted(negatives),
                complexity=structural_complexity(target_concept),
            )
        )
    problems.sort(key=lambda p: p.id)
    return problems


def _expand(names: Sequence[str], namespace: str) -> list[str]:
    """Expand local names back into full IRIs where possible."""

    expanded = []
    for name in names:
        if name.startswith(("http://", "https://")):
            expanded.append(name)
        else:
            expanded.append(f"{namespace}{name}" if namespace else name)
    return expanded


def save_learning_problems(
    problems: Sequence[LearningProblem], path: Path
) -> None:
    """Serialize learning problems grouped by complexity level."""

    grouped: dict[str, list[dict[str, Any]]] = {}
    problems_tmp = copy.deepcopy(problems)
    for problem in problems_tmp:
        # Grouping by DL-expression length is a coarse proxy for difficulty,
        # but it is easy to compute and can be used to stratify the train/test split.
        grouped.setdefault(str(problem.complexity.dl_length), []).append(problem.to_dict())

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(grouped, handle, indent=2, ensure_ascii=False)
    logger.info("Wrote %d learning problems to %s", len(problems_tmp), path)


def load_learning_problems(path: Path) -> list[LearningProblem]:
    """Load learning problems from the grouped JSON artifact."""

    with path.open(encoding="utf-8") as handle:
        grouped = json.load(handle)
    problems = [
        LearningProblem.from_dict(payload)
        for problems in grouped.values()
        for payload in problems
    ]
    problems.sort(key=lambda p: p.id)
    return problems


_CONTINUOUS = {"extension_ratio", "atomic_baseline_f1"}

def _get_strata_key(stratify_by: str | None, problem: LearningProblem) -> str:
    if problem is None:
        raise AttributeError("Cannot get strata key: Problem is None")
    if stratify_by is None:
        return str(problem.complexity.dl_length)

    if stratify_by in Hardness.__dataclass_fields__:
        hardness = problem.complexity.hardness
        if hardness is None or not problem.complexity.is_annotated:
            raise ValueError(
                f"Cannot stratify by {stratify_by!r}: problem {problem.id} "
                f"is not hardness-annotated. Run the annotation stage first."
            )
        value = getattr(hardness, stratify_by)
        if stratify_by in _CONTINUOUS and value is not None:
            return f"q{min(int(float(value) * 5), 4)}"  # 5 fixed quintile bins
        return str(value)

    if stratify_by in Complexity.__dataclass_fields__:
        return str(getattr(problem.complexity, stratify_by))
    raise ValueError(f"Unknown stratify_by field {stratify_by!r}.")


def _key_order(k: str) -> tuple[int, float, str]:
    try:
        return (0, float(k), "")
    except ValueError:
        return (1, 0.0, k)


def split_learning_problems(
    problems: Sequence[LearningProblem],
    *,
    seed: str,
    ratios: tuple[float, float] = (0.8, 0.2),
    stratify_by: str | None = "dl_length",
) -> dict[str, list[LearningProblem]]:
    """
    Split learning problems into disjoint train/test sets.

    NCES must never be evaluated on a learning problem it trained on, so the
    split is applied to the problems themselves rather than to the examples.

    When ``stratify_by`` is set, problems are grouped by that complexity field
    and each group is split independently, so every split sees a comparable
    difficulty mix. Without stratification the test split's difficulty
    composition drifts across seeds, which inflates apparent variance between
    embedding conditions.
    """

    if not problems or len(problems) < 2:
        raise ValueError(
            f"At least two learning problems are required for a split. "
            f"stratification. Got {len(problems)} problems."
        )

    if not (0 < ratios[0] < 1 and 0 < ratios[1] < 1 and ratios[0] + ratios[1] == 1):
        raise ValueError(f"Invalid ratios {ratios}: must be positive and sum to 1.")
    strata: dict[str, list[LearningProblem]] = {}
    for problem in problems:
        strata.setdefault(_get_strata_key(stratify_by, problem), []).append(problem)

    split: dict[str, list[LearningProblem]] = {
        "train": [], "test": [],
    }

    for key in sorted(strata, key=_key_order):
        stratum = sorted(strata[key], key=lambda p: (_split_score(seed, p.id), p.id))
        total = len(stratum)
        n_train = max(1, round(total * ratios[0]))
        n_train = min(n_train, total)          # keep at least 0 for test
        split["train"].extend(stratum[:n_train])
        split["test"].extend(stratum[n_train:])

    if not split["test"]:
        # Guarantee a non-empty test split, as the unstratified path did.
        split["test"] = [split["train"].pop()]

    for name in ("train", "test"):
        split[name].sort(key=lambda p: p.id)
    return split


def save_split(
    split: dict[str, list[LearningProblem]], directory: Path
) -> None:
    """Persist each learning-problem split to its own artifact."""
    
    directory.mkdir(parents=True, exist_ok=True)
    for name, problems in split.items():
        payload = [problem.to_dict() for problem in problems]
        with (directory / f"{name}_problems.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)

def _stable_id(target_concept: str, positives: list[str], negatives: list[str]) -> str:
    h = hashlib.sha256()
    h.update(target_concept.encode("utf-8"))
    h.update(b"\x00")
    for iri in sorted(positives):
        h.update(iri.encode("utf-8"))
        h.update(b"\x01")
    h.update(b"\x00")
    for iri in sorted(negatives):
        h.update(iri.encode("utf-8"))
        h.update(b"\x01")
    return f"lp_{h.hexdigest()[:12]}"

def stable_id(learning_problem: LearningProblem) -> str:
    return _stable_id(
        target_concept=learning_problem.target_concept,
        positives=learning_problem.pos_example,
        negatives=learning_problem.neg_example,
    )


def _split_score(seed: str, problem_id: str) -> float:
    """Deterministic per-problem score, independent of all other problems."""
    digest = hashlib.sha256(f"{seed}:{problem_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)