# src/data/lp.py
"""Learning-problem generation and the canonical learning-problem schema.

A learning problem satisfies the benchmark constraint: for target concept
``T``, positive examples ``E+`` and negative examples ``E-``, the learner must
find a concept expression ``C`` such that in ``K' = K ∪ {T ≡ C}`` every
``e+ ∈ E+`` is entailed as ``C(e+)`` and no ``e- ∈ E-`` is.
"""

from __future__ import annotations

import json
import logging
import random
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from src.config import DataGenerationSettings
from src.data.complexity import Complexity, Hardness, structural_complexity
from src.data.ontology import local_name

logger = logging.getLogger(__name__)

#: Key used by ontolearn's ``LPs.json`` for positive examples.
_POS_KEY = "positive examples"
#: Key used by ontolearn's ``LPs.json`` for negative examples.
_NEG_KEY = "negative examples"


@dataclass(frozen=True)
class LearningProblemTruncated:
    """A learning problem with only the target concept and complexity.

    This is used to stratify learning problems before they are expanded into
    full IRIs.
    """

    id: str
    target_concept: str
    complexity: Complexity
    num_pos: int = field(default=0)
    num_neg: int = field(default=0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "target_concept": self.target_concept,
            "complexity": self.complexity.to_dict(),
            "num_pos": self.num_pos,
            "num_neg": self.num_neg,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> LearningProblemTruncated:
        return cls(
            id=str(payload["id"]),
            target_concept=str(payload["target_concept"]),
            complexity=Complexity.from_dict(payload["complexity"]),
            num_pos=int(payload.get("num_pos", 0)),
            num_neg=int(payload.get("num_neg", 0)),
        )
    

@dataclass(frozen=True)
class LearningProblem(LearningProblemTruncated):
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
        payload = asdict(self)
        payload["complexity"] = asdict(self.complexity)
        return payload

    def to_truncated(self) -> LearningProblemTruncated:
        return LearningProblemTruncated(
            id=self.id,
            target_concept=self.target_concept,
            complexity=self.complexity,
            num_pos=self.num_pos,
            num_neg=self.num_neg,
        )

    @classmethod
    def to_truncated_list(cls, problems: Iterable[LearningProblem]) -> list[LearningProblemTruncated]:
        return [
            problem.to_truncated()
            for problem in problems
        ]

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
            """Return a copy carrying an updated hardness object for a given complexity."""
            return self._with_complexity(
                self.annotate_complexity(complexity=complexity)
                .complexity.with_hardness(hardness=hardness)
            )

def generate_learning_problems(
    kb_path: Path,
    storage_path: Path,
    settings: DataGenerationSettings,
    *,
    seed: int,
) -> list[LearningProblem]:
    """Generate learning problems for one knowledge base.

    Wraps ``ontolearn.lp_generator.LPGen``, which writes ``LPs.json`` into
    ``storage_path``. The generated file is then normalised into the canonical
    project schema, with positive/negative examples expanded back to full IRIs.
    """
    from ontolearn.lp_generator import LPGen

    random.seed(seed)
    storage_path.mkdir(parents=True, exist_ok=True)

    kwargs = settings.lpgen_kwargs(kb_path, storage_path)
    logger.info(
        "Generating up to %d learning problems for %s",
        kwargs["max_num_lps"],
        kb_path.name,
    )
    LPGen(random_seed=seed, **kwargs).generate()

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
    for subject in graph.subjects():
        if isinstance(subject, URIRef):
            iri = str(subject)
            for separator in ("#", "/"):
                if separator in iri:
                    return iri.rsplit(separator, 1)[0] + separator
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
    for index, (target_concept, examples) in enumerate(entries):
        positives = _expand(examples.get(_POS_KEY, []), namespace)
        negatives = _expand(examples.get(_NEG_KEY, []), namespace)
        if not positives or not negatives:
            logger.debug("Skipping degenerate problem %r", target_concept)
            continue
        problems.append(
            LearningProblem(
                id=f"lp_{index:04d}",
                target_concept=target_concept,
                pos_example=sorted(positives),
                neg_example=sorted(negatives),
                complexity=structural_complexity(target_concept),
            )
        )
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
    for problem in problems:
        # Grouping by DL-expression length is a coarse proxy for difficulty,
        # but it is easy to compute and can be used to stratify the train/test split.
        grouped.setdefault(str(problem.complexity.dl_length), []).append(problem.to_dict())

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(grouped, handle, indent=2, ensure_ascii=False)
    logger.info("Wrote %d learning problems to %s", len(problems), path)


def load_learning_problems(path: Path) -> list[LearningProblem]:
    """Load learning problems from the grouped JSON artifact."""

    with path.open(encoding="utf-8") as handle:
        grouped = json.load(handle)
    return [
        LearningProblem.from_dict(payload)
        for problems in grouped.values()
        for payload in problems
    ]

def _get_strata_key(stratify_by: str | None, problem: LearningProblem) -> str:
    """
    Return a string key for stratifying learning problems.
    Missing or invalid keys default to DL-expression length.
    """

    if problem is None:
        raise AttributeError("Cannot get strata key: Problem is None")
    strata_target = problem.complexity
    invalid_strata = (
        stratify_by is None
        or stratify_by not in Complexity.__dataclass_fields__
    )
    if stratify_by in Hardness.__dataclass_fields__:
        strata_target = strata_target.hardness
        invalid_strata = strata_target is None or strata_target.atomic_baseline_f1 is None
    return (str(problem.complexity.dl_length)
            if invalid_strata 
            else str(getattr(strata_target, stratify_by)) # type: ignore //Already validated that stratify_by is a valid field of Complexity or Hardness
    ) 

def split_learning_problems(
    problems: Sequence[LearningProblem],
    *,
    seed: int,
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
            f"It might require even more depending on the ratios and "
            f"stratification. Got {len(problems)} problems."
        )

    strata = {}
    for problem in problems:
        strata.setdefault(_get_strata_key(stratify_by, problem), []).append(problem)

    split: dict[str, list[LearningProblem]] = {
        "train": [], "test": [],
    }

    for key in sorted(strata):
        stratum = sorted(strata[key], key=lambda p: p.id)
        random.Random(seed).shuffle(stratum)
        total = len(stratum)
        if 0 < ratios[0] < 1 and 0 < ratios[1] < 1 and ratios[0] + ratios[1] == 1:
            n_train = max(1, int(total * ratios[0]))
            split["train"].extend(stratum[:n_train])
            split["test"].extend(stratum[n_train:])
        else:
            raise ValueError(f"Invalid ratios {ratios}: must be positive and sum to 1.")

    if not split["test"]:
        # Guarantee a non-empty test split, as the unstratified path did.
        split["test"] = [split["train"].pop()]

    return _sort_then_shuffle_splits(split, seed=seed)


def _sort_then_shuffle_splits(splits: dict[str, list[LearningProblem]], seed: int) -> dict[str, list[LearningProblem]]:
    """Shuffle each split in-place with a deterministic seed."""
    tmp_splits = splits.copy()
    for problems in tmp_splits.values():
        problems.sort(key=lambda p: p.id)
        random.Random(seed).shuffle(problems)
    return tmp_splits

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