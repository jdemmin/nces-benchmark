#src/data/results.py
"""Per-run bookkeeping artifacts.

Per-problem scoring lives in ``src.eval`` (``Observation``,
``ExtensionMetrics``); this module only holds orchestration-level records:
ontology/learning-problem phase results, knowledge-base stats/failures, and
the per-(knowledge base, seed) bundle of observations, HPO trials, and
ranking quality that ``src.eval.suite``/``src.eval.rq1`` consume.
"""

import dataclasses
from dataclasses import dataclass, field
from typing import Any

from ontolearn.knowledge_base import KnowledgeBase

from src.data.lp import LearningProblem
from src.data.ontology import Triple
from src.eval.pairing import Observation
from src.eval.rq2 import Trial


@dataclass
class NCESStats:
    learner_name: str
    runtime_seconds: float
    degraded: bool

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class SingleRunResult:
    """Everything one (knowledge base, seed) run contributes to the suite.

    Filled in across every condition's training/evaluation, then flattened
    across (knowledge base, seed) by the caller into the suite-wide
    observations/trials/quality inputs ``src.eval.suite.analyse_suite`` and
    ``src.eval.rq1.quality_frame`` expect.
    """

    knowledge_base: str
    seed: int
    observations: list[Observation] = field(default_factory=list)
    trials: list[Trial] = field(default_factory=list)
    #: One row per condition: {"condition", "mrr", "hits@1", "hits@3", "hits@10", "mean_abl"}.
    quality: list[dict[str, Any]] = field(default_factory=list)
    condition_stats: dict[str, NCESStats] = field(default_factory=dict)
    runtime: float | None = None

    def set_runtime(self, runtime: float) -> None:
        self.runtime = runtime

    def to_dict(self) -> dict[str, Any]:
        return {
            "knowledge_base": self.knowledge_base,
            "seed": self.seed,
            "observations": [dataclasses.asdict(o) for o in self.observations],
            "trials": [t.to_dict() for t in self.trials],
            "quality": self.quality,
            "condition_stats": {
                name: stats.to_dict() for name, stats in self.condition_stats.items()
            },
            "runtime": self.runtime,
        }


@dataclass(frozen=True)
class KnowledgeBaseStats:
    knowledge_base_name: str
    number_of_individuals: int
    number_of_triples: int
    number_of_atomic_classes: int
    #: Local names shared by more than one distinct IRI; those IRIs are
    #: dropped from the entity vocabulary rather than merged (sec:meth:kbs).
    number_of_colliding_local_names: int = 0
    number_of_entities_dropped_to_collisions: int = 0

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class KnowledgeBaseFailure:
    knowledge_base_name: str
    error_message: str
    seed: int


@dataclass(frozen=True)
class OntologyParseResult:

    knowledge_base: KnowledgeBase
    all_individuals: list[str]
    triples: list[Triple]

    def to_dict(self) -> dict[str, Any]:
        """
        Knowledge base is omitted from the dictionary representation because it may contain
        complex objects not easily serializable to JSON.
        """
        return {
            "all_individuals": self.all_individuals,
            "triples": [triple.as_tuple() for triple in self.triples]
        }

    @classmethod
    def from_dict(cls, data: dict, knowledge_base: KnowledgeBase) -> "OntologyParseResult":
        all_individuals = data.get("all_individuals", [])
        triples_data = data.get("triples", [])
        triples = [Triple(*triple) for triple in triples_data]
        return cls(
            knowledge_base=knowledge_base,  # Knowledge base is omitted in the dictionary representation
            all_individuals=all_individuals,
            triples=triples
        )

@dataclass(frozen=True)
class OntologyPhaseResult:
    ontology_parse_result: OntologyParseResult
    counts: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ontology_parse_result": self.ontology_parse_result.to_dict()
                if self.ontology_parse_result else None,
            "counts": self.counts
        }

@dataclass(frozen=True)
class LearningProblemPhaseResult:
    target_extensions: dict[str, frozenset[str]]
    split: dict[str, list[LearningProblem]]
    unparsed: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_extensions": {
                key: list(value) for key, value in self.target_extensions.items()
            },
            "split": {
                key: [problem.to_dict() for problem in value] for key, value in self.split.items()
            },
            "unparsed": self.unparsed
        }
    
@dataclass(frozen=True)
class HardnessAnnotationResult:
    annotated_problems: list[LearningProblem]
    unparsed_problems: list[str]
    target_extensions: dict[str, frozenset[str]]

    def to_dict(self) -> dict:
        return {
            "annotated_problems": self.annotated_problems,
            "unparsed_problems": self.unparsed_problems,
            "target_extensions": {
                key: list(value) for key, value in self.target_extensions.items()
            }
        }
