#src/data/results.py

import dataclasses
from dataclasses import dataclass
from typing import Any

from ontolearn.knowledge_base import KnowledgeBase

from src.config import EmbeddingSettings
from src.data.lp import LearningProblem, LearningProblemTruncated
from src.data.ontology import Triple


@dataclass(frozen=True)
class SingleMetric:
    identifier: str
    mean: float
    variance: float
    std_dev: float
    # n: int

    def to_dict(self) -> dict:
        return {
            "identifier": self.identifier,
            "mean": self.mean,
            "variance": self.variance,
            "std_dev": self.std_dev,
            # "n": self.n
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SingleMetric":
        return cls(
            identifier=data.get("identifier", ""),
            mean=data.get("mean", 0.0),
            variance=data.get("variance", 0.0),
            std_dev=data.get("std_dev", 0.0),
            # n=data.get("n", 0)
        )

@dataclass(frozen=True)
class MeanMetricsResult:
    """
    Represents the mean evaluation metrics 
    across multiple learning problem results.
    """
    accuracy: SingleMetric
    precision: SingleMetric
    recall: SingleMetric
    f1_score: SingleMetric
    jaccard: SingleMetric
    semantic_equivalence_rate: SingleMetric
    intersection: SingleMetric
    union: SingleMetric
    lift: SingleMetric | None
    lp_count: int

    @classmethod
    def from_dict(cls, data: dict) -> "MeanMetricsResult":
        lift_mean_data = data.get("lift")
        lift_mean_data = float(lift_mean_data) if lift_mean_data is not None else None
        return cls(
            accuracy=cls._get_single_metric("accuracy", data),
            precision=cls._get_single_metric("precision", data),
            recall=cls._get_single_metric("recall", data),
            f1_score=cls._get_single_metric("f1_score", data),
            jaccard=cls._get_single_metric("jaccard", data),
            semantic_equivalence_rate=cls._get_single_metric("semantic_equivalence_rate", data),
            intersection=cls._get_single_metric("intersection", data),
            union=cls._get_single_metric("union", data),
            lift=cls._get_single_metric("lift", data) if lift_mean_data is not None else None,
            lp_count=data.get("lp_count", 0)
        )

    @classmethod
    def _get_single_metric(cls, key: str, data: dict) -> SingleMetric:
        return SingleMetric(
                identifier=key,
                mean=data.get("mean", 0.0),
                variance=data.get("variance", 0.0),
                std_dev=data.get("std_dev", 0.0),
                # n=data.get("n", 0)
        )

    
    def to_dict(self) -> dict:
        return {
            "accuracy": self.accuracy.to_dict(),
            "precision": self.precision.to_dict(),
            "recall": self.recall.to_dict(),
            "f1_score": self.f1_score.to_dict(),
            "jaccard": self.jaccard.to_dict(),
            "semantic_equivalence_rate": self.semantic_equivalence_rate.to_dict(),
            "intersection": self.intersection.to_dict(),
            "union": self.union.to_dict(),
            "lift": self.lift.to_dict() if self.lift is not None else None,
            "lp_count": self.lp_count
        }


@dataclass(frozen=True)
class MetricsResult:
    """
    Represents the evaluation metrics for a single
    learning problem result.
    """
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    jaccard: float
    semantic_equivalence: bool
    intersection: int
    union: int
    lift: float


    @classmethod
    def from_dict(cls, data: dict) -> "MetricsResult":
        return cls(
            accuracy=data.get("accuracy", 0.0),
            precision=data.get("precision", 0.0),
            recall=data.get("recall", 0.0),
            f1_score=data.get("f1_score", 0.0),
            jaccard=data.get("jaccard", 0.0),
            semantic_equivalence=data.get("semantic_equivalence", False),
            intersection=data.get("intersection", 0),
            union=data.get("union", 0),
            lift=data.get("lift", 0)
        )

    
    def to_dict(self) -> dict:
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "jaccard": self.jaccard,
            "semantic_equivalence": self.semantic_equivalence,
            "intersection": self.intersection,
            "union": self.union,
            "lift": self.lift
        }

    def to_mean_metrics(self) -> MeanMetricsResult:
        return MeanMetricsResult(
            accuracy=self._get_single_metric("accuracy"),
            precision=self._get_single_metric("precision"),
            recall=self._get_single_metric("recall"),
            f1_score=self._get_single_metric("f1_score"),
            jaccard=self._get_single_metric("jaccard"),
            # Mapping the semantic equivalence metric to the semantic equivalence rate in the mean metrics result.
            semantic_equivalence_rate=self._get_single_metric("semantic_equivalence"),
            intersection=self._get_single_metric("intersection"),
            union=self._get_single_metric("union"),
            lift=self._get_single_metric("lift"),
            lp_count=1,
        )

    def _get_single_metric(self, key: str) -> SingleMetric:
        return SingleMetric(
            identifier=key,
            mean=getattr(self, key) if hasattr(self, key) else 0.0,
            variance=0.0,
            std_dev=0.0,
            # n=1,
        )

class TargetExtensionStructure:
    """
    Represents the structure of a target extension,
    including the number of positive and negative
    examples.
    """
    positive: int
    negative: int
    total: int

    def __init__(self, positive: int, negative: int):
        self.positive = positive
        self.negative = negative
        self.total = positive + negative


    @classmethod
    def from_dict(cls, data: dict) -> "TargetExtensionStructure":
        positive = data.get("positive", 0)
        negative = data.get("negative", 0)
        return cls(positive=positive, negative=negative)

    def to_dict(self) -> dict:
        return {
            "positive": self.positive,
            "negative": self.negative,
            "total": self.total
        }

class LearningProblemResult:
    """
    Represents the result of a single learning problem,
    including the learning problem details, the
    hypothesis, the target extension, and the evaluation
    metrics. A given learning problem is truncated.
    There is no way to extract it back in its full form,
    but it is not needed for the report anyway.
    """
    learning_problem: LearningProblemTruncated
    hypothesis: str = ""
    target_extension: TargetExtensionStructure | None
    hypothesis_extension: TargetExtensionStructure | None
    metrics: MetricsResult | None
    runtime: float | None
    error: str | None

    def __init__(
        self,
        learning_problem: LearningProblem,
        target_extension: TargetExtensionStructure | None = None,
        hypothesis_extension: TargetExtensionStructure | None = None,
        metrics: MetricsResult | None = None,
        runtime: float | None = None,
        error: str | None = None,
        hypothesis: str = "",
    ):
        self.learning_problem = learning_problem.to_truncated()
        self.hypothesis = hypothesis
        self.target_extension = target_extension
        self.hypothesis_extension = hypothesis_extension
        self.metrics = metrics
        self.runtime = runtime
        self.error = error

    @classmethod
    def from_dict(cls, data: dict) -> "LearningProblemResult":
        learning_problem = LearningProblem.from_dict(data.get("learning_problem", {}))
        hypothesis = data.get("hypothesis", "")
        target_extension_data = data.get("target_extension")
        target_extension = TargetExtensionStructure.from_dict(target_extension_data) if target_extension_data else None
        hypothesis_extension_data = data.get("hypothesis_extension")
        hypothesis_extension = TargetExtensionStructure.from_dict(hypothesis_extension_data) if hypothesis_extension_data else None
        metrics_data = data.get("metrics")
        metrics = MetricsResult.from_dict(metrics_data) if metrics_data else None
        runtime = data.get("runtime")
        error = data.get("error")

        return cls(
            learning_problem=learning_problem,
            hypothesis=hypothesis,
            target_extension=target_extension,
            hypothesis_extension=hypothesis_extension,
            metrics=metrics,
            runtime=runtime,
            error=error,
        )

    
    def to_dict(self) -> dict:
        return {
            "learning_problem": self.learning_problem.to_dict(),
            "hypothesis": self.hypothesis,
            "target_extension": TargetExtensionStructure.to_dict(self.target_extension)
                if self.target_extension else None,
            "hypothesis_extension": TargetExtensionStructure.to_dict(self.hypothesis_extension)
                if self.hypothesis_extension else None,
            "metrics": MetricsResult.to_dict(self.metrics)
                if self.metrics else None,
            "error": self.error,
            "runtime": self.runtime,
        }

@dataclass
class NCESStats:
    learner_name: str
    runtime_seconds: float
    degraded: bool

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

class EmbeddingResult:
    """
    Represents the mean result of a single embedding evaluated by NCES
    across multiple learning problems. This class is NOT to be confused
    with the EmbeddingResultDice class, which is a subclass of
    EmbeddingResult and includes additional information specific to
    the DICE embedding method.
    """
    split_name: str
    mean_metrics: MeanMetricsResult | None
    learning_problem_results: list[LearningProblemResult]
    number_of_problems: int
    number_of_successful_problems: int
    embedding_settings: EmbeddingSettings
    nces_stats: NCESStats

    def __init__(
        self,
        split_name: str,
        learning_problem_results: list[LearningProblemResult],
        embedding_settings: EmbeddingSettings,
        nces_stats: NCESStats,
        number_of_problems: int,
        number_of_successful_problems: int,
        mean_metrics: MeanMetricsResult | None = None,
    ):
        self.split_name = split_name
        self.mean_metrics = mean_metrics
        self.learning_problem_results = learning_problem_results
        self.embedding_settings = embedding_settings
        self.nces_stats = nces_stats
        self.number_of_problems = number_of_problems
        self.number_of_successful_problems = number_of_successful_problems

    @classmethod
    def from_dict(cls, data: dict) -> "EmbeddingResult":
        mean_metrics_data = data.get("mean_metrics", {})
        mean_metrics = MeanMetricsResult.from_dict(mean_metrics_data)
        learning_problem_results_data = data.get("learning_problem_results", [])
        learning_problem_results = [
            LearningProblemResult.from_dict(lpr_data)
            for lpr_data in learning_problem_results_data
        ]
        number_of_successful_problems = data.get("number_of_successful_problems", 0)
        split_name = data.get("split_name", "")

        return cls(
            split_name=split_name,
            mean_metrics=mean_metrics,
            learning_problem_results=learning_problem_results,
            embedding_settings=EmbeddingSettings.from_dict(data.get("embedding_settings", {})),
            nces_stats=NCESStats(**data.get("nces_stats", {})),
            number_of_problems=mean_metrics.lp_count,
            number_of_successful_problems=number_of_successful_problems,
        )

    
    def to_dict(self,) -> dict:
        return {
            "mean_metrics": self.mean_metrics.to_dict() if self.mean_metrics else None,
            "learning_problem_results": [
                lpr.to_dict()
                for lpr in self.learning_problem_results
            ],
            "embedding_settings": self.embedding_settings.to_dict(),
            "nces_stats": self.nces_stats.to_dict(),
            "split_name": self.split_name,
            "number_of_problems": self.number_of_problems,
            "number_of_successful_problems": self.number_of_successful_problems,
        }


class ComplexityStratum:
    """
    Represents a complexity stratum, which is a grouping of learning problems
    based on their complexity. Each stratum contains a list of learning problems
    and the corresponding target extensions.
    """
    stratum_name: str
    bucket_size: int | None
    aggragate_per_bucket_value: dict[str, MeanMetricsResult] | None

    def __init__(
        self,
        stratum_name: str,
        bucket_size: int | None,
        aggragate_per_bucket_value: dict[str, MeanMetricsResult] | None
    ):
        self.stratum_name = stratum_name
        self.bucket_size = bucket_size
        self.aggragate_per_bucket_value = aggragate_per_bucket_value

    def to_dict(self) -> dict:
        return {
            "stratum_name": self.stratum_name,
            "bucket_size": self.bucket_size,
            "aggragate_per_bucket_value": {
                k: v.to_dict() for k, v in self.aggragate_per_bucket_value.items()
            } if self.aggragate_per_bucket_value else None
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ComplexityStratum":
        stratum_name = data.get("stratum_name", "")
        bucket_size = data.get("bucket_size", None)
        aggragate_per_bucket_value_data = data.get("aggragate_per_bucket_value", None)
        aggragate_per_bucket_value = {
            k: MeanMetricsResult.from_dict(v) for k, v in aggragate_per_bucket_value_data.items()
        } if aggragate_per_bucket_value_data else None
        return cls(
            stratum_name=stratum_name,
            bucket_size=bucket_size,
            aggragate_per_bucket_value=aggragate_per_bucket_value
        )


class SingleRunResult:
    """
    Represents the result of a single run across 
    a knowledge base.
    """
    knowledge_base: str
    random_embedding_result: EmbeddingResult | None
    dice_embedding_result: EmbeddingResult | None
    runtime: float | None

    def __init__(
        self,
        knowledge_base: str,
        dice_embedding_result: EmbeddingResult | None,
        random_embedding_result: EmbeddingResult | None = None,
        random_complexity_aggregates: list[ComplexityStratum] | None = None,
        dice_complexity_aggregates: list[ComplexityStratum] | None = None,
        runtime: float | None = None,
    ):
        self.knowledge_base = knowledge_base
        self.random_embedding_result = random_embedding_result
        self.random_complexity_aggregates = random_complexity_aggregates
        self.dice_complexity_aggregates = dice_complexity_aggregates
        self.dice_embedding_result = dice_embedding_result
        self.runtime = runtime

    def set_runtime(self, runtime: float) -> None:
        self.runtime = runtime

    @classmethod
    def from_dict(cls, data: dict) -> "SingleRunResult":
        knowledge_base = data.get("knowledge_base", "")
        random_embedding_data = data.get("random_embedding_result")
        random_embedding_result = EmbeddingResult.from_dict(random_embedding_data) if random_embedding_data else None
        random_complexity_aggregates_data = data.get("random_complexity_aggregates", [])
        random_complexity_aggregates = [ComplexityStratum.from_dict(d) for d in random_complexity_aggregates_data] if random_complexity_aggregates_data else None
        dice_embedding_data = data.get("dice_embedding_result")
        dice_embedding_result = EmbeddingResult.from_dict(dice_embedding_data) if dice_embedding_data else None
        dice_complexity_aggregates_data = data.get("dice_complexity_aggregates", [])
        dice_complexity_aggregates = [ComplexityStratum.from_dict(d) for d in dice_complexity_aggregates_data] if dice_complexity_aggregates_data else None
        runtime = data.get("runtime", 0.0)
        return cls(
            knowledge_base=knowledge_base,
            random_embedding_result=random_embedding_result,
            random_complexity_aggregates=random_complexity_aggregates,
            dice_complexity_aggregates=dice_complexity_aggregates,
            dice_embedding_result=dice_embedding_result,
            runtime=runtime
        )


    def to_dict(self) -> dict:
        return {
            "knowledge_base": self.knowledge_base,
            "random_embedding_result": self.random_embedding_result.to_dict()
                if self.random_embedding_result else None,
            "random_complexity_aggregates": [s.to_dict() for s in self.random_complexity_aggregates]
                if self.random_complexity_aggregates else None,
            "dice_complexity_aggregates": [s.to_dict() for s in self.dice_complexity_aggregates]
                if self.dice_complexity_aggregates else None,
            "dice_embedding_result": self.dice_embedding_result.to_dict()
                if self.dice_embedding_result else None,
            "runtime": self.runtime
        }

    def get_conditions(self) -> list[str]:
        conditions = []
        if self.random_embedding_result:
            conditions.append("random")
        if self.dice_embedding_result:
            conditions.append("dice")
        return conditions

    def get_embedding_result(self, condition: str) -> EmbeddingResult | None:
        if condition == "random":
            return self.random_embedding_result
        elif condition == "dice":
            return self.dice_embedding_result
        else:
            raise ValueError(f"Unknown embedding condition: {condition}")

    def get_any_embedding_result(self) -> EmbeddingResult | None:
        if self.dice_embedding_result:
            return self.dice_embedding_result
        elif self.random_embedding_result:
            return self.random_embedding_result
        else:
            return None

    def get_number_of_problems(self) -> int:
        result = self.get_any_embedding_result()
        if result:
            return result.number_of_problems
        return 0


@dataclass(frozen=True)
class KnowledgeBaseStats:
    knowledge_base_name: str
    number_of_individuals: int
    number_of_triples: int
    number_of_atomic_classes: int

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class KnowledgeBaseFailure:
    knowledge_base_name: str
    error_message: str
    seed: int

@dataclass(frozen=True)
class KnowledgeBaseResult:
    """
    Represents the mean result of all seeded 
    runs across a knowledge base.
    """
    knowledge_base: str
    mean_random_metrics: MeanMetricsResult | None
    mean_dice_metrics: MeanMetricsResult | None

    @classmethod
    def from_dict(cls, data: dict) -> "KnowledgeBaseResult":
        knowledge_base = data.get("knowledge_base", "")
        mean_random_metrics_data = data.get("mean_random_metrics")
        mean_random_metrics = MeanMetricsResult.from_dict(mean_random_metrics_data) if mean_random_metrics_data else None
        mean_dice_metrics_data = data.get("mean_dice_metrics")
        mean_dice_metrics = MeanMetricsResult.from_dict(mean_dice_metrics_data) if mean_dice_metrics_data else None

        return cls(
            knowledge_base=knowledge_base,
            mean_random_metrics=mean_random_metrics,
            mean_dice_metrics=mean_dice_metrics
        )


    def to_dict(self) -> dict[str, Any]:
        return {
            "knowledge_base": self.knowledge_base,
            "mean_random_metrics": MeanMetricsResult.to_dict(self.mean_random_metrics)
                if self.mean_random_metrics else None,
            "mean_dice_metrics": MeanMetricsResult.to_dict(self.mean_dice_metrics)
                if self.mean_dice_metrics else None
        }


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

