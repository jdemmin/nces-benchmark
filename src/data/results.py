#src/data/results.py

import dataclasses
from dataclasses import dataclass
from typing import Any

from ontolearn.knowledge_base import KnowledgeBase

from src.config import EmbeddingSettings
from src.data.lp import LearningProblem, LearningProblemTruncated
from src.data.ontology import Triple


class MeanMetricsResult:
    """
    Represents the mean evaluation metrics 
    across multiple learning problem results.
    """
    mean_accuracy: float
    mean_precision: float
    mean_recall: float
    mean_f1_score: float
    mean_jaccard: float
    mean_semantic_equivalence: float
    mean_intersection: float
    mean_union: float
    mean_lift: float

    def __init__(
        self,
        mean_accuracy: float,
        mean_precision: float,
        mean_recall: float,
        mean_f1_score: float,
        mean_jaccard: float,
        mean_semantic_equivalence: float,
        mean_intersection: float,
        mean_union: float,
        mean_lift: float
    ):
        self.mean_accuracy = mean_accuracy
        self.mean_precision = mean_precision
        self.mean_recall = mean_recall
        self.mean_f1_score = mean_f1_score
        self.mean_jaccard = mean_jaccard
        self.mean_semantic_equivalence = mean_semantic_equivalence
        self.mean_intersection = mean_intersection
        self.mean_union = mean_union
        self.mean_lift = mean_lift


    @classmethod
    def from_dict(cls, data: dict) -> "MeanMetricsResult":
        return cls(
            mean_accuracy=data.get("mean_accuracy", 0.0),
            mean_precision=data.get("mean_precision", 0.0),
            mean_recall=data.get("mean_recall", 0.0),
            mean_f1_score=data.get("mean_f1_score", 0.0),
            mean_jaccard=data.get("mean_jaccard", 0.0),
            mean_semantic_equivalence=data.get("mean_semantic_equivalence", 0.0),
            mean_intersection=data.get("mean_intersection", 0.0),
            mean_union=data.get("mean_union", 0.0),
            mean_lift=data.get("mean_lift", 0.0)
        )

    
    def to_dict(self) -> dict:
        return {
            "mean_accuracy": self.mean_accuracy,
            "mean_precision": self.mean_precision,
            "mean_recall": self.mean_recall,
            "mean_f1_score": self.mean_f1_score,
            "mean_jaccard": self.mean_jaccard,
            "mean_semantic_equivalence": self.mean_semantic_equivalence,
            "mean_intersection": self.mean_intersection,
            "mean_union": self.mean_union,
            "mean_lift": self.mean_lift
        }

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

    def __init__(
        self,
        accuracy: float,
        precision: float,
        recall: float,
        f1_score: float,
        jaccard: float,
        semantic_equivalence: bool,
        intersection: int,
        union: int,
        lift: float
    ):
        self.accuracy = accuracy
        self.precision = precision
        self.recall = recall
        self.f1_score = f1_score
        self.jaccard = jaccard
        self.semantic_equivalence = semantic_equivalence
        self.intersection = intersection
        self.union = union
        self.lift = lift


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
            mean_accuracy=self.accuracy,
            mean_precision=self.precision,
            mean_recall=self.recall,
            mean_f1_score=self.f1_score,
            mean_jaccard=self.jaccard,
            mean_semantic_equivalence=float(1 if self.semantic_equivalence else 0),
            mean_intersection=float(self.intersection),
            mean_union=float(self.union),
            mean_lift=float(self.lift) if self.lift is not None else 0.0
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
    hypotesis: str = ""
    target_extension: TargetExtensionStructure | None
    metrics: MetricsResult | None
    runtime: float | None
    error: str | None

    def __init__(
        self,
        learning_problem: LearningProblem,
        target_extension: TargetExtensionStructure | None = None,
        metrics: MetricsResult | None = None,
        runtime: float | None = None,
        error: str | None = None,
        hypotesis: str = "",
    ):
        self.learning_problem = learning_problem.to_truncated()
        self.hypotesis = hypotesis
        self.target_extension = target_extension
        self.metrics = metrics
        self.runtime = runtime
        self.error = error

    @classmethod
    def from_dict(cls, data: dict) -> "LearningProblemResult":
        learning_problem = LearningProblem.from_dict(data.get("learning_problem", {}))
        hypotesis = data.get("hypotesis", "")
        target_extension_data = data.get("target_extension")
        target_extension = TargetExtensionStructure.from_dict(target_extension_data) if target_extension_data else None
        metrics_data = data.get("metrics")
        metrics = MetricsResult.from_dict(metrics_data) if metrics_data else None
        runtime = data.get("runtime")
        error = data.get("error")

        return cls(
            learning_problem=learning_problem,
            hypotesis=hypotesis,
            target_extension=target_extension,
            metrics=metrics,
            runtime=runtime,
            error=error
        )

    
    def to_dict(self) -> dict:
        return {
            "learning_problem": self.learning_problem.to_dict(),
            "hypotesis": self.hypotesis,
            "target_extension": TargetExtensionStructure.to_dict(self.target_extension)
                if self.target_extension else None,
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
        number_of_problems = len(learning_problem_results)
        number_of_successful_problems = data.get("number_of_successful_problems", 0)
        split_name = data.get("split_name", "")

        return cls(
            split_name=split_name,
            mean_metrics=mean_metrics,
            learning_problem_results=learning_problem_results,
            embedding_settings=EmbeddingSettings.from_dict(data.get("embedding_settings", {})),
            nces_stats=NCESStats(**data.get("nces_stats", {})),
            number_of_problems=number_of_problems,
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

class SingleRunResult:
    """
    Represents the result of a single run across 
    a knowledge base.
    """
    knowledge_base: str
    random_embedding_result: EmbeddingResult | None
    random_complexity_summary: dict[str, dict[str, MeanMetricsResult]] | None
    dice_complexity_summary: dict[str, dict[str, MeanMetricsResult]] | None
    dice_embedding_result: EmbeddingResult | None
    runtime: float | None

    def __init__(
        self,
        knowledge_base: str,
        dice_embedding_result: EmbeddingResult | None,
        random_embedding_result: EmbeddingResult | None = None,
        runtime: float | None = None,
    ):
        self.knowledge_base = knowledge_base
        self.random_embedding_result = random_embedding_result
        self.dice_embedding_result = dice_embedding_result
        self.runtime = runtime

    def set_runtime(self, runtime: float) -> None:
        self.runtime = runtime

    @classmethod
    def from_dict(cls, data: dict) -> "SingleRunResult":
        knowledge_base = data.get("knowledge_base", "")
        random_embedding_result = EmbeddingResult.from_dict(data.get("random_embedding_result", {}))
        dice_embedding_result = EmbeddingResult.from_dict(data.get("dice_embedding_result", {}))
        runtime = data.get("runtime", 0.0)
        return cls(
            knowledge_base=knowledge_base,
            random_embedding_result=random_embedding_result,
            dice_embedding_result=dice_embedding_result,
            runtime=runtime
        )


    def to_dict(self) -> dict:
        return {
            "knowledge_base": self.knowledge_base,
            "random_embedding_result": self.random_embedding_result.to_dict()
                if self.random_embedding_result else None,
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


@dataclass
class KnowledgeBaseStats:
    knowledge_base_name: str
    number_of_individuals: int
    number_of_triples: int
    number_of_atomic_classes: int


@dataclass
class KnowledgeBaseFailure:
    knowledge_base_name: str
    error_message: str
    seed: int


class KnowledgeBaseResult:
    """
    Represents the mean result of all seeded 
    runs across a knowledge base.
    """
    knowledge_base: str
    mean_random_metrics: MeanMetricsResult | None
    mean_dice_metrics: MeanMetricsResult | None

    def __init__(
        self,
        knowledge_base: str,
        mean_random_metrics: MeanMetricsResult | None = None,
        mean_dice_metrics: MeanMetricsResult | None = None,
    ):
        self.knowledge_base = knowledge_base
        self.mean_random_metrics = mean_random_metrics
        self.mean_dice_metrics = mean_dice_metrics

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


@dataclass
class OntologyParseResult:

    knowledge_base: KnowledgeBase
    all_individuals: list[str]
    triples: list[Triple]

    def __init__(
        self,
        knowledge_base: KnowledgeBase,
        all_individuals: list[str],
        triples: list[Triple]
    ):
        self.knowledge_base = knowledge_base
        self.all_individuals = all_individuals
        self.triples = triples


    @classmethod
    def from_dict(cls, data: dict) -> "OntologyParseResult":
        knowledge_base = data.get("knowledge_base", {})
        all_individuals = data.get("all_individuals", [])
        triples = data.get("triples", [])

        return cls(
            knowledge_base=knowledge_base,
            all_individuals=all_individuals,
            triples=triples
        )

    def to_dict(self) -> dict:
        return {
            "knowledge_base": self.knowledge_base,
            "all_individuals": self.all_individuals,
            "triples": self.triples,
        }


@dataclass
class HardnessAnnotationResult:
    annotated_problems: list[LearningProblem]
    unparsed_problems: list[str]
    target_extensions: dict[str, frozenset[str]]


class ComplexityStratum:
    """
    Represents a complexity stratum, which is a grouping of learning problems
    based on their complexity. Each stratum contains a list of learning problems
    and the corresponding target extensions.
    """
    stratum_name: str
    mean_metric_aggregates: dict[Any, MeanMetricsResult]

    def __init__(
        self,
        stratum_name: str,
        mean_metric_aggregates: dict[Any, MeanMetricsResult]
    ):
        self.stratum_name = stratum_name
        self.mean_metric_aggregates = mean_metric_aggregates

    def to_dict(self) -> dict:
        return {
            "stratum_name": self.stratum_name,
            "mean_metric_aggregates": {
                key: value.to_dict() for key, value in self.mean_metric_aggregates.items()
            }
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ComplexityStratum":
        stratum_name = data.get("stratum_name", "")
        mean_metric_aggregates_data = data.get("mean_metric_aggregates", {})
        mean_metric_aggregates = {
            key: MeanMetricsResult.from_dict(value) for key, value in mean_metric_aggregates_data.items()
        }
        return cls(
            stratum_name=stratum_name,
            mean_metric_aggregates=mean_metric_aggregates
        )
    

class ComplexityResult:
    """
    Represents the result of a complexity analysis, which includes
    multiple complexity strata. Each stratum contains a list of learning problems
    and the corresponding target extensions.
    """
    complexity_strata: dict[str, ComplexityStratum]

    def __init__(
        self,
        complexity_strata: dict[str, ComplexityStratum]
    ):
        self.complexity_strata = complexity_strata

    def to_dict(self) -> dict:
        return {
            "complexity_strata": {
                key: value.to_dict() for key, value in self.complexity_strata.items()
            }
        }



    @classmethod
    def from_dict(cls, data: dict) -> "ComplexityResult":
        complexity_strata_data = data.get("complexity_strata", {})
        complexity_strata = {
            key: ComplexityStratum.from_dict(value) for key, value in complexity_strata_data.items()
        }
        return cls(
            complexity_strata=complexity_strata
        )

