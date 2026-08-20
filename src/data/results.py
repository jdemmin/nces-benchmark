#src/data/results.py

import dataclasses
from dataclasses import dataclass

from src.config import EmbeddingSettings
from src.data.lp import LearningProblem


class MeanMetricsResult:
    """"
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
            mean_intersection=data.get("mean_intersection", 0),
            mean_union=data.get("mean_union", 0),
            mean_lift=data.get("mean_lift", 0)
        )

    @classmethod
    def to_dict(cls, mean_metrics_result: "MeanMetricsResult") -> dict:
        return {
            "mean_accuracy": mean_metrics_result.mean_accuracy,
            "mean_precision": mean_metrics_result.mean_precision,
            "mean_recall": mean_metrics_result.mean_recall,
            "mean_f1_score": mean_metrics_result.mean_f1_score,
            "mean_jaccard": mean_metrics_result.mean_jaccard,
            "mean_semantic_equivalence": mean_metrics_result.mean_semantic_equivalence,
            "mean_intersection": mean_metrics_result.mean_intersection,
            "mean_union": mean_metrics_result.mean_union,
            "mean_lift": mean_metrics_result.mean_lift
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
    lift: int

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
        lift: int
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

    @classmethod
    def to_dict(cls, metrics_result: "MetricsResult") -> dict:
        return {
            "accuracy": metrics_result.accuracy,
            "precision": metrics_result.precision,
            "recall": metrics_result.recall,
            "f1_score": metrics_result.f1_score,
            "jaccard": metrics_result.jaccard,
            "semantic_equivalence": metrics_result.semantic_equivalence,
            "intersection": metrics_result.intersection,
            "union": metrics_result.union,
            "lift": metrics_result.lift
        }

class LearningProblemResult:
    """
    Represents the result of a single learning problem,
    including the learning problem details, the
    hypothesis, the target extension, and the evaluation
    metrics.
    """
    learning_problem: LearningProblem
    hypotesis: str
    target_extension: dict[str, int]
    metrics: MetricsResult

    def __init__(
        self,
        learning_problem: LearningProblem,
        hypotesis: str,
        target_extension: dict[str, int],
        metrics: MetricsResult
    ):
        self.learning_problem = learning_problem
        self.hypotesis = hypotesis
        self.target_extension = target_extension
        self.metrics = metrics

    @classmethod
    def from_dict(cls, data: dict) -> "LearningProblemResult":
        learning_problem = LearningProblem.from_dict(data.get("learning_problem", {}))
        hypotesis = data.get("hypotesis", "")
        target_extension = data.get("target_extension", {})
        metrics_data = data.get("metrics", {})
        metrics = MetricsResult.from_dict(metrics_data)

        return cls(
            learning_problem=learning_problem,
            hypotesis=hypotesis,
            target_extension=target_extension,
            metrics=metrics
        )

    @classmethod
    def to_dict(cls, learning_problem_result: "LearningProblemResult") -> dict:
        return {
            "learning_problem": learning_problem_result.learning_problem.to_dict(),
            "hypotesis": learning_problem_result.hypotesis,
            "target_extension": learning_problem_result.target_extension,
            "metrics": MetricsResult.to_dict(learning_problem_result.metrics)
        }

@dataclass
class NCESStats:
    learner_name: str
    runtime_seconds: float

class EmbeddingResult:
    """
    Represents the mean result of a single embedding
    across multiple learning problems.
    """
    mean_metrics: MeanMetricsResult
    learning_problem_results: list[LearningProblemResult]
    embedding_settings: EmbeddingSettings
    nces_stats: NCESStats

    def __init__(
        self,
        mean_metrics: MeanMetricsResult,
        learning_problem_results: list[LearningProblemResult],
        embedding_settings: EmbeddingSettings,
        nces_stats: NCESStats
    ):
        self.mean_metrics = mean_metrics
        self.learning_problem_results = learning_problem_results
        self.embedding_settings = embedding_settings
        self.nces_stats = nces_stats

    @classmethod
    def from_dict(cls, data: dict) -> "EmbeddingResult":
        mean_metrics_data = data.get("mean_metrics", {})
        mean_metrics = MeanMetricsResult.from_dict(mean_metrics_data)
        learning_problem_results_data = data.get("learning_problem_results", [])
        learning_problem_results = [
            LearningProblemResult.from_dict(lpr_data)
            for lpr_data in learning_problem_results_data
        ]

        return cls(
            mean_metrics=mean_metrics,
            learning_problem_results=learning_problem_results,
            embedding_settings=EmbeddingSettings.from_dict(data.get("embedding_settings", {})),
            nces_stats=NCESStats(**data.get("nces_stats", {}))
        )

    @classmethod
    def to_dict(cls, embedding_result: "EmbeddingResult") -> dict:
        return {
            "mean_metrics": MeanMetricsResult.to_dict(embedding_result.mean_metrics),
            "learning_problem_results": [
                LearningProblemResult.to_dict(lpr)
                for lpr in embedding_result.learning_problem_results
            ],
            "embedding_settings": EmbeddingSettings.to_dict(embedding_result.embedding_settings),
            "nces_stats": dataclasses.asdict(embedding_result.nces_stats)
        }

class SingleRunResult:
    """
    Represents the result of a single run across 
    a knowledge base.
    """
    knowledge_base: str
    random_embedding_result: EmbeddingResult
    dice_embedding_result: EmbeddingResult

    def __init__(
        self,
        knowledge_base: str,
        random_embedding_result: EmbeddingResult,
        dice_embedding_result: EmbeddingResult,
    ):
        self.knowledge_base = knowledge_base
        self.random_embedding_result = random_embedding_result
        self.dice_embedding_result = dice_embedding_result

    @classmethod
    def from_dict(cls, data: dict) -> "SingleRunResult":
        knowledge_base = data.get("knowledge_base", "")
        random_embedding_result = EmbeddingResult.from_dict(data.get("random_embedding_result", {}))
        dice_embedding_result = EmbeddingResult.from_dict(data.get("dice_embedding_result", {}))

        return cls(
            knowledge_base=knowledge_base,
            random_embedding_result=random_embedding_result,
            dice_embedding_result=dice_embedding_result
        )

    @classmethod
    def to_dict(cls, single_run_result: "SingleRunResult") -> dict:
        return {
            "knowledge_base": single_run_result.knowledge_base,
            "random_embedding_result": EmbeddingResult.to_dict(single_run_result.random_embedding_result),
            "dice_embedding_result": EmbeddingResult.to_dict(single_run_result.dice_embedding_result)
        }

class KnowledgeBaseResult:
    """
    Represents the mean result of all seeded 
    runs across a knowledge base.
    """
    knowledge_base: str
    mean_metrics: MeanMetricsResult

    def __init__(
        self,
        knowledge_base: str,
        mean_metrics: MeanMetricsResult
    ):
        self.knowledge_base = knowledge_base
        self.mean_metrics = mean_metrics

    @classmethod
    def from_dict(cls, data: dict) -> "KnowledgeBaseResult":
        knowledge_base = data.get("knowledge_base", "")
        mean_metrics_data = data.get("mean_metrics", {})
        mean_metrics = MeanMetricsResult.from_dict(mean_metrics_data)

        return cls(
            knowledge_base=knowledge_base,
            mean_metrics=mean_metrics
        )

    @classmethod
    def to_dict(cls, knowledge_base_result: "KnowledgeBaseResult") -> dict:
        return {
            "knowledge_base": knowledge_base_result.knowledge_base,
            "mean_metrics": MeanMetricsResult.to_dict(knowledge_base_result.mean_metrics)
        }

