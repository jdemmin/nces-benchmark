# src/models/nces.py
"""NCES training-data preparation, training, and hypothesis evaluation."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

from src.benchmarking.metrics import (
    calculate_extension_metrics,
    calculate_metrics,
    compute_lift,
)
from src.config import EmbeddingSettings, NCESSettings
from src.data.lp import LearningProblem
from src.data.ontology import concept_extension
from src.data.results import (
    EmbeddingResult,
    LearningProblemResult,
    MetricsResult,
    NCESStats,
    TargetExtensionStructure,
)
from src.random_utils import seed_everything

logger = logging.getLogger(__name__)


def prepare_nces_training_data(
     problems: Sequence[LearningProblem], path: Path, seed: int
 ) -> list[tuple[str, dict[str, list[str]]]]:
    """Build and persist the NCES training data.

    NCES expects a sequence of ``(target_concept, {"positive examples": [...],
    "negative examples": [...]})`` tuples keyed on **local names**.
    """
    data = [problem.as_nces_datapoint() for problem in problems]
    if len(dict(data)) != len(data):
        logger.warning(
            "%d of %d learning problems share a target concept; "
            "the persisted artifact is keyed by concept and will collapse them",
            len(data) - len(dict(data)),
            len(data),
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            [{"target_concept": c, **ex} for c, ex in data],
            handle,
            indent=2,
            ensure_ascii=False,
        )
    return data


def build_nces(
    kb_path: Path,
    embeddings_path: Path,
    trained_models_dir: Path,
    settings: NCESSettings,
    m: int,
    *,
    load_pretrained: bool,
):
    """Construct an ``ontolearn.concept_learner.NCES`` instance.

    ``auto_train`` is disabled so that training is always explicit and driven
    by the benchmark's own train split.
    """
    from ontolearn.knowledge_base import KnowledgeBase
    from ontolearn.learners import NCES

    trained_models_dir.mkdir(parents=True, exist_ok=True)
    return NCES(
        knowledge_base=KnowledgeBase(path=str(kb_path)),
        path_of_embeddings=str(embeddings_path),
        path_of_trained_models=str(trained_models_dir),
        learner_names=settings.learner_names,
        quality_func=None,
        proj_dim=settings.proj_dim,
        rnn_n_layers=settings.rnn_n_layers,
        drop_prob=settings.drop_prob,
        num_heads=settings.num_heads,
        num_seeds=settings.num_seeds,
        m=m,
        max_length=settings.max_length,
        load_pretrained=load_pretrained,
        sorted_examples=settings.sorted_examples,
        verbose=0,
        num_predictions=settings.num_predictions,
        auto_train=False,
    )


def train_nces(
    kb_path: Path,
    embeddings_path: Path,
    trained_models_dir: Path,
    train_data: Sequence[tuple[str, dict[str, list[str]]]],
    settings: NCESSettings,
    m: int,
    seed: int,
) -> NCESStats:
    """Train NCES on the train split and save the weights.

    Upstream ``NCESTrainer.train`` initializes ``best_weights = (None, None)``
    and only replaces it when an epoch's mean *hard* accuracy is strictly
    greater than the running best, which starts at 0. When no epoch achieves
    non-zero hard accuracy -- routine for small learning-problem counts or
    short training -- it calls ``load_state_dict(None)`` and raises TypeError.
    The trained parameters are unaffected: ``train_step`` updates the module in
    place, so only the "restore the best epoch" step is lost. Catch the
    TypeError and persist the final-epoch weights instead.
    """

    seed_everything(seed)  # Ensure reproducibility for NCES training
    model = build_nces(
        kb_path,
        embeddings_path,
        trained_models_dir,
        settings,
        load_pretrained=False,
        m=m,
    )
    logger.info(
        "Training NCES %s for %d epochs on %d learning problems",
        settings.learner_name,
        settings.epochs,
        len(train_data),
    )
    logger.info("Starting NCES training.")
    started = time.perf_counter()
    degraded = False
    try:
        seed_everything(seed)  # Ensure reproducibility for NCES training
        model.train(
            data=sorted(train_data, key=lambda x: x[0]), # type: ignore // NUH UH
            epochs=settings.epochs,
            batch_size=settings.batch_size,
            # Force single-threaded data loading for reproducibility
            num_workers=0,
            learning_rate=settings.learning_rate,
            save_model=True,
            storage_path=str(trained_models_dir),
            record_runtime=True,
        )  
    except TypeError as error:
        if "state_dict to be dict-like" not in str(error):
            raise
        degraded = True
        logger.warning(
            "Upstream NCESTrainer never recorded best weights (no epoch "
            "exceeded zero hard accuracy); persisted final-epoch weights "
            "instead of best-epoch weights."
        )
        # Save the final-epoch weights to the trained models directory, since
        # the upstream trainer did not do so.
        _save_final_weights(model, trained_models_dir, settings)
        
    runtime = time.perf_counter() - started
    return NCESStats(
        learner_name=settings.learner_name,
        runtime_seconds=round(runtime, 3),
        degraded=degraded,
    )

def _save_final_weights(model, trained_models_dir: Path, settings: NCESSettings) -> None:
    """Write the artifacts upstream would have written after restoring weights."""

    import json as _json

    import numpy as np
    import torch

    models_dir = trained_models_dir
    models_dir.mkdir(parents=True, exist_ok=True)
    learners = (
        model.model 
        if isinstance(model.model, dict) 
        else {settings.learner_name: model.model}
    )
    for learner_name, module in learners.items():
        net = module["model"] if isinstance(module, dict) else module
        net = getattr(net, "module", net)  # unwrap DataParallel
        torch.save(
            net.state_dict(), models_dir / f"trained_{learner_name}.pt"
        )
        logger.info(f"Recorded net fingerprint: {_fingerprint(net)}.")

    with (models_dir / "config.json").open("w", encoding="utf-8") as handle:
        _json.dump(
            {
                "max_length": model.max_length,
                "proj_dim": model.proj_dim,
                "num_heads": model.num_heads,
                "num_seeds": model.num_seeds,
                "rnn_n_layers": model.rnn_n_layers,
            },
            handle,
        )
    with (models_dir / "vocab.json").open("w", encoding="utf-8") as handle:
        _json.dump(model.vocab, handle)
    np.save(models_dir / "inv_vocab.npy", model.inv_vocab)
    logger.info(
                "Saved final-epoch weights to %s",
                trained_models_dir / f"trained_{settings.learner_name}.pt",
            )


def evaluate_nces(
    kb_path: Path,
    embeddings_path: Path,
    trained_models_dir: Path,
    problems: Sequence[LearningProblem],
    settings: NCESSettings,
    m: int,
    target_extensions: Mapping[str, frozenset[str]] | None = None,
    *,
    knowledge_base,
    all_individuals: Sequence[str],
    split_name: str,
    trained_model_settings: EmbeddingSettings,
    seed: int,
    degraded: bool,
) -> EmbeddingResult:
    """Evaluate the trained NCES learner on a held-out learning-problem split.

    Each hypothesis is rendered to DL syntax, its extension is computed with
    the reasoner, and the extension is compared to the target extension.
    """
    
    if not problems:
        logger.warning(
            "No learning problems provided for evaluation; returning empty results."
        )
        return EmbeddingResult(
            split_name=split_name,
            learning_problem_results=[],
            embedding_settings=trained_model_settings,
            nces_stats=NCESStats(
                learner_name=settings.learner_name,
                runtime_seconds=0.0,
                degraded=degraded,
            ),
            number_of_problems=0,
            number_of_successful_problems=0,
        )
    eval_timer = time.perf_counter()
    seed_everything(seed)  # Ensure reproducibility for NCES evaluation
    model = build_nces(
        kb_path,
        embeddings_path,
        _get_valid_dir_path(trained_models_dir, settings),
        settings,
        load_pretrained=True,
        m=m,
    )
    # Evaluate each learning problem (fit) and collect the results.
    records = _build_records(
        problems=problems,
        model=model,
        knowledge_base=knowledge_base,
        target_extensions=target_extensions,
        all_individuals=all_individuals,
        seed=seed,
    )
    final_time = round(time.perf_counter() - eval_timer, 3)
    scored = [record for record in records if record.error is None]
    logger.info("Collecting mean metrics across %d successful problems", len(scored))
    mean_metrics = calculate_metrics([
        record.metrics for record in scored 
        if record is not None and record.metrics is not None and record.error is None
    ])
    logger.info("Finished computing mean metrics across %d successful problems", len(scored))
    logger.info(
        "NCES evaluation completed in %.3f seconds: %d problems, %d successful",
        time.perf_counter() - eval_timer,
        len(records),
        len(scored),
    )
    return EmbeddingResult(
        split_name=split_name,
        number_of_problems=len(records),
        number_of_successful_problems=len(scored),
        mean_metrics=mean_metrics,
        learning_problem_results=sorted(records, key=lambda r: r.learning_problem.id),
        embedding_settings=trained_model_settings,
        nces_stats=NCESStats(
            learner_name=settings.learner_name,
            runtime_seconds=final_time,
            degraded=degraded,
        ),
    )


def _get_valid_dir_path(path: Path, nces_settings: NCESSettings) -> Path:
    true_trained_model_path = path
    try:
        assert_model_dir_contains_needed_files(true_trained_model_path, nces_settings)
    except FileNotFoundError as e:
        try:
            logger.warning(
                "Trained model directory '%s' does not contain the expected files. "
                "Checking the 'trained_models' subdirectory.",
                true_trained_model_path,
            )
            assert_model_dir_contains_needed_files(true_trained_model_path / "trained_models", nces_settings)
            true_trained_model_path = true_trained_model_path / "trained_models"
            logger.info(
                "Found expected files in the 'trained_models' subdirectory: %s",
                true_trained_model_path,
            )
        except FileNotFoundError:
            logger.error(
                "Trained model directory '%s' does not contain the expected files. "
                "Please ensure that the NCES model was trained correctly and that the "
                "trained model files are present in the directory.",
                true_trained_model_path,
            )
            raise e
    return true_trained_model_path

def _fingerprint(net) -> float:
    return sum(float(p.detach().abs().sum()) for p in net.parameters())

# TODO: Keep track of the signature of the function to avoid accidental changes that break the benchmark.
def _build_records(
        problems, 
        model,
        knowledge_base, 
        target_extensions, 
        all_individuals,
        seed: int,
    ) -> list[LearningProblemResult]:
    from ontolearn.learning_problem import PosNegLPStandard
    from owlapy.class_expression import OWLClassExpression
    from owlapy.owl_individual import OWLNamedIndividual
    from owlapy.render import DLSyntaxObjectRenderer

    renderer = DLSyntaxObjectRenderer()
    records: list[LearningProblemResult] = []
    for problem in sorted(problems, key=lambda p: p.id):
        positives = {OWLNamedIndividual(iri) for iri in sorted(problem.pos_example)}
        negatives = {OWLNamedIndividual(iri) for iri in sorted(problem.neg_example)}
        lp = PosNegLPStandard(pos=set(positives), neg=set(negatives))
        started = time.perf_counter()
        try:
            seed_everything(seed)  # Ensure reproducibility for NCES fitting
            predictions = model.fit(lp)
            # returns Union type. Expect a single OWLClassExpression,
            # so check the type and raise if not.
            hypothesis = (model.best_hypotheses())
            if type(hypothesis) is list and len(hypothesis) > 0:
                hypothesis = hypothesis[0]
            if not (isinstance(hypothesis, OWLClassExpression) or hypothesis is None):
                logger.warning(
                    "NCES returned unexpected hypothesis type for problem %s: %s",
                    problem.id,
                    type(hypothesis),
                )
                raise TypeError(
                    f"Expected a single OWLClassExpression, got {type(hypothesis)}"
                )
        except Exception as error: # NoQA: BLE001
            logger.warning(
                "NCES failed on %s (%s): %s: %s",
                problem.id,
                problem.target_concept,
                type(error).__name__,
                error,
            )
            failed_learning_problem = LearningProblemResult(
                learning_problem=problem,
                error=type(error).__name__ + ": " + str(error),
            )
            records.append(failed_learning_problem)
            continue
        del predictions
        expression = getattr(hypothesis, "concept", hypothesis)
        hypothesis_dl = renderer.render(expression) if expression else ""
        predicted = (
            concept_extension(knowledge_base, hypothesis_dl)
            if hypothesis_dl
            else frozenset()
        )
        # Reuse the extension computed during hardness annotation; the two
        # stages must agree, including when the parse fallback fired.
        if target_extensions is not None and problem.id in target_extensions:
            target = target_extensions[problem.id]
        else:
            target = concept_extension(knowledge_base, problem.target_concept)
            target = target or frozenset(problem.pos_example)

        runtime = round(time.perf_counter() - started, 3)
        metrics = calculate_extension_metrics(predicted, target, all_individuals)
        lift = compute_lift(complexity=problem.complexity, f1=metrics.f1)
        if lift is None:
            logger.warning(
                "Learning problem %s has no hardness annotation; "
                "lift is undefined and will be reported as 0.0",
                problem.id,
            )
            lift = 0.0
        metric_result = MetricsResult(
            accuracy=metrics.accuracy,
            f1_score=metrics.f1,
            jaccard=metrics.jaccard,
            precision=metrics.precision,
            recall=metrics.recall,
            intersection=metrics.intersection,
            union=metrics.union,
            semantic_equivalence=metrics.semantic_equivalence,
            lift=lift, # type: ignore //is checked above
        )

        negative_extension = set(all_individuals) - set(target)

        learning_problem_result = LearningProblemResult(
            learning_problem=problem,
            hypothesis=hypothesis_dl,
            target_extension=TargetExtensionStructure(
                positive=len(target),
                negative=len(negative_extension),
            ),
            hypothesis_extension=TargetExtensionStructure(
                positive=len(predicted),
                negative=len(set(all_individuals) - set(predicted)),
            ),
            metrics=metric_result,
            runtime=runtime,
        )
        records.append(learning_problem_result)
    return records


def assert_model_dir_contains_needed_files(
        trained_models_dir: Path, 
        settings: NCESSettings
    ) -> None:
    """Check that the trained models directory contains the expected files."""
    if not trained_models_dir.exists():
        logger.error(
            "Trained models directory does not exist: %s", trained_models_dir
        )
        raise FileNotFoundError(
            f"Trained models directory does not exist: {trained_models_dir}"
        )

    expected = trained_models_dir / f"trained_{settings.learner_name}.pt"
    if not expected.is_file():
        contents = sorted(p.name for p in trained_models_dir.glob('*'))
        logger.error(
            "No trained NCES weights at %s; evaluation would score an "
            "untrained learner. "
            "Contents: %s",
            expected,
            contents,
        )
        raise FileNotFoundError(
            f"No trained NCES weights at {expected};evaluation would score an "
            "untrained learner. "
            f"Contents: {contents}"
        )
    expected_files = ["vocab.json", "inv_vocab.npy"]
    missing_files = [f for f in expected_files
                     if not (trained_models_dir / f).exists()]
    if len(missing_files) > 0:
        msg = (
            f"Trained models directory {trained_models_dir} is missing "
            f"expected files: {missing_files}. "
            f"Directory contents: {list(trained_models_dir.iterdir())}"
        )
        logger.error(msg)
        raise FileNotFoundError(msg)
