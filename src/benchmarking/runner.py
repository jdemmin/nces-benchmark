# src/benchmarking/runner.py
"""End-to-end benchmark orchestration across knowledge bases and seeds."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from src.config import BenchmarkConfiguration
from src.data.complexity import annotate_hardness
from src.data.lp import (
    LearningProblem,
    generate_learning_problems,
    save_learning_problems,
    save_split,
    split_learning_problems,
)
from src.data.ontology import (
    compute_atomic_class_extensions,
    concept_extension,
    individual_iris,
    load_knowledge_base,
    parse_triples,
)
from src.logging_utils import configure_logging
from src.models.dice import build_embeddings
from src.models.nces import (
    evaluate_nces,
    prepare_nces_training_data,
    train_nces,
)
from src.paths import OUTPUT_DIR, RunPaths, resolve_knowledge_base, run_paths

logger = logging.getLogger(__name__)


def run_single(
    knowledge_base_name: str,
    seed: int,
    config: BenchmarkConfiguration,
    output_dir: Path,
) -> dict[str, Any]:
    """Execute one benchmark run: one (knowledge base, seed) pair."""
    # ... ontology parsing ...
    kb_path = resolve_knowledge_base(knowledge_base_name)
    paths = run_paths(
        config.project.benchmark_name,
        seed,
        knowledge_base_name,
        output_dir=output_dir,
    )
    paths.mkdirs()
    handler = configure_logging(paths.logs_dir / f"{knowledge_base_name}.log")
    started = time.perf_counter()
    try:
        knowledge_base = load_knowledge_base(kb_path)
        all_individuals = individual_iris(knowledge_base)
        triples = parse_triples(kb_path)

        # --- Stage 2: learning-problem generation -------------------------------
        logger.info("\n----- Reached Stage 2 -----\n")
        problems = generate_learning_problems(
            kb_path,
            paths.nces_data_dir,
            config.data_generation,
            seed=seed,
        )
        if not problems:
            raise RuntimeError(
                f"No non-degenerate learning problems generated for {knowledge_base_name}."
            )

        # --- Stage 3: hardness annotation --------------------------------------
        logger.info("\n----- Reached Stage 3 -----\n")
        # Knowledge base only. No embedding-derived quantity may enter here, or
        # the benchmark's independent variable is contaminated.
        logger.info("Annotating hardness for %d learning problems", len(problems))
        atomic_extensions = compute_atomic_class_extensions(knowledge_base)
        universe = frozenset(all_individuals)

        # Cached so the evaluation stage does not recompute target extensions.
        target_extensions: dict[str, frozenset[str]] = {}
        unparsed: list[str] = []
        annotated: list[LearningProblem] = []

        for problem in problems:
            extension = concept_extension(knowledge_base, problem.target_concept)
            if not extension:
                # Same fallback the evaluation stage uses: sampled positives as
                # IRI strings. Must not be OWL objects -- they never compare
                # equal to the IRI strings in atomic_extensions, which would
                # silently zero out every hardness field.
                extension = frozenset(problem.pos_example)
                unparsed.append(problem.id)

            target_extensions[problem.id] = extension
            annotated.append(
                problem.annotate_complexity(
                    annotate_hardness(
                        problem.complexity,
                        target_extension=extension,
                        all_individuals=universe,
                        atomic_extensions=atomic_extensions,
                    )
                )
            )

        problems = annotated

        if unparsed:
            logger.warning(
                "%d of %d target concepts could not be parsed; used sampled "
                "positives as their extension (ids: %s%s)",
                len(unparsed),
                len(problems),
                ", ".join(unparsed[:5]),
                ", ..." if len(unparsed) > 5 else "",
            )

        _log_complexity_distribution(problems)
        save_learning_problems(problems, paths.nces_data_dir / "learning_problems.json")

        # --- Stage 4: learning-problem splitting -------------------------------
        logger.info("\n----- Reached Stage 4 -----\n")
        split = split_learning_problems(
            problems,
            seed=seed,
            stratify_by=config.project.stratify_by,
        )
        save_split(split, paths.nces_data_dir)
        logger.info(
            "Split %d learning problems into %d train / %d validation / %d test",
            len(problems),
            len(split["train"]),
            len(split["validation"]),
            len(split["test"]),
        )

        # --- Stage 5: embedding stage ------------------------------------------
        logger.info("\n----- Reached Stage 5 -----\n")
        embedding_report = run_embedding_stage(
            paths=paths,
            kb_path=kb_path,
            benchmark_settings=config,
            seed=seed,
        )

        # --- Stages 6-7: NCES training and evaluation, per condition -----------
        logger.info("\n----- Reached Stage 6/7 -----\n")
        train_data = prepare_nces_training_data(
            split["train"], paths.nces_data_dir / "nces_train_data.json"
        )

        conditions: dict[str, Any] = {}
        for condition in config.project.embedding_conditions:
            embeddings_path = embedding_report[condition].embeddings_path
            trained_models_dir = paths.trained_models_dir / condition

            training = train_nces(
                kb_path,
                Path(embeddings_path),
                trained_models_dir,
                train_data,
                config.nces,
            )
            evaluation = evaluate_nces(
                kb_path,
                Path(embeddings_path),
                trained_models_dir,
                split["test"],
                config.nces,
                knowledge_base=knowledge_base,
                all_individuals=all_individuals,
                target_extensions=target_extensions,
                split_name="test",
            )
            conditions[condition] = {"training": training, "evaluation": evaluation}
        return {
            "knowledge_base": knowledge_base_name,
            "seed": seed,
            "configuration": config.to_dict(),
            "num_individuals": len(all_individuals),
            "num_triples": len(triples),
            "num_learning_problems": len(problems),
            "num_unparsed_targets": len(unparsed),
            "num_atomic_classes": len(atomic_extensions),
            "split_sizes": {name: len(items) for name, items in split.items()},
            "embedding": embedding_report,
            "embedding_conditions": conditions,
            "runtime_seconds" : round(time.perf_counter() - started, 3)
        }
    finally:
        logger.info("\n----- All stages complete -----\n")
        if handler is not None:
            logging.getLogger().removeHandler(handler)
            handler.close()
        

def _log_complexity_distribution(problems: Sequence[LearningProblem]) -> None:
    """Log the spread along each complexity axis.

    Thin strata make stratified splitting degenerate -- a stratum of one or
    two problems is handed entirely to train -- so this is the signal for
    whether the configured ``stratify_by`` axis is viable.
    """
    axes: dict[str, dict[str, int]] = {
        "dl_length": {},
        "depth": {},
        "expressivity": {},
        "redundant": {},
    }
    for problem in problems:
        for axis, counts in axes.items():
            try:
                key = str(getattr(problem.complexity, axis))
            except AttributeError:
                key = str(getattr(problem.complexity.hardness, axis))
            counts[key] = counts.get(key, 0) + 1

    for axis, counts in axes.items():
        rendered = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
        logger.info("Complexity distribution by %s: %s", axis, rendered)
        thin = [k for k, v in counts.items() if v < 3]
        if thin and axis != "redundant":
            logger.warning(
                "Axis %r has strata with fewer than 3 learning problems (%s); "
                "stratifying on it will starve the validation and test splits",
                axis,
                ", ".join(sorted(thin)),
            )


def run_benchmark(
    config: BenchmarkConfiguration,
    *,
    knowledge_bases: Sequence[str] | None = None,
    seeds: Sequence[int] | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Run every (knowledge base, seed) combination and aggregate results."""
    selected_kbs = list(knowledge_bases or config.knowledge_bases)
    selected_seeds = list(seeds or config.project.seeds)
    base = output_dir or OUTPUT_DIR
    benchmark_dir = base / config.project.benchmark_name

    reports: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for kb_name in selected_kbs:
        kb_reports: list[dict[str, Any]] = []
        for seed in selected_seeds:
            logger.info("=== %s | seed %d ===", kb_name, seed)
            try:
                report = run_single(kb_name, seed, config, output_dir=base)
            except Exception as error:
                logger.exception("Benchmark run failed for %s seed %d", kb_name, seed)
                failures.append(
                    {"knowledge_base": kb_name, "seed": seed, "error": str(error)}
                )
                continue
            kb_reports.append(report)
            reports.append(report)

        if kb_reports:
            # write unaggregated reports
            for i in range(len(kb_reports)):
                if output_dir != None:
                    _write_json(path= benchmark_dir / "kb_reports" / f"kb_report_{i}.json", payload=kb_reports[i])
            _write_json(
                _summarise(kb_name, kb_reports),
                benchmark_dir / f"{kb_name}_summary.json",
            )

    summary = {
        "benchmark_name": config.project.benchmark_name,
        "knowledge_bases": selected_kbs,
        "seeds": selected_seeds,
        "num_runs": len(reports),
        "failures": failures,
        "configuration": config.to_dict(),
        "per_knowledge_base": {
            kb: _summarise(
                kb, [r for r in reports if r["knowledge_base"] == kb]
            )
            for kb in selected_kbs
            if any(r["knowledge_base"] == kb for r in reports)
        },
    }
    _write_json(summary, benchmark_dir / "benchmark_summary.json")
    return summary

def run_embedding_stage(
        paths: RunPaths,
        kb_path: Path,
        benchmark_settings: BenchmarkConfiguration,
        seed: int,
    )-> dict[str, Any]:
    
    return build_embeddings(
            kb_path,
            paths.embeddings_dir,
            paths.embeddings_data_dir,
            benchmark_settings.embedding,
            seed=seed,
            embedding_conditions=benchmark_settings.project.embedding_conditions,
        )


def _summarise(kb_name: str, reports: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate benchmark runs for one knowledge base across seeds."""
    conditions: dict[str, dict[str, list[float]]] = {}
    for report in reports:
        for condition, payload in report["embedding_conditions"].items():
            test = payload["evaluation"]
            bucket = conditions.setdefault(
                condition, {"mean_f1": [], "mean_accuracy": [], "sem_eq": []}
            )
            bucket["mean_f1"].append(
                float(test.get("mean_f1", 0.0))
            )
            bucket["mean_accuracy"].append(
                float(test.get("mean_accuracy", 0.0))
            )
            bucket["sem_eq"].append(
                float(test.get("semantic_equivalence_rate", 0.0))
            )

    return {
        "knowledge_base": kb_name,
        "num_runs": len(reports),
        "seeds": [report["seed"] for report in reports],
        "embedding_conditions": {
            condition: {
                "mean_f1": _avg(values["mean_f1"]),
                "mean_accuracy": _avg(values["mean_accuracy"]),
                "semantic_equivalence_rate": _avg(values["sem_eq"]),
                "per_seed_f1": values["mean_f1"],
            }
            for condition, values in conditions.items()
        },
    }


def _avg(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, default=str)
    logger.info("Wrote %s", path)
