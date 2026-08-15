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
from src.data.lp import (
    generate_learning_problems,
    save_learning_problems,
    save_split,
    split_learning_problems,
)
from src.data.ontology import individual_iris, load_knowledge_base
from src.logging_utils import configure_logging
from src.models.dice import build_embeddings
from src.models.nces import (
    evaluate_nces,
    prepare_nces_training_data,
    train_nces,
)
from src.paths import OUTPUT_DIR, resolve_knowledge_base, run_paths

logger = logging.getLogger(__name__)


def run_single(
    knowledge_base_name: str,
    seed: int,
    config: BenchmarkConfiguration,
    *,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Execute one benchmark run: one knowledge base, one seed.

    Stages: parse the knowledge base, generate learning problems, split them,
    build embeddings for every condition, then train and evaluate NCES per
    condition.
    """
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

        problems = generate_learning_problems(
            kb_path,
            paths.nces_data_dir,
            config.data_generation,
            seed=seed,
        )
        save_learning_problems(problems, paths.learning_problems_path)
        split = split_learning_problems(problems, seed=seed)
        save_split(split, paths.nces_data_dir)

        embeddings = build_embeddings(
            kb_path,
            paths.embeddings_dir,
            paths.embeddings_data_dir,
            config.embedding,
            seed=seed,
            conditions=config.project.embedding_conditions,
        )

        train_data = prepare_nces_training_data(
            split["train"], paths.nces_data_dir / "nces_train_data.json"
        )

        conditions: dict[str, Any] = {}
        for condition, embedding in embeddings.items():
            models_dir = paths.trained_models_dir / condition
            training = train_nces(
                kb_path,
                embedding.embeddings_path,
                models_dir,
                train_data,
                config.nces,
            )
            evaluation = {
                name: evaluate_nces(
                    kb_path,
                    embedding.embeddings_path,
                    models_dir,
                    split[name],
                    config.nces,
                    knowledge_base=knowledge_base,
                    all_individuals=all_individuals,
                    split_name=name,
                )
                for name in ("validation", "test")
            }
            conditions[condition] = {
                "best_embedding_config": embedding.to_dict(),
                "training": training,
                "evaluation": evaluation,
            }

        report = {
            "knowledge_base": knowledge_base_name,
            "knowledge_base_path": str(kb_path),
            "seed": seed,
            "benchmark_name": config.project.benchmark_name,
            "num_individuals": len(all_individuals),
            "num_learning_problems": len(problems),
            "split_sizes": {k: len(v) for k, v in split.items()},
            "embedding_conditions": conditions,
            "configuration": config.to_dict(),
            "runtime_seconds": round(time.perf_counter() - started, 3),
        }
        _write_json(report, paths.report_path)
        return report
    finally:
        if handler is not None:
            logging.getLogger().removeHandler(handler)
            handler.close()


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
            except Exception as error:  # noqa: BLE001 - keep the suite running
                logger.exception("Benchmark run failed for %s seed %d", kb_name, seed)
                failures.append(
                    {"knowledge_base": kb_name, "seed": seed, "error": str(error)}
                )
                continue
            kb_reports.append(report)
            reports.append(report)

        if kb_reports:
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


def _summarise(kb_name: str, reports: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate benchmark runs for one knowledge base across seeds."""
    conditions: dict[str, dict[str, list[float]]] = {}
    for report in reports:
        for condition, payload in report["embedding_conditions"].items():
            test = payload["evaluation"]["test"]
            bucket = conditions.setdefault(
                condition, {"mean_f1": [], "mean_accuracy": [], "sem_eq": []}
            )
            bucket["mean_f1"].append(float(test.get("mean_f1", 0.0)))
            bucket["mean_accuracy"].append(float(test.get("mean_accuracy", 0.0)))
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
