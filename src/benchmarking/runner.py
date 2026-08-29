# src/benchmarking/runner.py
"""End-to-end benchmark orchestration across knowledge bases and seeds."""

import json
import logging
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ontolearn.knowledge_base import KnowledgeBase

from src.benchmarking.metrics import (
    build_complexity_strata,
    get_complexity_summary,
    mean_embeddings_results,
)
from src.benchmarking.wilcoxon import (
    holm_adjust_across_complexity_strata,
    holm_adjust_across_kbs,
)
from src.config import BenchmarkConfiguration, DataGenerationSettings
from src.data.complexity import annotate_hardness
from src.data.lp import (
    LearningProblem,
    generate_learning_problems,
    load_learning_problems,
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
from src.data.results import (
    EmbeddingResult,
    HardnessAnnotationResult,
    KnowledgeBaseFailure,
    KnowledgeBaseResult,
    KnowledgeBaseStats,
    MeanMetricsResult,
    OntologyParseResult,
    SingleRunResult,
)
from src.logging_utils import configure_logging
from src.models.dice import EmbeddingResultDice, build_embeddings
from src.models.nces import (
    evaluate_nces,
    prepare_nces_training_data,
    train_nces,
)
from src.paths import (
    OUTPUT_DIR,
    RunPaths,
    resolve_knowledge_base,
    run_paths,
    update_false_dir_names,
)
from src.random_utils import seed_everything

logger = logging.getLogger(__name__)




def run_benchmark(
    config: BenchmarkConfiguration,
    *,
    knowledge_bases: Sequence[str] | None = None,
    seeds: Sequence[int] | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Run every (knowledge base, seed) combination and aggregate results."""
    _order_embedding_conditions(config.project.embedding_conditions)
    if (
        (config.project.embedding_conditions)[0] == "random"
        and not config.project.embedding_conditions[1:]
        and len(config.project.embedding_conditions) > 1
        ):
        logger.warning(
            "Random embeddings must follow after other embedding conditions." \
            "Otherwise, this can lead to random and dice differing in" \
            "embedding dimensionality." \
        )
        raise ValueError("Random embeddings must follow after other embedding conditions.")

    selected_kbs = list(knowledge_bases or config.knowledge_bases)
    selected_seeds = list(seeds or config.project.seeds)
    base = output_dir or OUTPUT_DIR
    logger.warning(
        "benchmark_name cannot contain train, valid or test in the name."
        "Replacing them with '_trian_', '_vaild_' and '_tset_' respectively."
    )
    updated_benchmark_name = update_false_dir_names(config.project.benchmark_name)
    benchmark_dir = base / Path(updated_benchmark_name)

    reports: dict[str, SingleRunResult] = {}
    failures: dict[str, KnowledgeBaseFailure] = {}

    for seed in selected_seeds:
        for kb_name in selected_kbs:
            logger.info("=== %s | seed %d ===", kb_name, seed)
            try:
                report = run_single(
                    kb_name,
                    seed,
                    config,
                    output_dir=base,
                    benchmark_name=updated_benchmark_name
                )
            except Exception as error:
                logger.exception("Benchmark run failed for %s seed %d", kb_name, seed)
                knowledge_base_failure = KnowledgeBaseFailure(
                    knowledge_base_name=kb_name,
                    seed=seed,
                    error_message=str(error),
                )
                failures[f"{kb_name}_{seed}"] = knowledge_base_failure
                continue
            reports[kb_name] = report
            _write_json(payload=holm_adjust_across_complexity_strata(report), path=benchmark_dir / f"{kb_name}_holm_adjusted_{seed}_across_strata.json")
        _write_json(payload=holm_adjust_across_kbs(reports), path=benchmark_dir / f"KBs_holm_adjusted_{seed}.json")
        # _write_complexity_summary(reports, benchmark_dir, kb_name)
        # _write_embeddings_summary(reports, benchmark_dir, kb_name)
    #_clean_benchmark_dir(benchmark_dir)
    summary = {
        "num_runs": len(reports),
        "failures": failures,
    }
    _write_json(payload=summary, path=benchmark_dir / "benchmark_summary.json")

    return summary


def _clean_benchmark_dir(benchmark_dir: Path):
    logger.info("Proceeding to zip the benchmark results directory %s", benchmark_dir)
    import shutil
    shutil.make_archive(str(benchmark_dir), 'zip', str(benchmark_dir))
    logger.info("Completed zipping the benchmark results directory %s", benchmark_dir)
    logger.info("Proceeding to remove the benchmark results directory %s. Excluding the zip archive.", benchmark_dir)
    shutil.rmtree(benchmark_dir)
    logger.info("Completed removal of the benchmark results directory %s. Excluding the zip archive.", benchmark_dir)


def _write_embeddings_summary(reports: dict[str, SingleRunResult], benchmark_dir: Path, kb_name: str):
    random_mean = mean_embeddings_results(
        [r.random_embedding_result for r in reports.values() if r.random_embedding_result is not None]
    )
    dice_mean = mean_embeddings_results(
        [r.dice_embedding_result for r in reports.values() if r.dice_embedding_result is not None]
    )
    knowledge_base_result = KnowledgeBaseResult(
        knowledge_base=kb_name,
        mean_random_metrics=random_mean,
        mean_dice_metrics=dice_mean,
    )
    _write_json(
        payload=knowledge_base_result.to_dict(),
        path=benchmark_dir / f"{kb_name}_mean_across_seeds.json",
    )


def _write_complexity_summary(reports: dict[str, SingleRunResult], benchmark_dir: Path, kb_name: str):
    complexity_summary = get_complexity_summary([
        # flattens list of lists of LearningProblemResults into a single list
        item for sublist in [
            r.random_embedding_result.learning_problem_results
            for r in reports.values() if r.random_embedding_result is not None
        ]
        for item in sublist
    ])
    _write_json(
        payload=dict(sorted(_convert_complexity_summary_to_dict(complexity_summary).items())),
        path=benchmark_dir / f"{kb_name}_mean_across_seeds_random_complexity_summary.json",
    )
    complexity_summary = get_complexity_summary([
        # flattens list of lists of LearningProblemResults into a single list
        item for sublist in [
            r.dice_embedding_result.learning_problem_results
            for r in reports.values() if r.dice_embedding_result is not None
        ]
        for item in sublist
    ])
    _write_json(
        payload=dict(sorted(_convert_complexity_summary_to_dict(complexity_summary).items())),
        path=benchmark_dir / f"{kb_name}_mean_across_seeds_dice_complexity_summary.json",
    )


def run_single(
    knowledge_base_name: str,
    seed: int,
    config: BenchmarkConfiguration,
    output_dir: Path,
    benchmark_name: str,
) -> SingleRunResult:
    """Execute one benchmark run: one (knowledge base, seed) pair."""

    seed_everything(seed)
    kb_path = resolve_knowledge_base(knowledge_base_name)
    paths = run_paths(
        benchmark_name,
        seed,
        knowledge_base_name,
        output_dir=output_dir,
    )
    paths.mkdirs()
    handler = configure_logging(paths.logs_dir / f"{knowledge_base_name}_{seed}.log")
    started = time.perf_counter()
    try:
        ontology_parse_result = _stage_parse_ontology(kb_path, seed=seed)
        knowledge_base = ontology_parse_result.knowledge_base
        logger.info("Completed Stage 1: Ontology parsing.")

        problems: list[LearningProblem] = []
        split: dict[str, list[LearningProblem]] = {"train": [], "test": []}
        target_extensions: dict[str, frozenset[str]] = {}

        current_data_settings_hash = _hash_data_generation_settings(config.data_generation)
        old_data_settings_hash = _read_data_generation_settings_hash(paths.nces_data_dir)

        hash_file_path = paths.nces_data_dir / "data_generation_settings_hash.txt"
        if old_data_settings_hash is None or old_data_settings_hash != current_data_settings_hash:
            import os

            logger.info("Data generation settings have changed. Updating hash file at %s", hash_file_path)
            if os.path.exists(hash_file_path):
                os.remove(hash_file_path)
            if os.path.exists(paths.learning_problems_path):
                os.remove(paths.learning_problems_path)
            with open(hash_file_path, "x", encoding="utf-8") as f:
                f.write(current_data_settings_hash)
            problems = generate_learning_problems(
                kb_path,
                paths.nces_data_dir,
                config.data_generation,
                seed=current_data_settings_hash,
            )
            logger.info("Completed Stage 2: Learning-problem generation for %d problems.", len(problems))
            if not problems:
                raise RuntimeError(
                    f"No non-degenerate learning problems generated for {knowledge_base_name}."
                )
            split = split_learning_problems(
                problems,
                stratify_by=config.project.stratify_by,
                seed=current_data_settings_hash,
            )
            save_split(split, paths.nces_data_dir)
            len_split_test = len(split["test"])
            len_split_train = len(split["train"])
            logger.info(
                "Split %d learning problems into %d train / %d test",
                len_split_train + len_split_test,
                len_split_train,
                len_split_test,
            )
            logger.info("Completed Stage 3: Learning-problem splitting.")

            unparsed: list[str] = []
        
            for key, value in split.items():
                logger.info(f"Annotating hardness for `{key}` split of size {len(value)}")
                annotation_result = _stage_hardness_annotation(
                    value, knowledge_base, ontology_parse_result.all_individuals
                )
                split[key] = annotation_result.annotated_problems
                unparsed.extend(annotation_result.unparsed_problems)
                target_extensions.update(annotation_result.target_extensions)
            logger.info("Completed Stage 4: Hardness annotation for %d learning problems", len_split_train + len_split_test)

            if unparsed:
                logger.warning(
                    "%d of %d target concepts could not be parsed; used sampled "
                    "positives as their extension (ids: %s%s)",
                    len(unparsed),
                    len(problems),
                    ", ".join(unparsed[:5]),
                    ", ..." if len(unparsed) > 5 else "",
                )
            problems = split["train"] + split["test"]
            _save_benchmark_learning_problems(sorted(problems, key=lambda p: p.id), paths.learning_problems_path)
        else:
            logger.info(
                "Data generation settings have not changed." \
                "Using cached learning problems and splits."
            )
            problems = _load_benchmark_learning_problems(paths.learning_problems_path)
            split = _load_split(paths.nces_data_dir)
            logger.info(
                "Loaded cached learning problems and splits."
            )
            logger.info("Starting Stage 5: Embedding generation.")
        
        embedding_report, m = _embedding_stage(
            paths=paths,
            kb_path=kb_path,
            benchmark_settings=config,
            seed=seed,
        )
        logger.info(
            "Completed Stage 5: Embedding generation for" \
            "all conditions and collected results."
        )
        single_run_result = _stage_train_eval_nces(
            split=split,
            paths=paths,
            kb_path=kb_path,
            knowledge_base=knowledge_base,
            all_individuals=ontology_parse_result.all_individuals,
            target_extensions=target_extensions,
            embedding_report=embedding_report,
            config=config,
            m=m,
            seed=seed
        )
        logger.info(
            "Completed Stage 6: NCES training and evaluation for all conditions."
        )
        single_run_result.set_runtime(round(time.perf_counter() - started, 3))
        # Write single-run result to JSON. Complexity summaries will be written separately.
        _write_json(
            payload=single_run_result.to_dict(),
            path=paths.nces_results_dir / "single_run_result.json",
        )
        logger.info("Wrote single-run result to %s", paths.nces_results_dir / "single_run_result.json")
        dice_complexity_summary = get_complexity_summary(
            single_run_result.dice_embedding_result.learning_problem_results
            if single_run_result.dice_embedding_result else []
        )
        _write_json(
            payload=dict(sorted(_convert_complexity_summary_to_dict(dice_complexity_summary).items())),
            path=paths.nces_results_dir / "dice_complexity_summary.json",
        )
        random_complexity_summary = get_complexity_summary(
            single_run_result.random_embedding_result.learning_problem_results
            if single_run_result.random_embedding_result else []
        )
        _write_json(
            payload=dict(sorted(_convert_complexity_summary_to_dict(random_complexity_summary).items())),
            path=paths.nces_results_dir / "random_complexity_summary.json",
        )
        logger.info(
            "Wrote complexity summaries to %s, %s",
            paths.nces_results_dir / "dice_complexity_summary.json",
            paths.nces_results_dir / "random_complexity_summary.json",
        )
        single_run_result.random_complexity_aggregates = build_complexity_strata(random_complexity_summary)
        single_run_result.dice_complexity_aggregates = build_complexity_strata(dice_complexity_summary)
        atomic_extensions = compute_atomic_class_extensions(knowledge_base)
        knowledge_base_stats = KnowledgeBaseStats(
            knowledge_base_name=knowledge_base_name,
            number_of_individuals=len(ontology_parse_result.all_individuals),
            number_of_triples=len(ontology_parse_result.triples),
            number_of_atomic_classes=len(atomic_extensions),
        )
        _write_json(
            payload=knowledge_base_stats.to_dict(),
            path=paths.kb_dir / f"knowledge_base_stats_{seed}.json",
        )
        logger.info(
            "Wrote knowledge-base stats to %s",
            paths.kb_dir / f"knowledge_base_stats_{seed}.json",
        )
        _remove_trials(path=paths.embeddings_dir)
        return single_run_result 
    finally:
        if handler is not None:
            logging.getLogger().removeHandler(handler)
            handler.close()
        

def _save_benchmark_learning_problems(problems: list[LearningProblem], path: Path) -> None:
    """Save the benchmark learning problems to the specified path."""
    with open(path, "w") as f:
        json.dump([problem.to_dict() for problem in problems], f, indent=4)

def _load_benchmark_learning_problems(path: Path) -> list[LearningProblem]:
    """Load the benchmark learning problems from the specified path."""
    if not path.exists():
        return []
    with open(path, "r") as f:
        return [LearningProblem.from_dict(d) for d in json.load(f)]

def _load_split(path: Path) -> dict[str, list[LearningProblem]]:
    """Load the data split from the specified directory."""
    split_file_path = path / "train_problems.json"
    split = {"train": [], "test": []}
    if split_file_path.exists():
        with open(split_file_path, "r") as f:
            split["train"].extend(LearningProblem.from_dict(d) for d in json.load(f))
            split["train"] = sorted(split["train"], key=lambda p: p.id)
    split_file_path = path / "test_problems.json"
    if split_file_path.exists():
        with open(split_file_path, "r") as f:
            split["test"].extend(LearningProblem.from_dict(d) for d in json.load(f))
            split["test"] = sorted(split["test"], key=lambda p: p.id)
    return split


def _read_data_generation_settings_hash(path: Path) -> str | None:
    """
    Read the previously stored hash of the data generation settings, if it exists.
    Given path to the directory containing the data generation settings hash file.
    """
    hash_file_path = path / "data_generation_settings_hash.txt"
    if hash_file_path.exists():
        with open(hash_file_path, "r") as f:
            return f.read().strip()
    return None


def _hash_data_generation_settings(data_generation_settings: DataGenerationSettings) -> str:
    """Compute a hash of the data generation settings for reproducibility."""
    import hashlib
    import json

    settings_json = json.dumps(data_generation_settings.__dict__, sort_keys=True)
    return hashlib.sha256(settings_json.encode("utf-8")).hexdigest()


def _remove_trials(path: Path) -> None:
    """Remove embeddings that are not used in the benchmark run."""

    logger.info(
        "Proceeding to remove unused embedding files with keyword ``trial`` in %s", path
    )
    for dir in path.iterdir():
        if dir.is_dir() and "trial" in dir.name:
            for subfile in dir.iterdir():
                if subfile.is_file():
                    logger.info("Removing unused embedding file %s", subfile)
                    subfile.unlink()
            dir.rmdir()
    logger.info(
        "Completed removal of unused embedding files with keyword ``trial`` in %s", path
    )


def _convert_complexity_summary_to_dict(
        complexity_summary: dict[str, dict[str, MeanMetricsResult]]
    ) -> dict[str, dict[str, dict[str, float]]]:
    """Convert a complexity summary to a dictionary of dictionaries of dictionaries."""

    return {
        axis: {
            complexity_value: mean_metrics.to_dict()
            for complexity_value, mean_metrics in complexity_dict.items()
        }
        for axis, complexity_dict in complexity_summary.items()
    }


def _order_embedding_conditions(embedding_conditions: list[str]) -> None:
    """Ensure that 'random' is always the last embedding condition."""

    if "random" == embedding_conditions[0] and len(embedding_conditions) > 1:
        embedding_conditions.remove("random")
        embedding_conditions.append("random")
        logger.warning(
                    "Random embedding condition must follow after dice embedding" \
                    "conditions. Otherwise, this can lead to a situation where" \
                    "random and the dice embedding differ in dimensionality,"
                )
        logger.info(
            "Your error has been corrected. The embedding conditions have been " \
            "reordered to ensure 'random' is last."
        )


def _embedding_stage(
        paths: RunPaths,
        kb_path: Path,
        benchmark_settings: BenchmarkConfiguration,
        seed: int,
    ) -> tuple[dict[str, EmbeddingResultDice], int]:
    """
    Run the embedding stage and return the report.
    """

    report, m = build_embeddings(
            kb_path=kb_path,
            embeddings_dir=paths.embeddings_dir,
            data_dir=paths.embeddings_data_dir,
            embedding_settings=benchmark_settings.embedding,
            seed=seed,
            embedding_conditions=benchmark_settings.project.embedding_conditions,
            nces_embedding_dim=benchmark_settings.nces.embedding_dim,
        )
    return report, m


def _write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, default=str)
    logger.info("Wrote %s", path)


def _stage_parse_ontology(kb_path: Path, seed: int) -> OntologyParseResult:
    """Parse the ontology and return triples and individual IRIs."""

    triples = parse_triples(kb_path, seed=seed)
    knowledge_base = load_knowledge_base(kb_path)
    all_individuals = individual_iris(knowledge_base)
    return OntologyParseResult(
        knowledge_base=knowledge_base,
        triples=triples,
        all_individuals=all_individuals,
    )


def _stage_hardness_annotation(
        problems: Sequence[LearningProblem], 
        knowledge_base: KnowledgeBase, 
        all_individuals: list[str]
        ) -> HardnessAnnotationResult:
    """
    Annotate hardness for learning problems
    and return annotated problems and unparsed targets.
    """

    atomic_extensions = compute_atomic_class_extensions(knowledge_base)
    universe = frozenset(all_individuals)
    target_extensions: dict[str, frozenset[str]] = {}
    unparsed: list[str] = []
    annotated: list[LearningProblem] = []
    for problem in problems:
        extension = concept_extension(knowledge_base, problem.target_concept)
        if not extension:
            # Same fallback the evaluation stage uses. Sampled positives as
            # IRI strings. Must not be OWL objects - they never compare
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
    return HardnessAnnotationResult(
        annotated_problems=annotated,
        unparsed_problems=unparsed,
        target_extensions=target_extensions
    )


def _stage_train_eval_nces(
        split: dict[str, list[LearningProblem]],
        paths: RunPaths, 
        kb_path: Path,
        knowledge_base: KnowledgeBase, 
        all_individuals: list[str], 
        target_extensions: dict[str, frozenset[str]], 
        embedding_report: dict[str, EmbeddingResultDice], 
        m: int,
        config: BenchmarkConfiguration, seed: int
    ) -> SingleRunResult:
    """
    Train and evaluate NCES for each embedding condition.
    Writes NCES stats.
    """
    seed_everything(seed)
    train_data = prepare_nces_training_data(
        split["train"], paths.nces_data_dir / "nces_train_data.json", seed=seed
    )
    logger.info("Prepared NCES training data")
    conditions: dict[str, EmbeddingResult] = {}
    for condition in config.project.embedding_conditions:
        embeddings_file_path = paths.entity_embeddings_path(
            model_name=config.embedding.model_name, random=(condition == "random")
        )
        # location where the trained model will be saved
        # and parent directory where the evaluation will read from
        trained_model_path = paths.nces_suffix_dir(condition)
        embedding_dim = config.nces.embedding_dim if condition == "random" else m
        logger.info(
            "Starting NCES training for condition '%s'" \
            "with embeddings from '%s'",
            condition,
            embeddings_file_path,
        )
        training = train_nces(
            kb_path=kb_path,
            embeddings_path=Path(embeddings_file_path),
            trained_models_dir=trained_model_path,
            train_data=train_data,
            settings=config.nces,
            m=embedding_dim,
            seed=seed,
        )
        _write_json(
            payload=training.to_dict(),
            path=trained_model_path / f"nces_training_stats_{condition}.json",
        )
        logger.info(
            "Completed NCES training for condition '%s'. Stats written to '%s'",
            condition,
            trained_model_path / f"nces_training_stats_{condition}.json",
        )
        logger.info(
            "Starting NCES evaluation for condition '%s' with embeddings from '%s'",
            condition,
            embeddings_file_path,
        )
        evaluation = evaluate_nces(
            kb_path=kb_path,
            embeddings_path=Path(embeddings_file_path),
            trained_models_dir=trained_model_path,
            problems=split["test"],
            settings=config.nces,
            knowledge_base=knowledge_base,
            all_individuals=all_individuals,
            target_extensions=target_extensions,
            split_name="test",
            m=embedding_dim,
            trained_model_settings=embedding_report[condition].embedding_settings,
            seed=seed,
            degraded=training.degraded,
        )
        logger.info(
            "Completed NCES evaluation for condition '%s'.",
            condition,
        )
        conditions[condition] = evaluation
    return SingleRunResult(
        knowledge_base=kb_path.name,
        random_embedding_result=conditions.get("random", None),
        dice_embedding_result=conditions.get("dice", None),
    )
