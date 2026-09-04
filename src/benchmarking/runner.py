# src/benchmarking/runner.py
"""End-to-end benchmark orchestration across knowledge bases and seeds."""

import json
import logging
import tempfile
import time
from collections.abc import Sequence
from os import makedirs
from pathlib import Path
from typing import Any

import pandas as pd
from ontolearn.knowledge_base import KnowledgeBase

from src.config import BenchmarkConfiguration, DataGenerationSettings, _read_json
from src.data.complexity import annotate_hardness
from src.data.lp import (
    LearningProblem,
    generate_learning_problems,
    save_split,
    split_learning_problems,
)
from src.data.ontology import (
    Triple,
    compute_atomic_class_extensions,
    concept_extension,
    individual_iris,
    load_knowledge_base,
    local_name_collisions,
    parse_triples,
)
from src.data.results import (
    HardnessAnnotationResult,
    KnowledgeBaseFailure,
    KnowledgeBaseStats,
    LearningProblemPhaseResult,
    OntologyParseResult,
    OntologyPhaseResult,
    SingleRunResult,
)
from src.eval.pairing import observations_to_frame
from src.eval.reasoning import ExtensionOracle
from src.eval.rq2 import Trial, trials_to_frame
from src.eval.suite import analyse_suite
from src.logging_utils import configure_logging
from src.models.dice import (
    BuildEmbeddingResult,
    build_embeddings,
    count_partitions,
    split_dicee_dataset,
    stage_partition,
)
from src.models.hpo_search_utils import MRRNotFound
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
)
from src.random_utils import seed_everything
from src.writing_utils import write_json

logger = logging.getLogger(__name__)




def run_benchmark(
    config: BenchmarkConfiguration,
    *,
    knowledge_bases: Sequence[str] | None = None,
    seeds: Sequence[int] | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Run every (knowledge base, seed) combination and aggregate results."""

    started = time.perf_counter()
    selected_kbs = sorted(knowledge_bases or config.knowledge_bases)
    data_generation_settings = sorted(config.data_generation, key=lambda d: d.kb)
    selected_seeds = list(seeds or config.project.seeds)

    base = Path(tempfile.TemporaryDirectory().name)
    benchmark_dir = base / "benchmark"

    reports: dict[str, dict[int, SingleRunResult]] = {}
    failures: dict[str, KnowledgeBaseFailure] = {}

    for kb_name in selected_kbs:
        data_generation_setting = next(
            (d for d in data_generation_settings if d.kb == kb_name), None
        )
        if data_generation_setting is None:
            raise ValueError(f"No data generation settings found for knowledge base {kb_name}")
        for seed in selected_seeds:
            logger.info("=== %s | seed %d ===", kb_name, seed)
            seed_everything(seed)
            kb_path = resolve_knowledge_base(kb_name)
            paths = run_paths(
                str(benchmark_dir),
                seed,
                kb_name,
                output_dir=base,
            )
            paths.mkdirs()
            handler = configure_logging(paths.logs_dir / f"{kb_name}_{seed}.log")
            ontology_phase_result, learning_problem_phase_result = _generate_train_artifacts(
                config=config, 
                kb_path=kb_path, 
                kb_name=kb_name, 
                paths=paths,
                data_generation_setting=data_generation_setting,
            )
            try:
                report = run_single(
                    kb_name,
                    seed,
                    config,
                    kb_path=kb_path,
                    paths=paths,
                    ontology_phase_result=ontology_phase_result,
                    learning_problem_phase_result=learning_problem_phase_result,
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
            reports.setdefault(kb_name, {})[seed] = report
    try:
        all_observations = [
            observation
            for kb_reports in reports.values()
            for report in kb_reports.values()
            for observation in report.observations
        ]
        all_trials = [
            trial
            for kb_reports in reports.values()
            for report in kb_reports.values()
            for trial in report.trials
        ]
        all_quality_rows = [
            row
            for kb_reports in reports.values()
            for report in kb_reports.values()
            for row in report.quality
        ]
        frame = observations_to_frame(all_observations)
        analysis = analyse_suite(
            frame,
            quality=pd.DataFrame(all_quality_rows),
            trials=trials_to_frame(all_trials),
        )
        analysis.write(
            benchmark_dir / f"{config.project.benchmark_name}_suite_evaluation.json"
        )
        write_json(
            payload={"observations": frame.to_dict(orient="records")},
            path=benchmark_dir / f"{config.project.benchmark_name}_paired_observations.json",
        )
        try:
            run_substudy(config, analysis, output_dir=benchmark_dir)
        except Exception:
            # The confirmatory main-suite analysis above must survive a
            # sub-study failure; sub-study failures are reported, not fatal.
            logger.exception("RQ2 configuration sub-study failed; main suite results are unaffected.")
    finally:
        if handler is not None:
            logging.getLogger().removeHandler(handler)
            handler.close()
        summary = {
            "num_runs": len(reports),
            "failures": failures,
            "elapsed_time": round(time.perf_counter() - started, 3),
        }
        out=output_dir or OUTPUT_DIR
        write_json(payload=summary, path=benchmark_dir / "benchmark_summary.json")
        _copy_from_temp_dir(from_path=benchmark_dir, to_path=out)
        _clean_dir(path=out, make_zip=True)
        
    return summary

def _copy_from_temp_dir(from_path: Path, to_path: Path):
    import shutil
    shutil.copytree(from_path, to_path, dirs_exist_ok=True)


def run_substudy(
    config: BenchmarkConfiguration,
    analysis: Any,
    *,
    output_dir: Path | None = None,
) -> dict[str, Any] | None:
    """Run the RQ2 configuration sub-study on the pre-selected (kb, architecture).

    Four configurations drawn from that architecture's own trial record
    (best/worst/median validation MRR, plus one deliberately extreme learning
    rate) are each trained, exported, and passed to NCES under the same
    benchmark seeds as the main suite, with every other factor unchanged.
    Observations are persisted for manual/future analysis; this is a
    four-arm factorial comparison, not a paired contrast against ``random``.
    """
    from dataclasses import replace

    from src.eval.pairing import Observation
    from src.models.dice import (
        export_entity_embeddings,
        get_csv_dimension,
        train_embedding_model,
    )

    selection = analysis.hyperparameters.get("substudy_selection")
    configurations = analysis.hyperparameters.get("substudy_configurations")
    if not selection or not configurations:
        logger.info(
            "No sub-study selection available; skipping the configuration sub-study."
        )
        return None

    knowledge_base_name = selection["knowledge_base"]
    architecture = selection["condition"]
    data_generation_setting = next(
        (d for d in config.data_generation if d.kb == knowledge_base_name), None
    )
    if data_generation_setting is None:
        raise ValueError(f"No data generation settings for {knowledge_base_name!r}.")

    base = Path(tempfile.TemporaryDirectory().name)
    substudy_dir = base / "substudy"
    observations: list[Observation] = []

    for seed in config.project.seeds:
        seed_everything(seed)
        kb_path = resolve_knowledge_base(knowledge_base_name)
        paths = run_paths(str(substudy_dir), seed, knowledge_base_name, output_dir=base)
        paths.mkdirs()
        ontology_phase_result, learning_problem_phase_result = _generate_train_artifacts(
            config=config,
            kb_path=kb_path,
            kb_name=knowledge_base_name,
            paths=paths,
            data_generation_setting=data_generation_setting,
        )
        ontology_parse_result = ontology_phase_result.ontology_parse_result
        train_data = prepare_nces_training_data(
            learning_problem_phase_result.split["train"],
            paths.nces_data_dir / "nces_train_data.json",
            seed=seed,
        )
        with ExtensionOracle(
            ontology_path=kb_path,
            cache_path=paths.ontology_parse_data_dir / "extension_cache.json",
        ) as oracle:
            for spec in configurations:
                condition = f"{architecture}__{spec['label']}"
                settings = replace(
                    config.embedding,
                    model_name=architecture,
                    batch_size=int(spec["batch_size"]),
                    learning_rate=float(spec["learning_rate"]),
                    epochs=int(spec["epochs"]),
                )
                run_dir = paths.embeddings_dir / condition / "run"
                seed_everything(seed)
                train_embedding_model(
                    paths.embeddings_data_dir, run_dir, settings, seed=seed
                )
                embeddings_path = paths.entity_embeddings_path(condition)
                export_entity_embeddings(run_dir, embeddings_path)
                m = get_csv_dimension(embeddings_path)
                trained_model_path = paths.nces_suffix_dir(condition)
                training = train_nces(
                    kb_path=kb_path,
                    embeddings_path=embeddings_path,
                    trained_models_dir=trained_model_path,
                    train_data=train_data,
                    settings=config.nces,
                    m=m,
                    seed=seed,
                )
                evaluation = evaluate_nces(
                    kb_path=kb_path,
                    embeddings_path=embeddings_path,
                    trained_models_dir=trained_model_path,
                    problems=learning_problem_phase_result.split["test"],
                    settings=config.nces,
                    oracle=oracle,
                    all_individuals=ontology_parse_result.all_individuals,
                    target_extensions=learning_problem_phase_result.target_extensions,
                    split_name="test",
                    condition=condition,
                    knowledge_base_name=knowledge_base_name,
                    m=m,
                    seed=seed,
                    degraded=training.degraded,
                )
                observations.extend(evaluation.observations)
        _clean_dir(path=paths.seed_dir, make_zip=False)

    frame = observations_to_frame(observations)
    out = output_dir or OUTPUT_DIR
    write_json(
        payload={
            "selection": selection,
            "configurations": configurations,
            "observations": frame.to_dict(orient="records"),
        },
        path=out / f"{config.project.benchmark_name}_substudy_observations.json",
    )
    logger.info(
        "Completed RQ2 configuration sub-study for %s/%s: %d observations",
        knowledge_base_name,
        architecture,
        len(observations),
    )
    return {
        "knowledge_base": knowledge_base_name,
        "condition": architecture,
        "n_observations": len(observations),
    }


def _generate_train_artifacts(
        config: BenchmarkConfiguration,
        data_generation_setting: DataGenerationSettings,
        kb_path: Path, 
        kb_name: str, 
        paths: RunPaths
    ):
    current_data_settings_hash: str = _hash_data_generation_settings(
        data_generation_settings=data_generation_setting
    )
    old_data_settings_hash: str | None = _read_data_generation_settings_hash(
        paths.nces_data_dir
    )
    has_no_ontology_parse_result: bool = old_data_settings_hash is None
    ontology_phase_result = _ontology_phase(
        has_no_ontology_parse_result=has_no_ontology_parse_result,
        kb_path=kb_path,
        knowledge_base_name=kb_name,
        paths=paths,
    )
    has_no_learning_problem_results: bool = (
        has_no_ontology_parse_result 
        or current_data_settings_hash != old_data_settings_hash
    )
    ontology_parse_result = ontology_phase_result.ontology_parse_result
    learning_problem_phase_result = _gather_learning_problems_phase(
        paths=paths,
        kb_path=kb_path,
        ontology_parse_result=ontology_parse_result,
        config=config,
        knowledge_base_name=kb_name,
        has_no_learning_problem_results=has_no_learning_problem_results,
        current_data_settings_hash=current_data_settings_hash,
        data_generation_setting=data_generation_setting,
    )
    return ontology_phase_result, learning_problem_phase_result


def _clean_dir(path: Path, make_zip: bool = False):
    import shutil
    if make_zip:
        logger.info("Proceeding to zip the directory %s", path)
        shutil.make_archive(str(path), 'zip', str(path))
        logger.info("Completed zipping the directory %s", path)
    logger.info("Proceeding to remove the directory %s. Excluding the zip archive.", path)
    shutil.rmtree(path)
    logger.info("Completed removal of the directory %s. Excluding the zip archive.", path)


def run_single(
    knowledge_base_name: str,
    seed: int,
    config: BenchmarkConfiguration,
    kb_path: Path,
    paths: RunPaths,
    ontology_phase_result: OntologyPhaseResult,
    learning_problem_phase_result: LearningProblemPhaseResult,
) -> SingleRunResult:
    """Execute one benchmark run: one (knowledge base, seed) pair."""
    started = time.perf_counter()
    ontology_parse_result: OntologyParseResult = ontology_phase_result.ontology_parse_result
    counts: dict[str, int] = ontology_phase_result.counts
    logger.info("Starting Stage 4: Embedding generation.")
    build_embedding_result = _embedding_stage(
        paths=paths,
        benchmark_settings=config,
        seed=seed,
        triples=ontology_parse_result.triples,
        counts=counts,
    )
    logger.info(
        "Completed Stage 4: Embedding generation for" \
        "all conditions and collected results."
    )
    split = learning_problem_phase_result.split
    target_extensions = learning_problem_phase_result.target_extensions
    with ExtensionOracle(
        ontology_path=kb_path,
        cache_path=paths.ontology_parse_data_dir / "extension_cache.json",
    ) as oracle:
        single_run_result = _stage_train_eval_nces(
            split=split,
            paths=paths,
            kb_path=kb_path,
            knowledge_base_name=knowledge_base_name,
            all_individuals=ontology_parse_result.all_individuals,
            target_extensions=target_extensions,
            build_embedding_result=build_embedding_result,
            config=config,
            seed=seed,
            oracle=oracle,
        )
    logger.info(
        "Completed Stage 5: NCES training and evaluation for all conditions."
    )
    logger.info(
        "Wrote single-run result to %s", 
        paths.nces_results_dir / "single_run_result.json"
    )
    atomic_extensions = compute_atomic_class_extensions(
        knowledge_base=ontology_parse_result.knowledge_base
    )
    entity_vocabulary = {
        triple.subject for triple in ontology_parse_result.triples
    } | {triple.object for triple in ontology_parse_result.triples}
    collisions = local_name_collisions(entity_vocabulary)
    if collisions:
        logger.warning(
            "%s: %d local name(s) collide across %d distinct IRIs; all "
            "colliding IRIs are dropped from the entity vocabulary.",
            knowledge_base_name,
            len(collisions),
            sum(len(members) for members in collisions.values()),
        )
    knowledge_base_stats = KnowledgeBaseStats(
        knowledge_base_name=knowledge_base_name,
        number_of_individuals=len(ontology_parse_result.all_individuals),
        number_of_triples=len(ontology_parse_result.triples),
        number_of_atomic_classes=len(atomic_extensions),
        number_of_colliding_local_names=len(collisions),
        number_of_entities_dropped_to_collisions=sum(
            len(members) for members in collisions.values()
        ),
    )
    write_json(
        payload=knowledge_base_stats,
        path=paths.seed_dir / f"knowledge_base_stats_{seed}.json",
    )
    logger.info(
        "Wrote knowledge-base stats to %s",
        paths.seed_dir / f"knowledge_base_stats_{seed}.json",
    )
    single_run_result.set_runtime(round(time.perf_counter() - started, 3))
    write_json(
        payload=single_run_result,
        path=paths.nces_results_dir / "single_run_result.json",
    )
    _remove_trials(path=paths.embeddings_dir)
    _clean_dir(path=paths.seed_dir, make_zip=True)
    return single_run_result 


def _gather_learning_problems_phase(
    paths: RunPaths, 
    has_no_learning_problem_results: bool,
    current_data_settings_hash: str, 
    kb_path: Path, knowledge_base_name: str, 
    ontology_parse_result: OntologyParseResult, 
    config: BenchmarkConfiguration,
    data_generation_setting: DataGenerationSettings,
    ) -> LearningProblemPhaseResult:
    
    split: dict[str, list[LearningProblem]]
    unparsed: list[str] = []
    target_extensions: dict[str, frozenset[str]] = {}

    hash_file_path = paths.nces_data_dir / "data_generation_settings_hash.txt"

    if not has_no_learning_problem_results:
        logger.info(
            "Data generation settings have not changed." \
            "Using cached learning problems and splits."
        )
        split = _load_split(paths.nces_data_dir)
        raw = _read_json(paths.nces_data_dir / "target_extensions.json")
        target_extensions = {
            key: frozenset(value) for key, value in raw.items()
        }
        with open(paths.nces_data_dir / "unparsed.json", "r") as f:
            unparsed = json.load(f)
        logger.info(
            "Loaded cached learning problems and splits."
        )
    else:
        logger.warning(
            f"Current data generation settings do not reflect the previous settings."
            f"Updating hash file at {hash_file_path}.")
        _clean_dir(path=paths.nces_data_dir, make_zip=False)
        makedirs(paths.nces_data_dir, exist_ok=True)
        with open(hash_file_path, "x", encoding="utf-8") as f:
            f.write(current_data_settings_hash)
        problems = generate_learning_problems(
            kb_path,
            paths.nces_data_dir,
            data_generation_setting,
            seed=current_data_settings_hash,
        )
        logger.info(
            "Completed Stage 2.1: Learning-problem generation for %d problems.",
            len(problems)
        )
        if not problems:
            raise RuntimeError(
                f"No non-degenerate learning problems generated for {knowledge_base_name}."
            )
        split = split_learning_problems(
            problems,
            stratify_by=config.project.stratify_by,
            seed=current_data_settings_hash,
        )
        len_split_test = len(split["test"])
        len_split_train = len(split["train"])
        logger.info(
            "Split %d learning problems into %d train / %d test",
            len_split_train + len_split_test,
            len_split_train,
            len_split_test,
        )
        logger.info("Completed Stage 2.2: Learning-problem splitting.")
        split, target_extensions, unparsed = _annotate_split(
            split, ontology_parse_result, len(problems)
        )
        save_split(split, paths.nces_data_dir)
        write_json(
            payload={k: sorted(v) for k, v in target_extensions.items()},
            path=paths.nces_data_dir / "target_extensions.json",
        )
        with open(paths.nces_data_dir / "unparsed.json", "w") as f:
            json.dump(unparsed, f, indent=4)
    return LearningProblemPhaseResult(
        target_extensions=target_extensions,
        split=split,
        unparsed=unparsed,
    )


def _annotate_split(
        split: dict[str, list[LearningProblem]], 
        ontology_parse_result: OntologyParseResult, 
        len_problems: int
    ) -> tuple[dict[str, list[LearningProblem]], dict[str, frozenset[str]], list[str]]:
    knowledge_base = ontology_parse_result.knowledge_base
    unparsed: list[str] = []
    target_extensions: dict[str, frozenset[str]] = {}
    for key, value in split.items():
        logger.info(f"Annotating hardness for `{key}` split of size {len(value)}")
        annotation_result = _stage_hardness_annotation(
            value, knowledge_base, ontology_parse_result.all_individuals
        )
        split[key] = annotation_result.annotated_problems
        unparsed.extend(annotation_result.unparsed_problems)
        target_extensions.update(annotation_result.target_extensions)
    logger.info("Completed Stage 2.3: Hardness annotation for splits")
    if unparsed:
        logger.warning(
            "%d of %d target concepts could not be parsed; used sampled "
            "positives as their extension (ids: %s%s)",
            len(unparsed),
            len_problems,
            ", ".join(unparsed[:5]),
            ", ..." if len(unparsed) > 5 else "",
        )
    return split, target_extensions, unparsed


def _ontology_phase(
        has_no_ontology_parse_result: bool | None, 
        kb_path: Path, 
        knowledge_base_name: str, 
        paths: RunPaths
    ) -> OntologyPhaseResult:
    # if old data is not none we can assume that the ontology has already been parsed
    if has_no_ontology_parse_result:
        logger.info("No previous ontology parse results found")
        ontology_parse_result = _parse_ontology(kb_path)
        write_json(
            payload=ontology_parse_result,
            path=paths.ontology_parse_data_dir / "ontology_parse_result.json"
        )
        split_triple = split_dicee_dataset(
            output_dir=paths.embeddings_data_dir,
            triples=ontology_parse_result.triples
        )
        stage_partition(
            output_dir=paths.embeddings_data_dir,
            partitions=split_triple
            )
        logger.info("Completed Stage 1: Ontology parsing.")
    else:
        ontology_parse_result = OntologyParseResult.from_dict(
            _read_json(paths.ontology_parse_data_dir / "ontology_parse_result.json"),
            knowledge_base=load_knowledge_base(resolve_knowledge_base(knowledge_base_name))
        )
        split_triple = split_dicee_dataset(
            output_dir=paths.embeddings_data_dir,
            triples=ontology_parse_result.triples
        )
    counts = count_partitions(
        partitions=split_triple
    )
    return OntologyPhaseResult(
        ontology_parse_result=ontology_parse_result,
        counts=counts
    )


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
    """Remove embeddings that are not used in the benchmark run.

    Each condition trains in its own subdirectory (``path/<condition>``), so
    trial artifacts are one level deeper than they used to be.
    """

    logger.info(
        "Proceeding to remove unused embedding files with keyword ``trial`` in %s", path
    )
    for condition_dir in path.iterdir():
        if not condition_dir.is_dir():
            continue
        for dir in condition_dir.iterdir():
            if dir.is_dir() and "trial" in dir.name:
                for subfile in dir.iterdir():
                    if subfile.is_file():
                        logger.info("Removing unused embedding file %s", subfile)
                        subfile.unlink()
                dir.rmdir()
    logger.info(
        "Completed removal of unused embedding files with keyword ``trial`` in %s", path
    )


def _embedding_stage(
        paths: RunPaths,
        benchmark_settings: BenchmarkConfiguration,
        seed: int,
        triples: list[Triple],
        counts: dict[str, int],
    ) -> BuildEmbeddingResult:
    """
    Run the embedding stage and return the report.
    """

    return build_embeddings(
            embeddings_dir=paths.embeddings_dir,
            data_dir=paths.embeddings_data_dir,
            embedding_settings=benchmark_settings.embedding,
            seed=seed,
            nces_embedding_dim=benchmark_settings.nces.embedding_dim,
            triples=triples,
            counts=counts,
            conditions=benchmark_settings.project.embedding_conditions,
    )


def _parse_ontology(kb_path: Path) -> OntologyParseResult:
    """Parse the ontology and return triples and individual IRIs."""

    triples = parse_triples(kb_path)
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


def _dice_trials_to_eval_trials(
    raw_trials: list[dict[str, Any]],
    *,
    condition: str,
    knowledge_base: str,
    seed: int,
) -> list[Trial]:
    """Convert one architecture's raw HPO trial records into RQ2's schema."""
    converted: list[Trial] = []
    for index, record in enumerate(raw_trials):
        error = record.get("error")
        score = record.get("score")
        converted.append(
            Trial(
                condition=condition,
                knowledge_base=knowledge_base,
                seed=seed,
                trial_index=int(record.get("trial", index)),
                configuration={
                    "batch_size": record.get("batch_size"),
                    "learning_rate": record.get("learning_rate"),
                    "epochs": record.get("epochs"),
                },
                score=None if score is None else float(score),
                used_validation_fallback=(
                    record.get("validation_error")
                    == MRRNotFound.ValidationUnavailable.value
                ),
                failed=error is not None or score is None,
                error=error,
            )
        )
    return converted


def _ranking_metrics_from_report(metrics: dict[str, Any]) -> dict[str, float | None]:
    """Validation MRR/Hits@k, falling back to test — same order as selection."""
    for section in ("Val", "Valid", "Test"):
        block = metrics.get(section)
        if isinstance(block, dict) and "MRR" in block:
            return {
                "mrr": block.get("MRR"),
                "hits_at_1": block.get("H@1"),
                "hits_at_3": block.get("H@3"),
                "hits_at_10": block.get("H@10"),
            }
    return {"mrr": None, "hits_at_1": None, "hits_at_3": None, "hits_at_10": None}


def _stage_train_eval_nces(
        split: dict[str, list[LearningProblem]],
        paths: RunPaths, 
        kb_path: Path,
        knowledge_base_name: str,
        all_individuals: list[str], 
        target_extensions: dict[str, frozenset[str]], 
        build_embedding_result: BuildEmbeddingResult,
        config: BenchmarkConfiguration, seed: int,
        oracle: ExtensionOracle,
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
    result = SingleRunResult(knowledge_base=knowledge_base_name, seed=seed)
    for condition in config.project.embedding_conditions:
        embeddings_file_path = paths.entity_embeddings_path(condition)
        # location where the trained model will be saved
        # and parent directory where the evaluation will read from
        trained_model_path = paths.nces_suffix_dir(condition)
        m = build_embedding_result.model_csv_dim[condition]
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
            m=m,
            seed=seed,
        )
        write_json(
            payload=training,
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
            oracle=oracle,
            all_individuals=all_individuals,
            target_extensions=target_extensions,
            split_name="test",
            condition=condition,
            knowledge_base_name=knowledge_base_name,
            m=m,
            seed=seed,
            degraded=training.degraded,
        )
        logger.info(
            "Completed NCES evaluation for condition '%s'.",
            condition,
        )
        result.observations.extend(evaluation.observations)
        result.condition_stats[condition] = evaluation.nces_stats

        if condition != "random":
            dice_result = build_embedding_result.results[condition]
            result.trials.extend(
                _dice_trials_to_eval_trials(
                    dice_result.search_trials,
                    condition=condition,
                    knowledge_base=knowledge_base_name,
                    seed=seed,
                )
            )
            abl_values = [
                v for o in evaluation.observations
                if (v := o.values.get("abl")) is not None
            ]
            result.quality.append(
                {
                    "condition": condition,
                    "knowledge_base": knowledge_base_name,
                    "seed": seed,
                    **_ranking_metrics_from_report(dice_result.metrics),
                    "mean_abl": (
                        sum(abl_values) / len(abl_values) if abl_values else None
                    ),
                }
            )
    return result