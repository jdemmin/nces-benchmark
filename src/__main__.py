# src/__main__.py
"""``nces-benchmark`` command-line entry point."""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import replace
from pathlib import Path

from src.config import DICE_MODELS, EMBEDDING_CONDITIONS, BenchmarkConfiguration
from src.logging_utils import configure_logging
from src.paths import INPUT_DIR, OUTPUT_DIR


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nces-benchmark",
        description="Benchmark NCES concept learning with DICE embeddings.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=INPUT_DIR,
        help="Directory holding the settings JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory for logs, learning problems, embeddings, and reports.",
    )
    parser.add_argument(
        "--benchmark-name",
        default=None,
        help="Benchmark name; overrides project_settings.json.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Knowledge-base names to evaluate.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help="Random seeds for splitting, DICE training, and NCES evaluation.",
    )
    parser.add_argument(
        "--embedding-conditions",
        nargs="+",
        choices=list(EMBEDDING_CONDITIONS),
        default=None,
        help="Embedding conditions to compare.",
    )
    parser.add_argument(
        "--embedding-model",
        choices=list(DICE_MODELS),
        default=None,
        help="DICE architecture; overrides embedding_settings.json.",
    )
    parser.add_argument(
        "--num-problems",
        type=int,
        default=None,
        help="Number of learning problems to generate per knowledge base.",
    )
    parser.add_argument(
        "--dice-epochs", type=int, default=None, help="DICE training epochs."
    )
    parser.add_argument(
        "--nces-epochs", type=int, default=None, help="NCES training epochs."
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(level=getattr(logging, args.log_level))

    config = BenchmarkConfiguration.load(args.input_dir)

    project = config.project
    if args.benchmark_name:
        project = replace(project, benchmark_name=args.benchmark_name)
    if args.seeds:
        project = replace(project, seeds=args.seeds)
    if args.embedding_conditions:
        project = replace(project, embedding_conditions=args.embedding_conditions)

    data_generation = config.data_generation
    if args.num_problems:
        data_generation = replace(
            data_generation, num_rand_samples=args.num_problems
        )

    embedding = config.embedding
    if args.embedding_model:
        embedding = replace(embedding, model_name=args.embedding_model)
    if args.dice_epochs:
        embedding = replace(embedding, epochs=args.dice_epochs)

    nces = config.nces
    if args.nces_epochs:
        nces = replace(nces, epochs=args.nces_epochs)

    config = replace(
        config,
        project=project,
        data_generation=data_generation,
        embedding=embedding,
        nces=nces,
    )

    # Imported here so that --help stays fast and dependency-free.
    from src.benchmarking.runner import run_benchmark

    summary = run_benchmark(
        config,
        knowledge_bases=args.datasets,
        seeds=args.seeds,
        output_dir=args.output_dir,
    )

    print(
        f"\nCompleted {summary['num_runs']} benchmark run(s); "
        f"{len(summary['failures'])} failure(s)."
    )
    return 1 if summary["failures"] and summary["num_runs"] == 0 else 0


if __name__ == "__main__":
    sys.exit(main())
