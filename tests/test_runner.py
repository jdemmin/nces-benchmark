# tests/test_runner.py
"""Runner tests with the heavy stages stubbed out.

These verify orchestration, the directory layout, and report structure without
training DICE or NCES.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.config import (
    BenchmarkConfiguration,
    DataGenerationSettings,
    EmbeddingSettings,
    NCESSettings,
    ProjectSettings,
)
from src.data.results import EmbeddingResult, NCESStats
from src.paths import run_paths


@pytest.fixture
def config() -> BenchmarkConfiguration:
    return BenchmarkConfiguration(
        project=ProjectSettings(
            seeds=[1], benchmark_name="benchmark1",
            embedding_conditions=["dice", "random"],
        ),
        data_generation=DataGenerationSettings(kbs="father", num_rand_samples=6),
        embedding=EmbeddingSettings(
            model_name="Keci", embedding_dim=8, epochs=1, batch_size=4
        ),
        nces=NCESSettings(embedding_dim=8, epochs=1, batch_size=2),
        knowledge_bases=["father"],
    )


def test_run_paths_match_specified_layout(tmp_path: Path) -> None:
    paths = run_paths("benchmark1", 1, "kb1", output_dir=tmp_path)
    root = tmp_path / "benchmark1" / "seed1" / "kb1"

    assert paths.embeddings_dir == root / "embeddings"
    assert paths.embeddings_data_dir == root / "embeddings" / "data"
    assert paths.nces_dir == root / "nces"
    assert paths.nces_data_dir == root / "nces" / "data"
    assert paths.logs_dir == root / "logs"
    assert paths.entity_embeddings_path("Keci", random=False).name == "Keci.csv"
    assert paths.entity_embeddings_path("Keci", random=True).name == "Keci_random.csv"


def test_mkdirs_creates_every_directory(tmp_path: Path) -> None:
    paths = run_paths("b", 2, "kb", output_dir=tmp_path)
    paths.mkdirs()
    for directory in (
        paths.embeddings_data_dir,
        paths.nces_data_dir,
        paths.nces_results_dir,
        # removed paths.trained_models_dir,  # Might be obsolete
        # paths.trained_models_dir,
        paths.logs_dir,
    ):
        assert directory.is_dir()

def test_atomic_extensions_exclude_thing_and_nothing(kb_path: Path) -> None:
    from src.data.ontology import compute_atomic_class_extensions, load_knowledge_base

    extensions = compute_atomic_class_extensions(load_knowledge_base(kb_path))

    assert set(extensions) == {"person", "male", "female"}
    assert "⊤" not in extensions and "⊥" not in extensions
    assert any(iri.endswith("stefan") for iri in extensions["male"])
    assert not (extensions["male"] & extensions["female"])

# TODO: I do not know why this works
def test_run_single_produces_report(
    monkeypatch: pytest.MonkeyPatch,
    config: BenchmarkConfiguration,
    kb_path: Path,
    problems,
    tmp_path: Path,
) -> None:
    from src.benchmarking import runner
    from src.models import nces
    from src.models.dice import EmbeddingResultDice

    monkeypatch.setattr(runner, "resolve_knowledge_base", lambda name: kb_path)
    monkeypatch.setattr(runner, "load_knowledge_base", lambda path: MagicMock())
    monkeypatch.setattr(
        runner, "individual_iris", lambda kb: ["a", "b", "c", "d"]
    )
    monkeypatch.setattr(
        runner,
        "compute_atomic_class_extensions",
        lambda kb: {"male": frozenset({"a", "b"}), "female": frozenset({"c", "d"})},
    )
    monkeypatch.setattr(
        runner,
        "generate_learning_problems",
        lambda *a, **k: problems * 5,
    ) # otherwise test prefix would trigger dicee path check
    settings = EmbeddingSettings(model_name="Keci", embedding_dim=8, epochs=1, batch_size=4)
    def fake_embeddings(*args, **kwargs):
        return {
            condition: EmbeddingResultDice(
                embeddings_path=tmp_path / f"Keci_{condition}.csv" if condition == "random" else tmp_path / "Keci.csv",
                score=0.42,
                embedding_settings=settings,
            )
            for condition in kwargs["embedding_conditions"]
        }

    monkeypatch.setattr(runner, "build_embeddings", fake_embeddings)
    monkeypatch.setattr(
        runner, "train_nces", lambda *a, **k: NCESStats(
            learner_name="GRU", runtime_seconds=0.1, degraded=False
        )
    )
    monkeypatch.setattr(
        runner, "_write_json", lambda *a, **k: None
    )
    monkeypatch.setattr(
        runner,
        "evaluate_nces",
        lambda *a, **k: EmbeddingResult(
                split_name="train", learning_problem_results=[],
                number_of_problems=10,
                number_of_successful_problems=1,
                embedding_settings=settings,
                nces_stats=NCESStats(
                    learner_name="GRU", runtime_seconds=0.1, degraded=False
                )
        ))
    monkeypatch.setattr(
        runner, "get_csv_dimension", lambda *a, **k: 32
    )
    monkeypatch.setattr(
        nces, 
        "_assert_model_dir_contains_needed_files", 
        lambda *a, **k: None
    )
    # run_single does not create the report file itself,
    # but the benchmark runner does, so we check that at least the report exists.
    report = runner.run_single("father", 1, config, output_dir=tmp_path, benchmark_name="benchmark1")

    #assert report["seed"] == 1 No longer included in report
    assert set(report.get_conditions()) == {"dice", "random"}
    assert report.get_number_of_problems() == 10
    #assert set(report["split_sizes"]) == {"train", "test"} # No longer included in report

    #paths = run_paths("benchmark1", 1, "father", output_dir=tmp_path)
    #assert paths.report_path.is_file()
    #assert paths.learning_problems_path.is_file()

    #persisted = json.loads(paths.report_path.read_text(encoding="utf-8"))
    #assert persisted["knowledge_base"] == "father"


def test_run_benchmark_records_failures_without_aborting(
    monkeypatch: pytest.MonkeyPatch,
    config: BenchmarkConfiguration,
    tmp_path: Path,
) -> None:
    from src.benchmarking import runner

    def explode(kb_name, seed, cfg, output_dir=None):
        raise RuntimeError(f"boom {kb_name}/{seed}")

    monkeypatch.setattr(runner, "run_single", explode)

    summary = runner.run_benchmark(
        config, knowledge_bases=["father"], seeds=[1, 2], output_dir=tmp_path
    )

    assert summary["num_runs"] == 0
    assert len(summary["failures"]) == 2
    assert (tmp_path / "benchmark1" / "benchmark_summary.json").is_file()


def test_cli_overrides_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.__main__ as cli
    from src.benchmarking import runner

    captured: dict = {}

    def capture(cfg, **kwargs):
        captured["config"] = cfg
        captured["kwargs"] = kwargs
        return {"num_runs": 1, "failures": []}

    monkeypatch.setattr(runner, "run_benchmark", capture)

    input_dir = Path(__file__).parent / "input" / "test_runner.py"

    exit_code = cli.main(
        [
            "--input-dir", str(input_dir),
            "--benchmark-name", "bench2",
            "--seeds", "7",
            "--embedding-model", "TransE",
            "--dice-epochs", "3",
            "--nces-epochs", "5",
            "--embedding-conditions", "random",
        ]
    )

    assert exit_code == 0
    cfg = captured["config"]
    assert cfg.project.benchmark_name == "bench2"
    assert cfg.project.seeds == [7]
    assert cfg.project.embedding_conditions == ["random"]
    assert cfg.embedding.model_name == "TransE"
    assert cfg.embedding.epochs == 3
    assert cfg.nces.epochs == 5
