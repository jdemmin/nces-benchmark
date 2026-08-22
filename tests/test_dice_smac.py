# tests/test_dice_smac.py
"""Tests for the SMAC-driven DICE hyperparameter search.

These exercise run_smac_search against synthetic objectives so no dicee
training, torch, or knowledge base is required.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import pytest

from src.config import EmbeddingSearchSpace, EmbeddingSettings
from src.models.dice_smac import (
    CRASH_COST,
    MAX_CONSECUTIVE_UNSCORED,
    SearchAborted,
    build_configuration_space,
    run_smac_search,
    settings_from_configuration,
)
from src.models.hpo_search_utils import MRRNotFound

smac = pytest.importorskip("smac", reason="SMAC3 is required for the HPO tests")


def _settings(**overrides: Any) -> EmbeddingSettings:
    base = EmbeddingSettings(
        model_name="Keci",
        embedding_dim=64,
        batch_size=64,
        epochs=5,
        learning_rate=0.1,
        hpo_backend="smac",
        n_trials=8,
        search_space=EmbeddingSearchSpace(
            embedding_dim_choices=(32, 64, 128),
            batch_size_choices=(32, 64, 128),
            learning_rate_bounds=(1e-3, 3e-1),
        ),
    )
    return base.with_overrides(**overrides)


def _mrr_report(mrr: float, *, section: str = "Val") -> dict[str, Any]:
    return {
        "Train": {"MRR": 0.99},
        section: {"MRR": mrr},
        "Test": {"MRR": mrr},
    }


def _selection_score(report: dict[str, Any]) -> tuple[float | None, str | None]:
    """Mirror of src.models.dice._selection_score, kept local on purpose.

    Importing the real function would drag in dicee at collection time.
    """
    for key in ("Val", "Valid", "Validation"):
        section = report.get(key)
        if isinstance(section, dict) and "MRR" in section:
            return float(section["MRR"]), None
    test = report.get("Test")
    if isinstance(test, dict) and "MRR" in test:
        return float(test["MRR"]), MRRNotFound.ValidationUnavailable.value
    train = report.get("Train")
    if isinstance(train, dict) and "MRR" in train:
        return None, MRRNotFound.TrainOnly.value
    return None, MRRNotFound.NoMRR.value



# --- Configuration space --------------------------------------------------


def test_configuration_space_contains_expected_hyperparameters() -> None:
    space = build_configuration_space(_settings(), seed=1)
    assert set(space.keys()) == {
        "embedding_dim",
        "batch_size",
        "learning_rate",
    }


def test_configuration_space_tunes_optional_hyperparameters() -> None:
    settings = _settings(
        search_space=EmbeddingSearchSpace(
            tune_epochs=True, tune_scoring_technique=True
        )
    )
    space = build_configuration_space(settings, seed=1)
    assert "epochs" in space
    assert "scoring_technique" in space


def test_default_configuration_snaps_onto_the_categorical_grid() -> None:
    # 100 is not a listed choice; it must snap to the nearest one (128).
    settings = _settings(embedding_dim=100)
    space = build_configuration_space(settings, seed=1)
    assert space.get_default_configuration()["embedding_dim"] == 128


def test_out_of_bounds_learning_rate_is_clamped_not_rejected() -> None:
    settings = _settings(learning_rate=5.0)
    space = build_configuration_space(settings, seed=1)
    assert space.get_default_configuration()["learning_rate"] == pytest.approx(
        0.3
    )


def test_settings_from_configuration_casts_types() -> None:
    settings = _settings()
    space = build_configuration_space(settings, seed=1)
    config = space.sample_configuration()
    result = settings_from_configuration(settings, config)
    assert isinstance(result.embedding_dim, int)
    assert isinstance(result.batch_size, int)
    assert isinstance(result.learning_rate, float)
    # Untuned fields must be inherited verbatim.
    assert result.model_name == settings.model_name
    assert result.scoring_technique == settings.scoring_technique



# --- Good path: the search must find the optimum of a synthetic objective



def test_search_finds_the_best_configuration(tmp_path: Path) -> None:
    """A synthetic objective peaks at dim=128; SMAC must select it."""
    calls: list[EmbeddingSettings] = []

    def train_fn(
        dataset_dir: Path, run_dir: Path, settings: EmbeddingSettings
    ) -> dict[str, Any]:
        calls.append(settings)
        run_dir.mkdir(parents=True, exist_ok=True)
        # Deterministic, unimodal in embedding_dim.
        mrr = {32: 0.10, 64: 0.30, 128: 0.70}[settings.embedding_dim]
        return _mrr_report(mrr)

    outcome = run_smac_search(
        tmp_path / "kg",
        tmp_path / "emb",
        _settings(n_trials=12),
        seed=1,
        train_fn=train_fn,
        score_fn=_selection_score,
    )

    assert outcome.best_settings.embedding_dim == 128
    assert outcome.incumbent_cost == pytest.approx(0.30)
    assert outcome.best_report["Val"]["MRR"] == pytest.approx(0.70)
    assert len(calls) == len(outcome.trials)


def test_best_run_dir_points_at_the_incumbent(tmp_path: Path) -> None:
    def train_fn(
        dataset_dir: Path, run_dir: Path, settings: EmbeddingSettings
    ) -> dict[str, Any]:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "marker.txt").write_text(str(settings.embedding_dim))
        return _mrr_report({32: 0.1, 64: 0.2, 128: 0.9}[settings.embedding_dim])

    outcome = run_smac_search(
        tmp_path / "kg",
        tmp_path / "emb",
        _settings(n_trials=10),
        seed=3,
        train_fn=train_fn,
        score_fn=_selection_score,
    )

    assert outcome.best_run_dir.is_dir()
    assert (outcome.best_run_dir / "marker.txt").read_text() == "128"


def test_default_configuration_is_evaluated(tmp_path: Path) -> None:
    """use_default_config must guarantee the hand-tuned config is tried."""
    seen: set[tuple[int, int]] = set()

    def train_fn(
        dataset_dir: Path, run_dir: Path, settings: EmbeddingSettings
    ) -> dict[str, Any]:
        seen.add((settings.embedding_dim, settings.batch_size))
        return _mrr_report(0.5)

    run_smac_search(
        tmp_path / "kg",
        tmp_path / "emb",
        _settings(n_trials=6, embedding_dim=64, batch_size=64),
        seed=1,
        train_fn=train_fn,
        score_fn=_selection_score,
    )
    assert (64, 64) in seen


def test_search_is_reproducible_for_a_fixed_seed(tmp_path: Path) -> None:
    def make_train_fn(sink: list[tuple[int, int]]):
        def train_fn(
            dataset_dir: Path, run_dir: Path, settings: EmbeddingSettings
        ) -> dict[str, Any]:
            sink.append((settings.embedding_dim, settings.batch_size))
            return _mrr_report(settings.embedding_dim / 1000.0)

        return train_fn

    first: list[tuple[int, int]] = []
    second: list[tuple[int, int]] = []
    for sink, directory in ((first, "a"), (second, "b")):
        run_smac_search(
            tmp_path / "kg",
            tmp_path / directory,
            _settings(n_trials=8),
            seed=42,
            train_fn=make_train_fn(sink),
            score_fn=_selection_score,
        )
    assert first == second



# --- Partial failure: crashes must be recorded, not fatal



def test_a_crashing_trial_does_not_abort_the_search() -> None:
    def train_fn(
        dataset_dir: Path, run_dir: Path, settings: EmbeddingSettings
    ) -> dict[str, Any]:
        if settings.embedding_dim == 128:
            raise RuntimeError("CUDA out of memory")
        return _mrr_report(0.4)

    tmp_path = Path(tempfile.mkdtemp())
    outcome = run_smac_search(
        tmp_path / "kg",
        tmp_path / "emb",
        _settings(n_trials=10),
        seed=7,
        train_fn=train_fn,
        score_fn=_selection_score,
    )

    assert outcome.best_settings.embedding_dim != 128
    crashed = [t for t in outcome.trials if "error" in t]
    assert crashed, "the crashing configuration should have been sampled"
    assert all(t["cost"] == CRASH_COST for t in crashed)
    assert "CUDA out of memory" in crashed[0]["error"]


def test_interleaved_failures_still_yield_an_incumbent(tmp_path: Path) -> None:
    """Failures must not trip the abort guard when successes interleave."""
    counter = {"n": 0}

    def train_fn(
        dataset_dir: Path, run_dir: Path, settings: EmbeddingSettings
    ) -> dict[str, Any]:
        counter["n"] += 1
        if counter["n"] % 2 == 0:
            raise RuntimeError("transient failure")
        return _mrr_report(0.45)

    outcome = run_smac_search(
        tmp_path / "kg",
        tmp_path / "emb",
        _settings(n_trials=8),
        seed=11,
        train_fn=train_fn,
        score_fn=_selection_score,
    )
    assert outcome.incumbent_cost == pytest.approx(0.55)


def test_test_mrr_fallback_is_scored_and_flagged(tmp_path: Path) -> None:
    def train_fn(
        dataset_dir: Path, run_dir: Path, settings: EmbeddingSettings
    ) -> dict[str, Any]:
        return {"Train": {"MRR": 0.99}, "Test": {"MRR": 0.42}}

    outcome = run_smac_search(
        tmp_path / "kg",
        tmp_path / "emb",
        _settings(n_trials=4),
        seed=5,
        train_fn=train_fn,
        score_fn=_selection_score,
    )
    assert outcome.incumbent_cost == pytest.approx(0.58)
    assert "test MRR" in (outcome.validation_error or "")


# --- Total failure: the regression tests for your actual error


def test_train_only_reports_abort_with_the_split_misrouting_diagnosis() -> None:
    """Reproduces the reported failure: dicee reported only a train MRR.

    This is the dicee ReadFromDisk substring-routing bug, and the message
    must say so instead of just pointing at search_trials.
    """
    tmp_path = Path(tempfile.mkdtemp())
    def train_fn(
        dataset_dir: Path, run_dir: Path, settings: EmbeddingSettings
    ) -> dict[str, Any]:
        # Exactly what dicee returns when valid/test are captured into
        # raw_train_set: a Train section and nothing else.
        return {"Train": {"MRR": 0.97, "H@1": 0.95}}

    with pytest.raises(SearchAborted) as excinfo:
        run_smac_search(
            tmp_path / "kg",
            tmp_path / "emb",
            _settings(n_trials=16),
            seed=1,
            train_fn=train_fn,
            score_fn=_selection_score,
        )

    message = str(excinfo.value)
    assert "ONLY a train MRR" in message
    assert "ReadFromDisk" in message
    assert "stage_dicee_dataset" in message
    assert "eval_model" in message


def test_total_failure_aborts_early_instead_of_spending_the_budget(
    tmp_path: Path,
) -> None:
    """A systematic misconfiguration must not burn all n_trials."""
    calls = {"n": 0}

    def train_fn(
        dataset_dir: Path, run_dir: Path, settings: EmbeddingSettings
    ) -> dict[str, Any]:
        calls["n"] += 1
        return {"Train": {"MRR": 0.97}}

    with pytest.raises(SearchAborted):
        run_smac_search(
            tmp_path / "kg",
            tmp_path / "emb",
            _settings(n_trials=64),
            seed=1,
            train_fn=train_fn,
            score_fn=_selection_score,
        )

    assert calls["n"] <= MAX_CONSECUTIVE_UNSCORED, (
        f"aborted only after {calls['n']} trials; the guard should stop at "
        f"{MAX_CONSECUTIVE_UNSCORED}"
    )


def test_all_trials_crashing_reports_the_distinct_errors(
    tmp_path: Path,
) -> None:
    def train_fn(
        dataset_dir: Path, run_dir: Path, settings: EmbeddingSettings
    ) -> dict[str, Any]:
        raise ValueError("dataset_dir contains no train.txt")

    with pytest.raises(SearchAborted) as excinfo:
        run_smac_search(
            tmp_path / "kg",
            tmp_path / "emb",
            _settings(n_trials=16),
            seed=1,
            train_fn=train_fn,
            score_fn=_selection_score,
        )

    message = str(excinfo.value)
    assert "raised" in message
    assert "no train.txt" in message


def test_empty_report_reports_the_missing_metric_cause(tmp_path: Path) -> None:
    def train_fn(
        dataset_dir: Path, run_dir: Path, settings: EmbeddingSettings
    ) -> dict[str, Any]:
        return {}

    with pytest.raises(SearchAborted) as excinfo:
        run_smac_search(
            tmp_path / "kg",
            tmp_path / "emb",
            _settings(n_trials=8),
            seed=1,
            train_fn=train_fn,
            score_fn=_selection_score,
        )

    message = str(excinfo.value)
    assert "no MRR at all" in message
    assert "eval_model" in message


def test_non_finite_score_is_treated_as_unscorable(tmp_path: Path) -> None:
    """A NaN MRR must not become a NaN cost and poison the surrogate."""

    def train_fn(
        dataset_dir: Path, run_dir: Path, settings: EmbeddingSettings
    ) -> dict[str, Any]:
        return _mrr_report(float("nan"))

    with pytest.raises(SearchAborted):
        run_smac_search(
            tmp_path / "kg",
            tmp_path / "emb",
            _settings(n_trials=8),
            seed=1,
            train_fn=train_fn,
            score_fn=_selection_score,
        )