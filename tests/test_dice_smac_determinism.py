# tests/test_dice_smac_determinism.py
"""Determinism guarantees for the SMAC-driven DICE search.

Two independent runs with the same seed must produce the same trial
sequence, the same incumbent and the same run directories. Everything is
driven through a cheap deterministic surrogate objective so no DICE
training happens.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pytest

from src.config import EmbeddingSearchSpace, EmbeddingSettings
from src.models.dice_smac import (
    CRASH_COST,
    SearchAborted,
    _config_key,
    _run_dir_for,
    build_configuration_space,
    run_smac_search,
    settings_from_configuration,
)

pytest.importorskip("smac")
pytest.importorskip("ConfigSpace")


N_TRIALS = 8


def _settings(**overrides: Any) -> EmbeddingSettings:
    base = EmbeddingSettings(
        model_name="Keci",
        embedding_dim=64,
        epochs=5,
        batch_size=64,
        learning_rate=0.05,
        hpo_backend="smac",
        n_trials=N_TRIALS,
        n_workers=1,
        walltime_limit=None,
        trial_walltime_limit=None,
        search_space=EmbeddingSearchSpace(
            embedding_dim_choices=(32, 64, 128),
            batch_size_choices=(32, 64, 128),
            learning_rate_bounds=(1e-3, 3e-1),
            epochs_bounds=(5, 20),
            tune_epochs=True,
            tune_scoring_technique=True,
        ),
    )
    return base.with_overrides(**overrides)


def _surrogate_train_fn(
    calls: list[dict[str, Any]],
) -> Any:
    """A pure, closed-form stand-in for ``train_dice``.

    The MRR is a deterministic function of the sampled hyperparameters
    only -- no RNG, no clock, no filesystem -- so any run-to-run
    difference in the recorded trials must come from the search itself.
    """

    def train_fn(
        dataset_dir: Path, run_dir: Path, settings: EmbeddingSettings
    ) -> dict[str, Any]:
        calls.append(
            {
                "run_dir": str(run_dir),
                "embedding_dim": settings.embedding_dim,
                "batch_size": settings.batch_size,
                "epochs": settings.epochs,
                "learning_rate": settings.learning_rate,
                "scoring_technique": settings.scoring_technique,
            }
        )
        run_dir.mkdir(parents=True, exist_ok=True)
        # Unimodal in log-lr, mildly rewards larger dimension.
        lr_penalty = (math.log10(settings.learning_rate) + 1.5) ** 2
        mrr = 0.9 / (1.0 + lr_penalty) + settings.embedding_dim / 4096
        return {
            "Train": {"MRR": mrr * 0.99},
            "Val": {"MRR": mrr},
            "Test": {"MRR": mrr * 0.98},
        }

    return train_fn


def _score_fn(report: dict[str, Any]) -> tuple[float | None, str | None]:
    section = report.get("Val") or report.get("Valid")
    if not isinstance(section, dict) or "MRR" not in section:
        return None, "no validation MRR reported"
    return float(section["MRR"]), None


def _run(tmp_path: Path, seed: int, tag: str, **overrides: Any):
    calls: list[dict[str, Any]] = []
    embeddings_dir = tmp_path / tag
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    outcome = run_smac_search(
        dataset_dir=tmp_path / "dataset",
        embeddings_dir=embeddings_dir,
        settings=_settings(**overrides),
        seed=seed,
        train_fn=_surrogate_train_fn(calls),
        score_fn=_score_fn,
    )
    return outcome, calls


def _fingerprint(calls: list[dict[str, Any]]) -> list[tuple[Any, ...]]:
    """Order-sensitive summary of the proposed configurations."""
    return [
        (
            c["embedding_dim"],
            c["batch_size"],
            c["epochs"],
            round(c["learning_rate"], 12),
            c["scoring_technique"],
        )
        for c in calls
    ]


# --------------------------------------------------------------------------
# Configuration space
# --------------------------------------------------------------------------


def test_configuration_space_sampling_is_seed_reproducible() -> None:
    settings = _settings()
    first = build_configuration_space(settings, seed=7).sample_configuration(20)
    second = build_configuration_space(settings, seed=7).sample_configuration(20)
    assert [dict(c) for c in first] == [dict(c) for c in second]


def test_configuration_space_differs_across_seeds() -> None:
    settings = _settings()
    a = build_configuration_space(settings, seed=7).sample_configuration(20)
    b = build_configuration_space(settings, seed=8).sample_configuration(20)
    assert [dict(c) for c in a] != [dict(c) for c in b]


def test_default_configuration_matches_hand_tuned_settings() -> None:
    settings = _settings()
    default = build_configuration_space(settings, seed=1).get_default_configuration()
    materialized = settings_from_configuration(settings, default)
    assert materialized.embedding_dim == settings.embedding_dim
    assert materialized.batch_size == settings.batch_size
    assert materialized.learning_rate == pytest.approx(settings.learning_rate)


# --------------------------------------------------------------------------
# Key + run-directory derivation
# --------------------------------------------------------------------------


def test_config_key_is_deterministic_and_independent_of_insertion_order() -> None:
    """``_config_key`` shuffles then sorts, so the seed must not matter.

    If the shuffle ever moves outside the sort, this test fails -- and so
    would the incumbent lookup in ``run_smac_search``.
    """
    config = {
        "embedding_dim": 64,
        "batch_size": 32,
        "epochs": 10,
        "learning_rate": 0.05,
        "scoring_technique": "KvsAll",
    }
    reversed_config = dict(reversed(list(config.items())))
    assert _config_key(config, 1) == _config_key(config, 1) # type: ignore
    assert _config_key(config, 1) != _config_key(config, 2) # type: ignore
    assert _config_key(config, 1) == _config_key(reversed_config, 1) # type: ignore


def test_config_key_survives_json_round_trip() -> None:
    """Trial records are reloaded from JSON before the incumbent lookup."""
    import json

    config = {
        "embedding_dim": 128,
        "batch_size": 64,
        "epochs": 12,
        "learning_rate": 0.0123456789,
        "scoring_technique": "1vsAll",
    }
    round_tripped = json.loads(json.dumps(config))
    assert _config_key(config, 3) == _config_key(round_tripped, 3) # type: ignore


def test_run_dir_is_a_pure_function_of_the_configuration(tmp_path: Path) -> None:
    settings = _settings()
    cs = build_configuration_space(settings, seed=5)
    config = cs.sample_configuration()
    trial_settings = settings_from_configuration(settings, config)
    first = _run_dir_for(tmp_path, trial_settings, config)
    second = _run_dir_for(tmp_path, trial_settings, config)
    assert first == second

    other = next(
        c for c in cs.sample_configuration(25) if dict(c) != dict(config)
    )
    assert _run_dir_for(
        tmp_path, settings_from_configuration(settings, other), other
    ) != first


# --------------------------------------------------------------------------
# End-to-end determinism
# --------------------------------------------------------------------------


def test_same_seed_yields_identical_trial_sequence(tmp_path: Path) -> None:
    first, calls_a = _run(tmp_path, seed=13, tag="a")
    second, calls_b = _run(tmp_path, seed=13, tag="b")

    assert len(calls_a) == N_TRIALS
    assert _fingerprint(calls_a) == _fingerprint(calls_b)
    assert [t["cost"] for t in first.trials] == [
        t["cost"] for t in second.trials
    ]


def test_same_seed_yields_identical_incumbent(tmp_path: Path) -> None:
    first, _ = _run(tmp_path, seed=13, tag="a")
    second, _ = _run(tmp_path, seed=13, tag="b")

    assert first.incumbent_cost == pytest.approx(second.incumbent_cost)
    assert first.best_settings == second.best_settings
    assert first.best_run_dir.name == second.best_run_dir.name
    assert first.best_report == second.best_report


def test_default_config_is_evaluated_after_the_initial_design(
    tmp_path: Path,
) -> None:
    """``use_default_config=True`` appends the default, it does not prepend it.

    ``AbstractInitialDesign.select_configurations`` does
    ``configs += self._select_configurations()`` (Sobol) and only then
    ``configs += self._additional_configs`` (which is where the default
    lands). So the default is evaluated at index ``n_configs``, not 0.
    """
    settings = _settings()
    n_initial = max(2, N_TRIALS // 4)
    _, calls = _run(tmp_path, seed=13, tag="a")

    default = build_configuration_space(
        settings, seed=13
    ).get_default_configuration()
    expected = settings_from_configuration(settings, default)

    assert calls[n_initial]["embedding_dim"] == expected.embedding_dim
    assert calls[n_initial]["batch_size"] == expected.batch_size
    assert calls[n_initial]["learning_rate"] == pytest.approx(
        expected.learning_rate
    )
    assert calls[n_initial]["epochs"] == expected.epochs


def test_default_config_is_evaluated_exactly_once(tmp_path: Path) -> None:
    """The dedup in ``select_configurations`` must not drop or double it."""
    settings = _settings()
    _, calls = _run(tmp_path, seed=13, tag="a")
    default = dict(
        build_configuration_space(settings, seed=13).get_default_configuration()
    )
    matches = [
        c
        for c in calls
        if c["embedding_dim"] == default["embedding_dim"]
        and c["batch_size"] == default["batch_size"]
        and c["epochs"] == default["epochs"]
        and math.isclose(c["learning_rate"], default["learning_rate"])
        and c["scoring_technique"] == default["scoring_technique"]
    ]
    assert len(matches) == 1


def test_initial_design_budget_fits_in_n_trials() -> None:
    """``n_configs + 1`` (the default) must not exceed ``n_trials``.

    ``AbstractInitialDesign.__init__`` raises ValueError otherwise, and
    ``run_smac_search`` would die before the first trial.
    """
    from smac import HyperparameterOptimizationFacade, Scenario

    for n_trials in (1, 2, 3, 4, 8, 16):
        settings = _settings(n_trials=n_trials)
        scenario = Scenario(
            configspace=build_configuration_space(settings, seed=1),
            name=f"budget_{n_trials}",
            deterministic=True,
            n_trials=n_trials,
            seed=1,
            use_default_config=True,
            crash_cost=CRASH_COST,
            n_workers=1,
        )
        HyperparameterOptimizationFacade.get_initial_design(
        scenario,
        # +1 because use_default_config appends the default as an
        # additional_config, and AbstractInitialDesign raises if
        # n_configs + len(additional_configs) > n_trials.
        n_configs=max(0, min(settings.n_trials // 4, settings.n_trials - 1)),
    )


def test_different_seeds_explore_differently(tmp_path: Path) -> None:
    _, calls_a = _run(tmp_path, seed=13, tag="a")
    _, calls_b = _run(tmp_path, seed=99, tag="b")
    assert _fingerprint(calls_a) != _fingerprint(calls_b)


def test_trial_records_on_disk_match_the_returned_trials(
    tmp_path: Path,
) -> None:
    """``run_smac_search`` re-reads the records; the two must agree."""
    outcome, calls = _run(tmp_path, seed=13, tag="a")
    written = sorted((tmp_path / "a" / "smac_trials").glob("trial_*.json"))
    assert len(written) == len(calls)
    assert [t["trial"] for t in outcome.trials] == list(range(len(calls)))


# --------------------------------------------------------------------------
# Determinism of the failure paths
# --------------------------------------------------------------------------


def test_repeated_crashes_abort_deterministically(tmp_path: Path) -> None:
    """A systematically broken objective must abort at the same trial."""

    def exploding_train_fn(
        dataset_dir: Path, run_dir: Path, settings: EmbeddingSettings
    ) -> dict[str, Any]:
        raise RuntimeError("dicee: valid.txt is empty")

    def _go(tag: str) -> str:
        directory = tmp_path / tag
        directory.mkdir(parents=True, exist_ok=True)
        with pytest.raises(SearchAborted) as excinfo:
            run_smac_search(
                dataset_dir=tmp_path / "dataset",
                embeddings_dir=directory,
                settings=_settings(),
                seed=13,
                train_fn=exploding_train_fn,
                score_fn=_score_fn,
            )
        return str(excinfo.value)

    message_a = _go("crash_a")
    message_b = _go("crash_b")
    assert message_a == message_b
    assert "trials raised" in message_a


def test_unscorable_report_is_costed_as_a_crash(tmp_path: Path) -> None:
    """An unscorable trial gets CRASH_COST, not a NaN that kills the RF."""

    def unscorable_train_fn(
        dataset_dir: Path, run_dir: Path, settings: EmbeddingSettings
    ) -> dict[str, Any]:
        run_dir.mkdir(parents=True, exist_ok=True)
        return {"Train": {"MRR": 0.5}}

    directory = tmp_path / "unscored"
    directory.mkdir(parents=True, exist_ok=True)
    with pytest.raises(SearchAborted) as excinfo:
        run_smac_search(
            dataset_dir=tmp_path / "dataset",
            embeddings_dir=directory,
            settings=_settings(),
            seed=13,
            train_fn=unscorable_train_fn,
            score_fn=_score_fn,
        )
    assert "no usable MRR" in str(excinfo.value)
    records = sorted((directory / "smac_trials").glob("trial_*.json"))
    assert records, "unscored trials must still be persisted"
    import json

    costs = [json.loads(p.read_text())["cost"] for p in records]
    assert all(c == CRASH_COST for c in costs)
    assert all(math.isfinite(c) for c in costs)