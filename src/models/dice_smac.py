# src/models/dice_smac.py
"""SMAC3-driven hyperparameter search for the DICE embedding stage.

Uses ``HyperparameterOptimizationFacade``, whose default surrogate model is
a random forest with log expected improvement -- the right choice here
because the search space mixes categorical (scoring technique), integer
(dimension, batch size, epochs) and log-scaled float (learning rate)
hyperparameters, and because every trial is expensive enough that we can
only afford tens of evaluations.

SMAC minimizes, so the cost is ``1 - MRR``. A crashed trial returns
``CRASH_COST`` instead of raising, which keeps the run alive and lets the
random forest learn that the region is unusable.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.config import EmbeddingSettings

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ConfigSpace import Configuration, ConfigurationSpace

logger = logging.getLogger(__name__)

#: Cost reported for a trial that raised. ``1 - MRR`` is bounded by 1.0, so
#: any value above that is unambiguously worse than every real trial while
#: still being finite (``np.inf`` would poison the random forest's fit).
CRASH_COST: float = 2.0

class SearchAborted(RuntimeError):
    """Raised when the search is unrecoverable and must stop immediately."""


#: Consecutive unscorable trials tolerated before aborting. A misconfigured
#: dataset directory makes EVERY trial unscorable, so there is no point in
#: spending the remaining budget to confirm it.
MAX_CONSECUTIVE_UNSCORED: int = 3

@dataclass
class SmacSearchOutcome:
    """Result of one SMAC search over the DICE hyperparameter space."""

    best_settings: EmbeddingSettings
    best_report: dict[str, Any]
    best_run_dir: Path
    trials: list[dict[str, Any]] = field(default_factory=list)
    validation_error: str | None = None
    incumbent_cost: float | None = None


def build_configuration_space(
    settings: EmbeddingSettings, *, seed: int
) -> ConfigurationSpace:
    """Build the ConfigSpace for the DICE embedding search.

    ``embedding_dim`` and ``batch_size`` are ordinal categoricals rather
    than uniform integers: DICE's multi-component models require the
    dimension to stay a multiple of the component count, and powers of two
    keep the exported CSV width predictable for NCES.
    """
    from ConfigSpace import (
        Categorical,
        ConfigurationSpace,
        Float,
        Integer,
    )

    space = settings.search_space
    cs = ConfigurationSpace(seed=seed)

    hyperparameters: list[Any] = [
        Categorical(
            "embedding_dim",
            list(space.embedding_dim_choices),
            default=_nearest_choice(
                settings.embedding_dim, space.embedding_dim_choices
            ),
            ordered=True,
        ),
        Categorical(
            "batch_size",
            list(space.batch_size_choices),
            default=_nearest_choice(
                settings.batch_size, space.batch_size_choices
            ),
            ordered=True,
        ),
        Float(
            "learning_rate",
            space.learning_rate_bounds,
            default=_clamp(
                settings.learning_rate, space.learning_rate_bounds
            ),
            log=True,
        ),
    ]

    if space.tune_epochs:
        hyperparameters.append(
            Integer(
                "epochs",
                space.epochs_bounds,
                default=int(_clamp(settings.epochs, space.epochs_bounds)),
            )
        )
    if space.tune_scoring_technique:
        default = (
            settings.scoring_technique
            if settings.scoring_technique in space.scoring_technique_choices
            else space.scoring_technique_choices[0]
        )
        hyperparameters.append(
            Categorical(
                "scoring_technique",
                list(space.scoring_technique_choices),
                default=default,
            )
        )

    cs.add(hyperparameters)
    return cs


def _nearest_choice(value: int, choices: tuple[int, ...]) -> int:
    """Snap the configured default onto the categorical grid."""
    return min(choices, key=lambda choice: abs(choice - value))


def _clamp(value: float, bounds: tuple[float, float]) -> float:
    low, high = bounds
    return max(low, min(high, value))


def settings_from_configuration(
    base: EmbeddingSettings, config: Configuration
) -> EmbeddingSettings:
    """Materialize the sampled configuration as ``EmbeddingSettings``."""
    overrides: dict[str, Any] = {}
    values = dict(config)
    for name in (
        "embedding_dim",
        "batch_size",
        "epochs",
        "learning_rate",
        "scoring_technique",
    ):
        if name in values:
            overrides[name] = values[name]
    if "embedding_dim" in overrides:
        overrides["embedding_dim"] = int(overrides["embedding_dim"])
    if "batch_size" in overrides:
        overrides["batch_size"] = int(overrides["batch_size"])
    if "epochs" in overrides:
        overrides["epochs"] = int(overrides["epochs"])
    if "learning_rate" in overrides:
        overrides["learning_rate"] = float(overrides["learning_rate"])
    return base.with_overrides(**overrides)


def run_smac_search(
    dataset_dir: Path,
    embeddings_dir: Path,
    settings: EmbeddingSettings,
    *,
    seed: int,
    train_fn: Callable[[Path, Path, EmbeddingSettings], dict[str, Any]],
    score_fn: Callable[[dict[str, Any]], tuple[float | None, str | None]],
) -> SmacSearchOutcome:
    """Optimize the DICE hyperparameters with SMAC3.

    ``train_fn`` and ``score_fn`` are injected so this module stays free of
    a circular import back into ``src.models.dice`` and so the search can be
    unit-tested against a cheap surrogate objective.
    """
    from smac import HyperparameterOptimizationFacade, Scenario

    trials: list[dict[str, Any]] = []
    # Keyed by the ConfigSpace repr so the incumbent can be mapped back to
    # its run directory and report without retraining.
    evaluated: dict[str, dict[str, Any]] = {}
    counter = {"index": 0}

    configspace = build_configuration_space(settings, seed=seed)

    consecutive = {"unscored": 0}
    # SMAC swallows target-function exceptions and records them as crashed
    # trials, so the abort has to be latched here and re-raised after
    # optimize() returns.
    aborted: dict[str, str | None] = {"reason": None}
    def _record_unscored(record: dict[str, Any]) -> float:
        record["cost"] = CRASH_COST
        trials.append(record)
        consecutive["unscored"] += 1
        if consecutive["unscored"] >= MAX_CONSECUTIVE_UNSCORED:
            # Abort rather than let SMAC spend the remaining budget: a
            # systematic misconfiguration cannot be escaped by resampling.
            aborted["reason"] = _diagnose_total_failure(trials)
        return CRASH_COST
    
    def target_function(config: Configuration, seed: int = 0) -> float:
        if aborted["reason"] is not None:
            # Budget cannot be cancelled mid-run; make the remaining trials
            # free instead of training on a known-broken dataset.
            # Leads to funky 62 NOPs
            return CRASH_COST
        index = counter["index"]
        counter["index"] += 1

        trial_settings = settings_from_configuration(settings, config)
        run_dir = (
            embeddings_dir
            / f"trial_{index:02d}_{trial_settings.model_name}"
        )
        record: dict[str, Any] = {
            "trial": index,
            "model_name": trial_settings.model_name,
            "embedding_dim": trial_settings.embedding_dim,
            "batch_size": trial_settings.batch_size,
            "epochs": trial_settings.epochs,
            "learning_rate": trial_settings.learning_rate,
            "scoring_technique": trial_settings.scoring_technique,
            "run_dir": str(run_dir),
        }

        try:
            report = train_fn(dataset_dir, run_dir, trial_settings)
        except Exception as error:  # noqa: BLE001 - one trial must not kill the run
            logger.warning("SMAC trial %d failed: %s", index, error)
            record["error"] = str(error)
            return _record_unscored(record)

        score, validation_error = score_fn(report)
        record["score"] = score
        record["validation_error"] = validation_error
        record["metrics"] = {
            section: report[section]
            for section in ("Train", "Val", "Valid", "Test")
            if isinstance(report.get(section), dict)
        }

        if score is None or not math.isfinite(score):
            # No usable selection metric: treat as a crash so SMAC keeps
            # exploring instead of converging on an unscorable region.
            return _record_unscored(record)
        consecutive["unscored"] = 0
        cost = 1.0 - float(score)
        record["cost"] = cost
        trials.append(record)
        evaluated[str(config)] = {
            "settings": trial_settings,
            "report": report,
            "run_dir": run_dir,
            "validation_error": validation_error,
            "cost": cost,
        }
        return cost

    scenario_kwargs: dict[str, Any] = {
        "configspace": configspace,
        "name": f"dice_{settings.model_name}_seed{seed}",
        "output_directory": embeddings_dir / "smac",
        # DICE fixes its own random seed via ``args.random_seed``, so a
        # repeated configuration yields the same MRR. Declaring the
        # objective deterministic stops SMAC from re-evaluating configs
        # under extra seeds and wasting the trial budget.
        "deterministic": True,
        "n_trials": settings.n_trials,
        "seed": seed,
        "n_workers": settings.n_workers,
        # Evaluate the configured defaults first so the search can never do
        # worse than the hand-tuned configuration.
        "use_default_config": True,
        # Without this, SMAC's default crash cost is np.inf, which the
        # random-forest surrogate cannot fit ("Input y contains NaN").
        "crash_cost": CRASH_COST,
    }
    if settings.walltime_limit is not None:
        scenario_kwargs["walltime_limit"] = settings.walltime_limit
    if settings.trial_walltime_limit is not None:
        scenario_kwargs["trial_walltime_limit"] = settings.trial_walltime_limit

    scenario = Scenario(**scenario_kwargs)

    # A random forest surrogate needs a handful of observations before its
    # predictions beat random sampling; 25% of the budget is the usual
    # compromise for expensive objectives.
    initial_design = HyperparameterOptimizationFacade.get_initial_design(
        scenario,
        n_configs=max(2, settings.n_trials // 4),
    )

    smac = HyperparameterOptimizationFacade(
        scenario,
        target_function,
        initial_design=initial_design,
        # Never warm-start from a previous benchmark run: the run history
        # would leak across seeds and break reproducibility.
        overwrite=True,
        logging_level=logging.WARNING,
    )

    logger.info(
        "Starting SMAC search for DICE %s (%d trials, RF surrogate)",
        settings.model_name,
        settings.n_trials,
    )
    try:
        incumbent = smac.optimize()
    except SearchAborted:
        raise
    except Exception as error:
        if aborted["reason"] is not None:
            raise SearchAborted(aborted["reason"]) from error
        if any(isinstance(e, SearchAborted) for e in _exception_chain(error)):
            raise SearchAborted(_diagnose_total_failure(trials)) from error
        raise

    if aborted["reason"] is not None:
        raise SearchAborted(aborted["reason"])
    
    if isinstance(incumbent, list):  # multi-objective safety net
        incumbent = incumbent[0]

    best = evaluated.get(str(incumbent))
    if best is None:
        # The incumbent was not scorable (all trials crashed, or SMAC
        # returned a config we never recorded). Fall back to the cheapest
        # successful trial.
        successful = [
            entry for entry in evaluated.values() if entry["cost"] < CRASH_COST
        ]
        if not successful:
            raise SearchAborted(_diagnose_total_failure(trials))
        best = min(successful, key=lambda entry: entry["cost"])

    logger.info(
        "SMAC selected dim=%d batch=%d lr=%.4g (cost=%.4f, MRR=%.4f)",
        best["settings"].embedding_dim,
        best["settings"].batch_size,
        best["settings"].learning_rate,
        best["cost"],
        1.0 - best["cost"],
    )

    return SmacSearchOutcome(
        best_settings=best["settings"],
        best_report=best["report"],
        best_run_dir=best["run_dir"],
        trials=trials,
        validation_error=best["validation_error"],
        incumbent_cost=best["cost"],
    )

def _diagnose_total_failure(trials: list[dict[str, Any]]) -> str:
    """Build an actionable message for a search where no trial scored.

    The three distinct causes have completely different fixes, so the
    message must name which one occurred instead of pointing at the trial
    log generically.
    """
    if not trials:
        return (
            "SMAC ran zero trials. Check that n_trials >= 1 and that "
            "walltime_limit is not smaller than a single training run."
        )

    crashed = [t for t in trials if "error" in t]
    train_only = [
        t
        for t in trials
        if "error" not in t
        and t.get("score") is None
        and "train MRR" in (t.get("validation_error") or "")
    ]
    unscored = [
        t
        for t in trials
        if "error" not in t and t.get("score") is None and t not in train_only
    ]

    lines = [
        (f"Every one of the {len(trials)} SMAC trials failed or produced no "
         f"usable MRR."
        )
    ]

    if train_only:
        lines.append(
            f"{len(train_only)}/{len(trials)} trials reported ONLY a train "
            f"MRR. This is the dicee split-misrouting bug: ReadFromDisk "
            f"routes files with `if 'train' in full_path`, so a "
            f"'train'/'valid'/'test' token anywhere in an ANCESTOR "
            f"directory captures valid.txt and test.txt into the training "
            f"set. Fix: stage the dataset via stage_dicee_dataset() into a "
            f"token-free directory, or rename the benchmark/output path. "
            f"Also verify eval_model='train_val_test'."
        )
    if unscored:
        example = unscored[0].get("validation_error") or "no MRR reported"
        lines.append(
            f"{len(unscored)}/{len(trials)} trials returned no MRR at all "
            f"(first reason: {example}). Check that valid.txt and test.txt "
            f"are non-empty and that eval_model is set."
        )
    if crashed:
        messages = sorted({str(t["error"])[:200] for t in crashed})
        lines.append(
            f"{len(crashed)}/{len(trials)} trials raised. Distinct errors: "
            + "; ".join(messages[:3])
        )

    return " ".join(lines)

def _exception_chain(error: BaseException) -> list[BaseException]:
    chain: list[BaseException] = []
    current: BaseException | None = error
    while current is not None and current not in chain:
        chain.append(current)
        current = current.__cause__ or current.__context__
    return chain