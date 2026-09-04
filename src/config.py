# src/config.py
"""Typed benchmark configuration loaded from the ``input/`` JSON files.

The JSON field names follow the project specification. The dataclasses
translate them into the keyword arguments that ``ontolearn``, ``dicee`` and
``NCES`` actually accept, which is documented per field below.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

from src.paths import INPUT_DIR

#: DICE architectures usable as an ``embedding model`` in this project.
DICE_MODELS: tuple[str, ...] = (
    "Decal",
    "Keci",
    "DualE",
    "ComplEx",
    "QMult",
    "OMult",
    "ConvQ",
    "ConvO",
    "ConEx",
    "TransE",
    "DistMult",
    "Shallom",
)

#: NCES recurrent/attention learners exposed by ontolearn.
NCES_LEARNERS: tuple[str, ...] = ("LSTM", "GRU", "SetTransformer")

#: Train / validation / test proportions for the RDF triple split.
SPLIT_RATIOS: tuple[float, float, float] = (0.8, 0.1, 0.1)

#: Embedding conditions compared by the benchmark: the twelve DICE
#: architectures plus the "random" control, the reference level.
EMBEDDING_CONDITIONS: tuple[str, ...] = (*DICE_MODELS, "random")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing settings file: {path}")
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"{path} must contain a JSON object.")
    return payload


@dataclass(frozen=True)
class DataGenerationSettings:
    """Learning-problem generation settings (``data_generation_settings.json``).

    These fields configure ``ontolearn.lp_generator.LPGen``, which internally
    builds the refinement operator conventionally named ``rho``:

        rho = ExpressRefinement(
            knowledge_base=kb,
            max_child_length=max_child_len,
            sample_fillers_count=sample_fillers_count,
            downsample=downsample_refinements,
            expressivity=refinement_expressivity,
            use_inverse=beyond_alc,
            use_card_restrictions=beyond_alc,
            use_numeric_datatypes=beyond_alc,
            use_time_datatypes=beyond_alc,
            use_boolean_datatype=beyond_alc,
        )

    ``rho`` is therefore not a parameter but the operator object itself:
    ``beyond_alc`` selects ALC (all five ``use_*`` flags off) versus
    ALCHIQ(D) (all five on), while ``refinement_expressivity`` and
    ``downsample_refinements`` become ``expressivity`` and ``downsample``.
    """

    kb: str = "semantic_bible"
    num_rand_samples: int = 150
    depth: int = 2
    max_child_len: int = 10
    refinement_expressivity: float = 0.2
    beyond_alc: bool = False
    downsample_refinements: bool = True
    sample_fillers_count: int = 10
    num_sub_roots: int = 50
    min_num_pos_examples: int = 1
    max_pos_neg_examples_per_lp: int | None = None

    def __post_init__(self) -> None:
        if not 0.0 < self.refinement_expressivity <= 1.0:
            raise ValueError(
                "refinement_expressivity must be in (0.0, 1.0], got "
                f"{self.refinement_expressivity}."
            )
        # Upstream requirement: ExpressRefinement cannot honour an
        # expressivity below 1.0 without downsampling its refinements.
        if self.refinement_expressivity < 1.0 and not self.downsample_refinements:
            raise ValueError(
                "downsample_refinements must be True when "
                "refinement_expressivity < 1.0 (LPGen/ExpressRefinement "
                "constraint)."
            )
        if self.depth < 1:
            raise ValueError(f"depth must be >= 1, got {self.depth}.")
        if self.min_num_pos_examples < 1:
            raise ValueError(
                f"min_num_pos_examples must be >= 1, got "
                f"{self.min_num_pos_examples}."
            )

    @property
    def refinement_operator_expressivity(self) -> str:
        """Human-readable DL name of the refinement operator, for logging."""
        return "ALCHIQ(D)" if self.beyond_alc else "ALC"

    def lpgen_kwargs(self, kb_path: Path, storage_path: Path) -> dict[str, Any]:
        """Keyword arguments for ``ontolearn.lp_generator.LPGen``.

        Two project field names differ from upstream: ``num_rand_samples``
        is upstream's ``max_num_lps``, and ``max_child_len`` is upstream's
        ``max_child_length``.
        """
        kwargs: dict[str, Any] = {
            "kb_path": str(kb_path),
            "storage_path": str(storage_path),
            "max_num_lps": self.num_rand_samples,
            "beyond_alc": self.beyond_alc,
            "depth": self.depth,
            "max_child_length": self.max_child_len,
            "refinement_expressivity": self.refinement_expressivity,
            "downsample_refinements": self.downsample_refinements,
            "sample_fillers_count": self.sample_fillers_count,
            "num_sub_roots": self.num_sub_roots,
            "min_num_pos_examples": self.min_num_pos_examples,
        }
        # Only present in ontolearn >= the 2025-09-29 example-limit commit.
        if self.max_pos_neg_examples_per_lp is not None:
            kwargs["max_pos_neg_examples_per_lp"] = (
                self.max_pos_neg_examples_per_lp
            )
        return kwargs

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> DataGenerationSettings:
        # Accept the legacy `rho` spelling: "ALCHIQD" meant beyond_alc=True.
        if "beyond_alc" in payload:
            beyond_alc = bool(payload["beyond_alc"])
        elif "rho" in payload:
            beyond_alc = str(payload["rho"]).strip().upper() in {
                "ALCHIQD",
                "ALCHIQ(D)",
                "BEYOND_ALC",
            }
        else:
            beyond_alc = False

        expressivity = float(payload.get("refinement_expressivity", 0.2))
        return cls(
            kb=str(payload.get("kbs", "semantic_bible")),
            num_rand_samples=int(payload.get("num_rand_samples", 150)),
            depth=int(payload.get("depth", 2)),
            max_child_len=int(payload.get("max_child_len", 10)),
            refinement_expressivity=expressivity,
            beyond_alc=beyond_alc,
            downsample_refinements=bool(
                payload.get("downsample_refinements", expressivity < 1.0)
            ),
            sample_fillers_count=int(payload.get("sample_fillers_count", 10)),
            num_sub_roots=int(payload.get("num_sub_roots", 50)),
            min_num_pos_examples=int(payload.get("min_num_pos_examples", 1)),
            max_pos_neg_examples_per_lp=payload.get(
                "max_pos_neg_examples_per_lp"
            ),
        )

#: Surrogate/HPO backends for the DICE embedding search.
HPO_BACKENDS: tuple[str, ...] = ("smac", "grid")


@dataclass(frozen=True)
class EmbeddingSearchSpace:
    """Bounds for the SMAC configuration space of the DICE search.

    ``embedding_dim`` is deliberately not a field here: it is fixed at 128
    for every condition and excluded from the search space so that no
    architecture is advantaged by capacity (methodology sec:meth:dimensionality).
    """

    batch_size_choices: tuple[int, ...] = (32, 64, 128, 256, 512)
    learning_rate_bounds: tuple[float, float] = (1e-3, 3e-1)
    epochs_bounds: tuple[int, int] = (25, 100)
    scoring_technique_choices: tuple[str, ...] = ("KvsAll", "NegSample", "AllvsAll", "1vsAll")
    tune_epochs: bool = True
    tune_scoring_technique: bool = False

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> EmbeddingSearchSpace:
        space = payload.get("search_space", {})
        if not isinstance(space, dict):
            raise TypeError("embedding_settings.search_space must be an object.")
        defaults = cls()
        return cls(
            batch_size_choices=tuple(
                int(v)
                for v in space.get(
                    "batch_size_choices", defaults.batch_size_choices
                )
            ),
            learning_rate_bounds=tuple(  # type: ignore[arg-type]
                float(v)
                for v in space.get(
                    "learning_rate_bounds", defaults.learning_rate_bounds
                )
            ),
            epochs_bounds=tuple(  # type: ignore[arg-type]
                int(v) for v in space.get("epochs_bounds", defaults.epochs_bounds)
            ),
            scoring_technique_choices=tuple(
                str(v)
                for v in space.get(
                    "scoring_technique_choices",
                    defaults.scoring_technique_choices,
                )
            ),
            tune_epochs=bool(space.get("tune_epochs", defaults.tune_epochs)),
            tune_scoring_technique=bool(
                space.get(
                    "tune_scoring_technique", defaults.tune_scoring_technique
                )
            ),
        )

@dataclass(frozen=True)
class EmbeddingSettings:
    """DICE embedding settings (``embedding_settings.json``)."""

    model_name: str = "Keci"
    embedding_dim: int = 128
    epochs: int = 150
    batch_size: int = 64
    scoring_technique: str = "KvsAll"
    trainer: str = "torchCPUTrainer"
    eval_model: str = "train_val_test"
    num_core: int = 0
    learning_rate: float = 0.1

    # --- hyperparameter search ------------------------------------------
    hpo_backend: str = "smac"
    n_trials: int = 32
    walltime_limit: float | None = None
    trial_walltime_limit: float | None = None
    n_workers: int = 1
    search_space: EmbeddingSearchSpace = field(
        default_factory=EmbeddingSearchSpace
    )

    def __post_init__(self) -> None:
        if self.model_name not in DICE_MODELS:
            raise ValueError(
                f"Unknown DICE model {self.model_name!r}. "
                f"Choose one of {', '.join(DICE_MODELS)}."
            )
        if self.hpo_backend not in HPO_BACKENDS:
            raise ValueError(
                f"Unknown hpo_backend {self.hpo_backend!r}. "
                f"Choose one of {', '.join(HPO_BACKENDS)}."
            )
        if self.n_trials < 1:
            raise ValueError(f"n_trials must be >= 1, got {self.n_trials}.")

    def search_grid(self) -> list[EmbeddingSettings]:
        """Legacy grid: base/halved batch, at the fixed embedding dimension.

        Kept as the ``hpo_backend="grid"`` fallback so a run can be
        reproduced without SMAC installed. ``embedding_dim`` is never
        varied: it is fixed for every condition.
        """
        batches = sorted({self.batch_size, max(1, self.batch_size // 2)})
        return [replace(self, batch_size=batch) for batch in batches]

    def with_overrides(self, **overrides: Any) -> EmbeddingSettings:
        """Return a copy with the SMAC-sampled values applied."""
        return replace(self, **overrides)

    @classmethod
    def from_json(cls, path: Path | None = None) -> EmbeddingSettings:
        try:
            payload = _read_json(path or INPUT_DIR / "embedding_settings.json")
        except FileNotFoundError:
            logger.info(
                "embedding_settings.json not found. Using default embedding "
                "settings."
            )
            return cls()
        walltime = payload.get("walltime_limit")
        trial_walltime = payload.get("trial_walltime_limit")
        return cls(
            model_name=str(payload.get("model_name", "Keci")),
            embedding_dim=int(payload.get("embedding_dim", 128)),
            epochs=int(payload.get("epochs", 150)),
            batch_size=int(payload.get("batch_size", 64)),
            scoring_technique=str(payload.get("scoring_technique", "KvsAll")),
            trainer=str(payload.get("trainer", "torchCPUTrainer")),
            eval_model=str(payload.get("eval_model", "train_val_test")),
            num_core=int(payload.get("num_core", 0)),
            learning_rate=float(payload.get("learning_rate", 0.1)),
            hpo_backend=str(payload.get("hpo_backend", "smac")),
            n_trials=int(payload.get("n_trials", 32)),
            walltime_limit=None if walltime is None else float(walltime),
            trial_walltime_limit=(
                None if trial_walltime is None else float(trial_walltime)
            ),
            n_workers=int(payload.get("n_workers", 1)),
            search_space=EmbeddingSearchSpace.from_payload(payload),
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EmbeddingSettings:
        walltime = data.get("walltime_limit")
        trial_walltime = data.get("trial_walltime_limit")
        return cls(
            model_name=str(data.get("model_name", "Keci")),
            embedding_dim=int(data.get("embedding_dim", 128)),
            epochs=int(data.get("epochs", 150)),
            batch_size=int(data.get("batch_size", 64)),
            scoring_technique=str(data.get("scoring_technique", "KvsAll")),
            trainer=str(data.get("trainer", "torchCPUTrainer")),
            eval_model=str(data.get("eval_model", "train_val_test")),
            num_core=int(data.get("num_core", 0)),
            learning_rate=float(data.get("learning_rate", 0.1)),
            hpo_backend=str(data.get("hpo_backend", "smac")),
            n_trials=int(data.get("n_trials", 32)),
            walltime_limit=None if walltime is None else float(walltime),
            trial_walltime_limit=(
                None if trial_walltime is None else float(trial_walltime)
            ),
            n_workers=int(data.get("n_workers", 1)),
            search_space=EmbeddingSearchSpace.from_payload(data),
        )

    
    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "embedding_dim": self.embedding_dim,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "scoring_technique": self.scoring_technique,
            "trainer": self.trainer,
            "eval_model": self.eval_model,
            "num_core": self.num_core,
            "learning_rate": self.learning_rate,
            "hpo_backend": self.hpo_backend,
            "n_trials": self.n_trials,
            "walltime_limit": self.walltime_limit,
            "trial_walltime_limit": self.trial_walltime_limit,
            "n_workers": self.n_workers,
            "search_space": vars(self.search_space),
        }


@dataclass(frozen=True)
class NCESSettings:
    """NCES learner settings (``nces_settings.json``)."""

    learner_name: str = "GRU"
    embedding_dim: int = 64
    epochs: int = 300
    batch_size: int = 32
    proj_dim: int = 40
    rnn_n_layers: int = 1
    drop_prob: float = 0.0
    num_heads: int = 2
    num_seeds: int = 1
    num_workers: int = 0
    max_length: int = 48
    num_predictions: int = 64
    learning_rate: float = 1e-4
    sorted_examples: bool = True

    def __post_init__(self) -> None:
        if self.learner_name not in NCES_LEARNERS:
            raise ValueError(
                f"Unknown NCES learner {self.learner_name!r}. "
                f"Choose one of {', '.join(NCES_LEARNERS)}."
            )

    @property
    def learner_names(self) -> list[str]:
        """Upstream ``NCES`` expects a list under ``learner_names``."""
        return [self.learner_name]

    @classmethod
    def from_json(cls, path: Path | None = None) -> NCESSettings:
        try:
            payload = _read_json(path or INPUT_DIR / "nces_settings.json")
        except FileNotFoundError:
            # Provide a default NCES settings if the file is missing.
            logger.info("nces_settings.json not found. Using default NCES settings.")
            return cls()
        return cls(
            learner_name=str(payload.get("learner_name", "GRU")),
            embedding_dim=int(payload.get("embedding_dim", 64)),
            epochs=int(payload.get("epochs", 300)),
            batch_size=int(payload.get("batch_size", 32)),
            proj_dim=int(payload.get("proj_dim", 40)),
            rnn_n_layers=int(payload.get("rnn_n_layers", 1)),
            drop_prob=float(payload.get("drop_prob", 0.0)),
            num_heads=int(payload.get("num_heads", 2)),
            num_seeds=int(payload.get("num_seeds", 1)),
            num_workers=int(payload.get("num_workers", 0)),
            max_length=int(payload.get("max_length", 48)),
            num_predictions=int(payload.get("num_predictions", 64)),
            learning_rate=float(payload.get("learning_rate", 1e-4)),
            sorted_examples=bool(payload.get("sorted_examples", True)),
        )
        


@dataclass(frozen=True)
class ProjectSettings:
    """General project settings (``project_settings.json``)."""

    seeds: list[int] = field(default_factory=lambda: [1, 2, 3, 4, 5])
    benchmark_name: str = "benchmark1"
    embedding_conditions: list[str] = field(
        default_factory=lambda: list(EMBEDDING_CONDITIONS)
    )
    stratify_by: str = "dl_length"

    def __post_init__(self) -> None:
        unknown = set(self.embedding_conditions) - set(EMBEDDING_CONDITIONS)
        if unknown:
            raise ValueError(
                f"Unknown embedding condition(s): {sorted(unknown)}. "
                f"Choose from {list(EMBEDDING_CONDITIONS)}."
            )
        if not self.seeds:
            raise ValueError("project_settings.json must define at least one seed.")

    @classmethod
    def from_json(cls, path: Path | None = None) -> ProjectSettings:
        try:
            payload = _read_json(path or INPUT_DIR / "project_settings.json")
        except FileNotFoundError:
            # Provide a default project settings if the file is missing. relevant
            # for CI/CD pipelines that don't have the input directory mounted.
            logger.info(
                "project_settings.json not found. Using default project settings."
            )
            return cls()
        return cls(
            seeds=[int(s) for s in payload.get("seeds", [1, 2, 3, 4, 5])],
            benchmark_name=str(payload.get("benchmark_name", "benchmark1")),
            embedding_conditions=list(
                payload.get("embedding_conditions", EMBEDDING_CONDITIONS)
            ),
            stratify_by=str(payload.get("stratify_by", cls.stratify_by)),
        )

@dataclass(frozen=True)
class BenchmarkConfiguration:
    """The complete benchmark configuration for one invocation."""

    project: ProjectSettings
    data_generation: list[DataGenerationSettings]
    embedding: EmbeddingSettings
    nces: NCESSettings
    knowledge_bases: list[str]

    @classmethod
    def load(cls, input_dir: Path | None = None) -> BenchmarkConfiguration:
        base = input_dir or INPUT_DIR
        project = ProjectSettings.from_json(base / "project_settings.json")

        raw_data_gen = _read_json(base / "data_generation_settings.json")
        data_generation_list: list[DataGenerationSettings] = []
        knowledge_bases: list[str] = []
        for data_generation_settings in raw_data_gen.values():
            for setting in data_generation_settings:
                setting = DataGenerationSettings.from_json(setting)
                knowledge_bases.append(setting.kb)
                data_generation_list.append(setting)
        embedding = EmbeddingSettings.from_json(base / "embedding_settings.json")
        nces = NCESSettings.from_json(base / "nces_settings.json")
        return cls(
            project=project,
            data_generation=data_generation_list,
            embedding=embedding,
            nces=nces,
            knowledge_bases=knowledge_bases,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the benchmark configuration for the report."""
        return {
            "seeds": self.project.seeds,
            "benchmark_name": self.project.benchmark_name,
            "embedding_conditions": self.project.embedding_conditions,
            "knowledge_bases": self.knowledge_bases,
            "data_generation": [vars(d) for d in self.data_generation],
            "embedding": vars(self.embedding),
            "nces": vars(self.nces),
            "split_ratios": list(SPLIT_RATIOS),
        }
