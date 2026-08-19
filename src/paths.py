# src/paths.py
"""Canonical filesystem layout for the benchmark suite.

The layout mirrors the project specification:

    Output/<benchmark>/seed<N>/<kb>/embeddings/{data,}
    Output/<benchmark>/seed<N>/<kb>/nces/{data,trained_models,}
    Output/<benchmark>/seed<N>/<kb>/logs
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


def update_false_dir_names(output_dir_name: str) -> str:
    tmp_name = output_dir_name
    return tmp_name.replace("train", "_trn_").replace("valid", "_vld_").replace("test", "_tst_")

PROJECT_ROOT = Path(update_false_dir_names(str(Path(__file__).resolve().parent.parent)))
DATASETS_DIR = PROJECT_ROOT / "datasets"
INPUT_DIR = PROJECT_ROOT / "input"
OUTPUT_DIR = PROJECT_ROOT / "Output"


@dataclass(frozen=True)
class RunPaths:
    """Every directory and artifact path used by one benchmark run.

    A benchmark run is one (benchmark name, seed, knowledge base) triple.
    """

    root: Path
    knowledge_base: str
    seed: int

    @property
    def kb_dir(self) -> Path:
        return self.root

    @property
    def embeddings_dir(self) -> Path:
        return self.root / "embeddings"

    @property
    def embeddings_data_dir(self) -> Path:
        return self.embeddings_dir / "data"

    @property
    def nces_dir(self) -> Path:
        return self.root / "nces"

    @property
    def nces_data_dir(self) -> Path:
        return self.nces_dir / "data"

    @property
    def trained_models_dir(self) -> Path:
        return self.nces_dir / "trained_models"

    @property
    def logs_dir(self) -> Path:
        return self.root / "logs"

    @property
    def learning_problems_path(self) -> Path:
        return self.nces_data_dir / "learning_problems.json"

    @property
    def report_path(self) -> Path:
        return self.nces_dir / "nces_report.json"

    @property
    def embedding_report_path(self) -> Path:
        return self.embeddings_dir / "embedding_report.json"

    def entity_embeddings_path(self, model_name: str, *, random: bool) -> Path:
        suffix = "_random" if random else ""
        return self.embeddings_dir / f"{model_name}{suffix}.csv"

    def dicee_run_dir(self, model_name: str) -> Path:
        return self.embeddings_dir / f"{model_name}_dicee_run"

    def mkdirs(self) -> None:
        for directory in (
            self.embeddings_data_dir,
            self.nces_data_dir,
            self.trained_models_dir,
            self.logs_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)


def run_paths(
    benchmark_name: str,
    seed: int,
    knowledge_base: str,
    *,
    output_dir: Path | None = None,
) -> RunPaths:
    """Build the :class:`RunPaths` for one benchmark run."""
    base = output_dir or OUTPUT_DIR
    root = base / benchmark_name / f"seed{seed}" / knowledge_base
    return RunPaths(root=root, knowledge_base=knowledge_base, seed=seed)


def resolve_knowledge_base(name_or_path: str) -> Path:
    """Resolve a knowledge-base name or path to an existing ``.owl`` file."""
    candidate = Path(name_or_path)
    if candidate.is_file():
        return candidate.resolve()

    stem = candidate.stem or candidate.name
    for probe in (
        DATASETS_DIR / f"{stem}.owl",
        DATASETS_DIR / stem / f"{stem}.owl",
    ):
        if probe.is_file():
            return probe.resolve()

    matches = sorted(DATASETS_DIR.rglob(f"{stem}.owl"))
    if matches:
        return matches[0].resolve()

    raise FileNotFoundError(
        f"No knowledge base found for {name_or_path!r}. "
        f"Place a .owl file under {DATASETS_DIR}."
    )
