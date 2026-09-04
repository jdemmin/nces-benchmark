# src/eval/plots.py
"""Figures. The link plot is the primary evidence for RQ1."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.eval.pairing import PairedDesign


def link_plot(quality: pd.DataFrame, path: Path, *, abl_column: str) -> None:
    """Downstream ABL against link-prediction MRR, one point per triple.

    The only analysis relating *variation in embedding quality* to
    *variation in downstream performance*, rather than comparing named
    architectures whose quality differences are unmeasured.
    """
    figure, axes = plt.subplots(figsize=(7, 5))
    for kb, group in quality.groupby("knowledge_base"):
        axes.scatter(group["mrr"], group[abl_column], label=str(kb), alpha=0.75)
    axes.axhline(0.0, linewidth=0.8, linestyle="--", color="grey")
    axes.set_xlabel("Link-prediction MRR")
    axes.set_ylabel("Mean atomic baseline lift (test split)")
    axes.legend(title="Knowledge base", fontsize="small")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=200)
    plt.close(figure)


def trend_plot(
    design: PairedDesign,
    outcome: str,
    path: Path,
    *,
    predictor: str = "depth",
) -> None:
    """Mean paired difference against a predictor, per-seed points overlaid."""
    column = f"d_{outcome}"
    usable = design.frame.dropna(subset=[column, predictor])
    if usable.empty:
        return

    figure, axes = plt.subplots(figsize=(7, 4.5))
    for seed, group in usable.groupby("seed"):
        means = group.groupby(predictor)[column].mean()
        axes.plot(
            means.index, means.to_numpy(), marker="o", alpha=0.45,
            linewidth=0.9, label=f"seed {seed}",
        )
    collapsed = (
        usable.groupby("problem_id")
        .agg(difference=(column, "mean"), key=(predictor, "first"))
        .groupby("key")["difference"]
        .mean()
    )
    axes.plot(
        collapsed.index, collapsed.to_numpy(), color="black",
        linewidth=2.0, marker="s", label="collapsed",
    )
    axes.axhline(0.0, linewidth=0.8, linestyle="--", color="grey")
    axes.set_xlabel(predictor)
    axes.set_ylabel(f"paired difference in {outcome}")
    axes.set_title(f"{design.condition} vs random — {design.knowledge_base}")
    axes.legend(fontsize="small", ncol=2)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=200)
    plt.close(figure)