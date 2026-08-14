# src/__init__.py
"""NCES benchmark package with lazy imports for heavy dependencies."""

from __future__ import annotations

from typing import Any

__all__ = ["BenchmarkConfiguration", "run_benchmark", "run_single"]
__version__ = "0.1.0"


def __getattr__(name: str) -> Any:
    """Lazy import so that ``import src`` does not pull in torch."""
    if name == "BenchmarkConfiguration":
        from src.config import BenchmarkConfiguration

        return BenchmarkConfiguration
    if name in {"run_benchmark", "run_single"}:
        from src.benchmarking import runner

        return getattr(runner, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
