# src/eval/failures.py
"""Three kinds of shortfall, kept apart in terminology.

An exception is raised only when the paired design itself is impossible.
Everything downstream degrades and reports: each layer is guarded
independently, so a failed layer is noted and the remaining layers still
produce estimates.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class ShortfallLedger:
    """Accumulates per-block notes and unavailable outcomes for one report."""

    block_notes: dict[str, str] = field(default_factory=dict)
    unavailable_outcomes: list[str] = field(default_factory=list)
    problem_errors: dict[str, int] = field(default_factory=dict)

    def note(self, block: str, message: str) -> None:
        self.block_notes[block] = message
        logger.warning("Block %s unavailable: %s", block, message)

    def mark_unavailable(self, outcome: str) -> None:
        if outcome not in self.unavailable_outcomes:
            self.unavailable_outcomes.append(outcome)

    def record_problem_error(self, condition: str) -> None:
        self.problem_errors[condition] = (
            self.problem_errors.get(condition, 0) + 1
        )

    def guard(self, block: str, thunk: Callable[[], T]) -> T | None:
        """Run one analysis block; skip it on failure, keep the rest."""
        try:
            return thunk()
        except Exception as exc:
            self.note(block, str(exc))
            return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "block_notes": self.block_notes,
            "unavailable_outcomes": self.unavailable_outcomes,
            "problem_errors": self.problem_errors,
        }