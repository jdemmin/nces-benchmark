# tests/eval/test_failures.py
"""Tests for the shortfall ledger.

Central contract: ``guard`` never propagates an ordinary exception, and a
failed block leaves the rest of the report intact.
"""

from __future__ import annotations

import logging

import pytest

from src.eval.failures import ShortfallLedger


@pytest.fixture()
def ledger() -> ShortfallLedger:
    return ShortfallLedger()


def test_fresh_ledger_is_empty(ledger):
    assert ledger.to_dict() == {
        "block_notes": {},
        "unavailable_outcomes": [],
        "problem_errors": {},
    }


def test_default_containers_are_not_shared():
    """Guards against a mutable-default regression."""
    first, second = ShortfallLedger(), ShortfallLedger()
    first.note("mechanism", "boom")
    first.mark_unavailable("abl")
    first.record_problem_error("treated")
    assert second.block_notes == {}
    assert second.unavailable_outcomes == []
    assert second.problem_errors == {}


def test_note_records_and_logs(ledger, caplog):
    with caplog.at_level(logging.WARNING, logger="src.eval.failures"):
        ledger.note("trend", "too few levels")
    assert ledger.block_notes == {"trend": "too few levels"}
    assert "Block trend unavailable: too few levels" in caplog.text


def test_note_overwrites_same_block(ledger):
    ledger.note("trend", "first")
    ledger.note("trend", "second")
    assert ledger.block_notes == {"trend": "second"}


def test_mark_unavailable_is_idempotent_and_ordered(ledger):
    for outcome in ("abl", "recall", "abl", "precision", "recall"):
        ledger.mark_unavailable(outcome)
    assert ledger.unavailable_outcomes == ["abl", "recall", "precision"]


def test_record_problem_error_counts_per_condition(ledger):
    for condition in ("treated", "treated", "control"):
        ledger.record_problem_error(condition)
    assert ledger.problem_errors == {"treated": 2, "control": 1}


def test_guard_returns_value_on_success(ledger):
    assert ledger.guard("mechanism", lambda: 42) == 42
    assert ledger.block_notes == {}


def test_guard_returns_falsy_values_untouched(ledger):
    """0, None and [] are legitimate results, not failures."""
    assert ledger.guard("a", lambda: 0) == 0
    assert ledger.guard("b", lambda: []) == []
    assert ledger.guard("c", lambda: None) is None
    assert ledger.block_notes == {}


def test_guard_swallows_exception_and_notes_it(ledger):
    result = ledger.guard("breakdown", lambda: 1 / 0)
    assert result is None
    assert "breakdown" in ledger.block_notes
    assert "division by zero" in ledger.block_notes["breakdown"]


def test_guard_records_empty_message_exception(ledger):
    def thunk():
        raise ValueError

    assert ledger.guard("mechanism", thunk) is None
    assert ledger.block_notes["mechanism"] == ""


def test_guard_isolates_blocks(ledger):
    """A failed layer must not prevent later layers from producing."""
    first = ledger.guard("mechanism", lambda: "ok")
    failed = ledger.guard("trend", lambda: (_ for _ in ()).throw(KeyError("d")))
    third = ledger.guard("breakdown", lambda: "also ok")
    assert (first, failed, third) == ("ok", None, "also ok")
    assert set(ledger.block_notes) == {"trend"}


@pytest.mark.parametrize("exc", [KeyboardInterrupt, SystemExit])
def test_guard_does_not_swallow_base_exceptions(ledger, exc):
    """Interrupts must abort the run, not be filed as a note."""

    def thunk():
        raise exc()

    with pytest.raises(exc):
        ledger.guard("mechanism", thunk)
    assert ledger.block_notes == {}


def test_guard_catches_memory_error(ledger):
    """Intentional: MemoryError degrades the block under this design."""

    def thunk():
        raise MemoryError("resample too large")

    assert ledger.guard("trend", thunk) is None
    assert "resample too large" in ledger.block_notes["trend"]


def test_to_dict_reflects_accumulated_state(ledger):
    ledger.note("trend", "unidentified")
    ledger.mark_unavailable("abl")
    ledger.record_problem_error("treated")
    assert ledger.to_dict() == {
        "block_notes": {"trend": "unidentified"},
        "unavailable_outcomes": ["abl"],
        "problem_errors": {"treated": 1},
    }


def test_to_dict_is_json_serialisable(ledger):
    import json

    ledger.note("trend", "unidentified")
    ledger.mark_unavailable("abl")
    ledger.record_problem_error("control")
    assert json.loads(json.dumps(ledger.to_dict())) == ledger.to_dict()


def test_to_dict_returns_defensive_copies(ledger):
    ledger.mark_unavailable("abl")
    payload = ledger.to_dict()
    payload["unavailable_outcomes"].append("injected")
    assert ledger.unavailable_outcomes == ["abl"]