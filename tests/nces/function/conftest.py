# tests/nces/function/conftest.py
"""Shared fixtures.

The NCES tests install fake third-party modules into ``sys.modules``. Every
such insertion goes through ``monkeypatch.setitem``, which unwinds per test,
but the real packages must not have been imported *before* the fake is
installed — an already-imported ``ontolearn`` would be shadowed only for the
lazy import, not for module-level ones. Asserting the invariant here makes a
future violation fail loudly instead of mysteriously.
"""

from __future__ import annotations

import sys

import pytest


@pytest.fixture(autouse=True)
def _no_leaked_fakes():
    before = {
        name: sys.modules.get(name)
        for name in ("ontolearn", "owlapy", "torch", "numpy")
    }
    yield
    for name, module in before.items():
        current = sys.modules.get(name)
        assert current is module, (
            f"{name} was left replaced in sys.modules; use "
            f"monkeypatch.setitem so the patch unwinds"
        )