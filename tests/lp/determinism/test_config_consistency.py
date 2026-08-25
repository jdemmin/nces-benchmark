"""Configuration defaults must agree with the functions they feed."""

from __future__ import annotations

import inspect
import json

from src.config import ProjectSettings
from src.data.complexity import Complexity, Hardness
from src.data.lp import split_learning_problems


def test_stratify_by_default_matches_split_signature():
    """ProjectSettings and split_learning_problems must not disagree."""
    signature_default = inspect.signature(
        split_learning_problems
    ).parameters["stratify_by"].default
    assert ProjectSettings().stratify_by == signature_default


def test_stratify_by_json_fallback_matches_dataclass_default(tmp_path):
    """from_json's inline default must equal the field default."""
    path = tmp_path / "project_settings.json"
    path.write_text(json.dumps({"seeds": [1]}), encoding="utf-8")
    assert ProjectSettings.from_json(path).stratify_by == ProjectSettings().stratify_by


def test_default_stratify_by_is_a_real_field():
    field = ProjectSettings().stratify_by
    assert (
        field in Complexity.__dataclass_fields__
        or field in Hardness.__dataclass_fields__
    )