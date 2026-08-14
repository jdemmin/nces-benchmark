# tests/test_config.py
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.config import (
    DataGenerationSettings,
    EmbeddingSettings,
    NCESSettings,
    ProjectSettings,
)


def test_beyond_alc_selects_expressivity_label() -> None:
    assert (
        DataGenerationSettings(beyond_alc=True).refinement_operator_expressivity
        == "ALCHIQ(D)"
    )
    assert (
        DataGenerationSettings(beyond_alc=False).refinement_operator_expressivity
        == "ALC"
    )


def test_legacy_rho_field_still_parses(tmp_path: Path) -> None:
    path = tmp_path / "data_generation_settings.json"
    path.write_text(json.dumps({"rho": "ALCHIQD"}))
    assert DataGenerationSettings.from_json(path).beyond_alc is True


def test_downsample_required_below_full_expressivity() -> None:
    with pytest.raises(ValueError, match="downsample_refinements must be True"):
        DataGenerationSettings(
            refinement_expressivity=0.2, downsample_refinements=False
        )


def test_full_expressivity_allows_downsampling_off() -> None:
    settings = DataGenerationSettings(
        refinement_expressivity=1.0, downsample_refinements=False
    )
    assert settings.lpgen_kwargs(Path("k.owl"), Path("o"))[
        "downsample_refinements"
    ] is False


def test_max_pos_neg_examples_omitted_when_unset(tmp_path: Path) -> None:
    kwargs = DataGenerationSettings().lpgen_kwargs(
        tmp_path / "kb.owl", tmp_path / "out"
    )
    assert "max_pos_neg_examples_per_lp" not in kwargs


def test_lpgen_kwargs_use_upstream_spelling(tmp_path: Path) -> None:
    kwargs = DataGenerationSettings(num_rand_samples=7, max_child_len=5).lpgen_kwargs(
        tmp_path / "kb.owl", tmp_path / "out"
    )
    assert kwargs["max_num_lps"] == 7
    assert kwargs["max_child_length"] == 5
    assert "max_child_len" not in kwargs


def test_search_grid_covers_dim_and_batch() -> None:
    grid = EmbeddingSettings(embedding_dim=32, batch_size=64).search_grid()
    assert {(s.embedding_dim, s.batch_size) for s in grid} == {
        (32, 32),
        (32, 64),
        (64, 32),
        (64, 64),
    }


def test_unknown_dice_model_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown DICE model"):
        EmbeddingSettings(model_name="NotAModel")


def test_nces_learner_names_is_a_list() -> None:
    assert NCESSettings(learner_name="GRU").learner_names == ["GRU"]


def test_project_settings_rejects_unknown_condition() -> None:
    with pytest.raises(ValueError, match="Unknown embedding condition"):
        ProjectSettings(embedding_conditions=["dice", "bogus"])


def test_settings_round_trip_from_json(tmp_path: Path) -> None:
    path = tmp_path / "embedding_settings.json"
    path.write_text(
        json.dumps(
            {
                "model_name": "TransE",
                "embedding_dim": 16,
                "epochs": 2,
                "batch_size": 8,
            }
        )
    )
    settings = EmbeddingSettings.from_json(path)
    assert (settings.model_name, settings.embedding_dim) == ("TransE", 16)
