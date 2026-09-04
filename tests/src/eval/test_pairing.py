"""Unit tests for src/eval/pairing.py."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.eval.pairing import (
    CONTROL,
    FALLBACK_OUTCOME,
    OUTCOMES,
    PRIMARY_OUTCOME,
    Observation,
    PairedDesign,
    PairedDesignImpossible,
    assemble,
    dataframe_records,
    observations_to_frame,
    primary_outcome,
    resample_seed_clusters,
)

KB = "family"
TREATMENT = "kge"


def make_observation(
    *,
    condition: str = TREATMENT,
    knowledge_base: str = KB,
    seed: int = 0,
    problem_id: str = "p1",
    hypothesis: str = "H",
    depth: int | None = 2,
    dl_length: int | None = 5,
    expressivity: str | None = "ALC",
    extension_ratio: float | None = 0.25,
    atomic_baseline_f1: float | None = 0.4,
    target_extension_size: int | None = 12,
    **values: float | None,
) -> Observation:
    """An Observation with every outcome defaulted to None."""
    filled: dict[str, float | None] = {outcome: None for outcome in OUTCOMES}
    filled.update(values)
    return Observation(
        condition=condition,
        knowledge_base=knowledge_base,
        seed=seed,
        problem_id=problem_id,
        hypothesis=hypothesis,
        depth=depth,
        dl_length=dl_length,
        expressivity=expressivity,
        extension_ratio=extension_ratio,
        atomic_baseline_f1=atomic_baseline_f1,
        target_extension_size=target_extension_size,
        values=filled,
    )


def make_pair(
    seed: int,
    problem_id: str,
    treated: float,
    control: float,
    outcome: str = PRIMARY_OUTCOME,
) -> list[Observation]:
    """A treated/control pair differing only in condition and value."""
    return [
        make_observation(
            condition=TREATMENT,
            seed=seed,
            problem_id=problem_id,
            hypothesis=f"H_t_{problem_id}",
            **{outcome: treated},
        ),
        make_observation(
            condition=CONTROL,
            seed=seed,
            problem_id=problem_id,
            hypothesis=f"H_c_{problem_id}",
            **{outcome: control},
        ),
    ]


def assemble_from(observations: list[Observation], **kwargs) -> PairedDesign:
    return assemble(
        observations_to_frame(observations),
        condition=TREATMENT,
        knowledge_base=KB,
        **kwargs,
    )


class TestDataframeRecords:
    def test_empty_frame_yields_empty_list(self):
        assert dataframe_records(pd.DataFrame()) == []

    def test_frame_with_columns_but_no_rows_yields_empty_list(self):
        frame = pd.DataFrame({"a": pd.Series(dtype="int64")})
        assert dataframe_records(frame) == []

    def test_numpy_scalars_become_json_serializable_natives(self):
        frame = pd.DataFrame(
            {
                "i": np.array([1], dtype=np.int64),
                "f": np.array([1.5], dtype=np.float32),
                "b": np.array([True]),
            }
        )
        records = dataframe_records(frame)
        json.dumps(records)  # must not raise
        assert type(records[0]["i"]) is int
        assert type(records[0]["f"]) is float
        assert type(records[0]["b"]) is bool

    def test_nan_becomes_none(self):
        frame = pd.DataFrame({"x": [1.0, np.nan]})
        assert dataframe_records(frame) == [{"x": 1.0}, {"x": None}]

    def test_infinity_is_silently_nulled(self):
        """Documents lossy behaviour: inf is indistinguishable from missing."""
        frame = pd.DataFrame({"x": [np.inf, -np.inf]})
        assert dataframe_records(frame) == [{"x": None}, {"x": None}]

    def test_none_in_object_column_becomes_none(self):
        frame = pd.DataFrame({"x": ["a", None]})
        assert dataframe_records(frame) == [{"x": "a"}, {"x": None}]


class TestObservationsToFrame:
    def test_empty_iterable_yields_empty_frame(self):
        frame = observations_to_frame([])
        assert frame.empty

    def test_every_outcome_becomes_a_column(self):
        frame = observations_to_frame([make_observation(abl=0.5)])
        for outcome in OUTCOMES:
            assert outcome in frame

    def test_metadata_is_preserved(self):
        obs = make_observation(depth=3, expressivity="EL", abl=0.1)
        row = observations_to_frame([obs]).iloc[0]
        assert row["condition"] == TREATMENT
        assert row["knowledge_base"] == KB
        assert row["problem_id"] == "p1"
        assert row["expressivity"] == "EL"
        assert row["depth"] == 3

    def test_semantic_equivalence_is_float_even_when_all_missing(self):
        frame = observations_to_frame([make_observation()])
        assert frame["semantic_equivalence"].dtype == np.dtype("float64")

    def test_boolean_semantic_equivalence_is_coerced_to_float(self):
        frame = observations_to_frame(
            [make_observation(semantic_equivalence=True)]
        )
        assert frame["semantic_equivalence"].dtype == np.dtype("float64")
        assert frame["semantic_equivalence"].iloc[0] == 1.0

    @pytest.mark.xfail(
        reason="BUG: only semantic_equivalence is cast; other all-None "
        "outcome columns stay object dtype and break arithmetic in assemble()",
        strict=True,
    )
    def test_all_none_outcome_columns_are_numeric(self):
        frame = observations_to_frame([make_observation(f1=0.5)])
        # abl was never supplied, so it is object dtype, not float64.
        assert frame[PRIMARY_OUTCOME].dtype == np.dtype("float64")


class TestAssembleGuards:
    def test_missing_treatment_raises(self):
        observations = [make_observation(condition=CONTROL, abl=0.5)]
        with pytest.raises(PairedDesignImpossible, match="No runs"):
            assemble_from(observations)

    def test_missing_control_raises(self):
        observations = [make_observation(condition=TREATMENT, abl=0.5)]
        with pytest.raises(PairedDesignImpossible, match="No runs"):
            assemble_from(observations)

    def test_disjoint_seeds_raises(self):
        observations = [
            make_observation(condition=TREATMENT, seed=0, abl=0.5),
            make_observation(condition=CONTROL, seed=1, abl=0.2),
        ]
        with pytest.raises(PairedDesignImpossible, match="No seed carries"):
            assemble_from(observations)

    def test_other_knowledge_bases_are_excluded(self):
        observations = [
            *make_pair(0, "p1", 0.8, 0.3),
            make_observation(
                condition=TREATMENT, knowledge_base="other", abl=1.0
            ),
            make_observation(
                condition=CONTROL, knowledge_base="other", abl=0.0
            ),
        ]
        design = assemble_from(observations)
        assert design.n_observations == 1
        assert design.knowledge_base == KB

    def test_wrong_knowledge_base_only_raises(self):
        observations = [
            make_observation(
                condition=TREATMENT, knowledge_base="other", abl=1.0
            ),
            make_observation(
                condition=CONTROL, knowledge_base="other", abl=0.0
            ),
        ]
        with pytest.raises(PairedDesignImpossible):
            assemble_from(observations)


class TestAssembleDifferences:
    def test_difference_is_treated_minus_control(self):
        design = assemble_from(make_pair(0, "p1", 0.8, 0.3))
        row = design.frame.iloc[0]
        assert row[f"{PRIMARY_OUTCOME}_treated"] == pytest.approx(0.8)
        assert row[f"{PRIMARY_OUTCOME}_control"] == pytest.approx(0.3)
        assert row[f"d_{PRIMARY_OUTCOME}"] == pytest.approx(0.5)

    def test_hypotheses_are_kept_from_both_arms(self):
        design = assemble_from(make_pair(0, "p1", 0.8, 0.3))
        row = design.frame.iloc[0]
        assert row["hypothesis_treated"] == "H_t_p1"
        assert row["hypothesis_control"] == "H_c_p1"

    def test_metadata_columns_survive_the_merge_unsuffixed(self):
        design = assemble_from(make_pair(0, "p1", 0.8, 0.3))
        for column in (
            "depth",
            "dl_length",
            "expressivity",
            "extension_ratio",
            "atomic_baseline_f1",
            "target_extension_size",
        ):
            assert column in design.frame

    def test_seeds_are_the_sorted_intersection(self):
        observations = [
            *make_pair(2, "p1", 0.8, 0.3),
            *make_pair(0, "p1", 0.7, 0.4),
            make_observation(condition=TREATMENT, seed=9, abl=0.5),
        ]
        design = assemble_from(observations)
        assert design.seeds == (0, 2)

    def test_treated_only_problem_is_reported_unpaired(self):
        observations = [
            *make_pair(0, "p1", 0.8, 0.3),
            make_observation(
                condition=TREATMENT, seed=0, problem_id="p2", abl=0.9
            ),
        ]
        design = assemble_from(observations)
        assert design.unpaired_problem_ids == ("p2",)
        assert design.n_observations == 1
        assert any("unpaired" in note for note in design.notes)

    def test_control_only_problem_is_reported_unpaired(self):
        observations = [
            *make_pair(0, "p1", 0.8, 0.3),
            make_observation(
                condition=CONTROL, seed=0, problem_id="p2", abl=0.1
            ),
        ]
        design = assemble_from(observations)
        assert design.unpaired_problem_ids == ("p2",)

    def test_no_notes_when_design_is_complete(self):
        design = assemble_from(make_pair(0, "p1", 0.8, 0.3))
        assert design.notes == []
        assert design.unpaired_problem_ids == ()

    def test_failures_are_carried_through(self):
        design = assemble_from(
            make_pair(0, "p1", 0.8, 0.3), failures={"timeout": 3}
        )
        assert design.failures == {"timeout": 3}

    def test_failures_default_to_empty_mapping(self):
        design = assemble_from(make_pair(0, "p1", 0.8, 0.3))
        assert design.failures == {}

    def test_missing_value_on_one_side_yields_nan_difference(self):
        observations = [
            *make_pair(0, "p1", 0.8, 0.3),
            make_observation(
                condition=TREATMENT, seed=0, problem_id="p2", abl=0.9
            ),
            make_observation(
                condition=CONTROL, seed=0, problem_id="p2", abl=None
            ),
        ]
        design = assemble_from(observations)
        # p2 is *paired* (both rows exist) but its difference is undefined.
        assert design.n_problems == 2
        differences = design.frame.set_index("problem_id")[
            f"d_{PRIMARY_OUTCOME}"
        ]
        assert differences["p1"] == pytest.approx(0.5)
        assert pd.isna(differences["p2"])


class TestAssembleUnpairedAccounting:
    @pytest.mark.xfail(
        reason="BUG: unpaired ids are pooled across seeds, so a problem "
        "paired in one seed masks its unpaired rows in another",
        strict=True,
    )
    def test_partially_paired_problem_is_reported(self):
        observations = [
            *make_pair(0, "p1", 0.8, 0.3),
            # seed 1 has the treated arm only.
            make_observation(
                condition=TREATMENT, seed=1, problem_id="p1", abl=0.9
            ),
            make_observation(
                condition=CONTROL, seed=1, problem_id="p2", abl=0.1
            ),
            make_observation(
                condition=TREATMENT, seed=1, problem_id="p2", abl=0.4
            ),
        ]
        design = assemble_from(observations)
        # p1's seed-1 row was silently dropped by the inner join.
        assert design.n_observations == 3
        assert "p1" in design.unpaired_problem_ids


class TestAssembleDuplicates:
    @pytest.mark.xfail(
        reason="BUG: no uniqueness check on (seed, problem_id); duplicate "
        "artifacts fan out into a Cartesian product and overweight a problem",
        strict=True,
    )
    def test_duplicate_rows_do_not_fan_out(self):
        observations = [
            *make_pair(0, "p1", 0.8, 0.3),
            *make_pair(0, "p1", 0.8, 0.3),
        ]
        design = assemble_from(observations)
        assert design.n_observations <= 2


class TestPrimarySubstitution:
    def test_primary_outcome_used_when_available(self):
        design = assemble_from(make_pair(0, "p1", 0.8, 0.3))
        assert design.substituted_primary is False
        assert primary_outcome(design) == PRIMARY_OUTCOME

    def test_fallback_used_when_primary_all_missing(self):
        observations = [
            *make_pair(0, "p1", 0.8, 0.3, outcome=FALLBACK_OUTCOME),
            # keep abl float-typed but entirely NaN
            make_observation(
                condition=TREATMENT, seed=1, problem_id="p9", abl=np.nan, f1=0.5
            ),
            make_observation(
                condition=CONTROL, seed=1, problem_id="p9", abl=np.nan, f1=0.2
            ),
        ]
        design = assemble_from(observations)
        assert design.substituted_primary is True
        assert primary_outcome(design) == FALLBACK_OUTCOME
        assert any("hardness annotation" in note for note in design.notes)

    @pytest.mark.xfail(
        reason="BUG: all-None abl leaves an object-dtype column, so the "
        "treated - control subtraction raises instead of degrading",
        strict=True,
    )
    def test_absent_primary_degrades_instead_of_raising(self):
        design = assemble_from(
            make_pair(0, "p1", 0.8, 0.3, outcome=FALLBACK_OUTCOME)
        )
        assert design.substituted_primary is True

    @pytest.mark.xfail(
        reason="BUG: substitution does not verify the fallback exists, so "
        "primary_outcome() can name an entirely empty outcome",
        strict=True,
    )
    def test_substitution_requires_a_usable_fallback(self):
        observations = [
            *make_pair(0, "p1", 0.8, 0.3, outcome="accuracy"),
            make_observation(
                condition=TREATMENT,
                seed=1,
                problem_id="p9",
                abl=np.nan,
                f1=np.nan,
                accuracy=0.5,
            ),
            make_observation(
                condition=CONTROL,
                seed=1,
                problem_id="p9",
                abl=np.nan,
                f1=np.nan,
                accuracy=0.2,
            ),
        ]
        design = assemble_from(observations)
        assert primary_outcome(design) in design.available_outcomes()


class TestEmptyIntersection:
    def _design(self) -> PairedDesign:
        observations = [
            make_observation(
                condition=TREATMENT, seed=0, problem_id="p1", abl=0.8
            ),
            make_observation(
                condition=CONTROL, seed=0, problem_id="p2", abl=0.3
            ),
        ]
        return assemble_from(observations)

    def test_shared_seed_with_no_shared_problems_yields_empty_design(self):
        design = self._design()
        assert design.n_observations == 0
        assert design.n_problems == 0
        assert set(design.unpaired_problem_ids) == {"p1", "p2"}

    @pytest.mark.xfail(
        reason="BUG: total pairing failure is misreported as a missing "
        "hardness annotation rather than raising PairedDesignImpossible",
        strict=True,
    )
    def test_total_pairing_failure_is_diagnosed_honestly(self):
        design = self._design()
        assert not any("hardness annotation" in n for n in design.notes)


class TestAvailableOutcomes:
    def test_lists_only_outcomes_with_data_in_canonical_order(self):
        observations = [
            *make_pair(0, "p1", 0.8, 0.3, outcome=PRIMARY_OUTCOME),
            make_observation(
                condition=TREATMENT, seed=0, problem_id="p2", abl=0.1, recall=0.9
            ),
            make_observation(
                condition=CONTROL, seed=0, problem_id="p2", abl=0.1, recall=0.4
            ),
        ]
        design = assemble_from(observations)
        assert design.available_outcomes() == (PRIMARY_OUTCOME, "recall")

    def test_empty_when_no_outcome_has_data(self):
        observations = [
            *make_pair(0, "p1", np.nan, np.nan),
        ]
        design = assemble_from(observations)
        assert design.available_outcomes() == ()


class TestCollapse:
    def test_averages_a_problem_across_its_seeds(self):
        observations = [
            *make_pair(0, "p1", 0.8, 0.3),  # +0.5
            *make_pair(1, "p1", 0.5, 0.4),  # +0.1
        ]
        collapsed = assemble_from(observations).collapse(PRIMARY_OUTCOME)
        assert collapsed.index.tolist() == ["p1"]
        assert collapsed["p1"] == pytest.approx(0.3)

    def test_index_is_sorted_by_problem_id(self):
        observations = [
            *make_pair(0, "pc", 0.9, 0.1),
            *make_pair(0, "pa", 0.8, 0.3),
            *make_pair(0, "pb", 0.7, 0.5),
        ]
        collapsed = assemble_from(observations).collapse(PRIMARY_OUTCOME)
        assert collapsed.index.tolist() == ["pa", "pb", "pc"]

    def test_nan_differences_are_dropped_before_averaging(self):
        observations = [
            *make_pair(0, "p1", 0.8, 0.3),  # +0.5
            make_observation(
                condition=TREATMENT, seed=1, problem_id="p1", abl=np.nan
            ),
            make_observation(
                condition=CONTROL, seed=1, problem_id="p1", abl=0.4
            ),
        ]
        collapsed = assemble_from(observations).collapse(PRIMARY_OUTCOME)
        assert collapsed["p1"] == pytest.approx(0.5)

    def test_problem_with_only_nan_differences_is_absent(self):
        observations = [
            *make_pair(0, "p1", 0.8, 0.3),
            make_observation(
                condition=TREATMENT, seed=0, problem_id="p2", abl=np.nan
            ),
            make_observation(
                condition=CONTROL, seed=0, problem_id="p2", abl=np.nan
            ),
        ]
        collapsed = assemble_from(observations).collapse(PRIMARY_OUTCOME)
        assert collapsed.index.tolist() == ["p1"]

    def test_unknown_outcome_yields_empty_float_series(self):
        design = assemble_from(make_pair(0, "p1", 0.8, 0.3))
        collapsed = design.collapse("not_an_outcome")
        assert collapsed.empty
        assert collapsed.dtype == np.dtype("float64")


class TestSeedMeans:
    def test_averages_within_each_seed(self):
        observations = [
            *make_pair(0, "p1", 0.8, 0.3),  # +0.5
            *make_pair(0, "p2", 0.6, 0.5),  # +0.1
            *make_pair(1, "p1", 0.4, 0.4),  # 0.0
        ]
        means = assemble_from(observations).seed_means(PRIMARY_OUTCOME)
        assert means.index.tolist() == [0, 1]
        assert means[0] == pytest.approx(0.3)
        assert means[1] == pytest.approx(0.0)

    def test_unknown_outcome_yields_empty_float_series(self):
        design = assemble_from(make_pair(0, "p1", 0.8, 0.3))
        assert design.seed_means("nope").empty

    def test_collapse_and_seed_means_agree_on_the_grand_mean(self):
        observations = [
            *make_pair(0, "p1", 0.8, 0.3),
            *make_pair(0, "p2", 0.6, 0.5),
            *make_pair(1, "p1", 0.4, 0.4),
            *make_pair(1, "p2", 0.9, 0.2),
        ]
        design = assemble_from(observations)
        # Balanced design: both collapses must give the same grand mean.
        assert design.collapse(PRIMARY_OUTCOME).mean() == pytest.approx(
            design.seed_means(PRIMARY_OUTCOME).mean()
        )


class TestResampleSeedClusters:
    def _frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "seed": [0, 0, 1, 1, 2],
                "problem_id": ["p1", "p2", "p1", "p2", "p1"],
                "d_abl": [0.1, 0.2, 0.3, 0.4, 0.5],
            }
        )

    def test_draws_the_same_number_of_seeds(self):
        rng = np.random.default_rng(0)
        resampled = resample_seed_clusters(self._frame(), [0, 1, 2], rng)
        assert resampled["seed"].nunique() <= 3
        assert len(resampled) >= 1

    def test_is_deterministic_for_a_fixed_seed(self):
        frame = self._frame()
        a = resample_seed_clusters(frame, [0, 1, 2], np.random.default_rng(7))
        b = resample_seed_clusters(frame, [0, 1, 2], np.random.default_rng(7))
        pd.testing.assert_frame_equal(a, b)

    def test_whole_clusters_are_kept_intact(self):
        rng = np.random.default_rng(3)
        resampled = resample_seed_clusters(self._frame(), [0, 1, 2], rng)
        # Every drawn seed contributes all of its rows, never a subset.
        for seed, group in resampled.groupby("seed"):
            expected = (self._frame()["seed"] == seed).sum()
            assert len(group) % expected == 0

    def test_single_seed_resample_reproduces_that_cluster(self):
        rng = np.random.default_rng(0)
        frame = self._frame()
        resampled = resample_seed_clusters(frame, [1], rng)
        pd.testing.assert_frame_equal(
            resampled.reset_index(drop=True),
            frame[frame["seed"] == 1].reset_index(drop=True),
        )

    def test_index_is_reset(self):
        rng = np.random.default_rng(1)
        resampled = resample_seed_clusters(self._frame(), [0, 1, 2], rng)
        assert resampled.index.tolist() == list(range(len(resampled)))

    def test_repeated_draws_duplicate_rows(self):
        """A seed drawn twice must appear twice, or the interval is too tight."""
        rng = np.random.default_rng(0)
        drawn = np.random.default_rng(0).choice([0, 1, 2], size=3, replace=True)
        resampled = resample_seed_clusters(self._frame(), [0, 1, 2], rng)
        expected = sum(
            (self._frame()["seed"] == seed).sum() for seed in drawn
        )
        assert len(resampled) == expected

    def test_empty_seed_sequence_raises(self):
        rng = np.random.default_rng(0)
        with pytest.raises(ValueError):
            resample_seed_clusters(self._frame(), [], rng)

    def test_unknown_seed_silently_shrinks_the_resample(self):
        """Documents that seeds absent from the frame contribute no rows."""
        rng = np.random.default_rng(0)
        resampled = resample_seed_clusters(self._frame(), [99], rng)
        assert resampled.empty


class TestRoundTrip:
    def test_design_frame_is_json_serializable(self):
        design = assemble_from(
            [*make_pair(0, "p1", 0.8, 0.3), *make_pair(1, "p2", 0.6, 0.6)]
        )
        records = dataframe_records(design.frame)
        json.dumps(records)
        assert len(records) == 2

    def test_design_frame_omits_condition_and_knowledge_base(self):
        """Documents that provenance lives only on the dataclass."""
        design = assemble_from(make_pair(0, "p1", 0.8, 0.3))
        assert "condition" not in design.frame
        assert "knowledge_base" not in design.frame
        assert design.condition == TREATMENT