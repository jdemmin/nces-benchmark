"""Determinism and stability guarantees of split_learning_problems."""

from __future__ import annotations

import random

import pytest
from lp.determinism.conftest import (
    make_hardness,
    make_problem,
    make_problems,
)

from src.data.lp import (
    _get_strata_key,
    _split_score,
    split_learning_problems,
)


def ids(problems) -> set[str]:
    return {p.id for p in problems}


class TestBasicInvariants:
    def test_train_and_test_are_disjoint(self, population):
        split = split_learning_problems(population, seed=1)
        assert not (ids(split["train"]) & ids(split["test"]))

    def test_split_is_a_partition(self, population):
        split = split_learning_problems(population, seed=1)
        assert ids(split["train"]) | ids(split["test"]) == ids(population)
        assert len(split["train"]) + len(split["test"]) == len(population)

    def test_no_duplicates_within_a_split(self, population):
        split = split_learning_problems(population, seed=1)
        for name in ("train", "test"):
            assert len(ids(split[name])) == len(split[name])

    def test_both_splits_are_non_empty(self, population):
        split = split_learning_problems(population, seed=1)
        assert split["train"]
        assert split["test"]

    def test_ratio_is_approximately_honoured(self, population):
        split = split_learning_problems(population, seed=1, ratios=(0.8, 0.2))
        frac = len(split["train"]) / len(population)
        # Per-stratum rounding means this cannot be exact; 12 strata-ish
        # rounding slack on 60 problems.
        assert 0.72 <= frac <= 0.90

    def test_minimum_two_problems_required(self):
        with pytest.raises(ValueError, match="At least two"):
            split_learning_problems([make_problem("A")], seed=1)

    @pytest.mark.parametrize(
        "ratios", [(0.0, 1.0), (1.0, 0.0), (0.5, 0.4), (0.7, 0.7), (-0.2, 1.2)]
    )
    def test_invalid_ratios_rejected(self, population, ratios):
        with pytest.raises(ValueError, match="Invalid ratios"):
            split_learning_problems(population, seed=1, ratios=ratios)


class TestReproducibility:
    def test_same_seed_same_split(self, population):
        a = split_learning_problems(population, seed=7)
        b = split_learning_problems(population, seed=7)
        assert ids(a["test"]) == ids(b["test"])
        assert ids(a["train"]) == ids(b["train"])

    def test_output_order_is_deterministic(self, population):
        a = split_learning_problems(population, seed=7)
        b = split_learning_problems(population, seed=7)
        for name in ("train", "test"):
            assert [p.id for p in a[name]] == [p.id for p in b[name]]

    def test_output_is_sorted_by_id(self, population):
        split = split_learning_problems(population, seed=7)
        for name in ("train", "test"):
            assert [p.id for p in split[name]] == sorted(p.id for p in split[name])

    def test_different_seeds_give_different_splits(self, population):
        a = split_learning_problems(population, seed=1)
        b = split_learning_problems(population, seed=2)
        assert ids(a["test"]) != ids(b["test"])

    def test_unaffected_by_global_rng_state(self, population):
        random.seed(1234)
        a = split_learning_problems(population, seed=7)
        for _ in range(1000):
            random.random()
        b = split_learning_problems(population, seed=7)
        assert ids(a["test"]) == ids(b["test"])

    def test_stable_across_processes(self, population, tmp_path):
        """Hash-based scoring must not depend on PYTHONHASHSEED."""
        import json
        import subprocess
        import sys
        import textwrap

        payload = tmp_path / "problems.json"
        payload.write_text(
            json.dumps({"0": [p.to_dict() for p in population]}), encoding="utf-8"
        )
        script = textwrap.dedent(
            f"""
            import json
            from pathlib import Path
            from src.data.lp import load_learning_problems, split_learning_problems
            problems = load_learning_problems(Path({str(payload)!r}))
            split = split_learning_problems(problems, seed=7)
            print(json.dumps(sorted(p.id for p in split["test"])))
            """
        )
        outputs = []
        for hashseed in ("0", "1", "42"):
            result = subprocess.run(
                [sys.executable, "-c", script],
                capture_output=True,
                text=True,
                check=True,
                env={"PYTHONHASHSEED": hashseed, "PATH": "/usr/bin:/bin"},
            )
            outputs.append(result.stdout.strip())
        assert len(set(outputs)) == 1, "split varies with PYTHONHASHSEED"


class TestPermutationInvariance:
    """Input ordering must not influence membership."""

    @pytest.mark.parametrize("shuffle_seed", [1, 2, 3, 99, 12345])
    def test_membership_invariant_under_input_permutation(
        self, population, shuffle_seed
    ):
        reference = split_learning_problems(population, seed=7)
        shuffled = list(population)
        random.Random(shuffle_seed).shuffle(shuffled)
        actual = split_learning_problems(shuffled, seed=7)
        assert ids(actual["test"]) == ids(reference["test"])
        assert ids(actual["train"]) == ids(reference["train"])

    def test_reversed_input_gives_same_split(self, population):
        reference = split_learning_problems(population, seed=7)
        actual = split_learning_problems(list(reversed(population)), seed=7)
        assert ids(actual["test"]) == ids(reference["test"])


class TestCrossRunLeakage:
    """The regression tests for the observed test→train leakage."""

    def test_new_stratum_does_not_move_existing_problems(self, population):
        reference = split_learning_problems(population, seed=7)
        grown = population + [make_problem("BrandNew", dl_length=999)]
        actual = split_learning_problems(grown, seed=7)

        leaked = ids(reference["test"]) & ids(actual["train"])
        assert not leaked, f"{len(leaked)} test problems leaked into train"

    def test_appending_to_one_stratum_does_not_perturb_others(self, population):
        """Adding to dl_length=1 must leave the dl_length=10 stratum intact."""
        reference = split_learning_problems(population, seed=7)
        ref_len10 = {p.id for p in reference["test"] if p.complexity.dl_length == 10}

        grown = population + [
            make_problem(f"Extra{i}", dl_length=1) for i in range(5)
        ]
        actual = split_learning_problems(grown, seed=7)
        act_len10 = {p.id for p in actual["test"] if p.complexity.dl_length == 10}

        assert ref_len10 == act_len10

    def test_removing_a_problem_does_not_flip_others(self, population):
        reference = split_learning_problems(population, seed=7)
        dropped = population[0]
        shrunk = [p for p in population if p.id != dropped.id]
        actual = split_learning_problems(shrunk, seed=7)

        expected_test = ids(reference["test"]) - {dropped.id}
        leaked = expected_test & ids(actual["train"])
        assert not leaked, f"{len(leaked)} problems flipped test→train on removal"

    def test_split_depends_only_on_problem_ids(self, population):
        """Same ids + different non-id content ⇒ same partition by id."""
        reference = split_learning_problems(population, seed=7)
        # Re-annotate every problem's complexity; ids are unchanged.

        mutated = [
            p.annotate_hardness(p.complexity, make_hardness(extension_size=99))
            for p in population
        ]
        actual = split_learning_problems(mutated, seed=7)
        assert ids(actual["test"]) == ids(reference["test"])


class TestStratification:
    def test_every_stratum_contributes_to_train(self, population):
        split = split_learning_problems(population, seed=7)
        all_keys = {_get_strata_key("dl_length", p) for p in population}
        train_keys = {_get_strata_key("dl_length", p) for p in split["train"]}
        assert train_keys == all_keys

    def test_difficulty_mix_is_comparable(self, population):
        """Mean dl_length should not diverge wildly between splits."""
        split = split_learning_problems(population, seed=7)
        mean = lambda ps: sum(p.complexity.dl_length for p in ps) / len(ps)
        assert abs(mean(split["train"]) - mean(split["test"])) < 1.5

    def test_stratify_by_none_falls_back_to_dl_length(self, population):
        split = split_learning_problems(population, seed=7, stratify_by=None)
        assert ids(split["train"]) | ids(split["test"]) == ids(population)

    def test_unknown_stratify_field_raises(self, population):
        with pytest.raises(ValueError, match="Unknown stratify_by"):
            split_learning_problems(population, seed=7, stratify_by="nonsense")

    def test_unannotated_hardness_stratification_raises(self, population):
        with pytest.raises(ValueError, match="not hardness-annotated"):
            split_learning_problems(
                population, seed=7, stratify_by="atomic_baseline_f1"
            )

    def test_annotated_hardness_stratification_works(self, annotated_population):
        split = split_learning_problems(
            annotated_population, seed=7, stratify_by="atomic_baseline_f1"
        )
        assert split["train"] and split["test"]
        assert not (ids(split["train"]) & ids(split["test"]))

    def test_continuous_field_is_binned_not_exploded(self, annotated_population):
        """Float strata must bucket, else every stratum has size 1."""
        keys = {
            _get_strata_key("atomic_baseline_f1", p) for p in annotated_population
        }
        assert len(keys) <= 5, f"continuous field produced {len(keys)} strata"

    def test_partial_annotation_is_rejected(self, population, annotated_population):
        mixed = population[:10] + annotated_population[10:20]
        with pytest.raises(ValueError, match="not hardness-annotated"):
            split_learning_problems(
                mixed, seed=7, stratify_by="atomic_baseline_f1"
            )

    def test_numeric_strata_keys_sort_numerically(self):
        from src.data.lp import _key_order

        keys = ["10", "2", "3", "1"]
        assert sorted(keys, key=_key_order) == ["1", "2", "3", "10"]

    def test_mixed_numeric_and_string_keys_sort_stably(self):
        from src.data.lp import _key_order

        keys = ["10", "q3", "2", "True", "q1"]
        ordered = sorted(keys, key=_key_order)
        assert ordered[:2] == ["2", "10"]
        assert ordered[2:] == sorted(["q3", "True", "q1"])


class TestSmallAndDegenerateInputs:
    def test_two_problems_one_each(self):
        problems = [make_problem("A", dl_length=1), make_problem("B", dl_length=1)]
        split = split_learning_problems(problems, seed=7)
        assert len(split["train"]) == 1
        assert len(split["test"]) == 1

    def test_singleton_strata_still_yield_a_test_set(self):
        """Every stratum of size 1 goes to train ⇒ rescue branch must fire."""
        problems = [make_problem(f"C{i}", dl_length=i) for i in range(5)]
        split = split_learning_problems(problems, seed=7)
        assert split["test"], "rescue branch failed to populate test"
        assert not (ids(split["train"]) & ids(split["test"]))
        assert ids(split["train"]) | ids(split["test"]) == ids(problems)

    def test_rescue_branch_is_deterministic(self):
        problems = [make_problem(f"C{i}", dl_length=i) for i in range(5)]
        a = split_learning_problems(problems, seed=7)
        b = split_learning_problems(problems, seed=7)
        assert ids(a["test"]) == ids(b["test"])

    def test_all_problems_in_one_stratum(self):
        problems = make_problems(20, lengths=(4,))
        split = split_learning_problems(problems, seed=7)
        assert len(split["test"]) == 4
        assert len(split["train"]) == 16


class TestSplitScore:
    def test_score_in_unit_interval(self):
        for i in range(500):
            assert 0.0 <= _split_score(1, f"lp_{i}") < 1.0

    def test_score_depends_on_seed(self):
        assert _split_score(1, "lp_abc") != _split_score(2, "lp_abc")

    def test_score_depends_on_id(self):
        assert _split_score(1, "lp_abc") != _split_score(1, "lp_abd")

    def test_score_is_pure(self):
        assert _split_score(3, "lp_x") == _split_score(3, "lp_x")

    def test_scores_are_roughly_uniform(self):
        scores = [_split_score(1, f"lp_{i:05d}") for i in range(5000)]
        buckets = [0] * 10
        for s in scores:
            buckets[min(int(s * 10), 9)] += 1
        assert all(350 < b < 650 for b in buckets), buckets