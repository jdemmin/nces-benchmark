# tests/eval/test_rq1.py
"""Unit tests for src.eval.rq1."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.eval.rq1 import (
    LinkSummary,
    RankingConcordance,
    condition_rankings,
    conditional_table,
    link_summary,
    quality_frame,
    ranking_concordance,
    sign_agreement,
)

# --------------------------------------------------------------------------
# Minimal stand-ins for the inference-layer records. Kept structural rather
# than importing the real ones so that a change to Interval/Diagnostics
# cannot make these tests fail for reasons unrelated to rq1.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class FakeInterval:
    low: float
    high: float

    @property
    def excludes_zero(self) -> bool:
        return (self.low > 0.0) or (self.high < 0.0)


@dataclass(frozen=True)
class FakeTest:
    p_value: float | None = None
    hodges_lehmann: float | None = None
    wins: int = 0
    losses: int = 0
    ties: int = 0


@dataclass(frozen=True)
class FakeContrast:
    condition: str
    knowledge_base: str
    estimate: float | None
    outcome: str = "abl"
    interval: FakeInterval | None = None
    test: FakeTest | None = None
    n_problems: int = 300
    n_seeds: int = 5
    substituted_primary: bool = False
    n_observations: int = 1500
    diagnostics: Any = None
    seed_spread: Any = None
    notes: list[str] = field(default_factory=list)


def contrast(condition: str, kb: str, estimate: float | None, **kw: Any):
    return FakeContrast(
        condition=condition, knowledge_base=kb, estimate=estimate, **kw
    )


# --------------------------------------------------------------------------
# conditional_table
# --------------------------------------------------------------------------

EXPECTED_TABLE_COLUMNS = {
    "condition",
    "outcome",
    "estimate",
    "ci_low",
    "ci_high",
    "excludes_zero",
    "p_value",
    "hodges_lehmann",
    "wins",
    "losses",
    "ties",
    "n_problems",
    "n_seeds",
}


class TestConditionalTable:
    def test_filters_to_requested_knowledge_base(self):
        results = [
            contrast("Keci", "vicodi", 0.10),
            contrast("TransE", "vicodi", 0.02),
            contrast("Keci", "mutagenesis", 0.99),
        ]
        table = conditional_table(results, "vicodi")
        assert set(table["condition"]) == {"Keci", "TransE"}
        assert 0.99 not in set(table["estimate"])

    def test_sorted_descending_by_estimate(self):
        results = [
            contrast("low", "kb", -0.05),
            contrast("high", "kb", 0.20),
            contrast("mid", "kb", 0.01),
        ]
        table = conditional_table(results, "kb")
        assert list(table["condition"]) == ["high", "mid", "low"]

    def test_none_estimates_sort_last(self):
        results = [
            contrast("has_none", "kb", None),
            contrast("has_value", "kb", -1.0),
        ]
        table = conditional_table(results, "kb")
        assert list(table["condition"]) == ["has_value", "has_none"]

    def test_index_is_reset_after_sorting(self):
        results = [contrast("a", "kb", 0.0), contrast("b", "kb", 1.0)]
        table = conditional_table(results, "kb")
        assert list(table.index) == [0, 1]

    def test_unpacks_interval_and_test_fields(self):
        results = [
            contrast(
                "Keci",
                "kb",
                0.15,
                interval=FakeInterval(0.05, 0.25),
                test=FakeTest(
                    p_value=0.01, hodges_lehmann=0.12, wins=7, losses=2, ties=1
                ),
            )
        ]
        row = conditional_table(results, "kb").iloc[0]
        assert row["ci_low"] == 0.05
        assert row["ci_high"] == 0.25
        assert bool(row["excludes_zero"]) is True
        assert row["p_value"] == 0.01
        assert row["hodges_lehmann"] == 0.12
        assert (row["wins"], row["losses"], row["ties"]) == (7, 2, 1)

    def test_missing_interval_and_test_become_none(self):
        results = [contrast("Keci", "kb", 0.15, interval=None, test=None)]
        row = conditional_table(results, "kb").iloc[0]
        for column in (
            "ci_low",
            "ci_high",
            "excludes_zero",
            "p_value",
            "hodges_lehmann",
            "wins",
            "losses",
            "ties",
        ):
            assert row[column] is None or pd.isna(row[column])

    def test_interval_spanning_zero_reports_false(self):
        results = [
            contrast("Keci", "kb", 0.01, interval=FakeInterval(-0.2, 0.3))
        ]
        assert bool(conditional_table(results, "kb").iloc[0]["excludes_zero"])\
            is False

    def test_unknown_knowledge_base_returns_empty(self):
        table = conditional_table([contrast("a", "kb", 0.1)], "absent")
        assert table.empty

    def test_empty_input_returns_empty(self):
        assert conditional_table([], "kb").empty

    @pytest.mark.xfail(
        reason="empty result loses the output schema; downstream column "
        "access raises KeyError",
        strict=True,
    )
    def test_empty_result_preserves_columns(self):
        table = conditional_table([], "kb")
        assert EXPECTED_TABLE_COLUMNS.issubset(set(table.columns))

    @pytest.mark.xfail(
        reason="no outcome filter: multiple outcomes per condition are "
        "emitted as separate rows and sorted together",
        strict=True,
    )
    def test_one_row_per_condition_across_outcomes(self):
        results = [
            contrast("Keci", "kb", 0.10, outcome="abl"),
            contrast("Keci", "kb", 0.90, outcome="recall"),
        ]
        table = conditional_table(results, "kb")
        assert len(table) == 1


# --------------------------------------------------------------------------
# condition_rankings
# --------------------------------------------------------------------------


class TestConditionRankings:
    def test_ranks_within_each_knowledge_base_independently(self):
        results = [
            contrast("A", "kb1", 0.3),
            contrast("B", "kb1", 0.1),
            contrast("A", "kb2", 0.1),
            contrast("B", "kb2", 0.3),
        ]
        assert condition_rankings(results) == {
            "kb1": ["A", "B"],
            "kb2": ["B", "A"],
        }

    def test_skips_none_estimates(self):
        results = [contrast("A", "kb", None), contrast("B", "kb", 0.1)]
        assert condition_rankings(results) == {"kb": ["B"]}

    def test_kb_absent_when_all_estimates_none(self):
        assert condition_rankings([contrast("A", "kb", None)]) == {}

    def test_ties_broken_alphabetically(self):
        results = [
            contrast("zeta", "kb", 0.5),
            contrast("alpha", "kb", 0.5),
            contrast("mid", "kb", 0.5),
        ]
        assert condition_rankings(results)["kb"] == ["alpha", "mid", "zeta"]

    def test_negative_estimates_ordered_correctly(self):
        results = [
            contrast("worst", "kb", -0.9),
            contrast("best", "kb", 0.9),
            contrast("zero", "kb", 0.0),
        ]
        assert condition_rankings(results)["kb"] == ["best", "zero", "worst"]

    def test_empty_input(self):
        assert condition_rankings([]) == {}

    @pytest.mark.xfail(
        reason="no outcome filter: a condition measured on two outcomes "
        "appears twice in the ranking",
        strict=True,
    )
    def test_condition_appears_at_most_once(self):
        results = [
            contrast("Keci", "kb", 0.10, outcome="abl"),
            contrast("Keci", "kb", 0.90, outcome="recall"),
        ]
        ranking = condition_rankings(results)["kb"]
        assert len(ranking) == len(set(ranking))


# --------------------------------------------------------------------------
# ranking_concordance
# --------------------------------------------------------------------------


def concordance_map(
    concordances: list[RankingConcordance],
) -> dict[tuple[str, str], RankingConcordance]:
    return {(c.kb_a, c.kb_b): c for c in concordances}


class TestRankingConcordance:
    @staticmethod
    def _four_conditions(kb: str, estimates: list[float]):
        names = ["c1", "c2", "c3", "c4"]
        return [contrast(n, kb, e) for n, e in zip(names, estimates)]

    def test_pairs_are_unordered_and_sorted(self):
        results = (
            self._four_conditions("bbb", [1.0, 2.0, 3.0, 4.0])
            + self._four_conditions("aaa", [1.0, 2.0, 3.0, 4.0])
        )
        pairs = [(c.kb_a, c.kb_b) for c in ranking_concordance(results)]
        assert pairs == [("aaa", "bbb")]

    def test_number_of_pairs_is_n_choose_2(self):
        results = []
        for kb in ("kb1", "kb2", "kb3", "kb4"):
            results += self._four_conditions(kb, [1.0, 2.0, 3.0, 4.0])
        assert len(ranking_concordance(results)) == 6

    def test_identical_ordering_gives_tau_one(self):
        results = (
            self._four_conditions("kb1", [0.1, 0.2, 0.3, 0.4])
            + self._four_conditions("kb2", [1.0, 2.0, 3.0, 4.0])
        )
        (result,) = ranking_concordance(results)
        assert result.kendall_tau == pytest.approx(1.0)
        assert result.n_conditions == 4

    def test_reversed_ordering_gives_tau_minus_one(self):
        results = (
            self._four_conditions("kb1", [0.1, 0.2, 0.3, 0.4])
            + self._four_conditions("kb2", [4.0, 3.0, 2.0, 1.0])
        )
        (result,) = ranking_concordance(results)
        assert result.kendall_tau == pytest.approx(-1.0)

    def test_tau_is_scale_invariant(self):
        base = self._four_conditions("kb1", [0.1, 0.2, 0.3, 0.4])
        small = self._four_conditions("kb2", [1e-6, 2e-6, 3e-6, 4e-6])
        (result,) = ranking_concordance(base + small)
        assert result.kendall_tau == pytest.approx(1.0)

    def test_only_shared_conditions_are_compared(self):
        results = [
            contrast("shared1", "kb1", 1.0),
            contrast("shared2", "kb1", 2.0),
            contrast("shared3", "kb1", 3.0),
            contrast("only_kb1", "kb1", 99.0),
            contrast("shared1", "kb2", 1.0),
            contrast("shared2", "kb2", 2.0),
            contrast("shared3", "kb2", 3.0),
            contrast("only_kb2", "kb2", -99.0),
        ]
        (result,) = ranking_concordance(results)
        assert result.n_conditions == 3
        assert result.kendall_tau == pytest.approx(1.0)

    def test_fewer_than_three_shared_returns_none_tau(self):
        results = [
            contrast("a", "kb1", 1.0),
            contrast("b", "kb1", 2.0),
            contrast("a", "kb2", 1.0),
            contrast("b", "kb2", 2.0),
        ]
        (result,) = ranking_concordance(results)
        assert result.kendall_tau is None
        assert result.p_value is None
        assert result.n_conditions == 2

    def test_no_shared_conditions_reports_zero(self):
        results = [
            contrast("x1", "kb1", 1.0),
            contrast("x2", "kb1", 2.0),
            contrast("x3", "kb1", 3.0),
            contrast("y1", "kb2", 1.0),
            contrast("y2", "kb2", 2.0),
            contrast("y3", "kb2", 3.0),
        ]
        (result,) = ranking_concordance(results)
        assert result.n_conditions == 0
        assert result.kendall_tau is None

    def test_all_constant_estimates_yield_none_not_nan(self):
        results = (
            self._four_conditions("kb1", [0.5, 0.5, 0.5, 0.5])
            + self._four_conditions("kb2", [1.0, 2.0, 3.0, 4.0])
        )
        (result,) = ranking_concordance(results)
        assert result.kendall_tau is None
        assert result.p_value is None

    def test_none_estimates_excluded_from_shared_set(self):
        results = [
            contrast("a", "kb1", 1.0),
            contrast("b", "kb1", 2.0),
            contrast("c", "kb1", 3.0),
            contrast("a", "kb2", 1.0),
            contrast("b", "kb2", 2.0),
            contrast("c", "kb2", None),
        ]
        (result,) = ranking_concordance(results)
        assert result.n_conditions == 2

    def test_single_knowledge_base_produces_no_pairs(self):
        results = self._four_conditions("kb1", [1.0, 2.0, 3.0, 4.0])
        assert ranking_concordance(results) == []

    def test_empty_input(self):
        assert ranking_concordance([]) == []

    def test_outputs_are_json_serialisable_floats(self):
        results = (
            self._four_conditions("kb1", [0.1, 0.2, 0.3, 0.4])
            + self._four_conditions("kb2", [0.4, 0.1, 0.3, 0.2])
        )
        (result,) = ranking_concordance(results)
        assert type(result.kendall_tau) is float
        assert type(result.p_value) is float
        json.dumps(result.to_dict())

    def test_to_dict_keys(self):
        results = (
            self._four_conditions("kb1", [0.1, 0.2, 0.3, 0.4])
            + self._four_conditions("kb2", [0.4, 0.1, 0.3, 0.2])
        )
        assert set(ranking_concordance(results)[0].to_dict()) == {
            "kb_a",
            "kb_b",
            "kendall_tau",
            "p_value",
            "n_conditions",
        }

    @pytest.mark.xfail(
        reason="no outcome filter: estimates from different outcomes "
        "overwrite each other in the per-kb dict",
        strict=True,
    )
    def test_multiple_outcomes_do_not_corrupt_tau(self):
        results = []
        for kb, order in (("kb1", [1.0, 2.0, 3.0]), ("kb2", [1.0, 2.0, 3.0])):
            for name, value in zip(["a", "b", "c"], order):
                results.append(contrast(name, kb, value, outcome="abl"))
                results.append(
                    contrast(name, kb, -value, outcome="hypothesis_size")
                )
        (result,) = ranking_concordance(results)
        assert result.kendall_tau == pytest.approx(1.0)


# --------------------------------------------------------------------------
# sign_agreement
# --------------------------------------------------------------------------

EXPECTED_SIGN_COLUMNS = {
    "condition",
    "n_knowledge_bases",
    "n_positive",
    "n_negative",
    "unanimous",
    "mean_estimate_unpooled",
}


class TestSignAgreement:
    def test_counts_positive_and_negative_per_condition(self):
        results = [
            contrast("Keci", "kb1", 0.1),
            contrast("Keci", "kb2", 0.2),
            contrast("Keci", "kb3", -0.3),
            contrast("Keci", "kb4", 0.4),
        ]
        row = sign_agreement(results).iloc[0]
        assert row["n_knowledge_bases"] == 4
        assert row["n_positive"] == 3
        assert row["n_negative"] == 1
        assert bool(row["unanimous"]) is False

    def test_all_positive_is_unanimous(self):
        results = [
            contrast("Keci", f"kb{i}", 0.1 * (i + 1)) for i in range(4)
        ]
        assert bool(sign_agreement(results).iloc[0]["unanimous"]) is True

    def test_all_negative_is_unanimous(self):
        results = [
            contrast("Keci", f"kb{i}", -0.1 * (i + 1)) for i in range(4)
        ]
        row = sign_agreement(results).iloc[0]
        assert bool(row["unanimous"]) is True
        assert row["n_positive"] == 0
        assert row["n_negative"] == 4

    def test_mean_is_unpooled_arithmetic_mean(self):
        results = [
            contrast("Keci", "kb1", 0.0),
            contrast("Keci", "kb2", 1.0),
        ]
        assert sign_agreement(results).iloc[0][
            "mean_estimate_unpooled"
        ] == pytest.approx(0.5)

    def test_exact_zero_counts_as_neither_sign(self):
        results = [
            contrast("Keci", "kb1", 0.0),
            contrast("Keci", "kb2", 0.0),
        ]
        row = sign_agreement(results).iloc[0]
        assert row["n_positive"] == 0
        assert row["n_negative"] == 0
        assert row["n_knowledge_bases"] == 2

    def test_ordered_by_n_positive_then_mean(self):
        results = [
            contrast("weak", "kb1", 0.01),
            contrast("weak", "kb2", 0.01),
            contrast("strong", "kb1", 0.50),
            contrast("strong", "kb2", 0.50),
            contrast("mixed", "kb1", 0.90),
            contrast("mixed", "kb2", -0.90),
        ]
        assert list(sign_agreement(results)["condition"]) == [
            "strong",
            "weak",
            "mixed",
        ]

    def test_index_reset(self):
        results = [
            contrast("a", "kb1", -1.0),
            contrast("b", "kb1", 1.0),
        ]
        assert list(sign_agreement(results).index) == [0, 1]

    def test_none_estimates_dropped_from_counts(self):
        results = [
            contrast("Keci", "kb1", 0.1),
            contrast("Keci", "kb2", None),
        ]
        assert sign_agreement(results).iloc[0]["n_knowledge_bases"] == 1

    def test_all_none_returns_empty(self):
        assert sign_agreement([contrast("Keci", "kb1", None)]).empty

    def test_empty_input_returns_empty(self):
        assert sign_agreement([]).empty

    @pytest.mark.xfail(
        reason="empty path returns the intermediate frame, so the output "
        "schema is lost",
        strict=True,
    )
    def test_empty_result_preserves_columns(self):
        assert EXPECTED_SIGN_COLUMNS.issubset(set(sign_agreement([]).columns))

    @pytest.mark.xfail(
        reason="a single exact zero defeats `unanimous` even when every "
        "other knowledge base agrees in sign",
        strict=True,
    )
    def test_zero_does_not_defeat_unanimity(self):
        results = [
            contrast("Keci", "kb1", 0.1),
            contrast("Keci", "kb2", 0.2),
            contrast("Keci", "kb3", 0.0),
        ]
        assert bool(sign_agreement(results).iloc[0]["unanimous"]) is True

    @pytest.mark.xfail(
        reason="no outcome filter: n_knowledge_bases counts rows, not "
        "distinct knowledge bases",
        strict=True,
    )
    def test_n_knowledge_bases_counts_distinct_kbs(self):
        results = [
            contrast("Keci", "kb1", 0.1, outcome="abl"),
            contrast("Keci", "kb1", 0.2, outcome="recall"),
        ]
        assert sign_agreement(results).iloc[0]["n_knowledge_bases"] == 1

    @pytest.mark.xfail(
        reason="a NaN estimate passes the `is not None` filter and poisons "
        "the mean and the unanimity flag",
        strict=True,
    )
    def test_nan_estimate_is_excluded(self):
        results = [
            contrast("Keci", "kb1", 0.1),
            contrast("Keci", "kb2", float("nan")),
        ]
        row = sign_agreement(results).iloc[0]
        assert row["n_knowledge_bases"] == 1
        assert not math.isnan(row["mean_estimate_unpooled"])


# --------------------------------------------------------------------------
# link_summary
# --------------------------------------------------------------------------


def quality_rows(
    kb: str, pairs: list[tuple[float, float]], condition_prefix: str = "c"
) -> list[dict[str, Any]]:
    return [
        {
            "condition": f"{condition_prefix}{i}",
            "knowledge_base": kb,
            "seed": 1,
            "mrr": mrr,
            "mean_abl": abl,
        }
        for i, (mrr, abl) in enumerate(pairs)
    ]


class TestLinkSummary:
    def test_perfect_monotone_gives_positive_reading(self):
        frame = pd.DataFrame(
            quality_rows(
                "kb1",
                [(0.1, 0.01), (0.2, 0.02), (0.3, 0.03), (0.4, 0.04),
                 (0.5, 0.05), (0.6, 0.06)],
            )
        )
        (summary,) = link_summary(frame)
        assert summary.rho == pytest.approx(1.0)
        assert summary.p_value < 0.05
        assert "usable proxy" in summary.reading
        assert summary.n_points == 6

    def test_flat_relationship_gives_flat_reading(self):
        frame = pd.DataFrame(
            quality_rows(
                "kb1",
                [(0.1, 0.05), (0.2, 0.01), (0.3, 0.04), (0.4, 0.02),
                 (0.5, 0.03)],
            )
        )
        (summary,) = link_summary(frame)
        assert summary.p_value >= 0.05
        assert "identifier rather than as a carrier" in summary.reading

    def test_significant_negative_gets_its_own_reading(self):
        frame = pd.DataFrame(
            quality_rows(
                "kb1",
                [(0.1, 0.06), (0.2, 0.05), (0.3, 0.04), (0.4, 0.03),
                 (0.5, 0.02), (0.6, 0.01)],
            )
        )
        (summary,) = link_summary(frame)
        assert summary.rho == pytest.approx(-1.0)
        assert "unexpected" in summary.reading

    def test_computed_separately_per_knowledge_base(self):
        rows = quality_rows(
            "up",
            [(0.1, 0.01), (0.2, 0.02), (0.3, 0.03), (0.4, 0.04), (0.5, 0.05)],
        ) + quality_rows(
            "down",
            [(0.1, 0.05), (0.2, 0.04), (0.3, 0.03), (0.4, 0.02), (0.5, 0.01)],
        )
        summaries = {s.knowledge_base: s for s in link_summary(
            pd.DataFrame(rows)
        )}
        assert summaries["up"].rho == pytest.approx(1.0)
        assert summaries["down"].rho == pytest.approx(-1.0)

    def test_cross_kb_scale_offset_cannot_create_association(self):
        """The within-KB rho must not see the between-KB scale difference."""
        rows = quality_rows(
            "low",
            [(0.10, 0.9), (0.11, 0.1), (0.12, 0.5), (0.13, 0.3),
             (0.14, 0.7)],
        ) + quality_rows(
            "high",
            [(0.90, 9.0), (0.91, 1.0), (0.92, 5.0), (0.93, 3.0),
             (0.94, 7.0)],
        )
        for summary in link_summary(pd.DataFrame(rows)):
            assert summary.p_value >= 0.05

    def test_rows_with_missing_values_are_dropped(self):
        rows = quality_rows(
            "kb1",
            [(0.1, 0.01), (0.2, 0.02), (0.3, 0.03), (0.4, 0.04), (0.5, 0.05)],
        )
        rows.append(
            {
                "condition": "random",
                "knowledge_base": "kb1",
                "seed": 1,
                "mrr": None,
                "mean_abl": 0.5,
            }
        )
        (summary,) = link_summary(pd.DataFrame(rows))
        assert summary.n_points == 5

    def test_too_few_points_is_indeterminate(self):
        frame = pd.DataFrame(
            quality_rows("kb1", [(0.1, 0.01), (0.2, 0.02), (0.3, 0.03)])
        )
        (summary,) = link_summary(frame)
        assert summary.rho is None
        assert summary.p_value is None
        assert summary.reading.startswith("indeterminate")
        assert summary.n_points == 3

    def test_constant_mrr_is_indeterminate(self):
        frame = pd.DataFrame(
            quality_rows(
                "kb1",
                [(0.3, 0.01), (0.3, 0.02), (0.3, 0.03), (0.3, 0.04),
                 (0.3, 0.05)],
            )
        )
        (summary,) = link_summary(frame)
        assert summary.rho is None
        assert summary.reading.startswith("indeterminate")

    def test_respects_custom_column_names(self):
        frame = pd.DataFrame(
            {
                "knowledge_base": ["kb1"] * 5,
                "hits": [0.1, 0.2, 0.3, 0.4, 0.5],
                "score": [0.01, 0.02, 0.03, 0.04, 0.05],
            }
        )
        (summary,) = link_summary(
            frame, mrr_column="hits", abl_column="score"
        )
        assert summary.rho == pytest.approx(1.0)

    def test_empty_frame_returns_empty_list(self):
        assert link_summary(pd.DataFrame()) == []

    def test_to_dict_keys(self):
        frame = pd.DataFrame(
            quality_rows(
                "kb1",
                [(0.1, 0.01), (0.2, 0.02), (0.3, 0.03), (0.4, 0.04),
                 (0.5, 0.05)],
            )
        )
        assert set(link_summary(frame)[0].to_dict()) == {
            "knowledge_base",
            "spearman_rho",
            "p_value",
            "n_points",
            "reading",
        }

    def test_knowledge_base_coerced_to_str(self):
        frame = pd.DataFrame(
            {
                "knowledge_base": [1] * 5,
                "mrr": [0.1, 0.2, 0.3, 0.4, 0.5],
                "mean_abl": [0.01, 0.02, 0.03, 0.04, 0.05],
            }
        )
        assert type(link_summary(frame)[0].knowledge_base) is str

    @pytest.mark.xfail(
        reason="p_value is left as a numpy scalar, unlike rho; breaks "
        "json.dumps of the persisted artifact",
        strict=True,
    )
    def test_p_value_is_a_plain_float(self):
        frame = pd.DataFrame(
            quality_rows(
                "kb1",
                [(0.1, 0.01), (0.2, 0.05), (0.3, 0.02), (0.4, 0.04),
                 (0.5, 0.03)],
            )
        )
        (summary,) = link_summary(frame)
        assert type(summary.p_value) is float
        json.dumps(summary.to_dict())

    @pytest.mark.xfail(
        reason="constant ABL makes rho NaN, which is silently reported as "
        "the pre-registered flat/negative finding",
        strict=True,
    )
    def test_constant_abl_is_indeterminate_not_flat(self):
        frame = pd.DataFrame(
            quality_rows(
                "kb1",
                [(0.1, 0.0), (0.2, 0.0), (0.3, 0.0), (0.4, 0.0), (0.5, 0.0)],
            )
        )
        (summary,) = link_summary(frame)
        assert summary.reading.startswith("indeterminate")
        assert summary.rho is None

    @pytest.mark.xfail(
        reason="rho may be reported as NaN rather than None when the ABL "
        "column is degenerate",
        strict=True,
    )
    def test_rho_is_never_nan(self):
        frame = pd.DataFrame(
            quality_rows(
                "kb1",
                [(0.1, 0.0), (0.2, 0.0), (0.3, 0.0), (0.4, 0.0), (0.5, 0.0)],
            )
        )
        (summary,) = link_summary(frame)
        assert summary.rho is None or not math.isnan(summary.rho)

    @pytest.mark.xfail(
        reason="seeds of one condition are dependent but enter Spearman as "
        "independent points, so the p-value is anticonservative",
        strict=True,
    )
    def test_seed_replication_does_not_inflate_significance(self):
        """Five seeds of three conditions must not read as n=15."""
        rows: list[dict[str, Any]] = []
        for index, (mrr, abl) in enumerate(
            [(0.1, 0.01), (0.2, 0.02), (0.3, 0.03)]
        ):
            for seed in range(1, 6):
                rows.append(
                    {
                        "condition": f"c{index}",
                        "knowledge_base": "kb1",
                        "seed": seed,
                        "mrr": mrr + seed * 1e-6,
                        "mean_abl": abl + seed * 1e-6,
                    }
                )
        (summary,) = link_summary(pd.DataFrame(rows))
        assert summary.n_points == 3


# --------------------------------------------------------------------------
# quality_frame
# --------------------------------------------------------------------------


class TestQualityFrame:
    def test_joins_metrics_to_downstream_abl_on_full_key(self):
        key = ("Keci", "vicodi", 1)
        frame = quality_frame(
            {key: {"mrr": 0.4, "hits@1": 0.3, "hits@3": 0.5, "hits@10": 0.7}},
            {key: 0.12},
        )
        row = frame.iloc[0]
        assert row["condition"] == "Keci"
        assert row["knowledge_base"] == "vicodi"
        assert row["seed"] == 1
        assert row["mrr"] == 0.4
        assert row["hits_at_1"] == 0.3
        assert row["hits_at_3"] == 0.5
        assert row["hits_at_10"] == 0.7
        assert row["mean_abl"] == 0.12

    def test_hits_keys_are_renamed_to_valid_identifiers(self):
        frame = quality_frame({("c", "kb", 1): {"hits@10": 0.9}}, {})
        assert "hits_at_10" in frame.columns
        assert "hits@10" not in frame.columns

    def test_missing_metric_becomes_none(self):
        frame = quality_frame({("c", "kb", 1): {"mrr": 0.5}}, {})
        row = frame.iloc[0]
        assert row["mrr"] == 0.5
        for column in ("hits_at_1", "hits_at_3", "hits_at_10"):
            assert row[column] is None or pd.isna(row[column])

    def test_missing_abl_becomes_none(self):
        frame = quality_frame({("c", "kb", 1): {"mrr": 0.5}}, {})
        assert pd.isna(frame.iloc[0]["mean_abl"])

    def test_abl_without_matching_metrics_is_not_emitted(self):
        """Iteration is driven by ranking_metrics only."""
        frame = quality_frame({}, {("orphan", "kb", 1): 0.5})
        assert frame.empty

    def test_one_row_per_key(self):
        metrics = {
            (condition, kb, seed): {"mrr": 0.1}
            for condition in ("Keci", "TransE")
            for kb in ("kb1", "kb2")
            for seed in (1, 2, 3)
        }
        frame = quality_frame(metrics, {})
        assert len(frame) == 12
        assert not frame.duplicated(
            subset=["condition", "knowledge_base", "seed"]
        ).any()

    def test_zero_abl_is_preserved_not_treated_as_missing(self):
        key = ("c", "kb", 1)
        frame = quality_frame({key: {"mrr": 0.1}}, {key: 0.0})
        assert frame.iloc[0]["mean_abl"] == 0.0

    def test_empty_inputs_return_empty_frame(self):
        assert quality_frame({}, {}).empty

    def test_output_is_consumable_by_link_summary(self):
        metrics = {
            (f"c{i}", "kb1", 1): {"mrr": 0.1 * (i + 1)} for i in range(5)
        }
        abl = {(f"c{i}", "kb1", 1): 0.01 * (i + 1) for i in range(5)}
        (summary,) = link_summary(quality_frame(metrics, abl))
        assert summary.rho == pytest.approx(1.0)


# --------------------------------------------------------------------------
# Dataclass contracts
# --------------------------------------------------------------------------


class TestDataclassContracts:
    def test_ranking_concordance_is_frozen(self):
        record = RankingConcordance("a", "b", 1.0, 0.01, 4)
        with pytest.raises(Exception):
            record.kendall_tau = 0.5  # type: ignore[misc]

    def test_link_summary_is_frozen(self):
        record = LinkSummary("kb", 1.0, 0.01, 5, "reading")
        with pytest.raises(Exception):
            record.rho = 0.5  # type: ignore[misc]

    def test_none_tau_survives_json_round_trip(self):
        record = RankingConcordance("a", "b", None, None, 2)
        assert json.loads(json.dumps(record.to_dict()))["kendall_tau"] is None