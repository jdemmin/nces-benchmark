"""Tests for reasoner-backed extension computation and its cache.

The HermiT-backed `SyncReasoner` is slow and I/O bound, so the caching,
parse-failure and persistence contracts are exercised against an injected
fake. The invariants under test are those the benchmark relies on: an
expression is reasoned over at most once per run, malformed hypotheses are
distinguishable from legitimately empty extensions, and negative results
never enter the persisted cache.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pytest

from src.eval.reasoning import (
    ExtensionOracle,
    UnparseableExpression,
    _default_namespace,
)

NS = "http://example.org/kb#"
THING = "http://www.w3.org/2002/07/owl#Thing"
NOTHING = "http://www.w3.org/2002/07/owl#Nothing"


# --------------------------------------------------------------------------
# doubles
# --------------------------------------------------------------------------


class FakeEntity:
    """Stands in for owlapy entities, which expose IRIs via `.str`."""

    def __init__(self, iri: str) -> None:
        self.str = iri

    def __repr__(self) -> str:
        return f"FakeEntity({self.str!r})"


class FakeExpression:
    """Parsed class expression: an opaque token plus its known extension."""

    def __init__(self, text: str) -> None:
        self.text = text

    def __eq__(self, other: object) -> bool:
        return isinstance(other, FakeExpression) and other.text == self.text

    def __hash__(self) -> int:
        return hash(("FakeExpression", self.text))


class FakeParser:
    """Parses anything except strings containing '(('."""

    def __init__(self, namespace: str) -> None:
        self.namespace = namespace
        self.calls: list[str] = []

    def parse_expression(self, expression: str) -> FakeExpression:
        self.calls.append(expression)
        if "((" in expression:
            raise ValueError(f"malformed: {expression}")
        return FakeExpression(expression)


class FakeReasoner:
    """Records every `instances` call so caching can be asserted."""

    def __init__(self, extensions: dict[str, set[str]]) -> None:
        self._extensions = extensions
        self.calls: list[tuple[str, bool]] = []

    def instances(self, expression: Any, direct: bool = True) -> list[FakeEntity]:
        key = (
            expression.text
            if isinstance(expression, FakeExpression)
            else expression.str
        )
        self.calls.append((key, direct))
        return [FakeEntity(iri) for iri in sorted(self._extensions.get(key, ()))]

    @property
    def call_keys(self) -> list[str]:
        return [key for key, _ in self.calls]


class FakeOntology:
    def __init__(
        self,
        *,
        individuals: list[str],
        classes: list[str],
        ontology_iri: str = "http://example.org/kb",
    ) -> None:
        self._individuals = individuals
        self._classes = classes
        self._ontology_iri = ontology_iri

    def individuals_in_signature(self) -> list[FakeEntity]:
        return [FakeEntity(iri) for iri in self._individuals]

    def classes_in_signature(self) -> list[FakeEntity]:
        return [FakeEntity(iri) for iri in self._classes]

    def get_ontology_id(self) -> Any:
        outer = self

        class _Id:
            def get_ontology_iri(self) -> FakeEntity:
                return FakeEntity(outer._ontology_iri)

        return _Id()


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------

DEFAULT_EXTENSIONS: dict[str, set[str]] = {
    "Male": {f"{NS}a", f"{NS}b"},
    "Female": {f"{NS}c"},
    "Person": {f"{NS}a", f"{NS}b", f"{NS}c"},
    "Ghost": set(),
    "Male ⊓ Female": set(),
    "∃ hasChild.Person": {f"{NS}a"},
    THING: {f"{NS}a", f"{NS}b", f"{NS}c"},
    NOTHING: set(),
}

DEFAULT_INDIVIDUALS = [f"{NS}a", f"{NS}b", f"{NS}c"]
DEFAULT_CLASSES = [f"{NS}Male", f"{NS}Female", f"{NS}Person", THING, NOTHING]


@pytest.fixture
def ontology_path(tmp_path: Path) -> Path:
    path = tmp_path / "kb.owl"
    path.write_text("<!-- not parsed by the fake -->", encoding="utf-8")
    return path


@pytest.fixture
def wiring(monkeypatch: pytest.MonkeyPatch):
    """Replace `_ensure_reasoner` with in-memory doubles.

    Returns a factory so each test can declare its own extensions, classes
    and individuals while keeping the oracle's real caching logic intact.
    """

    def make(
        *,
        extensions: dict[str, set[str]] | None = None,
        individuals: list[str] | None = None,
        classes: list[str] | None = None,
        ontology_iri: str = "http://example.org/kb",
    ):
        state: dict[str, Any] = {"loads": 0}
        table = dict(DEFAULT_EXTENSIONS if extensions is None else extensions)
        # Atomic classes are reasoned over as OWLClass objects keyed by IRI.
        for iri in classes if classes is not None else DEFAULT_CLASSES:
            local = iri.rsplit("#", 1)[-1]
            table.setdefault(iri, set(table.get(local, set())))

        def fake_ensure(self: ExtensionOracle) -> None:
            if self._reasoner is not None:
                return
            state["loads"] += 1
            ontology = FakeOntology(
                individuals=(
                    DEFAULT_INDIVIDUALS if individuals is None else individuals
                ),
                classes=DEFAULT_CLASSES if classes is None else classes,
                ontology_iri=ontology_iri,
            )
            reasoner = FakeReasoner(table)
            parser = FakeParser(_default_namespace(ontology))
            self._ontology = ontology
            self._reasoner = reasoner
            self._parser = parser
            state["ontology"] = ontology
            state["reasoner"] = reasoner
            state["parser"] = parser

        monkeypatch.setattr(ExtensionOracle, "_ensure_reasoner", fake_ensure)
        return state

    return make


@pytest.fixture
def oracle(ontology_path: Path, wiring):
    state = wiring()
    instance = ExtensionOracle(ontology_path=ontology_path)
    instance._state = state  # type: ignore[attr-defined]
    return instance


# --------------------------------------------------------------------------
# construction / laziness
# --------------------------------------------------------------------------


class TestConstruction:
    def test_construction_does_not_load_the_reasoner(
        self, ontology_path: Path, wiring
    ) -> None:
        state = wiring()
        ExtensionOracle(ontology_path=ontology_path)
        assert state["loads"] == 0

    def test_construction_with_missing_cache_file_is_not_an_error(
        self, ontology_path: Path, tmp_path: Path, wiring
    ) -> None:
        wiring()
        instance = ExtensionOracle(
            ontology_path=ontology_path,
            cache_path=tmp_path / "absent" / "cache.json",
        )
        assert instance._cache == {}

    def test_cache_path_pointing_at_a_directory_is_ignored(
        self, ontology_path: Path, tmp_path: Path, wiring
    ) -> None:
        wiring()
        directory = tmp_path / "cache_dir"
        directory.mkdir()
        instance = ExtensionOracle(
            ontology_path=ontology_path, cache_path=directory
        )
        assert instance._cache == {}

    def test_reasoner_is_loaded_at_most_once(self, oracle: ExtensionOracle) -> None:
        oracle.extension("Male")
        oracle.extension("Female")
        _ = oracle.universe
        _ = oracle.atomic_extensions
        assert oracle._state["loads"] == 1  # type: ignore[attr-defined]

    def test_context_manager_returns_the_oracle(
        self, ontology_path: Path, wiring
    ) -> None:
        wiring()
        with ExtensionOracle(ontology_path=ontology_path) as instance:
            assert isinstance(instance, ExtensionOracle)


# --------------------------------------------------------------------------
# universe
# --------------------------------------------------------------------------


class TestUniverse:
    def test_universe_is_all_named_individuals(
        self, oracle: ExtensionOracle
    ) -> None:
        assert oracle.universe == frozenset(DEFAULT_INDIVIDUALS)

    def test_universe_is_a_frozenset_of_iri_strings(
        self, oracle: ExtensionOracle
    ) -> None:
        universe = oracle.universe
        assert isinstance(universe, frozenset)
        assert all(isinstance(iri, str) for iri in universe)

    def test_universe_is_memoised(self, oracle: ExtensionOracle) -> None:
        first = oracle.universe
        assert oracle.universe is first

    def test_universe_deduplicates_repeated_individuals(
        self, ontology_path: Path, wiring
    ) -> None:
        wiring(individuals=[f"{NS}a", f"{NS}a", f"{NS}b"])
        instance = ExtensionOracle(ontology_path=ontology_path)
        assert instance.universe == frozenset({f"{NS}a", f"{NS}b"})

    def test_empty_universe_is_representable(
        self, ontology_path: Path, wiring
    ) -> None:
        wiring(individuals=[])
        instance = ExtensionOracle(ontology_path=ontology_path)
        assert instance.universe == frozenset()

def test_universe_size_is_the_metric_denominator(
        oracle: ExtensionOracle
    ) -> None:
        # Guards against U being derived from the reasoner rather than the
        # ontology signature: an individual asserted in no class must still
        # count toward |U|, since it is a true negative for every hypothesis.
        assert len(oracle.universe) == 3
        assert f"{NS}c" in oracle.universe

def test_universe_includes_individuals_outside_every_extension(
        ontology_path: Path, wiring
    ) -> None:
        wiring(
            individuals=[f"{NS}a", f"{NS}b", f"{NS}c", f"{NS}orphan"],
            extensions={"Male": {f"{NS}a"}},
        )
        instance = ExtensionOracle(ontology_path=ontology_path)
        assert f"{NS}orphan" in instance.universe
        assert f"{NS}orphan" not in instance.extension("Male")


# --------------------------------------------------------------------------
# atomic extensions
# --------------------------------------------------------------------------


class TestAtomicExtensions:
    def test_thing_and_nothing_are_excluded(
        self, oracle: ExtensionOracle
    ) -> None:
        # The admissible baseline set is "all named classes excluding
        # owl:Thing and owl:Nothing"; including them would let the baseline
        # be attained by a trivial concept.
        atomic = oracle.atomic_extensions
        assert THING not in atomic
        assert NOTHING not in atomic

    def test_all_other_named_classes_are_present(
        self, oracle: ExtensionOracle
    ) -> None:
        assert set(oracle.atomic_extensions) == {
            f"{NS}Male",
            f"{NS}Female",
            f"{NS}Person",
        }

    def test_extensions_are_keyed_by_full_iri(
        self, oracle: ExtensionOracle
    ) -> None:
        assert all(key.startswith("http") for key in oracle.atomic_extensions)

    def test_values_are_frozensets_of_iri_strings(
        self, oracle: ExtensionOracle
    ) -> None:
        for extension in oracle.atomic_extensions.values():
            assert isinstance(extension, frozenset)
            assert all(isinstance(iri, str) for iri in extension)

    def test_instances_are_requested_non_directly(
        self, oracle: ExtensionOracle
    ) -> None:
        # `direct=False` is required for completeness of the atomic
        # baseline: a subclass's instances belong to its superclass's
        # extension, and a direct-only query would deflate the baseline.
        _ = oracle.atomic_extensions
        reasoner: FakeReasoner = oracle._state["reasoner"]  # type: ignore
        assert reasoner.calls, "no instance queries were issued"
        assert all(direct is False for _, direct in reasoner.calls)

    def test_atomic_extensions_are_memoised(
        self, oracle: ExtensionOracle
    ) -> None:
        first = oracle.atomic_extensions
        reasoner: FakeReasoner = oracle._state["reasoner"]  # type: ignore
        call_count = len(reasoner.calls)
        assert oracle.atomic_extensions is first
        assert len(reasoner.calls) == call_count

    def test_empty_class_extension_is_retained_not_dropped(
        self, ontology_path: Path, wiring
    ) -> None:
        # An unsatisfiable named class still participates in the baseline
        # maximisation (contributing F1 = 0), so it must not be filtered.
        wiring(
            classes=[f"{NS}Male", f"{NS}Ghost", THING, NOTHING],
            extensions={"Male": {f"{NS}a"}, f"{NS}Ghost": set()},
        )
        instance = ExtensionOracle(ontology_path=ontology_path)
        atomic = instance.atomic_extensions
        assert f"{NS}Ghost" in atomic
        assert atomic[f"{NS}Ghost"] == frozenset()

    def test_ontology_without_named_classes_yields_empty_mapping(
        self, ontology_path: Path, wiring
    ) -> None:
        wiring(classes=[THING, NOTHING])
        instance = ExtensionOracle(ontology_path=ontology_path)
        assert instance.atomic_extensions == {}

    def test_atomic_extensions_do_not_populate_the_expression_cache(
        self, ontology_path: Path, tmp_path: Path, wiring
    ) -> None:
        # The cache is keyed by *rendered expression string*; atomic classes
        # are reasoned over as OWLClass objects and must not leak IRI-keyed
        # entries that could later be mistaken for expression hits.
        wiring()
        cache = tmp_path / "cache.json"
        with ExtensionOracle(
            ontology_path=ontology_path, cache_path=cache
        ) as instance:
            _ = instance.atomic_extensions
        payload = json.loads(cache.read_text(encoding="utf-8"))
        assert payload == {}


# --------------------------------------------------------------------------
# extension()
# --------------------------------------------------------------------------


class TestExtension:
    def test_returns_the_reasoner_result_as_a_frozenset(
        self, oracle: ExtensionOracle
    ) -> None:
        result = oracle.extension("Male")
        assert result == frozenset({f"{NS}a", f"{NS}b"})
        assert isinstance(result, frozenset)

    def test_satisfiable_but_empty_extension_is_a_valid_result(
        self, oracle: ExtensionOracle
    ) -> None:
        # Distinct from an unparseable hypothesis: this is a scored outcome
        # feeding the empty-hypothesis diagnostic.
        assert oracle.extension("Male ⊓ Female") == frozenset()

    def test_queries_use_direct_false(self, oracle: ExtensionOracle) -> None:
        oracle.extension("Person")
        reasoner: FakeReasoner = oracle._state["reasoner"]  # type: ignore
        assert reasoner.calls == [("Person", False)]

    def test_complex_dl_expression_is_forwarded_verbatim(
        self, oracle: ExtensionOracle
    ) -> None:
        expression = "∃ hasChild.Person"
        assert oracle.extension(expression) == frozenset({f"{NS}a"})
        parser: FakeParser = oracle._state["parser"]  # type: ignore
        assert parser.calls == [expression]

    def test_unknown_expression_yields_an_empty_extension(
        self, oracle: ExtensionOracle
    ) -> None:
        assert oracle.extension("Unmodelled") == frozenset()

    def test_reasoner_is_loaded_on_first_query(
        self, ontology_path: Path, wiring
    ) -> None:
        state = wiring()
        instance = ExtensionOracle(ontology_path=ontology_path)
        assert state["loads"] == 0
        instance.extension("Male")
        assert state["loads"] == 1


class TestExtensionCaching:
    def test_repeated_query_hits_the_cache(
        self, oracle: ExtensionOracle
    ) -> None:
        first = oracle.extension("Male")
        second = oracle.extension("Male")
        reasoner: FakeReasoner = oracle._state["reasoner"]  # type: ignore
        assert first == second
        assert reasoner.call_keys == ["Male"]

    def test_cached_result_is_returned_without_reparsing(
        self, oracle: ExtensionOracle
    ) -> None:
        oracle.extension("Male")
        oracle.extension("Male")
        parser: FakeParser = oracle._state["parser"]  # type: ignore
        assert parser.calls == ["Male"]

    def test_target_and_hypothesis_share_one_cache_entry(
        self, oracle: ExtensionOracle
    ) -> None:
        # A hypothesis that is byte-identical to the target string must not
        # be reasoned over twice within a run.
        target = oracle.extension("Person")
        hypothesis = oracle.extension("Person")
        reasoner: FakeReasoner = oracle._state["reasoner"]  # type: ignore
        assert target == hypothesis
        assert len(reasoner.calls) == 1

    def test_distinct_expressions_get_distinct_entries(
        self, oracle: ExtensionOracle
    ) -> None:
        oracle.extension("Male")
        oracle.extension("Female")
        assert set(oracle._cache) == {"Male", "Female"}

    def test_cache_is_keyed_by_exact_string(
        self, oracle: ExtensionOracle
    ) -> None:
        # Whitespace-differing renderings are separate keys by design; the
        # identical-hypothesis diagnostic likewise compares raw strings.
        oracle.extension("Male")
        oracle.extension(" Male")
        reasoner: FakeReasoner = oracle._state["reasoner"]  # type: ignore
        assert reasoner.call_keys == ["Male", " Male"]

    def test_preloaded_cache_prevents_reasoner_construction(
        self, ontology_path: Path, tmp_path: Path, wiring
    ) -> None:
        # The whole point of persisting the cache: a re-analysis over the
        # same expressions must never touch HermiT.
        cache = tmp_path / "cache.json"
        cache.write_text(json.dumps({"Male": [f"{NS}a", f"{NS}b"]}), "utf-8")
        state = wiring()
        instance = ExtensionOracle(
            ontology_path=ontology_path, cache_path=cache
        )
        assert instance.extension("Male") == frozenset({f"{NS}a", f"{NS}b"})
        assert state["loads"] == 0

    def test_empty_cached_extension_is_a_hit_not_a_miss(
        self, ontology_path: Path, tmp_path: Path, wiring
    ) -> None:
        # A falsy value must not be re-computed: `expression in self._cache`
        # is the correct membership test, `if self._cache.get(...)` is not.
        cache = tmp_path / "cache.json"
        cache.write_text(json.dumps({"Male ⊓ Female": []}), "utf-8")
        state = wiring()
        instance = ExtensionOracle(
            ontology_path=ontology_path, cache_path=cache
        )
        assert instance.extension("Male ⊓ Female") == frozenset()
        assert state["loads"] == 0


class TestExtensionParseFailure:
    def test_malformed_expression_raises_unparseable(
        self, oracle: ExtensionOracle
    ) -> None:
        with pytest.raises(UnparseableExpression):
            oracle.extension("Male ⊓ ((")

    def test_unparseable_is_a_runtime_error_subclass(self) -> None:
        assert issubclass(UnparseableExpression, RuntimeError)

    def test_exception_carries_the_offending_expression(
        self, oracle: ExtensionOracle
    ) -> None:
        expression = "Male ⊓ (("
        with pytest.raises(UnparseableExpression) as info:
            oracle.extension(expression)
        assert expression in str(info.value)

    def test_underlying_parser_error_is_chained(
        self, oracle: ExtensionOracle
    ) -> None:
        with pytest.raises(UnparseableExpression) as info:
            oracle.extension("((")
        assert isinstance(info.value.__cause__, ValueError)

    def test_bare_exceptions_from_the_parser_are_wrapped(
        self, ontology_path: Path, wiring, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # owlapy raises bare exceptions here; nothing may escape unwrapped.
        state = wiring()
        instance = ExtensionOracle(ontology_path=ontology_path)
        instance.extension("Male")  # force wiring

        def explode(_expression: str) -> None:
            raise Exception("jpype boom")

        monkeypatch.setattr(state["parser"], "parse_expression", explode)
        with pytest.raises(UnparseableExpression):
            instance.extension("Female")

    def test_failure_does_not_poison_the_cache(
        self, oracle: ExtensionOracle
    ) -> None:
        with pytest.raises(UnparseableExpression):
            oracle.extension("((")
        assert "((" not in oracle._cache

    def test_failure_does_not_query_the_reasoner(
        self, oracle: ExtensionOracle
    ) -> None:
        with pytest.raises(UnparseableExpression):
            oracle.extension("((")
        reasoner: FakeReasoner = oracle._state["reasoner"]  # type: ignore
        assert reasoner.calls == []

    def test_oracle_remains_usable_after_a_failure(
        self, oracle: ExtensionOracle
    ) -> None:
        with pytest.raises(UnparseableExpression):
            oracle.extension("((")
        assert oracle.extension("Male") == frozenset({f"{NS}a", f"{NS}b"})


# --------------------------------------------------------------------------
# extension_or_none()
# --------------------------------------------------------------------------


class TestExtensionOrNone:
    def test_valid_expression_returns_the_extension(
        self, oracle: ExtensionOracle
    ) -> None:
        assert oracle.extension_or_none("Male") == frozenset(
            {f"{NS}a", f"{NS}b"}
        )

    def test_malformed_expression_returns_none(
        self, oracle: ExtensionOracle
    ) -> None:
        # None means "no usable value" -> the problem becomes unpaired and
        # is excluded from every aggregate (tier-3 failure semantics).
        assert oracle.extension_or_none("Male ⊓ ((") is None

    @pytest.mark.parametrize("expression", ["", " ", "\t", "\n", "   \n  "])
    def test_blank_expression_returns_empty_set_not_none(
        self, oracle: ExtensionOracle, expression: str
    ) -> None:
        # A learner that emitted nothing produced an empty hypothesis: a
        # scored outcome (|P| = 0, F1 = 0), not a measurement failure.
        assert oracle.extension_or_none(expression) == frozenset()

    def test_blank_expression_is_distinguishable_from_malformed(
        self, oracle: ExtensionOracle
    ) -> None:
        empty = oracle.extension_or_none("")
        broken = oracle.extension_or_none("((")
        assert empty is not None
        assert broken is None

    def test_blank_expression_does_not_load_the_reasoner(
        self, ontology_path: Path, wiring
    ) -> None:
        state = wiring()
        instance = ExtensionOracle(ontology_path=ontology_path)
        assert instance.extension_or_none("") == frozenset()
        assert state["loads"] == 0

    def test_blank_expression_is_not_cached(
        self, oracle: ExtensionOracle
    ) -> None:
        oracle.extension_or_none("")
        assert oracle._cache == {}

    def test_unparseable_expression_is_logged_at_debug(
        self, oracle: ExtensionOracle, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.DEBUG, logger="src.eval.reasoning"):
            oracle.extension_or_none("((")
        assert any(
            "Unparseable expression" in record.message
            for record in caplog.records
        )

    def test_empty_extension_is_returned_as_a_set_not_none(
        self, oracle: ExtensionOracle
    ) -> None:
        # Satisfiable-but-empty must never be conflated with unparseable.
        assert oracle.extension_or_none("Male ⊓ Female") == frozenset()

    def test_repeated_call_uses_the_cache(
        self, oracle: ExtensionOracle
    ) -> None:
        oracle.extension_or_none("Male")
        oracle.extension_or_none("Male")
        reasoner: FakeReasoner = oracle._state["reasoner"]  # type: ignore
        assert reasoner.call_keys == ["Male"]

    def test_malformed_expression_is_retried_each_call(
        self, oracle: ExtensionOracle
    ) -> None:
        # Negative results are deliberately uncached; the cost is bounded
        # and the persisted artifact stays clean.
        oracle.extension_or_none("((")
        oracle.extension_or_none("((")
        parser: FakeParser = oracle._state["parser"]  # type: ignore
        assert parser.calls == ["((", "(("]


# --------------------------------------------------------------------------
# prime()
# --------------------------------------------------------------------------


class TestPrime:
    def test_warms_the_cache_for_every_expression(
        self, oracle: ExtensionOracle
    ) -> None:
        oracle.prime(iter(["Male", "Female", "Person"]))
        assert set(oracle._cache) == {"Male", "Female", "Person"}

    def test_malformed_entries_do_not_abort_the_warm_up(
        self, oracle: ExtensionOracle
    ) -> None:
        # The hardness-annotation stage primes over generated targets; one
        # bad rendering must not take down the batch.
        oracle.prime(iter(["Male", "((", "Female"]))
        assert set(oracle._cache) == {"Male", "Female"}

    def test_accepts_any_iterable(self, oracle: ExtensionOracle) -> None:
        oracle.prime(expression for expression in ("Male", "Person"))
        assert set(oracle._cache) == {"Male", "Person"}

    def test_empty_iterable_is_a_no_op(
        self, ontology_path: Path, wiring
    ) -> None:
        state = wiring()
        instance = ExtensionOracle(ontology_path=ontology_path)
        instance.prime(iter([]))
        assert instance._cache == {}
        assert state["loads"] == 0

    def test_priming_makes_subsequent_scoring_reasoner_free(
        self, oracle: ExtensionOracle
    ) -> None:
        oracle.prime(iter(["Male", "Female"]))
        reasoner: FakeReasoner = oracle._state["reasoner"]  # type: ignore
        before = len(reasoner.calls)
        oracle.extension("Male")
        oracle.extension("Female")
        assert len(reasoner.calls) == before

    def test_duplicate_expressions_are_reasoned_over_once(
        self, oracle: ExtensionOracle
    ) -> None:
        oracle.prime(iter(["Male", "Male", "Male"]))
        reasoner: FakeReasoner = oracle._state["reasoner"]  # type: ignore
        assert reasoner.call_keys == ["Male"]


# --------------------------------------------------------------------------
# cache persistence
# --------------------------------------------------------------------------


class TestCachePersistence:
    def test_close_writes_the_cache(
        self, ontology_path: Path, tmp_path: Path, wiring
    ) -> None:
        wiring()
        cache = tmp_path / "cache.json"
        instance = ExtensionOracle(
            ontology_path=ontology_path, cache_path=cache
        )
        instance.extension("Male")
        instance.close()
        assert json.loads(cache.read_text(encoding="utf-8")) == {
            "Male": [f"{NS}a", f"{NS}b"]
        }

    def test_close_creates_missing_parent_directories(
        self, ontology_path: Path, tmp_path: Path, wiring
    ) -> None:
        wiring()
        cache = tmp_path / "nested" / "deeper" / "cache.json"
        with ExtensionOracle(
            ontology_path=ontology_path, cache_path=cache
        ) as instance:
            instance.extension("Male")
        assert cache.is_file()

    def test_close_without_a_cache_path_is_harmless(
        self, oracle: ExtensionOracle
    ) -> None:
        oracle.extension("Male")
        oracle.close()  # must not raise

    def test_close_drops_the_reasoner_handle(
        self, oracle: ExtensionOracle
    ) -> None:
        oracle.extension("Male")
        oracle.close()
        assert oracle._reasoner is None

    def test_cache_survives_close_in_memory(
        self, oracle: ExtensionOracle
    ) -> None:
        oracle.extension("Male")
        oracle.close()
        assert "Male" in oracle._cache

    def test_close_is_idempotent(
        self, ontology_path: Path, tmp_path: Path, wiring
    ) -> None:
        wiring()
        cache = tmp_path / "cache.json"
        instance = ExtensionOracle(
            ontology_path=ontology_path, cache_path=cache
        )
        instance.extension("Male")
        instance.close()
        instance.close()
        assert json.loads(cache.read_text(encoding="utf-8")) == {
            "Male": [f"{NS}a", f"{NS}b"]
        }

    def test_context_manager_exit_persists_the_cache(
        self, ontology_path: Path, tmp_path: Path, wiring
    ) -> None:
        wiring()
        cache = tmp_path / "cache.json"
        with ExtensionOracle(
            ontology_path=ontology_path, cache_path=cache
        ) as instance:
            instance.extension("Person")
        assert "Person" in json.loads(cache.read_text(encoding="utf-8"))

    def test_cache_is_persisted_even_when_the_body_raises(
        self, ontology_path: Path, tmp_path: Path, wiring
    ) -> None:
        # Reasoning is the expensive stage; a crash downstream must not
        # discard work already paid for.
        wiring()
        cache = tmp_path / "cache.json"
        with pytest.raises(RuntimeError):
            with ExtensionOracle(
                ontology_path=ontology_path, cache_path=cache
            ) as instance:
                instance.extension("Male")
                raise RuntimeError("learner blew up")
        assert "Male" in json.loads(cache.read_text(encoding="utf-8"))

    def test_empty_cache_is_written_as_an_empty_object(
        self, ontology_path: Path, tmp_path: Path, wiring
    ) -> None:
        wiring()
        cache = tmp_path / "cache.json"
        with ExtensionOracle(ontology_path=ontology_path, cache_path=cache):
            pass
        assert json.loads(cache.read_text(encoding="utf-8")) == {}


class TestCacheFormat:
    def test_keys_and_values_are_sorted_for_byte_stability(
        self, ontology_path: Path, tmp_path: Path, wiring
    ) -> None:
        # Deterministic bytes make the artifact diffable across runs.
        wiring(
            extensions={
                "Zeta": {f"{NS}c", f"{NS}a", f"{NS}b"},
                "Alpha": {f"{NS}b"},
            }
        )
        cache = tmp_path / "cache.json"
        with ExtensionOracle(
            ontology_path=ontology_path, cache_path=cache
        ) as instance:
            instance.extension("Zeta")
            instance.extension("Alpha")
        payload = json.loads(cache.read_text(encoding="utf-8"))
        assert list(payload) == ["Alpha", "Zeta"]
        assert payload["Zeta"] == sorted(payload["Zeta"])

    def test_write_is_byte_identical_across_insertion_orders(
        self, ontology_path: Path, tmp_path: Path, wiring
    ) -> None:
        wiring(extensions={"Alpha": {f"{NS}b"}, "Zeta": {f"{NS}a", f"{NS}c"}})
        first, second = tmp_path / "a.json", tmp_path / "b.json"
        with ExtensionOracle(
            ontology_path=ontology_path, cache_path=first
        ) as instance:
            instance.extension("Alpha")
            instance.extension("Zeta")
        with ExtensionOracle(
            ontology_path=ontology_path, cache_path=second
        ) as instance:
            instance.extension("Zeta")
            instance.extension("Alpha")
        assert first.read_bytes() == second.read_bytes()

    def test_non_ascii_expressions_are_written_unescaped(
        self, ontology_path: Path, tmp_path: Path, wiring
    ) -> None:
        # DL syntax is full of non-ASCII operators; escaping them would
        # make the artifact unreadable.
        wiring()
        cache = tmp_path / "cache.json"
        with ExtensionOracle(
            ontology_path=ontology_path, cache_path=cache
        ) as instance:
            instance.extension("∃ hasChild.Person")
        assert "∃ hasChild.Person" in cache.read_text(encoding="utf-8")

    def test_dl_operators_round_trip_unchanged(
        self, ontology_path: Path, tmp_path: Path, wiring
    ) -> None:
        expressions = [
            "Male ⊓ Female",
            "∃ hasChild.Person",
            "¬Male ⊔ Person",
            "≥ 2 hasChild.⊤",
        ]
        wiring(extensions={name: {f"{NS}a"} for name in expressions})
        cache = tmp_path / "cache.json"
        with ExtensionOracle(
            ontology_path=ontology_path, cache_path=cache
        ) as instance:
            for expression in expressions:
                instance.extension(expression)
        reloaded = ExtensionOracle(
            ontology_path=ontology_path, cache_path=cache
        )
        assert set(reloaded._cache) == set(expressions)


class TestCacheRoundTrip:
    def test_reload_reproduces_every_extension(
        self, ontology_path: Path, tmp_path: Path, wiring
    ) -> None:
        wiring()
        cache = tmp_path / "cache.json"
        with ExtensionOracle(
            ontology_path=ontology_path, cache_path=cache
        ) as writer:
            for expression in ("Male", "Female", "Male ⊓ Female"):
                writer.extension(expression)
            original = dict(writer._cache)

        reader = ExtensionOracle(ontology_path=ontology_path, cache_path=cache)
        assert reader._cache == original

    def test_reloaded_values_are_frozensets(
        self, ontology_path: Path, tmp_path: Path, wiring
    ) -> None:
        wiring()
        cache = tmp_path / "cache.json"
        cache.write_text(json.dumps({"Male": [f"{NS}a"]}), encoding="utf-8")
        reader = ExtensionOracle(ontology_path=ontology_path, cache_path=cache)
        assert isinstance(reader._cache["Male"], frozenset)

    def test_reload_logs_the_entry_count(
        self,
        ontology_path: Path,
        tmp_path: Path,
        wiring,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        wiring()
        cache = tmp_path / "cache.json"
        cache.write_text(
            json.dumps({"Male": [f"{NS}a"], "Female": [f"{NS}c"]}),
            encoding="utf-8",
        )
        with caplog.at_level(logging.INFO, logger="src.eval.reasoning"):
            ExtensionOracle(ontology_path=ontology_path, cache_path=cache)
        assert any("Loaded 2 cached" in r.message for r in caplog.records)

    def test_a_second_run_extends_rather_than_replaces_the_cache(
        self, ontology_path: Path, tmp_path: Path, wiring
    ) -> None:
        # Successive seeds share one cache file per KB; the union must grow.
        wiring()
        cache = tmp_path / "cache.json"
        with ExtensionOracle(
            ontology_path=ontology_path, cache_path=cache
        ) as first:
            first.extension("Male")
        with ExtensionOracle(
            ontology_path=ontology_path, cache_path=cache
        ) as second:
            second.extension("Female")
        payload = json.loads(cache.read_text(encoding="utf-8"))
        assert set(payload) == {"Male", "Female"}

    def test_reloaded_extension_matches_a_freshly_reasoned_one(
        self, ontology_path: Path, tmp_path: Path, wiring
    ) -> None:
        wiring()
        cache = tmp_path / "cache.json"
        with ExtensionOracle(
            ontology_path=ontology_path, cache_path=cache
        ) as writer:
            expected = writer.extension("Person")
        reader = ExtensionOracle(ontology_path=ontology_path, cache_path=cache)
        assert reader.extension("Person") == expected


# --------------------------------------------------------------------------
# namespace resolution
# --------------------------------------------------------------------------


class TestDefaultNamespace:
    @pytest.mark.parametrize(
        ("ontology_iri", "expected"),
        [
            ("http://example.org/kb", "http://example.org/kb#"),
            ("http://example.org/kb#", "http://example.org/kb#"),
            ("http://example.org/kb/", "http://example.org/kb/"),
            (
                "http://dl-learner.org/carcinogenesis",
                "http://dl-learner.org/carcinogenesis#",
            ),
            (
                "http://dl-learner.org/mutagenesis#",
                "http://dl-learner.org/mutagenesis#",
            ),
            ("http://vicodi.org/ontology", "http://vicodi.org/ontology#"),
        ],
    )
    def test_separator_is_appended_only_when_absent(
        self, ontology_iri: str, expected: str
    ) -> None:
        ontology = FakeOntology(
            individuals=[], classes=[], ontology_iri=ontology_iri
        )
        assert _default_namespace(ontology) == expected

    def test_result_always_ends_with_a_separator(self) -> None:
        for iri in ("http://a/b", "http://a/b#", "http://a/b/"):
            ontology = FakeOntology(
                individuals=[], classes=[], ontology_iri=iri
            )
            assert _default_namespace(ontology).endswith(("#", "/"))

    def test_namespace_is_idempotent(self) -> None:
        once = _default_namespace(
            FakeOntology(individuals=[], classes=[], ontology_iri="http://a/b")
        )
        twice = _default_namespace(
            FakeOntology(individuals=[], classes=[], ontology_iri=once)
        )
        assert once == twice

    def test_parser_receives_the_resolved_namespace(
        self, oracle: ExtensionOracle
    ) -> None:
        oracle.extension("Male")
        parser: FakeParser = oracle._state["parser"]  # type: ignore
        assert parser.namespace == "http://example.org/kb#"


# --------------------------------------------------------------------------
# integration with metrics
# --------------------------------------------------------------------------


class TestOracleMetricsIntegration:
    """The oracle's outputs must drop straight into `score_extensions`."""

    def test_oracle_outputs_score_without_conversion(
        self, oracle: ExtensionOracle
    ) -> None:
        from src.eval.metrics import score_extensions

        metrics = score_extensions(
            hypothesis_extension=oracle.extension("Male"),
            target_extension=oracle.extension("Person"),
            universe_size=len(oracle.universe),
            atomic_baseline=None,
        )
        assert metrics.precision == 1.0
        assert metrics.recall == pytest.approx(2 / 3)
        assert metrics.abl is None

    def test_confusion_matrix_is_consistent_for_real_extensions(
        self, oracle: ExtensionOracle
    ) -> None:
        # Extensions drawn from the same universe can never produce a
        # negative cell -- the inconsistency check guards wiring errors.
        from src.eval.metrics import confusion_matrix

        universe = oracle.universe
        for expression in ("Male", "Female", "Person", "Male ⊓ Female"):
            matrix = confusion_matrix(
                hypothesis_extension=oracle.extension(expression),
                target_extension=oracle.extension("Person"),
                universe_size=len(universe),
            )
            assert matrix.consistent

    def test_atomic_baseline_is_the_best_single_class_f1(
        self, oracle: ExtensionOracle
    ) -> None:
        from src.eval.metrics import score_extensions

        target = oracle.extension("Person")
        universe_size = len(oracle.universe)
        baseline = max(
            score_extensions(
                hypothesis_extension=extension,
                target_extension=target,
                universe_size=universe_size,
                atomic_baseline=None,
            ).f1
            for extension in oracle.atomic_extensions.values()
        )
        # Person is itself a named class, so the baseline is attainable.
        assert baseline == pytest.approx(1.0)

    def test_redundant_target_is_detectable_from_atomic_extensions(
        self, oracle: ExtensionOracle
    ) -> None:
        # The `redundant` hardness flag: target extension equals some
        # atomic class extension.
        target = oracle.extension("Person")
        assert any(
            extension == target
            for extension in oracle.atomic_extensions.values()
        )

    def test_unparseable_hypothesis_is_excluded_before_scoring(
        self, oracle: ExtensionOracle
    ) -> None:
        # Tier-3 semantics: no metrics object is ever built for a malformed
        # hypothesis, so it cannot contribute a spurious F1 = 0.
        assert oracle.extension_or_none("Male ⊓ ((") is None


# --------------------------------------------------------------------------
# real reasoner (opt-in)
# --------------------------------------------------------------------------

FAMILY_ONTOLOGY = f"""<?xml version="1.0"?>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
         xmlns:owl="http://www.w3.org/2002/07/owl#"
         xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#"
         xml:base="http://example.org/family"
         xmlns="http://example.org/family#">
  <owl:Ontology rdf:about="http://example.org/family"/>
  <owl:Class rdf:about="#Person"/>
  <owl:Class rdf:about="#Male"><rdfs:subClassOf rdf:resource="#Person"/></owl:Class>
  <owl:Class rdf:about="#Female"><rdfs:subClassOf rdf:resource="#Person"/></owl:Class>
  <owl:ObjectProperty rdf:about="#hasChild"/>
  <Male rdf:about="#alice_father"/>
  <Female rdf:about="#alice"/>
  <Male rdf:about="#bob"/>
</rdf:RDF>
"""


@pytest.fixture
def real_ontology(tmp_path: Path) -> Path:
    path = tmp_path / "family.owl"
    path.write_text(FAMILY_ONTOLOGY, encoding="utf-8")
    return path


@pytest.mark.integration
class TestRealReasoner:
    """Opt-in checks against OWLAPY/HermiT. Run with `-m integration`."""

    def test_universe_matches_the_asserted_individuals(
        self, real_ontology: Path
    ) -> None:
        pytest.importorskip("owlapy")
        with ExtensionOracle(ontology_path=real_ontology) as oracle:
            local = {iri.rsplit("#", 1)[-1] for iri in oracle.universe}
            assert local == {"alice_father", "alice", "bob"}

    def test_atomic_extensions_respect_subsumption(
        self, real_ontology: Path
    ) -> None:
        # `direct=False` must pull Male/Female instances into Person.
        pytest.importorskip("owlapy")
        with ExtensionOracle(ontology_path=real_ontology) as oracle:
            atomic = oracle.atomic_extensions
            person = next(v for k, v in atomic.items() if k.endswith("#Person"))
            assert len(person) == 3

    def test_thing_and_nothing_are_absent(self, real_ontology: Path) -> None:
        pytest.importorskip("owlapy")
        with ExtensionOracle(ontology_path=real_ontology) as oracle:
            assert THING not in oracle.atomic_extensions
            assert NOTHING not in oracle.atomic_extensions

    def test_dl_conjunction_is_parsed_and_reasoned(
        self, real_ontology: Path
    ) -> None:
        pytest.importorskip("owlapy")
        with ExtensionOracle(ontology_path=real_ontology) as oracle:
            assert oracle.extension("Male ⊓ Female") == frozenset()
            assert len(oracle.extension("Male")) == 2

    def test_garbage_raises_unparseable(self, real_ontology: Path) -> None:
        pytest.importorskip("owlapy")
        with ExtensionOracle(ontology_path=real_ontology) as oracle:
            assert oracle.extension_or_none("⊓⊓ ((( not dl") is None

    def test_cache_reload_matches_fresh_reasoning(
        self, real_ontology: Path, tmp_path: Path
    ) -> None:
        pytest.importorskip("owlapy")
        cache = tmp_path / "cache.json"
        with ExtensionOracle(
            ontology_path=real_ontology, cache_path=cache
        ) as first:
            expected = first.extension("Male")
        with ExtensionOracle(
            ontology_path=real_ontology, cache_path=cache
        ) as second:
            assert second.extension("Male") == expected
            assert second._reasoner is None  # never loaded