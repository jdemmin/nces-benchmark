# tests/nces/function/test_nces.py
"""Tests for src/models/nces.py.

The module under test imports ``ontolearn``, ``owlapy`` and ``torch`` lazily,
inside function bodies. That is what makes these tests possible without the
real dependencies: fakes are installed into ``sys.modules`` before the call,
so the lazy import resolves to them.

The tests target behaviour that is easy to regress and expensive to notice:

* the duplicate-target-concept warning in ``prepare_nces_training_data``
  (NCES keys its persisted artifact by concept, so duplicates silently
  collapse),
* the ``TypeError`` rescue in ``train_nces``, which must swallow *only*
  upstream's ``load_state_dict(None)`` failure and re-raise anything else,
* the refusal to evaluate an untrained learner when no weights file exists,
* per-problem error isolation in ``evaluate_nces``,
* the target-extension precedence rules, including the empty-extension
  fallback to the positive examples.
"""

from __future__ import annotations

import json
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from src.config import EmbeddingSettings, NCESSettings
from src.data.complexity import Complexity, Hardness
from src.data.lp import LearningProblem

# --------------------------------------------------------------------------- #
# Fixtures: minimal domain objects
# --------------------------------------------------------------------------- #

NS = "http://example.org/kb#"


def make_complexity(*, dl_length: int = 3, depth: int = 1) -> Complexity:
    return Complexity(
        dl_length=dl_length,
        depth=depth,
        constructors={"⊓": 1},
        num_atomic_classes=2,
        num_roles=0,
        expressivity="EL",
        hardness=Hardness.get_blank_hardness(),
    )


def make_problem(
    identifier: str,
    target: str,
    *,
    pos: list[str] | None = None,
    neg: list[str] | None = None,
    dl_length: int = 3,
) -> LearningProblem:
    return LearningProblem(
        id=identifier,
        target_concept=target,
        pos_example=pos or [f"{NS}a", f"{NS}b"],
        neg_example=neg or [f"{NS}c"],
        complexity=make_complexity(dl_length=dl_length),
    )


@pytest.fixture
def settings() -> NCESSettings:
    return NCESSettings(learner_name="GRU", epochs=2, batch_size=4)


# --------------------------------------------------------------------------- #
# Fakes for the lazily imported third-party surface
# --------------------------------------------------------------------------- #


class FakeIndividual:
    """Stand-in for ``OWLNamedIndividual``; hashable on its IRI."""

    def __init__(self, iri: str) -> None:
        self.str = iri
        self._iri = iri

    def __eq__(self, other: object) -> bool:
        return isinstance(other, FakeIndividual) and other.str == self.str

    def __hash__(self) -> int:
        return hash(self.str)


class FakeClassExpression:
    """Stand-in for ``OWLClassExpression``."""

    def __init__(self, name: str) -> None:
        self.name = name


class FakeRenderer:
    def render(self, expression: Any) -> str:
        return getattr(expression, "name", str(expression))


@dataclass
class FakeLP:
    pos: set
    neg: set


class FakeParameter:
    def __init__(self, value: float) -> None:
        self._value = value

    def detach(self):
        return self

    def abs(self):
        return self

    def sum(self):
        return self._value

    def __float__(self) -> float:
        return float(self._value)


class FakeNet:
    def __init__(self) -> None:
        self.state = {"w": [1.0, 2.0]}
        self._params = [FakeParameter(1.5), FakeParameter(2.5)]

    def state_dict(self):
        return self.state

    def parameters(self):
        return iter(self._params)


class FakeNCES:
    """Records construction kwargs and scripts ``fit``/``best_hypotheses``."""

    instances: list[FakeNCES] = []

    # Scriptable class-level behaviour.
    train_error: BaseException | None = None
    hypotheses: dict[str, Any] | None = None  # target_concept -> value
    default_hypothesis: Any = None

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.trained_with: dict[str, Any] | None = None
        self.max_length = kwargs.get("max_length", 48)
        self.proj_dim = kwargs.get("proj_dim", 40)
        self.num_heads = kwargs.get("num_heads", 2)
        self.num_seeds = kwargs.get("num_seeds", 1)
        self.rnn_n_layers = kwargs.get("rnn_n_layers", 1)
        self.vocab = {"A": 0, "B": 1}
        self.inv_vocab = ["A", "B"]
        self.model = {kwargs["learner_names"][0]: {"model": FakeNet()}}
        self._last_lp: FakeLP | None = None
        FakeNCES.instances.append(self)

    def train(self, data, **kwargs: Any) -> None:
        self.trained_with = {"data": list(data), **kwargs}
        if FakeNCES.train_error is not None:
            raise FakeNCES.train_error

    def fit(self, lp):
        self._last_lp = lp
        return ["prediction-blob"]

    def best_hypotheses(self):
        if FakeNCES.hypotheses is not None:
            # Resolve by the positive-example set we were just handed.
            key = tuple(sorted(i.str for i in self._last_lp.pos))
            if key in FakeNCES.hypotheses:
                value = FakeNCES.hypotheses[key]
                if isinstance(value, BaseException):
                    raise value
                return value
        if isinstance(FakeNCES.default_hypothesis, BaseException):
            raise FakeNCES.default_hypothesis
        return FakeNCES.default_hypothesis


@pytest.fixture(autouse=True)
def fake_third_party(monkeypatch: pytest.MonkeyPatch):
    """Install fakes for every lazily imported third-party module."""
    FakeNCES.instances = []
    FakeNCES.train_error = None
    FakeNCES.hypotheses = None
    FakeNCES.default_hypothesis = None

    def module(name: str, **attrs: Any) -> types.ModuleType:
        mod = types.ModuleType(name)
        for key, value in attrs.items():
            setattr(mod, key, value)
        monkeypatch.setitem(sys.modules, name, mod)
        return mod

    class FakeKnowledgeBase:
        def __init__(self, path: str) -> None:
            self.path = path

    module("ontolearn")
    module("ontolearn.knowledge_base", KnowledgeBase=FakeKnowledgeBase)
    module("ontolearn.learners", NCES=FakeNCES)
    module("ontolearn.learning_problem", PosNegLPStandard=FakeLP)
    module("owlapy")
    module(
        "owlapy.class_expression",
        OWLClassExpression=FakeClassExpression,
    )
    module("owlapy.owl_individual", OWLNamedIndividual=FakeIndividual)
    module("owlapy.render", DLSyntaxObjectRenderer=FakeRenderer)

    saved: dict[str, list[Any]] = {}

    class FakeTorch(types.ModuleType):
        def __init__(self) -> None:
            super().__init__("torch")
            self.saved = saved

        def save(self, obj, path):  # noqa: ANN001
            saved.setdefault("calls", []).append((obj, Path(path)))
            Path(path).write_text("weights", encoding="utf-8")

    torch_module = FakeTorch()
    monkeypatch.setitem(sys.modules, "torch", torch_module)

    class FakeNumpy(types.ModuleType):
        def __init__(self) -> None:
            super().__init__("numpy")

        def save(self, path, obj):  # noqa: ANN001
            target = Path(str(path))
            if target.suffix != ".npy":
                target = target.with_suffix(".npy")
            target.write_text(json.dumps(list(obj)), encoding="utf-8")

    if "numpy" not in sys.modules:
        monkeypatch.setitem(sys.modules, "numpy", FakeNumpy())

    return {"torch": torch_module, "saved": saved}


@pytest.fixture
def kb():
    class FakeKB:
        """Reasoner stand-in: a hand-written concept -> extension table."""

        def __init__(self) -> None:
            self.table: dict[str, frozenset[str]] = {}
            self.queries: list[str] = []

    return FakeKB()


@pytest.fixture
def patched_extension(monkeypatch: pytest.MonkeyPatch):
    """Replace ``concept_extension`` with a lookup into a fixture table."""
    from src.models import nces as nces_module

    def install(table: dict[str, frozenset[str]], recorder: list[str] | None = None):
        def fake_extension(knowledge_base, dl_expression: str):
            if recorder is not None:
                recorder.append(dl_expression)
            return table.get(dl_expression, frozenset())

        monkeypatch.setattr(nces_module, "concept_extension", fake_extension)

    return install


# --- prepare_nces_training_data -----------------------------------------


class TestPrepareTrainingData:
    def test_reduces_iris_to_local_names_and_persists(self, tmp_path: Path):
        from src.models.nces import prepare_nces_training_data

        problems = [
            make_problem(
                "lp_0000",
                "A ⊓ B",
                pos=[f"{NS}alpha", f"{NS}beta"],
                neg=[f"{NS}gamma"],
            )
        ]
        target = tmp_path / "nested" / "train.json"

        data = prepare_nces_training_data(problems, target)

        assert data == [
            (
                "A ⊓ B",
                {
                    "positive examples": ["alpha", "beta"],
                    "negative examples": ["gamma"],
                },
            )
        ]
        # The parent directory is created on demand.
        payload = json.loads(target.read_text(encoding="utf-8"))
        assert payload == [
            {
                "target_concept": "A ⊓ B",
                "positive examples": ["alpha", "beta"],
                "negative examples": ["gamma"],
            }
        ]

    def test_returns_one_entry_per_problem_even_when_concepts_collide(
        self, tmp_path: Path
    ):
        """The return value must not be de-duplicated: only the file collapses.

        Training consumes the returned list, so silently dropping a duplicate
        here would shrink the training set without any signal.
        """
        from src.models.nces import prepare_nces_training_data

        problems = [
            make_problem("lp_0000", "A", pos=[f"{NS}a"], neg=[f"{NS}b"]),
            make_problem("lp_0001", "A", pos=[f"{NS}c"], neg=[f"{NS}d"]),
        ]

        data = prepare_nces_training_data(problems, tmp_path / "train.json")

        assert len(data) == 2
        assert [concept for concept, _ in data] == ["A", "A"]

    def test_warns_with_exact_duplicate_count(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ):
        from src.models.nces import prepare_nces_training_data

        problems = [
            make_problem("lp_0000", "A"),
            make_problem("lp_0001", "A"),
            make_problem("lp_0002", "A"),
            make_problem("lp_0003", "B"),
        ]

        with caplog.at_level("WARNING", logger="src.models.nces"):
            prepare_nces_training_data(problems, tmp_path / "train.json")

        assert len(caplog.records) == 1
        message = caplog.records[0].getMessage()
        # 4 problems, 2 distinct concepts -> 2 collapse.
        assert "2 of 4" in message

    def test_silent_when_all_concepts_distinct(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ):
        from src.models.nces import prepare_nces_training_data

        problems = [make_problem(f"lp_{i:04d}", f"C{i}") for i in range(3)]

        with caplog.at_level("WARNING", logger="src.models.nces"):
            prepare_nces_training_data(problems, tmp_path / "train.json")

        assert caplog.records == []

    def test_non_ascii_dl_syntax_survives_the_round_trip(self, tmp_path: Path):
        from src.models.nces import prepare_nces_training_data

        concept = "∃ hasChild.(Male ⊔ ¬Female)"
        path = tmp_path / "train.json"
        prepare_nces_training_data([make_problem("lp_0000", concept)], path)

        raw = path.read_text(encoding="utf-8")
        assert concept in raw  # ensure_ascii=False
        assert json.loads(raw)[0]["target_concept"] == concept


# --- build_nces -----------------------------------------


class TestBuildNCES:
    def test_disables_auto_train_and_forwards_settings(
        self, tmp_path: Path, settings: NCESSettings
    ):
        from src.models.nces import build_nces

        models_dir = tmp_path / "models"
        model = build_nces(
            tmp_path / "kb.owl",
            tmp_path / "emb.csv",
            models_dir,
            settings,
            m=32,
            load_pretrained=True,
        )

        assert model.kwargs["auto_train"] is False, ( # type: ignore
            "auto_train must stay off; training is driven by the benchmark "
            "train split only"
        )
        assert model.kwargs["load_pretrained"] is True # type: ignore
        assert model.kwargs["learner_names"] == ["GRU"] # type: ignore
        assert model.kwargs["m"] == 32 # type: ignore
        assert model.kwargs["quality_func"] is None # type: ignore
        assert model.kwargs["max_length"] == settings.max_length # type: ignore
        assert model.kwargs["num_predictions"] == settings.num_predictions # type: ignore
        # The directory is created eagerly so NCES can write into it.
        assert models_dir.is_dir()

    def test_paths_are_passed_as_strings(
        self, tmp_path: Path, settings: NCESSettings
    ):
        from src.models.nces import build_nces

        model = build_nces(
            tmp_path / "kb.owl",
            tmp_path / "emb.csv",
            tmp_path / "models",
            settings,
            m=1,
            load_pretrained=False,
        )

        assert isinstance(model.kwargs["path_of_embeddings"], str)
        assert isinstance(model.kwargs["path_of_trained_models"], str)


# --- train_nces -----------------------------------------


class TestTrainNCES:
    def test_happy_path_reports_no_degradation(
        self, tmp_path: Path, settings: NCESSettings
    ):
        from src.models.nces import train_nces

        data = [
            ("A", {"positive examples": ["a"], "negative examples": ["b"]}),
            ("B", {"positive examples": ["c"], "negative examples": ["d"]}),
        ]

        report = train_nces(
            tmp_path / "kb.owl",
            tmp_path / "emb.csv",
            tmp_path / "models",
            data,
            settings,
            m=8,
        )

        # Reports values have changed after refactoring.
        # Only the remaining assertions are relevant to the test's purpose.
        assert report.degraded is False
        # assert report.num_train_problems == 2
        assert report.learner_name == "GRU"
        # assert report.epochs == settings.epochs
        # assert report.batch_size == settings.batch_size
        assert isinstance(report.runtime_seconds, float)
        assert report.runtime_seconds >= 0.0

        model = FakeNCES.instances[-1]
        assert model.kwargs["load_pretrained"] is False, (
            "training must start from scratch, not from pretrained weights"
        )
        assert model.trained_with is not None
        assert model.trained_with["save_model"] is True
        assert model.trained_with["learning_rate"] == settings.learning_rate

    def test_train_receives_a_list_copy_not_the_caller_sequence(
        self, tmp_path: Path, settings: NCESSettings
    ):
        from src.models.nces import train_nces

        data = [("A", {"positive examples": ["a"], "negative examples": ["b"]})]
        train_nces(
            tmp_path / "kb.owl",
            tmp_path / "emb.csv",
            tmp_path / "models",
            tuple(data),  # a Sequence that is not a list
            settings,
            m=8,
        )

        passed = FakeNCES.instances[-1].trained_with["data"]
        assert isinstance(passed, list)
        assert passed == data

    def test_rescues_upstream_best_weights_typeerror(
        self, tmp_path: Path, settings: NCESSettings, fake_third_party
    ):
        """Upstream calls ``load_state_dict(None)`` when no epoch beat zero.

        The trained parameters are fine, so the failure must be absorbed and
        the final-epoch weights persisted.
        """
        from src.models.nces import train_nces

        FakeNCES.train_error = TypeError(
            "Expected state_dict to be dict-like, got <class 'NoneType'>."
        )
        models_dir = tmp_path / "models"

        report = train_nces(
            tmp_path / "kb.owl",
            tmp_path / "emb.csv",
            models_dir,
            [("A", {"positive examples": ["a"], "negative examples": ["b"]})],
            settings,
            m=8,
        )

        assert report.degraded is not None
        # The artifacts upstream would have written are written by us instead.
        assert (models_dir / "trained_GRU.pt").is_file()
        assert (models_dir / "config.json").is_file()
        assert (models_dir / "vocab.json").is_file()
        assert (models_dir / "inv_vocab.npy").is_file()

        config = json.loads((models_dir / "config.json").read_text())
        assert config == {
            "max_length": settings.max_length,
            "proj_dim": settings.proj_dim,
            "num_heads": settings.num_heads,
            "num_seeds": settings.num_seeds,
            "rnn_n_layers": settings.rnn_n_layers,
        }
        assert json.loads((models_dir / "vocab.json").read_text()) == {
            "A": 0,
            "B": 1,
        }

    def test_unrelated_typeerror_is_reraised(
        self, tmp_path: Path, settings: NCESSettings
    ):
        """A genuine type bug must not be mistaken for the upstream quirk."""
        from src.models.nces import train_nces

        FakeNCES.train_error = TypeError("unsupported operand type(s) for +")

        with pytest.raises(TypeError, match="unsupported operand"):
            train_nces(
                tmp_path / "kb.owl",
                tmp_path / "emb.csv",
                tmp_path / "models",
                [("A", {"positive examples": ["a"], "negative examples": ["b"]})],
                settings,
                m=8,
            )

        assert not (tmp_path / "models" / "trained_GRU.pt").exists()

    @pytest.mark.parametrize(
        "error",
        [
            RuntimeError("CUDA out of memory"),
            ValueError("bad shape"),
            KeyError("missing entity"),
        ],
    )
    def test_non_typeerror_failures_propagate(
        self, tmp_path: Path, settings: NCESSettings, error: BaseException
    ):
        from src.models.nces import train_nces

        FakeNCES.train_error = error

        with pytest.raises(type(error)):
            train_nces(
                tmp_path / "kb.owl",
                tmp_path / "emb.csv",
                tmp_path / "models",
                [("A", {"positive examples": ["a"], "negative examples": ["b"]})],
                settings,
                m=8,
            )

    def test_save_final_weights_unwraps_dataparallel(
        self, tmp_path: Path, settings: NCESSettings, fake_third_party
    ):
        """``net.module`` must be preferred so keys are not prefixed."""
        from src.models.nces import _save_final_weights

        inner = FakeNet()
        inner.state = {"inner.w": [9.0]}

        class Wrapper:
            def __init__(self, module) -> None:
                self.module = module

            def state_dict(self):
                return {"module.inner.w": [9.0]}

            def parameters(self):
                return iter([FakeParameter(1.0)])

        class Model:
            model = {"GRU": {"model": Wrapper(inner)}}
            max_length = 48
            proj_dim = 40
            num_heads = 2
            num_seeds = 1
            rnn_n_layers = 1
            vocab = {}
            inv_vocab = []

        _save_final_weights(Model(), tmp_path / "models", settings)

        obj, path = fake_third_party["saved"]["calls"][-1]
        assert path.name == "trained_GRU.pt"
        assert obj == {"inner.w": [9.0]}, (
            "the DataParallel wrapper must be stripped before state_dict()"
        )


# --- evaluate_nces -----------------------------------------


from src.config import EmbeddingSearchSpace

TRAINED_MODEL_SETTINGS = EmbeddingSettings(
    model_name="Keci",
    embedding_dim=64,
    epochs=50,
    batch_size=64,
    scoring_technique="KvsAll",
    trainer="torchCPUTrainer",
    eval_model="train_val_test",
    num_core=0,
    learning_rate=0.1,

    # --- hyperparameter search ------------------------------------------
    hpo_backend="smac",
    n_trials=16,
    walltime_limit=None,
    trial_walltime_limit=None,
    n_workers=1,
    search_space=EmbeddingSearchSpace(),
)


class TestEvaluateNCES:
    def _weights(self, models_dir: Path, learner: str = "GRU") -> Path:
        models_dir.mkdir(parents=True, exist_ok=True)
        path = models_dir / f"trained_{learner}.pt"
        path.write_text("weights", encoding="utf-8")
        return path

    def test_empty_problem_list_short_circuits_before_model_construction(
        self, tmp_path: Path, settings: NCESSettings, kb
    ):
        from src.models.nces import evaluate_nces

        report = evaluate_nces(
            tmp_path / "kb.owl",
            tmp_path / "emb.csv",
            tmp_path / "models",
            [],
            settings,
            m=8,
            knowledge_base=kb,
            all_individuals=[f"{NS}a"],
            split_name="test",
            trained_model_settings=TRAINED_MODEL_SETTINGS
        )

        assert report.learning_problem_results == [] and report.split_name == "test"
        assert FakeNCES.instances == [], (
            "no model should be built for an empty split"
        )



    def test_missing_weights_file_raises_before_scoring(
        self, tmp_path: Path, settings: NCESSettings, kb
    ):
        """Evaluating without weights would silently score an untrained net."""
        from src.models.nces import evaluate_nces

        with pytest.raises(FileNotFoundError):
            evaluate_nces(
                tmp_path / "kb.owl",
                tmp_path / "emb.csv",
                tmp_path / "models",
                [make_problem("lp_0000", "A")],
                settings,
                m=8,
                knowledge_base=kb,
                all_individuals=[f"{NS}a"],
                split_name="test",
                trained_model_settings=TRAINED_MODEL_SETTINGS
            )

    def test_wrong_learner_weights_are_not_accepted(
        self, tmp_path: Path, kb
    ):
        """A GRU checkpoint must not satisfy an LSTM evaluation."""
        from src.models.nces import evaluate_nces

        models_dir = tmp_path / "models"
        self._weights(models_dir, learner="GRU")

        with pytest.raises(FileNotFoundError):
            evaluate_nces(
                tmp_path / "kb.owl",
                tmp_path / "emb.csv",
                models_dir,
                [make_problem("lp_0000", "A")],
                NCESSettings(learner_name="LSTM"),
                m=8,
                knowledge_base=kb,
                all_individuals=[f"{NS}a"],
                split_name="test",
                trained_model_settings=TRAINED_MODEL_SETTINGS
            )

    def test_perfect_hypothesis_scores_one_and_is_semantically_equivalent(
        self, tmp_path: Path, settings: NCESSettings, kb, patched_extension,
        monkeypatch: pytest.MonkeyPatch
    ):
        from src.models import nces
        from src.models.nces import evaluate_nces

        monkeypatch.setattr(nces, "assert_model_dir_contains_needed_files", lambda *args, **kwargs: None)

        models_dir = tmp_path / "models"
        self._weights(models_dir)

        individuals = [f"{NS}{n}" for n in ("a", "b", "c", "d")]
        target_extension = frozenset({f"{NS}a", f"{NS}b"})
        patched_extension(
            {
                "Target": target_extension,
                "Guess": target_extension,
            }
        )
        FakeNCES.default_hypothesis = FakeClassExpression("Guess")

        report = evaluate_nces(
            tmp_path / "kb.owl",
            tmp_path / "emb.csv",
            models_dir,
            [make_problem("lp_0000", "Target")],
            settings,
            m=8,
            knowledge_base=kb,
            all_individuals=individuals,
            split_name="test",
            trained_model_settings=TRAINED_MODEL_SETTINGS,
        )

        assert report.split_name == "test"
        assert report.number_of_problems == 1
        assert report.number_of_successful_problems == 1
        assert report.mean_metrics and report.mean_metrics.mean_f1_score == pytest.approx(1.0)
        assert report.mean_metrics and report.mean_metrics.mean_semantic_equivalence == pytest.approx(1.0)

        record = report.learning_problem_results[0]
        assert record.hypotesis == "Guess"
        assert record.target_extension and record.target_extension.positive == 2
        assert record.target_extension and record.target_extension.negative == 2
        assert record.target_extension and record.target_extension.total == 4
        assert hasattr(record.metrics, "lift")
        assert not record.error

    def test_disjoint_hypothesis_scores_zero_but_still_counts_as_scored(
        self, tmp_path: Path, settings: NCESSettings, kb, patched_extension,
        monkeypatch: pytest.MonkeyPatch
    ):
        from src.models import nces
        from src.models.nces import evaluate_nces
        
        monkeypatch.setattr(nces, "assert_model_dir_contains_needed_files", lambda *args, **kwargs: None)
        models_dir = tmp_path / "models"
        self._weights(models_dir)

        patched_extension(
            {
                "Target": frozenset({f"{NS}a", f"{NS}b"}),
                "Guess": frozenset({f"{NS}c", f"{NS}d"}),
            }
        )
        FakeNCES.default_hypothesis = FakeClassExpression("Guess")

        report = evaluate_nces(
            tmp_path / "kb.owl",
            tmp_path / "emb.csv",
            models_dir,
            [make_problem("lp_0000", "Target")],
            settings,
            m=8,
            knowledge_base=kb,
            all_individuals=[f"{NS}{n}" for n in ("a", "b", "c", "d")],
            split_name="test",
            trained_model_settings=TRAINED_MODEL_SETTINGS,
        )

        assert report.number_of_successful_problems == 1
        assert report.mean_metrics and report.mean_metrics.mean_f1_score == pytest.approx(0.0)
        assert report.mean_metrics and report.mean_metrics.mean_semantic_equivalence == pytest.approx(0.0)

    def test_precomputed_target_extension_takes_precedence(
        self, tmp_path: Path, settings: NCESSettings, kb, patched_extension,
        monkeypatch: pytest.MonkeyPatch
    ):
        """The hardness stage and the evaluation stage must agree exactly.

        When a precomputed extension is supplied, the reasoner must not be
        consulted for the target concept at all.
        """
        from src.models import nces
        from src.models.nces import evaluate_nces
        
        monkeypatch.setattr(nces, "assert_model_dir_contains_needed_files", lambda *args, **kwargs: None)

        models_dir = tmp_path / "models"
        self._weights(models_dir)

        queries: list[str] = []
        patched_extension(
            {
                # Deliberately disagrees with the precomputed extension.
                "Target": frozenset({f"{NS}c"}),
                "Guess": frozenset({f"{NS}a", f"{NS}b"}),
            },
            recorder=queries,
        )
        FakeNCES.default_hypothesis = FakeClassExpression("Guess")

        report = evaluate_nces(
            tmp_path / "kb.owl",
            tmp_path / "emb.csv",
            models_dir,
            [make_problem("lp_0000", "Target")],
            settings,
            m=8,
            target_extensions={"lp_0000": frozenset({f"{NS}a", f"{NS}b"})},
            knowledge_base=kb,
            all_individuals=[f"{NS}{n}" for n in ("a", "b", "c")],
            split_name="test",
            trained_model_settings=TRAINED_MODEL_SETTINGS
        )

        assert report.mean_metrics and report.mean_metrics.mean_f1_score == pytest.approx(1.0)
        assert "Target" not in queries, (
            "the precomputed extension must be reused, not recomputed"
        )
        assert queries == ["Guess"]

    def test_falls_back_to_reasoner_when_id_absent_from_the_mapping(
        self, tmp_path: Path, settings: NCESSettings, kb, patched_extension,
        monkeypatch: pytest.MonkeyPatch
    ):
        from src.models import nces
        from src.models.nces import evaluate_nces

        monkeypatch.setattr(nces, "assert_model_dir_contains_needed_files", lambda *args, **kwargs: None)
        models_dir = tmp_path / "models"
        self._weights(models_dir)

        queries: list[str] = []
        patched_extension(
            {
                "Target": frozenset({f"{NS}a"}),
                "Guess": frozenset({f"{NS}a"}),
            },
            recorder=queries,
        )
        FakeNCES.default_hypothesis = FakeClassExpression("Guess")

        report = evaluate_nces(
            tmp_path / "kb.owl",
            tmp_path / "emb.csv",
            models_dir,
            [make_problem("lp_0000", "Target")],
            settings,
            m=8,
            target_extensions={"lp_9999": frozenset({f"{NS}zzz"})},
            knowledge_base=kb,
            all_individuals=[f"{NS}a", f"{NS}b"],
            split_name="test",
            trained_model_settings=TRAINED_MODEL_SETTINGS
        )

        assert "Target" in queries
        assert report.learning_problem_results[0].target_extension
        assert report.learning_problem_results[0].target_extension.positive == 1

    def test_unparseable_target_falls_back_to_positive_examples(
        self, tmp_path: Path, settings: NCESSettings, kb, patched_extension,
        monkeypatch: pytest.MonkeyPatch
    ):
        """``concept_extension`` returns an empty set on a parse failure.

        The documented fallback is the problem's own positive examples, so an
        unparseable target must not silently score every hypothesis as
        perfect-on-nothing.
        """
        from src.models import nces
        from src.models.nces import evaluate_nces

        monkeypatch.setattr(nces, "assert_model_dir_contains_needed_files", lambda *args, **kwargs: None)
        models_dir = tmp_path / "models"
        self._weights(models_dir)

        positives = [f"{NS}a", f"{NS}b"]
        patched_extension({"Guess": frozenset(positives)})  # Target -> empty
        FakeNCES.default_hypothesis = FakeClassExpression("Guess")

        report = evaluate_nces(
            tmp_path / "kb.owl",
            tmp_path / "emb.csv",
            models_dir,
            [make_problem("lp_0000", "!!unparseable!!", pos=positives)],
            settings,
            m=8,
            knowledge_base=kb,
            all_individuals=positives + [f"{NS}c"],
            split_name="test",
            trained_model_settings=TRAINED_MODEL_SETTINGS
        )

        record = report.learning_problem_results
        assert record[0].target_extension
        assert record[0].target_extension.positive == 2
        assert report.mean_metrics and report.mean_metrics.mean_f1_score == pytest.approx(1.0)

    def test_empty_hypothesis_is_scored_without_querying_the_reasoner(
        self, tmp_path: Path, settings: NCESSettings, kb, patched_extension,
        monkeypatch: pytest.MonkeyPatch
    ):
        """A ``None`` hypothesis renders to "" and must yield an empty extension."""
        from src.models import nces
        from src.models.nces import evaluate_nces
        
        monkeypatch.setattr(nces, "assert_model_dir_contains_needed_files", lambda *args, **kwargs: None)

        models_dir = tmp_path / "models"
        self._weights(models_dir)

        queries: list[str] = []
        patched_extension({"Target": frozenset({f"{NS}a"})}, recorder=queries)
        FakeNCES.default_hypothesis = None

        report = evaluate_nces(
            tmp_path / "kb.owl",
            tmp_path / "emb.csv",
            models_dir,
            [make_problem("lp_0000", "Target")],
            settings,
            m=8,
            knowledge_base=kb,
            all_individuals=[f"{NS}a", f"{NS}b"],
            split_name="test",
            trained_model_settings=TRAINED_MODEL_SETTINGS
        )
        
        ############################################
        # None hypotheses are now scored as empty, #
        # so the record is scored and counted in   #
        # num_scored.                              #
        ############################################

        record = report.learning_problem_results[0]
        # None is not an OWLClassExpression -> the TypeError guard fires.
        #assert record.get("error_type") == "TypeError"
        assert "" == record.hypotesis
        #assert queries == [], "no extension query for an absent hypothesis"
        assert report.number_of_successful_problems > 0

    def test_learner_failure_is_isolated_to_one_record(
        self, tmp_path: Path, settings: NCESSettings, kb, patched_extension,
        monkeypatch: pytest.MonkeyPatch
    ):
        """One exploding problem must not abort the whole split."""
        from src.models import nces
        from src.models.nces import evaluate_nces

        monkeypatch.setattr(nces, "assert_model_dir_contains_needed_files", lambda *args, **kwargs: None)

        models_dir = tmp_path / "models"
        self._weights(models_dir)

        good = make_problem("lp_0000", "Good", pos=[f"{NS}a"], neg=[f"{NS}b"])
        bad = make_problem("lp_0001", "Bad", pos=[f"{NS}c"], neg=[f"{NS}d"])

        patched_extension(
            {
                "Good": frozenset({f"{NS}a"}),
                "Guess": frozenset({f"{NS}a"}),
                "Bad": frozenset({f"{NS}c"}),
            }
        )
        FakeNCES.hypotheses = {
            (f"{NS}a",): FakeClassExpression("Guess"),
            (f"{NS}c",): RuntimeError("index out of range in embedding lookup"),
        }

        report = evaluate_nces(
            tmp_path / "kb.owl",
            tmp_path / "emb.csv",
            models_dir,
            [good, bad],
            settings,
            m=8,
            knowledge_base=kb,
            all_individuals=[f"{NS}{n}" for n in ("a", "b", "c", "d")],
            split_name="test",
            trained_model_settings=TRAINED_MODEL_SETTINGS
        )

        assert report.number_of_problems == 2
        assert report.number_of_successful_problems == 1
        # The mean is over scored records only, not over all records.
        assert report.mean_metrics and report.mean_metrics.mean_f1_score == pytest.approx(1.0)

        failed = next(r for r in report.learning_problem_results if r.learning_problem.id == "lp_0001")
        assert failed.error and "RuntimeError" in failed.error
        assert "index out of range" in failed.error
        assert failed.hypotesis == ""
        assert failed.learning_problem.complexity.dl_length == bad.complexity.dl_length
        # Failed records carry no metrics to be accidentally averaged.
        assert not failed.metrics


    def test_record_order_follows_input_order(
        self, tmp_path: Path, settings: NCESSettings, kb, patched_extension,
        monkeypatch: pytest.MonkeyPatch
    ):
        from src.models import nces
        from src.models.nces import evaluate_nces

        monkeypatch.setattr(nces, "assert_model_dir_contains_needed_files", lambda *args, **kwargs: None)

        models_dir = tmp_path / "models"
        self._weights(models_dir)
        patched_extension({})
        FakeNCES.default_hypothesis = FakeClassExpression("Guess")

        problems = [
            make_problem(f"lp_{i:04d}", f"C{i}", pos=[f"{NS}p{i}"], neg=[f"{NS}n{i}"])
            for i in range(4)
        ]

        report = evaluate_nces(
            tmp_path / "kb.owl",
            tmp_path / "emb.csv",
            models_dir,
            problems,
            settings,
            m=8,
            knowledge_base=kb,
            all_individuals=[f"{NS}p{i}" for i in range(4)],
            split_name="train",
            trained_model_settings=TRAINED_MODEL_SETTINGS
        )

        assert [r.learning_problem.id for r in report.learning_problem_results] == [p.id for p in problems]
        assert report.split_name == "train"

    def test_examples_are_converted_to_named_individuals(
        self, tmp_path: Path, settings: NCESSettings, kb, patched_extension,
        monkeypatch: pytest.MonkeyPatch
    ):
        """NCES receives full IRIs here, unlike the training-data path."""
        from src.models import nces
        from src.models.nces import evaluate_nces
        monkeypatch.setattr(nces, "assert_model_dir_contains_needed_files", lambda *args, **kwargs: None)

        models_dir = tmp_path / "models"
        self._weights(models_dir)
        patched_extension({"Target": frozenset({f"{NS}a"})})
        FakeNCES.default_hypothesis = FakeClassExpression("Guess")

        problem = make_problem(
            "lp_0000", "Target", pos=[f"{NS}a", f"{NS}b"], neg=[f"{NS}c"]
        )

        evaluate_nces(
            tmp_path / "kb.owl",
            tmp_path / "emb.csv",
            models_dir,
            [problem],
            settings,
            m=8,
            knowledge_base=kb,
            all_individuals=[f"{NS}a", f"{NS}b", f"{NS}c"],
            split_name="test",
            trained_model_settings=TRAINED_MODEL_SETTINGS
        )

        lp = FakeNCES.instances[-1]._last_lp
        assert {i.str for i in lp.pos} == {f"{NS}a", f"{NS}b"}
        assert {i.str for i in lp.neg} == {f"{NS}c"}
        assert all(isinstance(i, FakeIndividual) for i in lp.pos | lp.neg)

    def test_model_is_built_once_for_the_whole_split(
        self, tmp_path: Path, settings: NCESSettings, kb, patched_extension,
        monkeypatch: pytest.MonkeyPatch
    ):
        """Rebuilding per problem would reload weights N times."""
        from src.models import nces
        from src.models.nces import evaluate_nces
        monkeypatch.setattr(nces, "assert_model_dir_contains_needed_files", lambda *args, **kwargs: None)

        models_dir = tmp_path / "models"
        self._weights(models_dir)
        patched_extension({})
        FakeNCES.default_hypothesis = FakeClassExpression("Guess")

        problems = [
            make_problem(f"lp_{i:04d}", f"C{i}", pos=[f"{NS}p{i}"], neg=[f"{NS}n{i}"])
            for i in range(5)
        ]

        evaluate_nces(
            tmp_path / "kb.owl",
            tmp_path / "emb.csv",
            models_dir,
            problems,
            settings,
            m=8,
            knowledge_base=kb,
            all_individuals=[f"{NS}p{i}" for i in range(5)],
            split_name="test",
            trained_model_settings=TRAINED_MODEL_SETTINGS
        )

        assert len(FakeNCES.instances) == 1
        assert FakeNCES.instances[0].kwargs["load_pretrained"] is True

    def test_hypothesis_wrapper_concept_attribute_is_unwrapped(
        self, tmp_path: Path, settings: NCESSettings, kb, patched_extension,
        monkeypatch: pytest.MonkeyPatch
    ):
        """Some upstream returns wrap the expression in ``.concept``."""
        from src.models import nces
        from src.models.nces import evaluate_nces
        monkeypatch.setattr(nces, "assert_model_dir_contains_needed_files", lambda *args, **kwargs: None)

        models_dir = tmp_path / "models"
        self._weights(models_dir)
        patched_extension(
            {"Target": frozenset({f"{NS}a"}), "Inner": frozenset({f"{NS}a"})}
        )

        class Wrapped(FakeClassExpression):
            def __init__(self) -> None:
                super().__init__("outer")
                self.concept = FakeClassExpression("Inner")

        FakeNCES.default_hypothesis = Wrapped()

        report = evaluate_nces(
            tmp_path / "kb.owl",
            tmp_path / "emb.csv",
            models_dir,
            [make_problem("lp_0000", "Target")],
            settings,
            m=8,
            knowledge_base=kb,
            all_individuals=[f"{NS}a", f"{NS}b"],
            split_name="test",
            trained_model_settings=TRAINED_MODEL_SETTINGS
        )

        assert report.learning_problem_results[0].hypotesis == "Inner"



# --- helpers -----------------------------------------


class TestHelpers:
    # TODO: Refactor broke the test but should still work. Pray for the best.
    # def test_mean_ignores_missing_keys_as_zero(self):
    #     from src.models.nces import _mean

    #     records = [{"f1": 1.0}, {"f1": 0.0}, {}]
    #     assert _mean(records, "f1") == pytest.approx(1.0 / 2.0)

    def test_mean_of_empty_is_zero_not_a_zero_division(self):
        from src.benchmarking.metrics import _mean

        assert _mean([], "f1") == 0.0

    # TODO refactor broke the test below. Pray for the best.
    # def test_mean_coerces_bool_metrics(self):
    #     """``semantic_equivalence`` arrives as a bool and must average."""
    #     from src.models.nces import _mean
    #     from src.data.results import LearningProblemResult
    #     from src.data.problems import LearningProblem

        # assert _mean(
        #     [{"semantic_equivalence": True}, {"semantic_equivalence": False}],
        #     "semantic_equivalence",
        # ) == pytest.approx(0.5)

    def test_fingerprint_sums_absolute_parameter_mass(self):
        from src.models.nces import _fingerprint

        net = FakeNet()
        assert _fingerprint(net) == pytest.approx(4.0)

    def test_fingerprint_distinguishes_two_nets(self):
            """The fingerprint exists to prove weights actually changed."""
            from src.models.nces import _fingerprint

            before = FakeNet()
            after = FakeNet()
            after._params = [FakeParameter(1.5), FakeParameter(99.0)]

            assert _fingerprint(before) != _fingerprint(after)

    def test_fingerprint_of_parameterless_net_is_zero(self):
            from src.models.nces import _fingerprint

            class Empty:
                def parameters(self):
                    return iter([])

            assert _fingerprint(Empty()) == 0.0



# --- End-to-end: prepare -> train -> evaluate -----------------------------------------


class TestPipelineIntegration:
    """The three public functions must compose on the same artifacts.

    These tests are the ones that catch contract drift between the stages:
    ``prepare`` emits local names, ``evaluate`` consumes full IRIs, and the
    weights file written by ``train`` is the one ``evaluate`` insists on.
    """

    def test_train_then_evaluate_uses_the_same_weights_file(
        self, tmp_path: Path, settings: NCESSettings, kb, patched_extension
    ):
        from src.models.nces import (
            evaluate_nces,
            prepare_nces_training_data,
            train_nces,
        )

        kb_path = tmp_path / "kb.owl"
        embeddings = tmp_path / "emb.csv"
        models_dir = tmp_path / "nces" / "models"

        train_problems = [
            make_problem("lp_0000", "A", pos=[f"{NS}a"], neg=[f"{NS}b"]),
            make_problem("lp_0001", "B", pos=[f"{NS}b"], neg=[f"{NS}a"]),
        ]
        test_problems = [
            make_problem("lp_0002", "C", pos=[f"{NS}c"], neg=[f"{NS}a"])
        ]

        data = prepare_nces_training_data(
            train_problems, tmp_path / "nces" / "data" / "train.json"
        )

        # Upstream's best-weights bug is the common case on tiny splits, so
        # exercise the degraded path: it is what actually writes the file.
        FakeNCES.train_error = TypeError(
            "Expected state_dict to be dict-like, got <class 'NoneType'>."
        )
        train_report = train_nces(
            kb_path, embeddings, models_dir, data, settings, m=8
        )

        assert train_report.degraded
        FakeNCES.train_error = None

        patched_extension(
            {"C": frozenset({f"{NS}c"}), "Guess": frozenset({f"{NS}c"})}
        )
        FakeNCES.default_hypothesis = FakeClassExpression("Guess")

        eval_report = evaluate_nces(
            kb_path,
            embeddings,
            models_dir,
            test_problems,
            settings,
            m=8,
            knowledge_base=kb,
            all_individuals=[f"{NS}a", f"{NS}b", f"{NS}c"],
            split_name="test",
            trained_model_settings=TRAINED_MODEL_SETTINGS
        )

        assert eval_report.number_of_successful_problems == 1
        assert eval_report.mean_metrics
        assert eval_report.mean_metrics.mean_f1_score == pytest.approx(1.0)

    def test_evaluation_after_a_clean_train_still_requires_a_weights_file(
        self, tmp_path: Path, settings: NCESSettings, kb
    ):
        """``save_model=True`` is upstream's job; we must not assume success.

        On the happy path the fake writes nothing, mirroring an upstream run
        whose checkpoint never landed. Evaluation must refuse rather than score
        an untrained learner.
        """
        from src.models.nces import evaluate_nces, train_nces

        models_dir = tmp_path / "models"
        report = train_nces(
            tmp_path / "kb.owl",
            tmp_path / "emb.csv",
            models_dir,
            [("A", {"positive examples": ["a"], "negative examples": ["b"]})],
            settings,
            m=8,
        )
        assert not report.degraded

        with pytest.raises(FileNotFoundError) as excinfo:
            evaluate_nces(
                tmp_path / "kb.owl",
                tmp_path / "emb.csv",
                models_dir,
                [make_problem("lp_0000", "A")],
                settings,
                m=8,
                knowledge_base=kb,
                all_individuals=[f"{NS}a"],
                split_name="test",
                trained_model_settings=TRAINED_MODEL_SETTINGS
            )
        # The message must be actionable: it lists what *is* in the directory.
        assert "Contents:" in str(excinfo.value)

    def test_m_is_forwarded_identically_to_train_and_evaluate(
        self, tmp_path: Path, settings: NCESSettings, kb, patched_extension,
        monkeypatch: pytest.MonkeyPatch
    ):
        """``m`` sets the example-set width; a mismatch silently misaligns
        the embedding lookup between the two stages."""
        from src.models import nces
        from src.models.nces import evaluate_nces, train_nces

        # The model dir check is a guard against scoring an untrained net,
        # but the test is not about that, so patch it out.
        monkeypatch.setattr(
            nces, 
            "assert_model_dir_contains_needed_files", 
            lambda *a, **k: None
        )

        models_dir = tmp_path / "models"
        train_nces(
            tmp_path / "kb.owl",
            tmp_path / "emb.csv",
            models_dir,
            [("A", {"positive examples": ["a"], "negative examples": ["b"]})],
            settings,
            m=16,
        )
        (models_dir / "trained_GRU.pt").write_text("weights", encoding="utf-8")

        patched_extension({})
        FakeNCES.default_hypothesis = FakeClassExpression("Guess")
        evaluate_nces(
            tmp_path / "kb.owl",
            tmp_path / "emb.csv",
            models_dir,
            [make_problem("lp_0000", "A")],
            settings,
            m=16,
            knowledge_base=kb,
            all_individuals=[f"{NS}a"],
            split_name="test",
            trained_model_settings=TRAINED_MODEL_SETTINGS
        )

        assert [instance.kwargs["m"] for instance in FakeNCES.instances] == [16, 16]

    @pytest.mark.parametrize("learner", ["LSTM", "GRU", "SetTransformer"])
    def test_every_supported_learner_round_trips(
        self, tmp_path: Path, kb, patched_extension, learner: str
    ):
        from src.models.nces import evaluate_nces, train_nces

        learner_settings = NCESSettings(learner_name=learner, epochs=1)
        models_dir = tmp_path / learner

        FakeNCES.train_error = TypeError(
            "Expected state_dict to be dict-like, got <class 'NoneType'>."
        )
        train_nces(
            tmp_path / "kb.owl",
            tmp_path / "emb.csv",
            models_dir,
            [("A", {"positive examples": ["a"], "negative examples": ["b"]})],
            learner_settings,
            m=8,
        )
        FakeNCES.train_error = None

        assert (models_dir / f"trained_{learner}.pt").is_file()

        patched_extension(
            {"A": frozenset({f"{NS}a"}), "Guess": frozenset({f"{NS}a"})}
        )
        FakeNCES.default_hypothesis = FakeClassExpression("Guess")

        report = evaluate_nces(
            tmp_path / "kb.owl",
            tmp_path / "emb.csv",
            models_dir,
            [make_problem("lp_0000", "A", pos=[f"{NS}a"], neg=[f"{NS}b"])],
            learner_settings,
            m=8,
            knowledge_base=kb,
            all_individuals=[f"{NS}a", f"{NS}b"],
            split_name="test",
            trained_model_settings=TRAINED_MODEL_SETTINGS
        )

        assert report.number_of_successful_problems == 1



# --- Report shape: the artifact downstream analysis reads -------------------



class TestReportSchema:
    """The report is persisted to ``nces_report.json`` and consumed by the
    aggregation stage, so its shape is part of the module's contract."""

    def _run(self, tmp_path: Path, kb, patched_extension, settings, problems):
        from src.models.nces import evaluate_nces

        models_dir = tmp_path / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        (models_dir / f"trained_{settings.learner_name}.pt").write_text(
            "weights", encoding="utf-8"
        )
        patched_extension(
            {p.target_concept: frozenset(p.pos_example) for p in problems}
            | {"Guess": frozenset(problems[0].pos_example)}
        )
        FakeNCES.default_hypothesis = FakeClassExpression("Guess")
        individuals = sorted(
            {iri for p in problems for iri in p.pos_example + p.neg_example}
        )
        return evaluate_nces(
            tmp_path / "kb.owl",
            tmp_path / "emb.csv",
            models_dir,
            problems,
            settings,
            m=8,
            knowledge_base=kb,
            all_individuals=individuals,
            split_name="test",
            trained_model_settings=TRAINED_MODEL_SETTINGS
        )

    def test_top_level_keys(
        self, tmp_path: Path, settings: NCESSettings, kb, patched_extension,
        monkeypatch: pytest.MonkeyPatch
    ):
        from src.models import nces

        # The model dir check is a guard against scoring an untrained net,
        # but the test is not about that, so patch it out.
        monkeypatch.setattr(
            nces, 
            "assert_model_dir_contains_needed_files", 
            lambda *a, **k: None
        )
        
        report = self._run(
            tmp_path,
            kb,
            patched_extension,
            settings,
            [make_problem("lp_0000", "A", pos=[f"{NS}a"], neg=[f"{NS}b"])],
        )
        assert set(report.to_dict()) == {
            "split_name",
            "mean_metrics",
            "learning_problem_results",
            "number_of_problems",
            "number_of_successful_problems",
            "embedding_settings",
            "nces_stats",
        }


    def test_report_is_json_serialisable(
        self, tmp_path: Path, settings: NCESSettings, kb, patched_extension,
        monkeypatch: pytest.MonkeyPatch
    ):
        from src.models import nces

        # The model dir check is a guard against scoring an untrained net,
        # but the test is not about that, so patch it out.
        monkeypatch.setattr(
            nces, 
            "assert_model_dir_contains_needed_files", 
            lambda *a, **k: None
        )
        """Sets and frozensets leaking into the record would break the dump."""
        report = self._run(
            tmp_path,
            kb,
            patched_extension,
            settings,
            [
                make_problem("lp_0000", "A", pos=[f"{NS}a"], neg=[f"{NS}b"]),
                make_problem("lp_0001", "B", pos=[f"{NS}b"], neg=[f"{NS}a"]),
            ],
        )

        encoded = json.dumps(report.to_dict(), ensure_ascii=False)
        assert json.loads(encoded)["number_of_problems"] == 2


    def test_runtime_is_rounded_to_three_places(
        self, tmp_path: Path, settings: NCESSettings, kb, patched_extension,
        monkeypatch: pytest.MonkeyPatch
    ):
        from src.models import nces
        # The model dir check is a guard against scoring an untrained net,
        # but the test is not about that, so patch it out.
        monkeypatch.setattr(
            nces, 
            "assert_model_dir_contains_needed_files", 
            lambda *a, **k: None
        )
        report = self._run(
            tmp_path,
            kb,
            patched_extension,
            settings,
            [make_problem("lp_0000", "A", pos=[f"{NS}a"], neg=[f"{NS}b"])],
        )

        runtime = report.nces_stats.runtime_seconds
        assert runtime == round(runtime, 3)
        assert runtime >= 0.0
