# tests/conftest.py
import pytest

from src.data.complexity import Complexity, Hardness
from src.data.lp import LearningProblem


@pytest.fixture
def problems() -> list[LearningProblem]:
    return [
        LearningProblem(
            id="lp_0000",
            target_concept="male",
            pos_example=["http://example.com/father#stefan",
                         "http://example.com/father#markus"],
            neg_example=["http://example.com/father#anna",
                         "http://example.com/father#michelle"],
            complexity=Complexity(
                dl_length=1,
                num_atomic_classes=1,
                num_roles=0,
                expressivity="EL",
                hardness=Hardness.get_blank_hardness(),
                depth=0,
                constructors={},
            ),
        )
    ]