import pytest
from pydantic import ValidationError

from supervisor.types import ChoiceOption, ChoiceQuestion, ScoreQuestion


def test_choice_question_needs_two_options():
    with pytest.raises(ValidationError):
        ChoiceQuestion(prompt="p", options=[ChoiceOption(label="A", text="only one")])


def test_choice_question_rejects_duplicate_labels():
    with pytest.raises(ValidationError):
        ChoiceQuestion(
            prompt="p",
            options=[ChoiceOption(label="A", text="x"), ChoiceOption(label="A", text="y")],
        )


def test_score_question_valid():
    q = ScoreQuestion(
        prompt="p",
        scale=[ChoiceOption(label="A", text="low"), ChoiceOption(label="B", text="high")],
    )
    assert len(q.scale) == 2
