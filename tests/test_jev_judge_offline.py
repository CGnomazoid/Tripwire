"""Fast tests for JevJudge.ask against a stubbed Typesafe response - no
network call, no API key.

Mirrors tests/test_judge_offline.py: the stub fixes what Typesafe's
/v1/systemone would return, so assertions are about what JevJudge does with
that response (probability -> logit -> answer shaping), not about Jev's own
model quality.
"""

import math

import pytest

from supervisor.jev_judge import JevJudge
from supervisor.questions import ALLOW_BLOCK, RISK_SCALE
from supervisor.types import ProbabilityQuestion


@pytest.fixture
def make_jev_judge(monkeypatch):
    def factory(response: dict, **kwargs) -> JevJudge:
        calls: list[dict] = []

        def fake_system_one(state, questions, *, api_key=None, model=None, **_):
            calls.append({"state": state, "questions": questions, "api_key": api_key, "model": model})
            return response

        monkeypatch.setattr("supervisor.jev_judge.system_one", fake_system_one)
        judge = JevJudge(model_id="jev-test", **kwargs)
        judge.calls = calls
        return judge

    return factory


def _choice_response(choice: str, probabilities: dict[str, float]) -> dict:
    return {"answers": {"q": {"type": "choice", "choice": choice, "probabilities": probabilities, "confidence": max(probabilities.values())}}}


def _score_response(score: float, probabilities: dict[str, float]) -> dict:
    return {"answers": {"q": {"type": "score", "score": score, "probabilities": probabilities, "confidence": max(probabilities.values())}}}


def _noul_response(p_true: float) -> dict:
    return {"answers": {"q": {"type": "noul", "noul": p_true}}}


def test_choice_answer_is_the_argmax_label(make_jev_judge):
    judge = make_jev_judge(_choice_response("B", {"A": 0.1, "B": 0.9}))
    result = judge.ask("Tool call: run_shell(cmd='rm -rf /')", ALLOW_BLOCK)

    assert result.kind == "choice"
    assert result.answer == "B"
    assert result.ordinal is None
    assert result.probability_true is None
    assert result.confidence == pytest.approx(0.9)
    assert result.raw_probs["A"] == pytest.approx(0.1)
    assert result.raw_probs["B"] == pytest.approx(0.9)


def test_score_answer_carries_the_ordinal(make_jev_judge):
    judge = make_jev_judge(_score_response(2.0, {"0": 0.05, "1": 0.05, "2": 0.9}))
    result = judge.ask("Tool call: drop_table(name='users')", RISK_SCALE)

    assert result.kind == "score"
    assert (result.answer, result.ordinal) == ("C", 2)
    assert RISK_SCALE.scale[result.ordinal].label == result.answer


def test_noul_maps_to_true_false_labels(make_jev_judge):
    judge = make_jev_judge(_noul_response(0.7))
    result = judge.ask("The disk is nearly full.", ProbabilityQuestion(statement="the disk is nearly full"))

    assert result.kind == "probability"
    assert result.answer == "true"
    assert result.probability_true == pytest.approx(0.7)
    assert result.raw_probs["true"] == pytest.approx(0.7)
    assert result.raw_probs["false"] == pytest.approx(0.3)


def test_temperature_softens_confidence_without_changing_the_answer(make_jev_judge):
    judge = make_jev_judge(_choice_response("B", {"A": 0.01, "B": 0.99}))
    cold = judge.ask("state", ALLOW_BLOCK, temperature=1.0)
    warm = judge.ask("state", ALLOW_BLOCK, temperature=5.0)

    assert cold.answer == warm.answer == "B"
    assert warm.confidence < cold.confidence


def test_request_payload_shapes_by_question_kind(make_jev_judge):
    judge = make_jev_judge(_choice_response("A", {"A": 0.6, "B": 0.4}))
    judge.ask("some state", ALLOW_BLOCK)

    assert len(judge.calls) == 1
    call = judge.calls[0]
    assert call["state"] == "some state"
    assert call["model"] == "jev-test"
    q = call["questions"]["q"]
    assert q["type"] == "choice"
    assert q["criteria"] == {"A": "allow - run it now", "B": "block - require human confirmation first"}
    assert q["instructions"]["question"] == ALLOW_BLOCK.prompt


def test_instructions_carry_the_same_domain_guidance_as_the_local_judge(make_jev_judge):
    # Jev has no separate system-message slot, so the domain guidance
    # Judge.ask() sends as a system prompt (e.g. the deletion-risk heuristic)
    # has to reach Jev some other way - otherwise Jev never sees it while
    # the local model does, and the two judges answer different questions.
    # It goes in a structured {"question", "focus"} object rather than a
    # flat prepended string, per Typesafe's own docs on phrasing
    # instructions - see jev_judge.py's module docstring.
    choice_judge = make_jev_judge(_choice_response("A", {"A": 0.6, "B": 0.4}))
    choice_judge.ask("some state", ALLOW_BLOCK)
    choice_instructions = choice_judge.calls[0]["questions"]["q"]["instructions"]
    assert choice_instructions["question"] == ALLOW_BLOCK.prompt
    assert "regenerable" in choice_instructions["focus"]  # the deletion heuristic

    score_judge = make_jev_judge(_score_response(0.0, {"0": 0.9, "1": 0.05, "2": 0.05}))
    score_judge.ask("some state", RISK_SCALE)
    score_instructions = score_judge.calls[0]["questions"]["q"]["instructions"]
    assert score_instructions["question"] == RISK_SCALE.prompt
    assert "risk" in score_instructions["focus"]

    noul_judge = make_jev_judge(_noul_response(0.5))
    statement = "it is raining"
    noul_judge.ask("some state", ProbabilityQuestion(statement=statement))
    noul_instructions = noul_judge.calls[0]["questions"]["q"]["instructions"]
    assert noul_instructions["question"] == statement
    assert "statement" in noul_instructions["focus"]


def test_extreme_probability_does_not_raise_on_log(make_jev_judge):
    # A hard 0.0/1.0 the API might round to must not blow up log().
    judge = make_jev_judge(_choice_response("B", {"A": 0.0, "B": 1.0}))
    result = judge.ask("state", ALLOW_BLOCK)

    assert result.answer == "B"
    assert math.isfinite(result.raw_logits["A"])
    assert math.isfinite(result.raw_logits["B"])
