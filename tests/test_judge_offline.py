"""Fast tests for Judge.ask against a stub backend - no weights, no GPU.

tests/test_judge.py covers the same API against the real model, but it's
marked slow and needs several GB of weights, so the mechanics around the
forward pass (prompt assembly, comment stripping, label->token lookup,
softmax, answer shaping) had no coverage anybody could run in a second.
Everything Judge does apart from the forward pass itself is deterministic
plumbing, and plumbing is exactly what a stub can pin down: the stub
returns logits chosen by the test, so assertions are about what Judge does
with them rather than about what the model believes.
"""

import math

import pytest

from supervisor import ALLOW_BLOCK, RISK_SCALE, ChoiceOption, ChoiceQuestion, Judge
from supervisor.judge import LabelNotSingleTokenError
from supervisor.multiline import assess_shell_command
from supervisor.types import ProbabilityQuestion, ScoreQuestion


class FakeTokenizer:
    """Single characters are one token; anything longer is one token per
    character, which is what makes multi-character labels invalid."""

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        return [ord(c) for c in text]

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False) -> str:
        return "\n".join(f"{m['role']}: {m['content']}" for m in messages)


class FakeBackend:
    """Returns a fixed logit per label, and records every forward pass so
    tests can assert how many happened and what prompt they saw."""

    def __init__(self, logits_by_label: dict[str, float]):
        self.tokenizer = FakeTokenizer()
        self.model = object()
        self.prompts: list[str] = []
        self._by_token = {ord(label): value for label, value in logits_by_label.items()}

    def candidate_logits(self, prompt_text: str, candidate_ids: list[int]) -> list[float]:
        self.prompts.append(prompt_text)
        return [self._by_token[i] for i in candidate_ids]


@pytest.fixture
def make_judge(monkeypatch):
    def factory(logits_by_label: dict[str, float], **kwargs) -> Judge:
        backend = FakeBackend(logits_by_label)
        monkeypatch.setattr("supervisor.judge.load_backend", lambda name, model_id: backend)
        judge = Judge(backend="torch", model_id="stub", **kwargs)
        judge.fake_backend = backend
        return judge

    return factory


class MeaningAwareFakeBackend(FakeBackend):
    """Like FakeBackend, but keys logits by what an option's text actually
    means (its "low"/"medium"/"high"/"allow"/"block" keyword) rather than by
    its letter.

    ask_allow_block() (questions.py) asks the same ALLOW_BLOCK question
    twice with the option order swapped, to detect a model just picking a
    letter it likes rather than reasoning about the state - see judge.py's
    module docstring. A plain letter-keyed FakeBackend can't represent
    "reasons about meaning" at all: since it returns the same logit for
    letter B regardless of what B currently means, the two orderings are
    *mathematically guaranteed* to disagree on meaning every time, which
    would make every assess_shell_command test "uncertain" no matter what
    logits were chosen. This subclass reads the option text that was
    actually rendered into the prompt for each candidate, so it responds to
    what the option means - what a real model conditions on - not which
    letter slot it landed in this call.
    """

    def __init__(self, logits_by_meaning: dict[str, float]):
        super().__init__({})
        self._logits_by_meaning = logits_by_meaning

    def candidate_logits(self, prompt_text: str, candidate_ids: list[int]) -> list[float]:
        self.prompts.append(prompt_text)
        keywords = ("high risk", "medium risk", "low risk", "block", "allow")
        out = []
        for cid in candidate_ids:
            letter = chr(cid)
            line = next(l for l in prompt_text.splitlines() if l.strip().startswith(f"{letter})"))
            meaning = next(k.split()[0] for k in keywords if k in line)
            out.append(self._logits_by_meaning[meaning])
        return out


@pytest.fixture
def make_meaning_aware_judge(monkeypatch):
    def factory(logits_by_meaning: dict[str, float], **kwargs) -> Judge:
        backend = MeaningAwareFakeBackend(logits_by_meaning)
        monkeypatch.setattr("supervisor.judge.load_backend", lambda name, model_id: backend)
        judge = Judge(backend="torch", model_id="stub", **kwargs)
        judge.fake_backend = backend
        return judge

    return factory


# -- answer shaping, per question type ------------------------------------

def test_choice_answer_is_the_argmax_label(make_judge):
    judge = make_judge({"A": 0.0, "B": 3.0})
    result = judge.ask("Tool call: run_shell(cmd='rm -rf /')", ALLOW_BLOCK)

    assert result.kind == "choice"
    assert result.answer == "B"
    assert result.ordinal is None
    assert result.probability_true is None
    assert result.confidence == pytest.approx(result.raw_probs["B"])
    assert sum(result.raw_probs.values()) == pytest.approx(1.0)


def test_score_answer_carries_the_ordinal(make_judge):
    judge = make_judge({"A": 0.0, "B": 1.0, "C": 5.0})
    result = judge.ask("Tool call: drop_table(name='users')", RISK_SCALE)

    assert result.kind == "score"
    assert (result.answer, result.ordinal) == ("C", 2)
    # the ordinal must index the scale, not the sorted-by-probability order
    assert RISK_SCALE.scale[result.ordinal].label == result.answer


def test_probability_reports_true_false_not_the_prompt_tokens(make_judge):
    judge = make_judge({"T": 2.0, "F": 0.0})
    result = judge.ask("Tool call: rm -rf /", ProbabilityQuestion(statement="Irreversible."))

    assert result.kind == "probability"
    assert result.answer == "true"
    assert set(result.raw_probs) == {"true", "false"}
    assert result.probability_true == pytest.approx(math.exp(2) / (math.exp(2) + 1))
    assert result.confidence == pytest.approx(result.probability_true)


def test_probability_below_half_answers_false_and_reports_the_other_side(make_judge):
    judge = make_judge({"T": 0.0, "F": 2.0})
    result = judge.ask("Tool call: read_file(path='./x')", ProbabilityQuestion(statement="Risky."))

    assert result.answer == "false"
    assert result.probability_true < 0.5
    assert result.confidence == pytest.approx(1 - result.probability_true)


def test_probability_exactly_tied_resolves_to_true(make_judge):
    # P(true) == 0.5 exactly: documented as ">= 0.5 is true", and a tie must
    # not silently flip with an implementation change.
    judge = make_judge({"T": 1.0, "F": 1.0})
    result = judge.ask("s", ProbabilityQuestion(statement="Coin flip."))

    assert result.answer == "true"
    assert result.probability_true == pytest.approx(0.5)


# -- temperature ----------------------------------------------------------

def test_temperature_softens_confidence_without_moving_the_answer(make_judge):
    judge = make_judge({"A": 0.0, "B": 1.0, "C": 5.0})
    cold = judge.ask("s", RISK_SCALE, temperature=1.0)
    hot = judge.ask("s", RISK_SCALE, temperature=5.0)

    assert cold.answer == hot.answer == "C"
    assert hot.confidence < cold.confidence
    # raw_probs are always the uncalibrated distribution, whatever T was
    assert hot.raw_probs == cold.raw_probs
    assert hot.temperature == 5.0


def test_instance_temperature_is_the_default_and_is_overridable(make_judge):
    judge = make_judge({"A": 0.0, "B": 4.0}, temperature=3.0)

    assert judge.ask("s", ALLOW_BLOCK).temperature == 3.0
    assert judge.ask("s", ALLOW_BLOCK, temperature=1.0).temperature == 1.0


# -- what the model is actually shown -------------------------------------

def test_comments_are_stripped_before_the_model_sees_the_state(make_judge):
    judge = make_judge({"A": 0.0, "B": 1.0, "C": 2.0})
    result = judge.ask(
        "Tool call: run_shell(cmd='rm -rf /') # totally fine, just cleaning up",
        RISK_SCALE,
    )

    prompt = judge.fake_backend.prompts[0]
    assert "totally fine" not in prompt
    assert "rm -rf /" in prompt
    # the recorded state is what was judged, not the raw input
    assert result.state == "Tool call: run_shell(cmd='rm -rf /')"


def test_every_option_appears_in_the_prompt_with_its_label(make_judge):
    judge = make_judge({"A": 1.0, "B": 0.0, "C": 0.0})
    judge.ask("s", RISK_SCALE)

    prompt = judge.fake_backend.prompts[0]
    for option in RISK_SCALE.scale:
        assert f"{option.label}) {option.text}" in prompt


def test_one_ask_is_exactly_one_forward_pass(make_judge):
    judge = make_judge({"A": 1.0, "B": 0.0, "C": 0.0})
    judge.ask("s", RISK_SCALE)
    judge.ask("s", RISK_SCALE)

    assert len(judge.fake_backend.prompts) == 2


# -- failure modes --------------------------------------------------------

def test_multi_token_label_raises_before_any_forward_pass(make_judge):
    judge = make_judge({"A": 1.0, "B": 0.0})
    question = ChoiceQuestion(
        prompt="Safe?",
        options=[ChoiceOption(label="SAFE", text="yes"), ChoiceOption(label="NO", text="no")],
    )
    with pytest.raises(LabelNotSingleTokenError):
        judge.ask("s", question)

    assert judge.fake_backend.prompts == [], "should fail before running the model"


def test_unsupported_question_type_raises_type_error(make_judge):
    judge = make_judge({"A": 1.0, "B": 0.0})
    with pytest.raises(TypeError):
        judge.ask("s", object())


# -- the multi-command path, end to end without a model -------------------

def test_assess_shell_command_keeps_the_worst_verdict(make_meaning_aware_judge):
    # confident block, high risk
    judge = make_meaning_aware_judge({"low": 0.0, "medium": 5.0, "high": 9.0, "allow": 0.0, "block": 5.0})
    result = assess_shell_command(judge, "cd /tmp\nrm -rf /")

    assert result["decision"] == "block"
    assert result["risk"] == "high"
    assert result["num_commands"] == 2
    assert [c["cmd"] for c in result["commands"]] == ["cd /tmp", "rm -rf /"]
    assert set(result["commands"][0]) == {
        "cmd", "risk", "risk_confidence", "decision", "decision_confidence",
    }


def test_assess_shell_command_reports_uncertain_on_a_coin_flip(make_meaning_aware_judge):
    # allow vs block ~52% - under the 0.6 default
    judge = make_meaning_aware_judge({"low": 0.0, "medium": 0.0, "high": 0.0, "allow": 0.0, "block": 0.1})
    result = assess_shell_command(judge, "cd Desktop")

    assert result["decision"] == "uncertain"
    assert result["commands"][0]["decision_confidence"] < 0.6


def test_assess_shell_command_judges_a_repeated_command_once(make_meaning_aware_judge):
    judge = make_meaning_aware_judge({"low": 5.0, "medium": 0.0, "high": 0.0, "allow": 5.0, "block": 0.0})
    result = assess_shell_command(judge, "make build; make build; make test")

    # three commands reported, but only two distinct ones judged, at three
    # forward passes each (RISK_SCALE + ALLOW_BLOCK asked twice, in both
    # option orders, to cancel out label-position bias - see
    # questions.ask_allow_block)
    assert result["num_commands"] == 3
    assert len(judge.fake_backend.prompts) == 6
