"""Typed questions and answers for the supervisor judge.

Mirrors Jev's three question types: Choice, Score, Probability. All three
reduce to the same underlying mechanism (single forward pass, read logits at
candidate-answer token positions) but have different ergonomics for callers.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, Field

from supervisor.calibration import softmax


class ChoiceOption(BaseModel):
    """One labeled option. `label` must tokenize to a single token (enforced
    at ask-time against the loaded tokenizer, not here) - single letters
    ("A", "B", "C") are the safe default."""

    label: str
    text: str


def _check_options(options: list[ChoiceOption]) -> list[ChoiceOption]:
    if len(options) < 2:
        raise ValueError("needs at least 2 options to choose between")
    labels = [o.label for o in options]
    if len(set(labels)) != len(labels):
        # two options sharing a label would read the same logit twice
        raise ValueError(f"option labels must be unique, got {labels}")
    return options


Options = Annotated[list[ChoiceOption], AfterValidator(_check_options)]
"""At least two options with distinct labels - what both option-list
question kinds need for the logit read to mean anything."""


class ChoiceQuestion(BaseModel):
    """Pick one of N labeled options."""

    kind: Literal["choice"] = "choice"
    prompt: str
    options: Options


class ScoreQuestion(BaseModel):
    """Place the state on an ordered scale. Same mechanism as Choice, but the
    options are ordinal (e.g. low/medium/high risk) so the answer also carries
    an ordinal index (0 = lowest)."""

    kind: Literal["score"] = "score"
    prompt: str
    scale: Options  # ordered low -> high


class ProbabilityQuestion(BaseModel):
    """P(statement is true), as a single calibrated float in [0, 1]."""

    kind: Literal["probability"] = "probability"
    statement: str


Question = ChoiceQuestion | ScoreQuestion | ProbabilityQuestion


class JudgeAnswer(BaseModel):
    """Structured result of Judge.ask()."""

    kind: Literal["choice", "score", "probability"]
    answer: str
    """The chosen label ("A", "B", ...). For probability questions this is
    "true" or "false" (the higher-probability side)."""

    ordinal: int | None = None
    """For score questions: index of the chosen scale point (0 = lowest).
    None for choice/probability."""

    confidence: float
    """Calibrated probability assigned to `answer`, in [0, 1]."""

    probability_true: float | None = None
    """For probability questions only: calibrated P(statement is true)."""

    raw_logits: dict[str, float]
    """Pre-softmax logit per candidate label, straight from the single
    forward pass. Feed these into calibration.fit_temperature."""

    raw_probs: dict[str, float]
    """Uncalibrated (temperature=1) probability mass per candidate label,
    renormalized over just the candidate set."""

    calibrated_probs: dict[str, float]
    """Same candidate set as raw_probs, after temperature scaling."""

    temperature: float = 1.0
    """Temperature used to produce calibrated_probs. 1.0 means uncalibrated."""

    latency_ms: float
    state: str
    question: Question = Field(discriminator="kind")

    def with_temperature(self, temperature: float) -> JudgeAnswer:
        """This answer re-scaled to another temperature from its stored raw
        logits - no forward pass. `answer` can't change (softmax(logits / T)
        has the same argmax for every T > 0); confidence, calibrated_probs
        and probability_true do.

        Only meaningful for an answer straight out of Judge.ask(). One
        merged by questions.combine_allow_block no longer has a confidence
        derived from its logits: re-scale the inputs, then combine.
        """
        labels = list(self.raw_logits)
        probs = softmax(list(self.raw_logits.values()), temperature)
        update: dict[str, object] = {
            "confidence": probs[labels.index(self.answer)],
            "calibrated_probs": dict(zip(labels, probs)),
            "temperature": temperature,
        }
        if self.kind == "probability":
            update["probability_true"] = probs[labels.index("true")]
        return self.model_copy(update=update)
