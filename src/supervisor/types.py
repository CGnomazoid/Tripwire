"""Typed questions and answers for the supervisor judge.

Mirrors Jev's three question types: Choice, Score, Probability. All three
reduce to the same underlying mechanism (single forward pass, read logits at
candidate-answer token positions) but have different ergonomics for callers.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class ChoiceOption(BaseModel):
    """One labeled option. `label` must tokenize to a single token (enforced
    at ask-time against the loaded tokenizer, not here) - single letters
    ("A", "B", "C") are the safe default."""

    label: str
    text: str


class ChoiceQuestion(BaseModel):
    """Pick one of N labeled options."""

    kind: Literal["choice"] = "choice"
    prompt: str
    options: list[ChoiceOption]

    @field_validator("options")
    @classmethod
    def _at_least_two(cls, v: list[ChoiceOption]) -> list[ChoiceOption]:
        if len(v) < 2:
            raise ValueError("choice question needs at least 2 options")
        labels = [o.label for o in v]
        if len(set(labels)) != len(labels):
            raise ValueError("option labels must be unique")
        return v


class ScoreQuestion(BaseModel):
    """Place the state on an ordered scale. Same mechanism as Choice, but the
    options are ordinal (e.g. low/medium/high risk) so the answer also carries
    an ordinal index (0 = lowest)."""

    kind: Literal["score"] = "score"
    prompt: str
    scale: list[ChoiceOption]  # ordered low -> high

    @field_validator("scale")
    @classmethod
    def _at_least_two(cls, v: list[ChoiceOption]) -> list[ChoiceOption]:
        if len(v) < 2:
            raise ValueError("score question needs at least 2 scale points")
        labels = [o.label for o in v]
        if len(set(labels)) != len(labels):
            raise ValueError("scale labels must be unique")
        return v


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
