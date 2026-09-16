"""Temperature scaling + expected calibration error (ECE).

Temperature scaling: a single scalar T > 0 applied to logits before softmax
(softmax(logits / T)). T > 1 softens an overconfident model, T < 1 sharpens
an underconfident one. It's fit by minimizing NLL over a held-out labeled
set, using a 1-D scan since this is a convex, single-parameter problem (no
need to pull in scipy for this).

ECE: bin predictions by confidence, compare each bin's average confidence to
its actual accuracy, take the bin-size-weighted average gap. Standard
calibration diagnostic; see Guo et al. 2017 ("On Calibration of Modern
Neural Networks").
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from supervisor.paths import CALIBRATION_PATH

MIN_TEMPERATURE = 1e-6
"""Floor applied to any caller-supplied temperature: softmax(logits / T) is
undefined at T=0 and meaningless below it."""


def softmax(logits: list[float], temperature: float = 1.0) -> list[float]:
    """Temperature-scaled softmax over a handful of candidate logits.

    The one implementation of this in the project - Judge reads calibrated
    and uncalibrated probabilities off it, and run_eval.py re-scales stored
    logits with it without re-running the model. Shifts by the max before
    exponentiating, which is algebraically a no-op and keeps exp() from
    overflowing on the large logit gaps the real model produces.
    """
    t = max(temperature, MIN_TEMPERATURE)
    scaled = [x / t for x in logits]
    m = max(scaled)
    exps = [math.exp(x - m) for x in scaled]
    total = sum(exps)
    return [e / total for e in exps]


def load_temperature(path: Path = CALIBRATION_PATH) -> float:
    """Read the fitted temperature written by scripts/run_eval.py, falling
    back to 1.0 (uncalibrated) when it's missing or unusable.

    Falling back rather than raising is deliberate: a fresh clone has no
    calibration.json until the eval has been run, and an uncalibrated judge
    still answers correctly - only its reported confidence is less honest.
    Refusing to start would be a worse failure than saying so and going.
    """
    try:
        temperature = float(json.loads(path.read_text())["temperature"])
    except (OSError, ValueError, KeyError, TypeError):
        return 1.0
    if not math.isfinite(temperature) or temperature <= 0:
        return 1.0
    return temperature


def _nll(logit_true: float, logit_other: list[float], temperature: float) -> float:
    """Negative log-likelihood of the true class under softmax(logits / T),
    for one example, given the true class's logit and the other candidates'
    logits (binary or multi-way).

    Computed in log space rather than as -log(softmax(...)[0]) on purpose:
    at the logit gaps the real model produces, dividing by a small T
    underflows the true class's probability to exactly 0.0 and -log(0) is a
    domain error. Subtracting the log-sum-exp keeps it finite.
    """
    t = max(temperature, MIN_TEMPERATURE)
    scaled_all = [l / t for l in ([logit_true] + logit_other)]
    m = max(scaled_all)
    denom = sum(math.exp(x - m) for x in scaled_all)
    return -((scaled_all[0] - m) - math.log(denom))


@dataclass
class CalibrationExample:
    """One (candidate logits, index of the correct candidate) pair, fed to
    fit_temperature. `logits` should be the *raw* (pre-temperature) logits
    Judge stored, in the same order as the candidate labels."""

    logits: list[float]
    correct_index: int


def fit_temperature(
    examples: list[CalibrationExample], t_min: float = 0.05, t_max: float = 8.0, steps: int = 400
) -> float:
    """Grid-search the scalar temperature minimizing mean NLL over
    `examples`. Grid search over a convex 1-D objective is simple, doesn't
    need a gradient, and is plenty precise for `steps` in the hundreds."""
    if not examples:
        return 1.0
    if not 0 < t_min <= t_max:
        raise ValueError(f"need 0 < t_min <= t_max, got {t_min} and {t_max}")
    if steps < 2:
        raise ValueError(f"need at least 2 grid steps, got {steps}")

    best_t, best_nll = 1.0, math.inf
    log_min, log_max = math.log(t_min), math.log(t_max)
    for i in range(steps):
        t = math.exp(log_min + (log_max - log_min) * i / (steps - 1))
        total = 0.0
        for ex in examples:
            true_logit = ex.logits[ex.correct_index]
            other_logits = [l for j, l in enumerate(ex.logits) if j != ex.correct_index]
            total += _nll(true_logit, other_logits, t)
        mean_nll = total / len(examples)
        if mean_nll < best_nll:
            best_nll, best_t = mean_nll, t
    return best_t


@dataclass
class ECEResult:
    ece: float
    bin_edges: list[float]
    bin_confidence: list[float | None]
    bin_accuracy: list[float | None]
    bin_count: list[int]


def expected_calibration_error(
    confidences: list[float], correct: list[bool], n_bins: int = 10
) -> ECEResult:
    """Standard equal-width-bin ECE. `confidences` should be the probability
    assigned to whichever answer was actually chosen (i.e. always >= 1/num_options
    for that question, not the raw P(true) for a probability question)."""
    if len(confidences) != len(correct):
        raise ValueError(
            f"confidences and correct must be the same length, got "
            f"{len(confidences)} and {len(correct)}"
        )
    if n_bins < 1:
        raise ValueError(f"need at least 1 bin, got {n_bins}")
    edges = [i / n_bins for i in range(n_bins + 1)]
    bin_conf_sum = [0.0] * n_bins
    bin_acc_sum = [0.0] * n_bins
    bin_count = [0] * n_bins

    for conf, is_correct in zip(confidences, correct):
        idx = min(int(conf * n_bins), n_bins - 1)
        bin_conf_sum[idx] += conf
        bin_acc_sum[idx] += 1.0 if is_correct else 0.0
        bin_count[idx] += 1

    n = len(confidences)
    ece = 0.0
    if n == 0:
        return ECEResult(
            ece=0.0, bin_edges=edges, bin_confidence=[None] * n_bins,
            bin_accuracy=[None] * n_bins, bin_count=bin_count,
        )
    bin_confidence: list[float | None] = []
    bin_accuracy: list[float | None] = []
    for i in range(n_bins):
        if bin_count[i] == 0:
            bin_confidence.append(None)
            bin_accuracy.append(None)
            continue
        avg_conf = bin_conf_sum[i] / bin_count[i]
        avg_acc = bin_acc_sum[i] / bin_count[i]
        bin_confidence.append(avg_conf)
        bin_accuracy.append(avg_acc)
        ece += (bin_count[i] / n) * abs(avg_conf - avg_acc)

    return ECEResult(
        ece=ece,
        bin_edges=edges,
        bin_confidence=bin_confidence,
        bin_accuracy=bin_accuracy,
        bin_count=bin_count,
    )
