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

import math
from dataclasses import dataclass


def _nll(logit_true: float, logit_other: list[float], temperature: float) -> float:
    """Negative log-likelihood of the true class under softmax(logits / T),
    for one example, given the true class's logit and the other candidates'
    logits (binary or multi-way)."""
    t = max(temperature, 1e-6)
    scaled_true = logit_true / t
    scaled_all = [l / t for l in ([logit_true] + logit_other)]
    m = max(scaled_all)
    denom = sum(math.exp(x - m) for x in scaled_all)
    log_p_true = (scaled_true - m) - math.log(denom)
    return -log_p_true


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
    assert len(confidences) == len(correct)
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
