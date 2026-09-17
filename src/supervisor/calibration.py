"""Temperature scaling + expected calibration error (ECE).

Temperature scaling: a single scalar T > 0 applied to logits before softmax
(softmax(logits / T)). T > 1 softens an overconfident model, T < 1 sharpens
an underconfident one. It's fit by minimizing NLL over a held-out labeled
set, using a golden-section search since this is a single-parameter,
unimodal problem (no need to pull in scipy for this).

ECE: bin predictions by confidence, compare each bin's average confidence to
its actual accuracy, take the bin-size-weighted average gap. Standard
calibration diagnostic; see Guo et al. 2017 ("On Calibration of Modern
Neural Networks").
"""

from __future__ import annotations

import math
from dataclasses import dataclass

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


def _nll(logits: list[float], correct_index: int, temperature: float) -> float:
    """Negative log-likelihood of the correct candidate under
    softmax(logits / T), for one example.

    Computed in log space (log-sum-exp minus the correct logit) rather than
    as -log(softmax(...)[i]) on purpose: at the logit gaps the real model
    produces, dividing by a small T underflows the correct candidate's
    probability to exactly 0.0 and -log(0) is a domain error.
    """
    t = max(temperature, MIN_TEMPERATURE)
    scaled = [x / t for x in logits]
    m = max(scaled)
    log_sum_exp = m + math.log(sum(math.exp(x - m) for x in scaled))
    return log_sum_exp - scaled[correct_index]


@dataclass
class CalibrationExample:
    """One (candidate logits, index of the correct candidate) pair, fed to
    fit_temperature. `logits` should be the *raw* (pre-temperature) logits
    Judge stored, in the same order as the candidate labels."""

    logits: list[float]
    correct_index: int


def fit_temperature(
    examples: list[CalibrationExample], t_min: float = 0.05, t_max: float = 100.0, tolerance: float = 1e-4
) -> float:
    """The temperature in [t_min, t_max] minimizing mean NLL over `examples`,
    to within a relative `tolerance`.

    Mean NLL is convex in 1/T (a sum of log-sum-exps of linear functions,
    minus a linear term), so it has a single minimum along log T and a
    golden-section search there converges on it exactly - about 90
    evaluations for the default range and tolerance.

    This used to be a 400-point grid over [0.05, 8.0], which had two
    problems. The ceiling was too low: Qwen2.5-14B's stored fit was
    7.999999999999998, the edge of the grid rather than a minimum. And the
    grid's spacing leaked into reported numbers: the objective is flat near
    its minimum (for the 7B model, mean NLL at T=6.047 and T=6.100 differs
    in the sixth decimal) while held-out ECE is not, since a small shift in
    T moves predictions across ECE's bin edges - so where the grid happened
    to have a point decided a headline metric.
    """
    if not examples:
        return 1.0
    if not 0 < t_min <= t_max:
        raise ValueError(f"need 0 < t_min <= t_max, got {t_min} and {t_max}")
    if tolerance <= 0:
        raise ValueError(f"need a positive tolerance, got {tolerance}")

    def mean_nll(log_t: float) -> float:
        t = math.exp(log_t)
        return sum(_nll(ex.logits, ex.correct_index, t) for ex in examples) / len(examples)

    inv_phi = (math.sqrt(5) - 1) / 2
    lo, hi = math.log(t_min), math.log(t_max)
    a, b = hi - inv_phi * (hi - lo), lo + inv_phi * (hi - lo)
    f_a, f_b = mean_nll(a), mean_nll(b)
    while hi - lo > tolerance:
        # keep the side holding the lower of the two interior points; one of
        # them carries over as an interior point of the narrower bracket
        if f_a < f_b:
            hi, b, f_b = b, a, f_a
            a = hi - inv_phi * (hi - lo)
            f_a = mean_nll(a)
        else:
            lo, a, f_a = a, b, f_b
            b = lo + inv_phi * (hi - lo)
            f_b = mean_nll(b)
    return math.exp((lo + hi) / 2)


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
