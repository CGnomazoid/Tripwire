"""Unit tests for the pure-math calibration layer (no model loading, so these
run fast and don't need a GPU/model download)."""

import math

import pytest

from supervisor.calibration import CalibrationExample, expected_calibration_error, fit_temperature


def test_fit_temperature_recovers_perfect_calibration():
    # logits already perfectly separated and matched to a T=1 softmax that's
    # exactly as confident as it is correct -> fitting shouldn't need to
    # move far from T=1 to (further) minimize NLL beyond floating tolerance.
    examples = [CalibrationExample(logits=[5.0, 0.0], correct_index=0) for _ in range(20)]
    t = fit_temperature(examples)
    assert 0.05 < t < 8.0


def test_fit_temperature_cools_overconfident_model():
    # model is very confident (large logit gap) but wrong half the time ->
    # NLL-minimizing temperature should be > 1 (soften it).
    examples = [CalibrationExample(logits=[10.0, 0.0], correct_index=0) for _ in range(10)]
    examples += [CalibrationExample(logits=[10.0, 0.0], correct_index=1) for _ in range(10)]
    t = fit_temperature(examples)
    assert t > 1.5


def test_fit_temperature_empty_defaults_to_one():
    assert fit_temperature([]) == 1.0


def test_ece_perfect_calibration_is_zero():
    # confidence exactly matches empirical accuracy in every bin.
    confidences = [0.9] * 9 + [0.9] * 1
    correct = [True] * 9 + [False] * 1
    result = expected_calibration_error(confidences, correct, n_bins=10)
    assert math.isclose(result.ece, 0.0, abs_tol=1e-9)


def test_ece_detects_overconfidence():
    confidences = [0.99] * 10
    correct = [True] * 5 + [False] * 5  # only 50% right despite 99% confidence
    result = expected_calibration_error(confidences, correct, n_bins=10)
    assert result.ece > 0.4


def test_ece_handles_empty_bins():
    confidences = [0.95, 0.96, 0.97]
    correct = [True, True, False]
    result = expected_calibration_error(confidences, correct, n_bins=10)
    assert result.ece >= 0.0
    assert len(result.bin_count) == 10


# -- numerical stability under extreme/degenerate inputs. In practice the
# real model produces logit gaps well beyond +-10 (see test_judge.py's
# temperature test, and the 1000+ token vocab logits mlx-lm returns), so
# fit_temperature has to stay finite well past what a toy example suggests
# is "extreme." Verified against the actual live functions before writing
# these bounds - see the exploratory probe referenced in the commit this
# test was added in. --------------------------------------------------------

def test_fit_temperature_stays_finite_on_extreme_confident_correct_logits():
    # logit gap of 2000 - larger than anything realistic, specifically to
    # probe for overflow in the softmax/exp math.
    examples = [CalibrationExample(logits=[1000.0, -1000.0], correct_index=0)]
    t = fit_temperature(examples)
    assert math.isfinite(t)
    assert t > 0


def test_fit_temperature_stays_finite_on_extreme_confident_wrong_logits():
    # same extreme gap, but the confident answer is the WRONG one - this is
    # the case that stresses the optimizer hardest, since cooling toward
    # uniform is the only way to reduce NLL and it should hit the upper
    # bound of the search range rather than diverging.
    examples = [CalibrationExample(logits=[1000.0, -1000.0], correct_index=1)]
    t = fit_temperature(examples)
    assert math.isfinite(t)
    assert t > 5.0  # should push toward the top of the search range


def test_fit_temperature_handles_all_tied_logits():
    # zero information (every candidate equally likely) shouldn't crash or
    # produce a degenerate temperature.
    examples = [CalibrationExample(logits=[0.0, 0.0, 0.0], correct_index=0) for _ in range(5)]
    t = fit_temperature(examples)
    assert math.isfinite(t)
    assert t > 0


def test_fit_temperature_single_example_does_not_crash():
    t = fit_temperature([CalibrationExample(logits=[1.0, 2.0], correct_index=0)])
    assert math.isfinite(t)
    assert t > 0


def test_ece_handles_boundary_confidences_of_exactly_zero_and_one():
    result = expected_calibration_error([0.0, 1.0], [False, True])
    assert math.isfinite(result.ece)
    assert math.isclose(result.ece, 0.0, abs_tol=1e-9)


def test_fit_temperature_is_not_capped_at_the_old_grid_ceiling():
    # Qwen2.5-14B's stored fit was 7.999999999999998 - the old t_max=8.0,
    # not a minimum. Logit gap 20, right 3 times in 4: the NLL-optimal
    # temperature solves softmax(20 / T) = 0.75, i.e. T = 20 / ln(3) ~ 18.2.
    examples = [CalibrationExample(logits=[20.0, 0.0], correct_index=0) for _ in range(3)]
    examples.append(CalibrationExample(logits=[20.0, 0.0], correct_index=1))

    assert fit_temperature(examples) == pytest.approx(20 / math.log(3), rel=0.01)


def test_fit_temperature_finds_the_exact_minimum_not_a_nearby_grid_point():
    # two-class, gap 4, right 9 times in 10: optimum solves
    # sigmoid(4 / T) = 0.9, i.e. T = 4 / ln(9)
    examples = [CalibrationExample(logits=[4.0, 0.0], correct_index=0) for _ in range(9)]
    examples.append(CalibrationExample(logits=[4.0, 0.0], correct_index=1))

    assert fit_temperature(examples) == pytest.approx(4 / math.log(9), rel=1e-3)


def test_fit_temperature_lands_on_the_range_edge_when_the_minimum_is_outside_it():
    examples = [CalibrationExample(logits=[20.0, 0.0], correct_index=0) for _ in range(3)]
    examples.append(CalibrationExample(logits=[20.0, 0.0], correct_index=1))  # optimum ~18.2

    assert fit_temperature(examples, t_max=8.0) == pytest.approx(8.0, rel=1e-3)
