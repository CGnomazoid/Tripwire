"""Unit tests for the pure-math calibration layer (no model loading, so these
run fast and don't need a GPU/model download)."""

import math

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
