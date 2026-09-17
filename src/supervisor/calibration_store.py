"""Reads/writes the per-model calibration store (data/calibration.json).

A fitted temperature is specific to one model - different models produce
raw logits with different confidence distributions, so a temperature fit
for one is wrong for another. The store is keyed by model_id so
switching models (SUPERVISOR_MODEL_ID, or Judge(model_id=...)) picks up the
right calibration automatically instead of silently applying whatever
model was last run through scripts/run_eval.py.
"""

from __future__ import annotations

import json
import math
import os

from supervisor.paths import CALIBRATION_PATH


def load_temperature(model_id: str) -> float:
    """Calibrated temperature for `model_id`, or 1.0 (uncalibrated) when
    there isn't a usable one: this model has never been run through
    scripts/run_eval.py, or the store is missing, unreadable, or holds a
    value softmax(logits / T) can't use.

    Falling back rather than raising is deliberate: a fresh clone has no
    calibration for a newly chosen model, and an uncalibrated judge still
    answers correctly - only its reported confidence is less honest.
    Refusing to start would be a worse failure than saying so and going.
    """
    try:
        temperature = float(json.loads(CALIBRATION_PATH.read_text())[model_id]["temperature"])
    except (OSError, ValueError, LookupError, TypeError):
        return 1.0
    if not math.isfinite(temperature) or temperature <= 0:
        return 1.0
    return temperature


def save_entry(model_id: str, entry: dict) -> None:
    """Merge `entry` into the store under `model_id`, preserving whatever
    other models' entries are already there.

    Unlike load_temperature, a store that exists but won't parse is an
    error here rather than treated as empty - overwriting it would silently
    destroy every other model's calibration. The write goes to a temp file
    first and is renamed into place, so an interrupted run can't leave a
    truncated store behind either.
    """
    store = json.loads(CALIBRATION_PATH.read_text()) if CALIBRATION_PATH.exists() else {}
    store[model_id] = entry
    CALIBRATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = CALIBRATION_PATH.with_name(CALIBRATION_PATH.name + ".tmp")
    tmp.write_text(json.dumps(store, indent=2))
    os.replace(tmp, CALIBRATION_PATH)
