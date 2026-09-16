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
from pathlib import Path

CALIBRATION_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "calibration.json"


def load_temperature(model_id: str) -> float:
    """Calibrated temperature for `model_id`, or 1.0 (uncalibrated) if this
    model has never been run through scripts/run_eval.py."""
    if not CALIBRATION_PATH.exists():
        return 1.0
    store = json.loads(CALIBRATION_PATH.read_text())
    entry = store.get(model_id)
    return entry["temperature"] if entry else 1.0


def save_entry(model_id: str, entry: dict) -> None:
    """Merge `entry` into the store under `model_id`, preserving whatever
    other models' entries are already there."""
    store = json.loads(CALIBRATION_PATH.read_text()) if CALIBRATION_PATH.exists() else {}
    store[model_id] = entry
    CALIBRATION_PATH.parent.mkdir(exist_ok=True)
    CALIBRATION_PATH.write_text(json.dumps(store, indent=2))
