"""Well-known paths inside the repo, resolved once.

Four modules used to each recompute `Path(__file__).parent.parent...` to
find the same two files; a typo in any one of them would have silently
pointed at a different data directory.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"

#: Fitted temperature + held-out metrics, written by scripts/run_eval.py.
CALIBRATION_PATH = DATA_DIR / "calibration.json"
#: Append-only record of judged calls (gitignored).
AUDIT_LOG_PATH = DATA_DIR / "audit_log.jsonl"
