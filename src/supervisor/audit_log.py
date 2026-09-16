"""Append-only audit log for judged calls.

Deliberately has nothing to do with risk scoring. `reason` (the calling
agent's stated intent, if it provides one) is captured here purely for
troubleshooting and for surfacing to a human when an action is blocked -
"the agent said it wanted to do this because...". It is never passed into
Judge.ask() and never touches the prompt the model sees, so it can't be used
to talk the judge into a different score - see mcp_server.py and readme.md
for why that separation matters.
"""

import json
import time
from pathlib import Path

from supervisor.paths import AUDIT_LOG_PATH

LOG_PATH: Path = AUDIT_LOG_PATH


def log_call(*, tool: str, state: str, reason: str | None, result: dict) -> None:
    record = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tool": tool,
        "state": state,
        "reason": reason,
        "result": result,
    }
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a") as f:
        f.write(json.dumps(record) + "\n")
