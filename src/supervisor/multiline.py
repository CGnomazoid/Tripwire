"""Splitting and aggregating judgments for multi-command / multi-line shell
input.

Motivation: a single risky line buried inside a long, mostly-benign shell
script can get diluted when judged as one opaque blob - the model's
attention is spread across everything else in the string, and very long
input risks running into context-length limits regardless. Splitting into
individual logical commands, judging each one separately, and keeping the
worst verdict means one dangerous line can't hide behind many benign ones,
and keeps each individual forward pass short no matter how long the overall
input is.

Deliberately does NOT split on `|` (pipe): a pipeline like `curl ... | sh`
is dangerous BECAUSE of the composition (untrusted output becomes code) -
splitting it apart would judge `curl ...` and `sh` each in isolation and
lose exactly the signal that makes the combination risky.
"""

from __future__ import annotations

from supervisor.judge import Judge
from supervisor.questions import ALLOW_BLOCK, RISK_SCALE

_RISK_ORDER = {"A": 0, "B": 1, "C": 2}
_RISK_NAMES = {"A": "low", "B": "medium", "C": "high"}


def split_commands(cmd: str) -> list[str]:
    """Split on top-level `&&`, `||`, `;`, and newlines - quote-aware, so a
    separator inside a quoted string doesn't get treated as a split point.
    Does not split on `|` (pipe) - see module docstring."""
    parts: list[str] = []
    current: list[str] = []
    i = 0
    n = len(cmd)
    quote: str | None = None
    while i < n:
        ch = cmd[i]
        if quote:
            current.append(ch)
            if ch == quote and cmd[i - 1] != "\\":
                quote = None
            i += 1
            continue
        if ch in ("'", '"'):
            quote = ch
            current.append(ch)
            i += 1
            continue
        if cmd[i : i + 2] in ("&&", "||"):
            parts.append("".join(current))
            current = []
            i += 2
            continue
        if ch in (";", "\n"):
            parts.append("".join(current))
            current = []
            i += 1
            continue
        current.append(ch)
        i += 1
    parts.append("".join(current))
    return [p.strip() for p in parts if p.strip()]


def assess_shell_command(judge: Judge, cmd: str) -> dict:
    """Split `cmd` into individual logical commands, judge each one on its
    own, and return the worst-case risk/decision across all of them plus a
    per-command breakdown so a human (or the caller) can see which specific
    line triggered it."""
    sub_commands = split_commands(cmd)
    per_command = []
    worst_risk = "A"
    should_block = False

    for sub in sub_commands:
        state = f"Tool call: run_shell(cmd='{sub}')"
        risk = judge.ask(state, RISK_SCALE)
        block = judge.ask(state, ALLOW_BLOCK)

        if _RISK_ORDER[risk.answer] > _RISK_ORDER[worst_risk]:
            worst_risk = risk.answer
        if block.answer == "B":
            should_block = True

        per_command.append({
            "cmd": sub,
            "risk": _RISK_NAMES[risk.answer],
            "risk_confidence": round(risk.confidence, 4),
            "decision": "block" if block.answer == "B" else "allow",
            "decision_confidence": round(block.confidence, 4),
        })

    return {
        "risk": _RISK_NAMES[worst_risk],
        "decision": "block" if should_block else "allow",
        "num_commands": len(sub_commands),
        "commands": per_command,
    }
