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
from supervisor.questions import (
    DECISION_ORDER,
    RISK_NAMES,
    RISK_ORDER,
    RISK_SCALE,
    ask_allow_block,
    decide,
)
from supervisor.quoting import quote_mask

_SEPARATORS = ("&&", "||", ";", "\n")


def split_commands(cmd: str) -> list[str]:
    """Split on top-level `&&`, `||`, `;`, and newlines - quote-aware, so a
    separator inside a quoted string doesn't get treated as a split point.
    Does not split on `|` (pipe) - see module docstring."""
    quoted = quote_mask(cmd)
    parts: list[str] = []
    start = 0
    i = 0
    n = len(cmd)
    while i < n:
        if quoted[i]:
            i += 1
            continue
        sep = next((s for s in _SEPARATORS if cmd.startswith(s, i)), None)
        if sep is None:
            i += 1
            continue
        parts.append(cmd[start:i])
        i += len(sep)
        start = i
    parts.append(cmd[start:])
    return [p.strip() for p in parts if p.strip()]


def assess_shell_command(judge: Judge, cmd: str, confidence_threshold: float | None = None) -> dict:
    """Split `cmd` into individual logical commands, judge each one on its
    own, and return the worst-case risk/decision across all of them plus a
    per-command breakdown so a human (or the caller) can see which specific
    line triggered it.

    Each command's decision is "block", "allow", or "uncertain" (confidence
    below `confidence_threshold`, default DEFAULT_UNCERTAIN_THRESHOLD - see
    questions.py). The aggregate decision is the worst of the three across
    all commands: one confident block wins over everything; absent that, one
    uncertain command is enough to make the whole thing uncertain rather
    than silently averaging it away.

    Judging is deterministic, so a command repeated within one `cmd` (a
    build script that runs the same step per target, say) is judged once and
    the verdict reused - two forward passes saved per duplicate line.
    """
    per_command: list[dict] = []
    seen: dict[str, dict] = {}
    worst_risk = "A"
    worst_decision = "allow"

    for sub in split_commands(cmd):
        verdict = seen.get(sub)
        if verdict is None:
            verdict = _judge_one(judge, sub, confidence_threshold)
            seen[sub] = verdict
        per_command.append(verdict)

        if RISK_ORDER[verdict["risk_label"]] > RISK_ORDER[worst_risk]:
            worst_risk = verdict["risk_label"]
        if DECISION_ORDER[verdict["decision"]] > DECISION_ORDER[worst_decision]:
            worst_decision = verdict["decision"]

    return {
        "risk": RISK_NAMES[worst_risk],
        "decision": worst_decision,
        "num_commands": len(per_command),
        "commands": [{k: v for k, v in c.items() if k != "risk_label"} for c in per_command],
    }


def _judge_one(judge: Judge, sub: str, confidence_threshold: float | None) -> dict:
    state = f"Tool call: run_shell(cmd='{sub}')"
    risk = judge.ask(state, RISK_SCALE)
    block = ask_allow_block(judge, state)
    return {
        "cmd": sub,
        "risk_label": risk.answer,
        "risk": RISK_NAMES[risk.answer],
        "risk_confidence": round(risk.confidence, 4),
        "decision": decide(block.answer, block.confidence, confidence_threshold),
        "decision_confidence": round(block.confidence, 4),
    }
