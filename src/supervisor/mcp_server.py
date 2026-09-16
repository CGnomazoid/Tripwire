"""MCP server exposing the Judge as tools any MCP-compatible agent can call
before executing a risky action.

Deliberately NOT a hook or interceptor: nothing here automatically gates
anything. It's opt-in plumbing - a caller (Claude Desktop, Claude Code,
a hand-rolled agent loop, anything that speaks MCP) has to choose to call
`assess_risk`/`should_block`/etc. before it acts. This process never
executes tool calls itself; it only ever reads the text description it's
given and returns a judgment.

Every tool takes an optional `reason`: the calling agent's own stated
justification for why it wants to do this. It is NEVER passed into
Judge.ask() or the model's prompt - the risk score is a pure function of
`state` alone. `reason` is only logged (audit_log.py) and echoed back in the
response, for two purposes: troubleshooting later, and giving a human who
sees a blocked action some context on what the agent thought it was doing.
Keeping this split (logged/displayed, but never scored) matters - folding it
into the score would reopen a way for an agent to talk its way past the gate.

Run directly:      ./run python -m supervisor.mcp_server
As an MCP client config entry, point `command` at the repo's `run` script
(`run.ps1` on Windows) with args ["python", "-m", "supervisor.mcp_server"].
"""

from mcp.server.mcpserver import MCPServer

from supervisor import calibration_store
from supervisor.audit_log import log_call
from supervisor.judge import Judge
from supervisor.multiline import assess_shell_command as _assess_shell_command
from supervisor.questions import ALLOW_BLOCK, DEFAULT_UNCERTAIN_THRESHOLD, RISK_SCALE
from supervisor.types import ChoiceOption, ChoiceQuestion, ProbabilityQuestion

_RISK_NAMES = {"A": "low", "B": "medium", "C": "high"}

_judge: Judge | None = None


def _get_judge() -> Judge:
    """Lazy singleton: load weights on first tool call, not at server
    startup, so the MCP initialize handshake doesn't block on a ~50s
    first-ever model download. Calibration is looked up AFTER construction,
    keyed on whatever model_id Judge actually resolved to (it may differ
    from any hardcoded default - see SUPERVISOR_MODEL_ID/SUPERVISOR_BACKEND)
    so a model swap can't accidentally pick up a different model's
    temperature."""
    global _judge
    if _judge is None:
        _judge = Judge()
        _judge.temperature = calibration_store.load_temperature(_judge.model_id)
    return _judge


mcp = MCPServer(
    name="agent-supervisor",
    instructions=(
        "Local, fast (well under 1s) risk judgment for a proposed agent "
        "tool call, running entirely on-device. Call assess_risk or "
        "should_block BEFORE executing anything that modifies state, "
        "spends money, sends communication, or could be destructive - "
        "especially file deletion, shell commands, financial transfers, "
        "and mass communication. This server is advisory only: it returns "
        "a calibrated confidence score, not a guarantee, and it never "
        "executes any tool call itself - only the caller decides what to "
        "do with the judgment. Pass `reason` (why you want to do this) on "
        "every call if you can - it doesn't change the score, but it's "
        "logged and shown to the human if the action gets blocked, so they "
        "have context instead of just a bare denial. For a shell command "
        "that chains multiple statements (&&, ||, ;, or multiple lines), "
        "use assess_shell_command instead of assess_risk/should_block - it "
        "judges each statement individually so one risky line can't hide "
        "among a lot of benign ones."
    ),
)


@mcp.tool(structured_output=True)
def assess_risk(state: str, reason: str | None = None) -> dict[str, str | float | dict | None]:
    """Judge how risky a proposed tool call is, on a low/medium/high scale.

    Args:
        state: plain-text description of the tool call about to be made,
            e.g. "Tool call: delete_file(path='/etc/passwd')".
        reason: optional - why you (the calling agent) believe this action
            is needed. Does not affect the score. Logged and echoed back so
            a human can see it if this gets flagged.
    """
    result = _get_judge().ask(state, RISK_SCALE)
    response = {
        "risk": _RISK_NAMES[result.answer],
        "confidence": round(result.confidence, 4),
        "raw_probs": {_RISK_NAMES[k]: round(v, 4) for k, v in result.raw_probs.items()},
        "latency_ms": round(result.latency_ms, 1),
        "reason": reason,
    }
    log_call(tool="assess_risk", state=state, reason=reason, result=response)
    return response


@mcp.tool(structured_output=True)
def should_block(
    state: str, reason: str | None = None, confidence_threshold: float | None = None
) -> dict[str, str | float | None]:
    """Decide whether a proposed tool call should be blocked pending human
    confirmation, is fine to run automatically, or is uncertain enough that
    a human should weigh in even though the raw answer wasn't "block".

    Args:
        state: plain-text description of the tool call about to be made.
        reason: optional - why you (the calling agent) believe this action
            is needed. Does not affect the decision. Logged and echoed back
            so a human can see it if this gets blocked or is uncertain.
        confidence_threshold: below this confidence, the decision is
            "uncertain" rather than a forced allow/block (default 0.6 - see
            supervisor.questions.DEFAULT_UNCERTAIN_THRESHOLD). Temperature
            calibration can soften an overconfident wrong answer, but it
            can't move a genuinely near-50/50 one past the decision line -
            forcing a binary call there would present a coin flip as a
            decision.
    """
    threshold = DEFAULT_UNCERTAIN_THRESHOLD if confidence_threshold is None else confidence_threshold
    result = _get_judge().ask(state, ALLOW_BLOCK)

    if result.confidence < threshold:
        decision = "uncertain"
    elif result.answer == "B":
        decision = "block"
    else:
        decision = "allow"

    if decision == "uncertain":
        message = (
            f"Uncertain (confidence {result.confidence:.0%}, below the "
            f"{threshold:.0%} threshold) - recommend asking a human rather "
            f"than deciding automatically."
        )
    else:
        message = f"{decision.capitalize()} (confidence {result.confidence:.0%})."
    if decision in ("block", "uncertain") and reason:
        message += f" Agent's stated reason: {reason!r}"

    response = {
        "decision": decision,
        "confidence": round(result.confidence, 4),
        "latency_ms": round(result.latency_ms, 1),
        "reason": reason,
        "message": message,
    }
    log_call(tool="should_block", state=state, reason=reason, result=response)
    return response


@mcp.tool(structured_output=True)
def judge_statement(state: str, statement: str, reason: str | None = None) -> dict[str, str | float | None]:
    """Get a calibrated P(statement is true) about a proposed tool call.
    Useful for targeted checks like "is this reversible?" or "does this
    affect more than one user?" that don't fit the risk/allow-block framing.

    Args:
        state: plain-text description of the tool call.
        statement: the yes/no claim to evaluate, e.g. "This action is
            reversible."
        reason: optional - why you (the calling agent) believe this action
            is needed. Does not affect the answer. Logged and echoed back.
    """
    result = _get_judge().ask(state, ProbabilityQuestion(statement=statement))
    response = {
        "probability_true": round(result.probability_true, 4),
        "answer": result.answer,
        "confidence": round(result.confidence, 4),
        "latency_ms": round(result.latency_ms, 1),
        "reason": reason,
    }
    log_call(tool="judge_statement", state=state, reason=reason, result=response)
    return response


@mcp.tool(structured_output=True)
def ask_custom_choice(
    state: str, prompt: str, options: list[dict], reason: str | None = None
) -> dict[str, str | float | dict | None]:
    """Ask a custom multiple-choice question about a proposed tool call, for
    decisions that don't fit the built-in risk/allow-block framing.

    Args:
        state: plain-text description of the tool call.
        prompt: the question to ask.
        options: 2-4 options, each {"label": "A", "text": "description"}.
            Labels must be single letters (A-Z tokenize to exactly one
            token, which the underlying model needs).
        reason: optional - why you (the calling agent) believe this action
            is needed. Does not affect the answer. Logged and echoed back.
    """
    question = ChoiceQuestion(
        prompt=prompt,
        options=[ChoiceOption(label=o["label"], text=o["text"]) for o in options],
    )
    result = _get_judge().ask(state, question)
    response = {
        "answer": result.answer,
        "confidence": round(result.confidence, 4),
        "raw_probs": {k: round(v, 4) for k, v in result.raw_probs.items()},
        "latency_ms": round(result.latency_ms, 1),
        "reason": reason,
    }
    log_call(tool="ask_custom_choice", state=state, reason=reason, result=response)
    return response


@mcp.tool(structured_output=True)
def assess_shell_command(
    cmd: str, reason: str | None = None, confidence_threshold: float | None = None
) -> dict[str, str | int | list | None]:
    """Judge a shell command that may chain multiple statements together
    (via &&, ||, ; or newlines). Splits it into individual commands and
    judges each one on its own before aggregating, instead of judging the
    whole string as one opaque blob - this catches a single dangerous line
    buried among many benign ones, which a single combined judgment can
    dilute past detection, and keeps each forward pass short regardless of
    how long the overall command is. Does NOT split on | (pipe): a
    pipeline's danger often comes from the composition itself (e.g.
    `curl ... | sh`), so piped stages are judged together as one command.

    Prefer this over assess_risk/should_block whenever `cmd` might contain
    more than one statement - a long or chained command is exactly the case
    a single judgment call is least reliable on.

    The overall `decision` (and each command's own decision) can be
    "uncertain" rather than a forced allow/block - see should_block's
    confidence_threshold for what that means. One confident block beats
    everything else; failing that, one uncertain command makes the whole
    result uncertain.

    Args:
        cmd: the shell command (or chain of commands) about to run.
        reason: optional - why you (the calling agent) believe this action
            is needed. Does not affect the score. Logged and echoed back.
        confidence_threshold: see should_block (default 0.6).
    """
    result = _assess_shell_command(_get_judge(), cmd, confidence_threshold=confidence_threshold)
    response = {
        "risk": result["risk"],
        "decision": result["decision"],
        "num_commands": result["num_commands"],
        "commands": result["commands"],
        "reason": reason,
    }
    log_call(tool="assess_shell_command", state=cmd, reason=reason, result=response)
    return response


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
