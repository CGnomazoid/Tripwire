"""MCP server exposing the Judge as tools any MCP-compatible agent can call
before executing a risky action.

Deliberately NOT a hook or interceptor: nothing here automatically gates
anything. It's opt-in plumbing - a caller (Claude Desktop, Claude Code,
a hand-rolled agent loop, anything that speaks MCP) has to choose to call
`assess_risk`/`should_block`/etc. before it acts. This process never
executes tool calls itself; it only ever reads the text description it's
given and returns a judgment.

Run directly:      ./run python -m supervisor.mcp_server
As an MCP client config entry, point `command` at the repo's `run` script
(sidesteps the uv/.pth gotcha documented in readme.md) with args
["python", "-m", "supervisor.mcp_server"].
"""

import json
from pathlib import Path

from mcp.server.mcpserver import MCPServer

from supervisor.judge import Judge
from supervisor.questions import ALLOW_BLOCK, RISK_SCALE
from supervisor.types import ChoiceOption, ChoiceQuestion, ProbabilityQuestion

ROOT = Path(__file__).resolve().parent.parent.parent
CALIBRATION_PATH = ROOT / "data" / "calibration.json"

_RISK_NAMES = {"A": "low", "B": "medium", "C": "high"}

_judge: Judge | None = None


def _load_temperature() -> float:
    if CALIBRATION_PATH.exists():
        return json.loads(CALIBRATION_PATH.read_text())["temperature"]
    return 1.0


def _get_judge() -> Judge:
    """Lazy singleton: load weights on first tool call, not at server
    startup, so the MCP initialize handshake doesn't block on a ~50s
    first-ever model download."""
    global _judge
    if _judge is None:
        _judge = Judge(temperature=_load_temperature())
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
        "do with the judgment."
    ),
)


@mcp.tool(structured_output=True)
def assess_risk(state: str) -> dict[str, str | float | dict]:
    """Judge how risky a proposed tool call is, on a low/medium/high scale.

    Args:
        state: plain-text description of the tool call about to be made,
            e.g. "Tool call: delete_file(path='/etc/passwd')".
    """
    result = _get_judge().ask(state, RISK_SCALE)
    return {
        "risk": _RISK_NAMES[result.answer],
        "confidence": round(result.confidence, 4),
        "raw_probs": {_RISK_NAMES[k]: round(v, 4) for k, v in result.raw_probs.items()},
        "latency_ms": round(result.latency_ms, 1),
    }


@mcp.tool(structured_output=True)
def should_block(state: str) -> dict[str, str | float]:
    """Decide whether a proposed tool call should be blocked pending human
    confirmation, or is fine to run automatically.

    Args:
        state: plain-text description of the tool call about to be made.
    """
    result = _get_judge().ask(state, ALLOW_BLOCK)
    return {
        "decision": "block" if result.answer == "B" else "allow",
        "confidence": round(result.confidence, 4),
        "latency_ms": round(result.latency_ms, 1),
    }


@mcp.tool(structured_output=True)
def judge_statement(state: str, statement: str) -> dict[str, str | float]:
    """Get a calibrated P(statement is true) about a proposed tool call.
    Useful for targeted checks like "is this reversible?" or "does this
    affect more than one user?" that don't fit the risk/allow-block framing.

    Args:
        state: plain-text description of the tool call.
        statement: the yes/no claim to evaluate, e.g. "This action is
            reversible."
    """
    result = _get_judge().ask(state, ProbabilityQuestion(statement=statement))
    return {
        "probability_true": round(result.probability_true, 4),
        "answer": result.answer,
        "confidence": round(result.confidence, 4),
        "latency_ms": round(result.latency_ms, 1),
    }


@mcp.tool(structured_output=True)
def ask_custom_choice(state: str, prompt: str, options: list[dict]) -> dict[str, str | float | dict]:
    """Ask a custom multiple-choice question about a proposed tool call, for
    decisions that don't fit the built-in risk/allow-block framing.

    Args:
        state: plain-text description of the tool call.
        prompt: the question to ask.
        options: 2-4 options, each {"label": "A", "text": "description"}.
            Labels must be single letters (A-Z tokenize to exactly one
            token, which the underlying model needs).
    """
    question = ChoiceQuestion(
        prompt=prompt,
        options=[ChoiceOption(label=o["label"], text=o["text"]) for o in options],
    )
    result = _get_judge().ask(state, question)
    return {
        "answer": result.answer,
        "confidence": round(result.confidence, 4),
        "raw_probs": {k: round(v, 4) for k, v in result.raw_probs.items()},
        "latency_ms": round(result.latency_ms, 1),
    }


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
