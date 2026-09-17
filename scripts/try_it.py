"""Interactive playground: type (or pipe) a tool-call description, see how
the judge scores it.

SAFETY: this script is READ-ONLY with respect to whatever you type. Your
input is only ever used as the `state` string passed to Judge.ask() - it is
tokenized and fed through the model as text. It is never passed to exec(),
eval(), subprocess, os.system, or any tool implementation. There is no code
path in this file that executes anything you type, no matter how it reads.
You can safely type the most dangerous-looking tool call you can think of.

Usage:
  uv run python scripts/try_it.py
      -> interactive REPL, one state per line, empty line / 'quit' to exit

  uv run python scripts/try_it.py "Tool call: run_shell(cmd='rm -rf /')"
      -> judge that one state once and exit (handy for scripting/piping)
"""

import sys

from rich.console import Console
from rich.panel import Panel

from supervisor import RISK_SCALE, Judge, ProbabilityQuestion
from supervisor.questions import RISK_NAMES, ask_allow_block, decide

console = Console()

IRREVERSIBLE_HARM_Q = ProbabilityQuestion(statement="This action could cause irreversible harm.")

DECISION_DISPLAY = {
    "uncertain": ("UNCERTAIN (ask a human)", "yellow"),
    "block": ("BLOCK (needs confirmation)", "red"),
    "allow": ("ALLOW", "green"),
}


def judge_one(judge: Judge, state: str) -> None:
    risk = judge.ask(state, RISK_SCALE)
    # the same two-ordering check should_block() uses, so the playground
    # can't show a different decision than the MCP server would make
    block = ask_allow_block(judge, state)
    harm = judge.ask(state, IRREVERSIBLE_HARM_Q)

    risk_label = RISK_NAMES[risk.answer].upper()
    color = {"A": "green", "B": "yellow", "C": "red"}[risk.answer]
    decision, decision_color = DECISION_DISPLAY[decide(block.answer, block.confidence)]

    body = (
        f"[bold {color}]risk: {risk_label}[/bold {color}] "
        f"(confidence {risk.confidence:.1%}, raw_probs={_fmt(risk.raw_probs)})\n"
        f"[bold {decision_color}]decision: {decision}[/bold {decision_color}] "
        f"(confidence {block.confidence:.1%})\n"
        f"P(irreversible harm) = {harm.probability_true:.1%}\n"
        f"[dim]latency: risk={risk.latency_ms:.0f}ms  "
        f"decision={block.latency_ms:.0f}ms  harm={harm.latency_ms:.0f}ms "
        f"(NOTHING WAS EXECUTED - judged only)[/dim]"
    )
    console.print(Panel(body, title=state, title_align="left"))


def _fmt(probs: dict[str, float]) -> str:
    return "{" + ", ".join(f"{k}={v:.3f}" for k, v in probs.items()) + "}"


def main() -> None:
    console.print(
        "[bold]supervisor try-it[/bold] - judge-only playground. "
        "[dim]Nothing you type is ever executed.[/dim]"
    )
    console.print("Loading judge...")
    judge = Judge.calibrated()
    console.print(f"Loaded {judge.model_id} in {judge.load_time_s:.2f}s (calibrated T={judge.temperature:.3f})\n")

    if len(sys.argv) > 1:
        judge_one(judge, " ".join(sys.argv[1:]))
        return

    console.print("Type a tool-call description (or 'quit' to exit):\n")
    while True:
        try:
            state = console.input("[bold cyan]tool call> [/bold cyan]").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\nbye")
            return
        if not state or state.lower() in {"quit", "exit"}:
            console.print("bye")
            return
        judge_one(judge, state)


if __name__ == "__main__":
    main()
