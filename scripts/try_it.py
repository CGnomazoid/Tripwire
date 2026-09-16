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

import json
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

from supervisor import ALLOW_BLOCK, RISK_SCALE, Judge, ProbabilityQuestion

ROOT = Path(__file__).resolve().parent.parent
CALIBRATION_PATH = ROOT / "data" / "calibration.json"

console = Console()

REVERSIBLE_Q = ProbabilityQuestion(statement="This action could cause irreversible harm.")


def load_temperature() -> float:
    if CALIBRATION_PATH.exists():
        return json.loads(CALIBRATION_PATH.read_text())["temperature"]
    return 1.0


def judge_one(judge: Judge, state: str) -> None:
    risk = judge.ask(state, RISK_SCALE)
    allow = judge.ask(state, ALLOW_BLOCK)
    harm = judge.ask(state, REVERSIBLE_Q)

    risk_label = {"A": "LOW", "B": "MEDIUM", "C": "HIGH"}[risk.answer]
    color = {"A": "green", "B": "yellow", "C": "red"}[risk.answer]
    decision = "BLOCK (needs confirmation)" if allow.answer == "B" else "ALLOW"
    decision_color = "red" if allow.answer == "B" else "green"

    body = (
        f"[bold {color}]risk: {risk_label}[/bold {color}] "
        f"(confidence {risk.confidence:.1%}, raw_probs={_fmt(risk.raw_probs)})\n"
        f"[bold {decision_color}]decision: {decision}[/bold {decision_color}] "
        f"(confidence {allow.confidence:.1%})\n"
        f"P(irreversible harm) = {harm.probability_true:.1%}\n"
        f"[dim]latency: risk={risk.latency_ms:.0f}ms  "
        f"allow={allow.latency_ms:.0f}ms  harm={harm.latency_ms:.0f}ms "
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
    temperature = load_temperature()
    console.print(f"Loading judge (calibrated T={temperature:.3f})...")
    judge = Judge(temperature=temperature)
    console.print(f"Loaded in {judge.load_time_s:.2f}s\n")

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
