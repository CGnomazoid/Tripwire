"""Step 1 of the build order: prove the mechanism works at all.

Loads the hand-written spike set (data/spike.jsonl), asks the canonical
risk-scale question for each state, and reports whether the raw
(uncalibrated) top answer matches the hand-labeled ground truth. No
calibration is applied here - that's a later step.

Also the source of the README's latency numbers: per-ask latency for the
risk-scale question (assess_risk), and for should_block's pair of
ALLOW_BLOCK orderings.

Run: uv run python scripts/run_spike.py
"""

import json

from rich.console import Console
from rich.table import Table

from supervisor import Judge, RISK_SCALE
from supervisor.paths import DATA_DIR
from supervisor.questions import ask_allow_block

DATA_PATH = DATA_DIR / "spike.jsonl"


def main() -> None:
    console = Console()
    examples = [json.loads(line) for line in DATA_PATH.read_text().splitlines() if line.strip()]

    console.print("[bold]Loading judge model...[/bold]")
    judge = Judge()
    console.print(f"Loaded in {judge.load_time_s:.2f}s\n")

    # Not timed: the first forward pass after loading compiles Metal kernels
    # and fills the backend's system-prompt prefix cache, a one-off that no
    # later call pays. Both question kinds timed below get their own warmup.
    judge.ask("Tool call: warm_up()", RISK_SCALE)
    ask_allow_block(judge, "Tool call: warm_up()")

    table = Table(title=f"Spike eval: raw (uncalibrated) accuracy, n={len(examples)}")
    table.add_column("state", overflow="fold", max_width=50)
    table.add_column("expected")
    table.add_column("got")
    table.add_column("confidence")
    table.add_column("ms")
    table.add_column("ok")

    correct = 0
    binary_correct = 0
    latencies = []
    for ex in examples:
        result = judge.ask(ex["state"], RISK_SCALE)
        latencies.append(result.latency_ms)
        ok = result.answer == ex["label"]
        correct += ok
        # the operationally meaningful decision: does this need to block for
        # human confirmation (C) or not (A/B)? label accuracy understates
        # real usefulness if most errors are A<->B mix-ups that don't change
        # this downstream call.
        expected_block = ex["label"] == "C"
        got_block = result.answer == "C"
        binary_correct += expected_block == got_block
        table.add_row(
            ex["state"],
            ex["label"],
            result.answer,
            f"{result.raw_probs[result.answer]:.3f}",
            f"{result.latency_ms:.0f}",
            "[green]y[/green]" if ok else "[red]N[/red]",
        )

    console.print(table)
    acc = correct / len(examples)
    binary_acc = binary_correct / len(examples)
    console.print(f"\n[bold]Raw 3-way label accuracy: {correct}/{len(examples)} = {acc:.1%}[/bold]")
    console.print(
        f"[bold]Derived block-vs-allow accuracy (C vs not-C): "
        f"{binary_correct}/{len(examples)} = {binary_acc:.1%}[/bold]"
    )
    console.print(f"Latency, RISK_SCALE ask (assess_risk): {_latency_summary(latencies)}")
    block_latencies = [ask_allow_block(judge, ex["state"]).latency_ms for ex in examples]
    console.print(f"Latency, both ALLOW_BLOCK orders (should_block): {_latency_summary(block_latencies)}")


def _latency_summary(latencies: list[float]) -> str:
    return (
        f"mean={sum(latencies)/len(latencies):.0f}ms "
        f"min={min(latencies):.0f}ms max={max(latencies):.0f}ms (n={len(latencies)})"
    )


if __name__ == "__main__":
    main()
