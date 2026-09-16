"""Step 1 of the build order: prove the mechanism works at all.

Loads the hand-written spike set (data/spike.jsonl), asks the canonical
risk-scale question for each state, and reports whether the raw
(uncalibrated) top answer matches the hand-labeled ground truth. No
calibration is applied here - that's a later step.

Run: uv run python scripts/run_spike.py
"""

import json
import time
from pathlib import Path

from rich.console import Console
from rich.table import Table

from supervisor import Judge, RISK_SCALE

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "spike.jsonl"


def main() -> None:
    console = Console()
    examples = [json.loads(line) for line in DATA_PATH.read_text().splitlines() if line.strip()]

    console.print(f"[bold]Loading judge model...[/bold]")
    judge = Judge()
    console.print(f"Loaded in {judge.load_time_s:.2f}s\n")

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
    console.print(
        f"Latency: mean={sum(latencies)/len(latencies):.0f}ms "
        f"min={min(latencies):.0f}ms max={max(latencies):.0f}ms"
    )


if __name__ == "__main__":
    main()
