"""Compare the local Judge against Typesafe's hosted Jev model on the same
held-out eval set run_eval.py uses (data/eval.jsonl), so "how much better is
Jev" has one number instead of a vibe.

Runs every example through both judges with the exact same question set and
split as run_eval.py (see its docstring for why the split and the direct-vs-
derived block accuracy exist), then fits a temperature for each judge on its
own held-out calibration split and reports accuracy / ECE / latency side by
side. Jev's probabilities already come out of the API pre-calibrated, so its
"raw (T=1)" numbers are Typesafe's own calibration, not an untrained model -
the fitted-T row shows whether refitting on top of that helps or hurts.

Needs a Typesafe API key: export TYPESAFE_API_KEY before running. Every
example makes 1-3 real network calls to api.typesafe.ai (RISK_SCALE +
ALLOW_BLOCK + ALLOW_BLOCK_SWAPPED for a "score" example, one NOUL call for a
"probability" example) - for the current eval set that's a few hundred
requests, so this costs Typesafe usage and takes a couple of minutes, unlike
the other scripts here which never leave the machine.

Run: uv run python scripts/compare_jev.py [--limit N]
"""

from __future__ import annotations

import argparse
import json
import random
import time

from rich.console import Console
from rich.table import Table

from supervisor import ALLOW_BLOCK, RISK_SCALE, Judge
from supervisor.calibration import CalibrationExample, expected_calibration_error, fit_temperature, softmax
from supervisor.jev_judge import JevJudge
from supervisor.paths import DATA_DIR
from supervisor.questions import ALLOW_BLOCK_SWAPPED, BLOCK_LABEL, combine_allow_block
from supervisor.types import JudgeAnswer, ProbabilityQuestion

DATA_PATH = DATA_DIR / "eval.jsonl"
SEED = 1234  # same seed as run_eval.py, so the split lines up example-for-example
CALIBRATION_FRACTION = 0.6


def _record(id_: str, kind: str, result: JudgeAnswer, correct_label: str) -> dict:
    labels = list(result.raw_logits)
    return {
        "id": id_,
        "kind": kind,
        "logits": list(result.raw_logits.values()),
        "correct_index": labels.index(correct_label),
        "chosen_index": labels.index(result.answer),
        "is_correct": labels.index(correct_label) == labels.index(result.answer),
        "latency_ms": result.latency_ms,
    }


def run_examples(console: Console, judge, examples: list[dict], label: str) -> tuple[list[dict], list[dict]]:
    """Returns (records, block_records) - block_records hold the deployed
    should_block() rule's plain+swapped pair per score example, kept
    separate exactly like run_eval.py does."""
    records: list[dict] = []
    block_records: list[dict] = []
    t0 = time.perf_counter()
    for i, ex in enumerate(examples):
        if ex["kind"] == "score":
            result = judge.ask(ex["state"], RISK_SCALE)
            records.append(_record(ex["id"], "score", result, ex["label"]))

            should_block = ex["label"] == "C"
            plain = judge.ask(ex["state"], ALLOW_BLOCK)
            swapped = judge.ask(ex["state"], ALLOW_BLOCK_SWAPPED)
            records.append(_record(
                ex["id"] + "-block", "choice_block", plain, BLOCK_LABEL if should_block else "A"
            ))
            block_records.append({"plain": plain, "swapped": swapped, "should_block": should_block})
        elif ex["kind"] == "probability":
            result = judge.ask(ex["state"], ProbabilityQuestion(statement=ex["statement"]))
            records.append(_record(ex["id"], "probability", result, "true" if ex["label"] else "false"))
        else:
            raise ValueError(f"unknown kind {ex['kind']!r}")
        if (i + 1) % 20 == 0 or i + 1 == len(examples):
            console.print(f"  [{label}] {i + 1}/{len(examples)} examples", end="\r")
    console.print(f"  [{label}] {len(examples)}/{len(examples)} examples done in {time.perf_counter() - t0:.1f}s")
    return records, block_records


def summarize(console: Console, label: str, records: list[dict], block_records: list[dict]) -> dict:
    rng = random.Random(SEED)
    shuffled = records[:]
    rng.shuffle(shuffled)
    n_cal = int(len(shuffled) * CALIBRATION_FRACTION)
    cal_records, test_records = shuffled[:n_cal], shuffled[n_cal:]

    cal_examples = [
        CalibrationExample(logits=r["logits"], correct_index=r["correct_index"]) for r in cal_records
    ]
    temperature = fit_temperature(cal_examples)

    def confidences_at(recs: list[dict], t: float) -> list[float]:
        return [softmax(r["logits"], t)[r["chosen_index"]] for r in recs]

    correctness = [r["is_correct"] for r in test_records]
    raw_ece = expected_calibration_error(confidences_at(test_records, 1.0), correctness)
    cal_ece = expected_calibration_error(confidences_at(test_records, temperature), correctness)
    acc = sum(correctness) / len(test_records)

    outcomes = [
        (combine_allow_block(b["plain"], b["swapped"]).answer == BLOCK_LABEL) == b["should_block"]
        for b in block_records
    ]
    block_acc = sum(outcomes) / len(outcomes) if outcomes else float("nan")

    latencies = sorted(r["latency_ms"] for r in records)
    p50 = latencies[len(latencies) // 2]
    p95 = latencies[int(len(latencies) * 0.95)]

    console.print(f"\n[bold]{label}[/bold]")
    console.print(f"  fitted temperature: T={temperature:.3f}")
    console.print(f"  held-out answer accuracy: {sum(correctness)}/{len(test_records)} = {acc:.1%}")
    console.print(f"  held-out should_block accuracy (deployed rule, forced): {block_acc:.1%} (n={len(outcomes)})")
    console.print(f"  ECE raw (T=1.0): {raw_ece.ece:.4f}   ECE fitted (T={temperature:.3f}): {cal_ece.ece:.4f}")
    console.print(f"  latency per ask: p50={p50:.0f}ms  p95={p95:.0f}ms  n={len(records)}")

    return {
        "label": label, "temperature": temperature, "accuracy": acc, "block_accuracy": block_acc,
        "ece_raw": raw_ece.ece, "ece_fitted": cal_ece.ece, "p50_ms": p50, "p95_ms": p95,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="only run the first N eval examples")
    parser.add_argument(
        "--skip-local", action="store_true",
        help="don't load/run the local judge (e.g. to avoid reheating a laptop for a rerun "
        "that only changed the Jev side) - prints just the Jev summary, no comparison table",
    )
    args = parser.parse_args()

    console = Console()
    examples = [json.loads(line) for line in DATA_PATH.read_text().splitlines() if line.strip()]
    if args.limit:
        examples = examples[: args.limit]
    console.print(f"Loaded {len(examples)} eval examples from {DATA_PATH}\n")

    local_summary = None
    if not args.skip_local:
        console.print("[bold]Loading local judge...[/bold]")
        local = Judge.calibrated()
        console.print(f"Loaded {local.model_id} ({local.backend_name}) in {local.load_time_s:.2f}s\n")
        console.print("[bold]Running local judge over the eval set...[/bold]")
        local_records, local_block = run_examples(console, local, examples, "local")
        local_summary = summarize(console, f"Local: {local.model_id} ({local.backend_name})", local_records, local_block)

    console.print("[bold]Connecting to Typesafe Jev...[/bold]")
    jev = JevJudge()
    console.print(f"Using model {jev.model_id}\n")

    console.print("[bold]Running Jev over the eval set...[/bold]")
    jev_records, jev_block = run_examples(console, jev, examples, "jev")
    jev_summary = summarize(console, f"Jev: {jev.model_id}", jev_records, jev_block)

    if local_summary is None:
        console.print("\n[dim]--skip-local set: no comparison table, just the Jev numbers above.[/dim]")
        return

    table = Table(title="Local vs. Jev")
    table.add_column("metric")
    table.add_column("local")
    table.add_column("jev")
    table.add_row("held-out accuracy", f"{local_summary['accuracy']:.1%}", f"{jev_summary['accuracy']:.1%}")
    table.add_row(
        "should_block accuracy", f"{local_summary['block_accuracy']:.1%}", f"{jev_summary['block_accuracy']:.1%}"
    )
    table.add_row("ECE (T=1.0)", f"{local_summary['ece_raw']:.4f}", f"{jev_summary['ece_raw']:.4f}")
    table.add_row("ECE (fitted T)", f"{local_summary['ece_fitted']:.4f}", f"{jev_summary['ece_fitted']:.4f}")
    table.add_row("latency p50", f"{local_summary['p50_ms']:.0f}ms", f"{jev_summary['p50_ms']:.0f}ms")
    table.add_row("latency p95", f"{local_summary['p95_ms']:.0f}ms", f"{jev_summary['p95_ms']:.0f}ms")
    console.print()
    console.print(table)


if __name__ == "__main__":
    main()
