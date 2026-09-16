"""Steps 3 + 6 of the build order: fit temperature-scaling calibration on a
held-out split, then honestly report accuracy + expected calibration error
(ECE) on the other split. Also reports the "does this need to block" derived
binary metric, since that's the operationally meaningful one for a tool-call
gate (see run_spike.py for why 3-way label accuracy understates usefulness).

Only runs the model once per example (temperature scaling is refit purely
from stored logits, no extra forward passes needed).

Run: uv run python scripts/run_eval.py
"""

import json
import random
from pathlib import Path

from rich.console import Console
from rich.table import Table

from supervisor import ALLOW_BLOCK, Judge, RISK_SCALE
from supervisor.calibration import CalibrationExample, expected_calibration_error, fit_temperature
from supervisor.types import ProbabilityQuestion

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "eval.jsonl"
CALIBRATION_OUT = Path(__file__).resolve().parent.parent / "data" / "calibration.json"
SEED = 1234
CALIBRATION_FRACTION = 0.6


def main() -> None:
    console = Console()
    examples = [json.loads(line) for line in DATA_PATH.read_text().splitlines() if line.strip()]

    console.print("[bold]Loading judge model...[/bold]")
    judge = Judge()
    console.print(f"Loaded in {judge.load_time_s:.2f}s\n")

    # -- run every example once, capture raw logits + correctness -----------
    # For every "score" example we also directly ask ALLOW_BLOCK against the
    # same state, as its own "choice_block" record. This is the actual
    # question should_block()/the MCP server ask in production - it's a
    # separate forward pass with its own prompt and options, not the same
    # thing as the derived is_block_true/is_block_pred below (which only
    # reads out the RISK_SCALE answer). Previously nothing in this script
    # ever exercised ALLOW_BLOCK directly, so its accuracy/calibration was
    # completely unmeasured despite being the code path callers rely on.
    records = []
    for ex in examples:
        if ex["kind"] == "score":
            result = judge.ask(ex["state"], RISK_SCALE)
            labels = list(result.raw_logits.keys())
            correct_index = labels.index(ex["label"])
            chosen_index = labels.index(result.answer)
        elif ex["kind"] == "probability":
            q = ProbabilityQuestion(statement=ex["statement"])
            result = judge.ask(ex["state"], q)
            labels = list(result.raw_logits.keys())  # ["true", "false"]
            correct_index = labels.index("true" if ex["label"] else "false")
            chosen_index = labels.index(result.answer)
        else:
            raise ValueError(f"unknown kind {ex['kind']!r}")

        records.append({
            "id": ex["id"],
            "kind": ex["kind"],
            "category": ex.get("category", ex["kind"]),
            "logits": [result.raw_logits[l] for l in labels],
            "labels": labels,
            "correct_index": correct_index,
            "chosen_index": chosen_index,
            "is_correct": correct_index == chosen_index,
            "is_block_true": ex["kind"] == "score" and ex["label"] == "C",
            "is_block_pred": ex["kind"] == "score" and result.answer == "C",
        })

        if ex["kind"] == "score":
            block_result = judge.ask(ex["state"], ALLOW_BLOCK)
            block_labels = list(block_result.raw_logits.keys())  # ["A", "B"]
            should_block = ex["label"] == "C"
            block_correct_index = block_labels.index("B" if should_block else "A")
            block_chosen_index = block_labels.index(block_result.answer)
            records.append({
                "id": ex["id"] + "-block",
                "kind": "choice_block",
                "category": ex.get("category", "choice_block"),
                "logits": [block_result.raw_logits[l] for l in block_labels],
                "labels": block_labels,
                "correct_index": block_correct_index,
                "chosen_index": block_chosen_index,
                "is_correct": block_correct_index == block_chosen_index,
                "is_block_true": should_block,
                "is_block_pred": block_result.answer == "B",
            })

    # -- deterministic calibration/test split --------------------------------
    rng = random.Random(SEED)
    shuffled = records[:]
    rng.shuffle(shuffled)
    n_cal = int(len(shuffled) * CALIBRATION_FRACTION)
    cal_records, test_records = shuffled[:n_cal], shuffled[n_cal:]

    cal_examples = [
        CalibrationExample(logits=r["logits"], correct_index=r["correct_index"]) for r in cal_records
    ]
    temperature = fit_temperature(cal_examples)
    console.print(f"[bold]Fitted temperature on {len(cal_records)} calibration examples: T={temperature:.3f}[/bold]\n")

    def confidences_at(recs: list[dict], t: float) -> list[float]:
        out = []
        for r in recs:
            scaled = [x / t for x in r["logits"]]
            m = max(scaled)
            exps = [pow(2.718281828459045, x - m) for x in scaled]
            denom = sum(exps)
            probs = [e / denom for e in exps]
            out.append(probs[r["chosen_index"]])
        return out

    raw_conf = confidences_at(test_records, 1.0)
    cal_conf = confidences_at(test_records, temperature)
    correctness = [r["is_correct"] for r in test_records]

    acc = sum(correctness) / len(test_records)

    # block-vs-allow accuracy, computed two ways, kept separate on purpose:
    # "derived" reads it off the RISK_SCALE answer (label==C -> should have
    # blocked); "direct" is the actual ALLOW_BLOCK question should_block()
    # asks in production. These are different forward passes and can (and
    # did) disagree - conflating them into one number, or including
    # probability-kind records (which always score as a trivial match since
    # both flags default False) was the previous bug here.
    score_test = [r for r in test_records if r["kind"] == "score"]
    block_test = [r for r in test_records if r["kind"] == "choice_block"]
    derived_block_acc = sum(r["is_block_true"] == r["is_block_pred"] for r in score_test) / len(score_test)
    direct_block_acc = sum(r["is_block_true"] == r["is_block_pred"] for r in block_test) / len(block_test)

    raw_ece = expected_calibration_error(raw_conf, correctness)
    cal_ece = expected_calibration_error(cal_conf, correctness)

    console.print(f"[bold]Held-out test set: n={len(test_records)}[/bold]")
    console.print(f"Answer accuracy (all kinds): {sum(correctness)}/{len(test_records)} = {acc:.1%}")
    console.print(f"Block-vs-allow accuracy, derived from RISK_SCALE: {derived_block_acc:.1%} (n={len(score_test)})")
    console.print(
        f"Block-vs-allow accuracy, DIRECT from ALLOW_BLOCK "
        f"(the actual should_block() question): {direct_block_acc:.1%} (n={len(block_test)})"
    )
    console.print(f"ECE before calibration (T=1.0): {raw_ece.ece:.4f}")
    console.print(f"ECE after calibration  (T={temperature:.3f}): {cal_ece.ece:.4f}\n")

    def reliability_table(title: str, ece_result) -> Table:
        t = Table(title=title)
        t.add_column("bin")
        t.add_column("n")
        t.add_column("avg confidence")
        t.add_column("avg accuracy")
        t.add_column("gap")
        for i in range(len(ece_result.bin_edges) - 1):
            lo, hi = ece_result.bin_edges[i], ece_result.bin_edges[i + 1]
            conf, acc_i, n = ece_result.bin_confidence[i], ece_result.bin_accuracy[i], ece_result.bin_count[i]
            if n == 0:
                continue
            gap = abs(conf - acc_i)
            t.add_row(f"{lo:.1f}-{hi:.1f}", str(n), f"{conf:.3f}", f"{acc_i:.3f}", f"{gap:.3f}")
        return t

    console.print(reliability_table("Reliability - raw (T=1.0)", raw_ece))
    console.print(reliability_table(f"Reliability - calibrated (T={temperature:.3f})", cal_ece))

    # per-category breakdown on full dataset (informational, not held-out).
    # RISK_SCALE and ALLOW_BLOCK kept in separate tables - same category
    # label, different question/forward-pass, shouldn't be averaged together.
    def category_table(title: str, kind: str) -> Table:
        t = Table(title=title)
        t.add_column("category")
        t.add_column("n")
        t.add_column("accuracy")
        cats: dict[str, list[bool]] = {}
        for r in records:
            if r["kind"] == kind:
                cats.setdefault(r["category"], []).append(r["is_correct"])
        for cat, vals in sorted(cats.items()):
            t.add_row(cat, str(len(vals)), f"{sum(vals)/len(vals):.1%}")
        return t

    console.print(category_table("Per-category RISK_SCALE accuracy (full dataset, informational)", "score"))
    console.print(category_table("Per-category ALLOW_BLOCK accuracy (full dataset, informational)", "choice_block"))
    console.print(category_table("Per-category PROBABILITY accuracy (full dataset, informational)", "probability"))

    CALIBRATION_OUT.write_text(json.dumps({
        "model_id": judge.model_id,
        "temperature": temperature,
        "n_calibration_examples": len(cal_records),
        "held_out_ece_raw": raw_ece.ece,
        "held_out_ece_calibrated": cal_ece.ece,
        "held_out_accuracy": acc,
        "held_out_block_accuracy_derived": derived_block_acc,
        "held_out_block_accuracy_direct": direct_block_acc,
    }, indent=2))
    console.print(f"\n[bold]Saved calibration to {CALIBRATION_OUT}[/bold]")


if __name__ == "__main__":
    main()
