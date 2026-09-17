"""Steps 3 + 6 of the build order: fit temperature-scaling calibration on a
held-out split, then honestly report accuracy + expected calibration error
(ECE) on the other split. Also reports block-vs-allow accuracy, including
for should_block() exactly as the MCP server runs it (see
report_should_block), since that's the operationally meaningful number for
a tool-call gate (see run_spike.py for why 3-way label accuracy understates
usefulness).

Runs the model once per example per question: temperature scaling is refit
purely from stored logits, and answers are re-scaled to the fitted
temperature from those same logits (JudgeAnswer.with_temperature), so
neither needs extra forward passes.

Run: uv run python scripts/run_eval.py
"""

import json
import random

from rich.console import Console
from rich.table import Table

from supervisor import ALLOW_BLOCK, RISK_SCALE, Judge, calibration_store
from supervisor.calibration import CalibrationExample, expected_calibration_error, fit_temperature, softmax
from supervisor.paths import DATA_DIR
from supervisor.questions import (
    ALLOW_BLOCK_SWAPPED,
    ALLOW_LABEL,
    BLOCK_LABEL,
    DEFAULT_UNCERTAIN_THRESHOLD,
    combine_allow_block,
    decide,
)
from supervisor.types import JudgeAnswer, ProbabilityQuestion

DATA_PATH = DATA_DIR / "eval.jsonl"
SEED = 1234
CALIBRATION_FRACTION = 0.6


def _record(
    id_: str, kind: str, category: str, result: JudgeAnswer, correct_label: str,
    is_block_true: bool, is_block_pred: bool,
) -> dict:
    labels = list(result.raw_logits)
    correct_index = labels.index(correct_label)
    chosen_index = labels.index(result.answer)
    return {
        "id": id_,
        "kind": kind,
        "category": category,
        "logits": list(result.raw_logits.values()),
        "labels": labels,
        "correct_index": correct_index,
        "chosen_index": chosen_index,
        "is_correct": correct_index == chosen_index,
        "is_block_true": is_block_true,
        "is_block_pred": is_block_pred,
    }


def main() -> None:
    console = Console()
    examples = [json.loads(line) for line in DATA_PATH.read_text().splitlines() if line.strip()]

    console.print("[bold]Loading judge model...[/bold]")
    judge = Judge()
    console.print(f"Loaded in {judge.load_time_s:.2f}s\n")

    # -- run every example once, capture raw logits + correctness -----------
    # For every "score" example we also directly ask ALLOW_BLOCK against the
    # same state, as its own "choice_block" record. This is the question
    # should_block()/the MCP server ask in production - it's a separate
    # forward pass with its own prompt and options, not the same thing as
    # the derived is_block_true/is_block_pred on the score record (which
    # only reads out the RISK_SCALE answer).
    #
    # should_block() also asks ALLOW_BLOCK_SWAPPED (questions.ask_allow_block),
    # so that pass is run too and kept alongside the plain one in
    # block_answers. It is deliberately NOT added to `records`: those feed
    # the temperature fit and the calibration/test split, and adding records
    # would change both, making numbers incomparable with earlier runs.
    records = []
    block_answers: dict[str, tuple[JudgeAnswer, JudgeAnswer, bool]] = {}
    for ex in examples:
        category = ex.get("category", ex["kind"])
        if ex["kind"] == "score":
            result = judge.ask(ex["state"], RISK_SCALE)
            records.append(_record(
                ex["id"], "score", category, result, ex["label"],
                is_block_true=ex["label"] == "C", is_block_pred=result.answer == "C",
            ))

            should_block = ex["label"] == "C"
            plain = judge.ask(ex["state"], ALLOW_BLOCK)
            swapped = judge.ask(ex["state"], ALLOW_BLOCK_SWAPPED)
            block_id = ex["id"] + "-block"
            records.append(_record(
                block_id, "choice_block", category, plain,
                BLOCK_LABEL if should_block else ALLOW_LABEL,
                is_block_true=should_block, is_block_pred=plain.answer == BLOCK_LABEL,
            ))
            block_answers[block_id] = (plain, swapped, should_block)
        elif ex["kind"] == "probability":
            result = judge.ask(ex["state"], ProbabilityQuestion(statement=ex["statement"]))
            records.append(_record(
                ex["id"], "probability", category, result, "true" if ex["label"] else "false",
                is_block_true=False, is_block_pred=False,
            ))
        else:
            raise ValueError(f"unknown kind {ex['kind']!r}")

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
        return [softmax(r["logits"], t)[r["chosen_index"]] for r in recs]

    raw_conf = confidences_at(test_records, 1.0)
    cal_conf = confidences_at(test_records, temperature)
    correctness = [r["is_correct"] for r in test_records]

    acc = sum(correctness) / len(test_records)

    # block-vs-allow accuracy, computed two ways, kept separate on purpose:
    # "derived" reads it off the RISK_SCALE answer (label==C -> should have
    # blocked); "direct" is the ALLOW_BLOCK question should_block() asks.
    # These are different forward passes and can (and did) disagree -
    # conflating them into one number, or including probability-kind
    # records (which always score as a trivial match since both flags
    # default False) was a previous bug here.
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
        f"(one option order, forced allow/block): {direct_block_acc:.1%} (n={len(block_test)})"
    )
    console.print(f"ECE before calibration (T=1.0): {raw_ece.ece:.4f}")
    console.print(f"ECE after calibration  (T={temperature:.3f}): {cal_ece.ece:.4f}\n")

    should_block_stats = report_should_block(
        console, [block_answers[r["id"]] for r in block_test], temperature
    )

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

    # Merged into the store under judge.model_id, not overwritten wholesale -
    # calibration.json holds one entry per model, so evaluating a different
    # model doesn't destroy an already-calibrated one (see calibration_store.py).
    calibration_store.save_entry(judge.model_id, {
        "temperature": temperature,
        "n_calibration_examples": len(cal_records),
        "held_out_ece_raw": raw_ece.ece,
        "held_out_ece_calibrated": cal_ece.ece,
        "held_out_accuracy": acc,
        "held_out_block_accuracy_derived": derived_block_acc,
        "held_out_block_accuracy_direct": direct_block_acc,
        "held_out_should_block_accuracy_of_decided": should_block_stats["accuracy_of_decided"],
        "held_out_should_block_uncertain_rate": should_block_stats["uncertain_rate"],
    })
    console.print(f"\n[bold]Saved calibration for {judge.model_id} to {calibration_store.CALIBRATION_PATH}[/bold]")


def report_should_block(
    console: Console, answers: list[tuple[JudgeAnswer, JudgeAnswer, bool]], temperature: float
) -> dict[str, float]:
    """Held-out allow/block/uncertain outcomes for three decision rules,
    each adding one piece of what should_block() actually does, so the
    effect of each piece is visible on its own:

    1. plain ALLOW_BLOCK, forced to allow/block (the DIRECT number above)
    2. plus the uncertain threshold on calibrated confidence
    3. plus the swapped-order agreement check - should_block() as deployed

    An "uncertain" isn't scored as right or wrong - it's handed to a human -
    so each rule reports how often it decides, how accurate it is when it
    does, and which way its mistakes go: a false allow (ran something that
    should have been confirmed) is the costly one for a gate.

    Returns the deployed rule's accuracy-of-decided and uncertain rate.
    """
    threshold = DEFAULT_UNCERTAIN_THRESHOLD

    def plain_forced(plain: JudgeAnswer, _: JudgeAnswer) -> str:
        return decide(plain.answer, plain.confidence, threshold=0.0)

    def plain_threshold(plain: JudgeAnswer, _: JudgeAnswer) -> str:
        cal = plain.with_temperature(temperature)
        return decide(cal.answer, cal.confidence, threshold)

    def deployed(plain: JudgeAnswer, swapped: JudgeAnswer) -> str:
        merged = combine_allow_block(plain.with_temperature(temperature), swapped.with_temperature(temperature))
        return decide(merged.answer, merged.confidence, threshold)

    table = Table(title=f"should_block() decision rules, held-out (n={len(answers)}, T={temperature:.3f})")
    for column in ("rule", "decided", "accuracy of decided", "uncertain", "false allow", "false block"):
        table.add_column(column)

    stats: dict[str, float] = {}
    rules = [
        ("ALLOW_BLOCK, forced allow/block", plain_forced),
        (f"+ uncertain below {threshold:.0%}", plain_threshold),
        ("+ both option orders agree (deployed)", deployed),
    ]
    for name, rule in rules:
        outcomes = [(rule(plain, swapped), should_block) for plain, swapped, should_block in answers]
        decided = [(d, truth) for d, truth in outcomes if d != "uncertain"]
        correct = sum((d == "block") == truth for d, truth in decided)
        false_allow = sum(d == "allow" and truth for d, truth in decided)
        false_block = sum(d == "block" and not truth for d, truth in decided)
        n_uncertain = len(outcomes) - len(decided)
        accuracy = correct / len(decided) if decided else 0.0
        table.add_row(
            name,
            f"{len(decided)}/{len(outcomes)}",
            f"{correct}/{len(decided)} = {accuracy:.1%}",
            f"{n_uncertain} ({n_uncertain / len(outcomes):.1%})",
            str(false_allow),
            str(false_block),
        )
        stats = {"accuracy_of_decided": accuracy, "uncertain_rate": n_uncertain / len(outcomes)}

    console.print(table)
    console.print()
    return stats


if __name__ == "__main__":
    main()
