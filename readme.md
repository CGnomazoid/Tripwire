# Local Agent Supervisor (working name: mini-jev / sentinel)

## One-line pitch
An open-weight, locally-run "System One" style judge model that sits inline with any agent framework and gates risky tool calls using typed questions (choice / score / probability) with calibrated confidence — no cloud API, no waitlist.

## Why this project exists
- Inspired by TypeSafe AI's Jev: a fast, non-generative "System One" model that answers typed questions (Choice / Score / Probability) against a text "state" and returns calibrated confidence, instead of token-by-token generation.
- The core mechanism behind Jev (single forward pass, read logprobs on candidate answer tokens, skip generation) is not proprietary — it's used in eval harnesses today. The likely defensible piece is calibration.
- Open-source agent frameworks (LangChain, CrewAI, AutoGen, OpenHands, local Ollama agents) have no standard, cheap, always-on way to catch an agent about to do something dumb or destructive. A fast local judge model that runs on every single tool call (not sampled) fills that gap.
- Goal: be a credible open-weight "local Jev" for agent safety/supervision, sized to run on a single consumer/prosumer GPU.

## Goal / definition of done for v1
A pip-installable Python package that:
1. Loads a small-to-mid open-weight model (start 7-8B, target 32B class once feasibility is proven)
2. Exposes a simple API: given a `state` (text) and a typed `question` (Choice/Score/Probability), returns a structured answer with a calibrated confidence score
3. Runs well under 1s per call on reasonable local hardware (target: sub-300ms as a stretch goal — be honest about actual measured numbers, don't fudge)
4. Ships with at least one worked demo: gating a real (if toy) agent's tool calls in real time
5. Includes an evaluation script that measures whether confidence scores are actually calibrated (does "80% confident" correspond to ~80% correct)

## Non-goals for v1
- Not trying to match TypeSafe/Jev's frontier-level intelligence or their RLCD research — "good enough to catch obviously bad actions" is the bar, not state-of-the-art judgment
- Not building a new agent framework — this augments existing ones
- Not doing multimodal input (video/image) — text state only, matching Jev's scope
- Not trying to be adversarially hardened against a user deliberately trying to fool their own supervisor — threat model is "catch honest mistakes," not "catch a malicious agent actively evading detection"

## Proposed architecture (NOT final — use your judgment)
```
[state text] + [typed question]
      -> tokenize, single forward pass
      -> read logits/logprobs at candidate-answer token positions
      -> softmax -> raw probability distribution
      -> calibration layer (temperature scaling initially) -> calibrated confidence
      -> structured output (JSON: {answer, confidence, raw_probs})
```

Three question types to support, mirroring Jev's design (a reasonable existing pattern to imitate, not required):
- **Choice**: pick one of N labeled options
- **Score**: place state on an ordered scale (e.g., low/medium/high risk)
- **Probability**: return P(statement is true) for a single statement

## Suggested build order (rough, adjust freely)
1. **Spike**: prove the mechanism works at all. Pick Qwen2.5-7B-Instruct or Llama-3.1-8B-Instruct, hand-write 10-20 (state, question, options, correct answer) pairs, do single-forward-pass logprob extraction, see if the raw (uncalibrated) top answer is usually right.
2. **Interface**: define the Python API surface (`Judge.ask(state, question)` returning a typed result) before optimizing anything.
3. **Calibration v0**: temperature scaling against a small held-out labeled set (50-200 examples). Measure calibration with reliability diagrams / ECE (expected calibration error).
4. **Speed pass**: quantize (int4/int8 via bitsandbytes or GGUF), measure real latency, consider vLLM or llama.cpp for serving if raw HF transformers is too slow.
5. **Demo integration**: wrap a trivial agent loop (doesn't need to be a "real" framework — a hand-rolled one is fine) and show it blocking/flagging a genuinely risky tool call in real time.
6. **Eval + writeup**: build the calibration benchmark script, run it, be honest about the numbers, write a README that doesn't oversell it.
7. **Stretch**: LoRA fine-tune on distilled data from a frontier model (Claude/GPT) to improve judgment quality beyond temperature scaling alone; try the 32B tier once 7-8B is proven out.

## Key open questions for Claude Code to figure out along the way
- HF transformers vs vLLM vs llama.cpp for the inference backend — tradeoffs between ease-of-iteration and real-world speed
- Best approach for candidate-answer token extraction when options are multi-token (a single-token option is cleaner but off-the-shelf models may not give you that luxury)
- How much calibration data is "enough" — start small and see
- Whether 7-8B judgment quality is good enough to be useful at all before investing in 32B

## Tech stack starting point (suggestions, not requirements)
Python, PyTorch, Hugging Face `transformers`, `bitsandbytes` or GGUF/llama.cpp for quantization, numpy/scikit-learn for the eval/calibration script, Poetry or plain pip for packaging.

## Note to future Claude
This is a fresh idea sketched out in conversation, not a spec handed down from on high. If you find a better way to structure the model interface, the calibration approach, or the build order, do that instead — the outline above is a starting point, not a constraint. Flag tradeoffs to Colin as you hit them rather than silently picking a path he might disagree with.

---

## Status (2026-09-16, first build session)

Steps 1-5 of the build order are done and working end to end. Numbers below are real measured output, not estimates.

**Backend choice: MLX, not HF transformers/vLLM/llama.cpp.** This is Apple Silicon (M4 Max, 36GB unified memory), no CUDA, no `torch` wheel currently installed for the system's Python. `mlx` + `mlx-lm` were already present, give direct Metal-accelerated access to raw logits after a single forward pass (no server round trip, no generation), and needed zero glue code to get candidate-token logprobs out. This is a better fit for "single forward pass, read logprobs" than going through Ollama's HTTP API. Ollama stays installed for other things but isn't in this project's path.

**Model: `mlx-community/Qwen2.5-7B-Instruct-4bit`** (auto-downloaded from HF on first run, ~4.3GB). Single-token option labels (`A`/`B`/`C`, `T`/`F`) verified against this tokenizer - the readme's flagged risk about multi-token options didn't end up mattering because single letters tokenize cleanly.

**Measured latency:** ~140ms steady-state forward pass on M4 Max; ~265-290ms end-to-end through the full `Judge.ask()` API (tokenize + forward + softmax). Comfortably under the 1s target and close to the 300ms stretch goal.

**Measured accuracy (honest numbers, see `scripts/run_spike.py` and `scripts/run_eval.py`):**
- Spike set (20 hand-picked, clearly-easy examples): 75% raw 3-way risk-label accuracy (low/medium/high), but **100%** on the operationally meaningful derived decision (does this need to block for human confirmation, i.e. is it high-risk or not). The model nails the two extremes and gets fuzzy specifically on "low vs medium" - which is the least consequential mistake to make.
- Larger eval set (97 examples spanning file ops, db, network, finance, comms, infra, code exec, account mgmt, plus 15 probability-type questions; 60/40 calibration/test split, seeded): held-out block-vs-allow accuracy **87.2%**, 3-way label accuracy 71.8%, probability-question accuracy 93.3%.
- Raw model is meaningfully overconfident: expected calibration error (ECE) starts at **0.236** (T=1.0) and drops to **0.112** after fitting a single scalar temperature (T≈3.33) via NLL minimization on the calibration split - temperature scaling roughly halves the calibration gap, as hoped, with no accuracy cost (T doesn't change the argmax, only the reported confidence).
- Weakest categories: `fs_delete` and `borderline` (both 50%) - worth another look before trusting this on real destructive-filesystem calls specifically. `account`, `fs_write`, `probability` are strong (93-100%).

**What's built:**
- `src/supervisor/` - the package: `types.py` (pydantic Choice/Score/Probability question + JudgeAnswer types), `judge.py` (the `Judge` class - loads the model once, `judge.ask(state, question)` does one forward pass and returns a calibrated structured answer), `calibration.py` (temperature-scaling fit + ECE), `questions.py` (canonical reusable risk-scale and allow/block questions).
- `scripts/gen_dataset.py` - generates `data/spike.jsonl` (20) and `data/eval.jsonl` (97) from hand-authored, hand-labeled tool-call scenarios (ground truth is a judgment call, not an oracle - see non-goals above).
- `scripts/run_spike.py` - step 1, raw mechanism sanity check.
- `scripts/run_eval.py` - steps 3+6, fits calibration and reports accuracy/ECE/reliability tables on a held-out split; writes `data/calibration.json`.
- `demo/agent_demo.py` - step 5, a toy scripted agent loop that actually executes benign tool calls into `demo/sandbox/` and blocks the risky ones (an unverified $500k transfer, a recursive delete) before they run, using the fitted calibration.
- `tests/` - fast pure-math unit tests for the calibration layer and pydantic validators (no model load needed).

**Not done yet (steps 4, 7, and general hardening):**
- No real speed/quantization pass beyond "already 4-bit via mlx-community." Haven't tried int8 vs 4-bit tradeoffs or a smaller model.
- Haven't tried the 32B tier, or a LoRA fine-tune on distilled frontier-model judgments (step 7 stretch goal).
- No reliability-diagram plot, just text tables (kept matplotlib off the dependency list on purpose - can add if wanted).
- Eval set is hand-authored by Claude, not reviewed by a human yet - the `borderline` category especially deserves a sanity check from Colin on whether the assigned ground-truth labels are actually the risk level he'd want.
- No pip-packaging/CLI polish yet (def-of-done item 1 - "pip-installable" - the package structure is there but nothing's been published or given a CLI entry point).

**Known environment gotcha (macOS + iCloud Drive + uv + recent CPython):** recent CPython point releases (confirmed on 3.13.12 and 3.14.3) patched `site.py` to silently skip any `.pth` file with the macOS "hidden" (`UF_HIDDEN`) flag set - a security fix, not a bug, but it collides with `uv`, which sets that flag on the editable-install `.pth` file it writes for this project. Initially this only seemed to happen after `uv sync`/`uv add`, but it turned out to reapply unpredictably on plain `uv run` too (this repo lives under `~/Library/Mobile Documents/com~apple~CloudDocs`, i.e. iCloud Drive - possibly related). Net effect: `import supervisor` can silently break with zero error output, at any time.

The durable fix: **use `./run` instead of `uv run` directly** everywhere below. `./run` is a one-line wrapper that passes `--env-file .env`, setting `PYTHONPATH=src` - the interpreter applies `PYTHONPATH` before `site.py`'s `.pth` scanning ever runs, so it's immune to whatever is flipping that flag. `scripts/setup.sh` still exists and clears the flag too, but `./run` is the actual fix; the flag-clear alone kept silently regressing.

**How to run things:**
```bash
./scripts/setup.sh          # one-time / after any dependency change
./run python scripts/run_spike.py   # ~20s, proves the mechanism
./run python scripts/run_eval.py    # ~30s, calibration + honest accuracy/ECE report
./run python demo/agent_demo.py     # the live gating demo (actually executes benign calls)
./run python scripts/try_it.py      # interactive judge-only playground, see below - never executes anything
./run pytest -q                     # fast unit tests, no model load
./run pytest tests/test_judge.py -v # judge integration tests (loads the model)
```

**`scripts/try_it.py`** - type or pipe in any tool-call description, including intentionally dangerous ones, and see how the judge scores it (risk scale, allow/block, P(irreversible harm)). It is read-only by construction: your input is only ever passed as the `state` string to `Judge.ask()` (tokenized and fed through the model as text) - there is no `exec`/`eval`/`subprocess`/`os.system` anywhere near it, so nothing you type can actually run, no matter how it reads. Single-shot mode: `./run python scripts/try_it.py "Tool call: run_shell(cmd='rm -rf /')"`.
