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
