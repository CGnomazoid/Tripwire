"""Pluggable inference backends.

Judge itself doesn't know or care whether it's running on Apple Silicon or
an Nvidia GPU - it only needs a `.tokenizer` (HF-compatible: `.encode()`,
`.apply_chat_template()`) and a `candidate_logits(prompt_text, candidate_ids)`
method that runs exactly one forward pass and returns the logits at the last
token position for the given candidate token ids, in order.

Two backends implement that interface:
- `mlx_backend.MLXBackend`   - Apple Silicon, via MLX. Default on macOS.
- `torch_backend.TorchBackend` - PyTorch + transformers. Uses CUDA when
  available (Windows/Linux + Nvidia), otherwise falls back to CPU. Default
  everywhere that isn't macOS.

Backend selection: pass `backend="mlx"` / `backend="torch"` explicitly, set
the `SUPERVISOR_BACKEND` env var, or leave it as `None`/`"auto"` to pick
based on the current platform.
"""

from __future__ import annotations

import os
import platform
from collections.abc import Sequence
from typing import Protocol


class InferenceBackend(Protocol):
    tokenizer: object
    model: object

    def candidate_logits(
        self, prompt_text: str, candidate_ids: list[int], prefix_ends: Sequence[int] = ()
    ) -> list[float]:
        """Run one forward pass over `prompt_text` and return the logits at
        the last token position for exactly `candidate_ids`, in order.

        `prefix_ends` are character offsets into `prompt_text` where a
        prefix other prompts are likely to share ends, in increasing order.
        A backend may reuse attention state for such a prefix instead of
        recomputing it (see MLXBackend) or ignore them entirely; either way
        the result must depend only on the arguments, never on which prompts
        came before.

        `prompt_text` is already rendered through the tokenizer's chat
        template, special tokens included, so encode it with
        add_special_tokens=False - otherwise a model whose tokenizer adds a
        BOS token (Llama 3, Mistral, ...) sees two of them."""
        ...


# Each backend's own reasonable default model, used when the caller doesn't
# pass model_id explicitly. The mlx-community repo ships a pre-quantized
# 4-bit MLX model; the torch backend quantizes the plain HF checkpoint
# itself at load time (see torch_backend.py) so the two stay comparable in
# both quality and footprint.
DEFAULT_MODEL_IDS = {
    "mlx": "mlx-community/Qwen2.5-7B-Instruct-4bit",
    "torch": "Qwen/Qwen2.5-7B-Instruct",
}

_KNOWN_BACKENDS = set(DEFAULT_MODEL_IDS)


def resolve_backend_name(requested: str | None) -> str:
    requested = requested or os.environ.get("SUPERVISOR_BACKEND")
    if not requested or requested == "auto":
        return "mlx" if platform.system() == "Darwin" else "torch"
    if requested not in _KNOWN_BACKENDS:
        raise ValueError(f"unknown backend {requested!r}; expected 'mlx', 'torch', or 'auto'")
    return requested


def load_backend(name: str, model_id: str) -> InferenceBackend:
    if name == "mlx":
        try:
            from supervisor.backends.mlx_backend import MLXBackend
        except ImportError as e:
            raise ImportError(
                "The MLX backend needs `mlx` and `mlx-lm`, which only install on "
                "Apple Silicon. Install them with `uv sync --extra mlx`, or pass "
                "backend='torch' (or set SUPERVISOR_BACKEND=torch) to use the "
                "PyTorch/CUDA backend instead."
            ) from e
        return MLXBackend(model_id)

    if name == "torch":
        try:
            from supervisor.backends.torch_backend import TorchBackend
        except ImportError as e:
            raise ImportError(
                "The PyTorch backend needs `torch` and `transformers`. Install "
                "them with `uv sync --extra cuda` (Nvidia GPU) or `uv sync "
                "--extra torch` (CPU-only)."
            ) from e
        return TorchBackend(model_id)

    raise ValueError(f"unknown backend {name!r}; expected 'mlx' or 'torch'")
