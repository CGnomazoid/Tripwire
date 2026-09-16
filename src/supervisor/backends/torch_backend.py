"""PyTorch backend: CUDA (Nvidia, e.g. Windows/Linux) when available,
otherwise CPU. Same single-forward-pass, read-candidate-logits contract as
the MLX backend - no autoregressive generation here either.

Not yet exercised against real Nvidia hardware as part of this project (the
reference machine is Apple Silicon) - the mechanism is the same one MLX uses
and transformers' documented API, but treat the CUDA path as unverified
until someone runs it for real. See the README.
"""

from __future__ import annotations

from typing import Any


class TorchBackend:
    def __init__(self, model_id: str, device: str | None = None, load_in_4bit: bool | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        # Default to 4-bit quantization on CUDA (via bitsandbytes) to keep
        # VRAM requirements in reach of a single consumer GPU, mirroring the
        # MLX backend's pre-quantized default model. Falls back to an
        # unquantized load if bitsandbytes isn't installed, or on CPU where
        # it isn't supported.
        quantize = load_in_4bit if load_in_4bit is not None else self.device == "cuda"
        model_kwargs: dict[str, Any] = {}
        if quantize:
            try:
                from transformers import BitsAndBytesConfig

                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                )
                model_kwargs["device_map"] = self.device
            except ImportError:
                quantize = False
        if not quantize:
            model_kwargs["torch_dtype"] = torch.float16 if self.device == "cuda" else torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
        if not quantize:
            self.model = self.model.to(self.device)
        self.model.eval()

    def candidate_logits(self, prompt_text: str, candidate_ids: list[int]) -> list[float]:
        torch = self._torch
        tokens = self.tokenizer.encode(prompt_text)
        input_ids = torch.tensor([tokens], device=self.model.device)
        with torch.no_grad():
            logits = self.model(input_ids).logits
        last = logits[0, -1, :]
        candidates = last[torch.tensor(candidate_ids, device=last.device)]
        return [float(x) for x in candidates.tolist()]
