"""MLX backend: Apple Silicon, Metal-accelerated, single forward pass."""

from __future__ import annotations


class MLXBackend:
    def __init__(self, model_id: str):
        import mlx.core as mx
        from mlx_lm import load as mlx_load

        self._mx = mx
        self.model, self.tokenizer = mlx_load(model_id)

    def candidate_logits(self, prompt_text: str, candidate_ids: list[int]) -> list[float]:
        mx = self._mx
        tokens = self.tokenizer.encode(prompt_text)
        input_ids = mx.array([tokens])
        logits = self.model(input_ids)
        last = logits[0, -1, :]
        candidates = last[mx.array(candidate_ids)]
        mx.eval(candidates)
        return [float(x) for x in candidates.tolist()]
