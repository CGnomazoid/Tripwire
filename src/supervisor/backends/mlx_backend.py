"""MLX backend: Apple Silicon, Metal-accelerated, single forward pass."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence

# Prefix KV entries kept at once. Each is roughly 57 KB per token for the
# default 7B model (28 layers x 4 KV heads x 128 dims x keys+values, fp16),
# and these prompts' reusable prefixes run 25-130 tokens: ~1.5-7.5 MB an
# entry, so a full cache is tens of MB beside ~4.3 GB of weights. Eight
# covers the three per-kind system prompts plus the per-state prefixes of
# the question pairs asked back to back (see candidate_logits).
PREFIX_CACHE_ENTRIES = 8


class MLXBackend:
    def __init__(self, model_id: str):
        import mlx.core as mx
        from mlx_lm import load as mlx_load
        from mlx_lm.models.cache import KVCache, make_prompt_cache

        self._mx = mx
        self._make_prompt_cache = make_prompt_cache
        self.model, self.tokenizer = mlx_load(model_id)
        # Prefix reuse needs plain, growable per-layer KV caches. Sliding-
        # window or recurrent layers (some other architectures) can't be
        # seeded from a stored prefix this way, so those models always run
        # the whole prompt in one pass instead.
        self._reuse_prefixes = all(type(c) is KVCache for c in make_prompt_cache(self.model))
        self._prefixes: OrderedDict[tuple[int, ...], list] = OrderedDict()

    def candidate_logits(
        self, prompt_text: str, candidate_ids: list[int], prefix_ends: Sequence[int] = ()
    ) -> list[float]:
        """See InferenceBackend.candidate_logits.

        The prompt is run in segments split at `prefix_ends`, and the
        attention keys/values for each prefix are kept, so a later prompt
        sharing that prefix only runs what comes after it. Judge marks two
        such points: the end of the per-kind system prompt (shared by every
        ask of that kind) and the point where a question's own options
        begin (shared by ALLOW_BLOCK and ALLOW_BLOCK_SWAPPED on one state).
        Prefill time grows about linearly with tokens, so should_block's
        second ordering costs a fraction of its first.

        Always splitting at the same points, cached or not, is what keeps
        this deterministic: a cold and a warm cache run the same
        computations on the same values, so the logits are bit-identical
        either way. They're not bit-identical to one unsplit pass, though -
        attention over [cached prefix + new tokens] takes different fp16
        kernel paths than one pass over everything, moving logits by a few
        hundredths. That's the same size as any other fp16 kernel change,
        and it's why scripts/run_eval.py numbers are measured through this
        path.
        """
        mx = self._mx
        # prompt_text is chat-template output, which already carries any
        # special tokens the model wants (see backends/__init__.py)
        tokens = self.tokenizer.encode(prompt_text, add_special_tokens=False)

        cache_state = None
        start = 0
        for end in self._token_boundaries(prompt_text, tokens, prefix_ends):
            key = tuple(tokens[:end])
            entry = self._prefixes.get(key)
            if entry is None:
                cache = self._seeded_cache(cache_state)
                # Only the cache is evaluated, so MLX never computes this
                # segment's output logits at all.
                self.model(mx.array([tokens[start:end]]), cache=cache)
                # contiguous(): KVCache over-allocates in 256-token steps,
                # and a slice of that buffer would keep all of it alive.
                entry = [(mx.contiguous(k), mx.contiguous(v)) for k, v in (c.state for c in cache)]
                mx.eval(entry)
                self._prefixes[key] = entry
                if len(self._prefixes) > PREFIX_CACHE_ENTRIES:
                    self._prefixes.popitem(last=False)
            else:
                self._prefixes.move_to_end(key)
            cache_state, start = entry, end

        logits = self.model(mx.array([tokens[start:]]), cache=self._seeded_cache(cache_state))
        candidates = logits[0, -1, :][mx.array(candidate_ids)]
        mx.eval(candidates)
        return [float(x) for x in candidates.tolist()]

    def _token_boundaries(self, prompt_text: str, tokens: list[int], prefix_ends: Sequence[int]) -> list[int]:
        """Token counts at which to split `tokens`, from character offsets.

        A character offset only becomes a split point if tokenizing the text
        before it gives exactly the leading tokens of the whole prompt - if
        a token straddles the offset, splitting there would feed the model
        different tokens than the unsplit prompt, so that offset is skipped.
        """
        if not self._reuse_prefixes:
            return []
        boundaries: list[int] = []
        for pos in prefix_ends:
            prefix = self.tokenizer.encode(prompt_text[:pos], add_special_tokens=False)
            n = len(prefix)
            if (not boundaries or n > boundaries[-1]) and 0 < n < len(tokens) and tokens[:n] == prefix:
                boundaries.append(n)
        return boundaries

    def _seeded_cache(self, state: list | None) -> list:
        """A fresh per-layer KV cache holding `state` (a stored prefix).
        Fresh every time, so running more tokens through it never writes
        into a stored entry."""
        cache = self._make_prompt_cache(self.model)
        if state is not None:
            for layer_cache, layer_state in zip(cache, state):
                layer_cache.state = layer_state
        return cache
