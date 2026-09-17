"""Shared quote-awareness for the two places that scan free-form command
text: sanitize.strip_comments and multiline.split_commands.

Both need the same question answered - "is this character inside a quoted
string?" - so that a `#` in a channel name or a `;` in an echo argument
isn't mistaken for a comment marker or a command separator. Having one
implementation means quote handling (escapes, recovery from unbalanced
quotes) is reasoned about and tested once instead of drifting between two
copies.
"""

from __future__ import annotations

_QUOTES = ("'", '"')
ESCAPE = "\\"


def quote_mask(text: str) -> list[bool]:
    """Per-character flags: `mask[i]` is True when `text[i]` sits inside a
    quoted span (the delimiting quote characters count as inside).

    A backslash escapes the following character inside a quoted span, so
    `'it\\'s'` stays one span.

    Unbalanced quotes are recovered from rather than swallowing the rest of
    the line: an opening quote that is never closed is treated as an
    ordinary character, and scanning carries on right after it. This
    matters because the common case of an unbalanced quote isn't a
    truncated string, it's an apostrophe in ordinary prose ("don't",
    "user's") - treating everything after it as quoted would hide a
    trailing `# comment` from the sanitizer, and treating the whole line as
    unquoted would let a `#` inside a genuine quoted span be stripped.
    Recovering at the offending quote gets both cases right.

    Where each quoted span would end is precomputed (see _closing_quotes),
    so deciding whether an opening quote is ever closed is a lookup rather
    than a rescan to the end of the text. Rescanning made this quadratic: a
    few kilobytes of `'\\'\\'\\'...` - every quote unbalanced - took seconds.
    """
    n = len(text)
    closers = {q: _closing_quotes(text, q) for q in _QUOTES}
    mask = [False] * n
    i = 0
    while i < n:
        end = closers[text[i]][i + 1] if text[i] in closers else None
        if end is None:
            # not a quote, or an opening quote that is never closed
            i += 1
            continue
        mask[i : end + 1] = [True] * (end + 1 - i)
        i = end + 1
    return mask


def _closing_quotes(text: str, quote: str) -> list[int | None]:
    """`result[i]` is the index of the `quote` that ends a span whose scan
    has reached position i (skipping backslash-escaped characters), or None
    if the span runs off the end of `text` unclosed.

    Filled right to left so each entry reuses one already computed: O(n)
    for the whole table. Two trailing None entries stand in for "past the
    end", so neither `i + 1` nor an escape's `i + 2` needs a bounds check.
    """
    n = len(text)
    result: list[int | None] = [None] * (n + 2)
    for i in range(n - 1, -1, -1):
        if text[i] == ESCAPE and i + 1 < n:
            result[i] = result[i + 2]
        elif text[i] == quote:
            result[i] = i
        else:
            result[i] = result[i + 1]
    return result
