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
    the line: if the scan reaches the end still inside a quote, that opening
    quote is reinterpreted as an ordinary character and scanning resumes
    right after it. This matters because the common case of an unbalanced
    quote isn't a truncated string, it's an apostrophe in ordinary prose
    ("don't", "user's") - treating everything after it as quoted would hide
    a trailing `# comment` from the sanitizer, and treating the whole line
    as unquoted would let a `#` inside a genuine quoted span be stripped.
    Recovering at the offending quote gets both cases right.
    """
    mask = [False] * len(text)
    n = len(text)
    i = 0
    quote: str | None = None
    open_at = -1
    while True:
        while i < n:
            ch = text[i]
            if quote is None:
                if ch in _QUOTES:
                    quote, open_at = ch, i
                    mask[i] = True
                i += 1
                continue
            mask[i] = True
            if ch == ESCAPE and i + 1 < n:
                mask[i + 1] = True
                i += 2
                continue
            if ch == quote:
                quote = None
            i += 1
        if quote is None:
            return mask
        # Unterminated: un-quote the opening delimiter and rescan from just
        # past it. Each restart moves the earliest unmatched quote strictly
        # forward, so this terminates.
        for j in range(open_at, n):
            mask[j] = False
        quote, i = None, open_at + 1
