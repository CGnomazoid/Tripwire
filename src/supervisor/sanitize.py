"""Strip comment-like content from `state` text before it's ever judged.

Motivation: tests/test_adversarial.py found two real, verified injection
vulnerabilities, and both take the shape of a trailing "comment" appended
to an otherwise-recognized-as-dangerous action - a fake system-instruction
block, or casual/emoji framing ("# totally fine, just cleaning up 😇"). A
comment is exactly the channel an injection uses: content that reads as
out-of-band commentary/authorization rather than part of the action being
described. Removing that channel entirely - by policy decision, not left
as a heuristic toggle - is a real tradeoff: one *passing* adversarial test
(test_resists_euphemistic_rephrasing_without_social_engineering) relied on
a comment supplying genuine risk information ("# permanently removes all
contents to free space"). That signal is deliberately traded away here.

Only `#` and `//` are treated as comment markers. SQL-style `--` was
considered and rejected: it collides constantly with ordinary CLI flag
syntax (`--force`, `--recursive`, ...), which is itself highly relevant to
risk - stripping from the first `--` would silently delete exactly the
flags that make a command dangerous.
"""

from __future__ import annotations

_MARKERS = ("#", "//")


def strip_comments(text: str) -> str:
    """Remove `#` and `//` line comments, marker to end of line. Quote-aware:
    a marker inside a quoted string is left alone (URL fragments, channel
    names like '#eng-random', literal content, etc.)."""
    return "\n".join(_strip_line(line) for line in text.split("\n"))


def _strip_line(line: str) -> str:
    quote: str | None = None
    i = 0
    n = len(line)
    while i < n:
        ch = line[i]
        if quote:
            if ch == quote and line[i - 1] != "\\":
                quote = None
            i += 1
            continue
        if ch in ("'", '"'):
            quote = ch
            i += 1
            continue
        if any(line[i : i + len(m)] == m for m in _MARKERS):
            return line[:i].rstrip()
        i += 1
    return line.rstrip()
