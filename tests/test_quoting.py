"""Fast, pure tests for quoting.quote_mask - the quote-awareness that both
comment stripping and command splitting rest on."""

import time

from supervisor.quoting import quote_mask


def _quoted(text: str) -> str:
    """The characters quote_mask marks as quoted, with the rest blanked."""
    return "".join(ch if inside else "_" for ch, inside in zip(text, quote_mask(text)))


def test_marks_a_quoted_span_including_its_delimiters():
    assert _quoted("a 'b c' d") == "__'b c'__"


def test_single_and_double_quotes_each_hide_the_other():
    assert _quoted("""x "it's" y 'say "hi"' z""") == """__"it's"___'say "hi"'__"""


def test_backslash_escapes_a_quote_inside_a_span():
    assert _quoted(r"'it\'s' x") == r"'it\'s'__"


def test_unbalanced_apostrophe_is_an_ordinary_character():
    # prose, not a truncated string: the rest of the line stays unquoted
    assert _quoted("don't # comment") == "_______________"


def test_recovers_at_the_unbalanced_quote_and_still_finds_later_spans():
    assert _quoted('don\'t say "x # y"') == '__________"x # y"'


def test_an_apostrophe_can_still_close_at_a_later_quote():
    # recovery only applies to a quote that is never closed - this one is
    assert _quoted("don't 'x' ok") == "___'t '_____"


def test_empty_string():
    assert quote_mask("") == []


def test_input_full_of_unbalanced_quotes_is_linear_not_quadratic():
    # Every quote here is escaped or unbalanced, which made the old
    # rescan-on-failure implementation take ~13s at this size (quadratic:
    # ~0.8s at a quarter of it). Linear is a few milliseconds.
    text = "'" + "\\'" * 20_000
    start = time.perf_counter()
    mask = quote_mask(text)
    assert time.perf_counter() - start < 1.0
    assert not any(mask)
