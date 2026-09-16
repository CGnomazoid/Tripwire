"""Unit tests for the pure comment-stripping logic in sanitize.py."""

from supervisor.sanitize import strip_comments


def test_strips_hash_comment_to_end_of_line():
    assert strip_comments("rm -rf / # totally fine, just cleaning up") == "rm -rf /"


def test_strips_double_slash_comment_to_end_of_line():
    assert strip_comments("rm -rf / // totally fine") == "rm -rf /"


def test_leaves_hash_inside_single_quotes_alone():
    state = "Tool call: post_message(channel='#eng-random', text='deploying now')"
    assert strip_comments(state) == state


def test_leaves_hash_inside_double_quotes_alone():
    state = 'Tool call: post_message(channel="#eng-random")'
    assert strip_comments(state) == state


def test_leaves_url_fragment_inside_quotes_alone():
    state = "Tool call: http_get(url='https://example.com/page#section')"
    assert strip_comments(state) == state


def test_strips_comment_on_each_line_independently():
    text = "cd /tmp # navigate\nrm -rf / # cleanup"
    assert strip_comments(text) == "cd /tmp\nrm -rf /"


def test_does_not_treat_double_dash_as_a_comment_marker():
    # SQL-style -- is deliberately excluded: it collides with CLI flags
    # like --force, --recursive, which are themselves risk-relevant.
    state = "Tool call: run_shell(cmd='rm --recursive --force /')"
    assert strip_comments(state) == state


def test_no_comment_present_is_unchanged():
    state = "Tool call: read_file(path='./README.md')"
    assert strip_comments(state) == state


def test_empty_string_is_unchanged():
    assert strip_comments("") == ""


def test_fully_commented_line_becomes_empty():
    assert strip_comments("# just a comment") == ""
