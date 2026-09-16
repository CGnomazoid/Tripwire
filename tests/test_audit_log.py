"""Fast, pure tests for the audit log - no model load needed."""

import json

from supervisor import audit_log


def test_log_call_writes_a_jsonl_record(tmp_path, monkeypatch):
    log_path = tmp_path / "audit_log.jsonl"
    monkeypatch.setattr(audit_log, "LOG_PATH", log_path)

    audit_log.log_call(
        tool="should_block",
        state="Tool call: rm -rf /",
        reason="testing",
        result={"decision": "block", "confidence": 0.99},
    )

    lines = log_path.read_text().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["tool"] == "should_block"
    assert record["state"] == "Tool call: rm -rf /"
    assert record["reason"] == "testing"
    assert record["result"] == {"decision": "block", "confidence": 0.99}
    assert "timestamp" in record


def test_log_call_appends_rather_than_overwrites(tmp_path, monkeypatch):
    log_path = tmp_path / "audit_log.jsonl"
    monkeypatch.setattr(audit_log, "LOG_PATH", log_path)

    audit_log.log_call(tool="a", state="s1", reason=None, result={})
    audit_log.log_call(tool="b", state="s2", reason=None, result={})

    lines = log_path.read_text().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["state"] == "s1"
    assert json.loads(lines[1])["state"] == "s2"


def test_log_call_reason_none_serializes_as_null_not_dropped(tmp_path, monkeypatch):
    log_path = tmp_path / "audit_log.jsonl"
    monkeypatch.setattr(audit_log, "LOG_PATH", log_path)

    audit_log.log_call(tool="a", state="s", reason=None, result={})

    record = json.loads(log_path.read_text().splitlines()[0])
    assert "reason" in record
    assert record["reason"] is None


def test_log_call_creates_parent_dir_if_missing(tmp_path, monkeypatch):
    log_path = tmp_path / "nested" / "audit_log.jsonl"
    monkeypatch.setattr(audit_log, "LOG_PATH", log_path)

    audit_log.log_call(tool="a", state="s", reason=None, result={})

    assert log_path.exists()
