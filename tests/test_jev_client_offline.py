"""Fast tests for jev_client.system_one's retry behavior - no real network
call. Stubs urllib.request.urlopen so these run in milliseconds."""

import io
import urllib.error

import pytest

from supervisor import jev_client


class _FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_succeeds_without_retry_on_first_try(monkeypatch):
    calls = []

    def fake_urlopen(request, timeout):
        calls.append(request)
        return _FakeResponse(b'{"answers": {}}')

    monkeypatch.setattr(jev_client.urllib.request, "urlopen", fake_urlopen)
    result = jev_client.system_one("state", {}, api_key="k")

    assert result == {"answers": {}}
    assert len(calls) == 1


def test_retries_a_dropped_connection_then_succeeds(monkeypatch):
    attempts = {"n": 0}

    def fake_urlopen(request, timeout):
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise urllib.error.URLError("handshake timed out")
        return _FakeResponse(b'{"answers": {}}')

    monkeypatch.setattr(jev_client.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(jev_client.time, "sleep", lambda _: None)
    result = jev_client.system_one("state", {}, api_key="k")

    assert result == {"answers": {}}
    assert attempts["n"] == 3


def test_gives_up_after_max_attempts(monkeypatch):
    attempts = {"n": 0}

    def fake_urlopen(request, timeout):
        attempts["n"] += 1
        raise urllib.error.URLError("still down")

    monkeypatch.setattr(jev_client.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(jev_client.time, "sleep", lambda _: None)

    with pytest.raises(jev_client.TypesafeAPIError):
        jev_client.system_one("state", {}, api_key="k")
    assert attempts["n"] == jev_client._MAX_ATTEMPTS


def test_does_not_retry_a_4xx(monkeypatch):
    attempts = {"n": 0}

    def fake_urlopen(request, timeout):
        attempts["n"] += 1
        raise urllib.error.HTTPError(request.full_url, 401, "unauthorized", {}, io.BytesIO(b"bad key"))

    monkeypatch.setattr(jev_client.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(jev_client.time, "sleep", lambda _: None)

    with pytest.raises(jev_client.TypesafeAPIError):
        jev_client.system_one("state", {}, api_key="k")
    assert attempts["n"] == 1


def test_retries_a_503(monkeypatch):
    attempts = {"n": 0}

    def fake_urlopen(request, timeout):
        attempts["n"] += 1
        if attempts["n"] < 2:
            raise urllib.error.HTTPError(request.full_url, 503, "unavailable", {}, io.BytesIO(b"busy"))
        return _FakeResponse(b'{"answers": {}}')

    monkeypatch.setattr(jev_client.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(jev_client.time, "sleep", lambda _: None)
    result = jev_client.system_one("state", {}, api_key="k")

    assert result == {"answers": {}}
    assert attempts["n"] == 2
