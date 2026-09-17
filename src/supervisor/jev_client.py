"""Minimal HTTP client for Typesafe's hosted Jev model (docs.typesafe.ai).

Hand-rolled with stdlib `urllib` rather than Typesafe's own SDK: this only
ever needs one endpoint (POST /v1/systemone) for benchmarking Jev against
the local Judge, so pulling in a whole client library bought nothing.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

from supervisor.paths import REPO_ROOT

DEFAULT_BASE_URL = "https://api.typesafe.ai"
DEFAULT_MODEL = "jev-latest"

ENV_PATH = REPO_ROOT / ".env"


def _load_dotenv() -> None:
    """Populate os.environ from a gitignored `.env` at the repo root, for
    TYPESAFE_API_KEY and friends - so a key set once doesn't need exporting
    in every new shell. Never overrides a variable the shell already set,
    and does nothing if `.env` doesn't exist.

    Hand-rolled instead of pulling in python-dotenv: the format needed here
    is `KEY=VALUE` lines plus comments, which is a few lines either way.
    """
    if not ENV_PATH.is_file():
        return
    for line in ENV_PATH.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'\"")
        os.environ.setdefault(key, value)


_load_dotenv()

# Retries apply only to errors a retry can plausibly fix: a dropped
# connection/TLS handshake (URLError) or the server's own "back off" status
# codes. A 4xx (bad request, bad auth, ...) is retried zero times - sending
# the same broken request again just wastes the attempt.
_MAX_ATTEMPTS = 4
_RETRY_STATUS = {429, 500, 502, 503, 504}
_BACKOFF_BASE_S = 0.5


class TypesafeAPIError(RuntimeError):
    """Raised on a non-2xx response from the Typesafe API."""


def system_one(
    state: str,
    questions: dict[str, dict],
    *,
    api_key: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    timeout: float = 30.0,
) -> dict:
    """POST one /v1/systemone request and return the parsed JSON response.

    `api_key` falls back to the TYPESAFE_API_KEY env var, `model` to
    TYPESAFE_DEFAULT_MODEL then DEFAULT_MODEL, `base_url` to
    TYPESAFE_BASE_URL then DEFAULT_BASE_URL - same env vars Typesafe's own
    SDK documents, so a shell that's already set up for it needs no
    changes.

    Retries a transient network failure or a 429/5xx up to _MAX_ATTEMPTS
    times with linear backoff - a benchmark run makes hundreds of these
    calls back to back, and losing the whole run to one dropped connection
    partway through wastes every call already made.
    """
    key = api_key or os.environ.get("TYPESAFE_API_KEY")
    if not key:
        raise RuntimeError(
            "no Typesafe API key: set the TYPESAFE_API_KEY env var or pass api_key="
        )
    url = (base_url or os.environ.get("TYPESAFE_BASE_URL") or DEFAULT_BASE_URL).rstrip("/") + "/v1/systemone"
    body = json.dumps({
        "state": state,
        "model": model or os.environ.get("TYPESAFE_DEFAULT_MODEL") or DEFAULT_MODEL,
        "questions": questions,
    }).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )

    last_error: Exception | None = None
    for attempt in range(_MAX_ATTEMPTS):
        if attempt > 0:
            time.sleep(_BACKOFF_BASE_S * attempt)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")
            last_error = TypesafeAPIError(f"Typesafe API returned {e.code}: {detail}")
            if e.code not in _RETRY_STATUS:
                raise last_error from e
        except urllib.error.URLError as e:
            last_error = TypesafeAPIError(f"Typesafe API request failed: {e}")
    raise last_error  # last_error is always set: the loop only exits without returning after >=1 failure
