"""
API Key Authentication Layer.

Provides a FastAPI dependency (verify_api_key) that protects
sensitive endpoints, plus a startup check (require_api_key_configured)
that fails loudly if no keys are configured instead of silently
running the API wide open.

Configuration
-------------
APP_API_KEYS: comma-separated list of valid API keys, e.g.
    APP_API_KEYS=key-one,key-two

Callers must send the key in the `X-API-Key` header.

Design notes
------------
- Keys are compared using hmac.compare_digest (constant-time) to
  avoid leaking timing information about how much of the key
  matched, which is the standard defense against timing side-channel
  attacks on secret comparison.
- require_api_key_configured() is meant to be called once at import/
  startup time (see backend/main.py). It raises RuntimeError rather
  than allowing the app to start with authentication effectively
  disabled.
"""

from __future__ import annotations

import hmac
import os

from fastapi import Header, HTTPException, status


def _load_valid_keys() -> set[str]:
    raw = os.environ.get("APP_API_KEYS", "")
    return {key.strip() for key in raw.split(",") if key.strip()}


def require_api_key_configured() -> None:
    """
    Fail loudly at startup if APP_API_KEYS is missing or empty.

    This is intentionally NOT a silent fallback -- an API meant to be
    protected should refuse to start rather than run unauthenticated
    because an environment variable was forgotten.
    """

    if not _load_valid_keys():
        raise RuntimeError(
            "APP_API_KEYS is not set (or empty). Set it to a "
            "comma-separated list of valid API keys before starting "
            "the backend, e.g. APP_API_KEYS=key-one,key-two."
        )


def _is_valid_key(candidate: str) -> bool:
    """
    Constant-time membership check against the configured keys.

    A simple `candidate in valid_keys` uses Python's normal string
    equality, which can short-circuit on the first differing
    character -- in principle leaking timing information about how
    much of a guessed key matched. Comparing against every configured
    key with hmac.compare_digest avoids that.
    """

    valid_keys = _load_valid_keys()
    match = False

    for key in valid_keys:
        if hmac.compare_digest(candidate, key):
            match = True

    return match


def verify_api_key(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> str:
    """
    FastAPI dependency: require a valid X-API-Key header.

    Use as: `_: str = Depends(verify_api_key)` on any endpoint that
    should not be publicly callable.

    Raises HTTPException(401) if the header is missing or invalid.
    """

    if not x_api_key or not _is_valid_key(x_api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid API key.",
        )

    return x_api_key