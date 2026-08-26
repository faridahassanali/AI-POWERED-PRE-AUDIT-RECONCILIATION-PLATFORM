"""
API Client — Streamlit -> FastAPI Backend.

Thin HTTP layer used exclusively by the Streamlit frontend (app.py)
to talk to backend/main.py. Nothing here touches business logic --
that stays in engine/. This module only knows how to make requests,
attach the API key, and unwrap the backend's {"status": ...} envelope.

Configuration
-------------
APP_API_BASE : base URL of the FastAPI backend (default http://127.0.0.1:8000)
APP_API_KEYS : same comma-separated list you set for the backend
               (APP_API_KEYS=key-one,key-two) -- the first key is used
               to authenticate outgoing requests.

Both must be set in the SAME terminal session that runs
`streamlit run frontend/app.py`, since environment variables don't
carry over between separate terminal windows.
"""

from __future__ import annotations

import os

import requests

API_BASE = os.environ.get("APP_API_BASE", "http://127.0.0.1:8000")

_raw_keys = os.environ.get("APP_API_KEYS", "")
API_KEY = _raw_keys.split(",")[0].strip() if _raw_keys else ""

HEADERS = {"X-API-Key": API_KEY}

DEFAULT_TIMEOUT = 15
AUDIT_TIMEOUT = 120  # running the full pipeline / calling the LLM can take longer


class BackendError(RuntimeError):
    """Raised when the backend responds but reports status != success."""


# Fields safe to merge back into the in-memory finding dict after a
# PATCH /findings/{id} call. Supabase's row also carries columns like
# created_at/updated_at that the AI input schema (engine.ai_input)
# doesn't expect -- merging the raw response wholesale breaks
# generate_ai_explanation_for_finding() with an
# "Additional properties are not allowed" error. Whitelisting keeps
# the finding dict's shape identical to what it was before Confirm/
# Reject, just with these fields updated.
REVIEW_MERGE_FIELDS = {
    "finding_status",
    "reviewed_by",
    "reviewer_notes",
    "review_timestamp",
}


def _unwrap(response: requests.Response) -> dict:
    """
    Raise on transport/HTTP errors, then unwrap the backend's
    {"status": "success"|"error"|"not_found", ...} envelope.
    """

    response.raise_for_status()
    data = response.json()

    status = data.get("status")

    if status == "not_found":
        raise BackendError(data.get("message", "Not found."))

    if status == "error":
        raise BackendError(data.get("message", "Unknown backend error."))

    return data


def run_audit() -> dict:
    """
    Trigger POST /audit-runs/execute.

    Runs the deterministic audit pipeline on the backend and persists
    the audit run + findings to Supabase. Returns the execution
    summary (audit_run_id, counts, report) -- call get_findings()
    afterwards to fetch the actual finding records.
    """

    response = requests.post(
        f"{API_BASE}/audit-runs/execute",
        headers=HEADERS,
        timeout=AUDIT_TIMEOUT,
    )

    return _unwrap(response)


def get_findings(**filters) -> list[dict]:
    """
    Fetch findings via GET /findings.

    Optional filters: status, severity, control_id, audit_run_id
    (passed straight through as query params, matching the backend's
    signature). Returns the list of finding dicts.
    """

    params = {key: value for key, value in filters.items() if value is not None}

    response = requests.get(
        f"{API_BASE}/findings",
        headers=HEADERS,
        params=params,
        timeout=DEFAULT_TIMEOUT,
    )

    data = _unwrap(response)
    return data["findings"]


def get_finding_policy(finding_id: str) -> dict:
    """Fetch policy context for one finding via GET /findings/{id}/policy."""

    response = requests.get(
        f"{API_BASE}/findings/{finding_id}/policy",
        headers=HEADERS,
        timeout=DEFAULT_TIMEOUT,
    )

    return _unwrap(response)


def update_finding(finding_id: str, **fields) -> dict:
    """
    Update a finding via PATCH /findings/{id}.

    Accepts any of: finding_status, reviewed_by, reviewer_notes,
    ai_explanation, ai_recommendation (matching FindingUpdate on the
    backend). Only pass the fields you actually want to change.
    Returns the updated finding record as stored in Supabase.

    Note: as of backend/main.py's update_finding(), confirming a
    finding (finding_status="CONFIRMED") also triggers the
    deterministic Stage 2 explanation automatically on the backend --
    the frontend no longer needs to call explain_finding() itself.
    """

    payload = {key: value for key, value in fields.items() if value is not None}

    response = requests.patch(
        f"{API_BASE}/findings/{finding_id}",
        json=payload,
        headers=HEADERS,
        timeout=DEFAULT_TIMEOUT,
    )

    data = _unwrap(response)
    return data["finding"]


def get_finding_reviews(finding_id: str) -> list[dict]:
    """Fetch review history via GET /findings/{id}/reviews."""

    response = requests.get(
        f"{API_BASE}/findings/{finding_id}/reviews",
        headers=HEADERS,
        timeout=DEFAULT_TIMEOUT,
    )

    data = _unwrap(response)
    return data["reviews"]


def generate_ai_explanation(finding_id: str) -> dict:
    """
    Trigger POST /findings/{id}/ai-explanation.

    Runs Stage 3 (RAG retrieval + LLM call + output validation) on
    the backend and persists the result. Uses AUDIT_TIMEOUT instead
    of DEFAULT_TIMEOUT since the LLM call can take a while.

    Returns a dict with at least: finding_id, ai_explanation,
    ai_recommendation.
    """

    response = requests.post(
        f"{API_BASE}/findings/{finding_id}/ai-explanation",
        headers=HEADERS,
        timeout=AUDIT_TIMEOUT,
    )

    data = _unwrap(response)
    return data["finding"]


def get_evaluation(audit_run_id: str) -> dict | None:
    """
    Fetch evaluation metrics via GET /audit-runs/{id}/evaluation.

    Returns None if no evaluation has been persisted yet for this
    run (backend responds with status "not_found") instead of
    raising -- callers can treat None exactly like the old
    evaluation=None placeholder state used before this endpoint
    existed.
    """

    if not audit_run_id:
        return None

    response = requests.get(
        f"{API_BASE}/audit-runs/{audit_run_id}/evaluation",
        headers=HEADERS,
        timeout=DEFAULT_TIMEOUT,
    )

    response.raise_for_status()
    data = response.json()

    if data.get("status") == "not_found":
        return None

    if data.get("status") == "error":
        raise BackendError(data.get("message", "Unknown backend error."))

    return data["evaluation"]