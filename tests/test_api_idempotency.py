# tests/test_api_idempotency.py

# tests/test_api_idempotency.py

import os
import uuid

import pytest
from dotenv import load_dotenv
from fastapi.testclient import TestClient

# Load .env BEFORE importing backend.main.
# backend.main validates APP_API_KEYS at import time.
load_dotenv()

from backend.main import app
from backend.database import supabase



client = TestClient(app)


# =========================================================
# HELPERS
# =========================================================

def _get_api_key() -> str:
    """
    Return the first configured API key.

    This test intentionally uses the real authentication
    dependency instead of overriding or mocking it.
    """

    api_keys = os.getenv("APP_API_KEYS", "")

    if not api_keys.strip():
        pytest.fail(
            "APP_API_KEYS is not configured. "
            "Set APP_API_KEYS in .env before running "
            "the API-level idempotency test."
        )

    key = next(
        (
            item.strip()
            for item in api_keys.split(",")
            if item.strip()
        ),
        None,
    )

    if not key:
        pytest.fail(
            "APP_API_KEYS is empty. "
            "Set at least one valid API key."
        )

    return key


# =========================================================
# API-LEVEL IDEMPOTENCY TEST
# =========================================================

def test_execute_audit_idempotency_at_api_level():
    """
    Verify idempotency through the real HTTP API.

    First request:
        -> audit executes
        -> status = success
        -> audit_run_id is returned

    Second request with the same Idempotency-Key:
        -> audit must NOT execute again
        -> status = duplicate
        -> same audit_run_id is returned

    Finally:
        -> database must contain exactly one audit_run
           for this Idempotency-Key.
    """

    api_key = _get_api_key()

    # -----------------------------------------------------
    # 1. Unique idempotency key
    # -----------------------------------------------------

    idempotency_key = (
        f"pytest-api-idempotency-{uuid.uuid4().hex}"
    )

    headers = {
        "X-API-Key": api_key,
        "Idempotency-Key": idempotency_key,
    }

    # -----------------------------------------------------
    # 2. FIRST REQUEST
    # -----------------------------------------------------

    first_response = client.post(
        "/audit-runs/execute",
        headers=headers,
    )

    assert first_response.status_code == 200, (
        "First audit execution failed.\n"
        f"Status: {first_response.status_code}\n"
        f"Response: {first_response.text}"
    )

    first_body = first_response.json()

    assert first_body["status"] == "success", (
        f"Unexpected first response: {first_body}"
    )

    first_audit_run_id = first_body.get("audit_run_id")

    assert first_audit_run_id, (
        "First request did not return audit_run_id."
    )

    # -----------------------------------------------------
    # 3. SECOND REQUEST
    #    SAME Idempotency-Key
    # -----------------------------------------------------

    second_response = client.post(
        "/audit-runs/execute",
        headers=headers,
    )

    assert second_response.status_code == 200, (
        "Second idempotent request failed.\n"
        f"Status: {second_response.status_code}\n"
        f"Response: {second_response.text}"
    )

    second_body = second_response.json()

    # -----------------------------------------------------
    # 4. MUST BE DUPLICATE
    # -----------------------------------------------------

    assert second_body["status"] == "duplicate", (
        "Second request was not treated as duplicate.\n"
        f"Response: {second_body}"
    )

    # -----------------------------------------------------
    # 5. SAME AUDIT RUN ID
    # -----------------------------------------------------

    duplicate_audit_run = second_body.get("audit_run")

    assert duplicate_audit_run is not None, (
        "Duplicate response did not contain audit_run."
    )

    second_audit_run_id = duplicate_audit_run.get(
        "audit_run_id"
    )

    assert second_audit_run_id == first_audit_run_id, (
        "Idempotency violation: second request returned "
        "a different audit_run_id.\n"
        f"First:  {first_audit_run_id}\n"
        f"Second: {second_audit_run_id}"
    )

    # -----------------------------------------------------
    # 6. DATABASE VERIFICATION
    # -----------------------------------------------------

    db_response = (
        supabase
        .table("audit_runs")
        .select(
            "audit_run_id, idempotency_key"
        )
        .eq(
            "idempotency_key",
            idempotency_key,
        )
        .execute()
    )

    assert db_response.data, (
        "No audit_run was persisted for the "
        "Idempotency-Key."
    )

    # Exactly ONE row must exist.
    assert len(db_response.data) == 1, (
        "Idempotency violation: more than one audit_run "
        "was persisted for the same Idempotency-Key.\n"
        f"Rows: {db_response.data}"
    )

    persisted_run = db_response.data[0]

    assert persisted_run["audit_run_id"] == (
        first_audit_run_id
    )

    assert persisted_run["idempotency_key"] == (
        idempotency_key
    )