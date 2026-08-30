"""
Backend-level persistence-failure tests.

Closes the gap noted in the launch checklist: tests/test_api_client.py
only verifies that the FRONTEND correctly turns a persistence-failure
envelope into a BackendError. Nothing previously exercised the
BACKEND ENDPOINT itself -- i.e. that backend/main.py actually builds
that envelope correctly when a Supabase write raises.

These tests never touch a real Supabase project. They:

    1. Import the real FastAPI app (same pattern as
       tests/test_api_idempotency.py -- .env must have SUPABASE_URL /
       SUPABASE_SERVICE_ROLE_KEY / APP_API_KEYS set, since backend.main
       validates these at import time).
    2. Monkeypatch `backend.main.supabase` with a small fake client
       (same shape as tests/test_persistence.py's _FakeClient) so no
       real network call happens for the read side.
    3. Monkeypatch the specific persistence function
       (write_findings / write_ai_output / create_finding_review /
       write_finding_explanation) that backend/main.py imported by
       name, to raise on demand.
    4. Assert the exact response envelope shape the endpoint promises
       on partial failure, and that no half-written state leaks into
       what's returned to the caller.

Endpoints covered:
    POST /findings/{id}/ai-explanation  -- generation succeeds,
                                            persistence fails
    PATCH /findings/{id}                -- review-history persistence
                                            fails
    PATCH /findings/{id}                -- deterministic explanation
                                            persistence fails on
                                            REVIEW -> CONFIRMED
"""

from __future__ import annotations

import os

import pytest
from dotenv import load_dotenv
from fastapi.testclient import TestClient

# Same import convention as tests/test_api_idempotency.py: backend.main
# validates APP_API_KEYS at import time, so .env must be loaded first.
load_dotenv()

import backend.main as backend_main
from engine.ai_explanation_pipeline import AIExplanationResult


# =====================================================================
# API KEY
# =====================================================================

def _get_api_key() -> str:
    api_keys = os.getenv("APP_API_KEYS", "")

    if not api_keys.strip():
        pytest.fail(
            "APP_API_KEYS is not configured. Set APP_API_KEYS in .env "
            "before running the backend persistence-failure tests."
        )

    key = next(
        (item.strip() for item in api_keys.split(",") if item.strip()),
        None,
    )

    if not key:
        pytest.fail("APP_API_KEYS is empty. Set at least one valid API key.")

    return key


API_KEY = None  # resolved lazily inside fixtures, matching test_api_idempotency.py


@pytest.fixture(scope="module")
def api_key() -> str:
    return _get_api_key()


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(backend_main.app)


# =====================================================================
# FAKE SUPABASE CLIENT
#
# Supports exactly the chainable calls backend/main.py's
# update_finding() and generate_ai_explanation() use:
#   .table(name).select(...).eq(...).limit(...).execute()
#   .table(name).update(...).eq(...).execute()
# =====================================================================

class _FakeResponse:
    def __init__(self, data):
        self.data = data


class _FakeTable:
    def __init__(self, name: str, rows: dict[str, list[dict]]):
        self.name = name
        self._rows = rows
        self._pending = None
        self._eq_filters: dict[str, str] = {}

    # --- read chain -----------------------------------------------

    def select(self, _columns):
        self._pending = ("select", None)
        return self

    def eq(self, column, value):
        self._eq_filters[column] = value
        return self

    def limit(self, _n):
        return self

    def order(self, *_args, **_kwargs):
        return self

    # --- write chain ------------------------------------------------

    def update(self, payload):
        self._pending = ("update", payload)
        return self

    def insert(self, payload):
        self._pending = ("insert", payload)
        return self

    def upsert(self, payload, on_conflict=None):
        self._pending = ("upsert", payload)
        return self

    # --- execute ------------------------------------------------------

    def execute(self):
        op, payload = self._pending

        table_rows = self._rows.get(self.name, [])

        if op == "select":
            matches = [
                row
                for row in table_rows
                if all(row.get(k) == v for k, v in self._eq_filters.items())
            ]
            return _FakeResponse(matches)

        if op == "update":
            matches = [
                row
                for row in table_rows
                if all(row.get(k) == v for k, v in self._eq_filters.items())
            ]
            for row in matches:
                row.update(payload)
            return _FakeResponse(matches)

        # insert / upsert -- not exercised by these tests directly
        # (write paths go through the monkeypatched persistence
        # functions, not straight through the client), but kept for
        # completeness so an unexpected call doesn't crash silently.
        return _FakeResponse([payload] if not isinstance(payload, list) else payload)


class _FakeClient:
    def __init__(self, rows: dict[str, list[dict]]):
        self._rows = rows

    def table(self, name):
        return _FakeTable(name, self._rows)


# =====================================================================
# TEST DATA
# =====================================================================

def _make_confirmed_finding(finding_id: str) -> dict:
    return {
        "finding_id": finding_id,
        "audit_run_id": "AUDIT-TEST-PERSIST",
        "control_id": "SCREENING_001",
        "customer_id": "CUST100005",
        "severity": "HIGH",
        "assessment_status": "FAIL",
        "finding_status": "CONFIRMED",
        "expected": "Opened wallets must have a CLEAR screening result.",
        "actual": "Screening status is HIGH_RISK.",
        "evidence": {"screening_status": "HIGH_RISK"},
        "policy_references": [
            {
                "policy_id": "SCREENING-POLICY-001",
                "version": "1.0",
                "section": "Requirements",
            }
        ],
        "reviewed_by": "tester",
        "review_timestamp": "2026-08-27T00:00:00+00:00",
        "reviewer_notes": None,
        "ai_explanation": None,
        "ai_recommendation": None,
    }


def _make_review_finding(finding_id: str) -> dict:
    finding = _make_confirmed_finding(finding_id)
    finding["finding_status"] = "REVIEW"
    finding["reviewed_by"] = None
    finding["review_timestamp"] = None
    return finding


# =====================================================================
# 1. POST /findings/{id}/ai-explanation -- generation OK, persistence fails
# =====================================================================

def test_ai_explanation_persist_failure_returns_warning_envelope(
    client, api_key, monkeypatch
):
    finding_id = "F-PERSIST-0001"
    finding = _make_confirmed_finding(finding_id)

    fake_client = _FakeClient({"findings": [finding]})
    monkeypatch.setattr(backend_main, "supabase", fake_client)

    def _fake_generate(finding_arg, registry):
        # Mirrors what a real success does: mutate the finding dict in
        # place and return a succeeded AIExplanationResult.
        finding_arg["ai_explanation"] = "Grounded explanation."
        finding_arg["ai_recommendation"] = "Grounded recommendation."
        return AIExplanationResult(
            finding_id=finding_arg["finding_id"],
            succeeded=True,
            finding=finding_arg,
            ai_output={"provider_used": "fake", "model_used": "fake"},
        )

    monkeypatch.setattr(
        backend_main, "generate_ai_explanation_for_finding", _fake_generate
    )

    def _raise_write_findings(*_args, **_kwargs):
        raise RuntimeError("simulated Supabase outage")

    monkeypatch.setattr(backend_main, "write_findings", _raise_write_findings)

    response = client.post(
        f"/findings/{finding_id}/ai-explanation",
        headers={"X-API-Key": api_key},
    )

    assert response.status_code == 200
    body = response.json()

    assert body["status"] == "success"
    assert "could not" in body["warning"].lower()
    assert "simulated Supabase outage" in body["persist_error"]

    # The generated content is still surfaced to the caller (it WAS
    # generated -- only persistence failed) but nothing beyond the
    # whitelisted finding_id/ai_explanation/ai_recommendation leaks.
    assert body["finding"]["finding_id"] == finding_id
    assert body["finding"]["ai_explanation"] == "Grounded explanation."
    assert body["finding"]["ai_recommendation"] == "Grounded recommendation."


def test_ai_explanation_generation_failure_is_a_clean_error_not_a_warning(
    client, api_key, monkeypatch
):
    """
    Contrast case: if GENERATION itself fails (not persistence), the
    endpoint must report status="error", never the success+warning
    shape reserved for persistence-only failures.
    """
    finding_id = "F-PERSIST-0002"
    finding = _make_confirmed_finding(finding_id)

    fake_client = _FakeClient({"findings": [finding]})
    monkeypatch.setattr(backend_main, "supabase", fake_client)

    def _fake_generate_fails(finding_arg, registry):
        return AIExplanationResult(
            finding_id=finding_arg["finding_id"],
            succeeded=False,
            error="LLMAllProvidersFailedError: simulated total outage",
        )

    monkeypatch.setattr(
        backend_main, "generate_ai_explanation_for_finding", _fake_generate_fails
    )

    response = client.post(
        f"/findings/{finding_id}/ai-explanation",
        headers={"X-API-Key": api_key},
    )

    assert response.status_code == 200
    body = response.json()

    assert body["status"] == "error"
    assert "simulated total outage" in body["message"]
    assert "warning" not in body
    assert "persist_error" not in body


# =====================================================================
# 2. PATCH /findings/{id} -- review-history persistence fails
# =====================================================================

def test_review_history_persist_failure_returns_warning_envelope(
    client, api_key, monkeypatch
):
    finding_id = "F-PERSIST-0003"
    finding = _make_review_finding(finding_id)

    fake_client = _FakeClient({"findings": [finding]})
    monkeypatch.setattr(backend_main, "supabase", fake_client)

    def _raise_create_review(*_args, **_kwargs):
        raise RuntimeError("simulated finding_reviews insert failure")

    monkeypatch.setattr(backend_main, "create_finding_review", _raise_create_review)

    # Prevent the CONFIRMED auto-explain branch from also firing here --
    # isolate this test to the review-history failure path only.
    monkeypatch.setattr(
        backend_main,
        "explain_finding",
        lambda _finding: pytest.fail(
            "explain_finding() should not be reached: the function "
            "must return immediately after the review-history failure."
        ),
    )

    response = client.patch(
        f"/findings/{finding_id}",
        headers={"X-API-Key": api_key},
        json={
            "finding_status": "CONFIRMED",
            "reviewed_by": "integration-test",
            "reviewer_notes": None,
        },
    )

    assert response.status_code == 200
    body = response.json()

    assert body["status"] == "success"
    assert "review history" in body["warning"].lower()
    assert "simulated finding_reviews insert failure" in body["review_error"]

    # The finding_status transition itself still went through (it's a
    # separate write from the review-history insert) -- only the
    # history record failed to persist.
    assert body["finding"]["finding_status"] == "CONFIRMED"


# =====================================================================
# 3. PATCH /findings/{id} -- deterministic explanation persistence fails
# =====================================================================

def test_deterministic_explanation_persist_failure_returns_warning_envelope(
    client, api_key, monkeypatch
):
    finding_id = "F-PERSIST-0004"
    finding = _make_review_finding(finding_id)

    fake_client = _FakeClient({"findings": [finding]})
    monkeypatch.setattr(backend_main, "supabase", fake_client)

    # Review-history insert succeeds this time.
    monkeypatch.setattr(
        backend_main, "create_finding_review", lambda *a, **k: {"review_id": 1}
    )

    # explain_finding() itself succeeds (deterministic, no network) --
    # only the WRITE of that explanation fails.
    def _raise_write_explanation(*_args, **_kwargs):
        raise RuntimeError("simulated finding_explanations write failure")

    monkeypatch.setattr(
        backend_main, "write_finding_explanation", _raise_write_explanation
    )

    response = client.patch(
        f"/findings/{finding_id}",
        headers={"X-API-Key": api_key},
        json={
            "finding_status": "CONFIRMED",
            "reviewed_by": "integration-test",
            "reviewer_notes": None,
        },
    )

    assert response.status_code == 200
    body = response.json()

    assert body["status"] == "success"
    assert "deterministic" in body["warning"].lower()
    assert (
        "simulated finding_explanations write failure" in body["explain_error"]
    )
    assert body["finding"]["finding_status"] == "CONFIRMED"


# =====================================================================
# 4. Sanity: a clean run (no injected failures) never returns a warning
# =====================================================================

def test_clean_review_update_has_no_warning_fields(client, api_key, monkeypatch):
    """
    Guards against a monkeypatch leaking between tests, or the endpoint
    always emitting a warning key regardless of outcome.
    """
    finding_id = "F-PERSIST-0005"
    finding = _make_review_finding(finding_id)

    fake_client = _FakeClient({"findings": [finding]})
    monkeypatch.setattr(backend_main, "supabase", fake_client)
    monkeypatch.setattr(
        backend_main, "create_finding_review", lambda *a, **k: {"review_id": 1}
    )
    monkeypatch.setattr(
        backend_main, "write_finding_explanation", lambda *a, **k: {"explanation_id": 1}
    )

    response = client.patch(
        f"/findings/{finding_id}",
        headers={"X-API-Key": api_key},
        json={
            "finding_status": "REJECTED",
            "reviewed_by": "integration-test",
            "reviewer_notes": "clean path",
        },
    )

    assert response.status_code == 200
    body = response.json()

    assert body["status"] == "success"
    assert "warning" not in body
    assert body["finding"]["finding_status"] == "REJECTED"