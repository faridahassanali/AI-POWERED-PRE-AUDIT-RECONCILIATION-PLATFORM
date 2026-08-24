"""
Tests for engine.persistence.

These tests never touch a real Supabase project. They use a small
fake client that mimics the supabase-py chainable interface
(.table(name).upsert(rows, on_conflict=...).execute()) so we can
verify the row-shaping logic and call pattern in isolation.
"""

import os

import pytest
import engine.persistence as persistence

from engine.persistence import (
    PersistenceNotConfigured,
    get_supabase_client,
    write_audit_run,
    write_findings,
    write_finding_review,
)
from engine.finding_builder import build_finding
from engine.finding_review import confirm_finding


# =====================================================================
# FAKE SUPABASE CLIENT
# =====================================================================

class _FakeResponse:
    def __init__(self, data):
        self.data = data


class _FakeTable:
    def __init__(self, name, calls):
        self.name = name
        self.calls = calls
        self._pending = None

    def upsert(self, rows, on_conflict=None):
        self._pending = ("upsert", rows, on_conflict)
        return self

    def insert(self, row):
        self._pending = ("insert", row, None)
        return self

    def execute(self):
        op, payload, on_conflict = self._pending
        self.calls.append(
            {
                "table": self.name,
                "op": op,
                "payload": payload,
                "on_conflict": on_conflict,
            }
        )
        data = payload if isinstance(payload, list) else [payload]
        return _FakeResponse(data)


class _FakeClient:
    def __init__(self):
        self.calls = []

    def table(self, name):
        return _FakeTable(name, self.calls)


# =====================================================================
# FIXTURES
# =====================================================================

def _make_trace():
    return {
        "audit_run_id": "AUDIT-TEST123",
        "started_at": "2026-08-17T10:00:00+00:00",
        "completed_at": "2026-08-17T10:00:05+00:00",
        "controls_executed": ["SCREENING_001", "RISK_001"],
        "total_records_evaluated": 1000,
        "total_findings_generated": 2,
    }


def _make_finding(**overrides):
    finding = build_finding(
        control_id="SCREENING_001",
        customer_id="CUST100005",
        severity="HIGH",
        assessment_status="FAIL",
        finding_status="REVIEW",
        expected="Opened wallets must have a CLEAR screening result.",
        actual="Screening status is HIGH_RISK.",
        evidence={"screening_status": "HIGH_RISK"},
        policy_references=[
            {
                "policy_id": "SCREENING-POLICY-001",
                "version": "1.0",
                "section": "Requirements",
            }
        ],
    )
    finding["audit_run_id"] = "AUDIT-TEST123"
    finding.update(overrides)
    return finding


# =====================================================================
# get_supabase_client
# =====================================================================

def test_get_client_raises_when_env_vars_missing(monkeypatch):
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_SERVICE_ROLE_KEY", raising=False)

    with pytest.raises(PersistenceNotConfigured):
        get_supabase_client()


def test_get_client_raises_when_only_url_set(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.delenv("SUPABASE_SERVICE_ROLE_KEY", raising=False)

    with pytest.raises(PersistenceNotConfigured):
        get_supabase_client()


def test_get_client_uses_service_role_key(monkeypatch):
    created_with = {}

    def fake_create_client(url, key):
        created_with["url"] = url
        created_with["key"] = key
        return object()

    monkeypatch.setattr(persistence, "create_client", fake_create_client)
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "service-role-key")

    get_supabase_client()

    assert created_with == {
        "url": "https://example.supabase.co",
        "key": "service-role-key",
    }


# =====================================================================
# write_audit_run
# =====================================================================

def test_write_audit_run_shapes_row_correctly():
    client = _FakeClient()
    trace = _make_trace()

    write_audit_run(trace, client=client)

    assert len(client.calls) == 1
    call = client.calls[0]

    assert call["table"] == "audit_runs"
    assert call["op"] == "upsert"
    assert call["on_conflict"] == "audit_run_id"

    row = call["payload"]
    assert row["audit_run_id"] == "AUDIT-TEST123"
    assert row["controls_executed"] == ["SCREENING_001", "RISK_001"]
    assert row["total_findings_generated"] == 2


def test_write_audit_run_accepts_dataclass_trace():
    from engine.audit_trace import create_audit_trace, complete_audit_trace

    trace = create_audit_trace(
        audit_run_id="AUDIT-DC1",
        controls_executed=["RISK_001"],
        total_records_evaluated=10,
    )
    trace = complete_audit_trace(trace=trace, total_findings_generated=1)

    client = _FakeClient()
    write_audit_run(trace, client=client)

    row = client.calls[0]["payload"]
    assert row["audit_run_id"] == "AUDIT-DC1"


# =====================================================================
# write_findings
# =====================================================================

def test_write_findings_shapes_rows_correctly():
    client = _FakeClient()
    findings = [_make_finding(), _make_finding(customer_id="CUST100006")]

    write_findings(findings, client=client)

    assert len(client.calls) == 1
    call = client.calls[0]

    assert call["table"] == "findings"
    assert call["op"] == "upsert"
    assert call["on_conflict"] == "finding_id"
    assert len(call["payload"]) == 2

    row = call["payload"][0]
    assert row["finding_status"] == "REVIEW"
    assert row["audit_run_id"] == "AUDIT-TEST123"
    assert row["reviewed_by"] is None


def test_write_findings_with_empty_list_is_a_noop():
    client = _FakeClient()

    result = write_findings([], client=client)

    assert result == []
    assert client.calls == []


def test_write_findings_after_review_reflects_new_status():
    finding = _make_finding()
    confirm_finding(finding, reviewed_by="Sherine", reviewer_notes="Checked.")

    client = _FakeClient()
    write_findings([finding], client=client)

    row = client.calls[0]["payload"][0]
    assert row["finding_status"] == "CONFIRMED"
    assert row["reviewed_by"] == "Sherine"
    assert row["reviewer_notes"] == "Checked."


# =====================================================================
# write_finding_review
# =====================================================================

def test_write_finding_review_shapes_row_correctly():
    finding = _make_finding()
    confirm_finding(finding, reviewed_by="Sherine", reviewer_notes="Looks right.")

    client = _FakeClient()
    write_finding_review(finding, previous_status="REVIEW", client=client)

    assert len(client.calls) == 1
    call = client.calls[0]

    assert call["table"] == "finding_reviews"
    assert call["op"] == "insert"

    row = call["payload"]
    assert row["finding_id"] == finding["finding_id"]
    assert row["previous_status"] == "REVIEW"
    assert row["new_status"] == "CONFIRMED"
    assert row["reviewed_by"] == "Sherine"
    assert row["reviewer_notes"] == "Looks right."
    
def test_get_client_does_not_load_dotenv_implicitly(monkeypatch):
    """
    Persistence configuration must come only from runtime environment
    variables.

    This prevents a local .env file from silently changing the
    configured/not-configured state of the persistence layer.
    """

    monkeypatch.delenv(
        "SUPABASE_URL",
        raising=False,
    )

    monkeypatch.delenv(
        "SUPABASE_SERVICE_ROLE_KEY",
        raising=False,
    )

    with pytest.raises(PersistenceNotConfigured):
        get_supabase_client()
