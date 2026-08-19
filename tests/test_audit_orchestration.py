"""
Tests for engine.audit_orchestration.

These reuse the same fake-Supabase-client pattern as
tests/test_persistence.py so we never touch a real Supabase project.
"""

import pytest

from engine.audit_orchestration import (
    OrchestratedAuditResult,
    PersistenceOutcome,
    run_audit_and_persist,
)
from engine.persistence import PersistenceNotConfigured


# =====================================================================
# FAKE SUPABASE CLIENT (same shape as tests/test_persistence.py)
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
# run_audit_and_persist -- without Supabase configured
# =====================================================================

def test_run_audit_and_persist_skips_persistence_when_not_configured(monkeypatch):
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_SERVICE_ROLE_KEY", raising=False)

    result = run_audit_and_persist()

    assert isinstance(result, OrchestratedAuditResult)
    assert isinstance(result.persistence, PersistenceOutcome)
    assert result.persistence.attempted is True
    assert result.persistence.persisted is False
    assert result.persistence.reason is not None


def test_run_audit_and_persist_still_returns_full_pipeline_result_when_not_configured(monkeypatch):
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_SERVICE_ROLE_KEY", raising=False)

    result = run_audit_and_persist()

    # The whole point: a missing Supabase config must not degrade the
    # deterministic pipeline result in any way.
    assert result.pipeline_result.generated_findings
    assert result.pipeline_result.audit_trace.completed_at is not None
    assert result.pipeline_result.audit_output.audit_run_id == (
        result.pipeline_result.audit_trace.audit_run_id
    )


# =====================================================================
# run_audit_and_persist -- with a (fake) Supabase client
# =====================================================================

def test_run_audit_and_persist_writes_audit_run_and_findings_with_fake_client():
    client = _FakeClient()

    result = run_audit_and_persist(client=client)

    assert result.persistence.attempted is True
    assert result.persistence.persisted is True
    assert result.persistence.reason is None

    tables_written = {call["table"] for call in client.calls}
    assert "audit_runs" in tables_written
    assert "findings" in tables_written

    audit_run_call = next(c for c in client.calls if c["table"] == "audit_runs")
    assert (
        audit_run_call["payload"]["audit_run_id"]
        == result.pipeline_result.audit_trace.audit_run_id
    )

    findings_call = next(c for c in client.calls if c["table"] == "findings")
    assert len(findings_call["payload"]) == len(result.pipeline_result.generated_findings)


# =====================================================================
# Unexpected persistence errors must NOT be swallowed
# =====================================================================

def test_run_audit_and_persist_does_not_swallow_unexpected_errors():
    class _BrokenClient:
        def table(self, name):
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        run_audit_and_persist(client=_BrokenClient())


def test_persistence_not_configured_is_the_only_error_treated_as_optional():
    # Sanity check on the contract itself: PersistenceNotConfigured is a
    # RuntimeError subclass, but run_audit_and_persist must only special
    # case PersistenceNotConfigured, not RuntimeError in general (covered
    # by the previous test using a plain RuntimeError).
    assert issubclass(PersistenceNotConfigured, RuntimeError)