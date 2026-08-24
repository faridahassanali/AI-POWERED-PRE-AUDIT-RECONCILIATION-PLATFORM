from engine.audit_trace import (
    create_audit_trace,
    complete_audit_trace,
)


def test_create_audit_trace():

    trace = create_audit_trace(
        audit_run_id="AUDIT-001",
        controls_executed=[
            "SCREENING_001",
            "RISK_001",
            "ARABIC_NAME_001",
            "DORMANT_001",
            "RECON_001",
        ],
        total_records_evaluated=1000,
    )

    assert trace.audit_run_id == "AUDIT-001"

    assert trace.started_at is not None
    assert trace.completed_at is None

    assert len(trace.controls_executed) == 5

    assert trace.total_records_evaluated == 1000
    assert trace.total_findings_generated == 0


def test_complete_audit_trace():

    trace = create_audit_trace(
        audit_run_id="AUDIT-001",
        controls_executed=["SCREENING_001"],
        total_records_evaluated=100,
    )

    trace = complete_audit_trace(
        trace,
        total_findings_generated=25,
    )

    assert trace.completed_at is not None
    assert trace.total_findings_generated == 25
    
import pytest

from engine.audit_trace import (
    AuditTrace,
    complete_audit_trace,
    create_audit_trace,
    fail_audit_trace,
)


def test_create_audit_trace_starts_in_running_state():
    trace = create_audit_trace(
        audit_run_id="AUDIT-TEST-001",
        controls_executed=["SCREENING_001"],
        total_records_evaluated=100,
    )

    assert isinstance(trace, AuditTrace)
    assert trace.audit_run_id == "AUDIT-TEST-001"
    assert trace.status == "RUNNING"
    assert trace.started_at is not None
    assert trace.completed_at is None
    assert trace.error_type is None
    assert trace.error_message is None
    assert trace.total_records_evaluated == 100


def test_complete_audit_trace_marks_run_completed():
    trace = create_audit_trace(
        audit_run_id="AUDIT-TEST-002",
        controls_executed=["RISK_001"],
        total_records_evaluated=50,
    )

    complete_audit_trace(
        trace=trace,
        total_findings_generated=7,
    )

    assert trace.status == "COMPLETED"
    assert trace.completed_at is not None
    assert trace.total_findings_generated == 7
    assert trace.error_type is None
    assert trace.error_message is None


def test_fail_audit_trace_records_failure():
    trace = create_audit_trace(
        audit_run_id="AUDIT-TEST-003",
        controls_executed=["SCREENING_001"],
        total_records_evaluated=100,
    )

    error = ValueError("Invalid screening data")

    fail_audit_trace(
        trace=trace,
        error=error,
    )

    assert trace.status == "FAILED"
    assert trace.completed_at is not None
    assert trace.error_type == "ValueError"
    assert trace.error_message == "Invalid screening data"


def test_failed_trace_preserves_audit_identity():
    trace = create_audit_trace(
        audit_run_id="AUDIT-TEST-004",
        controls_executed=["RISK_001"],
        total_records_evaluated=25,
    )

    fail_audit_trace(
        trace,
        RuntimeError("Database unavailable"),
    )

    assert trace.audit_run_id == "AUDIT-TEST-004"
    assert trace.controls_executed == ["RISK_001"]
    assert trace.total_records_evaluated == 25
    assert trace.status == "FAILED"