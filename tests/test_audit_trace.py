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