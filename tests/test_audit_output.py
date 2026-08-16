from engine.audit_output import build_audit_output
from engine.audit_trace import (
    create_audit_trace,
    complete_audit_trace,
)


def test_build_audit_output():

    trace = create_audit_trace(
        audit_run_id="AUDIT-001",
        controls_executed=[
            "SCREENING_001",
            "RISK_001",
        ],
        total_records_evaluated=10,
    )

    finding = {
        "finding_id": "FIND-001",
        "audit_run_id": "AUDIT-001",
        "control_id": "SCREENING_001",
        "customer_id": "CUST001",
        "severity": "HIGH",
        "assessment_status": "FAIL",
        "finding_status": "REVIEW",
        "expected": "Screening must be CLEAR.",
        "actual": "Screening is PENDING.",
        "evidence": {
            "screening_status": "PENDING",
        },
        "policy_references": [
            {
                "policy_id": "SCREENING-POLICY-001",
                "version": "1.0",
                "section": "Requirements",
            }
        ],
    }

    explanation = {
        "finding_id": "FIND-001",
        "audit_run_id": "AUDIT-001",
        "control_id": "SCREENING_001",
        "customer_id": "CUST001",
    }

    complete_audit_trace(
        trace,
        total_findings_generated=1,
    )

    output = build_audit_output(
        audit_trace=trace,
        findings=[finding],
        explanations=[explanation],
        evaluation=None,
        report="test report",
    )

    assert output.audit_run_id == "AUDIT-001"

    assert len(output.findings) == 1

    assert output.findings[0].finding == finding

    assert (
        output.findings[0].explanation
        == explanation
    )

    assert output.audit_trace is trace


def test_output_does_not_modify_findings():

    trace = create_audit_trace(
        audit_run_id="AUDIT-002",
        controls_executed=["RISK_001"],
        total_records_evaluated=5,
    )

    finding = {
        "finding_id": "FIND-002",
        "audit_run_id": "AUDIT-002",
        "control_id": "RISK_001",
        "customer_id": "CUST002",
        "severity": "HIGH",
        "assessment_status": "FAIL",
        "finding_status": "REVIEW",
        "expected": "Risk must be consistent.",
        "actual": "Risk mismatch detected.",
        "evidence": {},
        "policy_references": [],
    }

    original = finding.copy()

    explanation = {
        "finding_id": "FIND-002",
        "audit_run_id": "AUDIT-002",
        "control_id": "RISK_001",
        "customer_id": "CUST002",
    }

    build_audit_output(
        audit_trace=trace,
        findings=[finding],
        explanations=[explanation],
        evaluation=None,
        report="test",
    )

    assert finding == original


def test_mismatched_finding_and_explanation_identity_is_rejected():

    trace = create_audit_trace(
        audit_run_id="AUDIT-004",
        controls_executed=["RISK_001"],
        total_records_evaluated=1,
    )

    finding = {
        "finding_id": "FIND-004",
        "audit_run_id": "AUDIT-004",
        "control_id": "RISK_001",
        "customer_id": "CUST004",
    }

    explanation = {
        "finding_id": "FIND-999",
        "audit_run_id": "AUDIT-004",
        "control_id": "RISK_001",
        "customer_id": "CUST004",
    }

    try:

        build_audit_output(
            audit_trace=trace,
            findings=[finding],
            explanations=[explanation],
            evaluation=None,
            report="test",
        )

        assert False, (
            "Expected ValueError was not raised."
        )

    except ValueError as exc:

        assert "finding_id" in str(exc)
        
def test_finding_with_wrong_audit_run_is_rejected():

    trace = create_audit_trace(
        audit_run_id="AUDIT-005",
        controls_executed=["RISK_001"],
        total_records_evaluated=1,
    )

    finding = {
        "finding_id": "FIND-005",
        "audit_run_id": "AUDIT-WRONG",
        "control_id": "RISK_001",
        "customer_id": "CUST005",
    }

    explanation = {
        "finding_id": "FIND-005",
        "audit_run_id": "AUDIT-WRONG",
        "control_id": "RISK_001",
        "customer_id": "CUST005",
    }

    try:

        build_audit_output(
            audit_trace=trace,
            findings=[finding],
            explanations=[explanation],
            evaluation=None,
            report="test",
        )

        assert False

    except ValueError as exc:

        assert "audit_run_id" in str(exc)