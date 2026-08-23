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
        
def test_partial_explanations_are_allowed():
    """
    Only confirmed findings may have AI explanations.
    Findings without explanations must remain in the output.
    """

    trace = create_audit_trace(
        audit_run_id="AUDIT-006",
        controls_executed=["RISK_001"],
        total_records_evaluated=3,
    )

    findings = [
        {
            "finding_id": "FIND-006-A",
            "audit_run_id": "AUDIT-006",
            "control_id": "RISK_001",
            "customer_id": "CUST006",
            "finding_status": "CONFIRMED",
        },
        {
            "finding_id": "FIND-006-B",
            "audit_run_id": "AUDIT-006",
            "control_id": "RISK_001",
            "customer_id": "CUST007",
            "finding_status": "REJECTED",
        },
        {
            "finding_id": "FIND-006-C",
            "audit_run_id": "AUDIT-006",
            "control_id": "RISK_001",
            "customer_id": "CUST008",
            "finding_status": "CONFIRMED",
        },
    ]

    explanations = [
        {
            "finding_id": "FIND-006-A",
            "audit_run_id": "AUDIT-006",
            "control_id": "RISK_001",
            "customer_id": "CUST006",
        },
        {
            "finding_id": "FIND-006-C",
            "audit_run_id": "AUDIT-006",
            "control_id": "RISK_001",
            "customer_id": "CUST008",
        },
    ]

    output = build_audit_output(
        audit_trace=trace,
        findings=findings,
        explanations=explanations,
        evaluation=None,
        report="test report",
    )

    assert len(output.findings) == 3

    assert (
        output.findings[0].explanation
        == explanations[0]
    )

    assert output.findings[1].explanation is None

    assert (
        output.findings[2].explanation
        == explanations[1]
    )
def test_explanations_are_matched_by_finding_id_not_position():
    """
    Explanations may arrive in a different order than findings.
    They must be matched using finding_id.
    """

    trace = create_audit_trace(
        audit_run_id="AUDIT-007",
        controls_executed=["RISK_001"],
        total_records_evaluated=2,
    )

    findings = [
        {
            "finding_id": "FIND-007-A",
            "audit_run_id": "AUDIT-007",
            "control_id": "RISK_001",
            "customer_id": "CUST007A",
        },
        {
            "finding_id": "FIND-007-B",
            "audit_run_id": "AUDIT-007",
            "control_id": "RISK_001",
            "customer_id": "CUST007B",
        },
    ]

    explanation_b = {
        "finding_id": "FIND-007-B",
        "audit_run_id": "AUDIT-007",
        "control_id": "RISK_001",
        "customer_id": "CUST007B",
    }

    explanation_a = {
        "finding_id": "FIND-007-A",
        "audit_run_id": "AUDIT-007",
        "control_id": "RISK_001",
        "customer_id": "CUST007A",
    }

    # Intentionally reversed order.
    explanations = [
        explanation_b,
        explanation_a,
    ]

    output = build_audit_output(
        audit_trace=trace,
        findings=findings,
        explanations=explanations,
        evaluation=None,
        report="test",
    )

    assert (
        output.findings[0].explanation
        == explanation_a
    )

    assert (
        output.findings[1].explanation
        == explanation_b
    )
    
def test_duplicate_explanation_for_same_finding_is_rejected():
    """
    The same finding must not receive multiple explanations.
    """

    trace = create_audit_trace(
        audit_run_id="AUDIT-008",
        controls_executed=["RISK_001"],
        total_records_evaluated=1,
    )

    finding = {
        "finding_id": "FIND-008",
        "audit_run_id": "AUDIT-008",
        "control_id": "RISK_001",
        "customer_id": "CUST008",
    }

    explanation = {
        "finding_id": "FIND-008",
        "audit_run_id": "AUDIT-008",
        "control_id": "RISK_001",
        "customer_id": "CUST008",
    }

    try:
        build_audit_output(
            audit_trace=trace,
            findings=[finding],
            explanations=[
                explanation,
                explanation.copy(),
            ],
            evaluation=None,
            report="test",
        )

        assert False, (
            "Expected ValueError for duplicate explanation."
        )

    except ValueError as exc:
        assert "Duplicate explanation" in str(exc)
        
def test_explanation_without_finding_id_is_rejected():
    """
    Every explanation must identify the finding it belongs to.
    """

    trace = create_audit_trace(
        audit_run_id="AUDIT-009",
        controls_executed=["RISK_001"],
        total_records_evaluated=1,
    )

    finding = {
        "finding_id": "FIND-009",
        "audit_run_id": "AUDIT-009",
        "control_id": "RISK_001",
        "customer_id": "CUST009",
    }

    explanation = {
        "audit_run_id": "AUDIT-009",
        "control_id": "RISK_001",
        "customer_id": "CUST009",
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
            "Expected ValueError for missing finding_id."
        )

    except ValueError as exc:
        assert "finding_id" in str(exc)
        
def test_no_explanations_are_allowed():
    """
    Pre-AI output is valid when no explanations exist.
    """

    trace = create_audit_trace(
        audit_run_id="AUDIT-010",
        controls_executed=["RISK_001"],
        total_records_evaluated=1,
    )

    finding = {
        "finding_id": "FIND-010",
        "audit_run_id": "AUDIT-010",
        "control_id": "RISK_001",
        "customer_id": "CUST010",
    }

    output = build_audit_output(
        audit_trace=trace,
        findings=[finding],
        explanations=None,
        evaluation=None,
        report="pre-ai report",
    )

    assert len(output.findings) == 1
    assert output.findings[0].explanation is None
    
def test_explanation_control_id_mismatch_is_rejected():
    trace = create_audit_trace(
        audit_run_id="AUDIT-011",
        controls_executed=["RISK_001"],
        total_records_evaluated=1,
    )

    finding = {
        "finding_id": "FIND-011",
        "audit_run_id": "AUDIT-011",
        "control_id": "RISK_001",
        "customer_id": "CUST011",
    }

    explanation = {
        "finding_id": "FIND-011",
        "audit_run_id": "AUDIT-011",
        "control_id": "WRONG_CONTROL",
        "customer_id": "CUST011",
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
        assert "control_id" in str(exc)  
        
def test_explanation_customer_id_mismatch_is_rejected():
    trace = create_audit_trace(
        audit_run_id="AUDIT-012",
        controls_executed=["RISK_001"],
        total_records_evaluated=1,
    )

    finding = {
        "finding_id": "FIND-012",
        "audit_run_id": "AUDIT-012",
        "control_id": "RISK_001",
        "customer_id": "CUST012",
    }

    explanation = {
        "finding_id": "FIND-012",
        "audit_run_id": "AUDIT-012",
        "control_id": "RISK_001",
        "customer_id": "WRONG_CUSTOMER",
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
        assert "customer_id" in str(exc)        