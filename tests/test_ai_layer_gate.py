import pytest

from engine.finding_builder import build_finding
from engine.finding_explainer import explain_finding


def make_finding(status="REVIEW"):
    return build_finding(
        control_id="SCREENING_001",
        customer_id="CUST100005",
        severity="HIGH",
        assessment_status="FAIL",
        finding_status=status,
        expected="Opened wallets must have a CLEAR screening result.",
        actual="Screening status is HIGH_RISK.",
        evidence={
            "screening_status": "HIGH_RISK",
            "screening_evidence_present": "True",
        },
        policy_references=[
            {
                "policy_id": "SCREENING-POLICY-001",
                "version": "1.0",
                "section": "Requirements",
            }
        ],
    )


def test_ai_layer_rejects_review_finding():
    finding = make_finding("REVIEW")

    with pytest.raises(ValueError):
        explain_finding(finding)


def test_ai_layer_rejects_rejected_finding():
    finding = make_finding("REJECTED")

    with pytest.raises(ValueError):
        explain_finding(finding)


def test_ai_layer_accepts_confirmed_finding():
    finding = make_finding("CONFIRMED")

    explanation = explain_finding(finding)

    assert explanation is not None
    assert explanation["finding_id"] == finding["finding_id"]
    assert explanation["audit_run_id"] == finding["audit_run_id"]
    assert explanation["control_id"] == finding["control_id"]
    assert explanation["customer_id"] == finding["customer_id"]
