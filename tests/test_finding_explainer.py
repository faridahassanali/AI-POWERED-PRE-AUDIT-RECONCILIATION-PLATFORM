import pytest

from engine.finding_explainer import explain_finding
from engine.finding_review import confirm_finding


def test_finding_explanation_contains_core_information():

    finding = {
        "finding_id": "FINDING-001",
        "audit_run_id": "AUDIT-001",

        "control_id": "RISK_001",
        "customer_id": "CUST100002",

        "severity": "HIGH",
        "assessment_status": "FAIL",
        "finding_status": "REVIEW",

        "expected": (
            "HIGH-risk opened wallets must have "
            "an approved risk exception."
        ),

        "actual": (
            "HIGH-risk opened wallet has no valid "
            "approved risk exception."
        ),

        "evidence": {
            "risk_level": "HIGH",
            "wallet_status": "OPENED",
            "risk_exception_approved": "False",
        },

        "policy_references": [
            {
                "policy_id": "RISK-POLICY-001",
                "version": "1.0",
                "section": "Requirements",
            }
        ],
    }

    # Human review must happen before explanation.
    confirm_finding(
        finding,
        reviewed_by="Sherine",
        reviewer_notes="Finding confirmed after evidence review.",
    )

    explanation = explain_finding(finding)

    assert explanation["finding_id"] == "FINDING-001"
    assert explanation["audit_run_id"] == "AUDIT-001"

    assert explanation["control_id"] == "RISK_001"
    assert explanation["customer_id"] == "CUST100002"

    assert explanation["severity"] == "HIGH"
    assert explanation["assessment_status"] == "FAIL"
    assert explanation["finding_status"] == "CONFIRMED"

    assert (
        explanation["expected_condition"]
        == finding["expected"]
    )

    assert (
        explanation["observed_condition"]
        == finding["actual"]
    )

    assert explanation["evidence"] == finding["evidence"]

    assert explanation["policy_references"] == [
        "RISK-POLICY-001 — version 1.0, section Requirements"
    ]


def test_explanation_does_not_change_finding():

    finding = {
        "finding_id": "FINDING-001",
        "audit_run_id": "AUDIT-001",

        "control_id": "SCREENING_001",
        "customer_id": "CUST100005",

        "severity": "CRITICAL",
        "assessment_status": "FAIL",
        "finding_status": "REVIEW",

        "expected": (
            "Opened wallets must have a CLEAR screening result."
        ),

        "actual": (
            "Screening status is HIGH_RISK."
        ),

        "evidence": {
            "screening_status": "HIGH_RISK",
        },

        "policy_references": [
            {
                "policy_id": "SCREENING-POLICY-001",
                "version": "1.0",
                "section": "Requirements",
            }
        ],
    }

    # Confirm before explanation.
    confirm_finding(
        finding,
        reviewed_by="Sherine",
    )

    original = finding.copy()

    explain_finding(finding)

    # Explanation must not change the finding.
    assert finding == original


def test_ai_layer_rejects_review_finding():

    finding = {
        "finding_id": "FINDING-002",
        "audit_run_id": "AUDIT-002",
        "control_id": "RISK_001",
        "customer_id": "CUST100006",

        "severity": "HIGH",
        "assessment_status": "FAIL",
        "finding_status": "REVIEW",

        "expected": "Risk requirement must be satisfied.",
        "actual": "Risk requirement was not satisfied.",

        "evidence": {},

        "policy_references": [],
    }

    # REVIEW findings must not reach the AI/explanation layer.
    with pytest.raises(ValueError, match="CONFIRMED"):
        explain_finding(finding)


def test_ai_layer_rejects_rejected_finding():

    finding = {
        "finding_id": "FINDING-003",
        "audit_run_id": "AUDIT-003",
        "control_id": "RISK_001",
        "customer_id": "CUST100007",

        "severity": "HIGH",
        "assessment_status": "FAIL",
        "finding_status": "REJECTED",

        "expected": "Risk requirement must be satisfied.",
        "actual": "Risk requirement was not satisfied.",

        "evidence": {},

        "policy_references": [],
    }

    # REJECTED findings must also be blocked.
    with pytest.raises(ValueError, match="CONFIRMED"):
        explain_finding(finding)


def test_confirmed_finding_can_be_explained():

    finding = {
        "finding_id": "FINDING-004",
        "audit_run_id": "AUDIT-004",
        "control_id": "SCREENING_001",
        "customer_id": "CUST100008",

        "severity": "HIGH",
        "assessment_status": "FAIL",
        "finding_status": "REVIEW",

        "expected": "Screening must be CLEAR.",
        "actual": "Screening is PENDING.",

        "evidence": {
            "screening_status": "PENDING",
        },

        "policy_references": [],
    }

    confirm_finding(
        finding,
        reviewed_by="Sherine",
    )

    explanation = explain_finding(finding)

    assert explanation["finding_status"] == "CONFIRMED"
    assert explanation["finding_id"] == finding["finding_id"]