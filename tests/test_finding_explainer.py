from engine.finding_explainer import explain_finding


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

    explanation = explain_finding(finding)

    assert explanation["finding_id"] == "FINDING-001"
    assert explanation["audit_run_id"] == "AUDIT-001"

    assert explanation["control_id"] == "RISK_001"
    assert explanation["customer_id"] == "CUST100002"

    assert explanation["severity"] == "HIGH"
    assert explanation["assessment_status"] == "FAIL"

    assert explanation["expected_condition"] == finding["expected"]
    assert explanation["observed_condition"] == finding["actual"]

    assert explanation["evidence"] == finding["evidence"]

    assert "RISK-POLICY-001" in explanation["policy_references"][0]


def test_explanation_does_not_change_finding():

    finding = {
        "finding_id": "FINDING-001",
        "audit_run_id": "AUDIT-001",
        "control_id": "SCREENING_001",
        "customer_id": "CUST100005",
        "severity": "CRITICAL",
        "assessment_status": "FAIL",
        "finding_status": "REVIEW",
        "expected": "Opened wallets must have a CLEAR screening result.",
        "actual": "Screening status is HIGH_RISK.",
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

    original = finding.copy()

    explain_finding(finding)

    assert finding == original