from engine.ground_truth_integrity import validate_ground_truth


CONTROL_IDS = {
    "SCREENING_001",
    "RISK_001",
    "ARABIC_NAME_001",
    "DORMANT_001",
    "RECON_001",
}


def test_valid_ground_truth():

    findings = [
        {
            "finding_id": "F-0001",
            "control_id": "SCREENING_001",
            "customer_id": "CUST100001",
            "severity": "HIGH",
            "assessment_status": "FAIL",
            "finding_status": "REVIEW",
            "evidence": {},
            "policy_references": [],
        }
    ]

    errors = validate_ground_truth(
        findings,
        CONTROL_IDS,
    )

    assert errors == []


def test_unknown_control_is_rejected():

    findings = [
        {
            "finding_id": "F-0001",
            "control_id": "UNKNOWN_999",
            "customer_id": "CUST100001",
            "severity": "HIGH",
            "assessment_status": "FAIL",
            "finding_status": "REVIEW",
            "evidence": {},
            "policy_references": [],
        }
    ]

    errors = validate_ground_truth(
        findings,
        CONTROL_IDS,
    )

    assert any("unknown control_id" in error for error in errors)


def test_duplicate_finding_id_is_rejected():

    findings = [
        {
            "finding_id": "F-0001",
            "control_id": "SCREENING_001",
            "customer_id": "CUST100001",
            "severity": "HIGH",
            "assessment_status": "FAIL",
            "finding_status": "REVIEW",
            "evidence": {},
            "policy_references": [],
        },
        {
            "finding_id": "F-0001",
            "control_id": "RISK_001",
            "customer_id": "CUST100002",
            "severity": "HIGH",
            "assessment_status": "FAIL",
            "finding_status": "REVIEW",
            "evidence": {},
            "policy_references": [],
        },
    ]

    errors = validate_ground_truth(
        findings,
        CONTROL_IDS,
    )

    assert any("duplicate finding_id" in error for error in errors)