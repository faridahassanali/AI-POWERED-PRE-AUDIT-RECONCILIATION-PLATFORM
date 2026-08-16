from engine.finding_integrity import (
    find_duplicate_findings,
    validate_finding_uniqueness,
)


def test_unique_findings_are_accepted():

    findings = [
        {
            "control_id": "SCREENING_001",
            "customer_id": "CUST100001",
            "assessment_status": "FAIL",
        },
        {
            "control_id": "RISK_001",
            "customer_id": "CUST100001",
            "assessment_status": "FAIL",
        },
        {
            "control_id": "SCREENING_001",
            "customer_id": "CUST100002",
            "assessment_status": "FAIL",
        },
    ]

    valid, duplicates = validate_finding_uniqueness(findings)

    assert valid is True
    assert duplicates == []


def test_duplicate_findings_are_detected():

    findings = [
        {
            "control_id": "SCREENING_001",
            "customer_id": "CUST100001",
            "assessment_status": "FAIL",
        },
        {
            "control_id": "SCREENING_001",
            "customer_id": "CUST100001",
            "assessment_status": "FAIL",
        },
    ]

    valid, duplicates = validate_finding_uniqueness(findings)

    assert valid is False
    assert len(duplicates) == 1


def test_different_controls_are_not_duplicates():

    findings = [
        {
            "control_id": "SCREENING_001",
            "customer_id": "CUST100001",
            "assessment_status": "FAIL",
        },
        {
            "control_id": "RISK_001",
            "customer_id": "CUST100001",
            "assessment_status": "FAIL",
        },
    ]

    duplicates = find_duplicate_findings(findings)

    assert duplicates == []