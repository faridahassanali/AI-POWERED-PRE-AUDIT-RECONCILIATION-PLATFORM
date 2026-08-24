import pytest

from engine.finding_integrity import (
    FindingIntegrityError,
    find_duplicate_findings,
    validate_finding_uniqueness,
    validate_unique_findings_or_raise,
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


def test_validate_unique_findings_or_raise_passes_when_no_duplicates():
    """Sanity check: distinct findings should never raise."""
    findings = [
        {
            "control_id": "SCREENING_001",
            "customer_id": "CUST100001",
            "assessment_status": "FAIL",
        },
        {
            "control_id": "SCREENING_001",
            "customer_id": "CUST100002",
            "assessment_status": "FAIL",
        },
    ]

    # Should not raise
    validate_unique_findings_or_raise(findings)


def test_validate_unique_findings_or_raise_detects_duplicate():
    """A genuine duplicate (same control_id + customer_id) must raise
    FindingIntegrityError, and the message must identify the duplicate
    so it's actionable in logs/CI output."""
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

    with pytest.raises(FindingIntegrityError) as exc_info:
        validate_unique_findings_or_raise(findings)

    error_message = str(exc_info.value)
    assert "SCREENING_001" in error_message
    assert "CUST100001" in error_message