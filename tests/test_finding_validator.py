"""
Tests for the Finding Schema Validator.

These tests verify that valid findings are accepted
and structurally invalid findings are rejected.
"""

import copy

import pytest

from engine.finding_validator import (
    FindingValidationError,
    get_validation_errors,
    validate_finding,
    validate_finding_or_raise,
)


@pytest.fixture
def valid_finding():
    """Return a valid finding that follows finding_schema.json."""

    return {
        "finding_id": "F-ABC12345",
        "audit_run_id": "RUN-ABC12345",
        "control_id": "SCREENING_001",
        "customer_id": "CUST100005",
        "severity": "HIGH",
        "assessment_status": "FAIL",
        "finding_status": "REVIEW",
        "expected": "Opened wallets must have a CLEAR screening result.",
        "actual": "Screening status is HIGH_RISK.",
        "evidence": {
            "customer_id": "CUST100005",
            "screening_status": "HIGH_RISK",
            "screening_evidence_present": True,
            "screening_reference": "SCR-100005",
            "wallet_status": "OPENED",
        },
        "policy_references": [
            {
                "policy_id": "SCREENING-POLICY-001",
                "version": "1.0",
                "section": "Requirements",
            }
        ],
        "reviewed_by": None,
        "review_timestamp": None,
        "reviewer_notes": None,
        "ai_explanation": None,
        "ai_recommendation": None,
    }


def test_valid_finding_is_accepted(valid_finding):
    """A correctly structured finding should be valid."""

    assert validate_finding(valid_finding) is True


def test_valid_finding_has_no_validation_errors(valid_finding):
    """A valid finding should return an empty error list."""

    errors = get_validation_errors(valid_finding)

    assert errors == []


def test_missing_required_field_is_rejected(valid_finding):
    """A finding missing a required field should be invalid."""

    finding = copy.deepcopy(valid_finding)

    del finding["control_id"]

    assert validate_finding(finding) is False

    errors = get_validation_errors(finding)

    assert any("control_id" in error for error in errors)


def test_invalid_severity_is_rejected(valid_finding):
    """A severity outside the schema enum should be rejected."""

    finding = copy.deepcopy(valid_finding)

    finding["severity"] = "BANANA"

    assert validate_finding(finding) is False

    errors = get_validation_errors(finding)

    assert any("severity" in error for error in errors)


def test_invalid_assessment_status_is_rejected(valid_finding):
    """An invalid assessment status should be rejected."""

    finding = copy.deepcopy(valid_finding)

    finding["assessment_status"] = "INVALID_STATUS"

    assert validate_finding(finding) is False

    errors = get_validation_errors(finding)

    assert any("assessment_status" in error for error in errors)


def test_invalid_finding_status_is_rejected(valid_finding):
    """An invalid finding status should be rejected."""

    finding = copy.deepcopy(valid_finding)

    finding["finding_status"] = "INVALID_STATUS"

    assert validate_finding(finding) is False

    errors = get_validation_errors(finding)

    assert any("finding_status" in error for error in errors)


def test_customer_id_can_be_none(valid_finding):
    """
    customer_id may be null.

    This is required for findings such as RECON_001 where
    a record may exist only in the final report.
    """

    finding = copy.deepcopy(valid_finding)

    finding["control_id"] = "RECON_001"
    finding["customer_id"] = None

    finding["expected"] = (
        "Final report must match the approved source."
    )

    finding["actual"] = (
        "Record exists only in the final report."
    )

    finding["evidence"] = {
        "record": "EXTRA"
    }

    finding["policy_references"] = [
        {
            "policy_id": "RECON-POLICY-001",
            "version": "1.0",
            "section": "Requirements",
        }
    ]

    assert validate_finding(finding) is True


def test_evidence_can_contain_control_specific_fields(valid_finding):
    """
    Evidence should allow control-specific fields.

    The main finding schema intentionally keeps evidence flexible.
    """

    finding = copy.deepcopy(valid_finding)

    finding["evidence"] = {
        "customer_id": "CUST100005",
        "screening_status": "HIGH_RISK",
        "screening_evidence_present": True,
        "screening_reference": "SCR-100005",
        "wallet_status": "OPENED",
        "extra_control_specific_field": "allowed",
        "nested_data": {
            "source": "screening_results.csv"
        },
    }

    assert validate_finding(finding) is True


def test_optional_ai_fields_can_be_none(valid_finding):
    """
    AI fields should initially be allowed to remain null.

    The deterministic audit engine does not depend on AI output.
    """

    finding = copy.deepcopy(valid_finding)

    finding["ai_explanation"] = None
    finding["ai_recommendation"] = None

    assert validate_finding(finding) is True


def test_optional_review_fields_can_be_none(valid_finding):
    """Review fields may initially be null."""

    finding = copy.deepcopy(valid_finding)

    finding["reviewed_by"] = None
    finding["review_timestamp"] = None
    finding["reviewer_notes"] = None

    assert validate_finding(finding) is True


def test_invalid_policy_reference_structure_is_rejected(valid_finding):
    """
    policy_references must follow the structure defined
    in finding_schema.json.
    """

    finding = copy.deepcopy(valid_finding)

    finding["policy_references"] = [
        {
            "policy_id": "SCREENING-POLICY-001",
            # version is intentionally missing
            "section": "Requirements",
        }
    ]

    assert validate_finding(finding) is False

    errors = get_validation_errors(finding)

    assert any("version" in error for error in errors)


def test_policy_references_must_be_a_list(valid_finding):
    """policy_references should not be a single object."""

    finding = copy.deepcopy(valid_finding)

    finding["policy_references"] = {
        "policy_id": "SCREENING-POLICY-001",
        "version": "1.0",
        "section": "Requirements",
    }

    assert validate_finding(finding) is False


def test_expected_must_be_string(valid_finding):
    """expected should contain a textual expected condition."""

    finding = copy.deepcopy(valid_finding)

    finding["expected"] = 123

    assert validate_finding(finding) is False

    errors = get_validation_errors(finding)

    assert any("expected" in error for error in errors)


def test_actual_must_be_string(valid_finding):
    """actual should contain a textual observed condition."""

    finding = copy.deepcopy(valid_finding)

    finding["actual"] = {
        "status": "HIGH_RISK"
    }

    assert validate_finding(finding) is False

    errors = get_validation_errors(finding)

    assert any("actual" in error for error in errors)


def test_validate_finding_or_raise_accepts_valid_finding(valid_finding):
    """validate_finding_or_raise should not raise for a valid finding."""

    validate_finding_or_raise(valid_finding)


def test_validate_finding_or_raise_raises_for_invalid_finding(
    valid_finding,
):
    """validate_finding_or_raise should raise for an invalid finding."""

    finding = copy.deepcopy(valid_finding)

    finding["severity"] = "INVALID"

    with pytest.raises(FindingValidationError):
        validate_finding_or_raise(finding)


def test_multiple_validation_errors_are_reported(valid_finding):
    """
    Multiple schema violations should be returned together,
    rather than stopping at the first error.
    """

    finding = copy.deepcopy(valid_finding)

    del finding["control_id"]
    finding["severity"] = "INVALID"
    finding["assessment_status"] = "INVALID"

    errors = get_validation_errors(finding)

    assert len(errors) >= 3

    assert any("control_id" in error for error in errors)
    assert any("severity" in error for error in errors)
    assert any("assessment_status" in error for error in errors)