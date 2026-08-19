import pytest

from engine.ai_input import (
    AIInputValidationError,
    build_ai_input,
)


def make_finding(status="CONFIRMED"):
    return {
        "finding_id": "F-12345678",
        "audit_run_id": "RUN-12345678",
        "control_id": "screening_001",
        "customer_id": "C-001",
        "severity": "HIGH",
        "assessment_status": "FAIL",
        "finding_status": status,
        "expected": "Screening evidence must be present.",
        "actual": "Screening evidence is missing.",
        "evidence": {
            "screening_status": "PENDING",
            "screening_evidence_present": False,
        },
        "policy_references": [
            {
                "policy_id": "POL-001",
                "version": "1.0",
                "section": "3.2",
            }
        ],
        "reviewed_by": "auditor_001",
        "review_timestamp": "2026-08-19T10:00:00+00:00",
        "reviewer_notes": "Evidence reviewed and finding confirmed.",
        "ai_explanation": None,
        "ai_recommendation": None,
    }


def test_confirmed_finding_can_become_ai_input():
    finding = make_finding("CONFIRMED")

    ai_input = build_ai_input(finding)

    assert ai_input["finding_id"] == "F-12345678"
    assert ai_input["finding_status"] == "CONFIRMED"
    assert ai_input["control_id"] == "screening_001"
    assert ai_input["reviewed_by"] == "auditor_001"


def test_review_finding_is_rejected():
    finding = make_finding("REVIEW")

    with pytest.raises(
        AIInputValidationError,
        match="CONFIRMED",
    ):
        build_ai_input(finding)


def test_rejected_finding_is_rejected():
    finding = make_finding("REJECTED")

    with pytest.raises(
        AIInputValidationError,
        match="CONFIRMED",
    ):
        build_ai_input(finding)


def test_resolved_finding_is_rejected():
    finding = make_finding("RESOLVED")

    with pytest.raises(
        AIInputValidationError,
        match="CONFIRMED",
    ):
        build_ai_input(finding)


def test_ai_outputs_are_not_part_of_ai_input():
    finding = make_finding("CONFIRMED")

    ai_input = build_ai_input(finding)

    assert "ai_explanation" not in ai_input
    assert "ai_recommendation" not in ai_input


def test_invalid_confirmed_finding_is_rejected():
    finding = make_finding("CONFIRMED")
    del finding["customer_id"]

    with pytest.raises(AIInputValidationError):
        build_ai_input(finding)