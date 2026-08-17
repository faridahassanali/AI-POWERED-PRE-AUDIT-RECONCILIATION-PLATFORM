import pytest

from engine.finding_builder import build_finding
from engine.finding_review import confirm_finding, reject_finding


def make_finding():
    return build_finding(
        control_id="SCREENING_001",
        customer_id="CUST100005",
        severity="HIGH",
        assessment_status="FAIL",
        finding_status="REVIEW",
        expected="Opened wallets must have a CLEAR screening result.",
        actual="Screening status is HIGH_RISK.",
        evidence={
            "screening_status": "HIGH_RISK",
            "screening_evidence_present": "True"
        },
        policy_references=[
            {
                "policy_id": "SCREENING-POLICY-001",
                "version": "1.0",
                "section": "Requirements"
            }
        ]
    )


def test_confirm_finding():
    finding = make_finding()

    result = confirm_finding(
        finding,
        reviewed_by="Sherine",
        reviewer_notes="Evidence verified."
    )

    assert result["finding_status"] == "CONFIRMED"
    assert result["reviewed_by"] == "Sherine"
    assert result["review_timestamp"] is not None
    assert result["reviewer_notes"] == "Evidence verified."


def test_reject_finding():
    finding = make_finding()

    result = reject_finding(
        finding,
        reviewed_by="Sherine",
        reviewer_notes="False positive."
    )

    assert result["finding_status"] == "REJECTED"
    assert result["reviewed_by"] == "Sherine"
    assert result["review_timestamp"] is not None
    assert result["reviewer_notes"] == "False positive."


def test_cannot_confirm_already_confirmed_finding():
    finding = make_finding()

    confirm_finding(
        finding,
        reviewed_by="Sherine"
    )

    with pytest.raises(ValueError):
        confirm_finding(
            finding,
            reviewed_by="Sherine"
        )


def test_cannot_reject_already_rejected_finding():
    finding = make_finding()

    reject_finding(
        finding,
        reviewed_by="Sherine"
    )

    with pytest.raises(ValueError):
        reject_finding(
            finding,
            reviewed_by="Sherine"
        )


def test_cannot_confirm_rejected_finding():
    finding = make_finding()

    reject_finding(
        finding,
        reviewed_by="Sherine"
    )

    with pytest.raises(ValueError):
        confirm_finding(
            finding,
            reviewed_by="Sherine"
        )


def test_reviewed_by_is_required():
    finding = make_finding()

    with pytest.raises(ValueError):
        confirm_finding(
            finding,
            reviewed_by=""
        )


def test_cannot_reject_confirmed_finding():
    finding = make_finding()

    confirm_finding(
        finding,
        reviewed_by="Sherine"
    )

    with pytest.raises(ValueError):
        reject_finding(
            finding,
            reviewed_by="Sherine"
        )