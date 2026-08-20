"""
AI Input Contract.

Builds the controlled input passed from a human-confirmed
audit finding to the AI explanation layer.

Only CONFIRMED findings are allowed to enter this contract.
REVIEW, REJECTED, and RESOLVED findings are blocked.
"""

from typing import Any

from engine.finding_validator import validate_finding_or_raise


class AIInputValidationError(ValueError):
    """Raised when a finding cannot be used as AI input."""


def build_ai_input(
    finding: dict[str, Any],
) -> dict[str, Any]:
    """
    Build the controlled AI input from a confirmed finding.

    Only findings with finding_status == "CONFIRMED" are allowed.

    Args:
        finding: A standardized audit finding.

    Returns:
        A structured AI input containing only the fields
        required by the AI explanation layer.

    Raises:
        AIInputValidationError:
            If the finding is not CONFIRMED or does not comply
            with the finding schema.
    """

    finding_status = finding.get("finding_status")

    if finding_status != "CONFIRMED":
        raise AIInputValidationError(
            "AI input is only allowed for CONFIRMED findings."
        )

    try:
        validate_finding_or_raise(finding)
    except Exception as exc:
        raise AIInputValidationError(
            f"Invalid finding cannot be used as AI input: {exc}"
        ) from exc

    return {
        "finding_id": finding["finding_id"],
        "audit_run_id": finding["audit_run_id"],
        "control_id": finding["control_id"],
        "customer_id": finding["customer_id"],
        "severity": finding["severity"],
        "assessment_status": finding["assessment_status"],
        "finding_status": finding["finding_status"],
        "expected": finding["expected"],
        "actual": finding["actual"],
        "evidence": finding["evidence"],
        "policy_references": finding["policy_references"],
        "reviewed_by": finding["reviewed_by"],
        "review_timestamp": finding["review_timestamp"],
        "reviewer_notes": finding["reviewer_notes"],
    }