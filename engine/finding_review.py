"""
Human Review Layer.

Handles human confirmation or rejection of audit findings.

Allowed transitions:

REVIEW -> CONFIRMED
REVIEW -> REJECTED
"""

from datetime import datetime, timezone
from typing import Any
from engine.finding_validator import validate_finding_or_raise


def _review_finding(
    finding: dict[str, Any],
    new_status: str,
    reviewed_by: str,
    reviewer_notes: str | None = None,
) -> dict[str, Any]:
    """
    Apply a human review decision to a finding.
    """

    # A finding can only be reviewed once it is in REVIEW state.
    if finding.get("finding_status") != "REVIEW":
        raise ValueError(
            "Finding must have status REVIEW before it can be reviewed."
        )

    # Reviewer identity is required.
    if not reviewed_by or not reviewed_by.strip():
        raise ValueError("reviewed_by is required.")

    # Only these two review decisions are allowed.
    if new_status not in {"CONFIRMED", "REJECTED"}:
        raise ValueError(
            f"Invalid review status: {new_status}"
        )

    # Update the finding with the review decision.
    finding["finding_status"] = new_status
    finding["reviewed_by"] = reviewed_by
    finding["review_timestamp"] = datetime.now(timezone.utc).isoformat()
    finding["reviewer_notes"] = reviewer_notes

    # Re-validate against the schema after mutation — a review
    # decision must never produce a finding that fails the contract
    # the rest of the pipeline (and the future AI layer) relies on.
    validate_finding_or_raise(finding)

    return finding


def confirm_finding(
    finding: dict[str, Any],
    reviewed_by: str,
    reviewer_notes: str | None = None,
) -> dict[str, Any]:
    """
    Confirm a finding after human review.

    REVIEW -> CONFIRMED
    """

    return _review_finding(
        finding=finding,
        new_status="CONFIRMED",
        reviewed_by=reviewed_by,
        reviewer_notes=reviewer_notes,
    )


def reject_finding(
    finding: dict[str, Any],
    reviewed_by: str,
    reviewer_notes: str | None = None,
) -> dict[str, Any]:
    """
    Reject a finding after human review.

    REVIEW -> REJECTED
    """

    return _review_finding(
        finding=finding,
        new_status="REJECTED",
        reviewed_by=reviewed_by,
        reviewer_notes=reviewer_notes,
    )