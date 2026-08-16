"""
Finding Builder.

Creates a standardized audit finding that follows
data/finding_schema.json.
"""

from typing import Any
from uuid import uuid4


def generate_finding_id() -> str:
    """Generate a unique finding ID."""
    return f"F-{uuid4().hex[:8].upper()}"


def generate_audit_run_id() -> str:
    """Generate a unique audit run ID."""
    return f"RUN-{uuid4().hex[:8].upper()}"


def build_finding(
    control_id: str,
    customer_id: str | None,
    severity: str,
    assessment_status: str,
    finding_status: str,
    expected: str,
    actual: str,
    evidence: dict[str, Any],
    policy_references: list[dict[str, str | None]],
    audit_run_id: str | None = None,
    finding_id: str | None = None,
    reviewed_by: str | None = None,
    review_timestamp: str | None = None,
    reviewer_notes: str | None = None,
    ai_explanation: str | None = None,
    ai_recommendation: str | None = None,
) -> dict[str, Any]:
    """
    Build one standardized audit finding.

    Required information comes from the audit control.
    Optional review and AI fields are initially None.
    """

    return {
        "finding_id": finding_id or generate_finding_id(),
        "audit_run_id": audit_run_id or generate_audit_run_id(),
        "control_id": control_id,
        "customer_id": customer_id,
        "severity": severity,
        "assessment_status": assessment_status,
        "finding_status": finding_status,
        "expected": expected,
        "actual": actual,
        "evidence": evidence,
        "policy_references": policy_references,
        "reviewed_by": reviewed_by,
        "review_timestamp": review_timestamp,
        "reviewer_notes": reviewer_notes,
        "ai_explanation": ai_explanation,
        "ai_recommendation": ai_recommendation,
    }