"""
AI Input Contract.

Builds the controlled input passed from a human-confirmed
audit finding to the AI explanation layer.

Only CONFIRMED findings are allowed to enter this contract.
REVIEW, REJECTED, and RESOLVED findings are blocked.

policy_context must be pre-resolved against the Policy Registry
(see engine.policy_context.get_verified_policy_context / Person A's
RAG <-> Registry bridge) BEFORE it reaches this function. An empty
policy_context means the finding's policy references didn't resolve
in the registry -- that's a block condition, not an "explain anyway
with no grounding" condition.
"""

from typing import Any

from engine.finding_validator import validate_finding_or_raise


class AIInputValidationError(ValueError):
    """Raised when a finding cannot be used as AI input."""


def build_ai_input(
    finding: dict[str, Any],
    policy_context: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Build the controlled AI input from a confirmed finding.

    Gate logic (per the AI Input Contract spec):
        finding is CONFIRMED
        -> policy_context resolves in the registry (non-empty)
        -> evidence is non-empty
        -> send. Anything else -> block, with a clear reason.

    Only findings with finding_status == "CONFIRMED" are allowed.
    policy_context must be pre-resolved and non-empty -- pass the
    output of engine.policy_context.get_verified_policy_context()
    (or equivalent), not raw/unverified RAG results.
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

    if not policy_context:
        raise AIInputValidationError(
            "AI input requires resolved policy_context -- "
            "the finding's policy references did not resolve in the "
            "registry, or nothing relevant was retrieved."
        )

    evidence = finding.get("evidence")
    if not evidence:
        raise AIInputValidationError(
            "AI input requires non-empty evidence."
        )

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
        "policy_context": policy_context,
    }