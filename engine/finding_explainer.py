"""
Finding Explainability / Audit Traceability Layer.

Transforms a validated finding into a structured,
auditor-readable explanation.

This layer is deterministic and does not make new
compliance decisions. It only explains an existing finding.
"""

from typing import Any


def _format_policy_references(
    policy_references: list[dict[str, Any]],
) -> list[str]:
    """
    Convert structured policy references into readable strings.
    """

    formatted = []

    for policy in policy_references:
        policy_id = policy.get("policy_id", "")
        version = policy.get("version", "")
        section = policy.get("section", "")

        parts = [policy_id]

        if version:
            parts.append(f"version {version}")

        if section:
            parts.append(f"section {section}")

        formatted.append(" — ".join([
            parts[0],
            ", ".join(parts[1:]) if len(parts) > 1 else "",
        ]).strip(" —"))

    return formatted


def explain_finding(
    finding: dict[str, Any],
) -> dict[str, Any]:
    """
    Build an auditor-friendly explanation for one finding.

    Important:
        This function does NOT determine whether a control passes
        or fails. It only explains an already-generated finding.
    """

    control_id = finding.get("control_id", "")
    customer_id = finding.get("customer_id")
    severity = finding.get("severity", "")
    assessment_status = finding.get("assessment_status", "")
    finding_status = finding.get("finding_status", "")

    expected = finding.get("expected", "")
    actual = finding.get("actual", "")

    evidence = finding.get("evidence", {})
    policy_references = finding.get("policy_references", [])

    explanation = {
        "finding_id": finding.get("finding_id"),
        "audit_run_id": finding.get("audit_run_id"),

        "control_id": control_id,
        "customer_id": customer_id,

        "severity": severity,
        "assessment_status": assessment_status,
        "finding_status": finding_status,

        "summary": (
            f"Control {control_id} failed for customer "
            f"{customer_id}."
        ),

        "expected_condition": expected,

        "observed_condition": actual,

        "evidence": evidence,

        "policy_references": _format_policy_references(
            policy_references
        ),

        "review_action": (
            "Review the provided evidence against the applicable "
            "policy requirement and determine the appropriate "
            "remediation or approval."
        ),
    }

    return explanation