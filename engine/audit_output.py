"""
Canonical Audit Output Contract.

Combines findings, explanations, audit trace,
evaluation, and report into one immutable-style
integration object.

This layer does not make compliance decisions.
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class AuditFindingOutput:
    """
    One finding together with its explanation.
    """

    finding: dict[str, Any]
    explanation: dict[str, Any]


@dataclass
class AuditOutput:
    """
    Canonical output of one complete audit execution.
    """

    audit_run_id: str

    audit_trace: Any

    findings: list[AuditFindingOutput]

    evaluation: Any

    report: str


def _validate_finding_explanation_pair(
    finding: dict[str, Any],
    explanation: dict[str, Any],
) -> None:
    """
    Ensure that a finding and its explanation
    belong to the same audit entity.
    """

    identity_fields = [
        "finding_id",
        "audit_run_id",
        "control_id",
        "customer_id",
    ]

    for field in identity_fields:

        finding_value = finding.get(field)
        explanation_value = explanation.get(field)

        if finding_value != explanation_value:
            raise ValueError(
                f"Finding and explanation mismatch for "
                f"'{field}': "
                f"{finding_value!r} != "
                f"{explanation_value!r}"
            )


def build_audit_output(
    audit_trace: Any,
    findings: list[dict[str, Any]],
    explanations: list[dict[str, Any]],
    evaluation: Any,
    report: str,
) -> AuditOutput:
    """
    Build the canonical audit output.

    The function validates that every finding has
    exactly one matching explanation.

    It does not modify the original findings.
    """

    if len(findings) != len(explanations):
        raise ValueError(
            "Each finding must have exactly one explanation."
        )

    audit_run_id = audit_trace.audit_run_id

    paired_findings: list[AuditFindingOutput] = []

    for finding, explanation in zip(
        findings,
        explanations,
    ):

        if finding.get("audit_run_id") != audit_run_id:
            raise ValueError(
                "Finding audit_run_id does not match "
                "the audit trace."
            )

        if explanation.get("audit_run_id") != audit_run_id:
            raise ValueError(
                "Explanation audit_run_id does not match "
                "the audit trace."
            )

        _validate_finding_explanation_pair(
            finding,
            explanation,
        )

        paired_findings.append(
            AuditFindingOutput(
                finding=dict(finding),
                explanation=dict(explanation),
            )
        )

    return AuditOutput(
        audit_run_id=audit_run_id,
        audit_trace=audit_trace,
        findings=paired_findings,
        evaluation=evaluation,
        report=report,
    )