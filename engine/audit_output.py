"""
Canonical Audit Output Contract.

Combines findings, optional explanations, audit trace,
evaluation, and report into one integration object.

During the pre-AI stage, findings may have no explanations
because they are still awaiting human review.

This layer does not make compliance decisions.
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class AuditFindingOutput:
    """
    One finding together with its optional explanation.

    Explanation is None while the finding is awaiting
    human review.
    """

    finding: dict[str, Any]
    explanation: dict[str, Any] | None


@dataclass
class AuditOutput:
    """
    Canonical output of one audit execution.
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
    explanations: list[dict[str, Any]] | None,
    evaluation: Any,
    report: str,
) -> AuditOutput:
    """
    Build the canonical audit output.

    Findings may be returned without explanations during
    the pre-AI stage while they are awaiting human review.

    If explanations are provided, there must be exactly one
    explanation for every finding and each explanation must
    belong to the corresponding finding.

    This function does not modify the original findings.
    """

    audit_run_id = audit_trace.audit_run_id

    # ---------------------------------------------------------
    # PRE-AI STAGE
    # ---------------------------------------------------------
    # No explanations have been generated yet.
    #
    # Every finding is still available for human review.
    # Therefore, an empty explanation list is valid.
    # ---------------------------------------------------------

    if explanations is None:
        explanations = []

    if len(explanations) not in {0, len(findings)}:
        raise ValueError(
            "Explanations must either be empty or contain "
            "exactly one explanation for every finding."
        )

    paired_findings: list[AuditFindingOutput] = []

    # ---------------------------------------------------------
    # BUILD OUTPUT
    # ---------------------------------------------------------

    for index, finding in enumerate(findings):

        # Every finding must belong to this audit run.
        if finding.get("audit_run_id") != audit_run_id:
            raise ValueError(
                "Finding audit_run_id does not match "
                "the audit trace."
            )

        # No explanation yet.
        explanation = None

        # If explanations exist, match them by position.
        if explanations:
            explanation = explanations[index]

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
                explanation=(
                    dict(explanation)
                    if explanation is not None
                    else None
                ),
            )
        )

    return AuditOutput(
        audit_run_id=audit_run_id,
        audit_trace=audit_trace,
        findings=paired_findings,
        evaluation=evaluation,
        report=report,
    )