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

    Findings may exist without explanations.

    Only findings that reached the AI stage will have
    an explanation. Therefore, explanations do not need
    to exist for every finding.

    Explanations are matched to findings using finding_id,
    not list position.

    Every supplied explanation must correspond to an
    existing finding in the same audit run.

    This function does not modify the original findings.
    """

    audit_run_id = audit_trace.audit_run_id

    # ---------------------------------------------------------
    # NORMALIZE EXPLANATIONS
    # ---------------------------------------------------------

    if explanations is None:
        explanations = []

    # ---------------------------------------------------------
    # BUILD FINDING INDEX
    # ---------------------------------------------------------
    # We use finding_id as the canonical identity between
    # findings and explanations.
    # ---------------------------------------------------------

    finding_by_id: dict[str, dict[str, Any]] = {}

    for finding in findings:

        finding_id = finding.get("finding_id")

        if not finding_id:
            raise ValueError(
                "Every finding must contain finding_id."
            )

        if finding_id in finding_by_id:
            raise ValueError(
                f"Duplicate finding for finding_id: "
                f"{finding_id}"
            )

        if finding.get("audit_run_id") != audit_run_id:
            raise ValueError(
                "Finding audit_run_id does not match "
                "the audit trace."
            )

        finding_by_id[finding_id] = finding

    # ---------------------------------------------------------
    # BUILD EXPLANATION INDEX
    # ---------------------------------------------------------
    # AI explanations may exist only for CONFIRMED findings.
    #
    # Example:
    #
    # findings:
    #   F001 -> CONFIRMED
    #   F002 -> REJECTED
    #   F003 -> CONFIRMED
    #
    # explanations:
    #   F001
    #   F003
    #
    # Therefore, explanations are matched by finding_id.
    # ---------------------------------------------------------

    explanation_by_finding_id: dict[str, dict[str, Any]] = {}

    for explanation in explanations:

        finding_id = explanation.get("finding_id")

        if not finding_id:
            raise ValueError(
                "Every explanation must contain finding_id."
            )

        if finding_id in explanation_by_finding_id:
            raise ValueError(
                f"Duplicate explanation for finding_id: "
                f"{finding_id}"
            )

        if explanation.get("audit_run_id") != audit_run_id:
            raise ValueError(
                "Explanation audit_run_id does not match "
                "the audit trace."
            )

        # -----------------------------------------------------
        # IMPORTANT:
        # Every explanation must belong to an actual finding.
        # -----------------------------------------------------

        if finding_id not in finding_by_id:
            raise ValueError(
                f"Explanation references unknown "
                f"finding_id: {finding_id}"
            )

        explanation_by_finding_id[finding_id] = explanation

    # ---------------------------------------------------------
    # BUILD OUTPUT
    # ---------------------------------------------------------

    paired_findings: list[AuditFindingOutput] = []

    for finding in findings:

        finding_id = finding["finding_id"]

        # -----------------------------------------------------
        # MATCH EXPLANATION BY FINDING ID
        # -----------------------------------------------------

        explanation = explanation_by_finding_id.get(
            finding_id
        )

        # -----------------------------------------------------
        # VALIDATE MATCHED EXPLANATION
        # -----------------------------------------------------

        if explanation is not None:

            _validate_finding_explanation_pair(
                finding,
                explanation,
            )

        # -----------------------------------------------------
        # BUILD FINDING OUTPUT
        # -----------------------------------------------------

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