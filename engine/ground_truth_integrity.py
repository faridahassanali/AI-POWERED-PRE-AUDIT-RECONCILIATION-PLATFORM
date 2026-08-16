from typing import Any


VALID_SEVERITIES = {
    "LOW",
    "MEDIUM",
    "HIGH",
    "CRITICAL",
}

VALID_ASSESSMENT_STATUSES = {
    "PASS",
    "FAIL",
    "UNKNOWN",
}

VALID_FINDING_STATUSES = {
    "REVIEW",
    "CONFIRMED",
    "DISMISSED",
}


def validate_ground_truth(
    findings: list[dict[str, Any]],
    control_ids: set[str],
) -> list[str]:
    """
    Validate the structural integrity of the ground-truth findings.

    Returns a list of validation errors.
    """

    errors = []
    seen_ids = set()

    for index, finding in enumerate(findings):

        prefix = f"Finding index {index}"

        finding_id = finding.get("finding_id")

        if not finding_id:
            errors.append(f"{prefix}: missing finding_id")

        elif finding_id in seen_ids:
            errors.append(
                f"{prefix}: duplicate finding_id '{finding_id}'"
            )

        else:
            seen_ids.add(finding_id)

        control_id = finding.get("control_id")

        if control_id not in control_ids:
            errors.append(
                f"{prefix}: unknown control_id '{control_id}'"
            )

        severity = finding.get("severity")

        if severity not in VALID_SEVERITIES:
            errors.append(
                f"{prefix}: invalid severity '{severity}'"
            )

        assessment_status = finding.get("assessment_status")

        if assessment_status not in VALID_ASSESSMENT_STATUSES:
            errors.append(
                f"{prefix}: invalid assessment_status "
                f"'{assessment_status}'"
            )

        finding_status = finding.get("finding_status")

        if finding_status not in VALID_FINDING_STATUSES:
            errors.append(
                f"{prefix}: invalid finding_status "
                f"'{finding_status}'"
            )

        if "evidence" not in finding:
            errors.append(
                f"{prefix}: missing evidence"
            )

        if "policy_references" not in finding:
            errors.append(
                f"{prefix}: missing policy_references"
            )

    return errors