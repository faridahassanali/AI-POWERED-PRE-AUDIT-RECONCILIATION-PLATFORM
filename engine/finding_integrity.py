from typing import Any


def find_duplicate_findings(
    findings: list[dict[str, Any]],
) -> list[tuple[str, str, str | None]]:
    """
    Find duplicate findings based on:

        control_id + customer_id + assessment_status

    Returns a list of duplicate keys.
    """

    seen = set()
    duplicates = []

    for finding in findings:
        key = (
            finding.get("control_id"),
            finding.get("customer_id"),
            finding.get("assessment_status"),
        )

        if key in seen:
            duplicates.append(key)
        else:
            seen.add(key)

    return duplicates


def validate_finding_uniqueness(
    findings: list[dict[str, Any]],
) -> tuple[bool, list[tuple[str, str, str | None]]]:
    """
    Validate that no duplicate finding exists.

    Returns:
        (is_valid, duplicate_keys)
    """

    duplicates = find_duplicate_findings(findings)

    return len(duplicates) == 0, duplicates


def validate_unique_findings(
    findings: list[dict[str, Any]],
) -> bool:
    """
    Validate that all findings are unique.

    This is the pipeline-facing helper.

    Returns:
        True  -> findings are unique
        False -> duplicate findings exist
    """

    is_valid, _ = validate_finding_uniqueness(findings)

    return is_valid