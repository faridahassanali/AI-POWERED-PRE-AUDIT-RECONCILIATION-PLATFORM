from typing import Any


class FindingIntegrityError(Exception):
    """Raised when the generated findings contain duplicates."""


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

    WARNING: this function only RETURNS a bool -- it does not raise
    and does not log. A caller that invokes this and discards the
    return value (as engine.audit_pipeline.run_audit() previously
    did) gets NO protection at all: duplicate findings pass straight
    through the pipeline with no error, no warning, nothing.

    Kept for backwards compatibility with any existing caller that
    genuinely wants a bool (e.g. a UI that wants to show a soft
    warning). For the pipeline's actual integrity gate, use
    validate_unique_findings_or_raise() instead, which cannot be
    silently ignored.
    """

    is_valid, _ = validate_finding_uniqueness(findings)

    return is_valid


def validate_unique_findings_or_raise(
    findings: list[dict[str, Any]],
) -> None:
    """
    FIX (bug): the pipeline's finding-integrity gate previously called
    validate_unique_findings() and discarded the boolean result, so
    duplicate findings could never actually stop or flag an audit run
    -- the check existed in name only.

    This is the pipeline-facing integrity gate. It cannot be
    accidentally ignored the way a discarded return value can: a
    duplicate always raises.

    Raises:
        FindingIntegrityError:
            If one or more duplicate findings exist, naming every
            duplicate (control_id, customer_id, assessment_status)
            key found, so the failure is actionable rather than a
            bare "invalid" flag.
    """

    is_valid, duplicates = validate_finding_uniqueness(findings)

    if not is_valid:

        duplicate_list = "\n- ".join(
            f"control_id={control_id!r}, customer_id={customer_id!r}, "
            f"assessment_status={assessment_status!r}"
            for control_id, customer_id, assessment_status in duplicates
        )

        raise FindingIntegrityError(
            "Duplicate findings detected in this audit run:\n- "
            + duplicate_list
        )