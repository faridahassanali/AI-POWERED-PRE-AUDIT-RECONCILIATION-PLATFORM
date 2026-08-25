"""
AI Output Validation.

Validates the normalized AI explanation dict
(LLMExplanation.to_dict(), produced by the LLM layer)
before it is allowed to be persisted or shown to a reviewer.

Four checks are performed:

1. Schema check
2. Policy Registry grounding check
3. Finding mutation / identity check
4. No invented evidence / citation check
"""

from typing import Any

from engine.policy_registry import PolicyRegistry


class AIOutputValidationError(ValueError):
    """Raised when an AI output cannot be accepted as-is."""


# =====================================================================
# 1. SCHEMA CHECK
# =====================================================================

REQUIRED_STRING_FIELDS = (
    "finding_id",
    "audit_run_id",
    "ai_explanation",
    "ai_recommendation",
    "provider_used",
    "model_used",
)


def validate_ai_output_schema(
    ai_output: dict[str, Any],
) -> list[str]:
    """
    Validate the required shape of an AI output.

    Returns:
        A list of validation errors.
        Empty list means the schema is valid.
    """

    errors: list[str] = []

    if not isinstance(ai_output, dict):
        return ["ai_output must be a dictionary."]

    for field in REQUIRED_STRING_FIELDS:
        value = ai_output.get(field)

        if field not in ai_output:
            errors.append(
                f"Missing required field: {field}"
            )
        elif not isinstance(value, str) or not value.strip():
            errors.append(
                f"{field} must be a non-empty string, got: {value!r}"
            )

    if "cited_policy_ids" not in ai_output:
        errors.append(
            "Missing required field: cited_policy_ids"
        )
    else:
        cited = ai_output["cited_policy_ids"]

        if not isinstance(cited, list):
            errors.append(
                "cited_policy_ids must be a list, got: "
                f"{type(cited).__name__}"
            )
        else:
            for i, policy_id in enumerate(cited):
                if not isinstance(policy_id, str) or not policy_id.strip():
                    errors.append(
                        f"cited_policy_ids[{i}] must be a non-empty "
                        f"string, got: {policy_id!r}"
                    )

    return errors


# =====================================================================
# 2. GROUNDING CHECK
# =====================================================================

def validate_grounding(
    ai_output: dict[str, Any],
    registry: PolicyRegistry,
) -> list[str]:
    """
    Verify that every cited policy exists in the Policy Registry.

    If ai_output isn't a dict, this returns no errors of its own --
    validate_ai_output_schema() already reports that failure, and
    this check has nothing meaningful to add on top of it.
    """

    if not isinstance(ai_output, dict):
        return []

    errors: list[str] = []

    cited_policy_ids = ai_output.get(
        "cited_policy_ids",
        [],
    )

    if not isinstance(cited_policy_ids, list):
        return errors

    for policy_id in cited_policy_ids:
        if not isinstance(policy_id, str):
            continue

        if not registry.contains(policy_id):
            errors.append(
                "Cited policy_id not found in registry: "
                f"{policy_id!r}"
            )

    return errors


# =====================================================================
# 3. FINDING MUTATION CHECK
# =====================================================================

PROTECTED_FIELDS = (
    "severity",
    "assessment_status",
    "finding_status",
)


def validate_no_finding_mutation(
    ai_output: dict[str, Any],
    finding_before: dict[str, Any],
    finding_after: dict[str, Any],
) -> list[str]:
    """
    Ensure the AI layer did not mutate the finding.

    Protected fields:
        severity
        assessment_status
        finding_status

    Also validates finding_id and audit_run_id identity.

    If ai_output isn't a dict, this returns no errors of its own --
    validate_ai_output_schema() already reports that failure.
    """

    if not isinstance(ai_output, dict):
        return []

    errors: list[str] = []

    if ai_output.get("finding_id") != finding_before.get(
        "finding_id"
    ):
        errors.append(
            "ai_output.finding_id does not match the source "
            "finding "
            f"(expected {finding_before.get('finding_id')!r}, "
            f"got {ai_output.get('finding_id')!r})."
        )

    if ai_output.get("audit_run_id") != finding_before.get(
        "audit_run_id"
    ):
        errors.append(
            "ai_output.audit_run_id does not match the source "
            "finding "
            f"(expected {finding_before.get('audit_run_id')!r}, "
            f"got {ai_output.get('audit_run_id')!r})."
        )

    for field in PROTECTED_FIELDS:
        before = finding_before.get(field)
        after = finding_after.get(field)

        if before != after:
            errors.append(
                f"Finding field '{field}' changed during AI "
                f"explanation: {before!r} -> {after!r}."
            )

    return errors


# =====================================================================
# 4. NO INVENTED EVIDENCE
# =====================================================================

def validate_no_invented_evidence(
    ai_output: dict[str, Any],
    ai_input: dict[str, Any],
) -> list[str]:
    """
    Ensure cited policies were actually present in the
    policy_context sent to the model.

    If ai_output isn't a dict, this returns no errors of its own --
    validate_ai_output_schema() already reports that failure.
    """

    if not isinstance(ai_output, dict):
        return []

    errors: list[str] = []

    cited_policy_ids = ai_output.get(
        "cited_policy_ids",
        [],
    )

    if not isinstance(cited_policy_ids, list):
        return errors

    policy_context = ai_input.get(
        "policy_context",
        [],
    )

    available_policy_ids = {
        str(chunk.get("policy_id"))
        for chunk in policy_context
        if isinstance(chunk, dict)
    }

    for policy_id in cited_policy_ids:
        if not isinstance(policy_id, str):
            continue

        if policy_id not in available_policy_ids:
            errors.append(
                "Cited policy_id was not present in the "
                "policy_context given to the model: "
                f"{policy_id!r} "
                f"(available: {sorted(available_policy_ids)})."
            )

    return errors


# =====================================================================
# AGGREGATOR
# =====================================================================

def get_ai_output_validation_errors(
    ai_output: dict[str, Any],
    ai_input: dict[str, Any],
    finding_before: dict[str, Any],
    finding_after: dict[str, Any],
    registry: PolicyRegistry,
) -> list[str]:
    """
    Run all validation checks and return all errors.
    """

    errors: list[str] = []

    errors += validate_ai_output_schema(
        ai_output
    )

    errors += validate_grounding(
        ai_output,
        registry,
    )

    errors += validate_no_finding_mutation(
        ai_output,
        finding_before,
        finding_after,
    )

    errors += validate_no_invented_evidence(
        ai_output,
        ai_input,
    )

    return errors


def validate_ai_output(
    ai_output: dict[str, Any],
    ai_input: dict[str, Any],
    finding_before: dict[str, Any],
    finding_after: dict[str, Any],
    registry: PolicyRegistry,
) -> bool:
    """
    Return True when the AI output passes all validation checks.
    """

    return (
        len(
            get_ai_output_validation_errors(
                ai_output,
                ai_input,
                finding_before,
                finding_after,
                registry,
            )
        )
        == 0
    )


def validate_ai_output_or_raise(
    ai_output: dict[str, Any],
    ai_input: dict[str, Any],
    finding_before: dict[str, Any],
    finding_after: dict[str, Any],
    registry: PolicyRegistry,
) -> None:
    """
    Validate an AI output and raise if any check fails.
    """

    errors = get_ai_output_validation_errors(
        ai_output,
        ai_input,
        finding_before,
        finding_after,
        registry,
    )

    if errors:
        error_message = (
            "AI output validation failed:\n- "
            + "\n- ".join(errors)
        )

        raise AIOutputValidationError(
            error_message
        )