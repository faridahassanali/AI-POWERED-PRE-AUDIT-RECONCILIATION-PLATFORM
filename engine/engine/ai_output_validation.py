"""
AI Output Validation.

Validates the normalized AI explanation dict (LLMExplanation.to_dict(),
produced by engine.llm.router.explain()) before it is allowed to be
persisted or shown to a reviewer.

This is the last gate between the LLM layer and persistence. It does
NOT trust the LLM layer's own internal safeguards (e.g.
engine.llm.base.validate_citations) -- it re-checks independently,
against the full Policy Registry rather than just the policy_context
slice that was sent to the model, and it also guards against the AI
layer accidentally mutating the original finding.

Four checks, run together by validate_ai_output():

    1. Schema check          -- required keys present, correct types.
    2. Grounding check        -- every cited_policy_id exists in the
                                  Policy Registry (not just in the
                                  policy_context that was sent).
    3. No mutation check      -- the AI layer did not change the
                                  finding's severity/status fields,
                                  and the explanation is actually
                                  attached to the right finding.
    4. No invented evidence   -- cited_policy_ids stay inside what
                                  the model was actually given
                                  (policy_context), so it can't cite
                                  something it was never shown.
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
    Check that ai_output has the exact shape produced by
    LLMExplanation.to_dict().

    Returns a list of human-readable error messages.
    An empty list means the shape is valid.
    """

    errors: list[str] = []

    if not isinstance(ai_output, dict):
        return ["ai_output must be a dictionary."]

    for field in REQUIRED_STRING_FIELDS:
        value = ai_output.get(field)

        if field not in ai_output:
            errors.append(f"Missing required field: {field}")
        elif not isinstance(value, str) or not value.strip():
            errors.append(
                f"{field} must be a non-empty string, got: {value!r}"
            )

    if "cited_policy_ids" not in ai_output:
        errors.append("Missing required field: cited_policy_ids")
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
# 2. GROUNDING CHECK (against the Policy Registry)
# =====================================================================

def validate_grounding(
    ai_output: dict[str, Any],
    registry: PolicyRegistry,
) -> list[str]:
    """
    Check that every policy_id the model cited actually exists in the
    Policy Registry.

    This is stricter than engine.llm.base.validate_citations(), which
    only checks against the policy_context slice sent to the model
    for that one call. This check re-verifies against the full,
    authoritative registry -- the real source of truth.
    """

    errors: list[str] = []

    cited_policy_ids = ai_output.get("cited_policy_ids", [])

    if not isinstance(cited_policy_ids, list):
        # Already reported by the schema check; nothing more to do here.
        return errors

    for policy_id in cited_policy_ids:
        if not isinstance(policy_id, str):
            continue

        if not registry.contains(policy_id):
            errors.append(
                f"Cited policy_id not found in registry: {policy_id!r}"
            )

    return errors


# =====================================================================
# 3. NO MUTATION CHECK (severity / status integrity + identity match)
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
)-> list[str]:
    """
    Check that:

      a) the AI explanation is actually attached to the finding it
         was generated for (finding_id / audit_run_id match), and

      b) generating the explanation did not change the finding's
         protected fields (severity, assessment_status,
         finding_status).

    finding_before / finding_after should be the same finding dict
    captured immediately before and immediately after the call to
    engine.llm.router.explain() -- this guards against an accidental
    mutation bug in the AI layer, not against the LLM's free-text
    content.
    """

    errors: list[str] = []

    if ai_output.get("finding_id") != finding_before.get("finding_id"):
        errors.append(
            "ai_output.finding_id does not match the source finding "
            f"(expected {finding_before.get('finding_id')!r}, got "
            f"{ai_output.get('finding_id')!r})."
        )

    if ai_output.get("audit_run_id") != finding_before.get("audit_run_id"):
        errors.append(
            "ai_output.audit_run_id does not match the source finding "
            f"(expected {finding_before.get('audit_run_id')!r}, got "
            f"{ai_output.get('audit_run_id')!r})."
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
# 4. NO INVENTED EVIDENCE (citations must stay inside what was sent)
# =====================================================================

def validate_no_invented_evidence(
    ai_output: dict[str, Any],
    ai_input: dict[str, Any],
) -> list[str]:
    """
    Check that every cited_policy_id is a subset of the policy_ids
    that were actually present in ai_input["policy_context"] -- i.e.
    the model didn't cite something it was never shown, even if that
    policy_id happens to exist somewhere in the registry.

    This is a narrower, complementary check to validate_grounding():
    validate_grounding() asks "does this policy exist at all?",
    this asks "was the model actually given this policy for this
    finding?".
    """

    errors: list[str] = []

    cited_policy_ids = ai_output.get("cited_policy_ids", [])

    if not isinstance(cited_policy_ids, list):
        return errors

    policy_context = ai_input.get("policy_context", [])

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
                f"Cited policy_id was not present in the "
                f"policy_context given to the model: {policy_id!r} "
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
    Run all four checks and return the combined list of error
    messages. An empty list means the AI output is valid.

    Args:
        ai_output: LLMExplanation.to_dict() result from
            engine.llm.router.explain().
        ai_input: the AI Input Contract dict that was sent to the
            LLM (engine.ai_input.build_ai_input() output).
        finding_before: the source finding, captured before the
            explain() call.
        finding_after: the same finding, captured after the
            explain() call (should normally be identical to
            finding_before).
        registry: the loaded PolicyRegistry.

    Returns:
        A list of human-readable validation error messages.
    """

    errors: list[str] = []

    # If the schema itself is broken, the other checks can't
    # meaningfully run against cited_policy_ids -- but they're
    # defensive enough (isinstance guards) to run anyway without
    # crashing, so we still collect everything in one pass.
    errors += validate_ai_output_schema(ai_output)
    errors += validate_grounding(ai_output, registry)
    errors += validate_no_finding_mutation(
        ai_output, finding_before, finding_after
    )
    errors += validate_no_invented_evidence(ai_output, ai_input)

    return errors


def validate_ai_output(
    ai_output: dict[str, Any],
    ai_input: dict[str, Any],
    finding_before: dict[str, Any],
    finding_after: dict[str, Any],
    registry: PolicyRegistry,
) -> bool:
    """
    Check whether an AI output is valid.

    Returns:
        True if valid, otherwise False.
    """

    return len(
        get_ai_output_validation_errors(
            ai_output, ai_input, finding_before, finding_after, registry
        )
    ) == 0


def validate_ai_output_or_raise(
    ai_output: dict[str, Any],
    ai_input: dict[str, Any],
    finding_before: dict[str, Any],
    finding_after: dict[str, Any],
    registry: PolicyRegistry,
) -> None:
    """
    Validate an AI output and raise an exception if invalid.

    Raises:
        AIOutputValidationError: if the AI output fails any check.
    """

    errors = get_ai_output_validation_errors(
        ai_output, ai_input, finding_before, finding_after, registry
    )

    if errors:
        error_message = (
            "AI output validation failed:\n- " + "\n- ".join(errors)
        )
        raise AIOutputValidationError(error_message)