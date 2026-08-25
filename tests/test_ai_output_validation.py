"""
Tests for AI Output Validation.

These tests verify that a valid, well-grounded AI explanation is
accepted, and that each failure mode (schema, grounding, mutation,
invented evidence) is caught independently.
"""

import copy

import pytest

from engine.ai_output_validation import (
    AIOutputValidationError,
    get_ai_output_validation_errors,
    validate_ai_output,
    validate_ai_output_or_raise,
    validate_ai_output_schema,
    validate_grounding,
    validate_no_finding_mutation,
    validate_no_invented_evidence,
)
from engine.policy_registry import PolicyRegistry


# =====================================================================
# FIXTURES
# =====================================================================

@pytest.fixture
def policy_section():
    """A minimal valid policy section."""

    return {
        "section": "Requirements",
        "content": "Customers must be screened before wallet opening.",
    }


@pytest.fixture
def registry(policy_section):
    """A PolicyRegistry containing exactly one known policy."""

    return PolicyRegistry(
        [
            {
                "policy_id": "SCREENING-POLICY-001",
                "version": "1.0",
                "title": "Screening Policy",
                "sections": [policy_section],
            }
        ]
    )


@pytest.fixture
def ai_input():
    """
    A minimal AI Input Contract dict (build_ai_input() output),
    including the policy_context the model was actually given.
    """

    return {
        "finding_id": "F-ABC12345",
        "audit_run_id": "RUN-ABC12345",
        "policy_context": [
            {
                "policy_id": "SCREENING-POLICY-001",
                "version": "1.0",
                "section": "Requirements",
                "content": (
                    "Customers must be screened before wallet opening."
                ),
            }
        ],
    }


@pytest.fixture
def finding_before():
    """The finding as it stood immediately before calling explain()."""

    return {
        "finding_id": "F-ABC12345",
        "audit_run_id": "RUN-ABC12345",
        "control_id": "SCREENING_001",
        "customer_id": "CUST100005",
        "severity": "HIGH",
        "assessment_status": "FAIL",
        "finding_status": "CONFIRMED",
    }


@pytest.fixture
def ai_output():
    """A valid LLMExplanation.to_dict() result."""

    return {
        "finding_id": "F-ABC12345",
        "audit_run_id": "RUN-ABC12345",
        "ai_explanation": (
            "The customer's wallet was opened without a CLEAR "
            "screening result, violating the screening policy."
        ),
        "ai_recommendation": (
            "Escalate to compliance for remediation before the "
            "wallet remains active."
        ),
        "cited_policy_ids": ["SCREENING-POLICY-001"],
        "provider_used": "groq",
        "model_used": "llama-3.1-70b",
    }


# =====================================================================
# VALID CASE
# =====================================================================

def test_valid_ai_output_is_accepted(
    ai_output, ai_input, finding_before, registry
):
    """A correctly structured, grounded AI output should be valid."""

    finding_after = copy.deepcopy(finding_before)

    assert (
        validate_ai_output(
            ai_output, ai_input, finding_before, finding_after, registry
        )
        is True
    )


def test_valid_ai_output_has_no_validation_errors(
    ai_output, ai_input, finding_before, registry
):
    """A valid AI output should return an empty error list."""

    finding_after = copy.deepcopy(finding_before)

    errors = get_ai_output_validation_errors(
        ai_output, ai_input, finding_before, finding_after, registry
    )

    assert errors == []


def test_validate_ai_output_or_raise_accepts_valid_output(
    ai_output, ai_input, finding_before, registry
):
    """validate_ai_output_or_raise should not raise for a valid output."""

    finding_after = copy.deepcopy(finding_before)

    validate_ai_output_or_raise(
        ai_output, ai_input, finding_before, finding_after, registry
    )


# =====================================================================
# 1. SCHEMA CHECK
# =====================================================================

def test_missing_required_field_is_rejected(ai_output):
    """An AI output missing a required field should be invalid."""

    output = copy.deepcopy(ai_output)

    del output["ai_recommendation"]

    errors = validate_ai_output_schema(output)

    assert any("ai_recommendation" in error for error in errors)


def test_empty_string_field_is_rejected(ai_output):
    """An empty ai_explanation should be rejected, not just missing."""

    output = copy.deepcopy(ai_output)

    output["ai_explanation"] = "   "

    errors = validate_ai_output_schema(output)

    assert any("ai_explanation" in error for error in errors)


def test_cited_policy_ids_must_be_a_list(ai_output):
    """cited_policy_ids should not be a single string."""

    output = copy.deepcopy(ai_output)

    output["cited_policy_ids"] = "SCREENING-POLICY-001"

    errors = validate_ai_output_schema(output)

    assert any("cited_policy_ids" in error for error in errors)


def test_cited_policy_ids_elements_must_be_strings(ai_output):
    """Non-string entries inside cited_policy_ids should be rejected."""

    output = copy.deepcopy(ai_output)

    output["cited_policy_ids"] = [123]

    errors = validate_ai_output_schema(output)

    assert any("cited_policy_ids[0]" in error for error in errors)


def test_valid_schema_has_no_errors(ai_output):
    """A correctly shaped AI output should return no schema errors."""

    assert validate_ai_output_schema(ai_output) == []


# =====================================================================
# 2. GROUNDING CHECK (against the registry)
# =====================================================================

def test_citation_not_in_registry_is_rejected(ai_output, registry):
    """A policy_id that doesn't exist in the registry should fail."""

    output = copy.deepcopy(ai_output)

    output["cited_policy_ids"] = ["MADE-UP-POLICY-999"]

    errors = validate_grounding(output, registry)

    assert any("MADE-UP-POLICY-999" in error for error in errors)


def test_citation_in_registry_but_not_in_policy_context_still_passes_grounding(
    ai_output, registry, policy_section
):
    """
    Grounding only checks registry membership -- a policy that
    exists in the registry but wasn't sent to the model passes this
    check. It should be caught separately by the invented-evidence
    check instead.
    """

    registry_with_extra = PolicyRegistry(
        [
            {
                "policy_id": "SCREENING-POLICY-001",
                "version": "1.0",
                "title": "Screening Policy",
                "sections": [policy_section],
            },
            {
                "policy_id": "AML-POLICY-002",
                "version": "1.0",
                "title": "AML Policy",
                "sections": [policy_section],
            },
        ]
    )

    output = copy.deepcopy(ai_output)
    output["cited_policy_ids"] = ["AML-POLICY-002"]

    errors = validate_grounding(output, registry_with_extra)

    assert errors == []


def test_valid_citation_passes_grounding(ai_output, registry):
    """A citation that exists in the registry should pass."""

    assert validate_grounding(ai_output, registry) == []


# =====================================================================
# 3. NO MUTATION CHECK
# =====================================================================

def test_severity_change_is_rejected(ai_output, finding_before):
    """A finding whose severity changed during explain() is rejected."""

    finding_after = copy.deepcopy(finding_before)
    finding_after["severity"] = "LOW"

    errors = validate_no_finding_mutation(
        ai_output, finding_before, finding_after
    )

    assert any("severity" in error for error in errors)


def test_assessment_status_change_is_rejected(ai_output, finding_before):
    """A changed assessment_status should be rejected."""

    finding_after = copy.deepcopy(finding_before)
    finding_after["assessment_status"] = "PASS"

    errors = validate_no_finding_mutation(
        ai_output, finding_before, finding_after
    )

    assert any("assessment_status" in error for error in errors)


def test_finding_status_change_is_rejected(ai_output, finding_before):
    """A changed finding_status should be rejected."""

    finding_after = copy.deepcopy(finding_before)
    finding_after["finding_status"] = "REJECTED"

    errors = validate_no_finding_mutation(
        ai_output, finding_before, finding_after
    )

    assert any("finding_status" in error for error in errors)


def test_finding_id_mismatch_is_rejected(ai_output, finding_before):
    """ai_output attached to the wrong finding_id should be rejected."""

    output = copy.deepcopy(ai_output)
    output["finding_id"] = "F-WRONG9999"

    finding_after = copy.deepcopy(finding_before)

    errors = validate_no_finding_mutation(
        output, finding_before, finding_after
    )

    assert any("finding_id" in error for error in errors)


def test_audit_run_id_mismatch_is_rejected(ai_output, finding_before):
    """ai_output attached to the wrong audit_run_id should be rejected."""

    output = copy.deepcopy(ai_output)
    output["audit_run_id"] = "RUN-WRONG999"

    finding_after = copy.deepcopy(finding_before)

    errors = validate_no_finding_mutation(
        output, finding_before, finding_after
    )

    assert any("audit_run_id" in error for error in errors)


def test_unchanged_finding_passes_mutation_check(ai_output, finding_before):
    """An identical before/after finding should pass."""

    finding_after = copy.deepcopy(finding_before)

    errors = validate_no_finding_mutation(
        ai_output, finding_before, finding_after
    )

    assert errors == []


# =====================================================================
# 4. NO INVENTED EVIDENCE
# =====================================================================

def test_citation_outside_policy_context_is_rejected(ai_output, ai_input):
    """
    A policy_id the model cited but was never given in
    policy_context should be rejected, even if it's a real policy
    elsewhere in the registry.
    """

    output = copy.deepcopy(ai_output)
    output["cited_policy_ids"] = ["AML-POLICY-002"]

    errors = validate_no_invented_evidence(output, ai_input)

    assert any("AML-POLICY-002" in error for error in errors)


def test_citation_inside_policy_context_passes(ai_output, ai_input):
    """A policy_id present in policy_context should pass."""

    assert validate_no_invented_evidence(ai_output, ai_input) == []


def test_empty_policy_context_rejects_any_citation(ai_output):
    """
    If no policy_context was resolved at all, any citation is
    invented by definition.
    """

    empty_ai_input = {"policy_context": []}

    errors = validate_no_invented_evidence(ai_output, empty_ai_input)

    assert any("SCREENING-POLICY-001" in error for error in errors)


# =====================================================================
# AGGREGATOR / MULTIPLE ERRORS
# =====================================================================

def test_multiple_validation_errors_are_reported(
    ai_output, ai_input, finding_before, registry
):
    """
    Several simultaneous violations should all be reported together,
    not just the first one encountered.
    """

    output = copy.deepcopy(ai_output)
    del output["ai_recommendation"]
    output["cited_policy_ids"] = ["MADE-UP-POLICY-999"]

    finding_after = copy.deepcopy(finding_before)
    finding_after["severity"] = "LOW"

    errors = get_ai_output_validation_errors(
        output, ai_input, finding_before, finding_after, registry
    )

    assert len(errors) >= 3
    assert any("ai_recommendation" in error for error in errors)
    assert any("MADE-UP-POLICY-999" in error for error in errors)
    assert any("severity" in error for error in errors)


def test_validate_ai_output_or_raise_raises_for_invalid_output(
    ai_output, ai_input, finding_before, registry
):
    """validate_ai_output_or_raise should raise for an invalid output."""

    output = copy.deepcopy(ai_output)
    output["cited_policy_ids"] = ["MADE-UP-POLICY-999"]

    finding_after = copy.deepcopy(finding_before)

    with pytest.raises(AIOutputValidationError):
        validate_ai_output_or_raise(
            output, ai_input, finding_before, finding_after, registry
        )
# =====================================================================
# NON-DICT ai_output (malformed LLM response / upstream parsing failure)
# =====================================================================

@pytest.mark.parametrize("bad_output", [None, "not a dict", [1, 2, 3], 42])
def test_non_dict_ai_output_fails_schema_cleanly(bad_output):
    """
    A malformed ai_output (e.g. None from a failed upstream parse,
    or an unexpected type) must be rejected with a clear schema
    error, not crash the caller.
    """

    errors = validate_ai_output_schema(bad_output)

    assert errors == ["ai_output must be a dictionary."]


@pytest.mark.parametrize("bad_output", [None, "not a dict", [1, 2, 3], 42])
def test_non_dict_ai_output_does_not_crash_grounding(bad_output, registry):
    """
    validate_grounding must not raise AttributeError when ai_output
    isn't a dict -- the schema check already reports this failure,
    so grounding should just be a no-op rather than crashing the
    whole validation pipeline.
    """

    errors = validate_grounding(bad_output, registry)

    assert errors == []


@pytest.mark.parametrize("bad_output", [None, "not a dict", [1, 2, 3], 42])
def test_non_dict_ai_output_does_not_crash_invented_evidence(
    bad_output, ai_input
):
    """
    validate_no_invented_evidence must not raise AttributeError when
    ai_output isn't a dict, for the same reason as grounding above.
    """

    errors = validate_no_invented_evidence(bad_output, ai_input)

    assert errors == []


@pytest.mark.parametrize("bad_output", [None, "not a dict", [1, 2, 3], 42])
def test_non_dict_ai_output_reported_cleanly_by_aggregator(
    bad_output, ai_input, finding_before, registry
):
    """
    The full aggregator must survive a malformed ai_output end to
    end and report exactly the schema failure -- not crash with an
    unrelated AttributeError from a downstream check.
    """

    finding_after = copy.deepcopy(finding_before)

    errors = get_ai_output_validation_errors(
        bad_output, ai_input, finding_before, finding_after, registry
    )

    assert "ai_output must be a dictionary." in errors        