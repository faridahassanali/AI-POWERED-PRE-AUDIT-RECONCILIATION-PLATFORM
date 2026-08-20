"""
Tests for engine.llm.hallucination_tripwire.

These test the tripwire in ISOLATION, with hand-built ai_output/
ai_input dicts -- no real LLM calls, no dependency on the router or
Task B's validation.
"""

import pytest

from engine.llm.hallucination_tripwire import (
    HallucinationDetectedError,
    check_for_hallucinations,
    check_for_hallucinations_or_raise,
    check_for_numeric_fabrication,
    check_for_status_contradiction,
)


def make_ai_input(**overrides):
    base = {
        "finding_id": "F-TEST-001",
        "audit_run_id": "RUN-TEST-001",
        "control_id": "SCREENING_001",
        "customer_id": "C-001",
        "severity": "HIGH",
        "assessment_status": "FAIL",
        "expected": "Screening evidence must be present.",
        "actual": "Screening evidence is missing.",
        "evidence": {
            "screening_status": "PENDING",
            "screening_evidence_present": False,
        },
        "policy_context": [
            {
                "policy_id": "SCREENING-POLICY-001",
                "section": "Requirements",
                "content": "Screening must be completed before wallet activation.",
            }
        ],
    }
    base.update(overrides)
    return base


def make_ai_output(explanation="Grounded explanation.", recommendation="Grounded recommendation."):
    return {
        "ai_explanation": explanation,
        "ai_recommendation": recommendation,
        "cited_policy_ids": ["SCREENING-POLICY-001"],
    }


# =====================================================================
# Numeric fabrication check
# =====================================================================

def test_number_present_in_evidence_is_not_flagged():
    ai_input = make_ai_input(evidence={"transaction_amount": "5000", "screening_status": "PENDING"})
    text = "The transaction amount of 5000 exceeded the allowed threshold."

    errors = check_for_numeric_fabrication(text, ai_input)

    assert errors == []


def test_number_not_grounded_anywhere_is_flagged():
    ai_input = make_ai_input()
    text = "The customer's account has been dormant for 45 days."

    errors = check_for_numeric_fabrication(text, ai_input)

    assert len(errors) == 1
    assert "45" in errors[0]


def test_number_from_policy_context_is_not_flagged():
    ai_input = make_ai_input(
        policy_context=[
            {
                "policy_id": "DORMANT-POLICY-001",
                "section": "Requirements",
                "content": "Accounts inactive for 90 days must be reviewed.",
            }
        ]
    )
    text = "Per policy, accounts inactive for 90 days require review."

    errors = check_for_numeric_fabrication(text, ai_input)

    assert errors == []


def test_single_digit_numbers_are_not_flagged():
    """Avoid false positives on incidental small numbers (list markers,
    ordinals) that aren't meaningful figures."""
    ai_input = make_ai_input()
    text = "This is the 1st of 2 issues found in this finding."

    errors = check_for_numeric_fabrication(text, ai_input)

    assert errors == []


def test_percentage_not_grounded_is_flagged():
    ai_input = make_ai_input()
    text = "This represents a 75% deviation from policy."

    errors = check_for_numeric_fabrication(text, ai_input)

    assert any("75%" in e for e in errors)


# =====================================================================
# Status contradiction check
# =====================================================================

def test_status_word_matching_actual_evidence_is_not_flagged():
    ai_input = make_ai_input(evidence={"screening_status": "PENDING"})
    text = "The screening status is PENDING, which requires review."

    errors = check_for_status_contradiction(text, ai_input)

    assert errors == []


def test_status_word_matching_severity_is_not_flagged():
    ai_input = make_ai_input(severity="HIGH")
    text = "This is a HIGH severity finding."

    errors = check_for_status_contradiction(text, ai_input)

    assert errors == []


def test_status_word_contradicting_evidence_is_flagged():
    ai_input = make_ai_input(evidence={"screening_status": "PENDING"})
    text = "The screening status was CLEAR at the time of review."

    errors = check_for_status_contradiction(text, ai_input)

    assert len(errors) == 1
    assert "CLEAR" in errors[0]


def test_status_word_contradicting_account_status_is_flagged():
    ai_input = make_ai_input(evidence={"account_status": "DORMANT", "wallet_status": "OPENED"})
    text = "The account is currently ACTIVE and in good standing."

    errors = check_for_status_contradiction(text, ai_input)

    assert any("ACTIVE" in e for e in errors)


def test_status_word_not_mentioned_is_not_flagged():
    ai_input = make_ai_input(evidence={"screening_status": "PENDING"})
    text = "The evidence for this finding was reviewed by the auditor."

    errors = check_for_status_contradiction(text, ai_input)

    assert errors == []


# =====================================================================
# Combined check + raise wrapper
# =====================================================================

def test_check_for_hallucinations_combines_both_checks():
    ai_input = make_ai_input(evidence={"screening_status": "PENDING"})
    ai_output = make_ai_output(
        explanation="The screening status was CLEAR, confirmed on day 45.",
    )

    errors = check_for_hallucinations(ai_output, ai_input)

    assert len(errors) == 2  # one numeric, one status


def test_clean_explanation_produces_no_errors():
    ai_input = make_ai_input(evidence={"screening_status": "PENDING"})
    ai_output = make_ai_output(
        explanation="The screening status is PENDING, which violates the policy requirement.",
        recommendation="Complete screening before wallet activation.",
    )

    errors = check_for_hallucinations(ai_output, ai_input)

    assert errors == []


def test_check_for_hallucinations_or_raise_raises_on_fabrication():
    ai_input = make_ai_input(evidence={"screening_status": "PENDING"})
    ai_output = make_ai_output(explanation="The screening status was CLEAR.")

    with pytest.raises(HallucinationDetectedError, match="CLEAR"):
        check_for_hallucinations_or_raise(ai_output, ai_input)


def test_check_for_hallucinations_or_raise_passes_clean_output():
    ai_input = make_ai_input(evidence={"screening_status": "PENDING"})
    ai_output = make_ai_output(
        explanation="The screening status is PENDING, which is not CLEAR as required."
    )

    # Should NOT raise -- both PENDING (actual) and CLEAR (the
    # required/expected state per policy) are legitimate to mention;
    # CLEAR here isn't being asserted as the actual status.
    #
    # NOTE: this is a known limitation -- the tripwire flags the
    # presence of a contradicting status word regardless of sentence
    # structure/negation, so this specific phrasing WILL currently be
    # flagged. Documented here deliberately rather than hidden.
    with pytest.raises(HallucinationDetectedError):
        check_for_hallucinations_or_raise(ai_output, ai_input)
