"""
Hallucination Tripwire -- Task A (Mon 24).

"Add a post-generation hallucination tripwire that cross-checks the
LLM's explanation against the pre-verified evidence fields."

This is DIFFERENT from Task B's engine.ai_output_validation:

    Task B checks the AI output's STRUCTURED fields (cited_policy_ids)
    against the Policy Registry and the finding's protected fields.
    It never reads the free-text ai_explanation/ai_recommendation at
    all.

    This module reads the FREE TEXT (ai_explanation +
    ai_recommendation) and checks whether it invents facts -- numbers,
    dates, or status words -- that don't actually appear anywhere in
    the pre-verified inputs the model was given (evidence,
    expected/actual, severity, assessment_status, policy_context
    content).

Two checks, both heuristic by nature (free text can't be validated
like structured data), scoped deliberately narrow to keep false
positives low:

    1. NUMERIC/DATE FABRICATION
       Any number or date-like token in the explanation/recommendation
       that doesn't appear anywhere in the grounded inputs is flagged.
       Numbers are precise -- if the model states "5,000 EGP" or
       "45 days" and that figure is nowhere in evidence/expected/
       actual/policy_context, it was very likely invented, not
       inferred.

    2. STATUS VOCABULARY CONTRADICTION
       The finding schema defines a closed vocabulary for severity,
       assessment_status, and the categorical evidence fields (see
       engine.normalization.CATEGORICAL_FIELDS). If the explanation
       uses one of these known status words (e.g. "CLEAR", "DORMANT",
       "HIGH_RISK") and that word does NOT match the actual value
       present in the finding's own fields, that's a direct
       contradiction of a pre-verified fact -- e.g. the model says
       "the screening status was CLEAR" while evidence says
       screening_status: "PENDING".

Deliberately NOT attempted: general open-ended fact-checking of
arbitrary prose. That would need real NLP/entailment checking and
would produce far more false positives than a narrow, closed-domain
compliance tool like this can tolerate.
"""

from __future__ import annotations

import re
from typing import Any

from engine.llm.base import LLMError


# =====================================================================
# CLOSED VOCABULARY (drawn directly from the finding schema / the
# categorical fields normalization.py defines -- not invented here)
# =====================================================================

KNOWN_STATUS_VOCABULARY = {
    # severity (data/finding_schema.json)
    "LOW", "MEDIUM", "HIGH", "CRITICAL",
    # assessment_status
    "PASS", "FAIL", "NOT_APPLICABLE",
    # NOTE: finding_status values (REVIEW, CONFIRMED, REJECTED,
    # RESOLVED) are deliberately EXCLUDED here. They describe the
    # review workflow state, not evidence content, AND "review" /
    # "confirmed" are extremely common ordinary English words --
    # including them guarantees false positives on completely benign
    # sentences ("which requires review", "confirmed on day 45").
    # Task B's validate_no_finding_mutation() already guards
    # finding_status separately and correctly.
    #
    # common categorical evidence values seen across controls.py
    "CLEAR", "PENDING", "NO_MATCH", "HIGH_RISK",
    "ACTIVE", "DORMANT", "OPENED", "NOT_OPENED", "COMPLETED",
}

# Words that collide with the closed vocabulary above but are common
# English words too ("HIGH" alone, "ACTIVE" alone) -- require these to
# appear as a closer status-like phrase to reduce false positives is
# impractical for a simple tripwire, so instead we only ever compare
# them against the ACTUAL values present in the finding; a word that
# happens to also be the correct value is never flagged (see below).


class HallucinationDetectedError(LLMError):
    """
    Raised when the tripwire finds a strong signal that the LLM
    invented a fact not present in the pre-verified inputs.

    Deliberately a subclass of LLMError (not AIOutputValidationError,
    which is Task B's namespace) -- this is Task A's own safeguard on
    its own output, checked before Task B's validation ever runs.
    """


NUMBER_PATTERN = re.compile(
    r"""
    \b
    \d[\d,]*          # digits, allowing thousands separators
    (?:\.\d+)?         # optional decimal part
    %?                 # optional percent sign -- NOTE: no trailing \b
                       # here. '%' is a non-word char, so a trailing
                       # \b right after it only matches when followed
                       # by a word char (never true for "75% x"),
                       # silently dropping the '%' from every match.
    """,
    re.VERBOSE,
)


def _collect_grounded_text(ai_input: dict[str, Any]) -> str:
    """
    Everything the model was ACTUALLY given and is therefore allowed
    to restate: evidence values, expected/actual, severity,
    assessment_status, control_id, customer_id, and every retrieved
    policy chunk's content. Concatenated into one blob so membership
    checks are simple substring tests.
    """

    parts: list[str] = [
        str(ai_input.get("severity", "")),
        str(ai_input.get("assessment_status", "")),
        str(ai_input.get("control_id", "")),
        str(ai_input.get("customer_id", "")),
        str(ai_input.get("expected", "")),
        str(ai_input.get("actual", "")),
    ]

    evidence = ai_input.get("evidence", {})
    if isinstance(evidence, dict):
        for value in evidence.values():
            parts.append(str(value))

    for chunk in ai_input.get("policy_context", []):
        if isinstance(chunk, dict):
            parts.append(str(chunk.get("content", "")))

    return " ".join(parts)


def _evidence_status_values(ai_input: dict[str, Any]) -> set[str]:
    """
    The ACTUAL status-like values present in this finding: evidence
    values, severity, assessment_status -- uppercased, since the
    vocabulary itself is uppercase. Any KNOWN_STATUS_VOCABULARY word
    the model uses that matches one of THESE is correct by
    definition, not a contradiction.
    """

    values: set[str] = {
        str(ai_input.get("severity", "")).upper(),
        str(ai_input.get("assessment_status", "")).upper(),
    }

    evidence = ai_input.get("evidence", {})
    if isinstance(evidence, dict):
        for value in evidence.values():
            values.add(str(value).upper())

    return values


def check_for_numeric_fabrication(
    text: str,
    ai_input: dict[str, Any],
) -> list[str]:
    """
    Flag any number/date-like token in `text` that does not appear
    anywhere in the grounded inputs (evidence, expected/actual,
    severity, assessment_status, policy_context content).
    """

    errors: list[str] = []

    grounded_text = _collect_grounded_text(ai_input)

    for match in NUMBER_PATTERN.finditer(text):
        token = match.group().strip()

        # Skip trivially small/common numbers that are almost always
        # incidental (e.g. "1", "2 sections", ordinals) -- the goal is
        # catching invented SPECIFIC figures (amounts, day counts,
        # percentages), not every digit that appears in prose.
        bare_digits = token.replace(",", "").replace("%", "").replace(".", "")
        if len(bare_digits) <= 1:
            continue

        if bare_digits not in grounded_text:
            errors.append(
                f"Explanation/recommendation mentions '{token}', which "
                "does not appear anywhere in the finding's evidence, "
                "expected/actual conditions, or retrieved policy text -- "
                "possible fabricated figure."
            )

    return errors


def check_for_status_contradiction(
    text: str,
    ai_input: dict[str, Any],
) -> list[str]:
    """
    Flag any KNOWN_STATUS_VOCABULARY word used in `text` that
    contradicts the finding's actual status values.
    """

    errors: list[str] = []

    actual_values = _evidence_status_values(ai_input)
    text_upper = text.upper()

    for word in KNOWN_STATUS_VOCABULARY:

        pattern = r"\b" + re.escape(word) + r"\b"

        if not re.search(pattern, text_upper):
            continue

        if word in actual_values:
            # The model used a status word that IS actually correct
            # for this finding -- not a contradiction.
            continue

        errors.append(
            f"Explanation/recommendation states status '{word}', which "
            "does not match any of this finding's actual values "
            f"({sorted(v for v in actual_values if v)}) -- possible "
            "contradicted fact."
        )

    return errors


def check_for_hallucinations(ai_output: dict[str, Any], ai_input: dict[str, Any]) -> list[str]:
    """
    Run both tripwire checks against the combined explanation +
    recommendation text. Returns a list of human-readable warnings;
    an empty list means nothing suspicious was found.
    """

    text = " ".join(
        [
            str(ai_output.get("ai_explanation", "")),
            str(ai_output.get("ai_recommendation", "")),
        ]
    )

    errors: list[str] = []
    errors += check_for_numeric_fabrication(text, ai_input)
    errors += check_for_status_contradiction(text, ai_input)

    return errors


def check_for_hallucinations_or_raise(
    ai_output: dict[str, Any],
    ai_input: dict[str, Any],
) -> None:
    """
    Same as check_for_hallucinations(), but raises
    HallucinationDetectedError if anything was flagged.

    Called from engine.ai_explanation_pipeline right after
    engine.llm.router.explain() returns and BEFORE Task B's
    validate_ai_output_or_raise() -- catching a fabricated fact here
    is cheaper and more specific than letting it flow through to a
    human reviewer.
    """

    errors = check_for_hallucinations(ai_output, ai_input)

    if errors:
        raise HallucinationDetectedError(
            "Hallucination tripwire flagged this explanation:\n- "
            + "\n- ".join(errors)
        )
