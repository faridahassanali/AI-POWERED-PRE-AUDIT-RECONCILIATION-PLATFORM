"""
AI Explanation Pipeline -- Stage 3.

Wires Task A (engine.llm.router.explain -- Groq primary / Gemini
fallback) to Task B (engine.ai_output_validation -- the schema /
grounding / no-mutation / no-invented-evidence gate) for CONFIRMED
findings that already have a resolved policy_context.

Kept as its own module rather than folded into
engine.audit_pipeline.explain_confirmed_findings() on purpose:

    - explain_confirmed_findings() (Stage 2) runs the DETERMINISTIC
      template explainer (engine.finding_explainer.explain_finding) --
      no network calls, no LLM, always available. It must keep
      working exactly as-is with zero dependency on Task A/B, same
      principle as engine.audit_pipeline / engine.persistence staying
      decoupled (see engine/audit_orchestration.py's docstring).
    - This module is Stage 3: the REAL AI explanation, which can fail
      (network, rate limits, a still-broken API key) in ways Stage 2
      never can. Keeping it separate means a Stage 3 failure can never
      take down Stage 2, and callers who only need the deterministic
      explanation never pay for or depend on this module.

Pipeline per finding:

    1. RAG.retriever.retrieve_for_finding()  -- resolve policy_context,
       grounded strictly to the finding's own policy_references.
    2. engine.ai_input.build_ai_input()       -- the confirmed-only /
       non-empty-policy_context / non-empty-evidence gate.
    3. engine.llm.router.explain()            -- Groq primary, Gemini
       fallback, normalized LLMExplanation.to_dict() output.
    4. engine.ai_output_validation.validate_ai_output_or_raise() --
       schema + registry grounding + no-mutation + no-invented-
       evidence, re-checked independently of step 3's own safeguards.
    5. On success: ai_explanation / ai_recommendation are attached
       directly onto the finding dict (mutated in place, same
       convention as engine.finding_review.confirm_finding), so
       engine.persistence.write_findings() picks them up with zero
       changes -- _finding_row() already reads
       finding.get("ai_explanation") / finding.get("ai_recommendation").

       Additionally, a small set of underscore-prefixed metadata keys
       (_ai_model_used, _ai_policy_context) are attached too -- these
       are NOT part of the finding_schema.json contract and are never
       read by _finding_row(); they exist purely so
       engine.persistence.write_ai_output() can build the
       public.ai_outputs row (model_name, retrieved_policy_context)
       without re-running retrieval or re-deriving which provider
       answered.

Failure handling: one finding failing (LLM outage, validation
rejection, etc.) does NOT stop the batch -- every other finding still
gets a result. Failures are never silent: each one is recorded with
its own error message in the returned AIExplanationResult, so a
caller (e.g. Person C's API layer) can decide what to do (retry,
surface to a reviewer, etc.) instead of the batch either fully
succeeding or fully dying on the first bad finding.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from engine.ai_input import AIInputValidationError, build_ai_input
from engine.ai_output_validation import (
    AIOutputValidationError,
    validate_ai_output_or_raise,
)
from engine.data_loader import DATA_DIR
from engine.llm.base import LLMError, LLMProvider
from engine.llm.hallucination_tripwire import check_for_hallucinations_or_raise
from engine.llm.router import explain
from engine.policy_registry import PolicyRegistry, load_policy_registry
from RAG.retriever import retrieve_for_finding


logger = logging.getLogger(__name__)


@dataclass
class AIExplanationResult:
    """
    Outcome of generating + validating an AI explanation for ONE
    finding. Exactly one of (finding/ai_output) or (error) is
    populated, controlled by `succeeded`.
    """

    finding_id: str
    succeeded: bool
    finding: dict[str, Any] | None = None
    ai_output: dict[str, Any] | None = None
    error: str | None = None


def generate_ai_explanation_for_finding(
    finding: dict[str, Any],
    registry: PolicyRegistry,
    primary: LLMProvider | None = None,
    fallback: LLMProvider | None = None,
) -> AIExplanationResult:
    """
    Run the full A+B chain for a single CONFIRMED finding.

    On success, `finding` is MUTATED IN PLACE (ai_explanation and
    ai_recommendation attached, plus the internal _ai_* metadata
    keys used by engine.persistence.write_ai_output()) and also
    returned via the result, so callers can either use the return
    value or rely on the mutation -- same convention as
    engine.finding_review.confirm_finding().

    Raises nothing -- every failure mode (RAG resolution, the gate,
    the LLM layer, output validation) is caught and returned as a
    failed AIExplanationResult instead, so a batch caller never needs
    a try/except around this function.

    primary/fallback are injectable straight through to
    engine.llm.router.explain(), purely for testing -- production
    callers should omit both.
    """

    finding_id = finding.get("finding_id", "<unknown>")

    try:
        policy_context = retrieve_for_finding(
            finding=finding,
            registry=registry,
        )

        ai_input = build_ai_input(
            finding,
            policy_context=policy_context,
        )

        # Snapshot immediately before the AI call, for the no-mutation
        # check below -- NOT before human review, which already
        # happened earlier in the pipeline. See
        # engine.ai_output_validation.validate_no_finding_mutation's
        # docstring: before/after must bracket the explain() call.
        finding_before_ai = copy.deepcopy(finding)

        ai_output = explain(
            ai_input,
            primary=primary,
            fallback=fallback,
        )

        # Task A's own safeguard on its own output -- runs BEFORE
        # Task B's validation, since a fabricated fact in the free
        # text is a different failure mode than a bad citation, and
        # catching it here is more specific than letting it flow
        # through to a human reviewer.
        check_for_hallucinations_or_raise(
            ai_output=ai_output,
            ai_input=ai_input,
        )

        validate_ai_output_or_raise(
            ai_output=ai_output,
            ai_input=ai_input,
            finding_before=finding_before_ai,
            finding_after=finding,
            registry=registry,
        )

    except (
        AIInputValidationError,
        LLMError,
        AIOutputValidationError,
        ValueError,
    ) as exc:

        logger.warning(
            "AI explanation failed for finding %s: %s: %s",
            finding_id,
            type(exc).__name__,
            exc,
        )

        return AIExplanationResult(
            finding_id=finding_id,
            succeeded=False,
            error=f"{type(exc).__name__}: {exc}",
        )

    # ---------------------------------------------------------------
    # Success -- attach to the finding so persistence picks it up
    # with zero changes (_finding_row() already reads these keys).
    #
    # The underscore-prefixed keys below are internal-only metadata
    # for engine.persistence.write_ai_output() -- they are NOT part
    # of finding_schema.json and _finding_row() never reads them, so
    # they never leak into the public.findings table.
    # ---------------------------------------------------------------

    finding["ai_explanation"] = ai_output["ai_explanation"]
    finding["ai_recommendation"] = ai_output["ai_recommendation"]
    finding["_ai_model_used"] = ai_output.get("model_used")
    finding["_ai_policy_context"] = ai_input.get("policy_context", [])

    return AIExplanationResult(
        finding_id=finding_id,
        succeeded=True,
        finding=finding,
        ai_output=ai_output,
    )


def generate_ai_explanations(
    findings: list[dict[str, Any]],
    registry: PolicyRegistry | None = None,
    data_dir: Path | str | None = None,
    primary: LLMProvider | None = None,
    fallback: LLMProvider | None = None,
) -> list[AIExplanationResult]:
    """
    Run generate_ai_explanation_for_finding() over a batch of
    CONFIRMED findings.

    A registry can be passed in (e.g. one already loaded once by the
    caller, to avoid re-parsing the policy files per call); otherwise
    one is loaded fresh from data_dir (defaults to the project's
    data/ directory, same convention as run_audit() and
    explain_confirmed_findings()).

    One finding failing never stops the batch -- see
    AIExplanationResult and generate_ai_explanation_for_finding()'s
    docstrings for the failure-handling contract.
    """

    if registry is None:
        registry = load_policy_registry(
            Path(data_dir) if data_dir is not None else DATA_DIR
        )

    return [
        generate_ai_explanation_for_finding(
            finding,
            registry=registry,
            primary=primary,
            fallback=fallback,
        )
        for finding in findings
    ]