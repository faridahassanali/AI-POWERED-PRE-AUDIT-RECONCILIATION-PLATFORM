"""
LLM Layer -- Base Contracts.

Defines the provider interface, the normalized output shape, and the
exception hierarchy shared by every provider (Groq, Gemini, ...).

Design principle (per the platform's core architecture):

    The deterministic engine has already decided WHAT happened
    (finding_status, severity, assessment_status, expected/actual).
    A human reviewer has already CONFIRMED the finding.
    RAG + Policy Registry have already resolved WHICH policy applies.

    The LLM's only job is to narrate those pre-verified facts in
    plain language. It never re-decides the finding, never invents
    a policy that wasn't in policy_context, and never guesses at
    facts not present in ai_input.

Every provider in this package must:

    1. Accept the same ai_input dict (the AI Input Contract produced
       by engine.ai_input.build_ai_input()).
    2. Return the same normalized shape (see LLMExplanation below),
       regardless of which underlying model answered.
    3. Raise one of the exceptions below -- never a raw SDK/HTTP
       exception -- so the router can decide whether to retry,
       fail over, or stop.
"""

from dataclasses import dataclass, field
from typing import Any, Protocol


# =====================================================================
# EXCEPTIONS
# =====================================================================

class LLMError(Exception):
    """Base class for every exception raised by this package."""


class LLMTransientError(LLMError):
    """
    A retryable / fail-over-able problem: timeout, rate limit (429),
    5xx, connection reset, etc.

    retry_after: seconds the server told us to wait, if it gave one
    (e.g. Groq's 429 Retry-After header). Optional -- providers that
    don't have this information just leave it None, and the router
    falls back to its own default backoff.

    The router treats this as "try again, then try the other
    provider" -- never a silent skip.
    """

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class LLMConfigError(LLMError):
    """
    A non-retryable configuration problem: missing/invalid API key,
    missing SDK, malformed base URL, etc.

    This is intentionally NOT treated as fail-over-able. A bad API
    key is a standing problem, not a blip -- silently routing every
    call to the fallback provider would hide a broken primary
    provider indefinitely, which is worse than a loud failure in a
    compliance system. The router re-raises this immediately.
    """


class LLMOutputError(LLMError):
    """
    The provider responded, but the response could not be parsed
    into the required JSON shape, or failed grounding validation
    (e.g. cited a policy_id that isn't in policy_context).

    Treated as retryable by the router (one retry with a stricter
    reminder), since it's usually a one-off formatting slip rather
    than a persistent problem.
    """


# =====================================================================
# NORMALIZED OUTPUT
# =====================================================================

@dataclass
class LLMExplanation:
    """
    The normalized shape every provider must return.

    finding_id / audit_run_id: carried through unchanged from
    ai_input, so the caller never has to cross-reference.

    ai_explanation / ai_recommendation: the model's narration,
    grounded strictly in ai_input["policy_context"].

    cited_policy_ids: the policy_id values the explanation actually
    relies on. Always a subset of the policy_ids present in
    ai_input["policy_context"] -- providers must enforce this
    themselves (see prompts.py) and the router double-checks it.

    provider_used / model_used: NOT shown to the human reviewer in
    the explanation text -- these exist purely for the audit_trace /
    persistence layer, so a later investigation or cost review can
    tell which provider actually generated a given explanation, even
    though the failover itself is invisible to the reviewer.
    """

    finding_id: str
    audit_run_id: str
    ai_explanation: str
    ai_recommendation: str
    cited_policy_ids: list[str]
    provider_used: str
    model_used: str
    raw_attempts: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "finding_id": self.finding_id,
            "audit_run_id": self.audit_run_id,
            "ai_explanation": self.ai_explanation,
            "ai_recommendation": self.ai_recommendation,
            "cited_policy_ids": self.cited_policy_ids,
            "provider_used": self.provider_used,
            "model_used": self.model_used,
        }


# =====================================================================
# PROVIDER INTERFACE
# =====================================================================

class LLMProvider(Protocol):
    """
    Every provider (GroqProvider, GeminiProvider, ...) implements
    this shape. The router only ever talks to this interface -- it
    has no idea which concrete provider it's calling.
    """

    name: str
    model: str

    def generate(self, ai_input: dict[str, Any]) -> LLMExplanation:
        """
        Call the underlying model and return a normalized
        LLMExplanation.

        Must raise LLMTransientError, LLMConfigError, or
        LLMOutputError on failure -- never let a raw requests/SDK
        exception escape.
        """
        ...


# =====================================================================
# SHARED GROUNDING VALIDATION
# =====================================================================

def validate_citations(
    cited_policy_ids: list[str],
    policy_context: list[dict[str, Any]],
) -> None:
    """
    Enforce that the model didn't cite a policy it wasn't given.

    This is the last line of defense for the grounding guarantee --
    even if the prompt is followed perfectly most of the time, this
    check is what makes the guarantee actually hold.
    """

    allowed = {
        str(chunk.get("policy_id"))
        for chunk in policy_context
    }

    cited = {
        str(policy_id)
        for policy_id in cited_policy_ids
    }

    unknown = cited - allowed

    if unknown:
        raise LLMOutputError(
            "Model cited policy_id(s) not present in policy_context: "
            f"{sorted(unknown)}. Allowed: {sorted(allowed)}."
        )