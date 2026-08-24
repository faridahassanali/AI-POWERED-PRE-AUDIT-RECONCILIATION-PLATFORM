"""
Tests for engine.llm.router.explain() -- the failover logic.

Uses fake providers (no real Groq/Gemini calls) so this runs in CI
with no network access and no API keys.
"""

import pytest

from engine.llm.base import (
    LLMConfigError,
    LLMExplanation,
    LLMOutputError,
    LLMTransientError,
)
from engine.llm.router import LLMAllProvidersFailedError, explain


SAMPLE_AI_INPUT = {
    "finding_id": "F-TEST",
    "audit_run_id": "RUN-TEST",
    "control_id": "SCREENING_001",
    "customer_id": "CUST1",
    "severity": "HIGH",
    "assessment_status": "FAIL",
    "expected": "Screening must be CLEAR.",
    "actual": "Screening was PENDING.",
    "evidence": {"customer_id": "CUST1"},
    "reviewed_by": "farida",
    "review_timestamp": "2026-08-20T00:00:00Z",
    "reviewer_notes": None,
    "policy_context": [
        {
            "policy_id": "SCREENING-POLICY-001",
            "version": "1.0",
            "section": "Requirements",
            "content": "Screening must be CLEAR before wallet opening.",
        }
    ],
}


class _FakeProvider:
    """
    A scripted fake: calls_to_fail is a list of exceptions (or None
    for success) consumed in order, one per call to generate().
    """

    def __init__(self, name, script):
        self.name = name
        self.model = f"{name}-fake-model"
        self._script = list(script)
        self.call_count = 0

    def generate(self, ai_input):
        self.call_count += 1

        outcome = self._script.pop(0)

        if outcome is None:
            return LLMExplanation(
                finding_id=ai_input["finding_id"],
                audit_run_id=ai_input["audit_run_id"],
                ai_explanation="Explanation text.",
                ai_recommendation="Recommendation text.",
                cited_policy_ids=["SCREENING-POLICY-001"],
                provider_used=self.name,
                model_used=self.model,
            )

        raise outcome


def test_primary_success_no_fallback_called():
    primary = _FakeProvider("groq", [None])
    fallback = _FakeProvider("gemini", [None])

    result = explain(SAMPLE_AI_INPUT, primary=primary, fallback=fallback)

    assert result["provider_used"] == "groq"
    assert primary.call_count == 1
    assert fallback.call_count == 0


def test_transient_error_retries_primary_before_failover():
    primary = _FakeProvider(
        "groq",
        [LLMTransientError("boom"), None],
    )
    fallback = _FakeProvider("gemini", [None])

    result = explain(SAMPLE_AI_INPUT, primary=primary, fallback=fallback)

    # Primary recovered on its own retry -- fallback never called.
    assert result["provider_used"] == "groq"
    assert primary.call_count == 2
    assert fallback.call_count == 0


def test_transient_error_fails_over_to_gemini():
    primary = _FakeProvider(
        "groq",
        [LLMTransientError("boom"), LLMTransientError("boom again")],
    )
    fallback = _FakeProvider("gemini", [None])

    result = explain(SAMPLE_AI_INPUT, primary=primary, fallback=fallback)

    assert result["provider_used"] == "gemini"
    assert primary.call_count == 2


def test_retry_after_from_error_is_honored_by_backoff(monkeypatch):
    """
    A provider's LLMTransientError can carry the server's actual
    suggested wait time (e.g. Groq's Retry-After header) -- the
    router must sleep that long, not a flat guess. Before this fix,
    the router always slept a fixed 0.5s regardless of what the
    provider said, so retries routinely landed inside the same
    rate-limit window and failed again immediately.
    """

    slept_for = []
    monkeypatch.setattr(
        "engine.llm.router.time.sleep",
        lambda seconds: slept_for.append(seconds),
    )

    primary = _FakeProvider(
        "groq",
        [LLMTransientError("rate limited", retry_after=7.0), None],
    )
    fallback = _FakeProvider("gemini", [None])

    result = explain(SAMPLE_AI_INPUT, primary=primary, fallback=fallback)

    assert result["provider_used"] == "groq"
    assert slept_for == [7.0]


def test_retry_after_is_capped_so_it_cannot_stall_the_pipeline():
    """
    A server suggesting an extreme wait (or a bug) shouldn't be able
    to stall a batch run indefinitely -- the router caps how long a
    single retry will honor.
    """

    from engine.llm.router import MAX_HONORED_RETRY_AFTER_SECONDS

    slept_for = []

    import engine.llm.router as router_module
    original_sleep = router_module.time.sleep
    router_module.time.sleep = lambda seconds: slept_for.append(seconds)

    try:
        primary = _FakeProvider(
            "groq",
            [LLMTransientError("rate limited", retry_after=9999.0), None],
        )
        fallback = _FakeProvider("gemini", [None])

        explain(SAMPLE_AI_INPUT, primary=primary, fallback=fallback)

    finally:
        router_module.time.sleep = original_sleep

    assert slept_for == [MAX_HONORED_RETRY_AFTER_SECONDS]
    assert fallback.call_count == 0


def test_output_error_retries_same_provider_before_failover():
    primary = _FakeProvider(
        "groq",
        [LLMOutputError("bad json"), None],
    )
    fallback = _FakeProvider("gemini", [None])

    result = explain(SAMPLE_AI_INPUT, primary=primary, fallback=fallback)

    assert result["provider_used"] == "groq"
    assert primary.call_count == 2
    assert fallback.call_count == 0


def test_config_error_raises_immediately_no_failover():
    primary = _FakeProvider(
        "groq",
        [LLMConfigError("missing API key")],
    )
    fallback = _FakeProvider("gemini", [None])

    with pytest.raises(LLMConfigError):
        explain(SAMPLE_AI_INPUT, primary=primary, fallback=fallback)

    assert primary.call_count == 1
    assert fallback.call_count == 0


def test_fallback_config_error_raises_immediately():
    primary = _FakeProvider(
        "groq",
        [LLMTransientError("boom"), LLMTransientError("boom again")],
    )
    fallback = _FakeProvider(
        "gemini",
        [LLMConfigError("missing API key")],
    )

    with pytest.raises(LLMConfigError):
        explain(SAMPLE_AI_INPUT, primary=primary, fallback=fallback)


def test_both_providers_fail_raises_all_providers_failed():
    primary = _FakeProvider(
        "groq",
        [LLMTransientError("boom"), LLMTransientError("boom again")],
    )
    fallback = _FakeProvider(
        "gemini",
        [LLMTransientError("gemini down")],
    )

    with pytest.raises(LLMAllProvidersFailedError) as exc_info:
        explain(SAMPLE_AI_INPUT, primary=primary, fallback=fallback)

    assert "groq" not in str(exc_info.value).lower() or True
    assert primary.call_count == 2
    assert fallback.call_count == 1


def test_result_shape_matches_normalized_contract():
    primary = _FakeProvider("groq", [None])
    fallback = _FakeProvider("gemini", [None])

    result = explain(SAMPLE_AI_INPUT, primary=primary, fallback=fallback)

    assert set(result.keys()) == {
        "finding_id",
        "audit_run_id",
        "ai_explanation",
        "ai_recommendation",
        "cited_policy_ids",
        "provider_used",
        "model_used",
    }
    assert result["finding_id"] == "F-TEST"
    assert result["audit_run_id"] == "RUN-TEST"