"""
Tests for engine.ai_explanation_pipeline -- the Stage 3 wiring that
connects Task A (engine.llm.router.explain) to Task B
(engine.ai_output_validation) for CONFIRMED findings.

Uses fake LLMProvider implementations throughout (same pattern as
tests/test_llm_router.py) -- no real network calls, no real API keys
needed.
"""

import copy

import pytest

from engine.ai_explanation_pipeline import (
    AIExplanationResult,
    generate_ai_explanation_for_finding,
    generate_ai_explanations,
)
from engine.audit_pipeline import run_audit
from engine.finding_review import confirm_finding
from engine.llm.base import LLMExplanation, LLMOutputError, LLMTransientError
from engine.policy_registry import load_policy_registry
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@pytest.fixture(scope="module")
def real_registry():
    return load_policy_registry(DATA_DIR)


@pytest.fixture
def confirmed_finding():
    """A real, deterministically-generated finding, confirmed."""
    result = run_audit()
    return confirm_finding(
        copy.deepcopy(result.generated_findings[0]), reviewed_by="test"
    )


class _FakeProvider:
    """A minimal LLMProvider -- always succeeds with a fixed, valid
    explanation unless configured otherwise."""

    name = "fake-provider"
    model = "fake-model"

    def __init__(self, cited_policy_ids=None, raises=None):
        self._cited_policy_ids = cited_policy_ids
        self._raises = raises

    def generate(self, ai_input):
        if self._raises is not None:
            raise self._raises

        cited = self._cited_policy_ids
        if cited is None:
            cited = [ai_input["policy_context"][0]["policy_id"]]

        return LLMExplanation(
            finding_id=ai_input["finding_id"],
            audit_run_id=ai_input["audit_run_id"],
            ai_explanation="This finding violates the cited policy.",
            ai_recommendation="Resolve the missing evidence.",
            cited_policy_ids=cited,
            provider_used=self.name,
            model_used=self.model,
        )


# =====================================================================
# Happy path -- success attaches ai_explanation/ai_recommendation
# =====================================================================

def test_success_attaches_explanation_to_the_finding(confirmed_finding, real_registry):
    result = generate_ai_explanation_for_finding(
        confirmed_finding,
        registry=real_registry,
        primary=_FakeProvider(),
        fallback=_FakeProvider(),
    )

    assert result.succeeded is True
    assert result.error is None
    assert confirmed_finding["ai_explanation"] == "This finding violates the cited policy."
    assert confirmed_finding["ai_recommendation"] == "Resolve the missing evidence."


def test_success_mutates_finding_in_place(confirmed_finding, real_registry):
    """Same convention as confirm_finding()/reject_finding() -- the
    caller's original dict is updated, not just the returned copy."""
    generate_ai_explanation_for_finding(
        confirmed_finding,
        registry=real_registry,
        primary=_FakeProvider(),
        fallback=_FakeProvider(),
    )

    assert confirmed_finding["ai_explanation"] is not None


def test_result_carries_the_raw_ai_output(confirmed_finding, real_registry):
    result = generate_ai_explanation_for_finding(
        confirmed_finding,
        registry=real_registry,
        primary=_FakeProvider(),
        fallback=_FakeProvider(),
    )

    assert result.ai_output["provider_used"] == "fake-provider"
    assert result.ai_output["finding_id"] == confirmed_finding["finding_id"]


# =====================================================================
# Gate enforcement -- non-CONFIRMED findings never reach the LLM
# =====================================================================

def test_review_status_finding_fails_before_any_llm_call(real_registry):
    result_pipeline = run_audit()
    review_finding = copy.deepcopy(result_pipeline.generated_findings[0])
    assert review_finding["finding_status"] == "REVIEW"

    called = {"count": 0}

    class _Spy(_FakeProvider):
        def generate(self, ai_input):
            called["count"] += 1
            return super().generate(ai_input)

    result = generate_ai_explanation_for_finding(
        review_finding,
        registry=real_registry,
        primary=_Spy(),
        fallback=_Spy(),
    )

    assert result.succeeded is False
    assert "CONFIRMED" in result.error
    assert called["count"] == 0


# =====================================================================
# LLM layer failure -- caught, recorded, finding left unmodified
# =====================================================================

def test_llm_failure_is_caught_and_recorded(confirmed_finding, real_registry):
    result = generate_ai_explanation_for_finding(
        confirmed_finding,
        registry=real_registry,
        primary=_FakeProvider(raises=LLMTransientError("simulated outage")),
        fallback=_FakeProvider(raises=LLMTransientError("simulated outage")),
    )

    assert result.succeeded is False
    assert "LLMAllProvidersFailedError" in result.error or "LLM" in result.error
    assert confirmed_finding.get("ai_explanation") is None


def test_llm_output_error_falls_back_and_still_succeeds(confirmed_finding, real_registry):
    """Sanity check that the router's own retry/failover still works
    when called through this module -- primary fails, fallback saves it."""
    result = generate_ai_explanation_for_finding(
        confirmed_finding,
        registry=real_registry,
        primary=_FakeProvider(raises=LLMOutputError("malformed once")),
        fallback=_FakeProvider(),
    )

    assert result.succeeded is True
    assert result.ai_output["provider_used"] == "fake-provider"


# =====================================================================
# Output validation failure -- caught, recorded, finding left unmodified
# =====================================================================

def test_citation_outside_registry_is_caught_by_validation(confirmed_finding, real_registry):
    """The model 'cites' a policy_id that doesn't exist anywhere in
    the registry -- Task B's grounding check must catch this even
    though the router's own validate_citations() only checks against
    policy_context, not the full registry."""
    result = generate_ai_explanation_for_finding(
        confirmed_finding,
        registry=real_registry,
        primary=_FakeProvider(cited_policy_ids=["TOTALLY-MADE-UP-POLICY"]),
        fallback=_FakeProvider(cited_policy_ids=["TOTALLY-MADE-UP-POLICY"]),
    )

    assert result.succeeded is False
    assert confirmed_finding.get("ai_explanation") is None


# =====================================================================
# Batch behavior -- one failure never stops the rest
# =====================================================================

def test_batch_one_failure_does_not_stop_the_others(real_registry):
    result_pipeline = run_audit()
    by_control = {}
    for f in result_pipeline.generated_findings:
        by_control.setdefault(f["control_id"], f)

    findings = [
        confirm_finding(copy.deepcopy(f), reviewed_by="test")
        for f in list(by_control.values())[:3]
    ]

    # Make the SECOND finding fail; first and third should still succeed.
    providers = [_FakeProvider(), _FakeProvider(raises=LLMTransientError("down")), _FakeProvider()]
    fallbacks = [_FakeProvider(), _FakeProvider(raises=LLMTransientError("down")), _FakeProvider()]

    results = [
        generate_ai_explanation_for_finding(
            f, registry=real_registry, primary=p, fallback=fb
        )
        for f, p, fb in zip(findings, providers, fallbacks)
    ]

    assert [r.succeeded for r in results] == [True, False, True]


def test_generate_ai_explanations_batch_wrapper(real_registry):
    result_pipeline = run_audit()
    findings = [
        confirm_finding(copy.deepcopy(result_pipeline.generated_findings[0]), reviewed_by="test")
    ]

    results = generate_ai_explanations(
        findings,
        registry=real_registry,
        primary=_FakeProvider(),
        fallback=_FakeProvider(),
    )

    assert len(results) == 1
    assert isinstance(results[0], AIExplanationResult)
    assert results[0].succeeded is True
