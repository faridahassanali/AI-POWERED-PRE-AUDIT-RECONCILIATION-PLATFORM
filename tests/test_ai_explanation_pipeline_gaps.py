"""
Additional integration tests for engine.ai_explanation_pipeline --
the full AI path: gate -> RAG -> LLM -> validation.

Complements tests/test_ai_explanation_pipeline.py (Farida's original
wiring tests). Reuses the same fixtures/fake-provider pattern so both
files stay consistent -- this file only adds the paths that weren't
covered yet:

    1. A REJECTED finding is blocked before any LLM call (the
       original file only covers REVIEW).
    2. An empty RAG policy_context blocks the finding before any
       LLM call (the gate must never send a finding with nothing to
       ground on).
    3. A finding mutated during the AI call is caught by the
       no-mutation check, at the full-pipeline level (not just the
       ai_output_validation unit tests).
    4. A citation that exists in the registry but was never part of
       this finding's own policy_context is caught as invented
       evidence, at the full-pipeline level.
    5. A schema-invalid LLM output (empty ai_explanation) is caught
       by the schema check before it's ever attached to the finding.
"""

import copy

import pytest

from engine.ai_explanation_pipeline import (
    generate_ai_explanation_for_finding,
)
from engine.audit_pipeline import run_audit
from engine.finding_review import confirm_finding, reject_finding
from engine.llm.base import LLMExplanation
from engine.policy_registry import load_policy_registry
from RAG.retriever import retrieve_for_finding
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
    """Same minimal fake provider as test_ai_explanation_pipeline.py."""

    name = "fake-provider"
    model = "fake-model"

    def __init__(self, cited_policy_ids=None, raises=None,
                 ai_explanation=None, mutate_finding=None):
        self._cited_policy_ids = cited_policy_ids
        self._raises = raises
        self._ai_explanation = ai_explanation
        self._mutate_finding = mutate_finding

    def generate(self, ai_input):
        if self._raises is not None:
            raise self._raises

        # Simulates a bug elsewhere touching the same finding object
        # while the AI call is in flight -- used only by the
        # mutation-check test below.
        if self._mutate_finding is not None:
            self._mutate_finding()

        cited = self._cited_policy_ids
        if cited is None:
            cited = [ai_input["policy_context"][0]["policy_id"]]

        return LLMExplanation(
            finding_id=ai_input["finding_id"],
            audit_run_id=ai_input["audit_run_id"],
            ai_explanation=(
                self._ai_explanation
                if self._ai_explanation is not None
                else "This finding violates the cited policy."
            ),
            ai_recommendation="Resolve the missing evidence.",
            cited_policy_ids=cited,
            provider_used=self.name,
            model_used=self.model,
        )


# =====================================================================
# 1. REJECTED finding -- blocked before any LLM call
# =====================================================================

def test_rejected_finding_fails_before_any_llm_call(real_registry):
    """
    A REJECTED finding must never reach the LLM, same as REVIEW.
    The original test file only checks REVIEW -- REJECTED is a
    distinct finding_status and needs its own guard test.
    """

    result_pipeline = run_audit()
    rejected_finding = reject_finding(
        copy.deepcopy(result_pipeline.generated_findings[0]),
        reviewed_by="test",
    )
    assert rejected_finding["finding_status"] == "REJECTED"

    called = {"count": 0}

    class _Spy(_FakeProvider):
        def generate(self, ai_input):
            called["count"] += 1
            return super().generate(ai_input)

    result = generate_ai_explanation_for_finding(
        rejected_finding,
        registry=real_registry,
        primary=_Spy(),
        fallback=_Spy(),
    )

    assert result.succeeded is False
    assert called["count"] == 0
    assert rejected_finding.get("ai_explanation") is None


# =====================================================================
# 2. Empty RAG policy_context -- blocked before any LLM call
# =====================================================================

def test_empty_policy_context_fails_before_any_llm_call(
    confirmed_finding, real_registry, monkeypatch
):
    """
    If RAG resolves no policy_context at all for this finding, the
    gate in build_ai_input() must block it -- the LLM must never be
    called with nothing to ground on.
    """

    import engine.ai_explanation_pipeline as pipeline_module

    monkeypatch.setattr(
        pipeline_module,
        "retrieve_for_finding",
        lambda finding, registry: [],
    )

    called = {"count": 0}

    class _Spy(_FakeProvider):
        def generate(self, ai_input):
            called["count"] += 1
            return super().generate(ai_input)

    result = generate_ai_explanation_for_finding(
        confirmed_finding,
        registry=real_registry,
        primary=_Spy(),
        fallback=_Spy(),
    )

    assert result.succeeded is False
    assert called["count"] == 0
    assert confirmed_finding.get("ai_explanation") is None


# =====================================================================
# 3. Finding mutated during the AI call -- caught by no-mutation check
# =====================================================================

def test_finding_mutated_during_ai_call_is_caught(
    confirmed_finding, real_registry
):
    """
    Simulates a bug where something touches the same finding object
    while the AI call is in flight (e.g. a stray write from another
    part of the system). The no-mutation check must catch the
    severity change even though the LLM output itself is otherwise
    perfectly valid, and the finding must be left without an
    ai_explanation attached.
    """

    def _mutate():
        confirmed_finding["severity"] = "LOW"

    result = generate_ai_explanation_for_finding(
        confirmed_finding,
        registry=real_registry,
        primary=_FakeProvider(mutate_finding=_mutate),
        fallback=_FakeProvider(mutate_finding=_mutate),
    )

    assert result.succeeded is False
    assert "severity" in result.error
    assert confirmed_finding.get("ai_explanation") is None


# =====================================================================
# 4. Invented evidence -- real registry policy, but not this
#    finding's own policy_context
# =====================================================================

def test_citation_outside_own_policy_context_is_caught(
    confirmed_finding, real_registry
):
    """
    The model cites a policy_id that is real and exists in the
    registry, but was never part of THIS finding's resolved
    policy_context -- i.e. the model cited something it wasn't
    actually given. This must be caught independently of the
    citation-outside-registry case already covered in the original
    test file.

    NOTE on a bug fixed here: `confirmed_finding` (straight out of
    run_audit() + confirm_finding()) never has a `policy_context` key
    at all -- that's only added later, inside
    generate_ai_explanation_for_finding() itself, via
    retrieve_for_finding(). Reading `confirmed_finding.get(
    "policy_context", [])` here always returned [], so nothing was
    ever actually excluded. Combined with PolicyRegistry.ids()
    returning a `set` (whose iteration order is randomized per-process
    by Python's string hash randomization), the "unrelated" policy_id
    picked below was effectively RANDOM every run -- occasionally
    coinciding with the finding's own real policy_id and making this
    test intermittently, non-deterministically fail. Fixed by
    resolving the finding's REAL policy_context the same way the
    pipeline does, and picking from a sorted (deterministic) id list.
    """

    real_policy_context = retrieve_for_finding(
        finding=confirmed_finding, registry=real_registry
    )
    own_policy_ids = {chunk["policy_id"] for chunk in real_policy_context}

    # sorted() for a deterministic pick -- no more dependence on set
    # iteration order / hash randomization.
    unrelated_policy_id = next(
        pid for pid in sorted(real_registry.ids()) if pid not in own_policy_ids
    )

    result = generate_ai_explanation_for_finding(
        confirmed_finding,
        registry=real_registry,
        primary=_FakeProvider(cited_policy_ids=[unrelated_policy_id]),
        fallback=_FakeProvider(cited_policy_ids=[unrelated_policy_id]),
    )

    assert result.succeeded is False
    assert confirmed_finding.get("ai_explanation") is None


# =====================================================================
# 5. Schema-invalid LLM output -- caught before attach
# =====================================================================

def test_empty_ai_explanation_is_caught_by_schema_check(
    confirmed_finding, real_registry
):
    """
    LLMExplanation itself doesn't enforce non-empty strings (it's a
    plain dataclass) -- ai_output_validation's schema check is the
    actual enforcement point. This confirms that enforcement holds
    through the full pipeline, not just in the validator's own unit
    tests.
    """

    result = generate_ai_explanation_for_finding(
        confirmed_finding,
        registry=real_registry,
        primary=_FakeProvider(ai_explanation=""),
        fallback=_FakeProvider(ai_explanation=""),
    )

    assert result.succeeded is False
    assert confirmed_finding.get("ai_explanation") is None