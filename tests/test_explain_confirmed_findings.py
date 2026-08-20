"""
Tests for engine.audit_pipeline.explain_confirmed_findings() -- Stage 2
of the pipeline.

This function had ZERO test coverage before: tests/test_ai_layer_gate.py
tests engine.finding_explainer.explain_finding() directly, never this
orchestration function, and nothing in the codebase called it either
(confirmed via grep before writing this file). It's the function that
wires human review -> RAG retrieval -> the AI Input Contract together,
so it deserves its own coverage.
"""

import copy

import pytest

from engine.ai_input import AIInputValidationError
from engine.audit_pipeline import explain_confirmed_findings, run_audit
from engine.finding_review import confirm_finding, reject_finding
from engine.policy_registry import load_policy_registry
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@pytest.fixture(scope="module")
def real_registry():
    return load_policy_registry(DATA_DIR)


@pytest.fixture(scope="module")
def one_finding_per_control():
    """
    One real, deterministically-generated finding per control_id.

    Module-scoped for speed (run_audit() reads real data files), but
    confirm_finding()/reject_finding() mutate their argument IN PLACE
    -- so every test that reviews one of these findings must
    copy.deepcopy() it first, or later tests in this module will see
    an already-reviewed finding and fail with "must have status REVIEW".
    """
    result = run_audit()
    by_control: dict[str, dict] = {}
    for finding in result.generated_findings:
        by_control.setdefault(finding["control_id"], finding)
    return list(by_control.values())


# =====================================================================
# Happy path -- every control type resolves and explains successfully
# =====================================================================

def test_explains_a_confirmed_finding_for_every_control_type(
    one_finding_per_control, real_registry
):
    confirmed = [
        confirm_finding(copy.deepcopy(f), reviewed_by="tester")
        for f in one_finding_per_control
    ]

    explanations = explain_confirmed_findings(confirmed, registry=real_registry)

    assert len(explanations) == len(confirmed)
    explained_ids = {e["finding_id"] for e in explanations}
    confirmed_ids = {f["finding_id"] for f in confirmed}
    assert explained_ids == confirmed_ids


def test_explanation_carries_policy_references_for_every_control(
    one_finding_per_control, real_registry
):
    """
    Regression test for the ARABIC_NAME_001 bug: engine/controls.py used
    to hardcode a stale section name ("Customer Data") that didn't exist
    in the actual policy file, so this control's policy reference could
    never resolve. If it silently regresses again, this test fails loud
    instead of the finding silently reaching this stage with no policy
    grounding.
    """
    confirmed = [
        confirm_finding(copy.deepcopy(f), reviewed_by="tester")
        for f in one_finding_per_control
    ]

    explanations = explain_confirmed_findings(confirmed, registry=real_registry)

    for explanation in explanations:
        assert explanation["policy_references"], (
            f"{explanation['control_id']} produced no resolved policy "
            "references -- its policy_references likely don't match "
            "the registry (policy_id/version/section)."
        )


# =====================================================================
# Gate enforcement -- non-CONFIRMED findings must never reach this stage
# =====================================================================

def test_review_finding_is_blocked(one_finding_per_control, real_registry):
    finding = copy.deepcopy(one_finding_per_control[0])  # still REVIEW

    with pytest.raises(AIInputValidationError, match="CONFIRMED"):
        explain_confirmed_findings([finding], registry=real_registry)


def test_rejected_finding_is_blocked(one_finding_per_control, real_registry):
    finding = reject_finding(
        copy.deepcopy(one_finding_per_control[0]), reviewed_by="tester"
    )

    with pytest.raises(AIInputValidationError, match="CONFIRMED"):
        explain_confirmed_findings([finding], registry=real_registry)


# =====================================================================
# A finding whose policy reference doesn't resolve must BLOCK, not skip
# =====================================================================

def test_unresolvable_policy_reference_blocks_rather_than_skips(real_registry):
    finding = confirm_finding(
        {
            "finding_id": "F-TEST-UNRESOLVED",
            "audit_run_id": "RUN-TEST",
            "control_id": "FAKE_CONTROL",
            "customer_id": "C-999",
            "severity": "HIGH",
            "assessment_status": "FAIL",
            "finding_status": "REVIEW",
            "expected": "n/a",
            "actual": "n/a",
            "evidence": {"some_field": "some_value"},
            "policy_references": [
                {
                    "policy_id": "DOES-NOT-EXIST-POLICY",
                    "version": "1.0",
                    "section": "Requirements",
                }
            ],
            "reviewed_by": None,
            "review_timestamp": None,
            "reviewer_notes": None,
        },
        reviewed_by="tester",
    )

    # RAG.retriever.resolve_policy_references() raises ValueError for
    # an unknown policy_id -- this must propagate, not be swallowed
    # into an empty (and therefore silently allowed) policy_context.
    with pytest.raises(ValueError):
        explain_confirmed_findings([finding], registry=real_registry)
