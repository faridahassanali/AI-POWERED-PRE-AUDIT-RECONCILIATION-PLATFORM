"""
Tests for engine.ai_orchestration.

The Stage 3 AI pipeline itself is tested separately in
test_ai_explanation_pipeline.py.

These tests focus only on the orchestration gate:

CONFIRMED findings enter the AI stage.

REVIEW and REJECTED findings are excluded.
"""

import copy
from pathlib import Path

from engine.ai_explanation_pipeline import (
    AIExplanationResult,
)
from engine.ai_orchestration import (
    run_ai_stage,
)
from engine.audit_pipeline import run_audit
from engine.finding_review import (
    confirm_finding,
    reject_finding,
)
from engine.llm.base import LLMExplanation
from engine.policy_registry import (
    load_policy_registry,
)


DATA_DIR = (
    Path(__file__).resolve().parent.parent / "data"
)


class _FakeProvider:
    """
    Minimal fake LLM provider for orchestration tests.
    """

    name = "fake-provider"
    model = "fake-model"

    def generate(self, ai_input):

        return LLMExplanation(
            finding_id=ai_input["finding_id"],
            audit_run_id=ai_input["audit_run_id"],
            ai_explanation=(
                "This finding violates the cited policy."
            ),
            ai_recommendation=(
                "Resolve the identified issue."
            ),
            cited_policy_ids=[
                ai_input["policy_context"][0]["policy_id"]
            ],
            provider_used=self.name,
            model_used=self.model,
        )


def test_ai_stage_processes_confirmed_findings_only():

    pipeline_result = run_audit()

    findings = [
        copy.deepcopy(
            pipeline_result.generated_findings[0]
        ),
        copy.deepcopy(
            pipeline_result.generated_findings[1]
        ),
    ]

    confirm_finding(
        findings[0],
        reviewed_by="test-reviewer",
    )

    reject_finding(
        findings[1],
        reviewed_by="test-reviewer",
    )

    registry = load_policy_registry(
        DATA_DIR
    )

    result = run_ai_stage(
        findings=findings,
        registry=registry,
        primary=_FakeProvider(),
        fallback=_FakeProvider(),
    )

    assert result.total_findings == 2
    assert result.confirmed_findings == 1
    assert result.successful_explanations == 1
    assert result.failed_explanations == 0

    assert len(result.results) == 1

    assert isinstance(
        result.results[0],
        AIExplanationResult,
    )

    assert result.results[0].succeeded is True

    assert (
        result.results[0].finding_id
        == findings[0]["finding_id"]
    )


def test_ai_stage_does_not_process_review_findings():

    pipeline_result = run_audit()

    findings = [
        copy.deepcopy(
            pipeline_result.generated_findings[0]
        )
    ]

    assert (
        findings[0]["finding_status"]
        == "REVIEW"
    )

    registry = load_policy_registry(
        DATA_DIR
    )

    result = run_ai_stage(
        findings=findings,
        registry=registry,
        primary=_FakeProvider(),
        fallback=_FakeProvider(),
    )

    assert result.total_findings == 1
    assert result.confirmed_findings == 0
    assert result.successful_explanations == 0
    assert result.failed_explanations == 0
    assert result.results == []


def test_ai_stage_does_not_process_rejected_findings():

    pipeline_result = run_audit()

    finding = copy.deepcopy(
        pipeline_result.generated_findings[0]
    )

    reject_finding(
        finding,
        reviewed_by="test-reviewer",
    )

    registry = load_policy_registry(
        DATA_DIR
    )

    result = run_ai_stage(
        findings=[finding],
        registry=registry,
        primary=_FakeProvider(),
        fallback=_FakeProvider(),
    )

    assert result.total_findings == 1
    assert result.confirmed_findings == 0
    assert result.successful_explanations == 0
    assert result.failed_explanations == 0
    assert result.results == []


def test_ai_stage_processes_only_confirmed_from_mixed_batch():

    pipeline_result = run_audit()

    findings = [
        copy.deepcopy(f)
        for f in pipeline_result.generated_findings[:4]
    ]

    confirm_finding(
        findings[0],
        reviewed_by="reviewer",
    )

    reject_finding(
        findings[1],
        reviewed_by="reviewer",
    )

    confirm_finding(
        findings[2],
        reviewed_by="reviewer",
    )

    reject_finding(
        findings[3],
        reviewed_by="reviewer",
    )

    registry = load_policy_registry(
        DATA_DIR
    )

    result = run_ai_stage(
        findings=findings,
        registry=registry,
        primary=_FakeProvider(),
        fallback=_FakeProvider(),
    )

    assert result.total_findings == 4
    assert result.confirmed_findings == 2
    assert result.successful_explanations == 2
    assert result.failed_explanations == 0

    assert len(result.results) == 2

    assert {
        result_item.finding_id
        for result_item in result.results
    } == {
        findings[0]["finding_id"],
        findings[2]["finding_id"],
    }