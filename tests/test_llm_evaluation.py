"""
Tests for engine.llm_evaluation -- uses fake providers throughout, so
this never makes a real Groq/Gemini call and needs no API keys.
"""

import json

import pytest

from engine.llm.base import LLMExplanation, LLMTransientError
from engine.llm_evaluation import (
    EvaluationReport,
    select_stratified_sample,
    run_evaluation,
    write_report,
)
from engine.policy_registry import load_policy_registry
from engine.data_loader import DATA_DIR


@pytest.fixture(scope="module")
def real_registry():
    return load_policy_registry(DATA_DIR)


class _AlwaysSucceedsProvider:
    name = "fake-primary"
    model = "fake-model"

    def generate(self, ai_input):
        return LLMExplanation(
            finding_id=ai_input["finding_id"],
            audit_run_id=ai_input["audit_run_id"],
            ai_explanation="Explanation grounded in the cited policy.",
            ai_recommendation="Take a concrete remediation step.",
            cited_policy_ids=[ai_input["policy_context"][0]["policy_id"]],
            provider_used=self.name,
            model_used=self.model,
        )


class _AlwaysFailsProvider:
    name = "fake-broken"
    model = "fake-model"

    def generate(self, ai_input):
        raise LLMTransientError("simulated outage")


# =====================================================================
# select_stratified_sample
# =====================================================================

def test_stratified_sample_caps_per_control():
    findings = (
        [{"control_id": "SCREENING_001", "finding_id": f"S-{i}"} for i in range(20)]
        + [{"control_id": "RISK_001", "finding_id": f"R-{i}"} for i in range(3)]
    )

    sample = select_stratified_sample(findings, per_control=5)

    screening = [f for f in sample if f["control_id"] == "SCREENING_001"]
    risk = [f for f in sample if f["control_id"] == "RISK_001"]

    assert len(screening) == 5
    assert len(risk) == 3  # fewer available than the cap -- takes all of them


def test_stratified_sample_covers_every_control():
    findings = [
        {"control_id": "A", "finding_id": "a1"},
        {"control_id": "B", "finding_id": "b1"},
        {"control_id": "C", "finding_id": "c1"},
    ]

    sample = select_stratified_sample(findings, per_control=5)

    assert {f["control_id"] for f in sample} == {"A", "B", "C"}


# =====================================================================
# run_evaluation -- end-to-end with fakes, against the real registry
# and real deterministic pipeline (run_audit / confirm_finding are
# both fast and make no network calls).
# =====================================================================

def test_run_evaluation_all_succeed(real_registry):

    report = run_evaluation(
        per_control=1,
        registry=real_registry,
        primary=_AlwaysSucceedsProvider(),
        delay_between_calls=0,
    )

    assert report.total > 0
    assert report.success_rate() == 1.0
    assert report.failed == []
    assert report.provider_distribution() == {"fake-primary": report.total}


def test_run_evaluation_all_fail_over_to_gemini_shape(real_registry):
    """
    Primary always fails (transient) -- router retries then fails
    over to `fallback`. `fallback` is also injected as a fake that
    always fails, so every item ends up in `failed`, with a clear
    error, not a crash.

    Deliberately does NOT rely on GEMINI_API_KEY being absent from
    the environment to make the fallback fail -- if a real .env with
    a valid key is present (as it will be on a developer machine, not
    just CI), the real GeminiProvider would genuinely succeed, making
    this test's outcome depend on whatever happens to be in the
    environment. Injecting both fakes keeps it deterministic.
    """

    report = run_evaluation(
        per_control=1,
        registry=real_registry,
        primary=_AlwaysFailsProvider(),
        fallback=_AlwaysFailsProvider(),
        delay_between_calls=0,
    )

    assert report.total > 0
    assert report.success_rate() == 0.0
    assert all(not item.succeeded for item in report.items)
    assert all(item.error_type for item in report.items)


def test_report_aggregates_are_consistent(real_registry):

    report = run_evaluation(
        per_control=1,
        registry=real_registry,
        primary=_AlwaysSucceedsProvider(),
        delay_between_calls=0,
    )

    breakdown = report.per_control_breakdown()
    total_from_breakdown = sum(
        counts["succeeded"] + counts["failed"] for counts in breakdown.values()
    )

    assert total_from_breakdown == report.total


# =====================================================================
# write_report -- valid JSON/Markdown on disk
# =====================================================================

def test_write_report_produces_valid_json_and_markdown(tmp_path, real_registry):

    report = run_evaluation(
        per_control=1,
        registry=real_registry,
        primary=_AlwaysSucceedsProvider(),
        delay_between_calls=0,
    )

    json_path, md_path = write_report(report, output_dir=tmp_path)

    assert json_path.exists()
    assert md_path.exists()

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["total"] == report.total
    assert payload["success_rate"] == report.success_rate()
    assert len(payload["items"]) == report.total

    markdown = md_path.read_text(encoding="utf-8")
    assert "# LLM/RAG Quality Evaluation" in markdown
    assert "Manual review" in markdown