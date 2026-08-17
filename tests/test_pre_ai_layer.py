from engine.audit_pipeline import run_audit
from engine.audit_output import build_audit_output


def test_pre_ai_pipeline_produces_complete_audit_result():
    """
    Verify that the deterministic pre-AI pipeline produces
    all artifacts required by the AI layer.

    Findings remain in REVIEW status and therefore do not
    receive explanations before human review.
    """

    result = run_audit()

    # Core audit artifacts
    assert result.generated_findings
    assert result.expected_findings
    assert result.evaluation is not None
    assert result.report

    # Traceability
    assert result.audit_trace is not None
    assert result.audit_trace.audit_run_id

    assert result.audit_trace.completed_at is not None
    assert result.audit_trace.total_records_evaluated > 0

    assert (
        result.audit_trace.total_findings_generated
        == len(result.generated_findings)
    )

    # Controls executed
    assert set(result.audit_trace.controls_executed) == {
        "SCREENING_001",
        "RISK_001",
        "ARABIC_NAME_001",
        "DORMANT_001",
        "RECON_001",
    }


def test_every_finding_is_traceable():
    """
    Every generated finding must be linked to the audit run.

    Findings are still awaiting human review, so explanations
    must NOT exist at the pre-AI stage.
    """

    result = run_audit()

    findings_by_id = {
        finding["finding_id"]: finding
        for finding in result.generated_findings
    }

    assert len(findings_by_id) == len(
        result.generated_findings
    )

    # All generated findings must belong to this audit run.
    for finding in result.generated_findings:

        assert finding["audit_run_id"] == (
            result.audit_trace.audit_run_id
        )

        # Findings must remain pending human review.
        assert finding["finding_status"] == "REVIEW"

    # No explanations are generated before human confirmation.
    assert result.explanations == []


def test_explanations_preserve_deterministic_finding_information():
    """
    Explanations are intentionally absent during the pre-AI stage.

    Explanation generation happens only after a finding has
    been CONFIRMED by a human reviewer.
    """

    result = run_audit()

    # No finding has passed the human review gate yet.
    assert result.explanations == []

    # Findings themselves remain deterministic and traceable.
    for finding in result.generated_findings:

        assert finding["finding_status"] == "REVIEW"

        assert finding["audit_run_id"] == (
            result.audit_trace.audit_run_id
        )


def test_pre_ai_evaluation_is_complete():
    """
    Ground-truth evaluation must be completed before the AI layer.
    """

    result = run_audit()

    evaluation = result.evaluation

    assert evaluation is not None

    assert evaluation.true_positives == 223
    assert evaluation.false_positives == 0
    assert evaluation.false_negatives == 0

    assert evaluation.precision == 1.0
    assert evaluation.recall == 1.0
    assert evaluation.f1_score == 1.0


def test_audit_output_is_ready_for_ai_consumption():
    """
    Verify that the integrated audit output contains the
    structured deterministic information that the future AI
    layer will consume.

    Explanations are intentionally None because findings
    are still awaiting human review.
    """

    result = run_audit()

    output = build_audit_output(
        audit_trace=result.audit_trace,
        findings=result.generated_findings,
        explanations=result.explanations,
        evaluation=result.evaluation,
        report=result.report,
    )

    assert output.audit_run_id == (
        result.audit_trace.audit_run_id
    )

    assert (
        len(output.findings)
        == len(result.generated_findings)
    )

    assert output.audit_trace is result.audit_trace

    for item in output.findings:

        assert item.finding is not None

        # Explanation must not exist before human confirmation.
        assert item.explanation is None

        assert (
            item.finding["audit_run_id"]
            == output.audit_run_id
        )

        assert item.finding["finding_status"] == "REVIEW"


def test_ai_layer_must_not_be_required_for_deterministic_audit():
    """
    The pre-AI audit pipeline must remain fully functional
    without any LLM/AI dependency.

    AI/explanations are not required for the deterministic
    audit stage.
    """

    result = run_audit()

    assert result.generated_findings

    # No explanations before human review.
    assert result.explanations == []

    assert result.audit_trace.completed_at is not None

    # Deterministic evaluation is already complete.
    assert result.evaluation.true_positives == 223
    assert result.evaluation.false_positives == 0
    assert result.evaluation.false_negatives == 0