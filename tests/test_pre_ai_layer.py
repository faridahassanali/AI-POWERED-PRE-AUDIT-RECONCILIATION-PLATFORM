from engine.audit_pipeline import run_audit
from engine.audit_output import build_audit_output


def test_pre_ai_pipeline_produces_complete_audit_result():
    """
    Verify that the deterministic pre-AI pipeline produces
    all artifacts required by the AI layer.
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
    Every generated finding must be linked to the audit run
    and have a corresponding explanation.
    """

    result = run_audit()

    findings_by_id = {
        finding["finding_id"]: finding
        for finding in result.generated_findings
    }

    explanations_by_id = {
        explanation["finding_id"]: explanation
        for explanation in result.explanations
    }

    assert len(findings_by_id) == len(result.generated_findings)
    assert len(explanations_by_id) == len(result.explanations)

    for finding in result.generated_findings:

        finding_id = finding["finding_id"]

        assert finding["audit_run_id"] == (
            result.audit_trace.audit_run_id
        )

        assert finding_id in explanations_by_id

        explanation = explanations_by_id[finding_id]

        assert explanation["finding_id"] == finding_id
        assert explanation["audit_run_id"] == (
            result.audit_trace.audit_run_id
        )
        assert explanation["control_id"] == finding["control_id"]
        assert explanation["customer_id"] == finding["customer_id"]


def test_explanations_preserve_deterministic_finding_information():
    """
    Explainability must describe the finding without changing
    the underlying deterministic decision.
    """

    result = run_audit()

    findings_by_id = {
        finding["finding_id"]: finding
        for finding in result.generated_findings
    }

    for explanation in result.explanations:

        finding = findings_by_id[explanation["finding_id"]]

        assert (
            explanation["control_id"]
            == finding["control_id"]
        )

        assert (
            explanation["customer_id"]
            == finding["customer_id"]
        )

        assert (
            explanation["severity"]
            == finding["severity"]
        )

        assert (
            explanation["assessment_status"]
            == finding["assessment_status"]
        )

        assert (
            explanation["finding_status"]
            == finding["finding_status"]
        )

        assert (
            explanation["expected_condition"]
            == finding["expected"]
        )

        assert (
            explanation["observed_condition"]
            == finding["actual"]
        )

        assert (
            explanation["evidence"]
            == finding["evidence"]
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
    structured information that the future AI layer will consume.
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
        assert item.explanation is not None

        assert (
            item.finding["finding_id"]
            == item.explanation["finding_id"]
        )

        assert (
            item.finding["audit_run_id"]
            == output.audit_run_id
        )

        assert (
            item.explanation["audit_run_id"]
            == output.audit_run_id
        )


def test_ai_layer_must_not_be_required_for_deterministic_audit():
    """
    The pre-AI audit pipeline must remain fully functional
    without any LLM/AI dependency.

    This guarantees that AI will be an enhancement layer,
    not the source of compliance decisions.
    """

    result = run_audit()

    assert result.generated_findings
    assert result.explanations
    assert result.audit_trace.completed_at is not None

    # Deterministic evaluation is already complete.
    assert result.evaluation.true_positives == 223
    assert result.evaluation.false_positives == 0
    assert result.evaluation.false_negatives == 0