"""
End-to-end tests for the complete audit pipeline.
"""

from engine.audit_pipeline import run_audit


def test_audit_pipeline_runs_end_to_end():

    result = run_audit()

    assert result is not None

    assert isinstance(
        result.generated_findings,
        list,
    )

    assert isinstance(
        result.expected_findings,
        list,
    )

    assert result.evaluation is not None

    assert isinstance(
        result.report,
        str,
    )


def test_audit_pipeline_produces_valid_metrics():

    result = run_audit()

    evaluation = result.evaluation

    assert evaluation.true_positives >= 0
    assert evaluation.false_positives >= 0
    assert evaluation.false_negatives >= 0

    assert 0.0 <= evaluation.precision <= 1.0
    assert 0.0 <= evaluation.recall <= 1.0
    assert 0.0 <= evaluation.f1_score <= 1.0


def test_audit_pipeline_report_contains_metrics():

    result = run_audit()

    report = result.report

    assert "GROUND TRUTH EVALUATION REPORT" in report
    assert "True Positives" in report
    assert "False Positives" in report
    assert "False Negatives" in report
    assert "Precision" in report
    assert "Recall" in report
    assert "F1 Score" in report


def test_generated_findings_have_required_fields():

    result = run_audit()

    required_fields = {
        "finding_id",
        "audit_run_id",
        "control_id",
        "customer_id",
        "severity",
        "assessment_status",
        "finding_status",
        "expected",
        "actual",
        "evidence",
        "policy_references",
    }

    for finding in result.generated_findings:

        assert required_fields.issubset(
            finding.keys()
        )


def test_pipeline_evaluation_counts_are_consistent():

    result = run_audit()

    evaluation = result.evaluation

    assert (
        evaluation.true_positives
        + evaluation.false_positives
        == len(
            {
                (
                    finding["control_id"],
                    finding.get("customer_id"),
                )
                for finding in result.generated_findings
            }
        )
    )

    assert (
        evaluation.true_positives
        + evaluation.false_negatives
        == len(
            {
                (
                    finding["control_id"],
                    finding.get("customer_id"),
                )
                for finding in result.expected_findings
            }
        )
    )
def test_run_audit_returns_failed_result_instead_of_raising(monkeypatch):
    """
    A failure anywhere in the pipeline (e.g. data loading) must not
    raise out of run_audit() -- it must be captured on the trace and
    returned as a FAILED AuditPipelineResult, so a failed run is
    never silently lost.
    """
    def _broken_load_data(*args, **kwargs):
        raise FileNotFoundError("simulated missing data file")

    monkeypatch.setattr(
        "engine.audit_pipeline.load_data",
        _broken_load_data,
    )

    result = run_audit()

    assert result is not None
    assert result.audit_trace.status == "FAILED"
    assert result.audit_trace.error_type == "FileNotFoundError"
    assert "simulated missing data file" in result.audit_trace.error_message
    assert result.generated_findings == []
    assert result.evaluation is not None    