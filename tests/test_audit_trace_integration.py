from engine.audit_pipeline import run_audit


def test_audit_pipeline_contains_audit_trace():
    result = run_audit()

    assert result.audit_trace is not None
    assert result.audit_trace.audit_run_id
    assert result.audit_trace.started_at
    assert result.audit_trace.completed_at


def test_audit_trace_contains_correct_counts():
    result = run_audit()

    assert (
        result.audit_trace.total_records_evaluated
        > 0
    )

    assert (
        result.audit_trace.total_findings_generated
        == len(result.generated_findings)
    )


def test_audit_trace_contains_executed_controls():
    result = run_audit()

    expected_controls = {
        "SCREENING_001",
        "RISK_001",
        "ARABIC_NAME_001",
        "DORMANT_001",
        "RECON_001",
    }

    assert set(
        result.audit_trace.controls_executed
    ) == expected_controls


def test_findings_are_linked_to_audit_run():
    result = run_audit()

    audit_run_id = result.audit_trace.audit_run_id

    for finding in result.generated_findings:
        assert finding["audit_run_id"] == audit_run_id


def test_every_finding_has_explanation():
    result = run_audit()

    assert len(result.explanations) == len(
        result.generated_findings
    )

    for explanation in result.explanations:
        assert explanation["finding_id"] is not None
        assert explanation["audit_run_id"] == (
            result.audit_trace.audit_run_id
        )
        assert explanation["control_id"]
        assert "summary" in explanation
        assert "expected_condition" in explanation
        assert "observed_condition" in explanation
        assert "evidence" in explanation
        assert "policy_references" in explanation
        assert "review_action" in explanation


def test_explanations_match_generated_findings():
    result = run_audit()

    for finding, explanation in zip(
        result.generated_findings,
        result.explanations,
    ):
        assert (
            explanation["finding_id"]
            == finding["finding_id"]
        )

        assert (
            explanation["audit_run_id"]
            == finding["audit_run_id"]
        )

        assert (
            explanation["control_id"]
            == finding["control_id"]
        )

        assert (
            explanation["customer_id"]
            == finding["customer_id"]
        )