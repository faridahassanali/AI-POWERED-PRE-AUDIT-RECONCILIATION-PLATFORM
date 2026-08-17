from engine.evaluation_report import (
    generate_evaluation_report,
)
from engine.ground_truth_evaluator import (
    evaluate_findings,
)


def make_finding(
    control_id,
    customer_id,
    severity="HIGH",
):
    return {
        "finding_id": "F-TEST",
        "audit_run_id": "RUN-TEST",
        "control_id": control_id,
        "customer_id": customer_id,
        "severity": severity,
        "assessment_status": "FAIL",
        "finding_status": "REVIEW",
        "expected": "Expected",
        "actual": "Actual",
        "evidence": {},
        "policy_references": [],
    }


def test_evaluation_report_contains_metrics():

    expected = [
        make_finding(
            "SCREENING_001",
            "CUST001",
        )
    ]

    generated = [
        make_finding(
            "SCREENING_001",
            "CUST001",
        )
    ]

    result = evaluate_findings(
        generated,
        expected,
    )

    report = generate_evaluation_report(result)

    assert "GROUND TRUTH EVALUATION REPORT" in report
    assert "Precision" in report
    assert "Recall" in report
    assert "F1 Score" in report
    assert "SCREENING_001" in report
    assert "100.00%" in report