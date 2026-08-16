from engine.ground_truth_evaluator import (
    FindingKey,
    evaluate_findings,
)


def make_finding(
    control_id,
    customer_id,
    severity="HIGH",
):
    return {
        "finding_id": "RANDOM-ID",
        "audit_run_id": "RUN-TEST",
        "control_id": control_id,
        "customer_id": customer_id,
        "severity": severity,
        "assessment_status": "FAIL",
        "finding_status": "REVIEW",
        "expected": "Expected condition",
        "actual": "Actual condition",
        "evidence": {},
        "policy_references": [],
    }


def test_perfect_match():
    expected = [
        make_finding(
            "SCREENING_001",
            "CUST001",
            "HIGH",
        ),
        make_finding(
            "RISK_001",
            "CUST002",
            "HIGH",
        ),
    ]

    generated = [
        make_finding(
            "SCREENING_001",
            "CUST001",
            "HIGH",
        ),
        make_finding(
            "RISK_001",
            "CUST002",
            "HIGH",
        ),
    ]

    result = evaluate_findings(
        generated,
        expected,
    )

    assert result.true_positives == 2
    assert result.false_positives == 0
    assert result.false_negatives == 0

    assert result.precision == 1.0
    assert result.recall == 1.0
    assert result.f1_score == 1.0


def test_false_positive_is_detected():
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
        ),
        make_finding(
            "RISK_001",
            "CUST999",
        ),
    ]

    result = evaluate_findings(
        generated,
        expected,
    )

    assert result.true_positives == 1
    assert result.false_positives == 1
    assert result.false_negatives == 0

    assert result.precision == 0.5
    assert result.recall == 1.0


def test_false_negative_is_detected():
    expected = [
        make_finding(
            "SCREENING_001",
            "CUST001",
        ),
        make_finding(
            "RISK_001",
            "CUST002",
        ),
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

    assert result.true_positives == 1
    assert result.false_positives == 0
    assert result.false_negatives == 1

    assert result.precision == 1.0
    assert result.recall == 0.5


def test_finding_id_does_not_affect_matching():
    expected = [
        {
            **make_finding(
                "SCREENING_001",
                "CUST001",
            ),
            "finding_id": "EXPECTED-123",
        }
    ]

    generated = [
        {
            **make_finding(
                "SCREENING_001",
                "CUST001",
            ),
            "finding_id": "GENERATED-999",
        }
    ]

    result = evaluate_findings(
        generated,
        expected,
    )

    assert result.true_positives == 1
    assert result.false_positives == 0
    assert result.false_negatives == 0


def test_control_is_part_of_finding_identity():
    expected = [
        make_finding(
            "SCREENING_001",
            "CUST001",
        )
    ]

    generated = [
        make_finding(
            "RISK_001",
            "CUST001",
        )
    ]

    result = evaluate_findings(
        generated,
        expected,
    )

    assert result.true_positives == 0
    assert result.false_positives == 1
    assert result.false_negatives == 1


def test_customer_id_none_is_supported():
    expected = [
        make_finding(
            "RECON_001",
            None,
        )
    ]

    generated = [
        make_finding(
            "RECON_001",
            None,
        )
    ]

    result = evaluate_findings(
        generated,
        expected,
    )

    assert result.true_positives == 1
    assert result.false_positives == 0
    assert result.false_negatives == 0


def test_per_control_metrics():
    expected = [
        make_finding(
            "SCREENING_001",
            "CUST001",
        ),
        make_finding(
            "SCREENING_001",
            "CUST002",
        ),
        make_finding(
            "RISK_001",
            "CUST003",
        ),
    ]

    generated = [
        make_finding(
            "SCREENING_001",
            "CUST001",
        ),
        make_finding(
            "SCREENING_001",
            "CUST999",
        ),
        make_finding(
            "RISK_001",
            "CUST003",
        ),
    ]

    result = evaluate_findings(
        generated,
        expected,
    )

    screening = result.per_control[
        "SCREENING_001"
    ]

    assert screening["true_positives"] == 1
    assert screening["false_positives"] == 1
    assert screening["false_negatives"] == 1

    risk = result.per_control["RISK_001"]

    assert risk["true_positives"] == 1
    assert risk["false_positives"] == 0
    assert risk["false_negatives"] == 0


def test_severity_metrics():
    expected = [
        make_finding(
            "SCREENING_001",
            "CUST001",
            "CRITICAL",
        ),
        make_finding(
            "RISK_001",
            "CUST002",
            "HIGH",
        ),
    ]

    generated = [
        make_finding(
            "SCREENING_001",
            "CUST001",
            "CRITICAL",
        ),
        make_finding(
            "RISK_001",
            "CUST002",
            "HIGH",
        ),
    ]

    result = evaluate_findings(
        generated,
        expected,
    )

    assert result.per_severity["CRITICAL"][
        "true_positives"
    ] == 1

    assert result.per_severity["HIGH"][
        "true_positives"
    ] == 1


def test_empty_generated_findings():
    expected = [
        make_finding(
            "SCREENING_001",
            "CUST001",
        )
    ]

    result = evaluate_findings(
        [],
        expected,
    )

    assert result.true_positives == 0
    assert result.false_positives == 0
    assert result.false_negatives == 1

    assert result.precision == 0.0
    assert result.recall == 0.0
    assert result.f1_score == 0.0


def test_empty_expected_findings():
    generated = [
        make_finding(
            "SCREENING_001",
            "CUST001",
        )
    ]

    result = evaluate_findings(
        generated,
        [],
    )

    assert result.true_positives == 0
    assert result.false_positives == 1
    assert result.false_negatives == 0

    assert result.precision == 0.0
    assert result.recall == 0.0
    assert result.f1_score == 0.0


def test_duplicate_generated_findings_do_not_inflate_metrics():
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
        ),
        make_finding(
            "SCREENING_001",
            "CUST001",
        ),
    ]

    result = evaluate_findings(
        generated,
        expected,
    )

    assert result.true_positives == 1
    assert result.false_positives == 0
    assert result.false_negatives == 0

    assert result.precision == 1.0
    assert result.recall == 1.0