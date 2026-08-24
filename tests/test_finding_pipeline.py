from engine.finding_builder import build_finding
from engine.finding_validator import validate_finding
from engine.ground_truth_evaluator import evaluate_findings


def test_finding_pipeline_end_to_end():

    finding = build_finding(
        audit_run_id="RUN-TEST00001",
        control_id="SCREENING_001",
        customer_id="CUST100005",
        severity="HIGH",
        assessment_status="FAIL",
        finding_status="REVIEW",
        expected=(
            "Opened wallets must have a CLEAR "
            "screening result."
        ),
        actual=(
            "Screening status is HIGH_RISK."
        ),
        evidence={
            "customer_id": "CUST100005",
            "screening_status": "HIGH_RISK",
            "screening_evidence_present": True,
            "wallet_status": "OPENED",
        },
        policy_references=[
            {
                "policy_id": "SCREENING-POLICY-001",
                "version": "1.0",
                "section": "Requirements",
            }
        ],
    )

    # 1. Validate
    assert validate_finding(finding) is True

    # 2. Create ground truth
    expected = [
        {
            **finding,
            "finding_id": "GROUND-TRUTH-001",
            "audit_run_id": "GROUND-TRUTH",
        }
    ]

    # 3. Evaluate
    result = evaluate_findings(
        generated_findings=[finding],
        expected_findings=expected,
    )

    assert result.true_positives == 1
    assert result.false_positives == 0
    assert result.false_negatives == 0

    assert result.precision == 1.0
    assert result.recall == 1.0
    assert result.f1_score == 1.0