from engine.finding_builder import build_finding


def test_build_finding_creates_required_fields():

    finding = build_finding(
        control_id="SCREENING_001",
        customer_id="CUST100005",
        severity="HIGH",
        assessment_status="FAIL",
        finding_status="REVIEW",
        expected="Opened wallets must have a CLEAR screening result.",
        actual="Screening status is HIGH_RISK.",
        evidence={
            "screening_status": "HIGH_RISK",
            "screening_evidence_present": "True"
        },
        policy_references=[
            {
                "policy_id": "SCREENING-POLICY-001",
                "version": "1.0",
                "section": "Requirements"
            }
        ]
    )

    assert "finding_id" in finding
    assert "audit_run_id" in finding
    assert finding["control_id"] == "SCREENING_001"
    assert finding["customer_id"] == "CUST100005"
    assert finding["severity"] == "HIGH"
    assert finding["assessment_status"] == "FAIL"
    assert finding["finding_status"] == "REVIEW"
    assert finding["evidence"] is not None
    assert finding["policy_references"] is not None


def test_finding_id_is_generated():

    finding = build_finding(
        control_id="RISK_001",
        customer_id="CUST100031",
        severity="HIGH",
        assessment_status="FAIL",
        finding_status="REVIEW",
        expected="HIGH-risk opened wallets must have an approved risk exception.",
        actual="HIGH-risk opened wallet has no valid approved risk exception.",
        evidence={
            "risk_level": "HIGH"
        },
        policy_references=[
            {
                "policy_id": "RISK-POLICY-001",
                "version": "1.0",
                "section": "Requirements"
            }
        ]
    )

    assert finding["finding_id"].startswith("F-")


def test_audit_run_id_is_generated():

    finding = build_finding(
        control_id="ARABIC_NAME_001",
        customer_id="CUST100087",
        severity="MEDIUM",
        assessment_status="FAIL",
        finding_status="REVIEW",
        expected="Customer must have an Arabic name.",
        actual="Arabic name is missing.",
        evidence={
            "name_ar": ""
        },
        policy_references=[
            {
                "policy_id": "DATA-POLICY-001",
                "version": "1.0",
                "section": "Requirements"
            }
        ]
    )

    assert finding["audit_run_id"].startswith("RUN-")


def test_customer_id_can_be_none():

    finding = build_finding(
        control_id="RECON_001",
        customer_id=None,
        severity="HIGH",
        assessment_status="FAIL",
        finding_status="REVIEW",
        expected="Final report must match the approved source.",
        actual="Record exists only in the final report.",
        evidence={
            "record": "EXTRA"
        },
        policy_references=[
            {
                "policy_id": "RECON-POLICY-001",
                "version": "1.0",
                "section": "Requirements"
            }
        ]
    )

    assert finding["customer_id"] is None