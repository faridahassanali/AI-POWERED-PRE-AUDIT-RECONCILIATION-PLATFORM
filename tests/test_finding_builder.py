from engine.finding_builder import build_finding

import pytest


# A fixed, explicit audit_run_id for all tests in this file.
# audit_run_id must now come from the caller -- it is no longer
# invented by build_finding() (see engine/finding_builder.py docstring
# for why: every finding in a real run must share ONE audit_run_id,
# not a random one per finding).
TEST_AUDIT_RUN_ID = "RUN-TEST-0001"


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
        ],
        audit_run_id=TEST_AUDIT_RUN_ID,
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
        ],
        audit_run_id=TEST_AUDIT_RUN_ID,
    )

    assert finding["finding_id"].startswith("F-")


def test_audit_run_id_is_stored_as_given():
    """
    FIX: this test used to be `test_audit_run_id_is_generated` and
    asserted that build_finding() invented its OWN random audit_run_id
    when none was given. That was exactly the bug: every finding in
    the same real audit run got a different, unrelated id, so findings
    from one run could never be grouped together for the Audit Trace.

    audit_run_id is now a required argument. build_finding() must
    store EXACTLY the id the caller (ultimately the audit run) gives
    it -- never invent its own.
    """

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
        ],
        audit_run_id=TEST_AUDIT_RUN_ID,
    )

    assert finding["audit_run_id"] == TEST_AUDIT_RUN_ID


def test_missing_audit_run_id_raises():
    """
    New test (FIX regression guard): calling build_finding() without
    audit_run_id must fail loudly, not silently default to a fresh
    random id. This is the exact contract that prevents the original
    bug from coming back.
    """

    with pytest.raises(TypeError):
        # audit_run_id omitted entirely -- missing required argument.
        build_finding(
            control_id="ARABIC_NAME_001",
            customer_id="CUST100087",
            severity="MEDIUM",
            assessment_status="FAIL",
            finding_status="REVIEW",
            expected="Customer must have an Arabic name.",
            actual="Arabic name is missing.",
            evidence={"name_ar": ""},
            policy_references=[
                {
                    "policy_id": "DATA-POLICY-001",
                    "version": "1.0",
                    "section": "Requirements",
                }
            ],
        )


def test_empty_string_audit_run_id_raises():
    """
    New test (FIX regression guard): an empty string is technically
    "provided" as an argument but is not a valid run id. build_finding()
    explicitly rejects falsy audit_run_id values with ValueError.
    """

    with pytest.raises(ValueError):
        build_finding(
            control_id="ARABIC_NAME_001",
            customer_id="CUST100087",
            severity="MEDIUM",
            assessment_status="FAIL",
            finding_status="REVIEW",
            expected="Customer must have an Arabic name.",
            actual="Arabic name is missing.",
            evidence={"name_ar": ""},
            policy_references=[
                {
                    "policy_id": "DATA-POLICY-001",
                    "version": "1.0",
                    "section": "Requirements",
                }
            ],
            audit_run_id="",
        )


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
        ],
        audit_run_id=TEST_AUDIT_RUN_ID,
    )

    assert finding["customer_id"] is None