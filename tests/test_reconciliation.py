"""
Tests for engine/reconciliation.py — RECON_001.

These tests verify the reconciliation logic using small
handcrafted datasets instead of relying only on the real dataset.
"""

import pandas as pd

from engine.reconciliation import reconciliation_001


# audit_run_id is now a required argument for reconciliation_001()
# (see engine/reconciliation.py / engine/finding_builder.py) -- every
# finding in a real run must share ONE audit_run_id, so it can no
# longer be invented internally per finding.
TEST_AUDIT_RUN_ID = "RUN-TEST-0001"


def make_tables(
    source_rows,
    report_rows,
    customer_rows=None,
    screening_rows=None,
):
    """
    Build a minimal set of tables required by reconciliation_001.
    """

    if customer_rows is None:
        customer_rows = []

    if screening_rows is None:
        screening_rows = []

    return {
        "approved_source_population": pd.DataFrame(
            source_rows,
            columns=["customer_id", "name_ar"],
        ),
        "customers": pd.DataFrame(
            customer_rows,
            columns=[
                "customer_id",
                "risk_level",
                "account_status",
                "wallet_status",
            ],
        ),
        "screening_results": pd.DataFrame(
            screening_rows,
            columns=[
                "customer_id",
                "screening_status",
            ],
        ),
        "wallet_initialization": pd.DataFrame(),
        "final_wallet_audit_report": pd.DataFrame(
            report_rows,
            columns=[
                "customer_id",
                "name_ar",
                "risk_level",
                "screening_status",
                "account_status",
                "wallet_status",
            ],
        ),
    }


def test_matching_records_pass():
    """
    Source and final report contain identical information.
    Therefore, there should be no reconciliation finding.
    """

    source = [
        {
            "customer_id": "CUST_TEST_001",
            "name_ar": "أحمد علي",
        }
    ]

    customers = [
        {
            "customer_id": "CUST_TEST_001",
            "risk_level": "LOW",
            "account_status": "ACTIVE",
            "wallet_status": "OPENED",
        }
    ]

    screening = [
        {
            "customer_id": "CUST_TEST_001",
            "screening_status": "CLEAR",
        }
    ]

    report = [
        {
            "customer_id": "CUST_TEST_001",
            "name_ar": "أحمد علي",
            "risk_level": "LOW",
            "screening_status": "CLEAR",
            "account_status": "ACTIVE",
            "wallet_status": "OPENED",
        }
    ]

    tables = make_tables(
        source,
        report,
        customers,
        screening,
    )

    findings = reconciliation_001(tables, audit_run_id=TEST_AUDIT_RUN_ID)

    assert findings == []


def test_risk_level_mismatch_fails():
    """
    Source says HIGH risk while final report says LOW.
    """

    source = [
        {
            "customer_id": "CUST_TEST_002",
            "name_ar": "محمد علي",
        }
    ]

    customers = [
        {
            "customer_id": "CUST_TEST_002",
            "risk_level": "HIGH",
            "account_status": "ACTIVE",
            "wallet_status": "OPENED",
        }
    ]

    screening = [
        {
            "customer_id": "CUST_TEST_002",
            "screening_status": "CLEAR",
        }
    ]

    report = [
        {
            "customer_id": "CUST_TEST_002",
            "name_ar": "محمد علي",
            "risk_level": "LOW",
            "screening_status": "CLEAR",
            "account_status": "ACTIVE",
            "wallet_status": "OPENED",
        }
    ]

    tables = make_tables(
        source,
        report,
        customers,
        screening,
    )

    findings = reconciliation_001(tables, audit_run_id=TEST_AUDIT_RUN_ID)

    assert len(findings) == 1
    assert findings[0]["control_id"] == "RECON_001"
    assert findings[0]["customer_id"] == "CUST_TEST_002"
    assert findings[0]["severity"] == "HIGH"
    assert findings[0]["assessment_status"] == "FAIL"
    assert findings[0]["audit_run_id"] == TEST_AUDIT_RUN_ID

    assert findings[0]["evidence"]["mismatches"]["risk_level"] == {
        "source": "HIGH",
        "report": "LOW",
    }


def test_screening_status_mismatch_fails():
    """
    Source says HIGH_RISK while final report says CLEAR.
    """

    source = [
        {
            "customer_id": "CUST_TEST_003",
            "name_ar": "سارة أحمد",
        }
    ]

    customers = [
        {
            "customer_id": "CUST_TEST_003",
            "risk_level": "HIGH",
            "account_status": "ACTIVE",
            "wallet_status": "OPENED",
        }
    ]

    screening = [
        {
            "customer_id": "CUST_TEST_003",
            "screening_status": "HIGH_RISK",
        }
    ]

    report = [
        {
            "customer_id": "CUST_TEST_003",
            "name_ar": "سارة أحمد",
            "risk_level": "HIGH",
            "screening_status": "CLEAR",
            "account_status": "ACTIVE",
            "wallet_status": "OPENED",
        }
    ]

    tables = make_tables(
        source,
        report,
        customers,
        screening,
    )

    findings = reconciliation_001(tables, audit_run_id=TEST_AUDIT_RUN_ID)

    assert len(findings) == 1

    mismatch = findings[0]["evidence"]["mismatches"]["screening_status"]

    assert mismatch["source"] == "HIGH_RISK"
    assert mismatch["report"] == "CLEAR"


def test_missing_record_from_final_report_fails():
    """
    Customer exists in the approved source but is missing
    from the final audit report.
    """

    source = [
        {
            "customer_id": "CUST_TEST_004",
            "name_ar": "علي حسن",
        }
    ]

    customers = [
        {
            "customer_id": "CUST_TEST_004",
            "risk_level": "LOW",
            "account_status": "ACTIVE",
            "wallet_status": "OPENED",
        }
    ]

    screening = [
        {
            "customer_id": "CUST_TEST_004",
            "screening_status": "CLEAR",
        }
    ]

    report = []

    tables = make_tables(
        source,
        report,
        customers,
        screening,
    )

    findings = reconciliation_001(tables, audit_run_id=TEST_AUDIT_RUN_ID)

    assert len(findings) == 1
    assert findings[0]["customer_id"] == "CUST_TEST_004"

    assert findings[0]["evidence"]["mismatches"]["record"] == {
        "source": "PRESENT",
        "report": "MISSING",
    }


def test_extra_record_in_final_report_fails():
    """
    Customer exists in the final report but not in the
    approved source population.
    """

    source = []

    customers = []

    screening = []

    report = [
        {
            "customer_id": "CUST_TEST_999",
            "name_ar": "محمود حسن",
            "risk_level": "LOW",
            "screening_status": "CLEAR",
            "account_status": "ACTIVE",
            "wallet_status": "OPENED",
        }
    ]

    tables = make_tables(
        source,
        report,
        customers,
        screening,
    )

    findings = reconciliation_001(tables, audit_run_id=TEST_AUDIT_RUN_ID)

    assert len(findings) == 1
    assert findings[0]["customer_id"] == "CUST_TEST_999"

    assert findings[0]["evidence"]["mismatches"]["record"] == {
        "source": "MISSING",
        "report": "PRESENT",
    }


def test_multiple_mismatches_are_combined_into_one_finding():
    """
    If the same customer has multiple mismatched fields,
    RECON_001 should create ONE finding containing all mismatches.
    """

    source = [
        {
            "customer_id": "CUST_TEST_005",
            "name_ar": "يوسف علي",
        }
    ]

    customers = [
        {
            "customer_id": "CUST_TEST_005",
            "risk_level": "HIGH",
            "account_status": "DORMANT",
            "wallet_status": "OPENED",
        }
    ]

    screening = [
        {
            "customer_id": "CUST_TEST_005",
            "screening_status": "HIGH_RISK",
        }
    ]

    report = [
        {
            "customer_id": "CUST_TEST_005",
            "name_ar": "يوسف علي",
            "risk_level": "LOW",
            "screening_status": "CLEAR",
            "account_status": "ACTIVE",
            "wallet_status": "OPENED",
        }
    ]

    tables = make_tables(
        source,
        report,
        customers,
        screening,
    )

    findings = reconciliation_001(tables, audit_run_id=TEST_AUDIT_RUN_ID)

    assert len(findings) == 1

    mismatches = findings[0]["evidence"]["mismatches"]

    assert "risk_level" in mismatches
    assert "screening_status" in mismatches
    assert "account_status" in mismatches

    assert mismatches["risk_level"] == {
        "source": "HIGH",
        "report": "LOW",
    }

    assert mismatches["screening_status"] == {
        "source": "HIGH_RISK",
        "report": "CLEAR",
    }

    assert mismatches["account_status"] == {
        "source": "DORMANT",
        "report": "ACTIVE",
    }