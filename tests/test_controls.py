"""
Tests for engine/controls.py — SCREENING_001.

Run with:  pytest tests/test_controls.py
(from inside the project root, with the venv active)

Two kinds of tests here:
  1. Unit tests — small handcrafted records, so we know exactly why a
     given case should PASS or FAIL, independent of the real dataset.
  2. Integration tests — run the control against the real V3 dataset and
     compare the result to expected_findings.csv (the ground truth).
"""

import sys
from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).resolve().parent.parent / "engine")
)

import pandas as pd
import pytest

from engine.controls import (
    screening_001,
    risk_001,
    arabic_name_001,
    dormant_001,
    run_all_controls,
)

from engine.reconciliation import reconciliation_001

from engine.data_loader import (
    load_data,
    build_unified_customer_record,
)
def make_record(**overrides) -> pd.Series:
    """Build a minimal customer record for SCREENING_001, with sensible
    defaults that PASS the control unless overridden."""
    base = {
        "customer_id": "CUST_TEST",
        "wallet_status": "OPENED",
        "screening_status": "CLEAR",
        "screening_evidence_present": "True",
        "screening_reference": "SCR-TEST",
    }
    base.update(overrides)
    return pd.Series(base)


# --- Unit tests -------------------------------------------------------

def test_clear_screening_with_evidence_passes():
    record = make_record()
    assert screening_001(record) is None


def test_wallet_not_opened_is_skipped_even_if_screening_bad():
    """The control only applies to opened wallets — anything else is out of scope."""
    record = make_record(wallet_status="PENDING", screening_status="HIGH_RISK")
    assert screening_001(record) is None


def test_high_risk_screening_is_critical():
    record = make_record(screening_status="HIGH_RISK")
    finding = screening_001(record)
    assert finding is not None
    assert finding["severity"] == "CRITICAL"
    assert finding["control_id"] == "SCREENING_001"
    assert finding["assessment_status"] == "FAIL"
    assert finding["finding_status"] == "REVIEW"


@pytest.mark.parametrize("status", ["PENDING", "NO_MATCH"])
def test_non_clear_non_high_risk_is_high_severity(status):
    record = make_record(screening_status=status)
    finding = screening_001(record)
    assert finding is not None
    assert finding["severity"] == "HIGH"


def test_missing_evidence_fails_even_if_status_clear():
    record = make_record(screening_evidence_present="False")
    finding = screening_001(record)
    assert finding is not None
    assert finding["severity"] == "HIGH"


def test_finding_contains_required_fields():
    record = make_record(screening_status="HIGH_RISK")
    finding = screening_001(record)
    required_keys = {
        "control_id", "customer_id", "severity", "assessment_status",
        "finding_status", "expected", "actual", "evidence", "policy_references",
    }
    assert required_keys.issubset(finding.keys()
                                  )

# --- RISK_001 unit tests --------------------------------------------

def test_high_risk_opened_without_exception_fails():
    record = make_record(
        customer_id="CUST_RISK_TEST",
        risk_level="HIGH",
        wallet_status="OPENED",
        risk_exception_approved="False",
        risk_exception_reference="",
        risk_exception_reviewer="",
    )

    finding = risk_001(record)

    assert finding is not None
    assert finding["control_id"] == "RISK_001"
    assert finding["severity"] == "HIGH"
    assert finding["assessment_status"] == "FAIL"
    assert finding["finding_status"] == "REVIEW"


def test_high_risk_opened_with_valid_exception_passes():
    record = make_record(
        risk_level="HIGH",
        wallet_status="OPENED",
        risk_exception_approved="True",
        risk_exception_reference="EXC-001",
        risk_exception_reviewer="Reviewer1",
    )

    assert risk_001(record) is None


def test_non_high_risk_is_not_applicable():
    record = make_record(
        risk_level="MEDIUM",
        wallet_status="OPENED",
        risk_exception_approved="False",
    )

    assert risk_001(record) is None


def test_high_risk_wallet_not_opened_is_not_applicable():
    record = make_record(
        risk_level="HIGH",
        wallet_status="PENDING",
        risk_exception_approved="False",
    )

    assert risk_001(record) is None   
    # --- ARABIC_NAME_001 unit tests -------------------------------------

def test_arabic_name_with_arabic_characters_passes():
    record = make_record(
        customer_id="CUST_AR_TEST",
        name_ar="أحمد يوسف",
    )

    assert arabic_name_001(record) is None


def test_empty_arabic_name_fails():
    record = make_record(
        customer_id="CUST_AR_TEST",
        name_ar="",
    )

    finding = arabic_name_001(record)

    assert finding is not None
    assert finding["control_id"] == "ARABIC_NAME_001"
    assert finding["severity"] == "MEDIUM"
    assert finding["assessment_status"] == "FAIL"
    assert finding["finding_status"] == "REVIEW"


def test_non_arabic_name_fails():
    record = make_record(
        customer_id="CUST_AR_TEST",
        name_ar="Ahmed Youssef",
    )

    finding = arabic_name_001(record)

    assert finding is not None
    assert finding["severity"] == "MEDIUM"


def test_mixed_arabic_and_english_name_passes():
    record = make_record(
        customer_id="CUST_AR_TEST",
        name_ar="Ahmed أحمد",
    )

    assert arabic_name_001(record) is None


def test_arabic_name_finding_contains_required_fields():
    record = make_record(
        customer_id="CUST_AR_TEST",
        name_ar="Ahmed Youssef",
    )

    finding = arabic_name_001(record)

    required_keys = {
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

    assert required_keys.issubset(finding.keys())    
  

# --- DORMANT_001 unit tests ----------------------------------------

def test_dormant_opened_completed_passes():
    record = make_record(
        customer_id="CUST_DORMANT_TEST",
        account_status="DORMANT",
        wallet_status="OPENED",
        last_transaction_date="2026-01-01",
        dormant_handling_status="COMPLETED",
    )

    assert dormant_001(record) is None


def test_dormant_opened_not_completed_fails():
    record = make_record(
        customer_id="CUST_DORMANT_TEST",
        account_status="DORMANT",
        wallet_status="OPENED",
        last_transaction_date="2026-01-01",
        dormant_handling_status="PENDING",
    )

    finding = dormant_001(record)

    assert finding is not None
    assert finding["control_id"] == "DORMANT_001"
    assert finding["severity"] == "HIGH"
    assert finding["assessment_status"] == "FAIL"
    assert finding["finding_status"] == "REVIEW"


def test_non_dormant_account_is_not_applicable():
    record = make_record(
        account_status="ACTIVE",
        wallet_status="OPENED",
        dormant_handling_status="NOT_REQUIRED",
    )

    assert dormant_001(record) is None


def test_dormant_wallet_not_opened_is_not_applicable():
    record = make_record(
        account_status="DORMANT",
        wallet_status="PENDING",
        dormant_handling_status="PENDING",
    )

    assert dormant_001(record) is None


def test_dormant_finding_contains_required_fields():
    record = make_record(
        customer_id="CUST_DORMANT_TEST",
        account_status="DORMANT",
        wallet_status="OPENED",
        last_transaction_date="2026-01-01",
        dormant_handling_status="PENDING",
    )

    finding = dormant_001(record)

    required_keys = {
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

    assert required_keys.issubset(finding.keys())
# --- Integration tests against the real V3 dataset ---------------------

@pytest.fixture(scope="module")
def unified_and_expected():
    tables = load_data()
    unified = build_unified_customer_record(tables)
    return unified, tables["expected_findings"]


def test_screening_001_matches_expected_count(unified_and_expected):
    unified, expected = unified_and_expected
    findings = [f for f in run_all_controls(unified) if f["control_id"] == "SCREENING_001"]
    expected_count = len(expected[expected["control_id"] == "SCREENING_001"])
    assert len(findings) == expected_count


def test_screening_001_matches_expected_customer_ids(unified_and_expected):
    unified, expected = unified_and_expected
    findings = [f for f in run_all_controls(unified) if f["control_id"] == "SCREENING_001"]
    my_ids = {f["customer_id"] for f in findings}
    expected_ids = set(expected[expected["control_id"] == "SCREENING_001"]["customer_id"])
    assert my_ids == expected_ids


def test_screening_001_severity_distribution_matches_expected(unified_and_expected):
    unified, expected = unified_and_expected
    findings = [f for f in run_all_controls(unified) if f["control_id"] == "SCREENING_001"]
    my_counts = pd.Series([f["severity"] for f in findings]).value_counts().to_dict()

    expected_screening = expected[expected["control_id"] == "SCREENING_001"]
    expected_counts = expected_screening["severity"].value_counts().to_dict()

    assert my_counts == expected_counts
    # --- DORMANT_001 integration tests ---------------------------------

def test_dormant_001_matches_expected_count(unified_and_expected):
    unified, expected = unified_and_expected

    findings = [
        f for f in run_all_controls(unified)
        if f["control_id"] == "DORMANT_001"
    ]

    expected_count = len(
        expected[expected["control_id"] == "DORMANT_001"]
    )

    assert len(findings) == expected_count


def test_dormant_001_matches_expected_customer_ids(unified_and_expected):
    unified, expected = unified_and_expected

    findings = [
        f for f in run_all_controls(unified)
        if f["control_id"] == "DORMANT_001"
    ]

    my_ids = {f["customer_id"] for f in findings}

    expected_ids = set(
        expected[
            expected["control_id"] == "DORMANT_001"
        ]["customer_id"]
    )

    assert my_ids == expected_ids


def test_dormant_001_severity_distribution_matches_expected(
    unified_and_expected
):
    unified, expected = unified_and_expected

    findings = [
        f for f in run_all_controls(unified)
        if f["control_id"] == "DORMANT_001"
    ]

    my_counts = pd.Series(
        [f["severity"] for f in findings]
    ).value_counts().to_dict()

    expected_dormant = expected[
        expected["control_id"] == "DORMANT_001"
    ]

    expected_counts = (
        expected_dormant["severity"]
        .value_counts()
        .to_dict()
    )

    assert my_counts == expected_counts
# --- RISK_001 integration tests -------------------------------------

def test_risk_001_matches_expected_count(unified_and_expected):
    unified, expected = unified_and_expected

    findings = [
        f for f in run_all_controls(unified)
        if f["control_id"] == "RISK_001"
    ]

    expected_count = len(
        expected[expected["control_id"] == "RISK_001"]
    )

    assert len(findings) == expected_count


def test_risk_001_matches_expected_customer_ids(unified_and_expected):
    unified, expected = unified_and_expected

    findings = [
        f for f in run_all_controls(unified)
        if f["control_id"] == "RISK_001"
    ]

    my_ids = {f["customer_id"] for f in findings}

    expected_ids = set(
        expected[
            expected["control_id"] == "RISK_001"
        ]["customer_id"]
    )

    assert my_ids == expected_ids


def test_risk_001_severity_distribution_matches_expected(
    unified_and_expected
):
    unified, expected = unified_and_expected

    findings = [
        f for f in run_all_controls(unified)
        if f["control_id"] == "RISK_001"
    ]

    my_counts = pd.Series(
        [f["severity"] for f in findings]
    ).value_counts().to_dict()

    expected_risk = expected[
        expected["control_id"] == "RISK_001"
    ]

    expected_counts = (
        expected_risk["severity"]
        .value_counts()
        .to_dict()
    )

    assert my_counts == expected_counts
    # --- ARABIC_NAME_001 integration tests ------------------------------

def test_arabic_name_001_matches_expected_count(unified_and_expected):
    unified, expected = unified_and_expected

    findings = [
        f for f in run_all_controls(unified)
        if f["control_id"] == "ARABIC_NAME_001"
    ]

    expected_count = len(
        expected[expected["control_id"] == "ARABIC_NAME_001"]
    )

    assert len(findings) == expected_count


def test_arabic_name_001_matches_expected_customer_ids(
    unified_and_expected
):
    unified, expected = unified_and_expected

    findings = [
        f for f in run_all_controls(unified)
        if f["control_id"] == "ARABIC_NAME_001"
    ]

    my_ids = {f["customer_id"] for f in findings}

    expected_ids = set(
        expected[
            expected["control_id"] == "ARABIC_NAME_001"
        ]["customer_id"]
    )

    assert my_ids == expected_ids


def test_arabic_name_001_severity_distribution_matches_expected(
    unified_and_expected
):
    unified, expected = unified_and_expected

    findings = [
        f for f in run_all_controls(unified)
        if f["control_id"] == "ARABIC_NAME_001"
    ]

    my_counts = pd.Series(
        [f["severity"] for f in findings]
    ).value_counts().to_dict()

    expected_arabic = expected[
        expected["control_id"] == "ARABIC_NAME_001"
    ]

    expected_counts = (
        expected_arabic["severity"]
        .value_counts()
        .to_dict()
    )

    assert my_counts == expected_counts
   # --- RECON_001 integration tests -------------------------------------

def test_recon_001_matches_expected_count(unified_and_expected):
    unified, expected = unified_and_expected

    tables = load_data()
    findings = reconciliation_001(tables)

    expected_count = len(
        expected[expected["control_id"] == "RECON_001"]
    )

    assert len(findings) == expected_count


def test_recon_001_matches_expected_customer_ids(unified_and_expected):
    unified, expected = unified_and_expected

    tables = load_data()
    findings = reconciliation_001(tables)

    my_ids = {
        f["customer_id"]
        for f in findings
    }

    expected_ids = set(
        expected[
            expected["control_id"] == "RECON_001"
        ]["customer_id"]
    )

    assert my_ids == expected_ids


def test_recon_001_severity_distribution_matches_expected(
    unified_and_expected
):
    unified, expected = unified_and_expected

    tables = load_data()
    findings = reconciliation_001(tables)

    my_counts = pd.Series(
        [f["severity"] for f in findings]
    ).value_counts().to_dict()

    expected_recon = expected[
        expected["control_id"] == "RECON_001"
    ]

    expected_counts = (
        expected_recon["severity"]
        .value_counts()
        .to_dict()
    )

    assert my_counts == expected_counts