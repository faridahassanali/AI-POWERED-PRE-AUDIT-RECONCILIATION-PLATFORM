import pandas as pd
import pytest

from engine.data_loader import load_data, build_unified_customer_record


@pytest.fixture(scope="module")
def tables():
    return load_data()


@pytest.fixture(scope="module")
def unified(tables):
    return build_unified_customer_record(tables)


def test_unified_record_has_one_row_per_customer(unified):
    """Every customer must appear exactly once in the unified record."""

    assert unified["customer_id"].notna().all()
    assert unified["customer_id"].ne("").all()

    duplicate_ids = unified[
        unified["customer_id"].duplicated(keep=False)
    ]["customer_id"].unique()

    assert len(duplicate_ids) == 0, (
        f"Duplicate customer IDs found in unified record: "
        f"{duplicate_ids.tolist()}"
    )


def test_unified_record_contains_required_control_fields(unified):
    """Unified records must contain fields required by the controls."""

    required_fields = {
        "customer_id",
        "name_ar",
        "risk_level",
        "screening_status",
        "screening_evidence_present",
        "screening_reference",
        "account_status",
        "wallet_status",
        "risk_exception_approved",
        "risk_exception_reference",
        "risk_exception_reviewer",
        "dormant_handling_status",
    }

    missing = required_fields - set(unified.columns)

    assert not missing, (
        f"Unified record is missing required control fields: "
        f"{sorted(missing)}"
    )


def test_unified_record_has_expected_customer_count(tables, unified):
    """
    The unified population should represent the approved source population.
    """

    expected_count = tables["approved_source_population"]["customer_id"].nunique()
    actual_count = unified["customer_id"].nunique()

    assert actual_count == expected_count, (
        f"Expected {expected_count} unique customers, "
        f"but unified record contains {actual_count}."
    )


def test_unified_record_has_no_nan_after_merge(unified):
    """
    Left joins (e.g. screening_results) must not leave real NaN
    values in the unified record. Every source table is loaded
    with fillna(""), so any customer with no matching row in a
    joined table must end up with "" too -- not NaN. Downstream
    control checks compare against "", and a NaN slipping through
    would make those checks silently fail to detect missing data.
    """

    nan_columns = [
        column
        for column in unified.columns
        if unified[column].isna().any()
    ]

    assert nan_columns == [], (
        f"Unified record contains real NaN values in columns: "
        f"{nan_columns}. Left joins likely reintroduced NaN for "
        f"unmatched customers -- check fillna('') is applied "
        f"after the merges."
    )