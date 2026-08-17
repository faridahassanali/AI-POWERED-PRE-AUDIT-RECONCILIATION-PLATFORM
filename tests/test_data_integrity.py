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