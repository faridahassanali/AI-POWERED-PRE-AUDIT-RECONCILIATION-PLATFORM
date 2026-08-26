"""
Tests for app.py (Streamlit Findings List + Finding Detail view).

Pure-function coverage:
    - get_finding_by_id
    - filter_findings
    - get_nav_state
    - sort_findings

Streamlit AppTest smoke coverage:
    - Generate AI explanation button for a CONFIRMED finding
      without existing AI output.

The Streamlit application lives at:

    frontend/app.py

The frontend communicates with the FastAPI backend through:

    frontend.api_client

Therefore the AppTest smoke test patches frontend.api_client
functions rather than the engine layer.
"""

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from frontend.app import (
    filter_findings,
    get_finding_by_id,
    get_nav_state,
    sort_findings,
)


# =========================================================
# FIXTURES
# =========================================================


@pytest.fixture
def sample_findings() -> list[dict]:
    """Four findings spanning different controls/severities/statuses."""

    return [
        {
            "finding_id": "F-AAAA0001",
            "audit_run_id": "RUN-0001",
            "control_id": "SCREENING_001",
            "customer_id": "CUST100005",
            "severity": "CRITICAL",
            "assessment_status": "FAIL",
            "finding_status": "REVIEW",
            "expected": (
                "Customer must have a CLEAR screening result."
            ),
            "actual": (
                "Screening result is not CLEAR."
            ),
            "evidence": {
                "screening_status": "PENDING"
            },
            "policy_references": [
                {
                    "policy_id": "POL-01",
                    "version": "1.0",
                    "section": "3.2",
                }
            ],
        },
        {
            "finding_id": "F-BBBB0002",
            "audit_run_id": "RUN-0001",
            "control_id": "ARABIC_NAME_001",
            "customer_id": "CUST100012",
            "severity": "MEDIUM",
            "assessment_status": "FAIL",
            "finding_status": "REVIEW",
            "expected": (
                "Customer record must contain an Arabic-script name."
            ),
            "actual": (
                "Arabic name is missing."
            ),
            "evidence": {
                "name_ar": None
            },
            "policy_references": [],
        },
        {
            "finding_id": "F-CCCC0003",
            "audit_run_id": "RUN-0001",
            "control_id": "SCREENING_001",
            "customer_id": "CUST100016",
            "severity": "HIGH",
            "assessment_status": "FAIL",
            "finding_status": "CONFIRMED",
            "expected": (
                "Customer must have a CLEAR screening result."
            ),
            "actual": (
                "Screening result is not CLEAR."
            ),
            "evidence": {
                "screening_status": "REJECTED"
            },
            "policy_references": [],
        },
        {
            "finding_id": "F-DDDD0004",
            "audit_run_id": "RUN-0001",
            "control_id": "RISK_001",
            "customer_id": None,
            "severity": "LOW",
            "assessment_status": "FAIL",
            "finding_status": "REJECTED",
            "expected": (
                "Risk score must be recalculated within 30 days."
            ),
            "actual": (
                "Risk score is stale."
            ),
            "evidence": {},
            "policy_references": [],
        },
    ]


@pytest.fixture
def sample_df(sample_findings) -> pd.DataFrame:
    return pd.DataFrame(sample_findings)


# =========================================================
# get_finding_by_id
# =========================================================


def test_get_finding_by_id_found(sample_findings):
    finding = get_finding_by_id(
        sample_findings,
        "F-BBBB0002",
    )

    assert finding is not None
    assert finding["control_id"] == "ARABIC_NAME_001"


def test_get_finding_by_id_not_found(sample_findings):
    assert (
        get_finding_by_id(
            sample_findings,
            "F-NOPE9999",
        )
        is None
    )


def test_get_finding_by_id_empty_list():
    assert (
        get_finding_by_id(
            [],
            "F-AAAA0001",
        )
        is None
    )


# =========================================================
# filter_findings — control/severity/status filters
# =========================================================


def test_filter_findings_no_filters_returns_all(sample_df):
    result = filter_findings(
        sample_df,
        selected_controls=sample_df[
            "control_id"
        ].unique().tolist(),
        selected_severities=sample_df[
            "severity"
        ].unique().tolist(),
        selected_statuses=sample_df[
            "finding_status"
        ].unique().tolist(),
    )

    assert len(result) == 4


def test_filter_findings_by_single_control(sample_df):
    result = filter_findings(
        sample_df,
        selected_controls=["SCREENING_001"],
        selected_severities=sample_df[
            "severity"
        ].unique().tolist(),
        selected_statuses=sample_df[
            "finding_status"
        ].unique().tolist(),
    )

    assert set(result["finding_id"]) == {
        "F-AAAA0001",
        "F-CCCC0003",
    }


def test_filter_findings_by_severity(sample_df):
    result = filter_findings(
        sample_df,
        selected_controls=sample_df[
            "control_id"
        ].unique().tolist(),
        selected_severities=["CRITICAL"],
        selected_statuses=sample_df[
            "finding_status"
        ].unique().tolist(),
    )

    assert result["finding_id"].tolist() == [
        "F-AAAA0001"
    ]


def test_filter_findings_by_status(sample_df):
    result = filter_findings(
        sample_df,
        selected_controls=sample_df[
            "control_id"
        ].unique().tolist(),
        selected_severities=sample_df[
            "severity"
        ].unique().tolist(),
        selected_statuses=["REJECTED"],
    )

    assert result["finding_id"].tolist() == [
        "F-DDDD0004"
    ]


def test_filter_findings_combined_filters_narrow_correctly(
    sample_df,
):
    result = filter_findings(
        sample_df,
        selected_controls=["SCREENING_001"],
        selected_severities=["HIGH"],
        selected_statuses=["CONFIRMED"],
    )

    assert result["finding_id"].tolist() == [
        "F-CCCC0003"
    ]


def test_filter_findings_empty_selection_returns_empty(
    sample_df,
):
    result = filter_findings(
        sample_df,
        selected_controls=[],
        selected_severities=sample_df[
            "severity"
        ].unique().tolist(),
        selected_statuses=sample_df[
            "finding_status"
        ].unique().tolist(),
    )

    assert result.empty


# =========================================================
# filter_findings — search
# =========================================================


def test_filter_findings_search_by_finding_id_case_insensitive(
    sample_df,
):
    result = filter_findings(
        sample_df,
        selected_controls=sample_df[
            "control_id"
        ].unique().tolist(),
        selected_severities=sample_df[
            "severity"
        ].unique().tolist(),
        selected_statuses=sample_df[
            "finding_status"
        ].unique().tolist(),
        search="f-aaaa0001",
    )

    assert result["finding_id"].tolist() == [
        "F-AAAA0001"
    ]


def test_filter_findings_search_by_customer_id(
    sample_df,
):
    result = filter_findings(
        sample_df,
        selected_controls=sample_df[
            "control_id"
        ].unique().tolist(),
        selected_severities=sample_df[
            "severity"
        ].unique().tolist(),
        selected_statuses=sample_df[
            "finding_status"
        ].unique().tolist(),
        search="100016",
    )

    assert result["finding_id"].tolist() == [
        "F-CCCC0003"
    ]


def test_filter_findings_search_with_null_customer_id_does_not_crash(
    sample_df,
):
    """
    F-DDDD0004 has customer_id=None.
    Searching must not raise on pandas .str operations.
    """

    result = filter_findings(
        sample_df,
        selected_controls=sample_df[
            "control_id"
        ].unique().tolist(),
        selected_severities=sample_df[
            "severity"
        ].unique().tolist(),
        selected_statuses=sample_df[
            "finding_status"
        ].unique().tolist(),
        search="dddd",
    )

    assert result["finding_id"].tolist() == [
        "F-DDDD0004"
    ]


def test_filter_findings_search_no_match_returns_empty(
    sample_df,
):
    result = filter_findings(
        sample_df,
        selected_controls=sample_df[
            "control_id"
        ].unique().tolist(),
        selected_severities=sample_df[
            "severity"
        ].unique().tolist(),
        selected_statuses=sample_df[
            "finding_status"
        ].unique().tolist(),
        search="does-not-exist",
    )

    assert result.empty


# =========================================================
# get_nav_state
# =========================================================


def test_get_nav_state_middle_position():
    ids = [
        "F-1",
        "F-2",
        "F-3",
    ]

    state = get_nav_state(
        ids,
        "F-2",
    )

    assert state == {
        "position": 1,
        "total": 3,
        "prev_id": "F-1",
        "next_id": "F-3",
    }


def test_get_nav_state_first_position_has_no_prev():
    ids = [
        "F-1",
        "F-2",
        "F-3",
    ]

    state = get_nav_state(
        ids,
        "F-1",
    )

    assert state["prev_id"] is None
    assert state["next_id"] == "F-2"


def test_get_nav_state_last_position_has_no_next():
    ids = [
        "F-1",
        "F-2",
        "F-3",
    ]

    state = get_nav_state(
        ids,
        "F-3",
    )

    assert state["next_id"] is None
    assert state["prev_id"] == "F-2"


def test_get_nav_state_single_item_has_no_prev_or_next():
    state = get_nav_state(
        ["F-1"],
        "F-1",
    )

    assert state["prev_id"] is None
    assert state["next_id"] is None
    assert state["total"] == 1


def test_get_nav_state_id_not_in_list_returns_none():
    """
    Simulates a stale selected_finding_id after filters change.
    """

    assert (
        get_nav_state(
            ["F-1", "F-2"],
            "F-STALE",
        )
        is None
    )


# =========================================================
# sort_findings
# =========================================================


def test_sort_findings_by_severity_critical_first(
    sample_df,
):
    result = sort_findings(
        sample_df,
        "Severity (Critical first)",
    )

    assert result["severity"].tolist() == [
        "CRITICAL",
        "HIGH",
        "MEDIUM",
        "LOW",
    ]


def test_sort_findings_by_finding_id(sample_df):
    result = sort_findings(
        sample_df,
        "Finding ID",
    )

    assert result["finding_id"].tolist() == sorted(
        sample_df["finding_id"]
    )


def test_sort_findings_by_control(sample_df):
    result = sort_findings(
        sample_df,
        "Control",
    )

    assert result["control_id"].tolist() == sorted(
        sample_df["control_id"]
    )


def test_sort_findings_by_customer_puts_null_last(
    sample_df,
):
    """
    F-DDDD0004 has customer_id=None.

    Sorting must:
        1. not crash
        2. put the null customer last
    """

    result = sort_findings(
        sample_df,
        "Customer",
    )

    assert result["finding_id"].tolist()[-1] == (
        "F-DDDD0004"
    )


def test_sort_findings_unrecognized_key_returns_unchanged(
    sample_df,
):
    result = sort_findings(
        sample_df,
        "Not A Real Option",
    )

    assert result["finding_id"].tolist() == (
        sample_df["finding_id"].tolist()
    )


# =========================================================
# GENERATE AI EXPLANATION BUTTON
# Streamlit AppTest smoke test
# =========================================================


def test_generate_button_appears_for_confirmed_finding_without_explanation(
    monkeypatch,
):
    """
    Verify that:

    1. A CONFIRMED finding without an AI explanation displays
       the "Generate AI explanation" button.

    2. Clicking the button calls the frontend API client.

    3. The returned explanation and recommendation are merged
       into the finding.

    Current architecture:

        Streamlit app
            |
            v
        frontend.api_client
            |
            v
        FastAPI backend
            |
            v
        engine / Supabase

    Therefore the test patches frontend.api_client rather than
    the engine layer.
    """

    import frontend.api_client as api_client_module

    # ---------------------------------------------------------
    # Confirmed finding without AI output
    # ---------------------------------------------------------

    confirmed_finding = {
        "finding_id": "F-BTN-0001",
        "audit_run_id": "RUN-BTN",
        "control_id": "SCREENING_001",
        "customer_id": "CUST999",
        "severity": "HIGH",
        "assessment_status": "FAIL",
        "finding_status": "CONFIRMED",
        "expected": (
            "Customer must have a CLEAR screening result."
        ),
        "actual": (
            "Screening result is PENDING."
        ),
        "evidence": {
            "customer_id": "CUST999",
            "screening_status": "PENDING",
        },
        "policy_references": [
            {
                "policy_id": "SCREENING-POLICY-001",
                "version": "1.0",
                "section": "Requirements",
            }
        ],
        "reviewed_by": "tester",
        "review_timestamp": (
            "2026-08-23T00:00:00Z"
        ),
        "reviewer_notes": None,
        "ai_explanation": None,
        "ai_recommendation": None,
    }

    # ---------------------------------------------------------
    # Fake backend: run audit
    # ---------------------------------------------------------

    monkeypatch.setattr(
        api_client_module,
        "run_audit",
        lambda: {
            "audit_run_id": "RUN-BTN",
        },
    )

    # ---------------------------------------------------------
    # Fake backend: get findings
    # ---------------------------------------------------------

    monkeypatch.setattr(
        api_client_module,
        "get_findings",
        lambda audit_run_id=None: [
            confirmed_finding
        ],
    )

    # ---------------------------------------------------------
    # Fake backend: get evaluation
    # ---------------------------------------------------------

    monkeypatch.setattr(
        api_client_module,
        "get_evaluation",
        lambda audit_run_id: None,
    )

    # ---------------------------------------------------------
    # Fake backend: generate AI explanation
    # ---------------------------------------------------------

    def fake_generate_ai_explanation(
        finding_id,
    ):
        assert finding_id == "F-BTN-0001"

        return {
            "finding_id": finding_id,
            "ai_explanation": (
                "Fake grounded explanation."
            ),
            "ai_recommendation": (
                "Fake recommendation."
            ),
        }

    monkeypatch.setattr(
        api_client_module,
        "generate_ai_explanation",
        fake_generate_ai_explanation,
    )

    # ---------------------------------------------------------
    # Start Streamlit AppTest
    # ---------------------------------------------------------
    #
    # IMPORTANT:
    #
    # The actual Streamlit app is located at:
    #
    #     frontend/app.py
    #
    # Since this test file is:
    #
    #     tests/test_app.py
    #
    # the correct relative path is:
    #
    #     ../frontend/app.py
    #
    # NOT:
    #
    #     ../app.py
    #
    # ---------------------------------------------------------

    at = AppTest.from_file(
        "../frontend/app.py"
    )

    # ---------------------------------------------------------
    # Open the finding detail view
    # ---------------------------------------------------------

    at.session_state["view"] = "detail"
    at.session_state["selected_finding_id"] = (
        "F-BTN-0001"
    )

    at.run()

    # ---------------------------------------------------------
    # The app itself must not crash
    # ---------------------------------------------------------

    assert not at.exception

    # ---------------------------------------------------------
    # Verify Generate AI explanation button appears
    # ---------------------------------------------------------

    generate_buttons = [
        button
        for button in at.button
        if "Generate AI explanation" in button.label
    ]

    assert len(generate_buttons) == 1

    # ---------------------------------------------------------
    # Click the button
    # ---------------------------------------------------------

    generate_buttons[0].click().run()

    # ---------------------------------------------------------
    # App must still not crash
    # ---------------------------------------------------------

    assert not at.exception

    # ---------------------------------------------------------
    # Verify returned AI output was merged
    # into the finding
    # ---------------------------------------------------------

    assert (
        confirmed_finding["ai_explanation"]
        == "Fake grounded explanation."
    )

    assert (
        confirmed_finding["ai_recommendation"]
        == "Fake recommendation."
    )