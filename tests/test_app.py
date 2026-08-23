"""
Tests for app.py (Streamlit Findings List + Finding Detail view).

Only the pure functions (filter_findings, get_finding_by_id,
get_nav_state) are tested directly here — they contain zero
Streamlit calls, so they run under plain pytest with no app
server needed.

Rendering functions (render_findings_list, render_finding_detail,
render_sidebar, render_navigation) are NOT covered here: they call
st.* widgets directly and would need streamlit.testing.v1.AppTest
for an end-to-end smoke test. That's a reasonable follow-up, not
required for this MVP.
"""

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from app import filter_findings, get_finding_by_id, get_nav_state, sort_findings


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
            "expected": "Customer must have a CLEAR screening result.",
            "actual": "Screening result is not CLEAR.",
            "evidence": {"screening_status": "PENDING"},
            "policy_references": [
                {"policy_id": "POL-01", "version": "1.0", "section": "3.2"}
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
            "expected": "Customer record must contain an Arabic-script name.",
            "actual": "Arabic name is missing.",
            "evidence": {"name_ar": None},
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
            "expected": "Customer must have a CLEAR screening result.",
            "actual": "Screening result is not CLEAR.",
            "evidence": {"screening_status": "REJECTED"},
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
            "expected": "Risk score must be recalculated within 30 days.",
            "actual": "Risk score is stale.",
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
    finding = get_finding_by_id(sample_findings, "F-BBBB0002")
    assert finding is not None
    assert finding["control_id"] == "ARABIC_NAME_001"


def test_get_finding_by_id_not_found(sample_findings):
    assert get_finding_by_id(sample_findings, "F-NOPE9999") is None


def test_get_finding_by_id_empty_list():
    assert get_finding_by_id([], "F-AAAA0001") is None


# =========================================================
# filter_findings — control/severity/status filters
# =========================================================

def test_filter_findings_no_filters_returns_all(sample_df):
    result = filter_findings(
        sample_df,
        selected_controls=sample_df["control_id"].unique().tolist(),
        selected_severities=sample_df["severity"].unique().tolist(),
        selected_statuses=sample_df["finding_status"].unique().tolist(),
    )
    assert len(result) == 4


def test_filter_findings_by_single_control(sample_df):
    result = filter_findings(
        sample_df,
        selected_controls=["SCREENING_001"],
        selected_severities=sample_df["severity"].unique().tolist(),
        selected_statuses=sample_df["finding_status"].unique().tolist(),
    )
    assert set(result["finding_id"]) == {"F-AAAA0001", "F-CCCC0003"}


def test_filter_findings_by_severity(sample_df):
    result = filter_findings(
        sample_df,
        selected_controls=sample_df["control_id"].unique().tolist(),
        selected_severities=["CRITICAL"],
        selected_statuses=sample_df["finding_status"].unique().tolist(),
    )
    assert result["finding_id"].tolist() == ["F-AAAA0001"]


def test_filter_findings_by_status(sample_df):
    result = filter_findings(
        sample_df,
        selected_controls=sample_df["control_id"].unique().tolist(),
        selected_severities=sample_df["severity"].unique().tolist(),
        selected_statuses=["REJECTED"],
    )
    assert result["finding_id"].tolist() == ["F-DDDD0004"]


def test_filter_findings_combined_filters_narrow_correctly(sample_df):
    result = filter_findings(
        sample_df,
        selected_controls=["SCREENING_001"],
        selected_severities=["HIGH"],
        selected_statuses=["CONFIRMED"],
    )
    assert result["finding_id"].tolist() == ["F-CCCC0003"]


def test_filter_findings_empty_selection_returns_empty(sample_df):
    result = filter_findings(
        sample_df,
        selected_controls=[],
        selected_severities=sample_df["severity"].unique().tolist(),
        selected_statuses=sample_df["finding_status"].unique().tolist(),
    )
    assert result.empty


# =========================================================
# filter_findings — search
# =========================================================

def test_filter_findings_search_by_finding_id_case_insensitive(sample_df):
    result = filter_findings(
        sample_df,
        selected_controls=sample_df["control_id"].unique().tolist(),
        selected_severities=sample_df["severity"].unique().tolist(),
        selected_statuses=sample_df["finding_status"].unique().tolist(),
        search="f-aaaa0001",
    )
    assert result["finding_id"].tolist() == ["F-AAAA0001"]


def test_filter_findings_search_by_customer_id(sample_df):
    result = filter_findings(
        sample_df,
        selected_controls=sample_df["control_id"].unique().tolist(),
        selected_severities=sample_df["severity"].unique().tolist(),
        selected_statuses=sample_df["finding_status"].unique().tolist(),
        search="100016",
    )
    assert result["finding_id"].tolist() == ["F-CCCC0003"]


def test_filter_findings_search_with_null_customer_id_does_not_crash(sample_df):
    # F-DDDD0004 has customer_id=None — must not raise on .str ops.
    result = filter_findings(
        sample_df,
        selected_controls=sample_df["control_id"].unique().tolist(),
        selected_severities=sample_df["severity"].unique().tolist(),
        selected_statuses=sample_df["finding_status"].unique().tolist(),
        search="dddd",
    )
    assert result["finding_id"].tolist() == ["F-DDDD0004"]


def test_filter_findings_search_no_match_returns_empty(sample_df):
    result = filter_findings(
        sample_df,
        selected_controls=sample_df["control_id"].unique().tolist(),
        selected_severities=sample_df["severity"].unique().tolist(),
        selected_statuses=sample_df["finding_status"].unique().tolist(),
        search="does-not-exist",
    )
    assert result.empty


# =========================================================
# get_nav_state
# =========================================================

def test_get_nav_state_middle_position():
    ids = ["F-1", "F-2", "F-3"]
    state = get_nav_state(ids, "F-2")
    assert state == {
        "position": 1,
        "total": 3,
        "prev_id": "F-1",
        "next_id": "F-3",
    }


def test_get_nav_state_first_position_has_no_prev():
    ids = ["F-1", "F-2", "F-3"]
    state = get_nav_state(ids, "F-1")
    assert state["prev_id"] is None
    assert state["next_id"] == "F-2"


def test_get_nav_state_last_position_has_no_next():
    ids = ["F-1", "F-2", "F-3"]
    state = get_nav_state(ids, "F-3")
    assert state["next_id"] is None
    assert state["prev_id"] == "F-2"


def test_get_nav_state_single_item_has_no_prev_or_next():
    state = get_nav_state(["F-1"], "F-1")
    assert state["prev_id"] is None
    assert state["next_id"] is None
    assert state["total"] == 1


def test_get_nav_state_id_not_in_list_returns_none():
    # e.g. stale selected_finding_id after filters changed.
    assert get_nav_state(["F-1", "F-2"], "F-STALE") is None


# =========================================================
# sort_findings
# =========================================================

def test_sort_findings_by_severity_critical_first(sample_df):
    result = sort_findings(sample_df, "Severity (Critical first)")
    assert result["severity"].tolist() == ["CRITICAL", "HIGH", "MEDIUM", "LOW"]


def test_sort_findings_by_finding_id(sample_df):
    result = sort_findings(sample_df, "Finding ID")
    assert result["finding_id"].tolist() == sorted(sample_df["finding_id"])


def test_sort_findings_by_control(sample_df):
    result = sort_findings(sample_df, "Control")
    assert result["control_id"].tolist() == sorted(sample_df["control_id"])


def test_sort_findings_by_customer_puts_null_last(sample_df):
    # F-DDDD0004 has customer_id=None and must not crash the sort,
    # and should not appear before real customer ids.
    result = sort_findings(sample_df, "Customer")
    assert result["finding_id"].tolist()[-1] == "F-DDDD0004"


def test_sort_findings_unrecognized_key_returns_unchanged(sample_df):
    result = sort_findings(sample_df, "Not A Real Option")
    assert result["finding_id"].tolist() == sample_df["finding_id"].tolist()


# =========================================================
# GENERATE AI EXPLANATION BUTTON (AppTest smoke test)
# =========================================================
#
# Unlike the pure-function tests above, this drives the actual
# Streamlit widget tree via streamlit.testing.v1.AppTest, since the
# button lives inside render_finding_detail() (a rendering function).
# app.generate_ai_explanation_for_finding is monkeypatched with a
# fake so this never makes a real Groq/Gemini call or needs API keys.

def test_generate_button_appears_for_confirmed_finding_without_explanation(monkeypatch):

    import types

    import engine.ai_explanation_pipeline as pipeline_module
    import engine.audit_pipeline as audit_pipeline_module
    import engine.policy_registry as policy_registry_module

    confirmed_finding = {
        "finding_id": "F-BTN-0001",
        "audit_run_id": "RUN-BTN",
        "control_id": "SCREENING_001",
        "customer_id": "CUST999",
        "severity": "HIGH",
        "assessment_status": "FAIL",
        "finding_status": "CONFIRMED",
        "expected": "Customer must have a CLEAR screening result.",
        "actual": "Screening result is PENDING.",
        "evidence": {"customer_id": "CUST999", "screening_status": "PENDING"},
        "policy_references": [
            {"policy_id": "SCREENING-POLICY-001", "version": "1.0", "section": "Requirements"}
        ],
        "reviewed_by": "tester",
        "review_timestamp": "2026-08-23T00:00:00Z",
        "reviewer_notes": None,
        "ai_explanation": None,
        "ai_recommendation": None,
    }

    # app.py execs fresh inside AppTest's own sandbox -- patching an
    # already-imported `app` module object does nothing there.
    # Patching the SOURCE modules works, because app.py's
    # `from x import y` statements re-resolve against these module
    # objects' current attributes on every fresh exec, and Python's
    # module cache (sys.modules) is shared process-wide.
    monkeypatch.setattr(
        audit_pipeline_module,
        "run_audit",
        lambda: types.SimpleNamespace(
            generated_findings=[confirmed_finding],
            evaluation={},
        ),
    )
    monkeypatch.setattr(
        policy_registry_module,
        "load_policy_registry",
        lambda data_dir: object(),
    )

    def _fake_generate(finding, registry, primary=None, fallback=None):
        finding["ai_explanation"] = "Fake grounded explanation."
        finding["ai_recommendation"] = "Fake recommendation."
        return types.SimpleNamespace(
            succeeded=True,
            error=None,
            finding_id=finding["finding_id"],
        )

    monkeypatch.setattr(
        pipeline_module,
        "generate_ai_explanation_for_finding",
        _fake_generate,
    )

    at = AppTest.from_file("../app.py")
    at.session_state["view"] = "detail"
    at.session_state["selected_finding_id"] = "F-BTN-0001"
    at.run()

    assert not at.exception

    generate_buttons = [
        b for b in at.button if "Generate AI explanation" in b.label
    ]
    assert len(generate_buttons) == 1

    generate_buttons[0].click().run()

    assert not at.exception
    assert confirmed_finding["ai_explanation"] == "Fake grounded explanation."
    assert confirmed_finding["ai_recommendation"] == "Fake recommendation."