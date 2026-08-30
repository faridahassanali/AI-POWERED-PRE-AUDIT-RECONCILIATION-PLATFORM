"""
Streamlit UI — Findings, Review Actions, and Dashboard.

Frontend phase deliverables:
    Person A:
        - Findings list
        - Finding detail view

    Person B:
        - Human review actions
        - Confirm / Reject
        - Reviewer notes

    Person C:
        - Dashboard summary
        - TP / FP / FN
        - Precision / Recall / F1
        - Findings by severity

Data source
-----------
Findings are loaded through the FastAPI backend (backend/main.py):

    frontend.api_client.run_audit()      -> POST /audit-runs/execute
    frontend.api_client.get_findings()   -> GET  /findings

The backend runs the deterministic audit pipeline and persists the
audit run + findings to Supabase itself. No mock findings are used.

Evaluation metrics (TP/FP/FN, Precision/Recall/F1) are fetched via
frontend.api_client.get_evaluation() -> GET /audit-runs/{id}/evaluation.

Persistence
-----------
Confirm / Reject decisions go through PATCH /findings/{id}, which the
backend persists to Supabase.

AI explanation generation (Stage 3) goes through
POST /findings/{id}/ai-explanation.
"""

from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import streamlit as st


# =========================================================
# PROJECT ROOT
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# =========================================================
# API CLIENT
# =========================================================

from frontend.api_client import (
    BackendError,
    generate_ai_explanation,
    get_evaluation,
    get_findings,
    run_audit,
    update_finding,
)


# =========================================================
# TESTABLE API FUNCTION RESOLUTION
# =========================================================

def _get_api_function(name: str):
    """
    Return the API function used by the frontend.

    Normally this returns the function imported from
    frontend.api_client.

    During pytest Streamlit AppTest execution, the test may import
    frontend.app first and monkeypatch functions on that module.
    AppTest then executes app.py in a separate module namespace.

    In that situation, prefer the already-imported frontend.app
    function so the test's monkeypatch is respected.
    """

    current_module = sys.modules.get(__name__)

    imported_app = sys.modules.get(
        "frontend.app"
    )

    if (
        imported_app is not None
        and imported_app is not current_module
    ):

        patched_function = getattr(
            imported_app,
            name,
            None,
        )

        if callable(patched_function):

            return patched_function

    return globals()[name]


# =========================================================
# REVIEW MERGE FIELDS
# =========================================================

REVIEW_MERGE_FIELDS = {
    "finding_status",
    "reviewed_by",
    "reviewer_notes",
    "review_timestamp",
    "ai_explanation",
    "ai_recommendation",
}


# =========================================================
# BRANDING / LOGO
# =========================================================

APP_DIR = Path(__file__).parent

LOGO_PATH = (
    APP_DIR
    / "assets"
    / "logo.png"
)

_LOGO_AVAILABLE = LOGO_PATH.exists()


def render_page_header(
    title: str,
    fallback_icon: str,
) -> None:
    """
    Render a page title with the bank logo beside it.

    Falls back to the emoji if assets/logo.png does not exist.
    """

    if _LOGO_AVAILABLE:

        logo_col, title_col = st.columns(
            [1, 6],
            vertical_alignment="center",
        )

        with logo_col:

            st.image(
                str(LOGO_PATH),
                width=120,
            )

        with title_col:

            st.title(title)

    else:

        st.title(
            f"{fallback_icon} {title}"
        )


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Pre-Audit Reconciliation",
    page_icon=(
        str(LOGO_PATH)
        if _LOGO_AVAILABLE
        else "🔎"
    ),
    layout="wide",
)


# =========================================================
# THEME / PALETTE
# =========================================================

BRAND_CRITICAL = "#A8321A"
BRAND_HIGH = "#D9531E"
BRAND_MEDIUM = "#E8A33D"
BRAND_LOW = "#F0C978"

SEVERITY_CHART_COLORS = {
    "CRITICAL": BRAND_CRITICAL,
    "HIGH": BRAND_HIGH,
    "MEDIUM": BRAND_MEDIUM,
    "LOW": BRAND_LOW,
}

SEVERITY_LEGEND = [
    (
        "CRITICAL",
        BRAND_CRITICAL,
        "Needs immediate attention",
    ),
    (
        "HIGH",
        BRAND_HIGH,
        "High priority — review soon",
    ),
    (
        "MEDIUM",
        BRAND_MEDIUM,
        "Moderate priority",
    ),
    (
        "LOW",
        BRAND_LOW,
        "Low priority / minor",
    ),
]

STATUS_CHART_COLORS = {
    "REVIEW": BRAND_MEDIUM,
    "CONFIRMED": BRAND_HIGH,
    "REJECTED": BRAND_CRITICAL,
}

STATUS_CHART_FALLBACK_COLOR = "#B8AFA0"

STATUS_LEGEND = [
    (
        "REVIEW",
        BRAND_MEDIUM,
        "Awaiting a reviewer decision",
    ),
    (
        "CONFIRMED",
        BRAND_HIGH,
        "Confirmed by a human reviewer",
    ),
    (
        "REJECTED",
        BRAND_CRITICAL,
        "Rejected by a human reviewer",
    ),
]


def render_color_legend(
    items: list[tuple[str, str, str]],
) -> None:
    """
    Render a small legend under a chart.
    """

    swatches = "".join(
        f"""
        <div style="display:flex;align-items:center;gap:6px;
                    margin:2px 14px 2px 0;">
            <span style="display:inline-block;width:12px;height:12px;
                         border-radius:3px;background:{color};
                         flex-shrink:0;"></span>
            <span style="font-size:0.85rem;">
                <b>{label}</b> — {description}
            </span>
        </div>
        """
        for label, color, description in items
    )

    st.markdown(
        f"""
        <div style="display:flex;flex-wrap:wrap;
                    padding:8px 2px 4px 2px;">
            {swatches}
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# DATA ACCESS
# =========================================================

def _get_page_load_idempotency_key() -> str:
    """
    Return one stable idempotency key per browser session.

    The key is reused across Streamlit reruns.
    """

    if (
        "page_load_idempotency_key"
        not in st.session_state
    ):

        st.session_state.page_load_idempotency_key = (
            f"page-load-{uuid.uuid4().hex}"
        )

    return st.session_state.page_load_idempotency_key


@st.cache_resource(
    show_spinner="Running audit pipeline..."
)
def load_pipeline_result(
    idempotency_key: str,
):
    """
    Execute/load the audit run and retrieve its findings/evaluation.

    The idempotency key is part of the cache key so the same browser
    session does not accidentally execute the audit repeatedly.
    """

    audit_run_id = ""

    findings: list[dict] = []

    evaluation = None

    try:

        run_result = _get_api_function(
            "run_audit"
        )(
            idempotency_key=idempotency_key
        )

        if not isinstance(
            run_result,
            dict,
        ):

            run_result = {}

        if (
            run_result.get("status")
            == "duplicate"
        ):

            audit_run = (
                run_result.get(
                    "audit_run",
                    {},
                )
                or {}
            )

            audit_run_id = audit_run.get(
                "audit_run_id",
                "",
            )

        else:

            audit_run_id = run_result.get(
                "audit_run_id",
                "",
            )

        if audit_run_id:

            findings = _get_api_function(
                "get_findings"
            )(
                audit_run_id=audit_run_id
            )

        else:

            findings = []

    except BackendError as exc:

        st.error(
            f"Backend error while loading findings: {exc}"
        )

    except Exception as exc:

        st.error(
            "Couldn't reach the backend. "
            "Make sure uvicorn is running "
            f"on {os.environ.get('APP_API_BASE', 'http://127.0.0.1:8000')} "
            "with the same APP_API_KEYS set. "
            f"({exc})"
        )

    if audit_run_id:

        try:

            evaluation = _get_api_function(
                "get_evaluation"
            )(
                audit_run_id
            )

        except BackendError:

            evaluation = None

        except Exception:

            evaluation = None

    return SimpleNamespace(
        generated_findings=findings,
        evaluation=evaluation,
    )


# =========================================================
# FINDING HELPERS
# =========================================================

def get_finding_by_id(
    findings: list[dict],
    finding_id: str,
) -> dict | None:
    """
    Return one finding by finding_id.
    """

    for finding in findings:

        if finding.get(
            "finding_id"
        ) == finding_id:

            return finding

    return None


# =========================================================
# SEVERITY STYLING
# =========================================================

SEVERITY_ORDER = {
    "CRITICAL": 0,
    "HIGH": 1,
    "MEDIUM": 2,
    "LOW": 3,
}

SEVERITY_COLOR = {
    "CRITICAL": "🔴",
    "HIGH": "🟠",
    "MEDIUM": "🟡",
    "LOW": "🟢",
}


# =========================================================
# PURE HELPERS
# =========================================================

def filter_findings(
    df: pd.DataFrame,
    selected_controls: list[str],
    selected_severities: list[str],
    selected_statuses: list[str],
    search: str = "",
) -> pd.DataFrame:
    """
    Apply control / severity / status filters and optional search.
    """

    filtered = df[
        df["control_id"].isin(
            selected_controls
        )
        & df["severity"].isin(
            selected_severities
        )
        & df["finding_status"].isin(
            selected_statuses
        )
    ].copy()

    if search:

        needle = search.strip().lower()

        finding_id_match = (
            filtered["finding_id"]
            .fillna("")
            .astype(str)
            .str.lower()
            .str.contains(
                needle,
                regex=False,
            )
        )

        customer_id_match = (
            filtered["customer_id"]
            .fillna("")
            .astype(str)
            .str.lower()
            .str.contains(
                needle,
                regex=False,
            )
        )

        filtered = filtered[
            finding_id_match
            | customer_id_match
        ]

    return filtered


def sort_findings(
    df: pd.DataFrame,
    sort_by: str,
) -> pd.DataFrame:
    """
    Sort findings according to selected criterion.
    """

    if sort_by == "Severity (Critical first)":

        rank = df["severity"].map(
            lambda severity: SEVERITY_ORDER.get(
                severity,
                99,
            )
        )

        return (
            df.assign(_rank=rank)
            .sort_values("_rank")
            .drop(columns="_rank")
        )

    if sort_by == "Finding ID":

        return df.sort_values(
            "finding_id"
        )

    if sort_by == "Control":

        return df.sort_values(
            "control_id"
        )

    if sort_by == "Customer":

        return df.sort_values(
            "customer_id",
            na_position="last",
        )

    return df


def get_nav_state(
    ids: list[str],
    current_id: str,
) -> dict | None:
    """
    Return Previous / Next navigation state.
    """

    if current_id not in ids:

        return None

    position = ids.index(
        current_id
    )

    return {
        "position": position,
        "total": len(ids),
        "prev_id": (
            ids[position - 1]
            if position > 0
            else None
        ),
        "next_id": (
            ids[position + 1]
            if position < len(ids) - 1
            else None
        ),
    }


# =========================================================
# REVIEWER IDENTITY
# =========================================================

def render_reviewer_identity() -> str:
    """
    Collect reviewer name once in the sidebar.
    """

    st.sidebar.header(
        "Reviewer"
    )

    return st.sidebar.text_input(
        "Reviewer name",
        key="reviewer_name",
        placeholder="e.g. Sherry",
    )


# =========================================================
# SIDEBAR
# =========================================================

def render_sidebar(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Render filters/search/sorting.
    """

    st.sidebar.header(
        "Filters"
    )

    search = st.sidebar.text_input(
        "Search by Finding ID or Customer ID",
        placeholder=(
            "e.g. F-547D4613 or CUST100016"
        ),
        key="filter_search",
    )

    control_options = sorted(
        df["control_id"]
        .dropna()
        .unique()
    )

    selected_controls = (
        st.sidebar.multiselect(
            "Control",
            control_options,
            default=control_options,
            key="filter_controls",
        )
    )

    severity_options = sorted(
        df["severity"]
        .dropna()
        .unique(),
        key=lambda severity: SEVERITY_ORDER.get(
            severity,
            99,
        ),
    )

    selected_severities = (
        st.sidebar.multiselect(
            "Severity",
            severity_options,
            default=severity_options,
            key="filter_severities",
        )
    )

    status_options = sorted(
        df["finding_status"]
        .dropna()
        .unique()
    )

    selected_statuses = (
        st.sidebar.multiselect(
            "Status",
            status_options,
            default=status_options,
            key="filter_statuses",
        )
    )

    sort_by = st.sidebar.selectbox(
        "Sort by",
        [
            "Severity (Critical first)",
            "Finding ID",
            "Control",
            "Customer",
        ],
        key="sort_by",
    )

    if st.sidebar.button(
        "Reset filters"
    ):

        for key in (
            "filter_search",
            "filter_controls",
            "filter_severities",
            "filter_statuses",
            "sort_by",
        ):

            st.session_state.pop(
                key,
                None,
            )

        st.session_state.filtered_ids = None

        st.rerun()

    filtered = filter_findings(
        df=df,
        selected_controls=selected_controls,
        selected_severities=selected_severities,
        selected_statuses=selected_statuses,
        search=search,
    )

    return sort_findings(
        filtered,
        sort_by,
    )


# =========================================================
# FINDINGS LIST SUMMARY
# =========================================================

def render_findings_summary(
    df: pd.DataFrame,
    filtered: pd.DataFrame,
) -> None:
    """
    Render compact metrics above findings table.
    """

    counts = filtered[
        "severity"
    ].value_counts()

    cols = st.columns(5)

    cols[0].metric(
        "Showing",
        f"{len(filtered)} / {len(df)}",
    )

    cols[1].metric(
        f"{SEVERITY_COLOR['CRITICAL']} Critical",
        int(
            counts.get(
                "CRITICAL",
                0,
            )
        ),
    )

    cols[2].metric(
        f"{SEVERITY_COLOR['HIGH']} High",
        int(
            counts.get(
                "HIGH",
                0,
            )
        ),
    )

    cols[3].metric(
        f"{SEVERITY_COLOR['MEDIUM']} Medium",
        int(
            counts.get(
                "MEDIUM",
                0,
            )
        ),
    )

    cols[4].metric(
        f"{SEVERITY_COLOR['LOW']} Low",
        int(
            counts.get(
                "LOW",
                0,
            )
        ),
    )


# =========================================================
# FINDINGS LIST VIEW
# =========================================================

def render_findings_list(
    findings: list[dict],
) -> None:
    """
    Render findings list with search, filters and sorting.
    """

    render_page_header(
        "Audit Findings",
        "🔎",
    )

    if not findings:

        st.info(
            "No findings were generated by the audit run."
        )

        return

    df = pd.DataFrame(
        findings
    )

    filtered = render_sidebar(
        df
    )

    render_findings_summary(
        df,
        filtered,
    )

    st.divider()

    if filtered.empty:

        st.warning(
            "No findings match the current filters/search."
        )

        return

    st.session_state.filtered_ids = (
        filtered[
            "finding_id"
        ].tolist()
    )

    filtered = filtered.copy()

    filtered["Severity"] = (
        filtered["severity"].map(
            lambda severity: (
                f"{SEVERITY_COLOR.get(severity, '')} "
                f"{severity}"
            )
        )
    )

    display_cols = {
        "finding_id": "Finding ID",
        "control_id": "Control",
        "customer_id": "Customer",
        "Severity": "Severity",
        "finding_status": "Status",
    }

    table = (
        filtered[
            list(display_cols.keys())
        ]
        .rename(
            columns=display_cols
        )
    )

    event = st.dataframe(
        table,
        width="stretch",
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
    )

    selected_rows = (
        event.selection.rows
        if event.selection
        else []
    )

    if selected_rows:

        selected_finding_id = (
            filtered.iloc[
                selected_rows[0]
            ]["finding_id"]
        )

        st.session_state.selected_finding_id = (
            selected_finding_id
        )

        st.session_state.view = (
            "detail"
        )

        st.rerun()


# =========================================================
# FINDING NAVIGATION
# =========================================================

def render_navigation(
    findings: list[dict],
    finding_id: str,
) -> None:
    """
    Render Back / Previous / Next navigation.
    """

    ids = (
        st.session_state.get(
            "filtered_ids"
        )
        or [
            finding["finding_id"]
            for finding in findings
        ]
    )

    nav_state = get_nav_state(
        ids,
        finding_id,
    )

    nav_col1, nav_col2, nav_col3, nav_col4 = (
        st.columns(
            [2, 1, 1, 2]
        )
    )

    with nav_col1:

        if st.button(
            "← Back to findings list"
        ):

            st.session_state.view = (
                "list"
            )

            st.session_state.selected_finding_id = (
                None
            )

            st.rerun()

    if nav_state is None:

        return

    with nav_col2:

        if st.button(
            "◀ Previous",
            disabled=(
                nav_state["prev_id"]
                is None
            ),
        ):

            st.session_state.selected_finding_id = (
                nav_state["prev_id"]
            )

            st.session_state.view = (
                "detail"
            )

            st.rerun()

    with nav_col3:

        if st.button(
            "Next ▶",
            disabled=(
                nav_state["next_id"]
                is None
            ),
        ):

            st.session_state.selected_finding_id = (
                nav_state["next_id"]
            )

            st.session_state.view = (
                "detail"
            )

            st.rerun()

    with nav_col4:

        st.caption(
            f"Finding "
            f"{nav_state['position'] + 1} "
            f"of "
            f"{nav_state['total']}"
        )


# =========================================================
# FINDING DETAIL
# =========================================================

def render_finding_detail(
    findings: list[dict],
    finding: dict,
) -> None:
    """
    Render full finding detail.
    """

    render_navigation(
        findings,
        finding["finding_id"],
    )

    severity = finding.get(
        "severity",
        "",
    )

    # IMPORTANT:
    # Keep this as a real st.title so AppTest can find it through
    # at.title and tests can verify that the selected finding opened.

    st.title(
        f"{SEVERITY_COLOR.get(severity, '')} "
        f"{finding['finding_id']}"
    )

    tab_overview, tab_evidence, tab_review, tab_ai = (
        st.tabs(
            [
                "Overview",
                "Evidence & Policy",
                "Review Info",
                "AI Output",
            ]
        )
    )

    # =====================================================
    # OVERVIEW
    # =====================================================

    with tab_overview:

        info_cols = st.columns(4)

        info_cols[0].metric(
            "Control",
            finding.get(
                "control_id",
                "—",
            ),
        )

        info_cols[1].metric(
            "Customer",
            finding.get(
                "customer_id"
            )
            or "—",
        )

        info_cols[2].metric(
            "Severity",
            severity or "—",
        )

        info_cols[3].metric(
            "Status",
            finding.get(
                "finding_status",
                "—",
            ),
        )

        st.caption(
            f"Audit Run: "
            f"{finding.get('audit_run_id', '—')}"
        )

        st.divider()

        st.subheader(
            "Expected vs Actual"
        )

        exp_col, act_col = st.columns(2)

        with exp_col:

            st.markdown(
                "**Expected**"
            )

            st.info(
                finding.get(
                    "expected",
                    "—",
                )
            )

        with act_col:

            st.markdown(
                "**Actual**"
            )

            st.warning(
                finding.get(
                    "actual",
                    "—",
                )
            )

        # -------------------------------------------------
        # OTHER FINDINGS FOR SAME CUSTOMER
        # -------------------------------------------------

        customer_id = finding.get(
            "customer_id"
        )

        if customer_id:

            siblings = [
                other
                for other in findings
                if (
                    other.get(
                        "customer_id"
                    )
                    == customer_id
                    and other.get(
                        "finding_id"
                    )
                    != finding.get(
                        "finding_id"
                    )
                )
            ]

            if siblings:

                st.divider()

                st.subheader(
                    f"Other findings for {customer_id}"
                )

                for sibling in siblings:

                    sibling_severity = (
                        sibling.get(
                            "severity",
                            "",
                        )
                    )

                    label = (
                        f"{SEVERITY_COLOR.get(sibling_severity, '')} "
                        f"{sibling['finding_id']} — "
                        f"{sibling['control_id']} "
                        f"({sibling.get('finding_status', '—')})"
                    )

                    if st.button(
                        label,
                        key=(
                            f"jump_"
                            f"{sibling['finding_id']}"
                        ),
                    ):

                        st.session_state.selected_finding_id = (
                            sibling[
                                "finding_id"
                            ]
                        )

                        st.session_state.view = (
                            "detail"
                        )

                        st.rerun()

    # =====================================================
    # EVIDENCE + POLICY
    # =====================================================

    with tab_evidence:

        st.subheader(
            "Evidence"
        )

        evidence = finding.get(
            "evidence"
        ) or {}

        if evidence:

            show_raw = st.toggle(
                "Show raw JSON",
                value=False,
                key=(
                    f"evidence_raw_"
                    f"{finding['finding_id']}"
                ),
            )

            if show_raw:

                st.json(
                    evidence
                )

            else:

                for key, value in evidence.items():

                    st.markdown(
                        f"**{key}:** {value}"
                    )

        else:

            st.caption(
                "No evidence recorded for this finding."
            )

        st.divider()

        st.subheader(
            "Policy References"
        )

        policy_refs = finding.get(
            "policy_references"
        ) or []

        if policy_refs:

            for policy in policy_refs:

                st.markdown(
                    f"**Policy ID:** "
                    f"`{policy.get('policy_id', '—')}`"
                )

                st.markdown(
                    f"**Version:** "
                    f"{policy.get('version', '—')}"
                )

                st.markdown(
                    f"**Section:** "
                    f"{policy.get('section', '—')}"
                )

                st.divider()

        else:

            st.caption(
                "No policy references recorded "
                "for this finding."
            )

    # =====================================================
    # REVIEW INFO + ACTIONS
    # =====================================================

    with tab_review:

        current_status = finding.get(
            "finding_status"
        )

        if current_status == "REVIEW":

            st.subheader(
                "Review Decision"
            )

            reviewer_name = (
                st.session_state.get(
                    "reviewer_name",
                    "",
                )
                .strip()
            )

            if not reviewer_name:

                st.warning(
                    "Enter your name in the "
                    "**Reviewer** field in the sidebar "
                    "before confirming or rejecting."
                )

            notes_key = (
                f"notes_{finding['finding_id']}"
            )

            reviewer_notes = st.text_area(
                "Reviewer notes (optional)",
                key=notes_key,
            )

            confirm_col, reject_col = (
                st.columns(2)
            )

            with confirm_col:

                if st.button(
                    "✅ Confirm finding",
                    disabled=not reviewer_name,
                    width="stretch",
                    key=(
                        f"confirm_"
                        f"{finding['finding_id']}"
                    ),
                ):

                    try:

                        updated = _get_api_function(
                            "update_finding"
                        )(
                            finding[
                                "finding_id"
                            ],
                            finding_status="CONFIRMED",
                            reviewed_by=reviewer_name,
                            reviewer_notes=(
                                reviewer_notes
                                or None
                            ),
                        )

                    except BackendError as exc:

                        st.error(
                            f"Could not confirm: {exc}"
                        )

                    else:

                        finding.update(
                            {
                                key: value
                                for key, value
                                in updated.items()
                                if key
                                in REVIEW_MERGE_FIELDS
                            }
                        )

                        st.success(
                            "Finding confirmed."
                        )

                        st.rerun()

            with reject_col:

                if st.button(
                    "❌ Reject finding",
                    disabled=not reviewer_name,
                    width="stretch",
                    key=(
                        f"reject_"
                        f"{finding['finding_id']}"
                    ),
                ):

                    try:

                        updated = _get_api_function(
                            "update_finding"
                        )(
                            finding[
                                "finding_id"
                            ],
                            finding_status="REJECTED",
                            reviewed_by=reviewer_name,
                            reviewer_notes=(
                                reviewer_notes
                                or None
                            ),
                        )

                    except BackendError as exc:

                        st.error(
                            f"Could not reject: {exc}"
                        )

                    else:

                        finding.update(
                            {
                                key: value
                                for key, value
                                in updated.items()
                                if key
                                in REVIEW_MERGE_FIELDS
                            }
                        )

                        st.success(
                            "Finding rejected."
                        )

                        st.rerun()

            st.divider()

        else:

            st.info(
                f"This finding is already "
                f"**{current_status}** — "
                "no further review action is available."
            )

            st.divider()

        rev_col1, rev_col2 = (
            st.columns(2)
        )

        with rev_col1:

            st.markdown(
                f"**Reviewed by:** "
                f"{finding.get('reviewed_by') or '—'}"
            )

            st.markdown(
                f"**Review timestamp:** "
                f"{finding.get('review_timestamp') or '—'}"
            )

        with rev_col2:

            notes = finding.get(
                "reviewer_notes"
            )

            st.markdown(
                "**Reviewer notes:**"
            )

            st.write(
                notes
                if notes
                else "—"
            )

    # =====================================================
    # AI OUTPUT
    # =====================================================

    with tab_ai:

        ai_explanation = finding.get(
            "ai_explanation"
        )

        ai_recommendation = finding.get(
            "ai_recommendation"
        )

        if (
            ai_explanation
            or ai_recommendation
        ):

            if ai_explanation:

                st.markdown(
                    "**Explanation**"
                )

                st.write(
                    ai_explanation
                )

            if ai_recommendation:

                st.markdown(
                    "**Recommendation**"
                )

                st.write(
                    ai_recommendation
                )

        elif (
            finding.get(
                "finding_status"
            )
            == "CONFIRMED"
        ):

            st.caption(
                "Not yet generated for this finding."
            )

            if st.button(
                "🤖 Generate AI explanation",
                width="stretch",
                key=(
                    f"generate_ai_"
                    f"{finding['finding_id']}"
                ),
            ):

                with st.spinner(
                    "Retrieving policy context and calling the LLM..."
                ):

                    try:

                        data = _get_api_function(
                            "generate_ai_explanation"
                        )(
                            finding[
                                "finding_id"
                            ]
                        )

                    except BackendError as exc:

                        st.error(
                            "Could not generate an "
                            f"AI explanation: {exc}"
                        )

                    except Exception as exc:

                        st.error(
                            "Could not generate an "
                            f"AI explanation: {exc}"
                        )

                    else:

                        if isinstance(
                            data,
                            dict,
                        ):

                            if (
                                "ai_explanation"
                                in data
                            ):

                                finding[
                                    "ai_explanation"
                                ] = data[
                                    "ai_explanation"
                                ]

                            if (
                                "ai_recommendation"
                                in data
                            ):

                                finding[
                                    "ai_recommendation"
                                ] = data[
                                    "ai_recommendation"
                                ]

                        st.success(
                            "AI explanation generated."
                        )

                        st.rerun()

        else:

            st.caption(
                "Not yet available — this finding "
                "must be CONFIRMED by a human reviewer "
                "before AI explanation is generated. "
                "Rejected findings never reach the AI stage."
            )


# =========================================================
# DASHBOARD HELPERS
# =========================================================

def _evaluation_value(
    evaluation,
    field: str,
    default=0,
):
    """
    Read a field from an object/dataclass or dictionary.
    """

    if hasattr(
        evaluation,
        field,
    ):

        return getattr(
            evaluation,
            field,
        )

    if isinstance(
        evaluation,
        dict,
    ):

        return evaluation.get(
            field,
            default,
        )

    return default


def _to_display_metric(
    value,
) -> str:
    """
    Format numeric metric values.
    """

    if isinstance(
        value,
        float,
    ):

        return f"{value:.3f}"

    return str(value)


# =========================================================
# DASHBOARD
# =========================================================

def render_dashboard(
    findings: list[dict],
    evaluation,
) -> None:
    """
    Render dashboard.
    """

    render_page_header(
        "Audit Dashboard",
        "📊",
    )

    st.caption(
        "Deterministic audit evaluation and finding summary"
    )

    if evaluation is None:

        st.caption(
            "⚠️ No evaluation has been recorded for this "
            "audit run yet — showing placeholders below."
        )

    # -----------------------------------------------------
    # TOP METRICS
    # -----------------------------------------------------

    tp = _evaluation_value(
        evaluation,
        "true_positives",
    )

    fp = _evaluation_value(
        evaluation,
        "false_positives",
    )

    fn = _evaluation_value(
        evaluation,
        "false_negatives",
    )

    precision = _evaluation_value(
        evaluation,
        "precision",
        0.0,
    )

    recall = _evaluation_value(
        evaluation,
        "recall",
        0.0,
    )

    f1_score = _evaluation_value(
        evaluation,
        "f1_score",
        0.0,
    )

    metric_row_1 = st.columns(3)

    metric_row_1[0].metric(
        "True Positives",
        _to_display_metric(tp),
    )

    metric_row_1[1].metric(
        "False Positives",
        _to_display_metric(fp),
    )

    metric_row_1[2].metric(
        "False Negatives",
        _to_display_metric(fn),
    )

    metric_row_2 = st.columns(3)

    metric_row_2[0].metric(
        "Precision",
        _to_display_metric(precision),
    )

    metric_row_2[1].metric(
        "Recall",
        _to_display_metric(recall),
    )

    metric_row_2[2].metric(
        "F1 Score",
        _to_display_metric(f1_score),
    )

    st.divider()

    # -----------------------------------------------------
    # FINDINGS BY SEVERITY
    # -----------------------------------------------------

    st.subheader(
        "Findings by Severity"
    )

    if findings:

        severity_counts = (
            pd.Series(
                [
                    finding.get(
                        "severity",
                        "UNKNOWN",
                    )
                    for finding in findings
                ]
            )
            .value_counts()
            .reindex(
                [
                    "CRITICAL",
                    "HIGH",
                    "MEDIUM",
                    "LOW",
                ],
                fill_value=0,
            )
        )

        severity_chart_df = (
            severity_counts
            .rename("Findings")
            .to_frame()
            .T
        )

        st.bar_chart(
            severity_chart_df,
            width="stretch",
            color=[
                SEVERITY_CHART_COLORS[
                    "CRITICAL"
                ],
                SEVERITY_CHART_COLORS[
                    "HIGH"
                ],
                SEVERITY_CHART_COLORS[
                    "MEDIUM"
                ],
                SEVERITY_CHART_COLORS[
                    "LOW"
                ],
            ],
        )

        render_color_legend(
            SEVERITY_LEGEND
        )

        severity_table = pd.DataFrame(
            {
                "Severity": [
                    "CRITICAL",
                    "HIGH",
                    "MEDIUM",
                    "LOW",
                ],
                "Count": [
                    int(
                        severity_counts.iloc[
                            index
                        ]
                    )
                    for index in range(
                        len(
                            severity_counts
                        )
                    )
                ],
            }
        )

        st.dataframe(
            severity_table,
            width="stretch",
            hide_index=True,
        )

    else:

        st.info(
            "No findings available."
        )

    st.divider()

    # -----------------------------------------------------
    # FINDINGS BY STATUS
    # -----------------------------------------------------

    st.subheader(
        "Findings by Status"
    )

    if findings:

        status_counts = (
            pd.Series(
                [
                    finding.get(
                        "finding_status",
                        "UNKNOWN",
                    )
                    for finding in findings
                ]
            )
            .value_counts()
        )

        status_chart_df = (
            status_counts
            .rename("Findings")
            .to_frame()
            .T
        )

        status_bar_colors = [
            STATUS_CHART_COLORS.get(
                status,
                STATUS_CHART_FALLBACK_COLOR,
            )
            for status
            in status_chart_df.columns
        ]

        st.bar_chart(
            status_chart_df,
            width="stretch",
            color=status_bar_colors,
        )

        present_statuses = set(
            status_chart_df.columns
        )

        status_legend_items = [
            item
            for item in STATUS_LEGEND
            if item[0]
            in present_statuses
        ]

        other_statuses = (
            present_statuses
            - {
                item[0]
                for item in STATUS_LEGEND
            }
        )

        for other_status in sorted(
            other_statuses
        ):

            status_legend_items.append(
                (
                    other_status,
                    STATUS_CHART_FALLBACK_COLOR,
                    "Other status",
                )
            )

        render_color_legend(
            status_legend_items
        )

    else:

        st.info(
            "No findings available."
        )

    st.divider()

    # -----------------------------------------------------
    # RUN SUMMARY
    # -----------------------------------------------------

    st.subheader(
        "Run Summary"
    )

    summary_cols = st.columns(3)

    summary_cols[0].metric(
        "Generated Findings",
        len(findings),
    )

    summary_cols[1].metric(
        "Unique Controls",
        len(
            {
                finding.get(
                    "control_id"
                )
                for finding in findings
            }
        ),
    )

    summary_cols[2].metric(
        "Unique Customers",
        len(
            {
                finding.get(
                    "customer_id"
                )
                for finding in findings
                if finding.get(
                    "customer_id"
                )
                is not None
            }
        ),
    )

    # -----------------------------------------------------
    # OPTIONAL PER-CONTROL EVALUATION
    # -----------------------------------------------------

    per_control = _evaluation_value(
        evaluation,
        "per_control",
        None,
    )

    if per_control:

        st.divider()

        st.subheader(
            "Evaluation by Control"
        )

        if isinstance(
            per_control,
            dict,
        ):

            rows = []

            for (
                control_id,
                result,
            ) in per_control.items():

                if hasattr(
                    result,
                    "__dict__",
                ):

                    row = {
                        "Control": control_id,
                        "TP": getattr(
                            result,
                            "true_positives",
                            getattr(
                                result,
                                "tp",
                                0,
                            ),
                        ),
                        "FP": getattr(
                            result,
                            "false_positives",
                            getattr(
                                result,
                                "fp",
                                0,
                            ),
                        ),
                        "FN": getattr(
                            result,
                            "false_negatives",
                            getattr(
                                result,
                                "fn",
                                0,
                            ),
                        ),
                        "Precision": getattr(
                            result,
                            "precision",
                            0.0,
                        ),
                        "Recall": getattr(
                            result,
                            "recall",
                            0.0,
                        ),
                        "F1": getattr(
                            result,
                            "f1_score",
                            0.0,
                        ),
                    }

                elif isinstance(
                    result,
                    dict,
                ):

                    row = {
                        "Control": control_id,
                        "TP": result.get(
                            "true_positives",
                            result.get(
                                "tp",
                                0,
                            ),
                        ),
                        "FP": result.get(
                            "false_positives",
                            result.get(
                                "fp",
                                0,
                            ),
                        ),
                        "FN": result.get(
                            "false_negatives",
                            result.get(
                                "fn",
                                0,
                            ),
                        ),
                        "Precision": result.get(
                            "precision",
                            0.0,
                        ),
                        "Recall": result.get(
                            "recall",
                            0.0,
                        ),
                        "F1": result.get(
                            "f1_score",
                            0.0,
                        ),
                    }

                else:

                    continue

                rows.append(row)

            if rows:

                st.dataframe(
                    pd.DataFrame(rows),
                    width="stretch",
                    hide_index=True,
                )


# =========================================================
# MAIN
# =========================================================

def main() -> None:
    """
    Run the Streamlit application.
    """

    # =====================================================
    # SESSION STATE
    # =====================================================

    if "view" not in st.session_state:

        st.session_state.view = "dashboard"

    if "selected_finding_id" not in st.session_state:

        st.session_state.selected_finding_id = None

    if "filtered_ids" not in st.session_state:

        st.session_state.filtered_ids = None

    # =====================================================
    # LOAD PIPELINE RESULT
    # =====================================================

    idempotency_key = _get_page_load_idempotency_key()

    result = load_pipeline_result(
        idempotency_key
    )

    findings = result.generated_findings

    evaluation = result.evaluation

    # =====================================================
    # SIDEBAR
    # =====================================================

    render_reviewer_identity()

    st.sidebar.divider()

    st.sidebar.header(
        "Navigation"
    )

    # IMPORTANT:
    # Detail view belongs to the Findings section.
    # When a finding is already selected, preserve the detail view.
    # The navigation radio only changes the top-level section.

    navigation_index = (
        0
        if st.session_state.view == "dashboard"
        else 1
    )

    selected_view = st.sidebar.radio(
        "Go to",
        [
            "Dashboard",
            "Findings",
        ],
        index=navigation_index,
        key="navigation_view",
    )

    # -----------------------------------------------------
    # HANDLE TOP-LEVEL NAVIGATION
    # -----------------------------------------------------

    if selected_view == "Dashboard":

        # Do not destroy an explicitly selected detail view.
        if (
            st.session_state.view != "detail"
            or not st.session_state.get(
                "selected_finding_id"
            )
        ):

            st.session_state.view = "dashboard"

            st.session_state.selected_finding_id = None

    elif selected_view == "Findings":

        # Preserve detail view if a finding is selected.
        if (
            st.session_state.view == "detail"
            and st.session_state.get(
                "selected_finding_id"
            )
        ):

            st.session_state.view = "detail"

        else:

            st.session_state.view = "list"

    st.sidebar.divider()

    st.sidebar.caption(
        "Findings, confirm/reject actions, and AI explanation "
        "generation all go through the FastAPI backend."
    )

    st.sidebar.divider()

    # =====================================================
    # RE-RUN AUDIT
    # =====================================================

    if st.sidebar.button(
        "🔄 Re-run audit"
    ):

        load_pipeline_result.clear()

        st.session_state.page_load_idempotency_key = (
            f"page-load-{uuid.uuid4().hex}"
        )

        st.session_state.view = "dashboard"

        st.session_state.selected_finding_id = None

        st.session_state.filtered_ids = None

        st.rerun()

    # =====================================================
    # PAGE ROUTING
    # =====================================================

    # -----------------------------------------------------
    # DASHBOARD
    # -----------------------------------------------------

    if st.session_state.view == "dashboard":

        render_dashboard(
            findings,
            evaluation,
        )

        return

    # -----------------------------------------------------
    # FINDING DETAIL
    # -----------------------------------------------------

    selected_finding_id = (
        st.session_state.get(
            "selected_finding_id"
        )
    )

    if (
        st.session_state.view == "detail"
        and selected_finding_id
    ):

        finding = get_finding_by_id(
            findings,
            selected_finding_id,
        )

        if finding is None:

            st.session_state.view = "list"

            st.session_state.selected_finding_id = None

            st.rerun()

            return

        render_finding_detail(
            findings,
            finding,
        )

        return

    # -----------------------------------------------------
    # FINDINGS LIST
    # -----------------------------------------------------

    st.session_state.view = "list"

    render_findings_list(
        findings
    )


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    main()