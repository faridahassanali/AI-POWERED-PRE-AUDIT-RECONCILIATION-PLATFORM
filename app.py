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
Findings and evaluation metrics are loaded from the real
engine.audit_pipeline.run_audit() result, run through
engine.audit_orchestration.run_audit_and_persist() so the audit run,
findings, and evaluation metrics are written to Supabase whenever
it's configured.

No mock findings are used.

Persistence
-----------
Every action that changes state is written to Supabase best-effort:
if SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY aren't set, the write is
skipped silently and the UI keeps working exactly as before —
persistence is never a hard requirement for using the app.

    - A fresh audit run           -> audit_runs, findings,
                                      audit_evaluations
    - A Confirm/Reject decision   -> finding_reviews, and (for
                                      Confirm only) an immediate
                                      deterministic Stage 2
                                      explanation -> finding_explanations
    - A generated AI explanation  -> ai_outputs

Backend/API integration can replace load_pipeline_result() later
without changing the UI structure.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from engine.ai_explanation_pipeline import generate_ai_explanation_for_finding
from engine.audit_orchestration import run_audit_and_persist
from engine.data_loader import DATA_DIR
from engine.finding_explainer import explain_finding
from engine.finding_review import (
    confirm_finding,
    reject_finding,
)
from engine.persistence import (
    PersistenceNotConfigured,
    write_ai_output,
    write_finding_explanation,
    write_finding_review,
)
from engine.policy_registry import load_policy_registry


# =========================================================
# BRANDING / LOGO
# =========================================================
# Put the bank logo file at assets/logo.png (next to this app.py) and it
# will replace the emoji icons automatically. If the file isn't there
# yet, the UI falls back to the emoji so the app never breaks.

APP_DIR = Path(__file__).parent
LOGO_PATH = APP_DIR / "assets" / "logo.png"
_LOGO_AVAILABLE = LOGO_PATH.exists()


def render_page_header(title: str, fallback_icon: str) -> None:
    """
    Render a page title with the bank logo beside it.

    Falls back to `fallback_icon + title` (the old emoji-based
    style) if assets/logo.png hasn't been added to the repo yet.
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
# Colors pulled from the app logo (gold top / orange-red bottom / near-black
# line). Kept muted/desaturated on purpose so the UI stays clean instead of
# looking "poppy". These are UI-only constants — no business logic here.

BRAND_CRITICAL = "#A8321A"   # deep brick red — most intense, most urgent
BRAND_HIGH = "#D9531E"       # primary orange-red (from the logo bottom)
BRAND_MEDIUM = "#E8A33D"     # muted gold (from the logo top)
BRAND_LOW = "#F0C978"        # light gold/cream — least intense, least urgent

# A single readable gradient: darkest/most saturated = most urgent,
# lightest = least urgent. Same brand family end-to-end, so every bar
# still visually "belongs" together instead of looking like unrelated
# random colors.
SEVERITY_CHART_COLORS = {
    "CRITICAL": BRAND_CRITICAL,
    "HIGH": BRAND_HIGH,
    "MEDIUM": BRAND_MEDIUM,
    "LOW": BRAND_LOW,
}

SEVERITY_LEGEND = [
    ("CRITICAL", BRAND_CRITICAL, "Needs immediate attention"),
    ("HIGH", BRAND_HIGH, "High priority — review soon"),
    ("MEDIUM", BRAND_MEDIUM, "Moderate priority"),
    ("LOW", BRAND_LOW, "Low priority / minor"),
]

# Same intensity logic applied to review status: darker = further along
# a "needs action" scale, lighter = settled/no action needed. Any status
# not listed here falls back to the neutral color below.
STATUS_CHART_COLORS = {
    "REVIEW": BRAND_MEDIUM,
    "CONFIRMED": BRAND_HIGH,
    "REJECTED": BRAND_CRITICAL,
}

STATUS_CHART_FALLBACK_COLOR = "#B8AFA0"  # neutral warm gray for any other status

STATUS_LEGEND = [
    ("REVIEW", BRAND_MEDIUM, "Awaiting a reviewer decision"),
    ("CONFIRMED", BRAND_HIGH, "Confirmed by a human reviewer"),
    ("REJECTED", BRAND_CRITICAL, "Rejected by a human reviewer"),
]


def render_color_legend(items: list[tuple[str, str, str]]) -> None:
    """
    Render a small "what does this color mean" legend under a chart.

    items: list of (label, hex_color, description) tuples.
    Pure display helper — does not touch any pipeline data.
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

@st.cache_resource(show_spinner="Running audit pipeline...")
def load_pipeline_result():
    """
    Run the real deterministic audit pipeline once per session,
    routed through run_audit_and_persist() so the audit run,
    findings, and evaluation metrics are written to Supabase
    whenever it's configured (best-effort -- the pipeline result is
    identical either way).

    The same result object supplies:
        - generated findings
        - evaluation metrics
        - audit trace
        - report
    """
    orchestrated = run_audit_and_persist()
    return orchestrated.pipeline_result


@st.cache_resource(show_spinner="Loading policy registry...")
def load_registry():
    """
    Load the Policy Registry once per session -- required by
    generate_ai_explanation_for_finding() to resolve a finding's
    policy_context via the RAG bridge before calling the LLM.
    """
    return load_policy_registry(DATA_DIR)


def get_finding_by_id(
    findings: list[dict],
    finding_id: str,
) -> dict | None:
    """Return one finding by finding_id."""

    for finding in findings:
        if finding["finding_id"] == finding_id:
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
    Apply control / severity / status filters and optional
    search across finding_id and customer_id.
    """

    filtered = df[
        df["control_id"].isin(selected_controls)
        & df["severity"].isin(selected_severities)
        & df["finding_status"].isin(selected_statuses)
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
            finding_id_match | customer_id_match
        ]

    return filtered


def sort_findings(
    df: pd.DataFrame,
    sort_by: str,
) -> pd.DataFrame:
    """Sort findings according to the selected criterion."""

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

    position = ids.index(current_id)

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
    """Collect reviewer name once in the sidebar."""

    st.sidebar.header("Reviewer")

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
    Render filters/search/sorting and return filtered findings.
    """

    st.sidebar.header("Filters")

    search = st.sidebar.text_input(
        "Search by Finding ID or Customer ID",
        placeholder="e.g. F-547D4613 or CUST100016",
        key="filter_search",
    )

    control_options = sorted(
        df["control_id"].dropna().unique()
    )

    selected_controls = st.sidebar.multiselect(
        "Control",
        control_options,
        default=control_options,
        key="filter_controls",
    )

    severity_options = sorted(
        df["severity"].dropna().unique(),
        key=lambda severity: SEVERITY_ORDER.get(
            severity,
            99,
        ),
    )

    selected_severities = st.sidebar.multiselect(
        "Severity",
        severity_options,
        default=severity_options,
        key="filter_severities",
    )

    status_options = sorted(
        df["finding_status"].dropna().unique()
    )

    selected_statuses = st.sidebar.multiselect(
        "Status",
        status_options,
        default=status_options,
        key="filter_statuses",
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

    if st.sidebar.button("Reset filters"):
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
    """Render compact metrics above the findings table."""

    counts = filtered["severity"].value_counts()

    cols = st.columns(5)

    cols[0].metric(
        "Showing",
        f"{len(filtered)} / {len(df)}",
    )

    cols[1].metric(
        f"{SEVERITY_COLOR['CRITICAL']} Critical",
        int(counts.get("CRITICAL", 0)),
    )

    cols[2].metric(
        f"{SEVERITY_COLOR['HIGH']} High",
        int(counts.get("HIGH", 0)),
    )

    cols[3].metric(
        f"{SEVERITY_COLOR['MEDIUM']} Medium",
        int(counts.get("MEDIUM", 0)),
    )

    cols[4].metric(
        f"{SEVERITY_COLOR['LOW']} Low",
        int(counts.get("LOW", 0)),
    )


# =========================================================
# FINDINGS LIST VIEW
# =========================================================

def render_findings_list(
    findings: list[dict],
) -> None:
    """Render findings list with search, filters and sorting."""

    render_page_header("Audit Findings", "🔎")

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
        filtered["finding_id"].tolist()
    )

    filtered = filtered.copy()

    filtered["Severity"] = filtered[
        "severity"
    ].map(
        lambda severity: (
            f"{SEVERITY_COLOR.get(severity, '')} "
            f"{severity}"
        )
    )

    display_cols = {
        "finding_id": "Finding ID",
        "control_id": "Control",
        "customer_id": "Customer",
        "Severity": "Severity",
        "finding_status": "Status",
    }

    table = filtered[
        list(display_cols.keys())
    ].rename(
        columns=display_cols
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

        st.session_state.view = "detail"

        st.rerun()


# =========================================================
# FINDING NAVIGATION
# =========================================================

def render_navigation(
    findings: list[dict],
    finding_id: str,
) -> None:
    """Render Back / Previous / Next navigation."""

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

    nav_col1, nav_col2, nav_col3, nav_col4 = st.columns(
        [2, 1, 1, 2]
    )

    with nav_col1:
        if st.button(
            "← Back to findings list"
        ):
            st.session_state.view = "list"
            st.session_state.selected_finding_id = None
            st.rerun()

    if nav_state is None:
        return

    with nav_col2:
        if st.button(
            "◀ Previous",
            disabled=nav_state["prev_id"] is None,
        ):
            st.session_state.selected_finding_id = (
                nav_state["prev_id"]
            )
            st.rerun()

    with nav_col3:
        if st.button(
            "Next ▶",
            disabled=nav_state["next_id"] is None,
        ):
            st.session_state.selected_finding_id = (
                nav_state["next_id"]
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
    """Render full finding detail."""

    render_navigation(
        findings,
        finding["finding_id"],
    )

    severity = finding.get(
        "severity",
        "",
    )

    st.title(
        f"{SEVERITY_COLOR.get(severity, '')} "
        f"{finding['finding_id']}"
    )

    tab_overview, tab_evidence, tab_review, tab_ai = st.tabs(
        [
            "Overview",
            "Evidence & Policy",
            "Review Info",
            "AI Output",
        ]
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
            ) or "—",
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

        # -----------------------------------------------
        # Other findings for same customer
        # -----------------------------------------------

        customer_id = finding.get(
            "customer_id"
        )

        if customer_id:

            siblings = [
                other
                for other in findings
                if (
                    other.get("customer_id")
                    == customer_id
                    and other["finding_id"]
                    != finding["finding_id"]
                )
            ]

            if siblings:

                st.divider()

                st.subheader(
                    f"Other findings for {customer_id}"
                )

                for sibling in siblings:

                    sibling_severity = sibling.get(
                        "severity",
                        "",
                    )

                    label = (
                        f"{SEVERITY_COLOR.get(sibling_severity, '')} "
                        f"{sibling['finding_id']} — "
                        f"{sibling['control_id']} "
                        f"({sibling.get('finding_status', '—')})"
                    )

                    if st.button(
                        label,
                        key=f"jump_{sibling['finding_id']}",
                    ):
                        st.session_state.selected_finding_id = (
                            sibling["finding_id"]
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
                key=f"evidence_raw_{finding['finding_id']}",
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
                st.session_state
                .get(
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

            confirm_col, reject_col = st.columns(2)

            with confirm_col:

                if st.button(
                    "✅ Confirm finding",
                    disabled=not reviewer_name,
                    width="stretch",
                    key=f"confirm_{finding['finding_id']}",
                ):

                    previous_status = current_status

                    try:

                        confirm_finding(
                            finding,
                            reviewed_by=reviewer_name,
                            reviewer_notes=(
                                reviewer_notes or None
                            ),
                        )

                    except ValueError as exc:

                        st.error(
                            f"Could not confirm: {exc}"
                        )

                    else:

                        try:
                            write_finding_review(
                                finding,
                                previous_status=previous_status,
                            )
                        except PersistenceNotConfigured:
                            # Supabase is optional -- the review
                            # decision still stands in memory even
                            # if it can't be persisted.
                            pass

                        # Immediately generate and persist the
                        # deterministic Stage 2 explanation -- no
                        # LLM, cannot hallucinate, always available
                        # the moment a finding is confirmed. This is
                        # separate from the AI explanation (Stage 3,
                        # generated on-demand below via the
                        # "Generate AI explanation" button).
                        try:
                            deterministic_explanation = explain_finding(
                                finding
                            )

                            try:
                                write_finding_explanation(
                                    deterministic_explanation
                                )
                            except PersistenceNotConfigured:
                                pass

                        except ValueError:
                            # explain_finding() only raises if the
                            # finding isn't CONFIRMED -- unreachable
                            # here since confirm_finding() just set
                            # it, but guarded rather than assumed.
                            pass

                        st.success(
                            "Finding confirmed."
                        )

                        st.rerun()

            with reject_col:

                if st.button(
                    "❌ Reject finding",
                    disabled=not reviewer_name,
                    width="stretch",
                    key=f"reject_{finding['finding_id']}",
                ):

                    previous_status = current_status

                    try:

                        reject_finding(
                            finding,
                            reviewed_by=reviewer_name,
                            reviewer_notes=(
                                reviewer_notes or None
                            ),
                        )

                    except ValueError as exc:

                        st.error(
                            f"Could not reject: {exc}"
                        )

                    else:

                        try:
                            write_finding_review(
                                finding,
                                previous_status=previous_status,
                            )
                        except PersistenceNotConfigured:
                            pass

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

        rev_col1, rev_col2 = st.columns(2)

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

        elif finding.get("finding_status") == "CONFIRMED":

            st.caption(
                "Not yet generated for this finding."
            )

            if st.button(
                "🤖 Generate AI explanation",
                width="stretch",
                key=f"generate_ai_{finding['finding_id']}",
            ):

                registry = load_registry()

                with st.spinner(
                    "Retrieving policy context and calling the LLM..."
                ):

                    result = generate_ai_explanation_for_finding(
                        finding,
                        registry=registry,
                    )

                if result.succeeded:

                    try:
                        write_ai_output(finding)
                    except PersistenceNotConfigured:
                        # Supabase is optional -- the explanation
                        # still shows in the UI even if it can't be
                        # persisted.
                        pass

                    st.success(
                        "AI explanation generated."
                    )

                    st.rerun()

                else:

                    st.error(
                        f"Could not generate an AI explanation: "
                        f"{result.error}"
                    )

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
    Read a field from either a dataclass/object or dict.

    Makes the dashboard a little more robust to future
    changes in EvaluationResult representation.
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
    Format numeric metric values for dashboard display.
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
    Render Person C dashboard.

    Uses EvaluationResult directly for:
        TP
        FP
        FN
        Precision
        Recall
        F1

    Severity distribution is based on the generated findings.
    """

    render_page_header("Audit Dashboard", "📊")

    st.caption(
        "Deterministic audit evaluation and finding summary"
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
        )

        # One column per severity level so each bar can carry its own
        # on-brand color instead of the Streamlit-default near-black bars.
        severity_chart_df = severity_chart_df.T

        st.bar_chart(
            severity_chart_df,
            width="stretch",
            color=[
                SEVERITY_CHART_COLORS["CRITICAL"],
                SEVERITY_CHART_COLORS["HIGH"],
                SEVERITY_CHART_COLORS["MEDIUM"],
                SEVERITY_CHART_COLORS["LOW"],
            ],
        )

        render_color_legend(SEVERITY_LEGEND)

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
                        severity_counts.iloc[index]
                    )
                    for index in range(
                        len(severity_counts)
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

        # One column per status (instead of one shared column) so every
        # bar can be given its own, consistent color regardless of the
        # order value_counts() happens to return.
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
            for status in status_chart_df.columns
        ]

        st.bar_chart(
            status_chart_df,
            width="stretch",
            color=status_bar_colors,
        )

        # Only show legend entries for statuses that are actually
        # present in this audit run, so the legend never mentions a
        # status the user can't currently see on the chart.
        present_statuses = set(status_chart_df.columns)

        status_legend_items = [
            item
            for item in STATUS_LEGEND
            if item[0] in present_statuses
        ]

        other_statuses = present_statuses - {
            item[0] for item in STATUS_LEGEND
        }

        for other_status in sorted(other_statuses):
            status_legend_items.append(
                (
                    other_status,
                    STATUS_CHART_FALLBACK_COLOR,
                    "Other status",
                )
            )

        render_color_legend(status_legend_items)

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
                ) is not None
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

            for control_id, result in per_control.items():

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
    """Run the Streamlit application."""

    # -----------------------------------------------------
    # SESSION STATE
    # -----------------------------------------------------

    if "view" not in st.session_state:
        st.session_state.view = "dashboard"

    if "selected_finding_id" not in st.session_state:
        st.session_state.selected_finding_id = None

    if "filtered_ids" not in st.session_state:
        st.session_state.filtered_ids = None

    # -----------------------------------------------------
    # LOAD PIPELINE RESULT ONCE
    # -----------------------------------------------------

    result = load_pipeline_result()

    findings = result.generated_findings
    evaluation = result.evaluation

    # -----------------------------------------------------
    # SIDEBAR
    # -----------------------------------------------------

    reviewer_name = render_reviewer_identity()

    st.sidebar.divider()

    st.sidebar.header(
        "Navigation"
    )

    selected_view = st.sidebar.radio(
        "Go to",
        [
            "Dashboard",
            "Findings",
        ],
        index=(
            0
            if st.session_state.view == "dashboard"
            else 1
        ),
    )

    if selected_view == "Dashboard":
        st.session_state.view = "dashboard"

    else:
        if (
            st.session_state.view
            == "dashboard"
        ):
            st.session_state.view = "list"

    st.sidebar.divider()

    st.sidebar.caption(
        "The UI uses the real deterministic audit pipeline. "
        "Audit runs, findings, reviews, deterministic "
        "explanations, and AI outputs are persisted to "
        "Supabase automatically when it's configured."
    )

    # -----------------------------------------------------
    # PAGE ROUTING
    # -----------------------------------------------------

    if st.session_state.view == "dashboard":

        render_dashboard(
            findings,
            evaluation,
        )

        return

    # -----------------------------------------------------
    # FINDINGS DETAIL
    # -----------------------------------------------------

    if (
        st.session_state.view == "detail"
        and st.session_state.selected_finding_id
    ):

        finding = get_finding_by_id(
            findings,
            st.session_state.selected_finding_id,
        )

        if finding is None:

            st.session_state.view = "list"
            st.session_state.selected_finding_id = None
            st.rerun()

        else:

            render_finding_detail(
                findings,
                finding,
            )

        return

    # -----------------------------------------------------
    # FINDINGS LIST
    # -----------------------------------------------------

    render_findings_list(
        findings
    )


if __name__ == "__main__":
    main()