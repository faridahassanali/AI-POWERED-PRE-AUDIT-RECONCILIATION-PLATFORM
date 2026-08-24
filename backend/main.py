from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel

from backend.database import supabase

from engine.audit_pipeline import run_audit
from engine.policy_registry import load_policy_registry
from RAG.retriever import retrieve_for_finding

from engine.persistence import (
    write_audit_run,
    write_findings,
    create_finding_review,
    get_finding_reviews,
)

# =========================================================
# POLICY REGISTRY
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

policy_registry = load_policy_registry(DATA_DIR)

app = FastAPI(
    title="AI-Powered Pre-Audit Platform",
    version="1.0.0",
)


# =========================================================
# REQUEST MODELS
# =========================================================

class AuditRunCreate(BaseModel):
    audit_run_id: str
    controls_executed: list[str]
    total_records_evaluated: int = 0
    total_findings_generated: int = 0


class AuditRunUpdate(BaseModel):
    controls_executed: list[str] | None = None
    total_records_evaluated: int | None = None
    total_findings_generated: int | None = None


class FindingCreate(BaseModel):
    finding_id: str
    audit_run_id: str
    control_id: str
    customer_id: str
    severity: str
    assessment_status: str
    finding_status: str
    expected: str
    actual: str
    evidence: dict
    policy_references: list[dict]


class FindingUpdate(BaseModel):
    finding_status: str | None = None
    reviewed_by: str | None = None
    reviewer_notes: str | None = None
    ai_explanation: str | None = None
    ai_recommendation: str | None = None


# =========================================================
# BASIC ENDPOINTS
# =========================================================

@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "Pre-Audit Backend",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
    }


# =========================================================
# DATABASE HEALTH
# =========================================================

@app.get("/database-health")
def database_health():
    try:
        (
            supabase
            .table("audit_runs")
            .select("*")
            .limit(1)
            .execute()
        )

        return {
            "status": "connected",
            "database": "supabase",
            "table": "audit_runs",
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
        }


# =========================================================
# REAL AUDIT EXECUTION
# =========================================================

@app.post("/audit-runs/execute")
def execute_audit():
    """
    Execute the existing deterministic audit pipeline.

    Flow:

        FastAPI
            ↓
        run_audit()
            ↓
        persistence
            ↓
        Supabase
    """

    try:

        # -----------------------------------------------------
        # 1. RUN EXISTING AUDIT ENGINE
        # -----------------------------------------------------

        result = run_audit()

        # -----------------------------------------------------
        # 2. SAVE AUDIT RUN
        # -----------------------------------------------------

        audit_run_result = write_audit_run(
            result.audit_trace
        )

        # -----------------------------------------------------
        # 3. SAVE GENERATED FINDINGS
        # -----------------------------------------------------

        findings_result = write_findings(
            result.generated_findings
        )

        # -----------------------------------------------------
        # 4. RETURN SUMMARY
        # -----------------------------------------------------

        return {
            "status": "success",
            "message": "Audit executed successfully.",
            "audit_run_id": (
                result.audit_trace.audit_run_id
            ),
            "total_records_evaluated": (
                result.audit_trace
                .total_records_evaluated
            ),
            "total_findings_generated": (
                result.audit_trace
                .total_findings_generated
            ),
            "audit_run_saved": (
                len(audit_run_result) > 0
            ),
            "findings_saved": len(
                findings_result
            ),
            "report": result.report,
        }

    except Exception as e:

        return {
            "status": "error",
            "message": str(e),
        }


# =========================================================
# AUDIT RUNS
# =========================================================

@app.get("/audit-runs")
def get_audit_runs():
    try:

        response = (
            supabase
            .table("audit_runs")
            .select("*")
            .order(
                "created_at",
                desc=True,
            )
            .execute()
        )

        return {
            "status": "success",
            "count": len(response.data),
            "audit_runs": response.data,
        }

    except Exception as e:

        return {
            "status": "error",
            "message": str(e),
        }


@app.get("/audit-runs/{audit_run_id}")
def get_audit_run(
    audit_run_id: str,
):
    try:

        response = (
            supabase
            .table("audit_runs")
            .select("*")
            .eq(
                "audit_run_id",
                audit_run_id,
            )
            .limit(1)
            .execute()
        )

        if not response.data:

            return {
                "status": "not_found",
                "message": (
                    f"Audit run '{audit_run_id}' "
                    "was not found."
                ),
            }

        return {
            "status": "success",
            "audit_run": response.data[0],
        }

    except Exception as e:

        return {
            "status": "error",
            "message": str(e),
        }


@app.post("/audit-runs")
def create_audit_run(
    audit_run: AuditRunCreate,
):
    try:

        row = {
            "audit_run_id": audit_run.audit_run_id,
            "started_at": (
                datetime.now(
                    timezone.utc
                ).isoformat()
            ),
            "controls_executed": (
                audit_run.controls_executed
            ),
            "total_records_evaluated": (
                audit_run.total_records_evaluated
            ),
            "total_findings_generated": (
                audit_run.total_findings_generated
            ),
        }

        response = (
            supabase
            .table("audit_runs")
            .insert(row)
            .execute()
        )

        return {
            "status": "success",
            "audit_run": response.data[0],
        }

    except Exception as e:

        return {
            "status": "error",
            "message": str(e),
        }


@app.patch("/audit-runs/{audit_run_id}")
def update_audit_run(
    audit_run_id: str,
    update: AuditRunUpdate,
):
    try:

        update_data = {}

        if update.controls_executed is not None:

            update_data["controls_executed"] = (
                update.controls_executed
            )

        if update.total_records_evaluated is not None:

            update_data["total_records_evaluated"] = (
                update.total_records_evaluated
            )

        if update.total_findings_generated is not None:

            update_data["total_findings_generated"] = (
                update.total_findings_generated
            )

        update_data["completed_at"] = (
            datetime.now(
                timezone.utc
            ).isoformat()
        )

        response = (
            supabase
            .table("audit_runs")
            .update(update_data)
            .eq(
                "audit_run_id",
                audit_run_id,
            )
            .execute()
        )

        if not response.data:

            return {
                "status": "not_found",
                "message": (
                    f"Audit run '{audit_run_id}' "
                    "was not found."
                ),
            }

        return {
            "status": "success",
            "audit_run": response.data[0],
        }

    except Exception as e:

        return {
            "status": "error",
            "message": str(e),
        }


# =========================================================
# FINDINGS
# =========================================================

@app.get("/findings")
def get_findings(
    status: str | None = None,
    severity: str | None = None,
    control_id: str | None = None,
    audit_run_id: str | None = None,
):
    """
    Get findings.

    Optional filters:

        ?status=REVIEW
        ?severity=HIGH
        ?control_id=RISK_001
        ?audit_run_id=AUDIT-xxxx

    Filters can be combined.
    """

    try:

        query = (
            supabase
            .table("findings")
            .select("*")
        )

        # -----------------------------------------------------
        # FILTER BY FINDING STATUS
        # -----------------------------------------------------

        if status is not None:

            query = query.eq(
                "finding_status",
                status,
            )

        # -----------------------------------------------------
        # FILTER BY SEVERITY
        # -----------------------------------------------------

        if severity is not None:

            query = query.eq(
                "severity",
                severity,
            )

        # -----------------------------------------------------
        # FILTER BY CONTROL
        # -----------------------------------------------------

        if control_id is not None:

            query = query.eq(
                "control_id",
                control_id,
            )

        # -----------------------------------------------------
        # FILTER BY AUDIT RUN
        # -----------------------------------------------------

        if audit_run_id is not None:

            query = query.eq(
                "audit_run_id",
                audit_run_id,
            )

        response = (
            query
            .order(
                "created_at",
                desc=True,
            )
            .execute()
        )

        return {
            "status": "success",
            "count": len(response.data),
            "filters": {
                "status": status,
                "severity": severity,
                "control_id": control_id,
                "audit_run_id": audit_run_id,
            },
            "findings": response.data,
        }

    except Exception as e:

        return {
            "status": "error",
            "message": str(e),
        }


# =========================================================
# DASHBOARD SUMMARY
# =========================================================

@app.get("/dashboard/summary")
def get_dashboard_summary():
    """
    Return a simple summary for a future dashboard.
    """

    try:

        response = (
            supabase
            .table("findings")
            .select(
                "finding_status, severity"
            )
            .execute()
        )

        findings = response.data

        summary = {
            "total_findings": len(findings),

            # Finding status
            "review": 0,
            "confirmed": 0,
            "rejected": 0,
            "resolved": 0,

            # Severity
            "critical": 0,
            "high": 0,
            "medium": 0,
            "low": 0,
        }

        for finding in findings:

            status = finding.get(
                "finding_status"
            )

            severity = finding.get(
                "severity"
            )

            # -------------------------------------------------
            # COUNT STATUS
            # -------------------------------------------------

            if status == "REVIEW":

                summary["review"] += 1

            elif status == "CONFIRMED":

                summary["confirmed"] += 1

            elif status == "REJECTED":

                summary["rejected"] += 1

            elif status == "RESOLVED":

                summary["resolved"] += 1

            # -------------------------------------------------
            # COUNT SEVERITY
            # -------------------------------------------------

            if severity == "CRITICAL":

                summary["critical"] += 1

            elif severity == "HIGH":

                summary["high"] += 1

            elif severity == "MEDIUM":

                summary["medium"] += 1

            elif severity == "LOW":

                summary["low"] += 1

        return {
            "status": "success",
            "summary": summary,
        }

    except Exception as e:

        return {
            "status": "error",
            "message": str(e),
        }


# =========================================================
# GET SINGLE FINDING
# =========================================================

# =========================================================
# FINDING POLICY RETRIEVAL
# =========================================================

@app.get("/findings/{finding_id}/policy")
def get_finding_policy(
    finding_id: str,
):
    """
    Retrieve the applicable policy context for one finding.

    Flow:

        Finding
            ↓
        policy_references
            ↓
        Policy Registry
            ↓
        RAG Retriever
            ↓
        Policy Context
    """

    try:

        # -----------------------------------------------------
        # 1. GET FINDING
        # -----------------------------------------------------

        response = (
            supabase
            .table("findings")
            .select("*")
            .eq(
                "finding_id",
                finding_id,
            )
            .limit(1)
            .execute()
        )

        if not response.data:

            return {
                "status": "not_found",
                "message": (
                    f"Finding '{finding_id}' "
                    "was not found."
                ),
            }

        finding = response.data[0]

        # -----------------------------------------------------
        # 2. RETRIEVE POLICY THROUGH RAG
        # -----------------------------------------------------

        policy_context = retrieve_for_finding(
            finding=finding,
            registry=policy_registry,
            top_k=3,
        )

        # -----------------------------------------------------
        # 3. RETURN POLICY CONTEXT
        # -----------------------------------------------------

        return {
            "status": "success",
            "finding_id": finding_id,
            "policy_references": (
                finding.get(
                    "policy_references",
                    [],
                )
            ),
            "count": len(policy_context),
            "policy_context": policy_context,
        }

    except Exception as e:

        return {
            "status": "error",
            "message": str(e),
        }
# =========================================================
# CREATE FINDING
# =========================================================

@app.post("/findings")
def create_finding(
    finding: FindingCreate,
):
    try:

        row = {
            "finding_id": finding.finding_id,
            "audit_run_id": finding.audit_run_id,
            "control_id": finding.control_id,
            "customer_id": finding.customer_id,
            "severity": finding.severity,
            "assessment_status": (
                finding.assessment_status
            ),
            "finding_status": (
                finding.finding_status
            ),
            "expected": finding.expected,
            "actual": finding.actual,
            "evidence": finding.evidence,
            "policy_references": (
                finding.policy_references
            ),
        }

        response = (
            supabase
            .table("findings")
            .insert(row)
            .execute()
        )

        return {
            "status": "success",
            "finding": response.data[0],
        }

    except Exception as e:

        return {
            "status": "error",
            "message": str(e),
        }


# =========================================================
# UPDATE FINDING / HUMAN REVIEW
# =========================================================

@app.patch("/findings/{finding_id}")
def update_finding(
    finding_id: str,
    update: FindingUpdate,
):
    try:

        # -----------------------------------------------------
        # 1. GET CURRENT FINDING
        # -----------------------------------------------------

        current_response = (
            supabase
            .table("findings")
            .select("*")
            .eq(
                "finding_id",
                finding_id,
            )
            .limit(1)
            .execute()
        )

        if not current_response.data:

            return {
                "status": "not_found",
                "message": (
                    f"Finding '{finding_id}' "
                    "was not found."
                ),
            }

        current_finding = (
            current_response.data[0]
        )

        previous_status = (
            current_finding.get(
                "finding_status"
            )
        )

        # -----------------------------------------------------
        # 2. BUILD UPDATE
        # -----------------------------------------------------

        update_data = {}

        if update.finding_status is not None:

            update_data["finding_status"] = (
                update.finding_status
            )

        if update.reviewed_by is not None:

            update_data["reviewed_by"] = (
                update.reviewed_by
            )

        if update.reviewer_notes is not None:

            update_data["reviewer_notes"] = (
                update.reviewer_notes
            )

        if update.ai_explanation is not None:

            update_data["ai_explanation"] = (
                update.ai_explanation
            )

        if update.ai_recommendation is not None:

            update_data["ai_recommendation"] = (
                update.ai_recommendation
            )

        # Only add review timestamp when this
        # is actually a human review/update.
        if (
            update.finding_status is not None
            or update.reviewed_by is not None
            or update.reviewer_notes is not None
        ):

            update_data["review_timestamp"] = (
                datetime.now(
                    timezone.utc
                ).isoformat()
            )

        update_data["updated_at"] = (
            datetime.now(
                timezone.utc
            ).isoformat()
        )

        # -----------------------------------------------------
        # 3. UPDATE FINDING
        # -----------------------------------------------------

        response = (
            supabase
            .table("findings")
            .update(update_data)
            .eq(
                "finding_id",
                finding_id,
            )
            .execute()
        )

        if not response.data:

            return {
                "status": "not_found",
                "message": (
                    f"Finding '{finding_id}' "
                    "was not found."
                ),
            }

        updated_finding = response.data[0]

        new_status = (
            updated_finding.get(
                "finding_status"
            )
        )

        # -----------------------------------------------------
        # 4. CREATE REVIEW HISTORY
        # -----------------------------------------------------

        # Only create history when the finding status
        # actually changes.
        if (
            update.finding_status is not None
            and previous_status != new_status
        ):

            try:

                create_finding_review(
                    finding_id=finding_id,
                    audit_run_id=(
                        updated_finding.get(
                            "audit_run_id"
                        )
                    ),
                    previous_status=previous_status,
                    new_status=new_status,
                    reviewed_by=(
                        update.reviewed_by
                    ),
                    reviewer_notes=(
                        update.reviewer_notes
                    ),
                )

            except Exception as review_error:

                return {
                    "status": "success",
                    "warning": (
                        "Finding was updated, "
                        "but review history could "
                        "not be recorded."
                    ),
                    "review_error": str(
                        review_error
                    ),
                    "finding": updated_finding,
                }

        return {
            "status": "success",
            "finding": updated_finding,
        }

    except Exception as e:

        return {
            "status": "error",
            "message": str(e),
        }


# =========================================================
# FINDING REVIEW HISTORY
# =========================================================

@app.get("/findings/{finding_id}/reviews")
def get_finding_review_history(
    finding_id: str,
):
    try:

        # First verify that the finding exists.
        finding_response = (
            supabase
            .table("findings")
            .select("finding_id")
            .eq(
                "finding_id",
                finding_id,
            )
            .limit(1)
            .execute()
        )

        if not finding_response.data:

            return {
                "status": "not_found",
                "message": (
                    f"Finding '{finding_id}' "
                    "was not found."
                ),
            }

        reviews = get_finding_reviews(
            finding_id
        )

        return {
            "status": "success",
            "count": len(reviews),
            "reviews": reviews,
        }

    except Exception as e:

        return {
            "status": "error",
            "message": str(e),
        }