from datetime import datetime, timezone

from fastapi import FastAPI
from pydantic import BaseModel

from backend.database import supabase

from engine.audit_pipeline import run_audit
from engine.persistence import (
    write_audit_run,
    write_findings,
)


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

    The pipeline itself remains independent from FastAPI
    and Supabase.

    This endpoint acts as the bridge:

        API
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
def get_findings():
    try:
        response = (
            supabase
            .table("findings")
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
            "findings": response.data,
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
        }


@app.get("/findings/{finding_id}")
def get_finding(
    finding_id: str,
):
    try:
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

        return {
            "status": "success",
            "finding": response.data[0],
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
        }


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


@app.patch("/findings/{finding_id}")
def update_finding(
    finding_id: str,
    update: FindingUpdate,
):
    try:
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

        return {
            "status": "success",
            "finding": response.data[0],
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
        }