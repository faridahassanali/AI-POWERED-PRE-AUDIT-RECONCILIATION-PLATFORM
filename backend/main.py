from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, Header
from pydantic import BaseModel

from fastapi.middleware.cors import CORSMiddleware
from backend.auth import verify_api_key, require_api_key_configured
from fastapi import Depends
import os

from backend.database import supabase, supabase_anon

from engine.audit_pipeline import run_audit
from engine.policy_registry import load_policy_registry
from RAG.retriever import retrieve_for_finding

from engine.ai_explanation_pipeline import generate_ai_explanation_for_finding
from engine.finding_explainer import explain_finding

from engine.persistence import (
    write_audit_run,
    write_findings,
    create_finding_review,
    get_finding_reviews,
    write_ai_output,
    write_finding_explanation,
    write_audit_evaluation,
    claim_audit_idempotency,
)

# =========================================================
# POLICY REGISTRY
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

policy_registry = load_policy_registry(DATA_DIR)


def _read_client():
    """
    Client used by read-only (GET) endpoints.

    Prefers the anon key, which is subject to the RLS read policies
    in supabase/migrations/20260826140000_rls_read_policies.sql --
    so these routes can never see more than those policies allow,
    even if a future endpoint is added carelessly.

    Falls back to the service-role client if SUPABASE_ANON_KEY isn't
    configured yet, so this doesn't break a deployment that hasn't
    added the key. Once SUPABASE_ANON_KEY is set, reads are RLS-
    restricted automatically -- no other code change needed.
    """
    return supabase_anon or supabase


# =========================================================
# AI INPUT SAFELIST
# =========================================================
# Only these keys are safe to hand to the AI pipeline. Rows read
# straight from Supabase also carry created_at/updated_at (and
# possibly other DB-only columns) which the AI input schema rejects
# with "Additional properties are not allowed" -- this whitelist
# mirrors the shape engine.persistence._finding_row() already
# writes, so it's the canonical "safe" shape for a finding
# everywhere else in the app.

_FINDING_AI_FIELDS = {
    "finding_id",
    "audit_run_id",
    "control_id",
    "customer_id",
    "severity",
    "assessment_status",
    "finding_status",
    "expected",
    "actual",
    "evidence",
    "policy_references",
    "reviewed_by",
    "review_timestamp",
    "reviewer_notes",
    "ai_explanation",
    "ai_recommendation",
}


def _clean_finding_for_ai(finding: dict) -> dict:
    return {
        key: value
        for key, value in finding.items()
        if key in _FINDING_AI_FIELDS
    }


app = FastAPI(
    title="AI-Powered Pre-Audit Platform",
    version="1.0.0",
)
require_api_key_configured()  
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("APP_CORS_ORIGINS", "").split(",") if os.environ.get("APP_CORS_ORIGINS") else [],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PATCH"],
    allow_headers=["X-API-Key", "Content-Type"],
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
    
class AuditExecuteRequest(BaseModel):
    audit_run_id: str
    idempotency_key: str


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
def execute_audit(
    idempotency_key: str = Header(..., alias="Idempotency-Key"),
    _: str = Depends(verify_api_key),
):
    """
    Execute the deterministic audit pipeline exactly once
    for each Idempotency-Key.

    Idempotency behavior:
    - First request:
        * atomically claims the Idempotency-Key
        * executes the audit
        * persists the audit result
        * returns success

    - Duplicate request:
        * does NOT execute the audit again
        * returns the existing audit_run
        * returns the original audit_run_id
    """

    try:
        # -----------------------------------------------------
        # 1. GENERATE AUDIT RUN ID
        # -----------------------------------------------------
        import uuid

        audit_run_id = f"AUDIT-{uuid.uuid4().hex}"

        # -----------------------------------------------------
        # 2. ATOMICALLY CLAIM IDEMPOTENCY KEY
        # -----------------------------------------------------
        claim_result = claim_audit_idempotency(
            audit_run_id=audit_run_id,
            idempotency_key=idempotency_key,
        )

        # -----------------------------------------------------
        # 3. DUPLICATE REQUEST
        # -----------------------------------------------------
        if not claim_result["claimed"]:

            existing_audit_run_id = claim_result.get(
                "audit_run_id"
            )

            # The atomic RPC should always return the
            # existing audit_run_id for a duplicate.
            if not existing_audit_run_id:
                return {
                    "status": "error",
                    "message": (
                        "Idempotency key was already claimed, "
                        "but no existing audit_run_id was returned."
                    ),
                }

            # -------------------------------------------------
            # Fetch the exact audit run returned by the RPC
            # -------------------------------------------------
            existing_response = (
                supabase
                .table("audit_runs")
                .select("*")
                .eq(
                    "audit_run_id",
                    existing_audit_run_id,
                )
                .limit(1)
                .execute()
            )

            if not existing_response.data:
                return {
                    "status": "error",
                    "message": (
                        "Idempotency key was already claimed, "
                        "but the existing audit run could not "
                        "be found."
                    ),
                }

            # -------------------------------------------------
            # RETURN DUPLICATE
            # -------------------------------------------------
            return {
                "status": "duplicate",
                "message": (
                    "Audit request with this Idempotency-Key "
                    "has already been claimed."
                ),
                "audit_run": existing_response.data[0],
            }

        # -----------------------------------------------------
        # 4. FIRST REQUEST → EXECUTE AUDIT
        # -----------------------------------------------------
        result = run_audit(
            audit_run_id=audit_run_id,
        )

        # -----------------------------------------------------
        # 4b. CHECK PIPELINE FAILURE
        # -----------------------------------------------------
        # FIX: run_audit() always returns a result, even when the
        # pipeline itself failed (audit_trace.status == "FAILED"),
        # instead of raising. Without this check, a failed run would
        # be persisted and reported to the caller as "success" --
        # the exact same class of bug fixed earlier in
        # engine.audit_orchestration.run_audit_and_persist(), which
        # this endpoint does not go through (it calls run_audit()
        # directly), so it needed the same guard independently.
        if result.audit_trace.status == "FAILED":
            return {
                "status": "error",
                "message": "Audit pipeline failed.",
                "audit_run_id": result.audit_trace.audit_run_id,
                "error_type": result.audit_trace.error_type,
                "error_message": result.audit_trace.error_message,
                "pre_audit_report": result.pre_audit_report,
            }

        # -----------------------------------------------------
        # 5. SAVE AUDIT RUN
        # -----------------------------------------------------
        audit_run_result = write_audit_run(
            result.audit_trace
        )

        # -----------------------------------------------------
        # 6. SAVE FINDINGS
        # -----------------------------------------------------
        findings_result = write_findings(
            result.generated_findings
        )

        # -----------------------------------------------------
        # 7. SAVE EVALUATION
        # -----------------------------------------------------
        try:
            write_audit_evaluation(
                result.evaluation,
                audit_run_id=result.audit_trace.audit_run_id,
            )
        except Exception as eval_error:
            evaluation_save_error = str(eval_error)
        else:
            evaluation_save_error = None

        # -----------------------------------------------------
        # 8. RETURN SUCCESS RESULT
        # -----------------------------------------------------
        return {
            "status": "success",
            "message": "Audit executed successfully.",
            "audit_run_id": (
                result.audit_trace.audit_run_id
            ),
            "total_records_evaluated": (
                result.audit_trace.total_records_evaluated
            ),
            "total_findings_generated": (
                result.audit_trace.total_findings_generated
            ),
            "audit_run_saved": (
                len(audit_run_result) > 0
            ),
            "findings_saved": (
                len(findings_result)
            ),
            "evaluation_saved": (
                evaluation_save_error is None
            ),
            "evaluation_save_error": (
                evaluation_save_error
            ),
            "report": result.report,
            "evaluation": {
                "true_positives": (
                    result.evaluation.true_positives
                ),
                "false_positives": (
                    result.evaluation.false_positives
                ),
                "false_negatives": (
                    result.evaluation.false_negatives
                ),
                "precision": (
                    result.evaluation.precision
                ),
                "recall": (
                    result.evaluation.recall
                ),
                "f1_score": (
                    result.evaluation.f1_score
                ),
            },
            "pre_audit_report": result.pre_audit_report,
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
            _read_client()
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
            _read_client()
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


# =========================================================
# AUDIT RUN EVALUATION
# =========================================================

@app.get("/audit-runs/{audit_run_id}/evaluation")
def get_audit_evaluation(
    audit_run_id: str,
):
    """
    Return the ground-truth evaluation metrics (TP/FP/FN,
    Precision/Recall/F1) for one audit run, if they've been
    persisted to public.audit_evaluations.
    """

    try:

        response = (
            _read_client()
            .table("audit_evaluations")
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
                    f"No evaluation found for audit run "
                    f"'{audit_run_id}'."
                ),
            }

        return {
            "status": "success",
            "evaluation": response.data[0],
        }

    except Exception as e:

        return {
            "status": "error",
            "message": str(e),
        }


@app.post("/audit-runs")
def create_audit_run(
    audit_run: AuditRunCreate,
    _: str = Depends(verify_api_key),
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
    _: str = Depends(verify_api_key),
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
            _read_client()
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
            _read_client()
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
            _read_client()
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
# AI EXPLANATION (STAGE 3)
# =========================================================

@app.post("/findings/{finding_id}/ai-explanation")
def generate_ai_explanation(
    finding_id: str,
    _: str = Depends(verify_api_key),
):
    """
    Generate (and persist) the LLM explanation/recommendation for
    one CONFIRMED finding.

    Flow:

        Finding (Supabase)
            ↓ cleaned to the AI-safe field whitelist
        generate_ai_explanation_for_finding()
            ↓ RAG retrieval -> LLM -> output validation
        write_findings() + write_ai_output()

    This replaces the frontend calling
    engine.ai_explanation_pipeline directly from inside the
    Streamlit process -- the frontend now only needs APP_API_KEYS,
    never SUPABASE_SERVICE_ROLE_KEY / GROQ_API_KEY / GEMINI_API_KEY.

    The whitelist in _clean_finding_for_ai() is what fixes the
    "Additional properties are not allowed ('created_at',
    'updated_at')" error -- rows read straight from Supabase carry
    those extra columns, which the AI input schema rejects.
    """

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

        finding = _clean_finding_for_ai(
            response.data[0]
        )

        result = generate_ai_explanation_for_finding(
            finding,
            registry=policy_registry,
        )

        if not result.succeeded:

            return {
                "status": "error",
                "message": result.error,
            }

        try:
            write_findings([finding])
            write_ai_output(finding)

        except Exception as persist_error:

            return {
                "status": "success",
                "warning": (
                    "AI explanation was generated, but could not "
                    "be persisted."
                ),
                "persist_error": str(persist_error),
                "finding": {
                    "finding_id": finding_id,
                    "ai_explanation": finding.get("ai_explanation"),
                    "ai_recommendation": finding.get("ai_recommendation"),
                },
            }

        return {
            "status": "success",
            "finding": {
                "finding_id": finding_id,
                "ai_explanation": finding.get("ai_explanation"),
                "ai_recommendation": finding.get("ai_recommendation"),
            },
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
    _: str = Depends(verify_api_key),
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
    _: str = Depends(verify_api_key),
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

        # -----------------------------------------------------
        # 4b. AUTO-GENERATE DETERMINISTIC EXPLANATION (STAGE 2)
        # -----------------------------------------------------
        # Runs automatically the moment a finding transitions to
        # CONFIRMED -- no LLM, cannot fail on network/API issues,
        # so it's safe to run inline here instead of leaving it to
        # the frontend to trigger as a separate call. This mirrors
        # what frontend/app.py used to do locally right after a
        # successful Confirm.

        if (
            update.finding_status == "CONFIRMED"
            and previous_status != "CONFIRMED"
        ):

            try:
                deterministic_explanation = explain_finding(
                    updated_finding
                )
                write_finding_explanation(
                    deterministic_explanation
                )

            except ValueError:
                # explain_finding() only raises if the finding
                # isn't CONFIRMED -- unreachable here since we just
                # set it above, guarded rather than assumed.
                pass

            except Exception as explain_error:

                return {
                    "status": "success",
                    "warning": (
                        "Finding was confirmed, but the "
                        "deterministic explanation could not be "
                        "generated/persisted."
                    ),
                    "explain_error": str(explain_error),
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