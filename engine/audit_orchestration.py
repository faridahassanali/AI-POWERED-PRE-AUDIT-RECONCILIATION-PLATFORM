"""
Audit Orchestration.

Wires the deterministic audit pipeline (engine.audit_pipeline.run_audit)
together with the Supabase persistence layer (engine.persistence).

Why this is a SEPARATE module
------------------------------
engine.persistence.py is explicit that it must never be imported by
engine.audit_pipeline itself, and test_pre_ai_layer.py enforces (via
test_ai_layer_must_not_be_required_for_deterministic_audit) that
run_audit() keeps working with ZERO Supabase dependency -- no
SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY required.

This module sits on top of both instead of merging them: it calls
run_audit() first (unchanged, still Supabase-free), and only then
attempts to persist the result -- its versioned policies, audit run,
findings, and ground-truth evaluation metrics (TP/FP/FN,
precision/recall/F1).
If Supabase isn't configured, the persistence step is skipped rather
than raised -- the pipeline result is still returned so callers who
don't care about Supabase never see a different failure mode than
they did before this module existed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from engine.audit_pipeline import AuditPipelineResult, run_audit
from engine.persistence import (
    PersistenceNotConfigured,
    get_supabase_client,
    write_audit_evaluation,
    write_audit_run,
    write_findings,
)
from engine.policy_persistence import sync_policy_registry
from engine.policy_registry import load_policy_registry
from engine.data_loader import DATA_DIR


@dataclass
class PersistenceOutcome:
    """
    Describes what happened when run_audit_and_persist() tried to
    persist the pipeline result.
    """

    attempted: bool
    persisted: bool
    reason: str | None = None


@dataclass
class OrchestratedAuditResult:
    """
    The deterministic pipeline result, plus what happened when we tried
    to persist it.
    """

    pipeline_result: AuditPipelineResult
    persistence: PersistenceOutcome


def run_audit_and_persist(
    data_dir: Path | str | None = None,
    client: Any | None = None,
) -> OrchestratedAuditResult:
    """
    Run Stage 1 of the audit pipeline, then persist the audit run,
    the generated findings, and the ground-truth evaluation metrics
    to Supabase.

    Persistence is best-effort:

    - If Supabase isn't configured (PersistenceNotConfigured -- missing
      env vars or the 'supabase' package), persistence is skipped and
      `persistence.persisted` is False. The pipeline result itself is
      still returned untouched.
    - Any other exception raised while persisting is NOT swallowed --
      only the "not configured" case is treated as expected/optional.

    A `client` can be passed in explicitly (e.g. a fake client in
    tests, or an already-constructed Supabase client); otherwise each
    write function builds its own via engine.persistence.get_supabase_client().
    """

    pipeline_result = run_audit(data_dir=data_dir)

    try:
        client = client or get_supabase_client()
        sync_policy_registry(load_policy_registry(DATA_DIR), client=client)
        write_audit_run(pipeline_result.audit_trace, client=client)
        write_findings(pipeline_result.generated_findings, client=client)
        write_audit_evaluation(
            pipeline_result.evaluation,
            audit_run_id=pipeline_result.audit_trace.audit_run_id,
            client=client,
        )
    except PersistenceNotConfigured as exc:
        return OrchestratedAuditResult(
            pipeline_result=pipeline_result,
            persistence=PersistenceOutcome(
                attempted=True,
                persisted=False,
                reason=str(exc),
            ),
        )

    return OrchestratedAuditResult(
        pipeline_result=pipeline_result,
        persistence=PersistenceOutcome(
            attempted=True,
            persisted=True,
        ),
    )
def run_audit_and_persist(
    data_dir: Path | str | None = None,
    client: Any | None = None,
) -> OrchestratedAuditResult:
    """
    Run Stage 1 of the audit pipeline, then persist the audit run,
    the generated findings, and the ground-truth evaluation metrics
    to Supabase.

    Persistence is best-effort:

    - If the pipeline itself failed (pipeline_result.audit_trace.status
      == "FAILED"), persistence is skipped entirely and
      `persistence.persisted` is False with a reason explaining why --
      a failed run must never be written to Supabase as if it
      succeeded.
    - If Supabase isn't configured (PersistenceNotConfigured -- missing
      env vars or the 'supabase' package), persistence is skipped and
      `persistence.persisted` is False. The pipeline result itself is
      still returned untouched.
    - Any other exception raised while persisting is NOT swallowed --
      only the "not configured" case is treated as expected/optional.

    A `client` can be passed in explicitly (e.g. a fake client in
    tests, or an already-constructed Supabase client); otherwise each
    write function builds its own via engine.persistence.get_supabase_client().
    """

    pipeline_result = run_audit(data_dir=data_dir)

    # FIX: run_audit() now always returns a result, even when the
    # pipeline itself failed (audit_trace.status == "FAILED"), instead
    # of raising. Without this check, a failed run would be persisted
    # to Supabase looking exactly like a successful one.
    if pipeline_result.audit_trace.status == "FAILED":
        return OrchestratedAuditResult(
            pipeline_result=pipeline_result,
            persistence=PersistenceOutcome(
                attempted=False,
                persisted=False,
                reason=(
                    "Pipeline run failed "
                    f"({pipeline_result.audit_trace.error_type}: "
                    f"{pipeline_result.audit_trace.error_message}); "
                    "skipping persistence."
                ),
            ),
        )

    try:
        client = client or get_supabase_client()
        sync_policy_registry(load_policy_registry(DATA_DIR), client=client)
        write_audit_run(pipeline_result.audit_trace, client=client)
        write_findings(pipeline_result.generated_findings, client=client)
        write_audit_evaluation(
            pipeline_result.evaluation,
            audit_run_id=pipeline_result.audit_trace.audit_run_id,
            client=client,
        )
    except PersistenceNotConfigured as exc:
        return OrchestratedAuditResult(
            pipeline_result=pipeline_result,
            persistence=PersistenceOutcome(
                attempted=True,
                persisted=False,
                reason=str(exc),
            ),
        )

    return OrchestratedAuditResult(
        pipeline_result=pipeline_result,
        persistence=PersistenceOutcome(
            attempted=True,
            persisted=True,
        ),
    )
