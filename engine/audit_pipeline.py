"""
End-to-End Audit Pipeline.

Stage 1 - Deterministic Pre-AI Audit
-------------------------------------

Data Loader
    ↓
Normalization
    ↓
Unified Customer Record
    ↓
Deterministic Controls
    ↓
Finding Validation
    ↓
Finding Integrity
    ↓
Human Review Gate
    ↓
Ground Truth Evaluation
    ↓
Evaluation Report
    ↓
Audit Output


Stage 2 - Post-Review Explanation
----------------------------------

Only CONFIRMED findings may be explained.

REVIEW    → blocked
REJECTED  → blocked
CONFIRMED → explanation allowed

The AI/explanation layer must never influence
the deterministic compliance decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd

from engine.data_loader import (
    DATA_DIR,
    load_data,
    build_unified_customer_record,
)

from engine.normalization import (
    normalize_dataframe,
)

from engine.finding_validator import (
    load_finding_schema,
    validate_finding_or_raise,
)

from engine.finding_integrity import (
    validate_unique_findings_or_raise,
)

from engine.ground_truth_evaluator import (
    EvaluationResult,
    evaluate_findings,
)

from engine.evaluation_report import (
    generate_evaluation_report,
)

from engine.controls import (
    run_all_controls,
)

from engine.finding_explainer import (
    explain_finding,
)

from engine.ai_input import (
    build_ai_input,
)

from engine.policy_registry import (
    PolicyRegistry,
    load_policy_registry,
)

from RAG.retriever import (
    retrieve_for_finding,
)

from engine.audit_trace import (
    create_audit_trace,
    complete_audit_trace,
    fail_audit_trace,
)

from engine.audit_output import (
    AuditOutput,
    build_audit_output,
)


@dataclass
class AuditPipelineResult:
    """
    Result of the deterministic pre-AI audit stage.
    """

    generated_findings: list[dict[str, Any]]

    expected_findings: list[dict[str, Any]]

    evaluation: EvaluationResult

    report: str

    # Empty during the pre-AI stage because findings
    # are still awaiting human review.
    explanations: list[dict[str, Any]]

    audit_trace: Any

    audit_output: AuditOutput


def _normalize_tables(
    tables: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """
    Normalize every loaded source table.

    Returns a new dictionary and does not modify
    the original loaded DataFrames.
    """

    normalized_tables: dict[str, pd.DataFrame] = {}

    for name, dataframe in tables.items():

        normalized_tables[name] = normalize_dataframe(
            dataframe
        )

    return normalized_tables


def _validate_generated_findings(
    findings: list[dict[str, Any]],
) -> None:
    """
    Validate every generated finding against
    the finding schema.

    FIX (perf): the schema is now loaded from disk and compiled into
    a validator ONCE per audit run, not once per finding. Previously
    each call to validate_finding_or_raise(finding) with no schema
    argument re-read finding_schema.json and rebuilt the
    Draft7Validator from scratch for every single finding -- harmless
    for a handful of findings, but unnecessary disk I/O that scales
    linearly with the size of the run.
    """

    schema = load_finding_schema()

    for finding in findings:

        validate_finding_or_raise(
            finding,
            schema=schema,
        )


def _validate_findings_share_audit_run_id(
    findings: list[dict[str, Any]],
    audit_run_id: str,
) -> None:
    """
    Defensive integrity check (FIX, replaces a previous silent
    overwrite bug).

    Previously, individual controls let build_finding() default
    audit_run_id to a fresh random value per finding, so findings
    from the same run had no shared identifier. That was masked here
    by unconditionally overwriting finding["audit_run_id"] on every
    finding after generation -- which hid the bug for this pipeline
    path but did nothing for any other caller of run_all_controls()
    or the individual control functions (e.g. tests, future direct
    callers).

    The root cause is now fixed in engine.controls / finding_builder:
    audit_run_id is a required argument threaded through every
    control call, and this pipeline passes its own audit_run_id
    (created in step 1) straight into run_all_controls().

    This function no longer overwrites anything. It only asserts the
    contract held -- if any finding comes back with a different or
    missing audit_run_id, that means a control forgot to thread the
    id through, and we want a loud failure here, not another silent
    patch.
    """

    for finding in findings:

        found_id = finding.get("audit_run_id")

        if found_id != audit_run_id:

            raise ValueError(
                "Finding integrity violation: expected all "
                f"generated findings to carry audit_run_id "
                f"'{audit_run_id}', but got '{found_id}' for "
                f"finding {finding.get('finding_id')!r} "
                f"(control_id={finding.get('control_id')!r}). "
                "This means a control did not thread audit_run_id "
                "through to build_finding() -- fix the control, "
                "do not patch the finding here."
            )


def _create_audit_trace(
    audit_run_id: str,
    unified: pd.DataFrame,
):
    """
    Create the initial audit trace.
    """

    controls_executed = [
        "SCREENING_001",
        "RISK_001",
        "ARABIC_NAME_001",
        "DORMANT_001",
        "RECON_001",
    ]

    return create_audit_trace(
        audit_run_id=audit_run_id,
        controls_executed=controls_executed,
        total_records_evaluated=len(unified),
    )


def run_audit(
    data_dir: Path | str | None = None,
) -> AuditPipelineResult:
    """
    Run Stage 1 of the audit pipeline.

    This stage:

    1. Loads data.
    2. Normalizes data.
    3. Builds unified customer records.
    4. Runs deterministic controls.
    5. Validates findings.
    6. Validates finding uniqueness.
    7. Creates the audit trace.
    8. Evaluates against ground truth.
    9. Generates the evaluation report.
    10. Builds the canonical audit output.

    IMPORTANT
    ---------

    Findings remain in REVIEW status.

    No explanations are generated here.

    Human review must happen before the
    explanation/AI stage.
    """

    # =========================================================
    # 1. CREATE AUDIT RUN ID
    # =========================================================

    audit_run_id = f"AUDIT-{uuid4().hex}"

    # =========================================================
    # 2. LOAD DATA
    # =========================================================

    if data_dir is None:

        tables = load_data()

    else:

        tables = load_data(
            Path(data_dir)
        )

    # =========================================================
    # 3. NORMALIZATION
    # =========================================================

    normalized_tables = _normalize_tables(
        tables
    )

    # =========================================================
    # 4. BUILD UNIFIED CUSTOMER RECORD
    # =========================================================

    unified = build_unified_customer_record(
        normalized_tables
    )

    # =========================================================
    # 5. CREATE AUDIT TRACE
    # =========================================================

    audit_trace = _create_audit_trace(
        audit_run_id=audit_run_id,
        unified=unified,
    )

    # =========================================================
    # 6. RUN DETERMINISTIC CONTROLS
    # =========================================================

    # FIX: pass the run's own audit_run_id straight into the
    # controls layer, instead of letting each control/build_finding
    # call invent its own id and patching it after the fact (see
    # _validate_findings_share_audit_run_id below).
    generated_findings = run_all_controls(
        unified=unified,
        tables=normalized_tables,
        audit_run_id=audit_run_id,
    )

    # =========================================================
    # 7. VERIFY AUDIT RUN ID INTEGRITY
    # =========================================================

    # FIX: this step used to unconditionally overwrite
    # finding["audit_run_id"] for every finding here. That silently
    # masked controls that weren't given the run's audit_run_id in
    # the first place. Now that engine.controls requires and threads
    # audit_run_id explicitly, this step only verifies the contract
    # held -- it raises loudly instead of patching quietly.
    _validate_findings_share_audit_run_id(
        generated_findings,
        audit_run_id,
    )

    # =========================================================
    # 8. FINDING VALIDATION
    # =========================================================

    _validate_generated_findings(
        generated_findings
    )

    # =========================================================
    # 9. FINDING INTEGRITY
    # =========================================================

    # FIX (bug): this used to call validate_unique_findings(), which
    # returns a bool that was never checked -- duplicate findings
    # could pass straight through the pipeline with no error and no
    # warning. validate_unique_findings_or_raise() raises
    # FindingIntegrityError instead, so it cannot be silently ignored
    # the way a discarded return value could.
    validate_unique_findings_or_raise(
        generated_findings
    )

    # =========================================================
    # 10. COMPLETE AUDIT TRACE
    # =========================================================

    audit_trace = complete_audit_trace(
        trace=audit_trace,
        total_findings_generated=len(
            generated_findings
        ),
    )

    # =========================================================
    # 11. LOAD EXPECTED FINDINGS / GROUND TRUTH
    # =========================================================

    expected_findings = normalized_tables[
        "expected_findings"
    ].to_dict(
        orient="records"
    )

    # =========================================================
    # 12. GROUND TRUTH EVALUATION
    # =========================================================

    evaluation = evaluate_findings(
        generated_findings=generated_findings,
        expected_findings=expected_findings,
    )

    # =========================================================
    # 13. EVALUATION REPORT
    # =========================================================

    report = generate_evaluation_report(
        evaluation
    )

    # =========================================================
    # 14. PRE-AI EXPLANATIONS
    # =========================================================

    # All generated findings are still in REVIEW status.
    # Therefore, NO explanations are generated here.

    explanations: list[dict[str, Any]] = []

    # =========================================================
    # 15. BUILD AUDIT OUTPUT
    # =========================================================

    audit_output = build_audit_output(
        audit_trace=audit_trace,
        findings=generated_findings,
        explanations=explanations,
        evaluation=evaluation,
        report=report,
    )

    # =========================================================
    # 16. RETURN PRE-AI RESULT
    # =========================================================

    return AuditPipelineResult(
        generated_findings=generated_findings,
        expected_findings=expected_findings,
        evaluation=evaluation,
        report=report,
        explanations=explanations,
        audit_trace=audit_trace,
        audit_output=audit_output,
    )


def explain_confirmed_findings(
    findings: list[dict[str, Any]],
    registry: PolicyRegistry | None = None,
    data_dir: Path | str | None = None,
) -> list[dict[str, Any]]:
    """
    Stage 2 of the audit pipeline.

    Generate explanations ONLY for findings that have
    passed human review and are CONFIRMED.

    For each finding, this:

    1. Resolves policy_context via RAG.retriever.retrieve_for_finding(),
       which grounds strictly to the finding's own policy_references
       (see RAG/retriever.py) -- it never substitutes a different,
       merely-similar policy.
    2. Builds the AI Input Contract via build_ai_input(), which
       enforces the confirmed-only gate AND requires a non-empty,
       resolved policy_context (see engine/ai_input.py).
    3. Runs the deterministic explainer.

    A registry can be passed in (e.g. one already loaded once by the
    caller); otherwise one is loaded fresh from data_dir (defaults to
    the project's data/ directory, same convention as run_audit()).

    If a finding's policy reference doesn't resolve in the registry,
    or resolves to nothing, that finding is BLOCKED (raises) rather
    than silently explained with no policy grounding -- this must
    never be caught and skipped silently, since a finding reaching
    this stage is expected to already have a valid, registry-backed
    policy reference from Stage 1.
    """

    if registry is None:
        registry = load_policy_registry(
            Path(data_dir) if data_dir is not None else DATA_DIR
        )

    explanations: list[dict[str, Any]] = []

    for finding in findings:

        policy_context = retrieve_for_finding(
            finding=finding,
            registry=registry,
        )

        ai_input = build_ai_input(
            finding,
            policy_context=policy_context,
        )

        explanation = explain_finding(
            ai_input
        )

        explanations.append(
            explanation
        )

    return explanations
def run_audit(
    data_dir: Path | str | None = None,
) -> AuditPipelineResult:
    """
    ...
    This function always returns an AuditPipelineResult, even on
    failure -- including failures during data loading itself.
    Callers must check `result.audit_trace.status`.
    """

    # =========================================================
    # 1. CREATE AUDIT RUN ID + INITIAL TRACE
    # =========================================================
    # The trace is created FIRST, before anything that can fail
    # (including data loading), so that a failure at any stage --
    # not just inside the deterministic controls -- is still
    # recorded on a trace instead of raising with nothing captured.
    # total_records_evaluated is corrected below once `unified`
    # exists; if we never get that far, it stays 0, which is
    # accurate for a run that never loaded any records.

    audit_run_id = f"AUDIT-{uuid4().hex}"

    audit_trace = create_audit_trace(
        audit_run_id=audit_run_id,
        controls_executed=[
            "SCREENING_001",
            "RISK_001",
            "ARABIC_NAME_001",
            "DORMANT_001",
            "RECON_001",
        ],
        total_records_evaluated=0,
    )

    try:
        # =====================================================
        # 2. LOAD DATA
        # =====================================================
        if data_dir is None:
            tables = load_data()
        else:
            tables = load_data(Path(data_dir))

        # =====================================================
        # 3. NORMALIZATION
        # =====================================================
        normalized_tables = _normalize_tables(tables)

        # =====================================================
        # 4. BUILD UNIFIED CUSTOMER RECORD
        # =====================================================
        unified = build_unified_customer_record(normalized_tables)

        audit_trace.total_records_evaluated = len(unified)

        # =====================================================
        # 5-8. DETERMINISTIC CONTROLS + VALIDATION
        # =====================================================
        generated_findings = run_all_controls(
            unified=unified,
            tables=normalized_tables,
            audit_run_id=audit_run_id,
        )

        _validate_findings_share_audit_run_id(
            generated_findings,
            audit_run_id,
        )

        _validate_generated_findings(generated_findings)

        validate_unique_findings_or_raise(generated_findings)

    except Exception as exc:
        fail_audit_trace(trace=audit_trace, error=exc)

        empty_evaluation = evaluate_findings(
            generated_findings=[],
            expected_findings=[],
        )

        failure_report = (
            "AUDIT RUN FAILED\n"
            f"audit_run_id: {audit_run_id}\n"
            f"error_type: {audit_trace.error_type}\n"
            f"error_message: {audit_trace.error_message}\n"
        )

        audit_output = build_audit_output(
            audit_trace=audit_trace,
            findings=[],
            explanations=[],
            evaluation=empty_evaluation,
            report=failure_report,
        )

        return AuditPipelineResult(
            generated_findings=[],
            expected_findings=[],
            evaluation=empty_evaluation,
            report=failure_report,
            explanations=[],
            audit_trace=audit_trace,
            audit_output=audit_output,
        )

    # =========================================================
    # 9. COMPLETE AUDIT TRACE
    # =========================================================

    audit_trace = complete_audit_trace(
        trace=audit_trace,
        total_findings_generated=len(generated_findings),
    )

    # =========================================================
    # 10. LOAD EXPECTED FINDINGS / GROUND TRUTH
    # =========================================================

    expected_findings = normalized_tables["expected_findings"].to_dict(
        orient="records"
    )

    # =========================================================
    # 11. GROUND TRUTH EVALUATION
    # =========================================================

    evaluation = evaluate_findings(
        generated_findings=generated_findings,
        expected_findings=expected_findings,
    )

    # =========================================================
    # 12. EVALUATION REPORT
    # =========================================================

    report = generate_evaluation_report(evaluation)

    # =========================================================
    # 13. PRE-AI EXPLANATIONS
    # =========================================================

    explanations: list[dict[str, Any]] = []

    # =========================================================
    # 14. BUILD AUDIT OUTPUT
    # =========================================================

    audit_output = build_audit_output(
        audit_trace=audit_trace,
        findings=generated_findings,
        explanations=explanations,
        evaluation=evaluation,
        report=report,
    )

    # =========================================================
    # 15. RETURN RESULT
    # =========================================================

    return AuditPipelineResult(
        generated_findings=generated_findings,
        expected_findings=expected_findings,
        evaluation=evaluation,
        report=report,
        explanations=explanations,
        audit_trace=audit_trace,
        audit_output=audit_output,
    )
