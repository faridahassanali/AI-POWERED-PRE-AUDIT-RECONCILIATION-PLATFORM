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
    load_data,
    build_unified_customer_record,
)

from engine.normalization import (
    normalize_dataframe,
)

from engine.finding_validator import (
    validate_finding_or_raise,
)

from engine.finding_integrity import (
    validate_unique_findings,
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

from engine.audit_trace import (
    create_audit_trace,
    complete_audit_trace,
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
    """

    for finding in findings:

        validate_finding_or_raise(
            finding
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

    generated_findings = run_all_controls(
        unified=unified,
        tables=normalized_tables,
    )

    # =========================================================
    # 7. ATTACH AUDIT RUN ID
    # =========================================================

    for finding in generated_findings:

        finding["audit_run_id"] = audit_run_id

    # =========================================================
    # 8. FINDING VALIDATION
    # =========================================================

    _validate_generated_findings(
        generated_findings
    )

    # =========================================================
    # 9. FINDING INTEGRITY
    # =========================================================

    validate_unique_findings(
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

    # IMPORTANT:
    #
    # All generated findings are still in REVIEW status.
    #
    # Therefore, NO explanations are generated here.
    #
    # This is the human review gate.
    #
    # The deterministic audit stage must finish before
    # any explanation or future AI processing occurs.

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
) -> list[dict[str, Any]]:
    """
    Stage 2 of the audit pipeline.

    Generate explanations ONLY for findings that have
    passed human review and are CONFIRMED.

    The AI Input Contract enforces the confirmed-only gate
    before the explanation layer is called.
    """

    explanations: list[dict[str, Any]] = []

    for finding in findings:

        ai_input = build_ai_input(
            finding
        )

        explanation = explain_finding(
            ai_input
        )

        explanations.append(
            explanation
        )

    return explanations