"""
End-to-End Audit Pipeline.

Pipeline:

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
Ground Truth Evaluation
    ↓
Evaluation Report
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from engine.data_loader import (
    load_data,
    build_unified_customer_record,
)

from engine.normalization import normalize_dataframe

from engine.finding_validator import validate_finding_or_raise

from engine.finding_integrity import validate_unique_findings

from engine.ground_truth_evaluator import (
    EvaluationResult,
    evaluate_findings,
)

from engine.evaluation_report import generate_evaluation_report

from engine.controls import run_all_controls

from uuid import uuid4

from engine.finding_explainer import explain_finding
from engine.audit_trace import (
    create_audit_trace,
    complete_audit_trace,
)

from engine.audit_output import (
    AuditOutput,
    build_audit_output,
)

from engine.audit_output import build_audit_output

@dataclass
class AuditPipelineResult:

    generated_findings: list[dict[str, Any]]
    expected_findings: list[dict[str, Any]]
    evaluation: EvaluationResult
    report: str

    explanations: list[dict[str, Any]]
    audit_trace: Any

    audit_output: Any


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
    Validate every generated finding against the
    finding schema.
    """

    for finding in findings:
        validate_finding_or_raise(finding)


def run_audit(
    data_dir: Path | str | None = None,
) -> AuditPipelineResult:
    """
    Run the complete pre-audit pipeline.

    Parameters
    ----------
    data_dir:
        Optional custom data directory.

        If omitted, the default data directory configured
        inside data_loader.py is used.

    Returns
    -------
    AuditPipelineResult
        Contains generated findings, expected findings,
        evaluation metrics, explanations, audit trace
        and formatted report.
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

    controls_executed = [
        "SCREENING_001",
        "RISK_001",
        "ARABIC_NAME_001",
        "DORMANT_001",
        "RECON_001",
    ]

    audit_trace = create_audit_trace(
        audit_run_id=audit_run_id,
        controls_executed=controls_executed,
        total_records_evaluated=len(unified),
    )

    # =========================================================
    # 6. RUN ALL CONTROLS
    # =========================================================

    generated_findings = run_all_controls(
        unified=unified,
        tables=normalized_tables,
    )

    # =========================================================
    # 7. ATTACH AUDIT RUN ID TO FINDINGS
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
    # 10. FINDING EXPLAINABILITY
    # =========================================================

    explanations = [
        explain_finding(finding)
        for finding in generated_findings
    ]

    # =========================================================
    # 11. COMPLETE AUDIT TRACE
    # =========================================================

    audit_trace = complete_audit_trace(
        trace=audit_trace,
        total_findings_generated=len(
            generated_findings
        ),
    )

    # =========================================================
    # 12. LOAD EXPECTED FINDINGS / GROUND TRUTH
    # =========================================================

    expected_findings = normalized_tables[
        "expected_findings"
    ].to_dict(
        orient="records"
    )

    # =========================================================
    # 13. GROUND TRUTH EVALUATION
    # =========================================================

    evaluation = evaluate_findings(
        generated_findings=generated_findings,
        expected_findings=expected_findings,
    )

    # =========================================================
    # 14. EVALUATION REPORT
    # =========================================================

    report = generate_evaluation_report(
        evaluation
    )
    
    # =========================================================
    # 15. AUDIT OUTPUT / EVIDENCE PACKAGING
    # =========================================================

    audit_output = build_audit_output(
    audit_trace=audit_trace,
    findings=generated_findings,
    explanations=explanations,
    evaluation=evaluation,
    report=report,
    )

    # =========================================================
    # 16. RETURN COMPLETE RESULT
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