"""
AI Orchestration Layer.

Connects human-reviewed findings to the existing
Stage 3 AI explanation pipeline.

Flow:

Human Review
    ↓
CONFIRMED findings only
    ↓
AI Explanation Pipeline
    ↓
RAG
    ↓
AI Input Validation
    ↓
LLM Router
    ↓
Hallucination Tripwire
    ↓
AI Output Validation

This module does not perform persistence
and does not import Supabase.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from engine.ai_explanation_pipeline import (
    AIExplanationResult,
    generate_ai_explanations,
)
from engine.data_loader import DATA_DIR
from engine.finding_review import get_confirmed_findings
from engine.llm.base import LLMProvider
from engine.policy_registry import (
    PolicyRegistry,
    load_policy_registry,
)


@dataclass
class AIOrchestrationResult:
    """
    Result of the AI orchestration stage.
    """

    total_findings: int
    confirmed_findings: int
    successful_explanations: int
    failed_explanations: int
    results: list[AIExplanationResult]


def run_ai_stage(
    findings: list[dict[str, Any]],
    registry: PolicyRegistry | None = None,
    data_dir: Path | str | None = None,
    primary: LLMProvider | None = None,
    fallback: LLMProvider | None = None,
) -> AIOrchestrationResult:
    """
    Run the AI explanation stage for human-confirmed findings.

    REVIEW and REJECTED findings never enter the AI pipeline.

    The existing Stage 3 pipeline remains responsible for:

        - policy retrieval
        - AI input validation
        - LLM generation
        - hallucination detection
        - AI output validation

    This function only orchestrates the entry into Stage 3.
    """

    # ---------------------------------------------------------
    # HUMAN REVIEW GATE
    # ---------------------------------------------------------
    # Only CONFIRMED findings are allowed to reach the
    # AI explanation pipeline.
    # ---------------------------------------------------------

    confirmed_findings = get_confirmed_findings(
        findings
    )

    # ---------------------------------------------------------
    # LOAD POLICY REGISTRY
    # ---------------------------------------------------------

    if registry is None:
        registry = load_policy_registry(
            Path(data_dir)
            if data_dir is not None
            else DATA_DIR
        )

    # ---------------------------------------------------------
    # RUN EXISTING AI PIPELINE
    # ---------------------------------------------------------

    results = generate_ai_explanations(
        findings=confirmed_findings,
        registry=registry,
        data_dir=data_dir,
        primary=primary,
        fallback=fallback,
    )

    # ---------------------------------------------------------
    # CALCULATE RESULTS
    # ---------------------------------------------------------

    successful_explanations = sum(
        1
        for result in results
        if result.succeeded
    )

    failed_explanations = sum(
        1
        for result in results
        if not result.succeeded
    )

    # ---------------------------------------------------------
    # RETURN ORCHESTRATION RESULT
    # ---------------------------------------------------------

    return AIOrchestrationResult(
        total_findings=len(findings),
        confirmed_findings=len(confirmed_findings),
        successful_explanations=successful_explanations,
        failed_explanations=failed_explanations,
        results=results,
    )