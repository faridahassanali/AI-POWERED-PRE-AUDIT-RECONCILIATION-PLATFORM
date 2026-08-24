"""
LLM/RAG Quality Evaluation -- Task A (Tue 25).

"Evaluate RAG/LLM quality on a sample of confirmed findings
(grounding, relevance, no unsupported claims); tune the prompt."

What this module does NOT re-check
-----------------------------------

Schema validity, registry grounding (cited policy_id exists),
no-mutation, and no-invented-evidence are already enforced by
engine.ai_output_validation, and numeric/status hallucinations are
already caught by engine.llm.hallucination_tripwire -- every item
that reaches this module's "succeeded" list has ALREADY passed all
of that. Re-implementing those checks here would be redundant.

What this module adds
----------------------

1. A STRATIFIED sample across all 5 control types, not a random
   slice -- SCREENING_001 alone is 127 of 223 findings, so a plain
   random sample would barely touch ARABIC_NAME_001 (11) or RISK_001
   (8).

2. Aggregate signals the per-finding pipeline doesn't surface on its
   own: success rate, which provider actually answered each call
   (Groq vs. Gemini -- a high fallback rate would mean Groq is
   flaky), and a failure breakdown by exception type.

3. RELEVANCE and "no unsupported claims beyond what the automated
   gates catch" are NOT automatable the same way grounding is --
   there's no ground truth for "is this explanation actually
   useful," and a second LLM call to judge the first would just move
   the trust problem, not solve it. This module produces a Markdown
   report with every explanation laid out for a human (Person A) to
   read and score, plus specific questions to ask per item -- same
   principle as the rest of this platform: automate what can be
   automated, keep a human gate on judgment calls.

NOT part of the pytest suite on purpose
----------------------------------------

run_evaluation() makes REAL Groq/Gemini API calls by default (unless
primary/fallback are injected, which is what tests/test_llm_
evaluation.py does with fakes). It's meant to be run manually:

    python -m engine.llm_evaluation
    python -m engine.llm_evaluation --per-control 8

Output goes to evaluation_reports/ (gitignored -- generated content,
not source) as both JSON (full detail, machine-readable) and
Markdown (for the manual relevance pass).
"""

from __future__ import annotations

import copy
import json
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from engine.ai_explanation_pipeline import generate_ai_explanation_for_finding
from engine.audit_pipeline import run_audit
from engine.data_loader import DATA_DIR
from engine.finding_review import confirm_finding
from engine.llm.base import LLMProvider
from engine.policy_registry import PolicyRegistry, load_policy_registry


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "evaluation_reports"
DEFAULT_PER_CONTROL = 5
DEFAULT_DELAY_BETWEEN_CALLS_SECONDS = 2.5


# =====================================================================
# DATA SHAPES
# =====================================================================

@dataclass
class EvaluationItem:
    finding_id: str
    control_id: str
    severity: str
    succeeded: bool
    provider_used: str | None = None
    model_used: str | None = None
    ai_explanation: str | None = None
    ai_recommendation: str | None = None
    cited_policy_ids: list[str] = field(default_factory=list)
    error: str | None = None
    error_type: str | None = None


@dataclass
class EvaluationReport:
    generated_at: str
    sample_size_per_control: int
    items: list[EvaluationItem]

    @property
    def total(self) -> int:
        return len(self.items)

    @property
    def succeeded(self) -> list[EvaluationItem]:
        return [item for item in self.items if item.succeeded]

    @property
    def failed(self) -> list[EvaluationItem]:
        return [item for item in self.items if not item.succeeded]

    def success_rate(self) -> float:
        return len(self.succeeded) / self.total if self.total else 0.0

    def provider_distribution(self) -> dict[str, int]:
        return dict(Counter(item.provider_used for item in self.succeeded))

    def failure_breakdown(self) -> dict[str, int]:
        return dict(Counter(item.error_type for item in self.failed))

    def per_control_breakdown(self) -> dict[str, dict[str, int]]:

        result: dict[str, dict[str, int]] = defaultdict(
            lambda: {"succeeded": 0, "failed": 0}
        )

        for item in self.items:
            key = "succeeded" if item.succeeded else "failed"
            result[item.control_id][key] += 1

        return dict(result)


# =====================================================================
# SAMPLING
# =====================================================================

def select_stratified_sample(
    findings: list[dict[str, Any]],
    per_control: int = DEFAULT_PER_CONTROL,
) -> list[dict[str, Any]]:
    """
    Up to `per_control` findings per control_id, so every control
    type is represented regardless of how skewed the raw finding
    counts are (SCREENING_001: 127, RISK_001: 8, out of 223 total).
    """

    by_control: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for finding in findings:
        by_control[finding["control_id"]].append(finding)

    sample: list[dict[str, Any]] = []

    for control_id in sorted(by_control):
        sample.extend(by_control[control_id][:per_control])

    return sample


# =====================================================================
# EVALUATION RUN
# =====================================================================

def run_evaluation(
    per_control: int = DEFAULT_PER_CONTROL,
    registry: PolicyRegistry | None = None,
    primary: LLMProvider | None = None,
    fallback: LLMProvider | None = None,
    delay_between_calls: float = DEFAULT_DELAY_BETWEEN_CALLS_SECONDS,
) -> EvaluationReport:
    """
    Run a stratified sample of CONFIRMED findings through the real
    Stage 3 pipeline (engine.ai_explanation_pipeline) and collect
    per-finding + aggregate results.

    primary/fallback default to None, which means
    generate_ai_explanation_for_finding() falls through to the real
    GroqProvider()/GeminiProvider() -- this makes real API calls
    unless a test injects fakes.

    delay_between_calls: seconds to sleep between findings. Matters
    because Gemini's free tier is far stricter on requests-per-minute
    than Groq's (roughly 10-15 RPM vs. Groq's 30 RPM) -- and once
    Groq starts rate-limiting (which happens well before this
    script's ~25-call sample exhausts Groq's own daily quota, simply
    from back-to-back calls with no spacing), nearly every finding
    falls over to Gemini in quick succession, which then hits ITS
    per-minute limit too. Spacing calls out keeps both providers
    comfortably under their per-minute ceilings for a run this size.
    Set to 0 to disable (e.g. in tests, where providers are fakes and
    there's no real rate limit to respect).
    """

    if registry is None:
        registry = load_policy_registry(DATA_DIR)

    audit_result = run_audit()

    sample = select_stratified_sample(
        audit_result.generated_findings,
        per_control=per_control,
    )

    confirmed = [
        confirm_finding(copy.deepcopy(finding), reviewed_by="llm-eval-script")
        for finding in sample
    ]

    items: list[EvaluationItem] = []

    for index, finding in enumerate(confirmed):

        if index > 0 and delay_between_calls > 0:
            time.sleep(delay_between_calls)

        result = generate_ai_explanation_for_finding(
            finding,
            registry=registry,
            primary=primary,
            fallback=fallback,
        )

        if result.succeeded:
            ai_output = result.ai_output or {}
            items.append(
                EvaluationItem(
                    finding_id=result.finding_id,
                    control_id=finding["control_id"],
                    severity=finding["severity"],
                    succeeded=True,
                    provider_used=ai_output.get("provider_used"),
                    model_used=ai_output.get("model_used"),
                    ai_explanation=ai_output.get("ai_explanation"),
                    ai_recommendation=ai_output.get("ai_recommendation"),
                    cited_policy_ids=ai_output.get("cited_policy_ids", []),
                )
            )
        else:
            error_type = (
                result.error.split(":", 1)[0] if result.error else "Unknown"
            )
            items.append(
                EvaluationItem(
                    finding_id=result.finding_id,
                    control_id=finding["control_id"],
                    severity=finding["severity"],
                    succeeded=False,
                    error=result.error,
                    error_type=error_type,
                )
            )

    return EvaluationReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        sample_size_per_control=per_control,
        items=items,
    )


# =====================================================================
# REPORT OUTPUT
# =====================================================================

def write_report(
    report: EvaluationReport,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> tuple[Path, Path]:
    """
    Write the report as JSON (full detail) and Markdown (for the
    manual relevance pass). Returns (json_path, md_path).
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")

    json_path = output_dir / f"llm_quality_eval_{timestamp}.json"
    md_path = output_dir / f"llm_quality_eval_{timestamp}.md"

    json_path.write_text(
        json.dumps(
            {
                "generated_at": report.generated_at,
                "sample_size_per_control": report.sample_size_per_control,
                "total": report.total,
                "success_rate": report.success_rate(),
                "provider_distribution": report.provider_distribution(),
                "failure_breakdown": report.failure_breakdown(),
                "per_control_breakdown": report.per_control_breakdown(),
                "items": [asdict(item) for item in report.items],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    md_path.write_text(_render_markdown(report), encoding="utf-8")

    return json_path, md_path


def _render_markdown(report: EvaluationReport) -> str:

    lines = [
        "# LLM/RAG Quality Evaluation",
        "",
        f"Generated: {report.generated_at}",
        f"Sample size per control: {report.sample_size_per_control}",
        f"Total sampled: {report.total}",
        f"Success rate: {report.success_rate():.0%}",
        "",
        "## Automated signals",
        "",
        "Everything below this line was already checked by the "
        "pipeline's own gates (schema, registry grounding, "
        "no-mutation, hallucination tripwire) -- a finding only "
        "appears in 'succeeded' if it passed all of them.",
        "",
        "### Provider distribution (successful calls)",
        "",
    ]

    provider_distribution = report.provider_distribution()
    if provider_distribution:
        for provider, count in provider_distribution.items():
            lines.append(f"- {provider}: {count}")
    else:
        lines.append("(no successful calls)")

    lines += ["", "### Failure breakdown", ""]

    failure_breakdown = report.failure_breakdown()
    if failure_breakdown:
        for error_type, count in failure_breakdown.items():
            lines.append(f"- {error_type}: {count}")
    else:
        lines.append("(none)")

    lines += ["", "### Per-control breakdown", ""]
    lines.append("| control_id | succeeded | failed |")
    lines.append("|---|---|---|")

    for control_id, counts in sorted(report.per_control_breakdown().items()):
        lines.append(
            f"| {control_id} | {counts['succeeded']} | {counts['failed']} |"
        )

    lines += [
        "",
        "## Manual review (relevance, no unsupported claims)",
        "",
        "For each item below, check:",
        "",
        "- Is the explanation specific to THIS finding, or generic "
        "boilerplate that could apply to any finding of this control?",
        "- Does the recommendation give a concrete next step, or "
        "vague advice ('review and take appropriate action')?",
        "- Does anything feel unsupported even though it passed the "
        "automated hallucination tripwire? (the tripwire only catches "
        "numbers/dates/status-vocabulary contradictions -- not every "
        "form of unsupported claim, e.g. an overstated causal link "
        "or a policy requirement paraphrased too loosely.)",
        "",
    ]

    for item in report.succeeded:
        lines += [
            f"### {item.finding_id} -- {item.control_id} ({item.severity})",
            "",
            f"**Provider:** {item.provider_used} ({item.model_used})",
            "",
            f"**Explanation:** {item.ai_explanation}",
            "",
            f"**Recommendation:** {item.ai_recommendation}",
            "",
            f"**Cited policies:** {', '.join(item.cited_policy_ids) or '(none)'}",
            "",
            "**Relevance (1-5):** ___   **Notes:** "
            "_______________________________",
            "",
            "---",
            "",
        ]

    if report.failed:
        lines += ["## Failed items", ""]
        for item in report.failed:
            lines += [
                f"### {item.finding_id} -- {item.control_id}",
                f"**Error:** {item.error}",
                "",
            ]

    return "\n".join(lines)


# =====================================================================
# CLI ENTRY POINT
# =====================================================================

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate RAG/LLM explanation quality on a stratified "
            "sample of confirmed findings. Makes REAL Groq/Gemini "
            "API calls -- run manually, not in CI."
        )
    )
    parser.add_argument(
        "--per-control",
        type=int,
        default=DEFAULT_PER_CONTROL,
        help=f"Max findings to sample per control_id (default: {DEFAULT_PER_CONTROL}).",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=DEFAULT_DELAY_BETWEEN_CALLS_SECONDS,
        help=(
            "Seconds to wait between findings, to stay under "
            "Gemini's stricter per-minute free-tier limit "
            f"(default: {DEFAULT_DELAY_BETWEEN_CALLS_SECONDS}). "
            "Increase this if you still see rate-limit failures."
        ),
    )
    args = parser.parse_args()

    print(
        f"Running evaluation with up to {args.per_control} finding(s) per "
        f"control, {args.delay}s between calls..."
    )

    evaluation_report = run_evaluation(
        per_control=args.per_control,
        delay_between_calls=args.delay,
    )

    json_report_path, md_report_path = write_report(evaluation_report)

    print(
        f"\nSampled: {evaluation_report.total}   "
        f"Success rate: {evaluation_report.success_rate():.0%}"
    )
    print(f"Provider distribution: {evaluation_report.provider_distribution()}")

    if evaluation_report.failed:
        print(f"Failure breakdown: {evaluation_report.failure_breakdown()}")

    print(f"\nJSON report:     {json_report_path}")
    print(f"Markdown report: {md_report_path}  (open this one for manual review)")