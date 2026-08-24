"""
Thursday Checkpoint -- Integration Test (real version).

"All 3 together: run one finding end-to-end through review -> gate ->
RAG retrieval -> Supabase persistence (LLM call still stubbed/mocked
at this point). Fix integration bugs."

Unlike an earlier draft of this test, EVERY stage below is the real,
merged production code -- nothing is monkeypatched or worked around,
because by the time this was written:

    - RAG/retriever.py (Task A) was merged: retrieve_for_finding()
      resolves a finding's policy_references directly against the
      Policy Registry -- it never substitutes a different policy.
    - engine/ai_input.py (Task B) requires a non-empty policy_context
      and blocks findings with missing evidence.
    - engine/audit_pipeline.py::explain_confirmed_findings() (Stage 2)
      now wires retrieve_for_finding() -> build_ai_input() ->
      explain_finding() together -- this was previously unwired dead
      code with zero callers and zero test coverage.
    - engine/persistence.py + engine/audit_orchestration.py (Task C)
      persist the audit run and findings.

Only the LLM call (Sunday's task, not built yet) is stubbed here.

Two variants:

    - test_checkpoint_end_to_end_with_fake_supabase(): the main
      checkpoint. Real registry, real retrieval, real gate, fake
      Supabase client (same pattern as test_persistence.py /
      test_audit_orchestration.py) so it runs in CI with no external
      infra.

    - test_checkpoint_rejected_finding_never_reaches_the_gate(): the
      reject path, confirming the gate actually blocks.
"""

import copy
from pathlib import Path
from typing import Any

import pytest

from engine.ai_input import AIInputValidationError, build_ai_input
from engine.audit_pipeline import explain_confirmed_findings, run_audit
from engine.finding_review import confirm_finding, reject_finding
from engine.persistence import write_audit_run, write_finding_review, write_findings
from engine.policy_registry import load_policy_registry
from RAG.retriever import retrieve_for_finding


DATA_DIR = Path(__file__).resolve().parent.parent / "data"


# =====================================================================
# FAKE SUPABASE CLIENT (same shape used across test_persistence.py,
# test_audit_orchestration.py, test_explain_confirmed_findings.py)
# =====================================================================

class _FakeResponse:
    def __init__(self, data):
        self.data = data


class _FakeTable:
    def __init__(self, name, calls):
        self.name = name
        self.calls = calls
        self._pending = None

    def upsert(self, rows, on_conflict=None):
        self._pending = ("upsert", rows, on_conflict)
        return self

    def insert(self, row):
        self._pending = ("insert", row, None)
        return self

    def execute(self):
        op, payload, on_conflict = self._pending
        self.calls.append(
            {"table": self.name, "op": op, "payload": payload, "on_conflict": on_conflict}
        )
        data = payload if isinstance(payload, list) else [payload]
        return _FakeResponse(data)


class _FakeClient:
    def __init__(self):
        self.calls = []

    def table(self, name):
        return _FakeTable(name, self.calls)


# =====================================================================
# STAGE 5 STUB -- the LLM call doesn't exist yet (Sunday's task)
# =====================================================================

def _stub_llm_call(ai_input: dict[str, Any]) -> dict[str, Any]:
    """
    Stands in for the real LLM integration. Doesn't call any model --
    just proves the gated, RAG-grounded ai_input reaches this point
    intact, which is all the checkpoint needs from this stage.
    """
    return {
        "finding_id": ai_input["finding_id"],
        "audit_run_id": ai_input["audit_run_id"],
        "ai_explanation": (
            f"[STUBBED] Explanation for {ai_input['control_id']} would be "
            f"generated here, grounded on {len(ai_input['policy_context'])} "
            "retrieved policy chunk(s)."
        ),
        "ai_recommendation": "[STUBBED] Recommendation would be generated here.",
        "cited_policy_ids": sorted(
            {c["policy_id"] for c in ai_input["policy_context"]}
        ),
    }


@pytest.fixture(scope="module")
def real_registry():
    return load_policy_registry(DATA_DIR)


# =====================================================================
# MAIN CHECKPOINT -- everything real except the LLM call
# =====================================================================

def test_checkpoint_end_to_end_with_fake_supabase(real_registry):
    client = _FakeClient()

    # 1. Deterministic engine.
    pipeline_result = run_audit()
    assert pipeline_result.generated_findings
    finding = copy.deepcopy(pipeline_result.generated_findings[0])
    assert finding["finding_status"] == "REVIEW"

    # 2. Human review.
    previous_status = finding["finding_status"]
    finding = confirm_finding(finding, reviewed_by="checkpoint-test")
    assert finding["finding_status"] == "CONFIRMED"

    write_finding_review(finding, previous_status=previous_status, client=client)

    # 3+4. RAG retrieval, grounded strictly to this finding's own
    # policy_references (Task A) -- then the gate (Task B).
    policy_context = retrieve_for_finding(finding=finding, registry=real_registry)
    assert policy_context, "retrieval returned nothing for a real finding"

    ai_input = build_ai_input(finding, policy_context=policy_context)
    assert ai_input["finding_status"] == "CONFIRMED"
    assert ai_input["policy_context"] == policy_context

    # 5. LLM call -- stubbed.
    ai_output = _stub_llm_call(ai_input)
    assert ai_output["cited_policy_ids"]

    # 6. Persistence (Task C).
    write_audit_run(pipeline_result.audit_trace, client=client)
    write_findings([finding], client=client)

    tables_written = {c["table"] for c in client.calls}
    assert {"audit_runs", "findings", "finding_reviews"} <= tables_written


def test_checkpoint_via_explain_confirmed_findings(real_registry):
    """
    Same flow, but through the actual Stage 2 orchestration function
    (engine.audit_pipeline.explain_confirmed_findings) instead of
    calling retrieve_for_finding()/build_ai_input() by hand -- this is
    the real code path a caller would use.
    """
    pipeline_result = run_audit()
    finding = confirm_finding(
        copy.deepcopy(pipeline_result.generated_findings[0]),
        reviewed_by="checkpoint-test",
    )

    explanations = explain_confirmed_findings([finding], registry=real_registry)

    assert len(explanations) == 1
    assert explanations[0]["finding_id"] == finding["finding_id"]
    assert explanations[0]["policy_references"]


def test_checkpoint_rejected_finding_never_reaches_the_gate(real_registry):
    pipeline_result = run_audit()
    finding = reject_finding(
        copy.deepcopy(pipeline_result.generated_findings[0]),
        reviewed_by="checkpoint-test",
    )

    with pytest.raises(AIInputValidationError, match="CONFIRMED"):
        explain_confirmed_findings([finding], registry=real_registry)
