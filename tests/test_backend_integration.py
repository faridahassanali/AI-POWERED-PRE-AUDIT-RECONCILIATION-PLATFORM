"""
Backend API integration tests.

These tests exercise backend/main.py through FastAPI TestClient while
keeping Supabase fully in-memory. The fake implements both the fluent
table API and the claim_audit_idempotency RPC now used by the endpoint.

Coverage map
------------
POST /audit-runs/execute            -- success, duplicate idempotency,
                                        evaluation-save failure, auth
GET  /audit-runs                    -- list
GET  /audit-runs/{id}               -- not_found, success
GET  /audit-runs/{id}/evaluation    -- not_found, success
PATCH /audit-runs/{id}              -- not_found, success
GET  /findings                      -- unfiltered, severity filter,
                                        combined status/control/run filter
GET  /dashboard/summary             -- status + severity counts
GET  /findings/{id}/policy          -- not_found, success
POST /findings                      -- auth required, success
PATCH /findings/{id}                -- (see test_confirm/reject section)
GET  /findings/{id}/reviews         -- not_found, success

Note: AI-explanation (Stage 3) and hallucination-guard paths have
their own dedicated test files (test_ai_explanation_pipeline.py,
test_hallucination_*.py) and are only partially re-exercised here at
the HTTP layer.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from engine.audit_trace import AuditTrace
from engine.ground_truth_evaluator import EvaluationResult

import backend.main as backend_main
import backend.auth as backend_auth
import engine.persistence as persistence


class _FakeQuery:
    def __init__(self, table):
        self._table = table
        self._op = None
        self._payload = None
        self._filters = []
        self._order_by = None
        self._order_desc = False
        self._limit_n = None
        self._on_conflict = None

    def select(self, *_args, **_kwargs):
        self._op = "select"
        return self

    def insert(self, payload):
        self._op = "insert"
        self._payload = payload
        return self

    def update(self, payload):
        self._op = "update"
        self._payload = payload
        return self

    def upsert(self, payload, on_conflict=None):
        self._op = "upsert"
        self._payload = payload
        self._on_conflict = on_conflict
        return self

    def eq(self, field, value):
        self._filters.append((field, value))
        return self

    def order(self, field, desc=False):
        self._order_by = field
        self._order_desc = desc
        return self

    def limit(self, n):
        self._limit_n = n
        return self

    def _matching_rows(self):
        rows = list(self._table.rows)
        for field, value in self._filters:
            rows = [row for row in rows if row.get(field) == value]
        if self._order_by is not None:
            rows.sort(
                key=lambda row: row.get(self._order_by) or "",
                reverse=self._order_desc,
            )
        if self._limit_n is not None:
            rows = rows[: self._limit_n]
        return rows

    def execute(self):
        if self._op == "select":
            return SimpleNamespace(data=self._matching_rows())

        if self._op == "insert":
            rows = self._payload if isinstance(self._payload, list) else [self._payload]
            inserted = []
            for row in rows:
                stored = dict(row)
                self._table.rows.append(stored)
                inserted.append(stored)
            return SimpleNamespace(data=inserted)

        if self._op == "update":
            matched = self._matching_rows()
            for row in matched:
                row.update(self._payload)
            return SimpleNamespace(data=matched)

        if self._op == "upsert":
            rows = self._payload if isinstance(self._payload, list) else [self._payload]
            key_field = self._on_conflict or "id"
            result = []
            for row in rows:
                existing = next(
                    (stored for stored in self._table.rows
                     if stored.get(key_field) == row.get(key_field)),
                    None,
                )
                if existing is None:
                    stored = dict(row)
                    self._table.rows.append(stored)
                    result.append(stored)
                else:
                    existing.update(row)
                    result.append(existing)
            return SimpleNamespace(data=result)

        raise RuntimeError(f"Unsupported fake query op: {self._op!r}")


class FakeTable:
    def __init__(self):
        self.rows = []

    def query(self):
        return _FakeQuery(self)


class FakeSupabaseClient:
    def __init__(self):
        self._tables = {}
        self._idempotency = {}

    def table(self, name):
        if name not in self._tables:
            self._tables[name] = FakeTable()
        return self._tables[name].query()

    def get_table(self, name):
        if name not in self._tables:
            self._tables[name] = FakeTable()
        return self._tables[name]

    def rpc(self, function_name, params):
        if function_name != "claim_audit_idempotency":
            raise RuntimeError(f"Unsupported fake RPC: {function_name}")

        key = params["p_idempotency_key"]
        audit_run_id = params["p_audit_run_id"]

        if key in self._idempotency:
            existing_id = self._idempotency[key]
            return SimpleNamespace(
                execute=lambda: SimpleNamespace(
                    data=[{"claimed": False, "audit_run_id": existing_id}]
                )
            )

        self._idempotency[key] = audit_run_id
        return SimpleNamespace(
            execute=lambda: SimpleNamespace(
                data=[{"claimed": True, "audit_run_id": audit_run_id}]
            )
        )


@pytest.fixture
def fake_db(monkeypatch):
    client = FakeSupabaseClient()
    monkeypatch.setattr(backend_main, "supabase", client)
    monkeypatch.setattr(backend_main, "supabase_anon", client)
    monkeypatch.setattr(persistence, "get_supabase_client", lambda: client)
    return client


@pytest.fixture
def api_client():
    return TestClient(backend_main.app)


@pytest.fixture
def auth_headers(monkeypatch):
    monkeypatch.setenv("APP_API_KEYS", "test-key")
    return {"X-API-Key": "test-key"}


@pytest.fixture
def execute_headers(auth_headers):
    def build(key):
        return {**auth_headers, "Idempotency-Key": key}
    return build


def _sample_result(audit_run_id="AUDIT-TEST-001", status="COMPLETED", error_type=None, error_message=None):
    evaluation = EvaluationResult(
        true_positives=2,
        false_positives=0,
        false_negatives=0,
        precision=1.0,
        recall=1.0,
        f1_score=1.0,
        matched=[],
        false_positive_findings=[],
        false_negative_findings=[],
        per_control={},
        per_severity={},
    )
    trace = AuditTrace(
        audit_run_id=audit_run_id,
        started_at="2026-01-01T00:00:00Z",
        completed_at="2026-01-01T00:01:00Z",
        controls_executed=["RISK_001"],
        total_records_evaluated=10,
        total_findings_generated=1,
        status=status,
        error_type=error_type,
        error_message=error_message,
    )
    finding = {
        "finding_id": "F-TEST-001",
        "audit_run_id": audit_run_id,
        "control_id": "RISK_001",
        "customer_id": "CUST100001",
        "severity": "HIGH",
        "assessment_status": "FAIL",
        "finding_status": "REVIEW",
        "expected": "Expected condition",
        "actual": "Actual condition",
        "evidence": {"field": "value"},
        "policy_references": [],
        "reviewed_by": None,
        "review_timestamp": None,
        "reviewer_notes": None,
        "ai_explanation": None,
        "ai_recommendation": None,
    }
    findings = [] if status == "FAILED" else [finding]
    return SimpleNamespace(
        audit_trace=trace,
        generated_findings=findings,
        evaluation=evaluation,
        report="fake report text",
        pre_audit_report="fake pre-audit report text",
        audit_output=SimpleNamespace(
            audit_run_id=audit_run_id,
            findings=[
                SimpleNamespace(finding=finding, explanation=None)
            ] if status != "FAILED" else [],
            report="fake report text",
            evaluation=evaluation,
            audit_trace=trace,
        ),
    )


def _sample_finding(**overrides) -> dict:
    """
    Standalone finding row for endpoints that don't go through
    run_audit()/_sample_result() -- GET /findings, /dashboard/summary,
    /findings/{id}/policy, /findings/{id}/reviews, POST /findings.
    """

    row = {
        "finding_id": "F-TEST-001",
        "audit_run_id": "AUDIT-TEST-001",
        "control_id": "RISK_001",
        "customer_id": "CUST100001",
        "severity": "HIGH",
        "assessment_status": "FAIL",
        "finding_status": "REVIEW",
        "expected": "Expected condition",
        "actual": "Actual condition",
        "evidence": {"field": "value"},
        "policy_references": [],
        "reviewed_by": None,
        "review_timestamp": None,
        "reviewer_notes": None,
        "ai_explanation": None,
        "ai_recommendation": None,
        "created_at": "2026-01-01T00:00:00Z",
        "updated_at": "2026-01-01T00:00:00Z",
    }
    row.update(overrides)
    return row


# =====================================================================
# POST /audit-runs/execute
# =====================================================================


def test_execute_audit_success_persists_run_findings_and_evaluation(
    api_client, execute_headers, fake_db, monkeypatch
):
    monkeypatch.setattr(
        backend_main,
        "run_audit",
        lambda audit_run_id: _sample_result(audit_run_id),
    )

    response = api_client.post(
        "/audit-runs/execute",
        headers=execute_headers("idem-success"),
    )

    assert response.status_code == 200, response.text
    body = response.json()

    assert body["status"] == "success"
    assert body["audit_run_id"].startswith("AUDIT-")
    assert body["total_findings_generated"] == 1
    assert body["audit_run_saved"] is True
    assert body["findings_saved"] == 1
    assert body["evaluation_saved"] is True
    assert body["evaluation_save_error"] is None
    assert body["pre_audit_report"] == "fake pre-audit report text"
    assert len(fake_db.get_table("audit_runs").rows) == 1
    assert len(fake_db.get_table("findings").rows) == 1
    assert len(fake_db.get_table("audit_evaluations").rows) == 1

def test_execute_audit_duplicate_idempotency_does_not_rerun_pipeline(
    api_client, execute_headers, fake_db, monkeypatch
):
    calls = []
    results = []

    def fake_run_audit(audit_run_id):
        calls.append(audit_run_id)
        result = _sample_result(audit_run_id)
        results.append(result)
        return result

    monkeypatch.setattr(backend_main, "run_audit", fake_run_audit)

    first = api_client.post(
        "/audit-runs/execute",
        headers=execute_headers("idem-duplicate"),
    )
    assert first.json()["status"] == "success"

    second = api_client.post(
        "/audit-runs/execute",
        headers=execute_headers("idem-duplicate"),
    )

    assert second.status_code == 200
    assert second.json()["status"] == "duplicate"
    assert second.json()["audit_run"]["audit_run_id"] == calls[0]
    assert len(calls) == 1


def test_execute_audit_reports_evaluation_save_failure_without_failing_run(
    api_client, execute_headers, fake_db, monkeypatch
):
    monkeypatch.setattr(
        backend_main,
        "run_audit",
        lambda audit_run_id: _sample_result(audit_run_id),
    )
    monkeypatch.setattr(
        backend_main,
        "write_audit_evaluation",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("audit_evaluations insert failed")
        ),
    )

    response = api_client.post(
        "/audit-runs/execute",
        headers=execute_headers("idem-eval-failure"),
    )

    body = response.json()
    assert response.status_code == 200
    assert body["status"] == "success"
    assert body["evaluation_saved"] is False
    assert "audit_evaluations insert failed" in body["evaluation_save_error"]
    assert len(fake_db.get_table("audit_runs").rows) == 1
    assert len(fake_db.get_table("findings").rows) == 1


def test_execute_audit_rejects_missing_or_wrong_api_key(api_client, fake_db):
    missing = api_client.post(
        "/audit-runs/execute",
        headers={"Idempotency-Key": "idem-missing"},
    )
    wrong = api_client.post(
        "/audit-runs/execute",
        headers={
            "X-API-Key": "wrong-key",
            "Idempotency-Key": "idem-wrong",
        },
    )
    assert missing.status_code == 401
    assert wrong.status_code == 401


# =====================================================================
# GET /audit-runs
# =====================================================================


def test_get_audit_runs_returns_seeded_rows(api_client, fake_db):
    fake_db.get_table("audit_runs").rows.extend(
        [
            {
                "audit_run_id": "AUDIT-A",
                "created_at": "2026-01-01T00:00:00Z",
            },
            {
                "audit_run_id": "AUDIT-B",
                "created_at": "2026-01-02T00:00:00Z",
            },
        ]
    )

    response = api_client.get("/audit-runs")

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "success"
    assert body["count"] == 2
    assert {row["audit_run_id"] for row in body["audit_runs"]} == {
        "AUDIT-A",
        "AUDIT-B",
    }


def test_get_audit_runs_empty_returns_zero_count(api_client, fake_db):
    response = api_client.get("/audit-runs")

    body = response.json()
    assert body["status"] == "success"
    assert body["count"] == 0
    assert body["audit_runs"] == []


# =====================================================================
# GET /audit-runs/{id}
# =====================================================================


def test_get_audit_run_not_found(api_client, fake_db):
    response = api_client.get("/audit-runs/DOES-NOT-EXIST")
    assert response.status_code == 200
    assert response.json()["status"] == "not_found"


def test_get_audit_run_success(api_client, fake_db):
    fake_db.get_table("audit_runs").rows.append(
        {
            "audit_run_id": "AUDIT-FOUND-001",
            "total_findings_generated": 5,
        }
    )

    response = api_client.get("/audit-runs/AUDIT-FOUND-001")

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "success"
    assert body["audit_run"]["audit_run_id"] == "AUDIT-FOUND-001"
    assert body["audit_run"]["total_findings_generated"] == 5


# =====================================================================
# GET /audit-runs/{id}/evaluation
# =====================================================================


def test_get_audit_evaluation_not_found(api_client, fake_db):
    response = api_client.get("/audit-runs/AUDIT-NOPE/evaluation")
    assert response.json()["status"] == "not_found"


def test_get_audit_evaluation_success(api_client, fake_db):
    fake_db.get_table("audit_evaluations").rows.append(
        {
            "audit_run_id": "AUDIT-EVAL-001",
            "true_positives": 5,
            "false_positives": 1,
            "false_negatives": 0,
            "precision": 0.833,
            "recall": 1.0,
            "f1_score": 0.909,
        }
    )

    response = api_client.get("/audit-runs/AUDIT-EVAL-001/evaluation")

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "success"
    assert body["evaluation"]["true_positives"] == 5
    assert body["evaluation"]["f1_score"] == pytest.approx(0.909)


# =====================================================================
# PATCH /audit-runs/{id}
# =====================================================================


def test_update_audit_run_requires_auth(api_client, fake_db):
    response = api_client.patch(
        "/audit-runs/AUDIT-ANY",
        json={"total_records_evaluated": 10},
    )
    assert response.status_code == 401


def test_update_audit_run_not_found(api_client, auth_headers, fake_db):
    response = api_client.patch(
        "/audit-runs/DOES-NOT-EXIST",
        json={"total_records_evaluated": 10},
        headers=auth_headers,
    )
    assert response.json()["status"] == "not_found"


def test_update_audit_run_success_sets_completed_at(
    api_client, auth_headers, fake_db
):
    fake_db.get_table("audit_runs").rows.append(
        {
            "audit_run_id": "AUDIT-UPDATE-001",
            "started_at": "2026-01-01T00:00:00Z",
            "completed_at": None,
            "controls_executed": ["RISK_001"],
            "total_records_evaluated": 10,
            "total_findings_generated": 1,
        }
    )

    response = api_client.patch(
        "/audit-runs/AUDIT-UPDATE-001",
        json={"total_records_evaluated": 999},
        headers=auth_headers,
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "success"
    assert body["audit_run"]["total_records_evaluated"] == 999
    assert body["audit_run"]["completed_at"] is not None


# =====================================================================
# GET /findings -- filters
# =====================================================================


def test_get_findings_returns_seeded_rows(api_client, fake_db):
    fake_db.get_table("findings").rows.extend(
        [
            _sample_finding(finding_id="F-A", severity="HIGH"),
            _sample_finding(finding_id="F-B", severity="LOW"),
        ]
    )

    response = api_client.get("/findings")

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "success"
    assert body["count"] == 2


def test_get_findings_filters_by_severity(api_client, fake_db):
    fake_db.get_table("findings").rows.extend(
        [
            _sample_finding(finding_id="F-A", severity="HIGH"),
            _sample_finding(finding_id="F-B", severity="LOW"),
        ]
    )

    response = api_client.get("/findings", params={"severity": "HIGH"})

    body = response.json()
    assert body["count"] == 1
    assert body["findings"][0]["finding_id"] == "F-A"


def test_get_findings_combined_status_control_and_audit_run_filters(
    api_client, fake_db
):
    fake_db.get_table("findings").rows.extend(
        [
            _sample_finding(
                finding_id="F-A",
                finding_status="REVIEW",
                control_id="RISK_001",
                audit_run_id="AUDIT-X",
            ),
            _sample_finding(
                finding_id="F-B",
                finding_status="CONFIRMED",
                control_id="SCREENING_001",
                audit_run_id="AUDIT-Y",
            ),
            _sample_finding(
                finding_id="F-C",
                finding_status="CONFIRMED",
                control_id="SCREENING_001",
                audit_run_id="AUDIT-X",
            ),
        ]
    )

    response = api_client.get(
        "/findings",
        params={
            "status": "CONFIRMED",
            "control_id": "SCREENING_001",
            "audit_run_id": "AUDIT-Y",
        },
    )

    body = response.json()
    assert body["count"] == 1
    assert body["findings"][0]["finding_id"] == "F-B"


def test_get_findings_no_matches_returns_empty_list(api_client, fake_db):
    fake_db.get_table("findings").rows.append(
        _sample_finding(finding_id="F-A", severity="LOW")
    )

    response = api_client.get("/findings", params={"severity": "CRITICAL"})

    body = response.json()
    assert body["count"] == 0
    assert body["findings"] == []


# =====================================================================
# GET /dashboard/summary
# =====================================================================


def test_dashboard_summary_counts_by_status_and_severity(api_client, fake_db):
    fake_db.get_table("findings").rows.extend(
        [
            _sample_finding(
                finding_id="F-1", finding_status="REVIEW", severity="CRITICAL"
            ),
            _sample_finding(
                finding_id="F-2", finding_status="CONFIRMED", severity="HIGH"
            ),
            _sample_finding(
                finding_id="F-3", finding_status="REJECTED", severity="MEDIUM"
            ),
            _sample_finding(
                finding_id="F-4", finding_status="RESOLVED", severity="LOW"
            ),
        ]
    )

    response = api_client.get("/dashboard/summary")

    assert response.status_code == 200, response.text
    summary = response.json()["summary"]

    assert summary["total_findings"] == 4
    assert summary["review"] == 1
    assert summary["confirmed"] == 1
    assert summary["rejected"] == 1
    assert summary["resolved"] == 1
    assert summary["critical"] == 1
    assert summary["high"] == 1
    assert summary["medium"] == 1
    assert summary["low"] == 1


def test_dashboard_summary_empty_findings(api_client, fake_db):
    response = api_client.get("/dashboard/summary")

    body = response.json()
    assert body["status"] == "success"
    assert body["summary"]["total_findings"] == 0


# =====================================================================
# GET /findings/{id}/policy
# =====================================================================


def test_get_finding_policy_not_found(api_client, fake_db):
    response = api_client.get("/findings/DOES-NOT-EXIST/policy")
    assert response.status_code == 200
    assert response.json()["status"] == "not_found"


def test_get_finding_policy_success(api_client, fake_db, monkeypatch):
    fake_db.get_table("findings").rows.append(
        _sample_finding(
            finding_id="F-POLICY-001",
            policy_references=[
                {"policy_id": "POL-01", "version": "1.0", "section": "3.2"}
            ],
        )
    )

    monkeypatch.setattr(
        backend_main,
        "retrieve_for_finding",
        lambda finding, registry, top_k=3: [
            {"policy_id": "POL-01", "text": "fake policy text"}
        ],
    )

    response = api_client.get("/findings/F-POLICY-001/policy")

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "success"
    assert body["count"] == 1
    assert body["policy_context"][0]["policy_id"] == "POL-01"
    assert body["policy_references"][0]["policy_id"] == "POL-01"


# =====================================================================
# POST /findings
# =====================================================================


def test_create_finding_requires_auth(api_client, fake_db):
    response = api_client.post("/findings", json={})
    assert response.status_code == 401


def test_create_finding_success(api_client, auth_headers, fake_db):
    payload = {
        "finding_id": "F-NEW-001",
        "audit_run_id": "AUDIT-NEW-001",
        "control_id": "RISK_001",
        "customer_id": "CUST100099",
        "severity": "HIGH",
        "assessment_status": "FAIL",
        "finding_status": "REVIEW",
        "expected": "Expected condition",
        "actual": "Actual condition",
        "evidence": {"field": "value"},
        "policy_references": [],
    }

    response = api_client.post(
        "/findings", json=payload, headers=auth_headers
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "success"
    assert body["finding"]["finding_id"] == "F-NEW-001"
    assert len(fake_db.get_table("findings").rows) == 1


def test_create_finding_missing_required_field_returns_422(
    api_client, auth_headers, fake_db
):
    payload = {
        "finding_id": "F-INCOMPLETE-001",
        # missing audit_run_id, control_id, etc.
    }

    response = api_client.post(
        "/findings", json=payload, headers=auth_headers
    )

    assert response.status_code == 422


# =====================================================================
# GET /findings/{id}/reviews
# =====================================================================


def test_get_finding_review_history_not_found(api_client, fake_db):
    response = api_client.get("/findings/DOES-NOT-EXIST/reviews")
    assert response.json()["status"] == "not_found"


def test_get_finding_review_history_success(api_client, fake_db):
    fake_db.get_table("findings").rows.append(
        _sample_finding(finding_id="F-REVIEWS-001")
    )
    fake_db.get_table("finding_reviews").rows.append(
        {
            "finding_id": "F-REVIEWS-001",
            "audit_run_id": "AUDIT-TEST-001",
            "previous_status": "REVIEW",
            "new_status": "CONFIRMED",
            "reviewed_by": "sherry",
            "reviewer_notes": None,
            "reviewed_at": "2026-01-01T00:00:00Z",
        }
    )

    response = api_client.get("/findings/F-REVIEWS-001/reviews")

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "success"
    assert body["count"] == 1
    assert body["reviews"][0]["new_status"] == "CONFIRMED"


def test_get_finding_review_history_empty_for_never_reviewed_finding(
    api_client, fake_db
):
    fake_db.get_table("findings").rows.append(
        _sample_finding(finding_id="F-NEVER-REVIEWED")
    )

    response = api_client.get("/findings/F-NEVER-REVIEWED/reviews")

    body = response.json()
    assert body["status"] == "success"
    assert body["count"] == 0
    assert body["reviews"] == []
def test_execute_audit_returns_error_when_pipeline_failed_without_persisting(
    api_client, execute_headers, fake_db, monkeypatch
):
    """
    A failed pipeline run (audit_trace.status == "FAILED") must be
    reported as an error to the caller and must NOT be persisted to
    Supabase as if it succeeded -- the same class of bug fixed
    earlier in engine.audit_orchestration.run_audit_and_persist(),
    guarded independently here since this endpoint calls run_audit()
    directly.
    """

    monkeypatch.setattr(
        backend_main,
        "run_audit",
        lambda audit_run_id: _sample_result(
            audit_run_id,
            status="FAILED",
            error_type="FileNotFoundError",
            error_message="simulated missing data file",
        ),
    )

    response = api_client.post(
        "/audit-runs/execute",
        headers=execute_headers("idem-pipeline-failed"),
    )

    assert response.status_code == 200, response.text
    body = response.json()

    assert body["status"] == "error"
    assert body["error_type"] == "FileNotFoundError"
    assert "simulated missing data file" in body["error_message"]
    assert "pre_audit_report" in body

    # Nothing should have been written to Supabase.
    assert len(fake_db.get_table("audit_runs").rows) == 0
    assert len(fake_db.get_table("findings").rows) == 0
    assert len(fake_db.get_table("audit_evaluations").rows) == 0    