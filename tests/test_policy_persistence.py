"""Tests for the Supabase policy sync and RAG registry bridge."""

from engine.policy_persistence import (
    load_policy_registry_from_supabase,
    sync_policy_registry,
)
from engine.policy_registry import PolicyRegistry
from RAG.retriever import resolve_policy_references


class _Response:
    def __init__(self, data):
        self.data = data


class _Table:
    def __init__(self, name, calls, rows):
        self.name = name
        self.calls = calls
        self.rows = rows
        self.pending = None

    def upsert(self, payload, on_conflict=None):
        self.pending = ("upsert", payload, on_conflict)
        return self

    def select(self, _columns):
        self.pending = ("select", None, None)
        return self

    def execute(self):
        operation, payload, conflict = self.pending
        self.calls.append(
            {
                "table": self.name,
                "operation": operation,
                "payload": payload,
                "on_conflict": conflict,
            }
        )
        if operation == "select":
            return _Response(self.rows.get(self.name, []))
        return _Response(payload)


class _Client:
    def __init__(self, rows=None):
        self.calls = []
        self.rows = rows or {}

    def table(self, name):
        return _Table(name, self.calls, self.rows)


def _policy(policy_id="POLICY-1", version="1.0", content="Version one"):
    return {
        "policy_id": policy_id,
        "version": version,
        "title": "Test policy",
        "sections": [{"section": "Requirements", "content": content}],
    }


def test_sync_policy_registry_writes_policy_and_versioned_sections():
    client = _Client()

    sync_policy_registry(PolicyRegistry([_policy()]), client=client)

    assert [call["table"] for call in client.calls] == [
        "policies",
        "policy_versions",
    ]
    assert client.calls[0]["payload"] == [
        {"policy_id": "POLICY-1", "policy_name": "Test policy"}
    ]
    assert client.calls[1]["on_conflict"] == "policy_id,version,section"
    assert client.calls[1]["payload"][0]["policy_text"] == "Version one"


def test_database_registry_preserves_versions_for_exact_rag_retrieval():
    client = _Client(
        {
            "policies": [
                {"policy_id": "POLICY-1", "policy_name": "Test policy"}
            ],
            "policy_versions": [
                {
                    "policy_id": "POLICY-1",
                    "version": "1.0",
                    "section": "Requirements",
                    "policy_text": "Version one requirement",
                },
                {
                    "policy_id": "POLICY-1",
                    "version": "2.0",
                    "section": "Requirements",
                    "policy_text": "Version two requirement",
                },
            ],
        }
    )

    registry = load_policy_registry_from_supabase(client=client)
    context = resolve_policy_references(
        {
            "policy_references": [
                {
                    "policy_id": "POLICY-1",
                    "version": "1.0",
                    "section": "Requirements",
                }
            ]
        },
        registry,
    )

    assert registry.contains("POLICY-1")
    assert len(context) == 1
    assert context[0]["version"] == "1.0"
    assert context[0]["content"] == "Version one requirement"
