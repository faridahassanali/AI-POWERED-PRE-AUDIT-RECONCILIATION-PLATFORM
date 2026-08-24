"""Synchronize policy Markdown with Supabase and read it back for RAG.

Markdown remains the authoring format.  Supabase is the runtime policy
store whenever it is configured, so the policy evidence used by RAG is
the same versioned evidence that is retained with audit results.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from engine.persistence import get_supabase_client


class DatabasePolicyRegistry:
    """Policy-registry interface backed by versioned Supabase rows.

    ``all()`` deliberately returns one policy object per policy-version.
    The RAG retriever then applies the finding's exact version and section
    reference, rather than silently selecting a newer version.
    """

    def __init__(self, policies: list[dict[str, Any]]):
        self._policies = policies
        self._policy_ids = {
            policy["policy_id"] for policy in policies
        }

    def all(self) -> list[dict[str, Any]]:
        return list(self._policies)

    def contains(self, policy_id: str) -> bool:
        return policy_id in self._policy_ids

    def __contains__(self, policy_id: str) -> bool:
        return self.contains(policy_id)


def sync_policy_registry(
    registry: Any,
    client: Any | None = None,
) -> None:
    """Upsert every locally authored policy and section into Supabase.

    This is additive: historical policy versions already in the database
    are retained so an older finding can still retrieve its exact evidence.
    """

    client = client or get_supabase_client()
    policies = registry.all()

    policy_rows: list[dict[str, Any]] = []
    version_rows: list[dict[str, Any]] = []

    for policy in policies:
        policy_rows.append(
            {
                "policy_id": policy["policy_id"],
                "policy_name": policy["title"],
            }
        )
        version_rows.extend(
            {
                "policy_id": policy["policy_id"],
                "version": policy["version"],
                "section": section["section"],
                "policy_text": section["content"],
            }
            for section in policy["sections"]
        )

    if policy_rows:
        (
            client.table("policies")
            .upsert(policy_rows, on_conflict="policy_id")
            .execute()
        )

    if version_rows:
        (
            client.table("policy_versions")
            .upsert(
                version_rows,
                on_conflict="policy_id,version,section",
            )
            .execute()
        )


def load_policy_registry_from_supabase(
    client: Any | None = None,
) -> DatabasePolicyRegistry:
    """Build a version-aware registry from Supabase policy tables."""

    client = client or get_supabase_client()
    policy_rows = (
        client.table("policies")
        .select("policy_id,policy_name")
        .execute()
        .data
        or []
    )
    version_rows = (
        client.table("policy_versions")
        .select("policy_id,version,section,policy_text")
        .execute()
        .data
        or []
    )

    names = {
        row["policy_id"]: row["policy_name"]
        for row in policy_rows
    }
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)

    for row in version_rows:
        if not row.get("section"):
            raise ValueError(
                "Supabase policy_versions rows must have a section."
            )
        grouped[(row["policy_id"], row["version"])].append(
            {
                "section": row["section"],
                "content": row["policy_text"],
            }
        )

    policies = [
        {
            "policy_id": policy_id,
            "version": version,
            "title": names.get(policy_id, policy_id),
            "sections": sections,
        }
        for (policy_id, version), sections in grouped.items()
    ]
    return DatabasePolicyRegistry(policies)
