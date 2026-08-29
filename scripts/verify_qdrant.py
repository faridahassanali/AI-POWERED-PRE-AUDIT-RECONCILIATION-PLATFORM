"""
Qdrant Preflight Verification.

Mirrors scripts/verify_database.py's pattern for the vector store:
catch a misconfigured/unindexed Qdrant loudly, before RAG semantic
fallback silently degrades to lexical search (see
RAG.retriever._try_semantic_fallback, which swallows every Qdrant
error and returns [] -- great for runtime resilience, terrible for
noticing a real misconfiguration during development/deploy).

Checks, in order:
    1. Reachable       -- is a Qdrant server listening at all?
    2. Collection       -- does policy_chunks exist?
    3. Point count > 0  -- has anything actually been indexed?
    4. Point count matches the number of chunked policy sections
       (RAG.chunker.chunk_all_policies()) -- catches a stale/partial
       index (e.g. policies edited after the last index run).

Usage
-----
    docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
    python -m RAG.vector_store        # index once
    python scripts/verify_qdrant.py

Exit code is 0 only if every check passes.
"""

from __future__ import annotations

import socket
import sys
from dataclasses import dataclass, field

from RAG.chunker import chunk_all_policies
from RAG.vector_store import (
    _COLLECTION_NAME,
    _QDRANT_HOST,
    _QDRANT_PORT,
    get_client,
)


@dataclass
class CheckResult:
    name: str
    passed: bool
    details: list[str] = field(default_factory=list)


@dataclass
class Report:
    results: list[CheckResult] = field(default_factory=list)

    def add(self, name: str, passed: bool, details: list[str] | None = None) -> None:
        self.results.append(CheckResult(name=name, passed=passed, details=details or []))

    @property
    def all_passed(self) -> bool:
        return all(r.passed for r in self.results)

    def render(self) -> str:
        lines = ["=" * 64, "QDRANT PREFLIGHT VERIFICATION", "=" * 64]
        for result in self.results:
            status = "PASS" if result.passed else "FAIL"
            lines.append("")
            lines.append(f"[{status}] {result.name}")
            for detail in result.details:
                lines.append(f"    - {detail}")
        lines.append("")
        lines.append("=" * 64)
        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)
        lines.append(f"RESULT: {passed}/{total} checks passed")
        lines.append("=" * 64)
        return "\n".join(lines)


def _is_reachable(host: str = _QDRANT_HOST, port: int = _QDRANT_PORT) -> bool:
    try:
        with socket.create_connection((host, port), timeout=2):
            return True
    except OSError:
        return False


def check_reachable() -> CheckResult:
    if _is_reachable():
        return CheckResult(
            "Reachable",
            True,
            [f"Connected to Qdrant at {_QDRANT_HOST}:{_QDRANT_PORT}."],
        )
    return CheckResult(
        "Reachable",
        False,
        [
            f"Could not reach Qdrant at {_QDRANT_HOST}:{_QDRANT_PORT}.",
            "Start it with: docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant",
        ],
    )


def check_collection_exists(client) -> CheckResult:
    try:
        existing = [c.name for c in client.get_collections().collections]
    except Exception as exc:
        return CheckResult("Collection", False, [f"Could not list collections: {exc}"])

    if _COLLECTION_NAME in existing:
        return CheckResult(
            "Collection",
            True,
            [f"Collection '{_COLLECTION_NAME}' exists."],
        )
    return CheckResult(
        "Collection",
        False,
        [
            f"Collection '{_COLLECTION_NAME}' does not exist.",
            "Run: python -m RAG.vector_store to index policies.",
        ],
    )


def check_point_count_positive(client) -> tuple[CheckResult, int | None]:
    try:
        count_result = client.count(collection_name=_COLLECTION_NAME, exact=True)
        point_count = count_result.count
    except Exception as exc:
        return CheckResult("Point Count > 0", False, [f"Could not count points: {exc}"]), None

    if point_count > 0:
        return (
            CheckResult("Point Count > 0", True, [f"{point_count} point(s) indexed."]),
            point_count,
        )
    return (
        CheckResult(
            "Point Count > 0",
            False,
            ["Collection exists but contains zero points -- nothing has been indexed."],
        ),
        point_count,
    )


def check_point_count_matches_chunks(point_count: int | None) -> CheckResult:
    if point_count is None:
        return CheckResult(
            "Point Count Matches Policy Chunks",
            False,
            ["Skipped -- point count could not be determined."],
        )

    try:
        chunks = chunk_all_policies()
    except Exception as exc:
        return CheckResult(
            "Point Count Matches Policy Chunks",
            False,
            [f"Could not chunk local policy files for comparison: {exc}"],
        )

    expected_count = len(chunks)

    if point_count == expected_count:
        return CheckResult(
            "Point Count Matches Policy Chunks",
            True,
            [f"{point_count} indexed points == {expected_count} chunked policy sections."],
        )

    return CheckResult(
        "Point Count Matches Policy Chunks",
        False,
        [
            f"Indexed point count ({point_count}) does not match the current "
            f"chunked policy sections ({expected_count}).",
            "This usually means policies were edited/added after the last "
            "index run -- re-run: python -m RAG.vector_store",
        ],
    )


def run_all_checks() -> Report:
    report = Report()

    reachable = check_reachable()
    report.results.append(reachable)
    if not reachable.passed:
        return report

    client = get_client()

    collection = check_collection_exists(client)
    report.results.append(collection)
    if not collection.passed:
        return report

    count_check, point_count = check_point_count_positive(client)
    report.results.append(count_check)

    report.results.append(check_point_count_matches_chunks(point_count))

    return report


def main() -> int:
    report = run_all_checks()
    print(report.render())
    return 0 if report.all_passed else 1


if __name__ == "__main__":
    sys.exit(main())